import Foundation
import Combine

/// The single source of truth for running a chat/agent turn. Both the text chat
/// window and the hands-free voice controller submit turns through here, so the
/// two never drift behaviourally. The engine is app-level (`AppState` owns it for
/// the app's lifetime) and window-independent: it appends messages, streams via
/// `APIClient`, runs the tool-calling loop, and writes everything into
/// `AppState.chatSessions`. Views observe `isGenerating` for spinners.
///
/// It conforms to `TurnRunning` so the voice controller can be unit-tested with a
/// fake runner instead of the concrete `APIClient`.
@MainActor
protocol TurnRunning: AnyObject {
    var isGenerating: Bool { get }
    /// Run one user turn against `sessionId`. Cancels any in-flight turn first.
    /// `approval` is invoked before every tool dispatch (agent mode) — returning
    /// false denies the call. Plain chat never calls `approval`.
    func runTurn(sessionId: UUID,
                 userText: String,
                 images: [ChatImage]?,
                 audio: [ChatAudio]?,
                 config: ChatTurnEngine.TurnConfig,
                 approval: @escaping (APIClient.ToolCall) async -> Bool)
    /// Cancel the current turn (replaces the View's `stopGenerating`).
    func stop()
}

@MainActor
final class ChatTurnEngine: ObservableObject, TurnRunning {
    /// Owning app state. `unowned` because the engine lives exactly as long as
    /// `AppState` does (it's a `lazy var` on it) — there is never a window where
    /// the engine outlives its owner.
    unowned let appState: AppState

    /// Drives every spinner in the app (text chat bubbles + the voice loading
    /// cue, indirectly via the controller's turn state).
    @Published private(set) var isGenerating = false

    private var generationTask: Task<Void, Never>?
    /// The session the in-flight turn is writing into — used by `stop()` so it
    /// clears the streaming flag on the right message even if `activeChatId`
    /// changed underneath it.
    private var activeTurnSessionId: UUID?

    init(appState: AppState) {
        self.appState = appState
    }

    /// Per-turn configuration. Built by the caller (chat window from its
    /// toolbar toggles; voice controller from its own voice-scoped toggles).
    struct TurnConfig {
        var agentMode: Bool
        var mcpMode: Bool
        var enableThinking: Bool
        var voiceStyle: Bool
        var workingDirectory: String?
    }

    // MARK: - Convenience accessors

    private var server: ServerManager { appState.server }
    private var mcpManager: MCPManager { appState.mcpManager }
    private func session(_ id: UUID) -> ChatSession? {
        appState.chatSessions.first { $0.id == id }
    }

    /// Apply a coalesced streaming batch into the session. Streamed tokens are
    /// batched (see `StreamCoalescer`) rather than written one at a time so the
    /// per-token `AppState.objectWillChange` churn can't re-render — and wedge —
    /// the open tray popover during the assistant's answer.
    private func applyStreamBatch(_ batch: (content: String, reasoning: String)?,
                                  to sessionId: UUID) {
        guard let batch else { return }
        if !batch.content.isEmpty { appState.updateLastMessage(in: sessionId, content: batch.content) }
        if !batch.reasoning.isEmpty { appState.updateLastMessage(in: sessionId, reasoning: batch.reasoning) }
    }

    // MARK: - Public API (TurnRunning)

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        if let sid = activeTurnSessionId {
            appState.updateLastMessage(in: sid, streaming: false)
            appState.saveChatHistory()
        }
        isGenerating = false
    }

    func runTurn(sessionId: UUID,
                 userText: String,
                 images: [ChatImage]?,
                 audio: [ChatAudio]?,
                 config: TurnConfig,
                 approval: @escaping (APIClient.ToolCall) async -> Bool) {
        // Cancel any in-flight turn first — a new submission supersedes it.
        generationTask?.cancel()
        generationTask = nil

        let text = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || images != nil || audio != nil,
              !isGenerating, server.status == .running else { return }

        activeTurnSessionId = sessionId

        if config.agentMode || config.mcpMode {
            runAgentTurn(sessionId: sessionId, text: text, images: images, audio: audio,
                         config: config, approval: approval)
        } else {
            runPlainTurn(sessionId: sessionId, text: text, images: images, audio: audio,
                         config: config)
        }
    }

    // MARK: - Plain chat

    private func runPlainTurn(sessionId: UUID, text: String,
                              images: [ChatImage]?, audio: [ChatAudio]?,
                              config: TurnConfig) {
        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = images
        userMsg.audio = audio
        appState.appendMessage(to: sessionId, message: userMsg)

        isGenerating = true
        let api = APIClient()

        // Build the request from the session (its source of truth). We append
        // the streaming placeholder AFTER this so it never lands in the
        // request — same pattern the agent loop uses. Image handling: only
        // the latest user message's images are sent (older turns' images are
        // stripped for bandwidth).
        let sessionMsgs = session(sessionId)?.messages ?? []
        let lastUserIdx = sessionMsgs.lastIndex { $0.role == .user }
        let history: [[String: Any]] = sessionMsgs.enumerated().map { i, msg in
            if i == lastUserIdx, msg.role == .user {
                let imgs = msg.images ?? []
                let clips = msg.audio ?? []
                if !imgs.isEmpty || !clips.isEmpty {
                    return ["role": "user", "content": Self.buildMultimodalContent(text: msg.content, images: imgs, audio: clips)]
                }
            }
            var d: [String: Any] = ["role": msg.role.rawValue, "content": msg.content]
            if msg.role == .assistant && msg.content.isEmpty { d.removeValue(forKey: "content") }
            return d
        }
        // Plain chat: no synthesized system message (see the long note in the
        // original ChatView implementation — a "formatNudge" system message was
        // routinely read by the model AS the user's input). Voice mode is the one
        // exception: prepend the voice-style guidance so spoken answers stay
        // short and free of URLs/Markdown.
        var messagesArray = history
        if config.voiceStyle {
            messagesArray.insert(["role": "system", "content": VoicePrompt.systemPrompt()], at: 0)
        }

        // Streaming placeholder for the UI — appended AFTER the request body is
        // built so it doesn't show up in the prompt.
        var assistantMsg = ChatMessage(role: .assistant, content: "")
        assistantMsg.isStreaming = true
        appState.appendMessage(to: sessionId, message: assistantMsg)

        generationTask = Task { [weak self] in
            await self?.streamPlainResponse(api: api, sessionId: sessionId,
                                            messages: messagesArray, config: config)
        }
    }

    private func streamPlainResponse(api: APIClient, sessionId: UUID,
                                     messages: [[String: Any]], config: TurnConfig) async {
        do {
            // Pin the request to the active model (server-resolved default if
            // nil) so hot-switch can finish in-flight requests on the old model.
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: appState.serverOptions.defaultTemperature,
                enableThinking: config.enableThinking || appState.serverOptions.defaultEnableThinking,
                defaults: APIClient.RequestDefaults.from(appState.serverOptions),
                modelId: server.modelInfo?.name
            )
            var coalescer = StreamCoalescer()
            for try await event in stream {
                try Task.checkCancellation()
                switch event {
                case .content(let text):
                    applyStreamBatch(coalescer.add(content: text, now: Date().timeIntervalSinceReferenceDate),
                                     to: sessionId)
                case .reasoning(let text):
                    applyStreamBatch(coalescer.add(reasoning: text, now: Date().timeIntervalSinceReferenceDate),
                                     to: sessionId)
                case .usage(let usage):
                    applyStreamBatch(coalescer.drain(), to: sessionId)
                    appState.updateLastMessage(in: sessionId, usage: usage)
                case .toolCalls:
                    break
                case .maxTokensReached:
                    // Plain chat is a single, always-terminal response — show the
                    // notice immediately (no agent loop to stack it). Flush any
                    // buffered text first so the notice lands after it, in order.
                    applyStreamBatch(coalescer.drain(), to: sessionId)
                    appState.updateLastMessage(in: sessionId, content: TruncationNotice.text(maxTokens: appState.maxTokens))
                case .done:
                    break
                }
            }
            applyStreamBatch(coalescer.drain(), to: sessionId)   // flush the trailing batch
        } catch is CancellationError {
            // Stopped by user
        } catch {
            print("[ChatTurnEngine] Chat error: \(error)")
            try? "Chat error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
            appState.updateLastMessage(in: sessionId, content: "\n\n[Error: \(error.localizedDescription)]")
        }
        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.saveChatHistory()
        isGenerating = false
        generationTask = nil
    }

    // MARK: - Agent mode (native tool calling)

    private func runAgentTurn(sessionId: UUID, text: String,
                              images: [ChatImage]?, audio: [ChatAudio]?,
                              config: TurnConfig,
                              approval: @escaping (APIClient.ToolCall) async -> Bool) {
        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = images
        userMsg.audio = audio
        appState.appendMessage(to: sessionId, message: userMsg)

        isGenerating = true
        let api = APIClient()
        let workDir = config.workingDirectory

        generationTask = Task { [weak self] in
            guard let self else { return }
            // Lazy-spawn MCP servers if MCP mode is on. Idempotent — already-connected servers are skipped.
            if config.mcpMode {
                // Inherit the chat's working directory so filesystem/shell MCP servers anchor at the
                // same dir the agent's built-in tools use. Per-entry `cwd` in mcp.json still wins.
                self.mcpManager.defaultCwd = config.workingDirectory
                await self.mcpManager.startEnabled()
                // Surface startup failures inline in chat — otherwise they're hidden behind the
                // marketplace gear icon and the user just sees "MCP doesn't seem to do anything".
                if !self.mcpManager.startErrors.isEmpty {
                    let lines = self.mcpManager.startErrors
                        .sorted(by: { $0.key < $1.key })
                        .map { "• **\($0.key)**: \($0.value)" }
                        .joined(separator: "\n")
                    let hint = self.mcpManager.sessions.isEmpty
                        ? "No MCP servers are connected — the model has no MCP tools available for this turn. Open the gear icon on the MCP pill to fix or disable broken servers."
                        : "Some MCP servers couldn't start. The model will only see tools from the ones that did connect."
                    let warning = ChatMessage(
                        role: .assistant,
                        content: "⚠️ MCP startup issues:\n\n\(lines)\n\n\(hint)"
                    )
                    self.appState.appendMessage(to: sessionId, message: warning)
                }
            }
            do {
                try await self.runAgentLoop(api: api, sessionId: sessionId, config: config,
                                            workingDirectory: workDir, approval: approval)
            } catch is CancellationError {
                // Stopped by user — stop() already cleared the streaming flag.
            } catch {
                print("[ChatTurnEngine] Agent error: \(error)")
                try? "Agent error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
                // Clear the spinner on the in-flight assistant message before appending the error;
                // otherwise GeneratingIndicator stays visible on the orphaned streaming bubble.
                self.appState.updateLastMessage(in: sessionId, streaming: false)
                var errorMsg = ChatMessage(role: .assistant, content: "[Error: \(error.localizedDescription)]")
                errorMsg.isStreaming = false
                self.appState.appendMessage(to: sessionId, message: errorMsg)
            }
            self.appState.saveChatHistory()
            self.isGenerating = false
            self.generationTask = nil
        }
    }

    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 150 iterations.
    private func runAgentLoop(api: APIClient, sessionId: UUID, config: TurnConfig,
                              workingDirectory initialWorkDir: String?,
                              approval: @escaping (APIClient.ToolCall) async -> Bool) async throws {
        var workingDirectory = initialWorkDir
        let maxIterations = 150
        let padRetryPolicy = RetryPolicy.aggressive
        let repetition = AgentEngine.RepetitionTracker()
        // Bail out if the model spends several rounds where every tool call
        // fails/blocks (e.g. an unresolvable tool name) instead of grinding to
        // `maxIterations` achieving nothing.
        var stuck = AgentEngine.StuckDetector()
        // Budget for recoverable failures: ghost/malformed tool calls, truncated
        // tool-call args, empty/pad responses. CONSECUTIVE, not cumulative — a
        // real tool round resets it (recordProgress below), so an isolated late
        // failure doesn't end a long, productive turn.
        var retry = AgentEngine.AgentRetryBudget()

        for iteration in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            let contextLength = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.modelInfo?.contextLength
            )
            var history = AgentEngine.buildAgentHistory(
                messages: session(sessionId)?.messages ?? [],
                contextLength: contextLength,
                maxTokens: appState.maxTokens,
                buildMultimodalContent: Self.buildMultimodalContent
            )
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let mcpToolsJSON = config.mcpMode ? mcpManager.toolDefinitionsJSON() : nil
            let mcpListing = config.mcpMode ? mcpManager.toolListingForPrompt() : ""
            var systemPrompt: String
            if config.agentMode {
                let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
                systemPrompt = AgentPrompt.systemPrompt + skills + AgentPrompt.memory + appState.agentMemory.contextSnippet()
                if let wd = workingDirectory {
                    systemPrompt += AgentEngine.workingDirectoryContext(wd)
                }
                if !mcpListing.isEmpty {
                    systemPrompt += "\n\n# MCP Tools\nIn addition to the built-in tools above, the user has connected these MCP servers. Their tools are namespaced as `<server>__<tool>`:\n\n\(mcpListing)"
                }
            } else {
                // MCP-only mode: minimal system prompt focused on MCP tool use, no shell/file rules.
                systemPrompt = AgentPrompt.mcpOnlySystemPrompt(toolListing: mcpListing)
            }
            // Ground every agent turn in the wall clock so "what time/date is it"
            // is answered from reality, not a hallucinated guess (and so recency
            // reasoning is correct). Voice and text agents both benefit.
            systemPrompt = SystemGrounding.dateTimeLine() + "\n\n" + systemPrompt
            // Voice mode: tools/thinking run silently; only the final answer is
            // spoken, so steer it to a short, speakable reply (no URLs/Markdown).
            if config.voiceStyle { systemPrompt = VoicePrompt.decorate(systemPrompt) }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            // Some models (e.g. Gemma 4 E4B) can't generate after tool results without
            // a user message. Add a nudge so the model knows to synthesize a response —
            // asks explicitly for a short plain-text summary when finished so the user
            // never sees a conversation that ends on a bare tool-call echo.
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is complete, reply with a short plain-text summary for the user (what got done, where it lives, any caveats) — no tool calls, no JSON. If more work is needed, make the next tool call."])
            }
            messages.append(contentsOf: history)

            AgentEngine.dumpDebugRequest(messages: messages, maxTokens: appState.maxTokens)

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            var maxTokensHit = false
            let combinedToolsJSON = Self.combinedToolsJSON(
                agentMode: config.agentMode,
                mcpToolsJSON: mcpToolsJSON
            )
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: appState.maxTokens,
                temperature: 0.7,
                enableThinking: config.enableThinking,
                toolsJSON: combinedToolsJSON,
                defaults: APIClient.RequestDefaults.from(appState.serverOptions),
                modelId: server.modelInfo?.name
            )

            // No client-side stream watchdog: long generations (large
            // contexts, big batches, slow sampling on big MoE) can legitimately
            // sit silent for minutes between events. The user keeps the Stop
            // button as the manual cancel; URLSession's own resource timeout
            // (set in APIClient) handles a truly broken socket.
            let streamTask = Task<(tcs: [APIClient.ToolCall], maxHit: Bool), Error> {
                var tcs: [APIClient.ToolCall] = []
                var maxHit = false
                var coalescer = StreamCoalescer()
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .content(let text):
                        self.applyStreamBatch(coalescer.add(content: text, now: Date().timeIntervalSinceReferenceDate),
                                              to: sessionId)
                    case .reasoning(let text):
                        self.applyStreamBatch(coalescer.add(reasoning: text, now: Date().timeIntervalSinceReferenceDate),
                                              to: sessionId)
                    case .usage(let usage):
                        self.applyStreamBatch(coalescer.drain(), to: sessionId)
                        appState.updateLastMessage(in: sessionId, usage: usage)
                    case .toolCalls(let calls):
                        tcs = calls
                    case .maxTokensReached:
                        // Just record it. The notice is surfaced once at the
                        // turn's terminal exit (see below) — appending here, per
                        // iteration, is what stacked duplicate banners on a
                        // multi-step agent turn.
                        maxHit = true
                    case .done:
                        break
                    }
                }
                // Flush the trailing batch so the message content is complete
                // before the post-stream truncation/pad checks read it back.
                self.applyStreamBatch(coalescer.drain(), to: sessionId)
                return (tcs, maxHit)
            }
            // Wire the user's Stop button through to the inner stream task.
            do {
                let result = try await withTaskCancellationHandler {
                    try await streamTask.value
                } onCancel: {
                    streamTask.cancel()
                }
                receivedToolCalls = result.tcs
                maxTokensHit = result.maxHit
            } catch is CancellationError {
                throw CancellationError()
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

            // Truncation recovery: if max_tokens was hit AND tool calls were received,
            // the tool call args are likely truncated (incomplete JSON). Don't execute them —
            // mark the broken message as non-replayable (preserves reasoning in the UI)
            // and nudge the model to try again more concisely.
            if maxTokensHit && !receivedToolCalls.isEmpty && retry.allowTruncationRetry() {
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   !appState.chatSessions[sIdx].messages.isEmpty {
                    let mIdx = appState.chatSessions[sIdx].messages.count - 1
                    appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    appState.chatSessions[sIdx].messages[mIdx].toolCalls = nil
                }
                let nudge = ChatMessage(role: .user, content: "[System: Your last response was cut off because the output was too long. The tool call was NOT executed. To avoid this, write shorter responses: use shell with heredoc (cat << 'EOF' > file) for file content instead of writeFile, or break large files into smaller pieces.]")
                appState.appendMessage(to: sessionId, message: nudge)
                continue
            }

            // Check for pad-only or empty responses — retry limited times.
            // Mark the empty message as failedRetry so it's hidden from API history
            // but its reasoning (if any) stays visible in the UI.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                let cleaned = lastContent
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    retry.pad += 1
                    if retry.pad <= padRetryPolicy.maxRetries {
                        let delay = padRetryPolicy.delay(for: retry.pad)
                        try? await Task.sleep(nanoseconds: delay)
                        continue
                    }
                    let errorMsg = ChatMessage(role: .assistant, content: "The model couldn't generate a response. Try rephrasing or starting a new chat.")
                    appState.appendMessage(to: sessionId, message: errorMsg)
                    return
                }
            }

            // If no tool calls, we're done — but make sure the user sees a
            // clean completion text. The model sometimes exits with a ghost
            // tool call (malformed <|tool_call>...<tool_call|> or <tool_call>
            // with bad args that didn't parse) as its final content; that's
            // ugly and uninformative. When we detect one, mark the garbled
            // turn as failedRetry (hidden from API history) and ask the model
            // for a plain-text summary before returning control to the user.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                // Match the full `<tool…` family — `<tool_call>`, `<tool_call name=…>`,
                // `<tool_calls>` wrapper, `<tool name=… arguments=…/>` self-closing,
                // Gemma 4 `<|tool_call>`/`<tool_call|>`, and `<function=` legacy. The
                // server-side `parseToolCalls` already handles all of these; this
                // check is the defense-in-depth that fires the retry nudge when
                // a new model variant slips through the parser before we recognize
                // it (the symptom: hundreds of completion_tokens but the assistant
                // turn ends with markup-as-content and no parsed tool_calls).
                let looksLikeGhostToolCall = lastContent.contains("<|tool_call>") ||
                    lastContent.contains("<tool_call>") ||
                    lastContent.contains("<tool_call ") ||
                    lastContent.contains("<tool_calls>") ||
                    lastContent.contains("<tool_calls ") ||
                    lastContent.contains("<tool_call|>") ||
                    lastContent.contains("<tool name=") ||
                    lastContent.contains("<function=")
                if looksLikeGhostToolCall && retry.allowGhostRetry() {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    let nudge = ChatMessage(role: .user, content: "[System: your last response contained a malformed tool-call tag. If you meant to call a tool, call it with proper JSON. If the task is complete, respond with a short plain-text summary of what you did — no tool tags, no JSON — just a sentence or two for the user.]")
                    appState.appendMessage(to: sessionId, message: nudge)
                    continue
                }
                // Terminal exit: a final answer with no more tool calls. If it
                // was cut off by the cap, surface the truncation notice exactly
                // once here — not per iteration in the stream loop above.
                if TruncationNotice.shouldShow(maxTokensHit: maxTokensHit, turnEnding: true, willRetry: false) {
                    appState.updateLastMessage(in: sessionId, content: TruncationNotice.text(maxTokens: appState.maxTokens))
                }
                return
            }

            // A round with real, parseable tool calls = progress. Reset the
            // recoverable-failure budget so an isolated *later* ghost/truncated
            // tool call gets its own retry instead of ending the turn (the budget
            // counts consecutive failures, not lifetime-of-turn ones).
            retry.recordProgress()

            // Track repetition for this round
            repetition.track(toolCalls: receivedToolCalls)

            // Store tool calls on the assistant message for history replay
            if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               !appState.chatSessions[sIdx].messages.isEmpty {
                let mIdx = appState.chatSessions[sIdx].messages.count - 1
                appState.chatSessions[sIdx].messages[mIdx].toolCalls = receivedToolCalls.map { tc in
                    let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    return SerializedToolCall(id: tc.id, name: tc.name, arguments: argsJson)
                }
            }

            // Show tool call summary as display-only message. Mark streaming so the GeneratingIndicator
            // keeps rendering underneath while tools execute — otherwise a slow / hung MCP tool looks
            // like the chat just froze with no feedback.
            let callSummary = receivedToolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                let display = args.isEmpty ? tc.rawArguments.prefix(200) : args[...]
                return "**\(tc.name)**(\(display))"
            }.joined(separator: "\n")
            var summaryMsg = ChatMessage(role: .assistant, content: callSummary)
            summaryMsg.isAgentSummary = true
            summaryMsg.isStreaming = true
            let summaryId = summaryMsg.id
            appState.appendMessage(to: sessionId, message: summaryMsg)
            // Stop the spinner on the summary regardless of how we leave the loop (success, throw, cancel).
            defer {
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == summaryId }) {
                    appState.chatSessions[sIdx].messages[mIdx].isStreaming = false
                }
            }

            // Execute each tool call. MCP-namespaced names (`<server>__<tool>`) route to MCPManager;
            // everything else flows through the existing AgentEngine dispatch.
            // Tool-approval gate: before every dispatch, ask via the injected
            // `approval` closure (the chat window's sheet, or the voice
            // controller's auto-approve/inline card). Deny short-circuits to a
            // fabricated error result so the agent loop can react and the
            // user's intent is visible in the transcript.
            var roundOutputs: [String] = []
            for tc in receivedToolCalls {
                try Task.checkCancellation()

                let approved = await approval(tc)
                guard approved else {
                    let denied = AgentEngine.ToolResult(
                        id: tc.id,
                        name: tc.name,
                        output: "Error: user denied this tool call. Do not retry this command; ask the user how to proceed or try a different approach."
                    )
                    // A user denial is a deliberate stop, not a stuck loop — don't
                    // count it toward the no-progress tally.
                    var deniedMsg = ChatMessage(role: .assistant, content: "**\(tc.name)** → denied by user")
                    deniedMsg.isAgentSummary = true
                    appState.appendMessage(to: sessionId, message: deniedMsg)
                    var toolMsg = ChatMessage(role: .system, content: denied.output)
                    toolMsg.toolCallId = denied.id
                    toolMsg.toolName = denied.name
                    appState.appendMessage(to: sessionId, message: toolMsg)
                    continue
                }

                // One execution path for built-in *and* MCP tools — the shared
                // repetition guard applies to both, so an MCP tool can no longer
                // loop forever (it routes to `mcpManager` internally via
                // `MCPToolRouting`).
                let result = await AgentEngine.executeToolCall(
                    tc, workingDirectory: &workingDirectory,
                    repetition: repetition, iteration: iteration,
                    agentMemory: appState.agentMemory,
                    mcpRouter: mcpManager
                )
                roundOutputs.append(result.output)

                // Show result in chat (display-only)
                var resultMsg = ChatMessage(role: .assistant, content: "**\(result.name)** → \(String(result.output.prefix(500)))")
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)

                // Store tool result as tool role message
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = result.id
                toolMsg.toolName = result.name

                // Extract screenshot image data and attach as vision input
                if result.name == "browse" && result.output.contains("data:image/jpeg;base64,") {
                    if let range = result.output.range(of: "data:image/jpeg;base64,") {
                        let remainder = result.output[range.upperBound...]
                        let b64End = remainder.firstIndex(of: "\n") ?? remainder.endIndex
                        let b64 = String(remainder[..<b64End])
                        if let jpegData = Data(base64Encoded: b64),
                           let chatImage = ChatImage(data: jpegData) as ChatImage? {
                            toolMsg.images = [chatImage]
                            toolMsg.content = "[screenshot captured]"
                        } else {
                            toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                        }
                    } else {
                        toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                    }
                } else {
                    toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                }
                appState.appendMessage(to: sessionId, message: toolMsg)
            }

            // Stop if the model made no progress for several consecutive rounds
            // (every tool call failed/blocked) rather than grinding to the cap.
            stuck.record(outputs: roundOutputs)
            if stuck.isStuck {
                let msg = ChatMessage(role: .assistant, content: "Stopped: the last \(AgentEngine.StuckDetector.limit) tool-call rounds all failed without making progress (often an unrecognized tool name). Tell me how you'd like to proceed.")
                appState.appendMessage(to: sessionId, message: msg)
                return
            }
        }

        // Max iterations reached
        let msg = ChatMessage(role: .assistant, content: "(Agent stopped after \(maxIterations) tool call rounds)")
        appState.appendMessage(to: sessionId, message: msg)
    }

    // MARK: - Tool JSON + multimodal content (relocated from ChatDetailView)

    /// Build the JSON tools array sent to the model. Concatenates agent tools (when agent mode is on) and
    /// MCP tools (when MCP mode is on). Returns nil when no tools should be advertised.
    /// `nonisolated` so it stays a pure helper callable off the main actor (unit tests).
    nonisolated static func combinedToolsJSON(agentMode: Bool, mcpToolsJSON: String?) -> String? {
        let agent = agentMode ? AgentPrompt.toolDefinitionsJSON : nil
        switch (agent, mcpToolsJSON) {
        case (nil, nil): return nil
        case (let a?, nil): return a
        case (nil, let m?): return m
        case (let a?, let m?):
            // Strip outer brackets and re-wrap. Both inputs are guaranteed to be JSON arrays.
            let aInner = a.trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            let mInner = m.trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            let aTrimmed = aInner.trimmingCharacters(in: .whitespacesAndNewlines)
            let mTrimmed = mInner.trimmingCharacters(in: .whitespacesAndNewlines)
            if aTrimmed.isEmpty { return "[\(mTrimmed)]" }
            if mTrimmed.isEmpty { return "[\(aTrimmed)]" }
            return "[\(aTrimmed),\(mTrimmed)]"
        }
    }

    /// Build OpenAI-style content blocks for a message with images (and,
    /// optionally, audio). Delegates to the pure, unit-tested `MultimodalContent`
    /// builder. Two overloads so the `buildAgentHistory` closure (images only)
    /// and the plain-chat path (images + audio) can both reference it.
    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage]) -> Any {
        MultimodalContent.build(text: text, images: images, audio: [])
    }

    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage], audio: [ChatAudio]) -> Any {
        MultimodalContent.build(text: text, images: images, audio: audio)
    }
}
