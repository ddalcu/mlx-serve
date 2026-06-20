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

/// Per-session memory for the "Allow all tools this session" decision. Keyed by
/// chat-session id so each tab remembers its own choice: the text-chat window
/// reuses one `ChatDetailView` across `sessionId` changes, so a single `Bool`
/// here is shared by every tab and was wiped on every switch. A `Set` keyed by
/// id has nothing to wipe on a switch — switching away and back preserves the
/// grant — and re-arming (Agent toggled off) clears only that one session.
struct SessionToolAllowList {
    private var allowed: Set<UUID> = []
    func allowsAll(_ id: UUID) -> Bool { allowed.contains(id) }
    mutating func allowAll(_ id: UUID) { allowed.insert(id) }
    /// Re-prompt this session on the next tool call (leaves other tabs alone).
    mutating func rearm(_ id: UUID) { allowed.remove(id) }
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
    /// changed underneath it, and by the composer so only the owning chat shows
    /// the Stop button. Published so the per-chat composer re-renders when the
    /// active turn moves between sessions. Always set before `isGenerating` flips
    /// true (the engine runs one turn at a time, so it never changes mid-turn).
    @Published private(set) var activeTurnSessionId: UUID?

    /// Live count of tokens the in-flight reply has produced — for the chat
    /// composer's live "gen:" readout and growing context bar. Counted by tallying
    /// streamed `.content`/`.reasoning` deltas (one per token for this server) and
    /// PUBLISHED only at the StreamCoalescer's ~20 Hz flush cadence, never per
    /// token — per-token @Published churn is exactly what StreamCoalescer exists to
    /// avoid. Reset at the start of each streamed round; reconciled to the
    /// authoritative `usage.completion_tokens` when the stream reports usage.
    @Published private(set) var liveCompletionTokens: Int = 0
    /// Non-published per-delta tally; promoted into `liveCompletionTokens` on flush.
    private var liveTokenAccum: Int = 0

    /// What the composer's primary button should show for a given chat. The
    /// engine runs one turn at a time app-wide, so a chat that doesn't own the
    /// active turn shows Send (disabled while busy), never the global Stop.
    enum ComposerState: Equatable {
        case idle             // free to send (subject to having content)
        case generatingHere   // this chat owns the in-flight turn → show Stop
        case busyElsewhere    // another chat is mid-turn → Send disabled
    }

    /// Pure decision for `ComposerState`; the instance accessor below feeds it
    /// the live engine state. `nonisolated` because it touches no actor state —
    /// just its arguments — so views and tests can call it freely.
    nonisolated static func composerState(isGenerating: Bool,
                                          activeTurnSessionId: UUID?,
                                          for sessionId: UUID) -> ComposerState {
        guard isGenerating else { return .idle }
        return activeTurnSessionId == sessionId ? .generatingHere : .busyElsewhere
    }

    func composerState(for sessionId: UUID) -> ComposerState {
        Self.composerState(isGenerating: isGenerating,
                           activeTurnSessionId: activeTurnSessionId,
                           for: sessionId)
    }

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
        /// Per-session document index (mini RAG). Non-nil while the user has a
        /// folder attached — advertises the searchDocuments tool and, with both
        /// Agent and MCP off, routes the turn through the loop in docs-only mode.
        var documentIndex: DocumentIndex? = nil
        /// Set when this turn is driven by the Telegram bridge — the originating
        /// chat id. Threaded into any `createTask` the agent makes so the task's
        /// result is pushed back to that chat. nil for in-app / voice turns.
        var telegramChatId: Int64? = nil
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

    /// Begin counting a fresh streamed reply (one per round). Zeroes the live
    /// token tally so the composer's "gen:" restarts from 0.
    private func beginLiveTokenCount() {
        liveTokenAccum = 0
        liveCompletionTokens = 0
    }

    /// Stream one text/reasoning delta into the session and tally it toward the
    /// live token count. `liveCompletionTokens` advances only when the coalescer
    /// actually flushes (≤20 Hz), so the live readout never adds per-token churn.
    private func streamDelta(content: String = "", reasoning: String = "",
                             coalescer: inout StreamCoalescer, to sessionId: UUID) {
        liveTokenAccum += 1
        if let batch = coalescer.add(content: content, reasoning: reasoning,
                                     now: Date().timeIntervalSinceReferenceDate) {
            applyStreamBatch(batch, to: sessionId)
            liveCompletionTokens = liveTokenAccum
        }
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

        if config.agentMode || config.mcpMode || config.documentIndex != nil {
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
        let useServerPreprocess = wantsServerImagePreprocess
        let history: [[String: Any]] = sessionMsgs.enumerated().map { i, msg in
            if i == lastUserIdx, msg.role == .user {
                let imgs = msg.images ?? []
                let clips = msg.audio ?? []
                if !imgs.isEmpty || !clips.isEmpty {
                    return ["role": "user", "content": Self.buildMultimodalContent(text: msg.content, images: imgs, audio: clips, serverPreprocess: useServerPreprocess)]
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
            beginLiveTokenCount()
            for try await event in stream {
                try Task.checkCancellation()
                switch event {
                case .content(let text):
                    streamDelta(content: text, coalescer: &coalescer, to: sessionId)
                case .reasoning(let text):
                    streamDelta(reasoning: text, coalescer: &coalescer, to: sessionId)
                case .usage(let usage):
                    applyStreamBatch(coalescer.drain(), to: sessionId)
                    appState.updateLastMessage(in: sessionId, usage: usage)
                    liveCompletionTokens = usage.completionTokens   // reconcile to the authoritative count
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

        // Resolve the LAN IP once per turn (not per iteration): it's a getifaddrs
        // enumeration that won't change mid-turn, and the agent loop rebuilds the
        // system prompt on every tool round.
        let lanIP = SystemGrounding.localIPAddress()

        for iteration in 0..<maxIterations {
            try Task.checkCancellation()

            // Build message history for API
            let contextLength = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.modelInfo?.contextLength
            )
            let useServerPreprocess = wantsServerImagePreprocess
            var history = AgentEngine.buildAgentHistory(
                messages: session(sessionId)?.messages ?? [],
                contextLength: contextLength,
                maxTokens: appState.maxTokens,
                buildMultimodalContent: { text, images in
                    Self.buildMultimodalContent(text: text, images: images, serverPreprocess: useServerPreprocess)
                }
            )
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let mcpToolsJSON = config.mcpMode ? mcpManager.toolDefinitionsJSON() : nil
            let mcpListing = config.mcpMode ? mcpManager.toolListingForPrompt() : ""
            var systemPrompt: String
            // Volatile context that changes mid-session — kept OUT of the
            // stable prefix and appended at the very end (see composeSystemPrompt).
            var agentVolatileTail = ""
            if config.agentMode {
                let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
                // Stable, cacheable core: base instructions + memory instructions
                // + MCP listing. The model's tool block (rendered by the chat
                // template) sits in front of all of this, so as long as this
                // prefix stays byte-identical the server reuses the whole
                // tool+instruction KV across requests.
                systemPrompt = AgentPrompt.systemPrompt + AgentPrompt.memory
                if !mcpListing.isEmpty {
                    systemPrompt += "\n\n# MCP Tools\nIn addition to the built-in tools above, the user has connected these MCP servers. Their tools are namespaced as `<server>__<tool>`:\n\n\(mcpListing)"
                }
                // Volatile context to the tail, ordered big-listing-then-tiny-
                // snippet so a shell command (which rewrites the recent-commands
                // snippet every turn) only re-prefills the snippet, not the
                // working-dir listing: matched skills (per message), the
                // working-dir listing (changes as files change), then the
                // learned recent-dirs/commands snippet (changes per command).
                agentVolatileTail += skills
                if let wd = workingDirectory {
                    agentVolatileTail += AgentEngine.workingDirectoryContext(wd)
                }
                agentVolatileTail += appState.agentMemory.contextSnippet()
            } else if config.mcpMode {
                // MCP-only mode: minimal system prompt focused on MCP tool use, no shell/file rules.
                systemPrompt = AgentPrompt.mcpOnlySystemPrompt(toolListing: mcpListing)
            } else {
                // Docs-only mode: plain chat with a document folder attached.
                let index = config.documentIndex
                systemPrompt = AgentPrompt.docsOnlySystemPrompt(
                    folderName: index?.folderName ?? "documents",
                    fileCount: indexedFileCount(index))
            }
            // Attached-docs section for the modes whose base prompt doesn't
            // already explain the searchDocuments tool.
            if let index = config.documentIndex, config.agentMode || config.mcpMode {
                systemPrompt += AgentPrompt.attachedDocumentsSection(
                    folderName: index.folderName, fileCount: indexedFileCount(index))
            }
            // Ground the turn in the wall clock (so "what day is it" is answered
            // from reality) and surface the Mac's LAN IP (so the agent hands out
            // reachable URLs). Date-ONLY (no clock time) and at the very END:
            // a per-minute timestamp at the front changed the first tokens every
            // request, so the KV prefix cache missed at token 0 and every agent
            // turn cold-re-prefilled the whole prompt — the cause of slow TTFB.
            var grounding = SystemGrounding.dateLine()
            let ipLine = SystemGrounding.localIPLine(ip: lanIP)
            if !ipLine.isEmpty { grounding += " " + ipLine }
            // Stable prefix first, all volatile content (skills / working-dir /
            // memory) + grounding last — so a mid-session change only re-prefills
            // the short tail, not the cached tool+instruction block.
            // Make the model aware of its actual per-response token cap so it
            // chunks large writes BEFORE truncating (a static "~200 lines" hint
            // gets ignored). Stable within the session → stays in the volatile
            // tail, before the date grounding, so the KV prefix still hits.
            systemPrompt = Self.composeSystemPrompt(stable: systemPrompt,
                                                    volatileTail: agentVolatileTail + AgentPrompt.outputBudgetGuidance(maxTokens: appState.maxTokens, contextLength: contextLength),
                                                    grounding: grounding)
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
                mcpToolsJSON: mcpToolsJSON,
                docsToolJSON: config.documentIndex != nil ? AgentPrompt.searchDocumentsToolJSON : nil
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
                self.beginLiveTokenCount()
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .content(let text):
                        self.streamDelta(content: text, coalescer: &coalescer, to: sessionId)
                    case .reasoning(let text):
                        self.streamDelta(reasoning: text, coalescer: &coalescer, to: sessionId)
                    case .usage(let usage):
                        self.applyStreamBatch(coalescer.drain(), to: sessionId)
                        appState.updateLastMessage(in: sessionId, usage: usage)
                        self.liveCompletionTokens = usage.completionTokens   // reconcile to the authoritative count
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
                let nudge = ChatMessage(role: .user, content: Self.truncatedToolCallNudge)
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

                // Truncation, NOT a ghost. When the cap was hit AND the content
                // still carries a tool-call opener with no matching close, the
                // call was cut off mid-emission and the server's parser couldn't
                // recover it (Step 1 recovers Hermes/XML, but a future format
                // could escape it). This is a *truncation* — route it to the
                // chunk/append nudge (budget 2) instead of falling through to the
                // ghost path's "call it with proper JSON" (useless: the JSON was
                // fine, just too long). Must precede the ghost check, which also
                // matches `<function=`/`<tool_call>`.
                if maxTokensHit && Self.hasUnclosedToolCallOpener(lastContent) && retry.allowTruncationRetry() {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    let nudge = ChatMessage(role: .user, content: Self.truncatedToolCallNudge)
                    appState.appendMessage(to: sessionId, message: nudge)
                    continue
                }
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
            var roundHandles: [String] = []
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
                    mcpRouter: mcpManager,
                    documentIndex: config.documentIndex,
                    createTask: { [weak self] goal, schedule in
                        await self?.createTaskFromAgent(
                            goal: goal, schedule: schedule,
                            telegramChatId: config.telegramChatId
                        ) ?? "Error: task creation unavailable."
                    },
                    processRegistry: appState.processRegistry,
                    sessionId: sessionId
                )
                roundOutputs.append(result.output)
                if let handle = result.backgroundHandle { roundHandles.append(handle) }

                // Build the model-facing tool message FIRST, so the visible
                // summary can mirror it 1:1 (same content the model receives —
                // no separate, smaller display cap).
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

                // Visible summary (display-only) — exactly what the model sees.
                var resultMsg = ChatMessage(role: .assistant,
                    content: AgentEngine.toolResultSummary(name: result.name, modelContent: toolMsg.content))
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)
                appState.appendMessage(to: sessionId, message: toolMsg)
            }

            // Attach any background-process handles this round started to the
            // call-summary message (located by the captured summaryId) so the
            // tool-call card can render a kill X for each live process.
            if !roundHandles.isEmpty,
               let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == summaryId }) {
                var handles = appState.chatSessions[sIdx].messages[mIdx].processHandles ?? []
                handles.append(contentsOf: roundHandles)
                appState.chatSessions[sIdx].messages[mIdx].processHandles = handles
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

    // MARK: - createTask tool (agent-scheduled background tasks)

    /// Backs the agent's `createTask` tool: build a `ScheduledTask` from a goal +
    /// optional natural-language schedule and hand it to the `TaskScheduler`. A
    /// one-shot ("now" / omitted) is created disabled and run immediately; a
    /// recurring schedule is added enabled and fires on its own. `telegramChatId`
    /// (set on Telegram-driven turns) is stamped on the task so each finished run
    /// reports back to that chat. Returns a short confirmation / error string for
    /// the model to relay.
    func createTaskFromAgent(goal: String, schedule: String?, telegramChatId: Int64?) async -> String {
        let trimmedGoal = goal.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedGoal.isEmpty else {
            return "Error: createTask needs a non-empty \"goal\" — the full instruction the task should carry out (it has no memory of this conversation)."
        }
        let scheduler = appState.taskScheduler
        let title = TaskScheduler.deriveTitle(from: trimmedGoal)
        let willNotify = telegramChatId != nil ? "message you here" : "notify you on the desktop"
        // Inherit MCP exposure from the originating context: the bot's MCP toggle
        // for Telegram-created tasks, the app's MCP mode for in-app ones.
        let useMCP = telegramChatId != nil ? appState.serverOptions.telegram.useMCP : appState.mcpMode

        switch TaskScheduler.scheduleIntent(schedule) {
        case .invalid:
            return "Couldn't understand the schedule “\(schedule ?? "")”. Use a phrase like “every day at 9am”, “every hour”, or “Mon Wed Fri at 8am” — or omit it / say “now” to run once immediately."
        case .once:
            let now = Date()
            let cal = Calendar.current
            let task = ScheduledTask(
                title: title, goal: trimmedGoal,
                trigger: .dailyAt(hour: cal.component(.hour, from: now),
                                  minute: cal.component(.minute, from: now)),
                scheduleText: "once", autonomy: .fullAuto, useMCP: useMCP, enabled: false,
                originTelegramChatId: telegramChatId, deleteAfterRun: true)
            scheduler.addTask(task)
            scheduler.runNow(task)
            return "✅ Created and started task “\(title)”. I'll \(willNotify) with the result when it finishes."
        case .recurring(let trigger):
            let task = ScheduledTask(
                title: title, goal: trimmedGoal, trigger: trigger,
                scheduleText: schedule?.trimmingCharacters(in: .whitespacesAndNewlines),
                autonomy: .fullAuto, useMCP: useMCP, enabled: true,
                originTelegramChatId: telegramChatId)
            scheduler.addTask(task)
            return "✅ Scheduled task “\(title)” to run \(ScheduleParser.describe(trigger)). I'll \(willNotify) with each result."
        }
    }

    // MARK: - Tool JSON + multimodal content (relocated from ChatDetailView)

    /// Build the JSON tools array sent to the model. Concatenates agent tools (when agent mode is on),
    /// MCP tools (when MCP mode is on), and the searchDocuments tool (when a folder is attached).
    /// Returns nil when no tools should be advertised.
    /// `nonisolated` so it stays a pure helper callable off the main actor (unit tests).
    /// Assemble the system prompt so cacheable content stays first and volatile
    /// content lasts. `stable` (base instructions + memory instructions + MCP +
    /// attached-docs, with the template's tool block rendered in front of all of
    /// it) must be byte-identical across a session for the server's KV prefix
    /// cache to reuse the tool+instruction block. `volatileTail` (matched skills,
    /// working-dir listing, learned recent-dirs/commands) and `grounding` (date +
    /// LAN IP) change mid-session, so they go LAST — a change there re-prefills
    /// only the short tail, not the big cached prefix. Pure → unit-tested.
    nonisolated static func composeSystemPrompt(stable: String, volatileTail: String, grounding: String) -> String {
        var p = stable + volatileTail
        if !grounding.isEmpty { p += "\n\n" + grounding }
        return p
    }

    /// Nudge for a tool call cut off by the token cap (the call was NOT
    /// executed). Shared by the two truncation paths: when the server still
    /// parsed the (truncated) call (args incomplete) and when it dropped it
    /// entirely (a format the parser couldn't recover, leaving only the opener
    /// in content). Steers the model to chunk + append rather than retry the
    /// same one-shot write or switch to an equally-capped heredoc.
    nonisolated static let truncatedToolCallNudge = "[System: Your last response was cut off because the output was too long, so the tool call was NOT executed. Write shorter responses: for a large file, write it in chunks — writeFile a first part, then writeFile again with append:\"true\" for each remaining chunk. (A shell heredoc has the same length limit, so don't switch to that.)]"

    /// True when `content` carries a tool-call OPENER with no matching close —
    /// the signature of a call cut off by the token cap (a *truncation*, not a
    /// malformed/"ghost" call). Routes maxTokensHit-with-empty-calls to the
    /// chunk/append nudge instead of the useless "call it with proper JSON"
    /// ghost nudge. Covers Hermes `<function=`→`</function>`, the
    /// `<tool_call>`/`<tool_calls>` wrappers→`</…>`, and Gemma 4
    /// `<|tool_call>`→`<tool_call|>`. Pure → unit-tested.
    nonisolated static func hasUnclosedToolCallOpener(_ content: String) -> Bool {
        func openNoClose(_ open: String, _ close: String) -> Bool {
            content.contains(open) && !content.contains(close)
        }
        return openNoClose("<function=", "</function>")
            || openNoClose("<tool_call>", "</tool_call>")
            || openNoClose("<tool_calls>", "</tool_calls>")
            || openNoClose("<|tool_call>", "<tool_call|>")
    }

    nonisolated static func combinedToolsJSON(agentMode: Bool, mcpToolsJSON: String?,
                                              docsToolJSON: String? = nil) -> String? {
        // Strip each array's outer brackets, drop empties, re-wrap as one array.
        let parts = [agentMode ? AgentPrompt.toolDefinitionsJSON : nil, mcpToolsJSON, docsToolJSON]
            .compactMap { $0 }
            .map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            .filter { !$0.isEmpty }
        guard !parts.isEmpty else { return nil }
        return "[\(parts.joined(separator: ","))]"
    }

    /// File count for prompt text — falls back to indexing-time totals so the
    /// prompt is sensible even if a turn races the tail of indexing.
    private func indexedFileCount(_ index: DocumentIndex?) -> Int {
        switch index?.state {
        case .ready(let files, _): return files
        case .indexing(_, let total): return total
        case .preparing, .failed, nil: return 0
        }
    }

    /// Build OpenAI-style content blocks for a message with images (and,
    /// optionally, audio). Delegates to the pure, unit-tested `MultimodalContent`
    /// builder. Two overloads so the `buildAgentHistory` closure (images only)
    /// and the plain-chat path (images + audio) can both reference it.
    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage], serverPreprocess: Bool = false) -> Any {
        MultimodalContent.build(text: text, images: images, audio: [], serverPreprocess: serverPreprocess)
    }

    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage], audio: [ChatAudio], serverPreprocess: Bool = false) -> Any {
        MultimodalContent.build(text: text, images: images, audio: audio, serverPreprocess: serverPreprocess)
    }

    /// Whether the loaded model wants server-side image preprocessing (Qwen3-VL):
    /// its `x-mlx-pixels` square format is Gemma-only, so Qwen sends raw images.
    var wantsServerImagePreprocess: Bool {
        (server.modelInfo?.architecture ?? "").hasPrefix("qwen")
    }
}
