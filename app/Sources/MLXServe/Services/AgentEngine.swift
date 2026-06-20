import Foundation

/// The slice of MCP behavior the agent executor needs, so `AgentEngine` can run
/// MCP tools through the *same* repetition guard as built-in tools without
/// depending on the concrete `MCPManager`. Production passes the real manager;
/// tests pass a fake. `MCPManager` already provides both methods.
@MainActor
protocol MCPToolRouting {
    func owns(toolName: String) -> Bool
    func executeToolCall(name: String, arguments: [String: String], rawArguments: String) async -> String
}

/// Shared agent engine — history building, tool execution, repetition tracking.
/// Used by both ChatView (production UI) and TestServer (test automation).
/// Eliminates duplication between the two, ensuring bug fixes apply everywhere.
@MainActor
enum AgentEngine {

    // MARK: - Token Estimation

    /// Rough token estimation: ~4 bytes per token (no tokenizer needed).
    static func roughTokenCount(_ s: String) -> Int {
        max(1, s.utf8.count / 4)
    }

    /// Estimate token cost for a message including role/format overhead.
    static func tokenCostForMessage(_ msg: ChatMessage) -> Int {
        var cost = 4  // role + formatting envelope
        cost += roughTokenCount(msg.content)
        if let tcs = msg.toolCalls {
            for tc in tcs {
                cost += roughTokenCount(tc.name) + roughTokenCount(tc.arguments) + 8
            }
        }
        return cost
    }

    // MARK: - Context Helpers

    /// Determine effective context length from user config or model metadata.
    static func effectiveContextLength(appContextSize: Int, modelContextLength: Int?) -> Int {
        if appContextSize > 0 { return appContextSize }
        if let modelCtx = modelContextLength, modelCtx > 0 { return modelCtx }
        return 32768  // safe default
    }

    // MARK: - History Building

    /// Build API-ready message history from chat messages with budget-aware truncation.
    ///
    /// Pins all user messages (they carry critical facts — name, preferences, task
    /// instructions) and the first assistant response (the plan). Walks backward from
    /// newest messages to fill the remaining budget. Progressively truncates older
    /// tool results when context is tight.
    ///
    /// - Parameters:
    ///   - messages: All chat messages in the session.
    ///   - contextLength: Effective context window size.
    ///   - maxTokens: Max generation tokens (capped to 40% of context for budget math).
    ///   - buildMultimodalContent: Optional closure to build image content blocks for the
    ///     last user message. Pass nil to skip image handling (e.g. in TestServer).
    static func buildAgentHistory(
        messages allMessages: [ChatMessage],
        contextLength: Int,
        maxTokens: Int,
        buildMultimodalContent: ((String, [ChatImage]) -> Any)? = nil
    ) -> [[String: Any]] {

        // --- Budget calculation ---

        let safetyBuffer = 1024
        let systemPromptCost = roughTokenCount(AgentPrompt.systemPrompt + AgentPrompt.memory)
        // Cap generation reservation to 40% of context so history always gets ≥60%.
        // The actual max_tokens sent to the API is unchanged — this only affects budget math.
        let effectiveMaxTokens = min(maxTokens, contextLength * 2 / 5)
        let budget = max(1024, contextLength - effectiveMaxTokens - safetyBuffer - systemPromptCost)

        // --- Pinning ---

        // Find the first user message and first assistant response (the plan).
        let firstUserIdx = allMessages.firstIndex { $0.role == .user && $0.toolCallId == nil }
        let firstAssistantIdx: Int? = {
            guard let uIdx = firstUserIdx else { return nil }
            let afterUser = allMessages.index(after: uIdx)
            guard afterUser < allMessages.count else { return nil }
            return allMessages[afterUser...].firstIndex {
                $0.role == .assistant && !$0.isAgentSummary && !$0.content.isEmpty
            }
        }()

        // Pin ALL user messages — they're short but carry critical context.
        let allUserIndices = allMessages.indices.filter {
            allMessages[$0].role == .user && allMessages[$0].toolCallId == nil
        }

        var userPinCost = 0
        for idx in allUserIndices {
            userPinCost += roughTokenCount(allMessages[idx].content) + 4
        }

        // Safety cap: if user messages exceed 30% of budget, pin only first + last
        let userBudgetCap = budget * 3 / 10
        let pinnedUserIndices: [Int]
        if userPinCost > userBudgetCap && allUserIndices.count > 2 {
            pinnedUserIndices = [allUserIndices.first!, allUserIndices.last!]
            userPinCost = pinnedUserIndices.reduce(0) {
                $0 + roughTokenCount(allMessages[$1].content) + 4
            }
        } else {
            pinnedUserIndices = allUserIndices
        }

        // Include first assistant response (plan) in pinned cost
        var pinnedCost = userPinCost
        if let idx = firstAssistantIdx {
            let content = allMessages[idx].content
            pinnedCost += roughTokenCount(String(content.prefix(500))) + 4
            if let tcs = allMessages[idx].toolCalls {
                for tc in tcs {
                    pinnedCost += roughTokenCount(tc.name) + roughTokenCount(tc.arguments) + 8
                }
            }
        }

        // --- Backward walk ---

        var remainingBudget = budget - pinnedCost
        var includeStartIdx = allMessages.count

        for i in stride(from: allMessages.count - 1, through: 0, by: -1) {
            let msg = allMessages[i]
            if msg.role == .system && msg.toolCallId == nil { continue }
            if msg.isAgentSummary { continue }
            if msg.failedRetry { continue }
            if msg.role == .assistant && msg.content.contains("couldn't generate a response") { continue }

            let cost = tokenCostForMessage(msg)
            if cost > remainingBudget { break }
            remainingBudget -= cost
            includeStartIdx = i
        }

        // Which pinned messages fell outside the backward-walk window?
        let pinnedUserOutside = pinnedUserIndices.filter { $0 < includeStartIdx }
        let needsPinAssistant = firstAssistantIdx != nil && firstAssistantIdx! < includeStartIdx

        // --- Auto-compact ---

        // When context is squeezed, truncate tool results more aggressively.
        let freeRatio = Double(remainingBudget + pinnedCost) / Double(budget + pinnedCost)
        let squeezed = freeRatio < 0.25
        let recentLimit = squeezed ? 500 : 2000
        let olderLimit = squeezed ? 100 : 500

        let window = Array(allMessages[includeStartIdx..<allMessages.count])
        let totalToolResults = window.filter { $0.toolCallId != nil }.count
        var toolResultsSeen = 0

        // --- Assemble history ---

        var history: [[String: Any]] = []

        // Emit pinned user messages that fell outside the window (in original order).
        // Strip images — server only processes the last user message's images.
        for idx in pinnedUserOutside {
            history.append(["role": "user", "content": allMessages[idx].content])
        }

        // Emit pinned first assistant response (the plan) if it fell outside.
        if needsPinAssistant, let idx = firstAssistantIdx {
            let msg = allMessages[idx]
            var content = msg.content
                .replacingOccurrences(of: "<pad>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if content.count > 500 {
                content = String(content.prefix(500)) + "..."
            }
            var dict: [String: Any] = ["role": "assistant"]
            if !content.isEmpty { dict["content"] = content }
            if let tcs = msg.toolCalls, !tcs.isEmpty {
                dict["tool_calls"] = tcs.map { tc -> [String: Any] in
                    [
                        "id": tc.id,
                        "type": "function",
                        "function": [
                            "name": tc.name,
                            "arguments": tc.arguments
                        ] as [String: Any]
                    ]
                }
            }
            history.append(dict)
        }

        // Only the last user message gets image content blocks.
        let lastUserMsgId: UUID? = buildMultimodalContent != nil
            ? window.last(where: { $0.role == .user && $0.toolCallId == nil })?.id
            : nil

        // Emit messages from the backward-walk window.
        for msg in window {
            if msg.failedRetry { continue }
            // Tool responses — progressive truncation
            if let callId = msg.toolCallId {
                let isRecent = toolResultsSeen >= totalToolResults - 2
                let limit = isRecent ? recentLimit : olderLimit
                toolResultsSeen += 1
                history.append([
                    "role": "tool",
                    "tool_call_id": callId,
                    "content": String(msg.content.prefix(limit)),
                ])
                continue
            }
            if msg.role == .system { continue }
            if msg.isAgentSummary { continue }
            if msg.role == .assistant && msg.content.contains("couldn't generate a response") { continue }

            // Assistant messages with tool_calls
            if msg.role == .assistant, let tcs = msg.toolCalls, !tcs.isEmpty {
                var dict: [String: Any] = ["role": "assistant"]
                let content = msg.content
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                dict["content"] = content.isEmpty ? "" : content
                dict["tool_calls"] = tcs.map { tc -> [String: Any] in
                    [
                        "id": tc.id,
                        "type": "function",
                        "function": [
                            "name": tc.name,
                            "arguments": tc.arguments
                        ] as [String: Any]
                    ]
                }
                history.append(dict)
                continue
            }

            // Regular messages
            if msg.role == .assistant && msg.content.isEmpty { continue }
            var content = msg.content
                .replacingOccurrences(of: "<pad>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if msg.role == .assistant && content.count > 500 {
                content = String(content.prefix(500)) + "..."
            }
            if content.isEmpty { continue }

            var dict: [String: Any] = ["role": msg.role.rawValue]
            if let multimodal = buildMultimodalContent,
               msg.id == lastUserMsgId,
               let imgs = msg.images, !imgs.isEmpty {
                dict["content"] = multimodal(content, imgs)
            } else {
                dict["content"] = content
            }
            history.append(dict)
        }

        return history
    }

    // MARK: - Tool Validation

    /// Check which required params are missing for a tool call.
    /// Recover a file body a model crammed into the `append` value. Gemma 4 12B
    /// (live, 2026-06-20) emits `{"append":"true,\n<entire file body>","path":…}`
    /// with NO `content` key — merging the flag and the body into one string —
    /// which left `content` "missing" and looped the agent on a bogus error. When
    /// `content` is absent/blank AND `append` is a boolean keyword (`true`/`false`)
    /// FOLLOWED by a separator and more text, split it: the leading boolean is the
    /// flag, the remainder (past a comma/whitespace/newline) becomes `content`. A
    /// clean boolean (`"true"` with nothing after it) is left untouched so a
    /// genuinely missing content still surfaces as an error rather than a fabricated
    /// empty file. Pure → unit-tested.
    nonisolated static func normalizeWriteFileArgs(_ arguments: [String: String]) -> [String: String] {
        let contentBlank = (arguments["content"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard contentBlank, let appendVal = arguments["append"] else { return arguments }
        let lower = appendVal.lowercased()
        let afterFlag: Substring
        let flag: String
        if lower.hasPrefix("true") {
            flag = "true"; afterFlag = appendVal.dropFirst("true".count)
        } else if lower.hasPrefix("false") {
            flag = "false"; afterFlag = appendVal.dropFirst("false".count)
        } else {
            return arguments
        }
        // The char right after the boolean must be a separator (or end) — else
        // this is a real word like "truely", not a flag-plus-body jam.
        if let first = afterFlag.first, !",\n\r\t ".contains(first) { return arguments }
        let body = String(afterFlag.drop(while: { ",\n\r\t ".contains($0) }))
        guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return arguments }
        var out = arguments
        out["append"] = flag
        out["content"] = body
        return out
    }

    static func missingRequiredParams(for toolName: String, arguments: [String: String]) -> [String] {
        for def in AgentPrompt.toolDefinitions {
            guard let fn = def["function"] as? [String: Any],
                  fn["name"] as? String == toolName,
                  let params = fn["parameters"] as? [String: Any],
                  let required = params["required"] as? [String] else { continue }
            return required.filter { key in
                guard let val = arguments[key] else { return true }
                return val.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
        }
        return []
    }

    /// Get the example JSON format from a tool's description.
    static func toolExample(for toolName: String) -> String {
        for def in AgentPrompt.toolDefinitions {
            guard let fn = def["function"] as? [String: Any],
                  fn["name"] as? String == toolName,
                  let desc = fn["description"] as? String,
                  let range = desc.range(of: "Example: ") else { continue }
            return String(desc[range.upperBound...])
        }
        return "{}"
    }

    // MARK: - Tool Repetition

    /// Write/control tools are never warned or blocked — they make forward progress.
    static let exemptTools: Set<String> = ["writeFile", "editFile", "shell", "cwd"]

    /// Tracks tool repetition across agent loop iterations.
    ///
    /// Three-phase system:
    /// 1. **Warning at 5 in 12**: Tool executes normally, result prefixed with warning.
    /// 2. **Soft block at 8 in 12**: Tool blocked for 3 iterations, then available again.
    /// 3. **Escalation**: Calling blocked tool during cooldown extends cooldown by 5.
    ///
    /// Tracking is arg-aware: `listFiles("src")` and `listFiles("lib")` are different entries.
    @MainActor
    class RepetitionTracker {
        var recentKeys: [String] = []       // sliding window of "name:arg" keys
        var warnings: Set<String> = []      // keys that have been warned
        var blockedUntil: [String: Int] = [:] // tool key → iteration when block expires

        /// Record tool calls from this round into the sliding window.
        func track(toolCalls: [APIClient.ToolCall]) {
            for tc in toolCalls {
                guard !AgentEngine.exemptTools.contains(tc.name) else { continue }
                recentKeys.append(AgentEngine.toolRepetitionKey(name: tc.name, arguments: tc.arguments))
            }
            if recentKeys.count > 12 {
                recentKeys = Array(recentKeys.suffix(12))
            }
        }

        /// Check if a tool call is blocked. Returns true if blocked.
        func isBlocked(name: String, arguments: [String: String], iteration: Int) -> Bool {
            guard !AgentEngine.exemptTools.contains(name) else { return false }
            let key = AgentEngine.toolRepetitionKey(name: name, arguments: arguments)

            // Check existing cooldown — extend on violation
            if let until = blockedUntil[key], iteration < until {
                blockedUntil[key] = until + 5
                return true
            }

            // Soft block at 8 occurrences in window
            if recentKeys.filter({ $0 == key }).count >= 8 {
                blockedUntil[key] = iteration + 3
                return true
            }

            return false
        }

        /// Apply warning prefix if tool key has appeared 5+ times. Returns modified output.
        func applyWarning(name: String, arguments: [String: String], output: String) -> String {
            guard !AgentEngine.exemptTools.contains(name), !output.hasPrefix("BLOCKED:") else {
                return output
            }
            let key = AgentEngine.toolRepetitionKey(name: name, arguments: arguments)
            let count = recentKeys.filter { $0 == key }.count
            guard count >= 5, !warnings.contains(key) else { return output }

            warnings.insert(key)
            let alt = AgentEngine.toolAlternativeSuggestion(for: name)
            return "WARNING: \(name) with these arguments has been called \(count) times recently. "
                + "Consider \(alt). Continued repetition will cause this tool to be temporarily blocked.\n\n"
                + output
        }
    }

    /// Primary argument key per tool — used to distinguish different invocations.
    private static let primaryArgKey: [String: String] = [
        "listFiles": "path", "readFile": "path", "searchFiles": "pattern",
        "browse": "url", "webSearch": "query", "searchDocuments": "query",
    ]

    /// Build a repetition key like "listFiles:src/lib" or "saveMemory" (no primary arg).
    static func toolRepetitionKey(name: String, arguments: [String: String]) -> String {
        if let argKey = primaryArgKey[name], let val = arguments[argKey], !val.isEmpty {
            return "\(name):\(val)"
        }
        // No designated primary arg (MCP tools like `dbhub__execute_sql`,
        // saveMemory, …): key on the FULL argument set so only *identical* calls
        // count as repetition. Distinct calls (e.g. different SQL queries) stay
        // independent and aren't over-blocked once they flow through the shared
        // block guard. Keys are sorted for a stable, dict-order-independent key.
        if arguments.isEmpty { return name }
        let serialized = arguments.keys.sorted()
            .map { "\($0)=\(arguments[$0] ?? "")" }
            .joined(separator: "\u{1}")
        return "\(name):\(serialized)"
    }

    /// Tool-specific block messages suggesting useful alternatives.
    static func toolBlockMessage(for toolName: String) -> String {
        let prefix = "BLOCKED: \(toolName) has been called too many times with the same arguments."
        switch toolName {
        case "listFiles":
            return "\(prefix) Use shell with `ls` instead: {\"command\": \"ls -la path/to/dir\"}"
        case "readFile":
            return "\(prefix) Use shell with `cat` instead: {\"command\": \"cat path/to/file\"}"
        case "searchFiles":
            return "\(prefix) Use shell with `grep -r` instead: {\"command\": \"grep -r 'pattern' path/\"}"
        case "browse":
            return "\(prefix) Try a different URL, or use webSearch to find what you need."
        case "webSearch":
            return "\(prefix) Try a different search query, or use browse to visit a specific URL."
        default:
            return "\(prefix) This tool is temporarily disabled. Try a different approach using shell."
        }
    }

    /// Suggest alternatives for the warning phase.
    static func toolAlternativeSuggestion(for toolName: String) -> String {
        switch toolName {
        case "listFiles": return "using shell with `ls` or trying a different path"
        case "readFile": return "using shell with `cat` or reading a different file"
        case "searchFiles": return "using shell with `grep -r` or a different search pattern"
        case "browse": return "trying a different URL or using webSearch"
        case "webSearch": return "trying a different query or using browse"
        default: return "a different approach"
        }
    }

    // MARK: - Tool Execution

    /// Result of executing a single tool call.
    struct ToolResult {
        let id: String
        let name: String
        var output: String
        /// Handle of a background process this call started/adopted (a `shell`
        /// with `run_in_background:"true"`, or the foreground backstop). Flows up
        /// to the chat tool-call card so it can show a kill X. nil otherwise.
        var backgroundHandle: String? = nil
    }

    /// Resolve a model-emitted tool name to a known tool, tolerating the common
    /// quirks that otherwise dead-loop the agent on "Unknown tool": surrounding
    /// whitespace, a trailing ':' (Gemma 4 12B leaks one via its `call:NAME:`
    /// format), a `functions.`/`tool.` namespace prefix, and case differences.
    /// Returns the canonical `AgentToolKind.rawValue` when a known tool matches,
    /// otherwise the cleaned name (so genuinely-unknown tools still surface).
    static func canonicalToolName(_ raw: String) -> String {
        var name = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["functions.", "function.", "tools.", "tool."] {
            if name.lowercased().hasPrefix(prefix) {
                name = String(name.dropFirst(prefix.count)).trimmingCharacters(in: .whitespacesAndNewlines)
                break
            }
        }
        while name.hasSuffix(":") {
            name = String(name.dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if AgentToolKind(rawValue: name) != nil { return name }
        if let match = AgentToolKind.allCases.first(where: {
            $0.rawValue.caseInsensitiveCompare(name) == .orderedSame
        }) {
            return match.rawValue
        }
        return name
    }

    /// Detects when the agent loop is making no progress — consecutive rounds in
    /// which every tool call failed or was blocked — so the loop can stop with a
    /// summary instead of grinding to the iteration cap on an unrecoverable name.
    struct StuckDetector {
        /// Consecutive no-progress rounds that trip the bail-out.
        static let limit = 5
        private(set) var consecutiveNoProgress = 0

        /// Record a round's tool outputs. A round counts as "no progress" when it
        /// ran at least one tool and *every* output was a failure or block.
        mutating func record(outputs: [String]) {
            guard !outputs.isEmpty else { return } // no tools ran → not a signal
            let allFailed = outputs.allSatisfy { StuckDetector.isFailure($0) }
            consecutiveNoProgress = allFailed ? consecutiveNoProgress + 1 : 0
        }

        var isStuck: Bool { consecutiveNoProgress >= StuckDetector.limit }

        /// An output that represents an unrecoverable or blocked tool call. Uses
        /// `contains` (not `hasPrefix`) because `applyWarning` prepends a
        /// "WARNING:" banner ahead of the underlying error.
        static func isFailure(_ output: String) -> Bool {
            output.contains("Error: Unknown tool")
                || output.contains("BLOCKED:")
                || output.hasPrefix("Error:")
        }
    }

    /// Per-turn budget for *recoverable* agent-loop failures: malformed/"ghost"
    /// tool calls, truncated tool-call args, and empty/pad responses.
    ///
    /// Counts **consecutive** failures, not lifetime-of-turn ones — a round that
    /// executes real tool calls calls `recordProgress()` to clear it. Without
    /// this, one early bad tool call (e.g. an `editFile` with unescaped quotes)
    /// permanently spent the budget, so the *next* bad call — even after lots of
    /// successful work — ended the turn: the "agent just stops mid-task" bug.
    struct AgentRetryBudget {
        private(set) var ghost = 0
        private(set) var truncation = 0
        /// Empty/pad-response retries. The limit + backoff live with the caller's
        /// `RetryPolicy`, so this stays a plain counter the loop reads and bumps.
        var pad = 0

        static let ghostLimit = 1
        static let truncationLimit = 2

        /// Consume one ghost-tool-call retry; false once the consecutive limit
        /// is hit (the loop then gives up for this stretch).
        mutating func allowGhostRetry() -> Bool {
            guard ghost < Self.ghostLimit else { return false }
            ghost += 1
            return true
        }

        /// Consume one truncated-tool-call retry; false once the limit is hit.
        mutating func allowTruncationRetry() -> Bool {
            guard truncation < Self.truncationLimit else { return false }
            truncation += 1
            return true
        }

        /// A round executed real tool calls — clear the consecutive-failure budget.
        mutating func recordProgress() {
            ghost = 0
            truncation = 0
            pad = 0
        }
    }

    /// Shared tool handler instances. Handlers are stateless — safe to reuse.
    private static var toolHandlers: [AgentToolKind: any ToolHandler] {
        [
            .shell: ShellHandler(),
            .readFile: ReadFileHandler(),
            .writeFile: WriteFileHandler(),
            .editFile: EditFileHandler(),
            .searchFiles: SearchFilesHandler(),
            .listFiles: ListFilesHandler(),
            .browse: BrowseHandler(),
            .webSearch: WebSearchHandler(),
            .saveMemory: SaveMemoryHandler(),
        ]
    }

    /// Execute a single tool call with cwd handling, validation, blocking, and warnings.
    ///
    /// Handles the full pipeline:
    /// 1. Smart fallback (editFile without find → writeFile)
    /// 2. Working directory changes (cwd tool)
    /// 3. Repetition blocking check
    /// 4. Required parameter validation
    /// 5. Handler dispatch
    /// 6. Warning injection
    static func executeToolCall(
        _ tc: APIClient.ToolCall,
        workingDirectory: inout String?,
        repetition: RepetitionTracker,
        iteration: Int,
        agentMemory: AgentMemory,
        mcpRouter: (any MCPToolRouting)? = nil,
        documentIndex: DocumentIndex? = nil,
        createTask: ((_ goal: String, _ schedule: String?) async -> String)? = nil,
        processRegistry: ProcessRegistry? = nil,
        sessionId: UUID? = nil
    ) async -> ToolResult {
        // Normalize the model-emitted name (strip a leaked trailing ':' etc.)
        // before resolving the tool. Repetition tracking stays on the raw
        // `tc.name` to match the caller's `RepetitionTracker.track(toolCalls:)`.
        let name = canonicalToolName(tc.name)

        // createTask is a meta-tool: it schedules an unattended run via the
        // TaskScheduler, which this static dispatcher can't reach. The caller
        // (ChatTurnEngine, which owns AppState) injects a closure; absent it, the
        // tool is gracefully unavailable. Handled before the repetition / MCP /
        // built-in dispatch since it isn't a ToolHandler and needs no workdir.
        if name == "createTask" {
            let out = createTask != nil
                ? await createTask!(tc.arguments["goal"] ?? "", tc.arguments["schedule"])
                : "Error: createTask isn't available in this context."
            return ToolResult(id: tc.id, name: name, output: out)
        }

        // ── Single repetition guard for ALL tools — built-in and MCP alike ──
        // MCP execution used to live in a separate branch that skipped this
        // entirely, so the model could loop on the same MCP call forever (the
        // 26B `dbhub__search_objects` ×31). Routing both kinds through here keeps
        // the block/warning logic in one place. `cwd` is exempt (see exemptTools)
        // so isBlocked short-circuits false for it.
        let output: String
        // Captures the handle of any background process the shell tool starts or
        // adopts this call, so it can ride back on ToolResult.backgroundHandle.
        let handleBox = ProcessHandleBox()
        if repetition.isBlocked(name: tc.name, arguments: tc.arguments, iteration: iteration) {
            output = toolBlockMessage(for: tc.name)
        } else if let mcpRouter, mcpRouter.owns(toolName: tc.name) {
            output = await mcpRouter.executeToolCall(
                name: tc.name, arguments: tc.arguments, rawArguments: tc.rawArguments)
        } else {
            output = await executeBuiltinTool(tc, name: name, workingDirectory: &workingDirectory,
                                              agentMemory: agentMemory, documentIndex: documentIndex,
                                              processRegistry: processRegistry, sessionId: sessionId,
                                              handleBox: handleBox)
        }

        // Apply warning if near repetition threshold (raw name — see above).
        // No-ops on a "BLOCKED:" output, so blocked calls pass through unchanged.
        let warned = repetition.applyWarning(name: tc.name, arguments: tc.arguments, output: output)
        return ToolResult(id: tc.id, name: name, output: warned, backgroundHandle: handleBox.handle)
    }

    /// Dispatch a built-in (non-MCP) tool: cwd, param validation, then the
    /// registered handler. Returns the raw output; the shared repetition guard
    /// and warning are applied by the caller (`executeToolCall`).
    private static func executeBuiltinTool(
        _ tc: APIClient.ToolCall,
        name: String,
        workingDirectory: inout String?,
        agentMemory: AgentMemory,
        documentIndex: DocumentIndex? = nil,
        processRegistry: ProcessRegistry? = nil,
        sessionId: UUID? = nil,
        handleBox: ProcessHandleBox? = nil
    ) async -> String {
        let tool = AgentToolKind(rawValue: name)

        // Smart fallback: editFile with content but no find → writeFile
        let effectiveTool: AgentToolKind?
        if tool == .editFile && tc.arguments["content"] != nil && tc.arguments["find"] == nil {
            effectiveTool = .writeFile
        } else {
            effectiveTool = tool
        }

        // For writeFile, recover a body the model crammed into `append` (Gemma 4
        // 12B emits `{"append":"true,\n<body>",…}` with no content key). No-op
        // when the call is already well-formed or it's not writeFile.
        let args = (effectiveTool == .writeFile) ? normalizeWriteFileArgs(tc.arguments) : tc.arguments

        if effectiveTool == .cwd {
            // Change working directory for subsequent calls
            guard let path = tc.arguments["path"] else {
                return "Error: cwd requires a path parameter. Example: {\"path\": \"myproject\"}"
            }
            let resolved: String
            if path.hasPrefix("/") || path.hasPrefix("~") {
                resolved = NSString(string: path).expandingTildeInPath
            } else if let wd = workingDirectory {
                resolved = (wd as NSString).appendingPathComponent(path)
            } else {
                resolved = path
            }
            let normalized = (resolved as NSString).standardizingPath
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: normalized, isDirectory: &isDir), isDir.boolValue {
                workingDirectory = normalized
                return "Changed working directory to \(normalized)"
            }
            return "Error: '\(normalized)' is not a directory"
        }

        // Validate required parameters
        let missing = missingRequiredParams(for: name, arguments: args)
        if !missing.isEmpty {
            if (name == "writeFile" || name == "editFile") && missing.contains("content") {
                // Honest diagnosis: content is absent, NOT necessarily truncated.
                // Real max_tokens truncation is intercepted upstream (ChatTurnEngine)
                // before execution, so reaching here means the body was misplaced or
                // omitted — most often crammed into `append`. Steer the model to the
                // right field instead of falsely claiming its output was cut off.
                return "Error: \(name) was called with no text in the `content` parameter, so nothing was written. Put the full file body in `content` — `append` is only a \"true\"/\"false\" flag and must never hold the text. For a long file, write the first part, then call again with `append`:\"true\" for each remaining chunk. Example: {\"path\": \"jfk.txt\", \"content\": \"<the text>\", \"append\": \"true\"}"
            }
            return "Error: \(name) missing required params: \(missing.joined(separator: ", ")). Example: \(toolExample(for: name))"
        }

        // Document search is stateful (the per-session index), so it dispatches
        // here instead of through the stateless handler registry.
        if effectiveTool == .searchDocuments {
            guard let documentIndex else {
                return "Error: no document folder is attached to this chat. Ask the user to attach one via the paperclip menu (Attach Folder for Q&A)."
            }
            return await documentIndex.search(query: tc.arguments["query"] ?? "")
        }

        // Background-process management — registry-backed, so dispatched inline
        // (like cwd/searchDocuments) rather than through the stateless handlers.
        if effectiveTool == .listProcesses || effectiveTool == .readProcessOutput || effectiveTool == .killProcess {
            return processToolOutput(effectiveTool!, arguments: tc.arguments,
                                     registry: processRegistry, sessionId: sessionId)
        }

        // Shell gets a fresh handler with the registry injected (the static
        // `toolHandlers` entry is stateless) so background-start and the
        // timeout-adopt backstop can register processes and report handles.
        if effectiveTool == .shell {
            // Weak models routinely emit a process-management TOOL as a shell
            // COMMAND (`shell {"command":"killProcess{handle:\"bg1\"}"}`), which
            // would just die as "command not found". Re-route it to the real
            // tool so process management works regardless of model capability.
            if let cmd = tc.arguments["command"],
               let reroute = processToolFromShellCommand(cmd) {
                let args = reroute.handle.map { ["handle": $0] } ?? [:]
                return processToolOutput(reroute.tool, arguments: args,
                                         registry: processRegistry, sessionId: sessionId)
            }
            let handler = ShellHandler(registry: processRegistry, sessionId: sessionId, handleBox: handleBox)
            do {
                let output = try await handler.execute(parameters: tc.arguments, workingDirectory: workingDirectory)
                if let cmd = tc.arguments["command"] { agentMemory.recordCommand(cmd) }
                return output
            } catch {
                let argsDesc = tc.arguments.isEmpty ? "none" : tc.arguments.map { "\($0.key)=\($0.value.prefix(30))" }.joined(separator: ", ")
                return "Error: \(error.localizedDescription). You sent args: [\(argsDesc)]. Example: \(toolExample(for: name))"
            }
        }

        guard let effectiveTool, let handler = toolHandlers[effectiveTool] else {
            return "Error: Unknown tool '\(name)'"
        }
        do {
            let output = try await handler.execute(parameters: args, workingDirectory: workingDirectory)
            if effectiveTool == .shell, let cmd = args["command"] {
                agentMemory.recordCommand(cmd)
            }
            return output
        } catch {
            let argsDesc = args.isEmpty ? "none" : args.map { "\($0.key)=\($0.value.prefix(30))" }.joined(separator: ", ")
            return "Error: \(error.localizedDescription). You sent args: [\(argsDesc)]. Example: \(toolExample(for: name))"
        }
    }

    /// Dispatch a background-process tool (`listProcesses`/`readProcessOutput`/
    /// `killProcess`) against the session-scoped registry. Returns a graceful,
    /// model-readable string in every case — unknown handles list the valid ones.
    /// `handle` is already validated present by `missingRequiredParams` for the
    /// two tools that require it.
    static func processToolOutput(_ tool: AgentToolKind, arguments: [String: String],
                                  registry: ProcessRegistry?, sessionId: UUID?) -> String {
        guard let registry else {
            return "Error: background process management isn't available in this context."
        }
        switch tool {
        case .listProcesses:
            let procs = registry.list(sessionId: sessionId)
            guard !procs.isEmpty else { return "No background processes have been started in this chat." }
            return procs.map { "\($0.handle) [\($0.statusLabel)] pid \($0.pid) — \($0.command.prefix(120))" }
                .joined(separator: "\n")
        case .readProcessOutput:
            let handle = arguments["handle"] ?? ""
            guard let out = registry.readOutput(handle: handle) else {
                return unknownHandleError(handle, registry: registry, sessionId: sessionId)
            }
            let status = registry.isAlive(handle: handle) ? "still running" : "exited"
            return out.isEmpty
                ? "(\(handle) \(status); no new output since the last read)"
                : "[\(handle) \(status)]\n\(out)"
        case .killProcess:
            let handle = arguments["handle"] ?? ""
            guard registry.list(sessionId: nil).contains(where: { $0.handle == handle }) else {
                return unknownHandleError(handle, registry: registry, sessionId: sessionId)
            }
            guard registry.isAlive(handle: handle) else {
                return "\(handle) is not running (it already exited or was killed)."
            }
            registry.kill(handle: handle)
            return "Killed \(handle)."
        default:
            return "Error: \(tool.rawValue) is not a process tool."
        }
    }

    /// Recognize a process-management tool that a weak model emitted as a shell
    /// command (first token is the tool name, in any casing). Returns the tool +
    /// the `bgN` handle plucked from anywhere in the command (nil for
    /// listProcesses / when absent). nil when it isn't a process-tool command, so
    /// real shell commands pass straight through. Pure + testable.
    static func processToolFromShellCommand(_ command: String) -> (tool: AgentToolKind, handle: String?)? {
        let trimmed = command.trimmingCharacters(in: .whitespaces)
        let firstToken = trimmed.split(whereSeparator: { " \t{(\"'".contains($0) }).first.map(String.init) ?? ""
        let tool: AgentToolKind
        switch firstToken.lowercased() {
        case "killprocess": tool = .killProcess
        case "readprocessoutput": tool = .readProcessOutput
        case "listprocesses": tool = .listProcesses
        default: return nil
        }
        return (tool, firstBgHandle(in: trimmed))
    }

    /// First `bg<digits>` token anywhere in a string (handles are `bg1`, `bg2`, …).
    static func firstBgHandle(in s: String) -> String? {
        let chars = Array(s)
        var i = 0
        while i + 1 < chars.count {
            if chars[i] == "b", chars[i + 1] == "g" {
                var j = i + 2
                var digits = ""
                while j < chars.count, chars[j].isNumber { digits.append(chars[j]); j += 1 }
                if !digits.isEmpty { return "bg" + digits }
            }
            i += 1
        }
        return nil
    }

    /// Helpful error for a handle that doesn't resolve — lists the live handles.
    private static func unknownHandleError(_ handle: String, registry: ProcessRegistry, sessionId: UUID?) -> String {
        let live = registry.list(sessionId: sessionId).filter { $0.status.isAlive }.map { $0.handle }
        if live.isEmpty {
            return "Error: no process with handle '\(handle)'. There are no running background processes — start one with shell {\"command\": \"…\", \"run_in_background\": \"true\"}."
        }
        return "Error: no process with handle '\(handle)'. Running processes: \(live.joined(separator: ", "))."
    }

    // MARK: - Tool Result Overflow

    /// Per-tool context caps (chars). Oversized results are saved to disk.
    static let toolResultCaps: [String: Int] = [
        "shell": 6000,
        "readFile": 8000,
        "searchFiles": 4000,
        "listFiles": 4000,
        "webSearch": 2000,
        "browse": 3000,
        "editFile": 2000,
        "writeFile": 2000,
        "saveMemory": 500,
        // Top-5 excerpts at ~1200 chars each plus headers — don't truncate
        // the retrieval the whole docs mode depends on.
        "searchDocuments": 10000,
    ]

    /// Truncate tool output, saving full result to disk if oversized.
    static func truncateWithOverflow(_ output: String, toolCallId: String, toolName: String) -> String {
        let maxChars = toolResultCaps[toolName] ?? 4000
        guard output.count > maxChars else { return output }

        // Save full output to disk for debugging
        let dir = NSString(string: "~/.mlx-serve/tool-output").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true, attributes: nil)
        let path = (dir as NSString).appendingPathComponent("\(toolCallId).txt")
        try? output.write(toFile: path, atomically: true, encoding: .utf8)

        let preview = String(output.prefix(maxChars))
        return "\(preview)\n\n[... truncated at \(maxChars) of \(output.count) chars]"
    }

    /// The visible tool-result summary shown in the chat transcript. Built from
    /// the **already-computed model-facing content** (the exact string put on the
    /// tool message the model receives), so the UI shows the result 1:1 with what
    /// the model saw — no separate, smaller display cap. The leading `**name** → `
    /// is the discriminator `ChatRowBuilder` uses to fold the row.
    static func toolResultSummary(name: String, modelContent: String) -> String {
        "**\(name)** → \(modelContent)"
    }

    /// Clean up overflow files older than 24 hours.
    static func cleanupOverflowFiles() {
        let dir = NSString(string: "~/.mlx-serve/tool-output").expandingTildeInPath
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: dir) else { return }
        let cutoff = Date().addingTimeInterval(-86400)
        for file in files {
            let path = (dir as NSString).appendingPathComponent(file)
            if let attrs = try? fm.attributesOfItem(atPath: path),
               let modified = attrs[.modificationDate] as? Date,
               modified < cutoff {
                try? fm.removeItem(atPath: path)
            }
        }
    }

    // MARK: - Workspace Context

    /// Build the working directory section for the system prompt.
    /// Includes the directory path and a listing of its contents so the model
    /// doesn't need to call listFiles on the root — it's always available.
    static func workingDirectoryContext(_ path: String) -> String {
        var section = "\n\n# Working Directory\nYour working directory is `\(path)`. All file tool operations are confined to this directory — paths that resolve outside it will be rejected. Use relative paths. Shell commands run here by default."

        let listing = directoryListing(path)
        if !listing.isEmpty {
            section += "\n\n## Files\nThis listing is always up-to-date — no need to call listFiles on the root directory.\n```\n\(listing)\n```"
        }

        return section
    }

    /// Compact directory listing: top-level entries with type indicators.
    /// Directories end with /, hidden files included, sorted, capped at 100 entries.
    private static func directoryListing(_ path: String) -> String {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: path) else { return "" }

        var entries: [String] = []
        for name in items.sorted() {
            var isDir: ObjCBool = false
            let fullPath = (path as NSString).appendingPathComponent(name)
            if fm.fileExists(atPath: fullPath, isDirectory: &isDir) {
                entries.append(isDir.boolValue ? "\(name)/" : name)
            } else {
                entries.append(name)
            }
            if entries.count >= 100 { break }
        }

        if entries.isEmpty { return "" }
        let suffix = items.count > 100 ? "\n... and \(items.count - 100) more" : ""
        return entries.joined(separator: "\n") + suffix
    }

    // MARK: - Debug

    /// Dump the exact request body to file for analysis.
    static func dumpDebugRequest(messages: [[String: Any]], maxTokens: Int) {
        do {
            let debugPath = NSString(string: "~/.mlx-serve/last-agent-request.json").expandingTildeInPath
            let messagesData = try JSONSerialization.data(withJSONObject: messages, options: .prettyPrinted)
            let messagesStr = String(data: messagesData, encoding: .utf8) ?? "[]"
            let debugJSON = """
            {
              "model": "mlx-serve",
              "max_tokens": \(maxTokens),
              "temperature": 0.7,
              "stream": true,
              "messages": \(messagesStr),
              "tools": \(AgentPrompt.toolDefinitionsJSON)
            }
            """
            try debugJSON.data(using: .utf8)?.write(to: URL(fileURLWithPath: debugPath))
        } catch { /* ignore debug errors */ }
    }
}
