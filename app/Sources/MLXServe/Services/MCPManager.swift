import Foundation
import MCP
import System

#if canImport(AppKit)
import AppKit
#endif

/// Owns the lifecycle of all enabled MCP server subprocesses, aggregates their tools, and routes
/// model-issued tool calls to the right server. Designed to live as an @EnvironmentObject so the
/// chat view and the marketplace sheet share the same instance.
@MainActor
final class MCPManager: ObservableObject, MCPToolRouting {

    /// One running server. Stdio sessions retain the child (a host `Process` or
    /// a `GuestMCPBridge`) and its stderr capture; HTTP sessions only retain the
    /// MCP Client. All stdio fields are nil for HTTP sessions.
    final class Session {
        let id: String
        /// nil for HTTP transport. `MCPChild` so a server running inside the
        /// Agent Sandbox guest satisfies the same contract as a host process.
        let child: MCPChild?
        private let stderrBox: StderrBox?
        let client: Client
        var tools: [Tool]

        /// Rolling stderr capture for surfacing server errors in the UI (stdio
        /// only). The spawner already drains the stream into `StderrBox`, so
        /// this just reads the tail rather than owning a second drain.
        var stderrTail: String { String((stderrBox?.snapshot() ?? "").suffix(4_000)) }

        /// Stdio transport — host process or guest bridge.
        init(id: String, child: MCPChild, stderr: StderrBox, client: Client, tools: [Tool]) {
            self.id = id
            self.child = child
            self.stderrBox = stderr
            self.client = client
            self.tools = tools
        }

        /// HTTP transport — no subprocess, just an MCP Client backed by `HTTPClientTransport`.
        init(id: String, client: Client, tools: [Tool]) {
            self.id = id
            self.child = nil
            self.stderrBox = nil
            self.client = client
            self.tools = tools
        }
    }

    /// Servers that have successfully connected and listed tools. Keyed by server id.
    @Published private(set) var sessions: [String: Session] = [:]

    /// Per-server failure messages for the marketplace UI.
    @Published private(set) var startErrors: [String: String] = [:]

    /// Last config loaded from disk. UI binds to this.
    @Published private(set) var config: MCPConfig = MCPConfigStore.load()

    /// True when at least one start-up attempt is in flight. Used to render a spinner in the UI.
    @Published private(set) var isStarting: Bool = false

    /// Fallback cwd used when an mcp.json entry doesn't pin its own `cwd`. ChatView sets this from
    /// the active chat session's `workingDirectory` before calling `startEnabled()`, so newly-spawned
    /// MCP servers anchor at the same dir the agent is using. Already-running sessions are NOT
    /// restarted when this changes — they'd need an off/on toggle (or app restart) to pick it up.
    var defaultCwd: String?

    // MARK: - Config

    func reloadConfig() {
        config = MCPConfigStore.load()
    }

    func saveConfig(_ newConfig: MCPConfig) throws {
        try MCPConfigStore.save(newConfig)
        config = newConfig
    }

    // MARK: - Lifecycle

    /// Spawn every enabled server. Idempotent — already-connected servers are skipped.
    /// Failures are recorded in `startErrors` and surfaced in the UI; one bad server does not abort the rest.
    func startEnabled() async {
        reloadConfig()
        isStarting = true
        defer { isStarting = false }

        // Purge stale errors from a previous attempt if the entry is now disabled or has been removed
        // entirely. Otherwise a server the user toggled off would keep flashing its old error in chat.
        for id in Array(startErrors.keys) {
            let stillEnabled = config.mcpServers[id]?.isEnabled ?? false
            if !stillEnabled { startErrors.removeValue(forKey: id) }
        }

        for (id, entry) in config.mcpServers where entry.isEnabled {
            if sessions[id] != nil { continue }
            do {
                let session: Session
                switch entry.transport {
                case .stdio:
                    session = try await spawnAndConnect(id: id, entry: entry)
                case .http:
                    session = try await connectHTTP(id: id, entry: entry)
                case .malformed:
                    throw MCPSpawnError.malformedEntry
                }
                sessions[id] = session
                startErrors.removeValue(forKey: id)
            } catch {
                // Prefer LocalizedError.errorDescription so users see the friendly message
                // (e.g. "MCP server exited before connecting…") rather than the raw enum form.
                startErrors[id] = (error as? LocalizedError)?.errorDescription ?? "\(error)"
            }
        }
        // Tear down any servers that were enabled before but are now disabled or removed.
        for id in sessions.keys where (config.mcpServers[id]?.isEnabled ?? false) == false {
            await stop(id: id)
        }
    }

    /// Stop and clean up a single server. Terminates the child process *before* disconnecting the
    /// MCP client — otherwise `disconnect()` can hang awaiting the transport read loop, which won't
    /// EOF until the child process closes its stdout pipe.
    func stop(id: String) async {
        guard let session = sessions.removeValue(forKey: id) else { return }
        // Stdio sessions: terminate the child first so its stdout pipe EOFs and the SDK's read loop
        // can exit, otherwise disconnect() hangs awaiting it. HTTP sessions just disconnect cleanly.
        if let child = session.child, child.isRunning {
            child.terminate()
        }
        await session.client.disconnect()
    }

    /// Stop every running server. Call on app termination or when MCP mode toggles off.
    func stopAll() async {
        let ids = Array(sessions.keys)
        for id in ids { await stop(id: id) }
        startErrors.removeAll()
    }

    // MARK: - Tool surfacing

    /// All tools across all connected servers, namespaced as `<serverID>__<toolName>` for the model.
    /// Returns nil when no MCP tools are available so callers can decide whether to splice or skip.
    func toolDefinitionsJSON() -> String? {
        let entries = sessions.values.flatMap { session -> [String] in
            session.tools.compactMap { tool in
                Self.encodeTool(tool, namespacedAs: Self.namespacedName(server: session.id, tool: tool.name))
            }
        }
        guard !entries.isEmpty else { return nil }
        return "[\(entries.joined(separator: ","))]"
    }

    /// Concise plain-text listing of available MCP tools for inclusion in the system prompt.
    func toolListingForPrompt() -> String {
        guard !sessions.isEmpty else { return "" }
        var lines: [String] = []
        for (id, session) in sessions.sorted(by: { $0.key < $1.key }) {
            lines.append("- \(id) (\(session.tools.count) tool\(session.tools.count == 1 ? "" : "s"))")
            for tool in session.tools.prefix(20) {
                let desc = tool.description?.split(separator: "\n").first.map(String.init) ?? ""
                let trimmed = desc.count > 100 ? String(desc.prefix(100)) + "…" : desc
                let qualified = Self.namespacedName(server: id, tool: tool.name)
                lines.append("  • \(qualified): \(trimmed)")
            }
            if session.tools.count > 20 {
                lines.append("  • …and \(session.tools.count - 20) more")
            }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Tool dispatch

    /// Returns true if the tool name uses our MCP namespace (i.e. contains `__` and a known server prefix).
    func owns(toolName: String) -> Bool {
        guard let (serverID, _) = Self.parseNamespacedName(toolName) else { return false }
        return sessions[serverID] != nil
    }

    /// How long to wait for a single MCP tool call before giving up. MCP servers can hang when their
    /// upstream is unreachable (Docker daemon down, k8s context broken, network call to GitHub stuck).
    /// Without a cap, the agent loop sits frozen on `await callTool` forever.
    /// Override via `MCP_TOOL_TIMEOUT` env var (in seconds) — used by integration tests to verify
    /// the watchdog without waiting 90s per run.
    nonisolated static var toolCallTimeoutSeconds: Double {
        if let raw = ProcessInfo.processInfo.environment["MCP_TOOL_TIMEOUT"],
           let v = Double(raw), v > 0 {
            return v
        }
        return 90
    }

    /// Execute a model-issued tool call against the right MCP server. Returns a string the model can read back.
    /// `arguments` is the parsed string→string map; `rawArguments` is the original JSON (preserves types).
    ///
    /// Hang resistance: `client.callTool` blocks on a `withCheckedThrowingContinuation` waiting for the
    /// JSON-RPC response, and that continuation does not honor task cancellation. Wrapping it in a
    /// TaskGroup-based timeout doesn't work either — the group's implicit destructor awaits all children.
    /// The reliable break-glass is `client.disconnect()`, which (per swift-sdk Client.swift) resumes
    /// every pending request with `MCPError.internalError("Client disconnected")` and releases the await.
    /// So we race a timer task against the call: if the timer wins, it disconnects the client, the call
    /// throws within milliseconds, we tear down the session, and the agent loop moves on.
    func executeToolCall(name: String, arguments: [String: String], rawArguments: String) async -> String {
        guard let (serverID, toolName) = Self.parseNamespacedName(name) else {
            return "Error: malformed MCP tool name '\(name)'. Expected '<server>__<tool>'."
        }
        guard let session = sessions[serverID] else {
            return "Error: MCP server '\(serverID)' is not connected. Open the MCP marketplace and verify it's enabled."
        }
        let mcpArgs = Self.convertArguments(rawArguments: rawArguments, fallback: arguments)
        let timeout = Self.toolCallTimeoutSeconds

        // Watchdog timer. We deliberately use a GCD timer (not a Swift Task) because the swift-sdk's
        // message-handling task can saturate Swift's cooperative thread pool with `AsyncThrowingStream`
        // iteration churn, which starves any Swift Task we'd spawn here — the watchdog Task simply
        // wouldn't fire. GCD timers run on dispatch queues outside the cooperative pool.
        //
        // When the timer fires we terminate the child process — its stdout pipe closes, StdioTransport's
        // readLoop wakes with EOF, the SDK's message-loop task exits, and the SDK's pending
        // continuations get resumed with `MCPError.internalError("Client disconnected")`. That makes
        // the in-flight `callTool` throw immediately, control returns to us, we surface a timeout
        // message, and drop the now-dead session so the next call respawns the server.
        let watchdogFired = WatchdogFlag()
        let watchdogQueue = DispatchQueue(label: "mcp.watchdog")
        let watchdogSource = DispatchSource.makeTimerSource(queue: watchdogQueue)
        watchdogSource.schedule(deadline: .now() + timeout)
        watchdogSource.setEventHandler { [weak session] in
            watchdogFired.set()
            guard let session else { return }
            // Stdio sessions: terminate the child so its stdout pipe EOFs and the SDK's StdioTransport
            // readLoop wakes. HTTP sessions: skip — there's no child to kill, just disconnect.
            if let child = session.child, child.isRunning {
                child.terminate()
            }
            // Then kick off the SDK's `disconnect()` on a detached Task. Disconnect resumes any
            // pending request continuations (line 302-304 of swift-sdk Client.swift), which makes
            // the in-flight `callTool` throw. We do this DETACHED so it doesn't run on whatever
            // actor the GCD handler thinks it's on.
            Task.detached { [client = session.client] in
                await client.disconnect()
            }
        }
        watchdogSource.resume()
        defer { watchdogSource.cancel() }

        do {
            let (content, isError) = try await session.client.callTool(name: toolName, arguments: mcpArgs)
            // Cancel the watchdog ASAP so we don't tear down a healthy session on slow but successful calls.
            watchdogSource.cancel()
            let rendered = Self.renderToolContent(content)
            if isError == true {
                return "MCP tool '\(toolName)' returned error:\n\(rendered)"
            }
            return rendered
        } catch {
            // If the watchdog fired, the error we caught is the "Client disconnected" wakeup, not a real
            // server failure. Surface a clear timeout message and drop the now-dead session so the next
            // call respawns the server.
            if watchdogFired.isSet {
                let stderrTail = session.stderrTail.trimmingCharacters(in: .whitespacesAndNewlines)
                let stderrSnippet = stderrTail.isEmpty ? "" : "\n\nServer stderr:\n\(String(stderrTail.suffix(600)))"
                if let child = session.child, child.isRunning { child.terminate() }
                sessions.removeValue(forKey: serverID)
                return "MCP tool '\(toolName)' timed out after \(Int(timeout))s. The '\(serverID)' server isn't responding — its upstream may be down (e.g. Docker daemon not running, kubeconfig context unreachable, network blocked).\(stderrSnippet)"
            }
            return "Error calling MCP tool '\(toolName)' on server '\(serverID)': \(error)"
        }
    }

    // MARK: - Naming

    /// Tool names sent to the model: `<server>__<tool>`. Double-underscore is rare in real tool names,
    /// keeping collision risk low while staying valid JSON identifier characters.
    nonisolated static func namespacedName(server: String, tool: String) -> String {
        // Server IDs from mcp.json are user-controlled; sanitize to keep a single split point.
        let safeServer = server.replacingOccurrences(of: "__", with: "_")
        return "\(safeServer)__\(tool)"
    }

    nonisolated static func parseNamespacedName(_ name: String) -> (server: String, tool: String)? {
        guard let range = name.range(of: "__") else { return nil }
        let server = String(name[..<range.lowerBound])
        let tool = String(name[range.upperBound...])
        guard !server.isEmpty, !tool.isEmpty else { return nil }
        return (server, tool)
    }

    // MARK: - Internal: spawn

    /// Connect to an HTTP-transport MCP server (no subprocess). Uses the SDK's `HTTPClientTransport`
    /// with SSE streaming enabled. Caps the handshake + tool listing at 30s.
    private func connectHTTP(id: String, entry: MCPServerEntry) async throws -> Session {
        guard let urlString = entry.url, !urlString.isEmpty,
              let url = URL(string: urlString) else {
            throw MCPSpawnError.malformedEntry
        }
        let transport = HTTPClientTransport(endpoint: url, streaming: true)
        let client = Client(name: "mlx-serve-mcp", version: "1.0.0")
        do {
            _ = try await withTimeout(seconds: 30) {
                _ = try await client.connect(transport: transport)
                return ()
            }
        } catch {
            throw MCPSpawnError.httpConnectFailed(url: urlString, detail: "\(error)")
        }
        let listed: [Tool]
        do {
            listed = try await withTimeout(seconds: 30) {
                let (tools, _) = try await client.listTools()
                return tools
            }
        } catch {
            await client.disconnect()
            throw MCPSpawnError.serverFailedToList(detail: "\(error)")
        }
        return Session(id: id, client: client, tools: listed)
    }

    /// Spawn one server, connect over stdio, perform handshake, list tools.
    ///
    /// WHERE it spawns is decided by `MCPSpawnerRouter`: on the host when the
    /// Agent Sandbox is off, inside the Linux guest when it is on (and always,
    /// in the Mac App Store build). Before that seam existed, a sandboxed
    /// session still ran its MCP servers on the host with the user's full
    /// filesystem permissions.
    private func spawnAndConnect(id: String, entry: MCPServerEntry) async throws -> Session {
        guard let command = entry.command, !command.isEmpty else {
            throw MCPSpawnError.malformedEntry
        }
        let args = entry.args ?? []
        let spawner = MCPSpawnerRouter.spawner(sandboxEnabled: AgentSandbox.shared.isEnabled)

        // Pre-flight: confirm the command exists WHERE IT WILL RUN. Cheaper and clearer than letting
        // the shell fail with `command not found` and waiting 30s for the MCP handshake to time out.
        let exists = await spawner.commandExists(command)
        if !exists {
            throw MCPSpawnError.commandNotFound(command: command, hint: Self.installHint(for: command))
        }

        // cwd resolution order: per-entry `cwd` in mcp.json > the active chat session's working dir
        // (set by ChatView before each spawn) > ~/.mlx-serve/workspace. So filesystem/shell MCP
        // servers automatically inherit the user's current chat cwd unless mcp.json pins one.
        let resolvedCwd = Self.resolveWorkingDirectory(entry.cwd ?? defaultCwd)

        // Per-server env overrides ONLY. Each spawner lays them over the right
        // base: the host spawner over the app's macOS environment (as before),
        // the guest spawner over the guest IMAGE's env — forwarding the whole
        // macOS environment there would clobber the guest's PATH/HOME with host
        // paths and break every server in the guest.
        let env = entry.env ?? [:]

        let stdio = try await spawner.open(command: command, args: args, cwd: resolvedCwd, env: env)
        let stderrBox = stdio.stderr
        let child = stdio.child
        let transport = StdioTransport(input: stdio.input, output: stdio.output)

        let client = Client(name: "mlx-serve-mcp", version: "1.0.0")

        // Race the MCP handshake against a child-exit watcher: if the server dies before connecting
        // (e.g. npx exits because the package wasn't found / install failed / runtime missing), we surface
        // the captured stderr immediately instead of waiting out the 30s connect timeout.
        do {
            try await Self.connectOrFailFast(client: client, transport: transport, child: child, stderrBox: stderrBox)
        } catch {
            if child.isRunning { child.terminate() }
            throw error
        }

        let listed: [Tool]
        do {
            listed = try await withTimeout(seconds: 30) {
                let (tools, _) = try await client.listTools()
                return tools
            }
        } catch {
            await client.disconnect()
            if child.isRunning { child.terminate() }
            // Surface stderr context if listTools failed because the child crashed.
            let tail = stderrBox.snapshot().trimmingCharacters(in: .whitespacesAndNewlines)
            if !tail.isEmpty {
                throw MCPSpawnError.serverFailedToList(detail: tail)
            }
            throw error
        }

        return Session(id: id, child: child, stderr: stderrBox, client: client, tools: listed)
    }

    /// Race `client.connect(transport:)` against process death. The first to finish wins; the loser
    /// keeps running but we abandon awaiting it.
    ///
    /// We deliberately avoid `withThrowingTaskGroup` here because its implicit destructor *awaits*
    /// every child task to complete before returning. The SDK's `client.connect()` blocks on a
    /// `withCheckedThrowingContinuation` waiting for an `initialize` JSON-RPC reply — and when the
    /// child process dies (docker-mcp does this in 0.58s when the daemon is unreachable), that reply
    /// never comes and the continuation never resumes. A TaskGroup approach would hang on the
    /// destructor's await even after we tried to cancel.
    ///
    /// Instead we use an unstructured `withCheckedContinuation` and Process's `terminationHandler`
    /// (synchronous, fires from a background queue the moment the OS reaps the child). Whoever
    /// resumes first wins; the other path keeps running but the parent has already moved on.
    private static func connectOrFailFast(client: Client, transport: StdioTransport, child: MCPChild, stderrBox: StderrBox) async throws {
        let resumed = OneShotResume()
        // Held so the death watcher can be cancelled once the handshake settles;
        // otherwise it keeps polling at 10 Hz for the server's whole lifetime.
        let watcher = TaskBox()
        defer { watcher.task?.cancel() }

        let result: Result<Void, Error> = await withCheckedContinuation { continuation in
            // Path A: the SDK's MCP handshake completes successfully → resume with .success.
            Task {
                do {
                    _ = try await client.connect(transport: transport)
                    resumed.tryResume {
                        continuation.resume(returning: .success(()))
                    }
                } catch {
                    resumed.tryResume {
                        continuation.resume(returning: .failure(error))
                    }
                }
            }
            // Path B: the server dies before the handshake completes → resume with the captured
            // stderr, rather than waiting out the 30s cap.
            //
            // Polled rather than `Process.terminationHandler`: a server running in the Agent Sandbox
            // guest is not a host process and has no such callback. 100 ms is invisible next to a spawn.
            watcher.task = Task {
                while !Task.isCancelled {
                    if !child.isRunning {
                        resumed.tryResume {
                            let tail = stderrBox.snapshot().trimmingCharacters(in: .whitespacesAndNewlines)
                            let err = MCPSpawnError.serverExitedEarly(status: child.exitStatus ?? -1, stderr: tail)
                            continuation.resume(returning: .failure(err))
                        }
                        return
                    }
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
            }
            // Belt-and-suspenders: hard cap at 30s in case neither path fires (process is alive but
            // the server never sends an initialize reply — would otherwise hang forever).
            Task {
                try? await Task.sleep(nanoseconds: 30_000_000_000)
                resumed.tryResume {
                    continuation.resume(returning: .failure(MCPManagerError.timeout(seconds: 30)))
                }
            }
        }
        try result.get()
    }

    /// Lets `connectOrFailFast` cancel the death watcher it starts inside the
    /// continuation body (which runs synchronously, so the task is set before
    /// the await resumes).
    private final class TaskBox: @unchecked Sendable {
        var task: Task<Void, Never>?
    }

    /// Run `command -v <name>` via a login shell. Returns true iff the command resolves on PATH.
    /// Login shell (`zsh -l`) so Homebrew, nvm, asdf, and `~/.local/bin` are honored.
    /// Command existence now depends on WHERE the server runs — see
    /// `MCPSpawner.commandExists`. `MCPManager` no longer probes the host directly.


    /// Human-readable install hint for the common runtimes our catalog depends on. Falls back to a
    /// generic message for unknown commands.
    nonisolated static func installHint(for command: String) -> String {
        switch command {
        case "npx", "npm", "node":
            return "Install Node.js from nodejs.org or via Homebrew: brew install node"
        case "uvx", "uv":
            return "Install uv: brew install uv  (or curl -LsSf https://astral.sh/uv/install.sh | sh)"
        case "docker":
            return "Install Docker Desktop from docker.com (also make sure the daemon is running)"
        case "python", "python3":
            return "Install Python from python.org or via Homebrew: brew install python"
        default:
            return "Install \(command) and make sure it's on your PATH (a login shell — Terminal — should be able to run `\(command) --version`)"
        }
    }

    /// Resolve the cwd for a spawned MCP server. Honors the optional `cwd` field on the entry and falls
    /// back to `~/.mlx-serve/workspace`. Tilde-expands and creates the directory if it doesn't yet exist.
    nonisolated static func resolveWorkingDirectory(_ override: String?) -> String {
        let raw: String
        if let override, !override.trimmingCharacters(in: .whitespaces).isEmpty {
            raw = override
        } else {
            raw = "~/.mlx-serve/workspace"
        }
        let path = NSString(string: raw).expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    /// Command-line quoting moved to `MCPCommandLine.shellEscape`, shared by the
    /// host and guest spawners so both build the same invocation.


    // MARK: - Internal: tool encoding/decoding

    /// Translate an MCP Tool into an OpenAI-format function tool JSON object string.
    nonisolated private static func encodeTool(_ tool: Tool, namespacedAs name: String) -> String? {
        // Encode the inputSchema (a Value) and the description into a {type:function, function:{...}} envelope.
        let encoder = JSONEncoder()
        // Stable key order is nice for debugging but not required by the API.
        encoder.outputFormatting = []
        guard let schemaData = try? encoder.encode(tool.inputSchema),
              let schemaStr = String(data: schemaData, encoding: .utf8) else {
            return nil
        }
        // Description: keep it short and JSON-safe.
        let descSource = tool.description ?? ""
        let truncated = descSource.count > 600 ? String(descSource.prefix(600)) + "…" : descSource
        let descData = (try? JSONEncoder().encode(truncated)) ?? Data("\"\"".utf8)
        let descJSON = String(data: descData, encoding: .utf8) ?? "\"\""
        let nameData = (try? JSONEncoder().encode(name)) ?? Data()
        let nameJSON = String(data: nameData, encoding: .utf8) ?? "\"\(name)\""
        return "{\"type\":\"function\",\"function\":{\"name\":\(nameJSON),\"description\":\(descJSON),\"parameters\":\(schemaStr)}}"
    }

    /// Build the [String: Value] argument map for callTool. Prefer the raw JSON string for type fidelity.
    /// Always returns a (possibly empty) dictionary — never nil — because some MCP servers (e.g. Azure
    /// DevOps) validate the `arguments` field with strict Zod schemas and reject `undefined`. Sending
    /// `{}` for a no-arg call satisfies those validators while still being a no-op for tools that ignore it.
    nonisolated static func convertArguments(rawArguments: String, fallback: [String: String]) -> [String: Value] {
        if let data = rawArguments.data(using: .utf8),
           !rawArguments.isEmpty,
           let parsed = try? JSONDecoder().decode([String: Value].self, from: data) {
            return parsed
        }
        // Fall back: best-effort per-key conversion (each value parsed as JSON, else string).
        var out: [String: Value] = [:]
        for (k, v) in fallback {
            if let data = v.data(using: .utf8),
               let parsed = try? JSONDecoder().decode(Value.self, from: data) {
                out[k] = parsed
            } else {
                out[k] = .string(v)
            }
        }
        return out
    }

    /// Flatten Tool.Content array into a string the model can read back. Non-text content is summarized.
    nonisolated static func renderToolContent(_ content: [Tool.Content]) -> String {
        var parts: [String] = []
        for item in content {
            switch item {
            case .text(let text, _, _):
                parts.append(text)
            case .image(_, let mimeType, _, _):
                parts.append("[image: \(mimeType)]")
            case .audio(_, let mimeType, _, _):
                parts.append("[audio: \(mimeType)]")
            case .resource(let resource, _, _):
                parts.append("[resource: \(resource.uri) (\(resource.mimeType ?? "unknown"))]")
            case .resourceLink(let uri, let name, _, _, let mimeType, _):
                parts.append("[resource link: \(name) — \(uri)\(mimeType.map { " (\($0))" } ?? "")]")
            }
        }
        return parts.isEmpty ? "(no content)" : parts.joined(separator: "\n")
    }
}

// MARK: - Timeout helper

/// Race a child task against a timeout; cancels and throws on expiry.
private func withTimeout<T: Sendable>(seconds: Double, operation: @escaping @Sendable () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw MCPManagerError.timeout(seconds: seconds)
        }
        guard let first = try await group.next() else {
            throw MCPManagerError.timeout(seconds: seconds)
        }
        group.cancelAll()
        return first
    }
}

enum MCPManagerError: Error, LocalizedError {
    case timeout(seconds: Double)

    var errorDescription: String? {
        switch self {
        case .timeout(let s): return "MCP operation timed out after \(s)s"
        }
    }
}

/// Errors thrown while spawning a curated MCP server. Distinct from generic timeouts so the marketplace
/// UI can render an actionable hint (install Node, start Docker, etc.) instead of a wall of text.
enum MCPSpawnError: LocalizedError {
    case commandNotFound(command: String, hint: String)
    case serverExitedEarly(status: Int32, stderr: String)
    case serverFailedToList(detail: String)
    case httpConnectFailed(url: String, detail: String)
    case malformedEntry
    /// Pre-baked-only policy (Mac App Store build): the server isn't in the
    /// bundled guest image, and fetching it at runtime would download code.
    case notPrebaked(command: String)

    var errorDescription: String? {
        switch self {
        case .commandNotFound(let command, let hint):
            return "'\(command)' not found on PATH. \(hint)"
        case .notPrebaked(let command):
            return "This build runs MCP servers from the bundled sandbox image; '\(command)' isn't in it. Use a server from the marketplace (Filesystem, GitHub) or an HTTP MCP server (\"url\" entry in mcp.json)."
        case .serverExitedEarly(let status, let stderr):
            let body = stderr.isEmpty ? "(no stderr captured)" : String(stderr.suffix(800))
            return "MCP server exited before connecting (status \(status)).\n\(body)"
        case .serverFailedToList(let detail):
            return "MCP server connected but listTools failed: \(String(detail.suffix(800)))"
        case .httpConnectFailed(let url, let detail):
            return "Couldn't reach MCP server at \(url): \(String(detail.suffix(400)))"
        case .malformedEntry:
            return "Malformed mcp.json entry: needs either a stdio `command` (with optional `args`) or an HTTP `url`."
        }
    }
}

/// One-shot Sendable boolean. Used by `executeToolCall` to know — after the call throws —
/// whether the throw came from our watchdog (timeout) or from a real server-side failure.
final class WatchdogFlag: @unchecked Sendable {
    private var fired = false
    private let lock = NSLock()
    func set() { lock.lock(); fired = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return fired }
}

/// Single-fire gate for racing two paths that both want to resume the same continuation.
/// `tryResume` runs its closure exactly once across all callers; subsequent calls are no-ops.
/// Required because Swift continuations crash if resumed more than once.
final class OneShotResume: @unchecked Sendable {
    private var done = false
    private let lock = NSLock()
    func tryResume(_ work: () -> Void) {
        lock.lock()
        let firstCall = !done
        done = true
        lock.unlock()
        if firstCall { work() }
    }
}

/// Tiny thread-safe accumulator for stderr chunks during spawn. The readabilityHandler runs on a
/// background queue, so we need a lock around the buffer.
final class StderrBox: @unchecked Sendable {
    private var buffer = ""
    private let lock = NSLock()

    func append(_ chunk: String) {
        lock.lock(); defer { lock.unlock() }
        buffer += chunk
        // Cap in case the server spews a huge log on crash.
        if buffer.count > 8000 {
            buffer = String(buffer.suffix(8000))
        }
    }

    func snapshot() -> String {
        lock.lock(); defer { lock.unlock() }
        return buffer
    }
}
