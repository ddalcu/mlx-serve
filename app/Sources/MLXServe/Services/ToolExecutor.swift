import Foundation

// MARK: - Error Type

enum ToolError: LocalizedError {
    case missingParameter(String)
    case executionFailed(String)
    case unsupportedTool(String)

    var errorDescription: String? {
        switch self {
        case .missingParameter(let p): "Missing parameter: \(p)"
        case .executionFailed(let msg): msg
        case .unsupportedTool(let t): "Unsupported tool: \(t)"
        }
    }
}

// MARK: - Handler Protocol

protocol ToolHandler: Sendable {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String
}

// MARK: - Shell

struct ShellHandler: ToolHandler {
    /// Max wall-clock before a FOREGROUND command is dealt with. Long enough for
    /// real installs/builds (`npm install`), bounded so a hang can't stall the
    /// agent. Injectable so tests can use a short bound.
    var timeoutSeconds: Double = 120

    /// When present, `run_in_background:"true"` registers a managed process and
    /// returns instantly, and a foreground command still alive at the timeout is
    /// ADOPTED (never killed) instead of being terminated. Absent (e.g. older
    /// call sites / unit tests that don't inject one) → today's behavior:
    /// foreground only, killed at the timeout.
    var registry: ProcessRegistry? = nil
    /// The chat session the spawned process belongs to (for scoped cleanup).
    var sessionId: UUID? = nil
    /// Out-parameter for the handle of any process started/adopted by this call,
    /// so the caller can surface it on `ToolResult.backgroundHandle`.
    var handleBox: ProcessHandleBox? = nil

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let command = parameters["command"] else {
            throw ToolError.missingParameter("command")
        }
        let cwd = workingDirectory ?? FileManager.default.currentDirectoryPath

        // ── Opt-in background start ──────────────────────────────────────────
        if parameters["run_in_background"] == "true" {
            guard let registry else {
                return "[cwd: \(cwd)]\nError: background execution isn't available in this context. Run the command in the foreground, or it will be stopped at the \(Int(timeoutSeconds))s timeout."
            }
            let (handle, pid): (String, Int32) = await MainActor.run {
                let managed = registry.start(command: command, workingDirectory: workingDirectory, sessionId: sessionId)
                return (managed.handle, managed.pid)
            }
            handleBox?.set(handle)
            return "[cwd: \(cwd)]\nStarted in background as \(handle) (pid \(pid)). It keeps running — poll it with readProcessOutput {\"handle\": \"\(handle)\"}, stop it with killProcess {\"handle\": \"\(handle)\"}."
        }

        let timeout = timeoutSeconds
        let registry = self.registry
        let sessionId = self.sessionId
        let handleBox = self.handleBox

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = ProcessRegistry.makeProcess(command: command, workingDirectory: workingDirectory)
                let outPipe = process.standardOutput as! Pipe
                let errPipe = process.standardError as! Pipe

                // Read both pipes incrementally so a kill never blocks on
                // `readDataToEndOfFile` waiting for orphaned grandchildren
                // (e.g. node spawned by npx) to release the write end.
                let lock = NSLock()
                var outData = Data()
                var errData = Data()
                outPipe.fileHandleForReading.readabilityHandler = { h in
                    let d = h.availableData
                    if !d.isEmpty { lock.lock(); outData.append(d); lock.unlock() }
                }
                errPipe.fileHandleForReading.readabilityHandler = { h in
                    let d = h.availableData
                    if !d.isEmpty { lock.lock(); errData.append(d); lock.unlock() }
                }

                // Signal exit via terminationHandler + a bounded semaphore wait
                // (NOT waitUntilExit) so a never-exiting server releases this
                // worker thread at the timeout instead of leaking it forever.
                let exitSem = DispatchSemaphore(value: 0)
                process.terminationHandler = { _ in exitSem.signal() }

                do {
                    try process.run()
                } catch {
                    continuation.resume(throwing: error)
                    return
                }
                let pid = process.processIdentifier

                // Single-resume guard: the adopt branch resumes from a hopped
                // MainActor task, so a near-simultaneous real exit must not
                // double-resume the continuation.
                let resumed = ManagedAtomicFlag()
                func resume(_ s: String) {
                    if resumed.testAndSet() { continuation.resume(returning: s) }
                }

                let timedOut = exitSem.wait(timeout: .now() + timeout) == .timedOut

                // Raced exit: the wait timed out but the process actually
                // finished right at the deadline → treat as a clean completion.
                if timedOut && process.isRunning {
                    // Hand the live process off to the registry (backstop), or —
                    // with no registry — preserve today's kill-and-report.
                    outPipe.fileHandleForReading.readabilityHandler = nil
                    errPipe.fileHandleForReading.readabilityHandler = nil
                    lock.lock(); let o = outData; let e = errData; lock.unlock()
                    if let registry {
                        var prior = String(data: o, encoding: .utf8) ?? ""
                        let errStr = String(data: e, encoding: .utf8) ?? ""
                        if !errStr.isEmpty { prior += errStr }
                        Task { @MainActor in
                            let managed = registry.adopt(process: process, command: command,
                                                         workingDirectory: workingDirectory,
                                                         sessionId: sessionId, priorOutput: prior)
                            handleBox?.set(managed.handle)
                            resume("[cwd: \(cwd)]\nStill running after \(Int(timeout))s — now managed in the background as \(managed.handle) (pid \(managed.pid)), NOT killed. Poll it with readProcessOutput {\"handle\": \"\(managed.handle)\"}, stop it with killProcess {\"handle\": \"\(managed.handle)\"}.")
                        }
                    } else {
                        process.terminate()
                        kill(pid, SIGKILL)
                        var result = String(data: o, encoding: .utf8) ?? ""
                        let errStr = String(data: e, encoding: .utf8) ?? ""
                        if !errStr.isEmpty { result += "\n[stderr]: \(errStr)" }
                        result += "\n[timed out after \(Int(timeout))s and was killed. If this command waits for input, re-run it with non-interactive flags — it cannot read stdin. If it is a long-running server, start it with run_in_background:\"true\".]"
                        resume("[cwd: \(cwd)]\n\(result)")
                    }
                    return
                }

                // Clean exit (or raced exit at the deadline).
                process.terminationHandler = nil
                outPipe.fileHandleForReading.readabilityHandler = nil
                errPipe.fileHandleForReading.readabilityHandler = nil
                // On a clean exit the pipe EOFs promptly, so drain any final
                // chunk the handler hadn't delivered yet.
                if let d = try? outPipe.fileHandleForReading.readToEnd() { lock.lock(); outData.append(d); lock.unlock() }
                if let d = try? errPipe.fileHandleForReading.readToEnd() { lock.lock(); errData.append(d); lock.unlock() }

                lock.lock(); let o = outData; let e = errData; lock.unlock()
                var result = String(data: o, encoding: .utf8) ?? ""
                let errStr = String(data: e, encoding: .utf8) ?? ""

                if !errStr.isEmpty { result += "\n[stderr]: \(errStr)" }
                if process.terminationStatus != 0 {
                    result += "\n[exit code: \(process.terminationStatus)]"
                } else if result.isEmpty {
                    result = "OK"
                }
                resume("[cwd: \(cwd)]\n\(result)")
            }
        }
    }
}

/// Minimal thread-safe test-and-set flag — guards the shell continuation against
/// a double resume when the adopt branch and a near-simultaneous real exit race.
final class ManagedAtomicFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var set = false
    /// Returns true exactly once (the first caller); false thereafter.
    func testAndSet() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if set { return false }
        set = true
        return true
    }
}

// MARK: - Read File

struct ReadFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"] else {
            throw ToolError.missingParameter("path")
        }

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        guard let data = FileManager.default.contents(atPath: fullPath),
              let content = String(data: data, encoding: .utf8) else {
            throw ToolError.executionFailed("Cannot read file: \(fullPath)")
        }

        let lines = content.components(separatedBy: "\n")
        let totalLines = lines.count
        let startLine = Int(parameters["startLine"] ?? "1") ?? 1
        let endLine = Int(parameters["endLine"] ?? "\(totalLines)") ?? totalLines
        let actualStart = max(1, startLine)
        let actualEnd = min(totalLines, endLine)

        guard actualStart <= actualEnd else {
            return "Invalid line range: \(startLine)-\(endLine) (file has \(totalLines) lines)"
        }

        let slice = lines[actualStart - 1..<actualEnd]
        // Add line numbers so the model can reference specific lines for editFile
        var numbered = slice.enumerated().map { (i, line) in
            "\(actualStart + i)| \(line)"
        }.joined(separator: "\n")

        // Add metadata header for large files so model knows to use line ranges
        if totalLines > 200 || content.utf8.count > 6000 {
            let header = "[File: \(path) | Lines: \(actualStart)-\(actualEnd) of \(totalLines) | \(content.utf8.count) bytes"
            if actualEnd < totalLines {
                numbered = header + " | Use startLine/endLine to read more]\n" + numbered
            } else {
                numbered = header + "]\n" + numbered
            }
        }

        return numbered
    }
}

// MARK: - Write File

struct WriteFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"], let content = parameters["content"] else {
            throw ToolError.missingParameter("path and content")
        }

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        let dir = (fullPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
        return "Wrote \(content.count) characters to \(path)"
    }
}

// MARK: - Edit File

struct EditFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"] else {
            throw ToolError.missingParameter("path")
        }

        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        guard let data = FileManager.default.contents(atPath: fullPath),
              var content = String(data: data, encoding: .utf8) else {
            throw ToolError.executionFailed("Cannot read file: \(fullPath)")
        }

        // Line-number-based editing: startLine/endLine + replace
        if let startStr = parameters["startLine"], let startLine = Int(startStr) {
            guard let replace = parameters["replace"] else {
                throw ToolError.executionFailed("editFile with startLine/endLine requires 'replace' parameter. You sent startLine=\(startStr) but no replace content. Example: {\"path\": \"file.js\", \"startLine\": \"5\", \"endLine\": \"8\", \"replace\": \"new code\"}")
            }
            let lines = content.components(separatedBy: "\n")
            let endLine = Int(parameters["endLine"] ?? startStr) ?? startLine
            let actualStart = max(1, startLine)
            let actualEnd = min(lines.count, endLine)

            guard actualStart <= actualEnd else {
                throw ToolError.executionFailed("Invalid line range: \(startLine)-\(endLine) (file has \(lines.count) lines)")
            }

            var newLines = Array(lines[0..<(actualStart - 1)])
            newLines.append(contentsOf: replace.components(separatedBy: "\n"))
            if actualEnd < lines.count {
                newLines.append(contentsOf: lines[actualEnd...])
            }
            content = newLines.joined(separator: "\n")
            try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
            return "Edited \(path) (replaced lines \(actualStart)-\(actualEnd))"
        }

        // Text-based editing: find + replace
        guard let find = parameters["find"], !find.isEmpty else {
            throw ToolError.missingParameter("Either 'find' or 'startLine' is required")
        }
        let replace = parameters["replace"] ?? ""

        guard content.contains(find) else {
            // Show nearby content to help the model correct its find pattern
            let lines = content.components(separatedBy: "\n")
            let preview = lines.prefix(10).enumerated()
                .map { "\($0.offset + 1)| \($0.element)" }.joined(separator: "\n")
            throw ToolError.executionFailed(
                "Pattern not found in \(path). Use readFile first to see exact content, or use startLine/endLine for line-based editing. First 10 lines:\n\(preview)"
            )
        }

        content = content.replacingOccurrences(of: find, with: replace)
        try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
        return "Edited \(path)"
    }
}

// MARK: - Search Files

struct SearchFilesHandler: ToolHandler {
    private let shellHandler = ShellHandler()

    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let pattern = parameters["pattern"] else {
            throw ToolError.missingParameter("pattern")
        }
        let path = parameters["path"] ?? "."
        // Confine search path to workspace
        let confinedPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        let maxResults = Int(parameters["maxResults"] ?? "100") ?? 100
        let escaped = shellEscape(pattern)
        let escapedPath = shellEscape(confinedPath)

        // Use ripgrep if available, fallback to grep
        let useRg = FileManager.default.fileExists(atPath: "/opt/homebrew/bin/rg")
            || FileManager.default.fileExists(atPath: "/usr/local/bin/rg")

        var command: String
        if useRg {
            command = "rg -n --no-heading"
            if let include = parameters["include"] {
                command += " -g \(shellEscape(include))"
            }
            if let context = parameters["context"], let ctx = Int(context), ctx > 0 {
                command += " -C \(min(ctx, 10))"
            }
            command += " \(escaped) \(escapedPath) 2>/dev/null | head -\(maxResults)"
        } else {
            command = "grep -rn"
            if let include = parameters["include"] {
                command += " --include=\(shellEscape(include))"
            }
            command += " \(escaped) \(escapedPath) 2>/dev/null | head -\(maxResults)"
        }
        return try await shellHandler.execute(parameters: ["command": command], workingDirectory: workingDirectory)
    }

    private func shellEscape(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}


// MARK: - List Files

struct ListFilesHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let path = parameters["path"] ?? "."
        let fullPath = try resolveAndConfine(path, workingDirectory: workingDirectory)
        let pattern = parameters["pattern"]
        let recursive = parameters["recursive"]?.lowercased() == "true"

        let fm = FileManager.default
        guard fm.fileExists(atPath: fullPath) else {
            throw ToolError.executionFailed("Directory not found: \(fullPath)")
        }

        var entries: [String] = []

        if recursive {
            guard let enumerator = fm.enumerator(atPath: fullPath) else {
                throw ToolError.executionFailed("Cannot enumerate: \(fullPath)")
            }
            // If pattern has no path separators (e.g. "*.swift"), match filename only
            let matchFilenameOnly = pattern != nil && !pattern!.contains("/")
            while let item = enumerator.nextObject() as? String {
                if let pattern {
                    let target = matchFilenameOnly ? (item as NSString).lastPathComponent : item
                    if matchesGlob(target, pattern: pattern) {
                        entries.append(item)
                    }
                } else {
                    entries.append(item)
                }
                if entries.count >= 200 { break }
            }
        } else {
            let items = try fm.contentsOfDirectory(atPath: fullPath)
            for item in items.sorted() {
                if let pattern {
                    if matchesGlob(item, pattern: pattern) {
                        entries.append(item)
                    }
                } else {
                    entries.append(item)
                }
                if entries.count >= 200 { break }
            }
        }

        if entries.isEmpty {
            return "No files found in \(path)" + (pattern != nil ? " matching '\(pattern!)'" : "")
        }
        let result = entries.joined(separator: "\n")
        let suffix = entries.count >= 200 ? "\n[... truncated at 200 entries]" : ""
        return result + suffix
    }

    /// Simple glob matching supporting * and ** wildcards.
    private func matchesGlob(_ path: String, pattern: String) -> Bool {
        // Convert glob to regex: * matches non-slash, ** matches anything
        var regex = "^"
        var i = pattern.startIndex
        while i < pattern.endIndex {
            let c = pattern[i]
            if c == "*" {
                let next = pattern.index(after: i)
                if next < pattern.endIndex && pattern[next] == "*" {
                    regex += ".*"
                    i = pattern.index(after: next)
                    // Skip trailing slash after **
                    if i < pattern.endIndex && pattern[i] == "/" {
                        i = pattern.index(after: i)
                    }
                    continue
                } else {
                    regex += "[^/]*"
                }
            } else if c == "?" {
                regex += "[^/]"
            } else if c == "." {
                regex += "\\."
            } else {
                regex += String(c)
            }
            i = pattern.index(after: i)
        }
        regex += "$"
        return path.range(of: regex, options: .regularExpression) != nil
    }
}

// MARK: - Browser tool timeout

/// Hard ceiling on any single browser-tool invocation. The inner WKWebView calls
/// already have their own navigate/evaluateJS timeouts, but a page can still
/// hang in ways that freeze the agent loop — this guarantees the tool returns
/// (with an error) within a bounded window so the loop keeps moving.
private let browserToolTimeoutSeconds: UInt64 = 30

private func withBrowserToolTimeout<T: Sendable>(
    _ description: String,
    _ body: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await body() }
        group.addTask {
            try await Task.sleep(nanoseconds: browserToolTimeoutSeconds * 1_000_000_000)
            throw ToolError.executionFailed("\(description) timed out after \(browserToolTimeoutSeconds)s")
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}

// MARK: - Web Search (DuckDuckGo)

struct WebSearchHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let query = parameters["query"], !query.isEmpty else {
            throw ToolError.missingParameter("query")
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let url = "https://html.duckduckgo.com/html/?q=\(encoded)"

        return try await withBrowserToolTimeout("webSearch") {
            let browser = await BrowserManager.shared
            _ = try await browser.navigate(to: url)
            // Wait for search results to render
            try await Task.sleep(nanoseconds: 500_000_000)

            // Extract structured search results instead of raw page text
            let js = """
            (function() {
                var results = [];
                var links = document.querySelectorAll('.result__a, .result__title a, a.result-link');
                if (links.length === 0) links = document.querySelectorAll('a[href*="//"]');
                var seen = new Set();
                for (var i = 0; i < Math.min(links.length, 8); i++) {
                    var a = links[i];
                    var href = a.href || '';
                    if (href.includes('duckduckgo.com') || seen.has(href)) continue;
                    seen.add(href);
                    var title = (a.textContent || '').trim();
                    var snippet = '';
                    var parent = a.closest('.result') || a.closest('.web-result') || a.parentElement;
                    if (parent) {
                        var snipEl = parent.querySelector('.result__snippet, .result-snippet');
                        if (snipEl) snippet = snipEl.textContent.trim();
                    }
                    if (title && href) results.push(title + '\\n' + href + (snippet ? '\\n' + snippet : ''));
                }
                return results.length > 0 ? results.join('\\n\\n') : document.body.innerText.substring(0, 2000);
            })()
            """
            let result = try await browser.evaluateJS(js)
            return "Search results for '\(query)':\n\n\(result)"
        }
    }
}

// MARK: - Built-in Browser (WKWebView)

struct BrowseHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"

        return try await withBrowserToolTimeout("browse(\(action))") {
            let browser = await BrowserManager.shared

            switch action {
            case "navigate":
                guard let url = parameters["url"] else { throw ToolError.missingParameter("url") }
                return try await browser.navigate(to: url)
            case "readText":
                // Navigate to URL first if provided, then read text
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                return try await browser.readText()
            case "readHTML":
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                return try await browser.readHTML()
            case "extractText":
                if let url = parameters["url"] { _ = try await browser.navigate(to: url) }
                guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
                return try await browser.extractText(selector: selector)
            case "click":
                guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
                return try await browser.click(selector: selector)
            case "screenshot":
                if let url = parameters["url"] {
                    _ = try await browser.navigate(to: url)
                }
                guard let data = await browser.takeScreenshot() else {
                    return "Failed to capture screenshot"
                }
                let base64 = data.base64EncodedString()
                return "[screenshot:\(data.count) bytes]\ndata:image/jpeg;base64,\(base64)"
            case "getInfo":
                return try await browser.getInfo()
            case "executeJS":
                guard let script = parameters["script"] ?? parameters["expression"] else {
                    throw ToolError.missingParameter("script")
                }
                return try await browser.evaluateJS(script)
            default:
                // Fallback: if URL is present, treat as navigate
                if let url = parameters["url"] {
                    return try await browser.navigate(to: url)
                }
                return try await browser.readText()
            }
        }
    }
}


// MARK: - Save Memory

struct SaveMemoryHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let memory = parameters["memory"] else {
            throw ToolError.missingParameter("memory")
        }
        AgentPrompt.saveMemory(memory)
        return "OK"
    }
}

// MARK: - Shared Helpers

private func resolvePath(_ path: String, workingDirectory: String?) -> String {
    if path.hasPrefix("/") || path.hasPrefix("~") {
        return NSString(string: path).expandingTildeInPath
    }
    if let wd = workingDirectory {
        return (wd as NSString).appendingPathComponent(path)
    }
    return path
}

/// Resolve a path and verify it stays within the working directory.
/// Returns the resolved absolute path, or throws if it escapes the workspace.
private func resolveAndConfine(_ path: String, workingDirectory: String?) throws -> String {
    let resolved: String
    if path.hasPrefix("/") || path.hasPrefix("~") {
        resolved = NSString(string: path).expandingTildeInPath
    } else if let wd = workingDirectory {
        resolved = (wd as NSString).appendingPathComponent(path)
    } else {
        return path // no workspace set, can't enforce
    }

    // Normalize to resolve ".." and symlinks for accurate prefix check
    let normalizedResolved = (resolved as NSString).standardizingPath

    guard let wd = workingDirectory else { return normalizedResolved }
    let normalizedWd = (wd as NSString).standardizingPath

    guard normalizedResolved == normalizedWd || normalizedResolved.hasPrefix(normalizedWd + "/") else {
        throw ToolError.executionFailed("Access denied: path '\(path)' resolves to '\(normalizedResolved)' which is outside the workspace '\(normalizedWd)'")
    }

    return normalizedResolved
}

// MARK: - Executor

@MainActor
class ToolExecutor: ObservableObject {
    @Published var currentStepIndex: Int?
    @Published var results: [StepResult] = []
    @Published var isExecuting = false

    private let handlers: [AgentToolKind: any ToolHandler] = [
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

    func executePlan(_ plan: AgentPlan, workingDirectory: String?) async -> [StepResult] {
        isExecuting = true
        results = []

        for (index, step) in plan.steps.enumerated() {
            currentStepIndex = index
            let start = DispatchTime.now()

            do {
                // Smart fallback: if editFile is called with content but no find/replace, use writeFile
                let effectiveTool: AgentToolKind
                if step.tool == .editFile && step.parameters["content"] != nil && step.parameters["find"] == nil {
                    effectiveTool = .writeFile
                } else {
                    effectiveTool = step.tool
                }
                guard let handler = handlers[effectiveTool] else {
                    throw ToolError.unsupportedTool(step.tool.rawValue)
                }
                let output = try await handler.execute(parameters: step.parameters, workingDirectory: workingDirectory)
                let elapsed = Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
                results.append(StepResult(stepId: step.id, status: .success, output: output, durationMs: elapsed))
            } catch {
                let elapsed = Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
                results.append(StepResult(stepId: step.id, status: .failed, output: "", error: error.localizedDescription, durationMs: elapsed))
            }
        }

        currentStepIndex = nil
        isExecuting = false
        return results
    }
}
