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
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let command = parameters["command"] else {
            throw ToolError.missingParameter("command")
        }

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let process = Process()
                    process.executableURL = URL(fileURLWithPath: "/bin/zsh")
                    process.arguments = ["-c", command]
                    if let wd = workingDirectory {
                        process.currentDirectoryURL = URL(fileURLWithPath: wd)
                    }

                    let stdout = Pipe()
                    let stderr = Pipe()
                    process.standardOutput = stdout
                    process.standardError = stderr

                    try process.run()

                    // 30s timeout
                    DispatchQueue.global().asyncAfter(deadline: .now() + 30) {
                        if process.isRunning { process.terminate() }
                    }

                    process.waitUntilExit()

                    let outData = stdout.fileHandleForReading.readDataToEndOfFile()
                    let errData = stderr.fileHandleForReading.readDataToEndOfFile()
                    var result = String(data: outData, encoding: .utf8) ?? ""
                    let errStr = String(data: errData, encoding: .utf8) ?? ""

                    if !errStr.isEmpty { result += "\n[stderr]: \(errStr)" }
                    if process.terminationStatus != 0 {
                        result += "\n[exit code: \(process.terminationStatus)]"
                    } else if result.isEmpty {
                        result = "OK"
                    }

                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - Read File

struct ReadFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"] else {
            throw ToolError.missingParameter("path")
        }

        let fullPath = resolvePath(path, workingDirectory: workingDirectory)
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
        guard let path = parameters["path"], var content = parameters["content"] else {
            throw ToolError.missingParameter("path and content")
        }

        // Unescape common sequences that smaller models double-escape
        content = content
            .replacingOccurrences(of: "\\n", with: "\n")
            .replacingOccurrences(of: "\\t", with: "\t")
            .replacingOccurrences(of: "\\\"", with: "\"")

        let fullPath = resolvePath(path, workingDirectory: workingDirectory)
        let dir = (fullPath as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try content.write(toFile: fullPath, atomically: true, encoding: .utf8)
        return "Wrote \(content.count) characters to \(path)"
    }
}

// MARK: - Edit File

struct EditFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"],
              let find = parameters["find"],
              let replace = parameters["replace"] else {
            throw ToolError.missingParameter("path, find, and replace")
        }

        let fullPath = resolvePath(path, workingDirectory: workingDirectory)
        guard let data = FileManager.default.contents(atPath: fullPath),
              var content = String(data: data, encoding: .utf8) else {
            throw ToolError.executionFailed("Cannot read file: \(fullPath)")
        }

        guard content.contains(find) else {
            throw ToolError.executionFailed("Pattern not found in \(path)")
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
        let maxResults = Int(parameters["maxResults"] ?? "100") ?? 100
        let escaped = shellEscape(pattern)
        let escapedPath = shellEscape(path)

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
        let fullPath = resolvePath(path, workingDirectory: workingDirectory)
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

// MARK: - Web Search (DuckDuckGo)

struct WebSearchHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let query = parameters["query"], !query.isEmpty else {
            throw ToolError.missingParameter("query")
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let url = "https://html.duckduckgo.com/html/?q=\(encoded)"

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
        let result = try await browser.webView.evaluateJavaScript(js)
        let text = (result as? String) ?? "No search results found"
        return "Search results for '\(query)':\n\n\(text)"
    }
}

// MARK: - Built-in Browser (WKWebView)

struct BrowseHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"
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
        case "click":
            guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
            return try await browser.click(selector: selector)
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
