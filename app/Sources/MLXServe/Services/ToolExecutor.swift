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
                    }

                    continuation.resume(returning: String(result.prefix(8192)))
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
        let startLine = Int(parameters["startLine"] ?? "1") ?? 1
        let endLine = Int(parameters["endLine"] ?? "\(lines.count)") ?? lines.count

        let slice = lines[max(0, startLine - 1)..<min(lines.count, endLine)]
        return String(slice.joined(separator: "\n").prefix(8192))
    }
}

// MARK: - Write File

struct WriteFileHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let path = parameters["path"], let content = parameters["content"] else {
            throw ToolError.missingParameter("path and content")
        }

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
        let escaped = shellEscape(pattern)
        let escapedPath = shellEscape(path)
        let command = "grep -rn \(escaped) \(escapedPath) 2>/dev/null | head -100"
        return try await shellHandler.execute(parameters: ["command": command], workingDirectory: workingDirectory)
    }

    private func shellEscape(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}


// MARK: - Web Search (DuckDuckGo)

struct WebSearchHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let query = parameters["query"] else {
            throw ToolError.missingParameter("query")
        }
        let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? query
        let url = "https://html.duckduckgo.com/html/?q=\(encoded)"

        let browser = await BrowserManager.shared
        if await browser.webView == nil {
            throw ToolError.executionFailed("Browser window is not open. Open it from the menu bar dropdown.")
        }
        return try await browser.navigate(to: url)
    }
}

// MARK: - Built-in Browser (WKWebView)

struct BrowseHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"
        let browser = await BrowserManager.shared

        // Check browser window is open
        if await browser.webView == nil {
            throw ToolError.executionFailed("Browser window is not open. Open it from the menu bar dropdown.")
        }

        switch action {
        case "navigate":
            guard let url = parameters["url"] else { throw ToolError.missingParameter("url") }
            return try await browser.navigate(to: url)
        case "readText":
            return try await browser.readText()
        case "readHTML":
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
        .browse: BrowseHandler(),
        .webSearch: WebSearchHandler(),
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
