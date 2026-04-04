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

// MARK: - Web Fetch

struct WebFetchHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let urlStr = parameters["url"], let url = URL(string: urlStr) else {
            throw ToolError.missingParameter("url")
        }

        let (data, _) = try await URLSession.shared.data(from: url)
        var text = String(data: data, encoding: .utf8) ?? ""

        // Strip HTML tags and collapse whitespace
        text = text.replacingOccurrences(of: "<script[^>]*>[\\s\\S]*?</script>", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "<style[^>]*>[\\s\\S]*?</style>", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        text = text.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        return String(text.prefix(8192))
    }
}

// MARK: - Safari Browse (AppleScript)

struct SafariBrowseHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"

        let script: String
        switch action {
        case "navigate":
            guard let url = parameters["url"] else { throw ToolError.missingParameter("url") }
            let escaped = url.replacingOccurrences(of: "\"", with: "\\\"")
            script = """
            tell application "Safari"
                activate
                if (count of windows) = 0 then make new document
                set URL of document 1 to "\(escaped)"
                delay 2
                return URL of document 1
            end tell
            """

        case "readText":
            script = """
            tell application "Safari"
                do JavaScript "document.body.innerText.substring(0, 8000)" in document 1
            end tell
            """

        case "click":
            guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
            let escaped = selector.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\"", with: "\\\"")
            script = """
            tell application "Safari"
                do JavaScript "document.querySelector('\(escaped)').click(); 'clicked'" in document 1
            end tell
            """

        case "getInfo":
            script = """
            tell application "Safari"
                set pageTitle to name of document 1
                set pageURL to URL of document 1
                return "Title: " & pageTitle & linefeed & "URL: " & pageURL
            end tell
            """

        case "executeJS":
            guard let js = parameters["script"] else { throw ToolError.missingParameter("script") }
            let escaped = js.replacingOccurrences(of: "\"", with: "\\\"")
            script = """
            tell application "Safari"
                do JavaScript "\(escaped)" in document 1
            end tell
            """

        default:
            throw ToolError.executionFailed("Unknown browse action: \(action)")
        }

        return try await runAppleScript(script)
    }
}

// MARK: - Chrome CDP (DevTools Protocol)

struct ChromeCDPHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        let action = parameters["action"] ?? "navigate"

        // Discover debugging target
        let discoveryURL = URL(string: "http://localhost:9222/json")!
        let (data, _) = try await URLSession.shared.data(from: discoveryURL)
        guard let tabs = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
              let wsURLStr = tabs.first?["webSocketDebuggerUrl"] as? String,
              let wsURL = URL(string: wsURLStr) else {
            throw ToolError.executionFailed("No Chrome debugging tabs found. Launch Chrome with --remote-debugging-port=9222")
        }

        // Connect WebSocket
        let ws = URLSession.shared.webSocketTask(with: wsURL)
        ws.resume()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        switch action {
        case "navigate":
            guard let url = parameters["url"] else { throw ToolError.missingParameter("url") }
            _ = try await sendCDP(ws: ws, method: "Page.navigate", params: ["url": url])
            // Wait for page load
            try await Task.sleep(nanoseconds: 2_000_000_000)
            return "Navigated to \(url)"

        case "readText":
            let result = try await sendCDP(ws: ws, method: "Runtime.evaluate",
                                           params: ["expression": "document.body.innerText.substring(0, 8000)"])
            return extractCDPValue(result)

        case "evaluate":
            guard let expression = parameters["expression"] else { throw ToolError.missingParameter("expression") }
            let result = try await sendCDP(ws: ws, method: "Runtime.evaluate", params: ["expression": expression])
            return extractCDPValue(result)

        case "click":
            guard let selector = parameters["selector"] else { throw ToolError.missingParameter("selector") }
            let js = "document.querySelector('\(selector.replacingOccurrences(of: "'", with: "\\'"))').click(); 'clicked'"
            let result = try await sendCDP(ws: ws, method: "Runtime.evaluate", params: ["expression": js])
            return "Clicked \(selector). \(extractCDPValue(result))"

        default:
            throw ToolError.executionFailed("Unknown Chrome action: \(action)")
        }
    }

    private func sendCDP(ws: URLSessionWebSocketTask, method: String, params: [String: Any]) async throws -> [String: Any] {
        let id = Int.random(in: 1...999999)
        let command: [String: Any] = ["id": id, "method": method, "params": params]
        let json = try JSONSerialization.data(withJSONObject: command)
        try await ws.send(.string(String(data: json, encoding: .utf8)!))

        // Read responses until we find matching ID (skip CDP events)
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            let message = try await ws.receive()
            let responseData: Data
            switch message {
            case .string(let text): responseData = text.data(using: .utf8) ?? Data()
            case .data(let d): responseData = d
            @unknown default: continue
            }
            guard let result = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
                  let responseId = result["id"] as? Int, responseId == id else {
                continue
            }
            return result
        }
        throw ToolError.executionFailed("CDP command timed out")
    }

    private func extractCDPValue(_ response: [String: Any]) -> String {
        if let result = response["result"] as? [String: Any],
           let inner = result["result"] as? [String: Any],
           let value = inner["value"] {
            return String(describing: value).prefix(8192).description
        }
        return String(describing: response).prefix(8192).description
    }
}

// MARK: - Generic AppleScript

struct AppleScriptHandler: ToolHandler {
    func execute(parameters: [String: String], workingDirectory: String?) async throws -> String {
        guard let source = parameters["source"] else {
            throw ToolError.missingParameter("source")
        }
        return try await runAppleScript(source)
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

private func runAppleScript(_ source: String) async throws -> String {
    try await withCheckedThrowingContinuation { continuation in
        DispatchQueue.global(qos: .userInitiated).async {
            var error: NSDictionary?
            guard let script = NSAppleScript(source: source) else {
                continuation.resume(throwing: ToolError.executionFailed("Failed to create AppleScript"))
                return
            }
            let result = script.executeAndReturnError(&error)
            if let error {
                let msg = error[NSAppleScript.errorMessage] as? String ?? "\(error)"
                continuation.resume(throwing: ToolError.executionFailed("AppleScript: \(msg)"))
            } else {
                continuation.resume(returning: result.stringValue ?? "(no output)")
            }
        }
    }
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
        .webFetch: WebFetchHandler(),
        .browse: SafariBrowseHandler(),
        .browseChrome: ChromeCDPHandler(),
        .applescript: AppleScriptHandler(),
    ]

    func executePlan(_ plan: AgentPlan, workingDirectory: String?) async -> [StepResult] {
        isExecuting = true
        results = []

        for (index, step) in plan.steps.enumerated() {
            currentStepIndex = index
            let start = DispatchTime.now()

            do {
                guard let handler = handlers[step.tool] else {
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
