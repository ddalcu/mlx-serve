import Foundation
import AppKit

enum AgentPrompt {

    static let skillManager = SkillManager()

    private static let mlxServeDir = NSString(string: "~/.mlx-serve").expandingTildeInPath
    private static let promptPath = (mlxServeDir as NSString).appendingPathComponent("system-prompt.md")
    private static let memoryPath = (mlxServeDir as NSString).appendingPathComponent("memory.md")

    private static let defaultPromptFile = """
        # System

        You are an autonomous macOS agent running on Apple Silicon. Act independently to complete tasks — do not ask the user for confirmation between steps. Execute multi-step tasks without pausing. Only respond to the user when the task is fully complete or if you hit a genuine ambiguity that cannot be resolved with tools.

        # Using Your Tools

        IMPORTANT: Use dedicated tools instead of shell equivalents:
        - readFile instead of `cat`, `head`, `tail`
        - writeFile instead of `echo >` or `cat <<EOF`
        - editFile instead of `sed` or `awk`
        - searchFiles instead of `grep` or `rg`
        - listFiles instead of `find` or `ls -R`

        Use shell only for: build/test commands, git operations, process management, installing packages, and commands with no dedicated tool equivalent.

        Tool arguments must be valid JSON: {"key": "value"}. Never omit required parameters.

        # File Editing Rules

        - ALWAYS readFile before editFile — you must see the exact text to match
        - readFile shows line numbers as "N| text". The `find` param must match the actual text AFTER the "N| " prefix, not including the line number
        - The `find` parameter must match the file content exactly, including whitespace and indentation
        - If editFile fails with "Pattern not found", readFile again to check the actual content
        - For large files, readFile shows a header with total line count. Use startLine/endLine to read the section you need to edit
        - writeFile overwrites the entire file — use editFile for partial modifications

        # Error Recovery

        - When a tool fails: 1) Read the error message 2) Check your parameters 3) Try a different approach
        - Do NOT repeat the same failing call with identical parameters
        - If a shell command fails, check the exit code and stderr for clues
        - If a file can't be read, verify the path exists with listFiles

        # Output Style

        - Be concise. Lead with actions, not reasoning
        - Don't narrate what you're about to do — just do it
        - When done, briefly summarize: what changed, which files, what to verify

        # Memory

        You have a saveMemory tool — use it to remember important context: user preferences, project details, recurring patterns. Memories persist across sessions.

        # Soul

        You are precise, resourceful, and action-oriented. You prefer doing over discussing.
        You treat the user's time as valuable. When you encounter obstacles, you adapt and find a way forward.
        You are honest about limitations and errors rather than hiding them.
        """

    private static let defaultUserPromptFile = """
        # Custom Instructions
        Add your project-specific rules, preferences, or personality tweaks here.
        These are appended to the base system prompt.
        """

    /// Load system prompt: hardcoded base + additive user customizations from `~/.mlx-serve/system-prompt.md`.
    static var systemPrompt: String {
        var prompt = defaultPromptFile
        ensureFile(at: promptPath, defaultContent: defaultUserPromptFile)
        let userPrompt = (try? String(contentsOfFile: promptPath, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !userPrompt.isEmpty {
            prompt += "\n\n# User Instructions\n" + userPrompt
        }
        return prompt
    }

    /// Load persistent memory from `~/.mlx-serve/memory.md`, capped at last 30 entries / ~2000 chars.
    static var memory: String {
        ensureFile(at: memoryPath, defaultContent: "")
        let content = (try? String(contentsOfFile: memoryPath, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !content.isEmpty else { return "" }
        // Keep only the last 30 entries to avoid bloating context
        let lines = content.components(separatedBy: "\n").filter { !$0.isEmpty }
        let capped = lines.suffix(30).joined(separator: "\n")
        let truncated = String(capped.prefix(2000))
        return "\n[Memories]\n\(truncated)"
    }

    /// Append a memory entry to `~/.mlx-serve/memory.md`.
    static func saveMemory(_ entry: String) {
        ensureFile(at: memoryPath, defaultContent: "")
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "- [\(timestamp)] \(entry)\n"
        if let handle = FileHandle(forWritingAtPath: memoryPath) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8) ?? Data())
            handle.closeFile()
        } else {
            try? line.write(toFile: memoryPath, atomically: true, encoding: .utf8)
        }
    }

    /// Open `system-prompt.md` in the user's default editor.
    static func openSystemPromptInEditor() {
        ensureFile(at: promptPath, defaultContent: defaultPromptFile)
        NSWorkspace.shared.open(URL(fileURLWithPath: promptPath))
    }

    private static func ensureFile(at path: String, defaultContent: String) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: path) {
            try? fm.createDirectory(atPath: mlxServeDir, withIntermediateDirectories: true)
            try? defaultContent.write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    /// OpenAI-format tool definitions with descriptions that guide parameter usage.
    static let toolDefinitions: [[String: Any]] = [
        [
            "type": "function",
            "function": [
                "name": "shell",
                "description": "Run a shell command. Example: {\"command\": \"ls -la /tmp\"}",
                "parameters": [
                    "type": "object",
                    "properties": ["command": ["type": "string", "description": "The shell command to execute"]],
                    "required": ["command"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "writeFile",
                "description": "Write content to a file (overwrites entire file). Use editFile for partial modifications. Example: {\"path\": \"src/main.swift\", \"content\": \"hello\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "File path (relative to working directory, or absolute)"],
                        "content": ["type": "string", "description": "File content to write"]
                    ],
                    "required": ["path", "content"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "readFile",
                "description": "Read a file's contents with optional line range. For large files, use startLine/endLine to read specific sections. Example: {\"path\": \"src/main.swift\", \"startLine\": \"10\", \"endLine\": \"50\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "File path (relative to working directory, or absolute)"],
                        "startLine": ["type": "string", "description": "First line to read (1-based, default: 1)"],
                        "endLine": ["type": "string", "description": "Last line to read (default: end of file)"]
                    ],
                    "required": ["path"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "editFile",
                "description": "Find and replace text in a file. You MUST provide all three params: path, find, replace. The find string must match the file content exactly. Always readFile first. Example: {\"path\": \"src/main.swift\", \"find\": \"old text\", \"replace\": \"new text\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "File path (relative to working directory, or absolute)"],
                        "find": ["type": "string", "description": "Exact text to find (must match file content)"],
                        "replace": ["type": "string", "description": "Replacement text"]
                    ],
                    "required": ["path", "find", "replace"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "searchFiles",
                "description": "Search file contents for a pattern (uses ripgrep if available). Returns matching lines with file paths and line numbers. Example: {\"pattern\": \"TODO\", \"include\": \"*.swift\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "pattern": ["type": "string", "description": "Text or regex pattern to search for"],
                        "path": ["type": "string", "description": "Directory to search in (default: working directory)"],
                        "include": ["type": "string", "description": "File glob filter (e.g. '*.swift', '*.ts')"],
                        "context": ["type": "string", "description": "Number of context lines around matches (0-10, default: 0)"],
                        "maxResults": ["type": "string", "description": "Max matches to return (default: 100)"]
                    ],
                    "required": ["pattern"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "listFiles",
                "description": "List files and directories. Use to explore project structure instead of shell ls/find. Returns paths matching the optional glob pattern. Example: {\"path\": \"src\", \"pattern\": \"*.swift\", \"recursive\": \"true\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "Directory to list (default: working directory)"],
                        "pattern": ["type": "string", "description": "Glob pattern to filter (e.g. '*.swift', '**/*.ts')"],
                        "recursive": ["type": "string", "description": "If 'true', search recursively (default: false)"]
                    ],
                    "required": [] as [String]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "browse",
                "description": "Browse a URL. Use action 'navigate' to load a page, then 'readText' to extract its text. Example: {\"action\": \"navigate\", \"url\": \"https://example.com\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "action": ["type": "string", "description": "navigate or readText"],
                        "url": ["type": "string", "description": "URL to browse"]
                    ],
                    "required": ["action", "url"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "webSearch",
                "description": "Search the web using DuckDuckGo. Example: {\"query\": \"latest news\"}",
                "parameters": [
                    "type": "object",
                    "properties": ["query": ["type": "string", "description": "Search query"]],
                    "required": ["query"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "saveMemory",
                "description": "Save a memory for future sessions. Use for user preferences, project context, or important facts. Example: {\"memory\": \"User prefers dark mode themes\"}",
                "parameters": [
                    "type": "object",
                    "properties": ["memory": ["type": "string", "description": "The memory to save"]],
                    "required": ["memory"]
                ]
            ] as [String: Any]
        ],
    ]
}

// MARK: - Prompt-based Skills

struct Skill {
    let name: String
    let description: String
    let triggers: [String]
    let body: String
}

class SkillManager {
    private let skillsDir: String
    private var skills: [Skill] = []
    private var lastModDate: Date?

    init() {
        skillsDir = NSString(string: "~/.mlx-serve/skills").expandingTildeInPath
        reload()
    }

    /// Returns skill index (always) + matching skill bodies (when triggered).
    func matchingSkills(for userMessage: String) -> String {
        reloadIfNeeded()
        guard !skills.isEmpty else { return "" }

        let lower = userMessage.lowercased()
        var result = "\nAvailable skills: " + skills.map { "\($0.name) (\($0.description))" }.joined(separator: ", ")

        let matched = skills.filter { skill in
            skill.triggers.contains { lower.contains($0) }
        }
        for skill in matched {
            result += "\n\n## Skill: \(skill.name)\n\(skill.body)"
        }

        return result
    }

    // MARK: - Private

    private func reloadIfNeeded() {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: skillsDir),
              let modDate = attrs[.modificationDate] as? Date else {
            if !skills.isEmpty { skills = [] }
            return
        }
        if lastModDate != modDate { reload() }
    }

    private func reload() {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(atPath: skillsDir) else {
            skills = []
            lastModDate = nil
            return
        }
        lastModDate = (try? fm.attributesOfItem(atPath: skillsDir))?[.modificationDate] as? Date
        skills = files.filter { $0.hasSuffix(".md") }.compactMap { file in
            let path = (skillsDir as NSString).appendingPathComponent(file)
            guard let content = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
            return parseSkill(content)
        }
    }

    private func parseSkill(_ content: String) -> Skill? {
        guard content.hasPrefix("---") else { return nil }
        let afterOpener = content.index(content.startIndex, offsetBy: 3)
        guard let closeRange = content.range(of: "\n---", range: afterOpener..<content.endIndex) else { return nil }

        let frontmatter = String(content[afterOpener..<closeRange.lowerBound])
        let body = String(content[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)

        var name = ""
        var description = ""
        var triggers: [String] = []

        for line in frontmatter.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard let colonIdx = trimmed.firstIndex(of: ":") else { continue }
            let key = trimmed[trimmed.startIndex..<colonIdx].trimmingCharacters(in: .whitespaces)
            let value = String(trimmed[trimmed.index(after: colonIdx)...]).trimmingCharacters(in: .whitespaces)

            switch key {
            case "name": name = value
            case "description": description = value
            case "trigger":
                triggers = value.components(separatedBy: ",")
                    .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
                    .filter { !$0.isEmpty }
            default: break
            }
        }

        guard !name.isEmpty, !triggers.isEmpty else { return nil }
        return Skill(name: name, description: description, triggers: triggers, body: body)
    }
}
