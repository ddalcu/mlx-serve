import Foundation
import AppKit

enum AgentPrompt {

    static let skillManager = SkillManager()

    private static let mlxServeDir = NSString(string: "~/.mlx-serve").expandingTildeInPath
    private static let promptPath = (mlxServeDir as NSString).appendingPathComponent("system-prompt.md")
    private static let memoryPath = (mlxServeDir as NSString).appendingPathComponent("memory.md")

    private static let defaultPromptFile = """
        # System Prompt

        You are an autonomous macOS agent. Act independently to complete tasks — do not ask the user for confirmation or permission between steps.
        When a task requires multiple steps, execute them all without pausing.
        If a command fails, diagnose the issue and retry with a corrected approach.
        If you need information, use your tools to find it rather than asking the user.
        Only respond to the user when the task is fully complete or if you hit a genuine ambiguity that cannot be resolved with tools.
        Tool arguments must be JSON: {"key": "value"}. Never omit required parameters.

        You have a `saveMemory` tool — use it to remember important context: user preferences, project details, recurring patterns, or anything that would help in future conversations. Memories persist across sessions.

        ## Soul

        You are precise, resourceful, and action-oriented. You prefer doing over discussing.
        You treat the user's time as valuable — don't narrate what you're about to do, just do it.
        When you encounter obstacles, you adapt and find a way forward.
        You are honest about limitations and errors rather than hiding them.
        """

    /// Load system prompt from `~/.mlx-serve/system-prompt.md`, seeding the file on first run.
    static var systemPrompt: String {
        ensureFile(at: promptPath, defaultContent: defaultPromptFile)
        return (try? String(contentsOfFile: promptPath, encoding: .utf8)) ?? defaultPromptFile
    }

    /// Load persistent memory from `~/.mlx-serve/memory.md`.
    static var memory: String {
        ensureFile(at: memoryPath, defaultContent: "")
        let content = (try? String(contentsOfFile: memoryPath, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return content.isEmpty ? "" : "\n[Memories]\n\(content)"
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
                "description": "Write content to a file. Example: {\"path\": \"/tmp/f.txt\", \"content\": \"hello\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "Absolute file path"],
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
                "description": "Read a file's contents. Example: {\"path\": \"/tmp/f.txt\"}",
                "parameters": [
                    "type": "object",
                    "properties": ["path": ["type": "string", "description": "Absolute file path to read"]],
                    "required": ["path"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "editFile",
                "description": "Find and replace text in a file. Example: {\"path\": \"/tmp/f.txt\", \"find\": \"old\", \"replace\": \"new\"}",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string", "description": "Absolute file path"],
                        "find": ["type": "string", "description": "Text to find"],
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
                "description": "Search files for a text pattern (grep). Example: {\"pattern\": \"TODO\"}",
                "parameters": [
                    "type": "object",
                    "properties": ["pattern": ["type": "string", "description": "Text pattern to search for"]],
                    "required": ["pattern"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "browse",
                "description": "Browse a URL. Use action 'navigate' to visit, 'readText' to extract page text. Example: {\"action\": \"readText\", \"url\": \"https://example.com\"}",
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
