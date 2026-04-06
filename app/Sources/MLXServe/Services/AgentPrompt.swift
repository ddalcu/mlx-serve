import Foundation

enum AgentPrompt {

    static let skillManager = SkillManager()

    static let systemPrompt = """
        You are a helpful macOS assistant with tool access. \
        When calling tools, ALWAYS pass arguments as JSON like {"key": "value"}. \
        Never omit required parameters — every tool call must include all required fields. \
        After receiving tool results, summarize what happened and answer the user. \
        For multi-step tasks, call tools one at a time.
        """

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
