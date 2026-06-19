import Foundation
import AppKit

enum AgentPrompt {

    static let skillManager = SkillManager()

    private static let mlxServeDir = NSString(string: "~/.mlx-serve").expandingTildeInPath
    private static let promptPath = (mlxServeDir as NSString).appendingPathComponent("system-prompt.md")
    private static let memoryPath = (mlxServeDir as NSString).appendingPathComponent("memory.md")

    static let defaultPromptFile = """
        # System

        You are an autonomous macOS agent running on Apple Silicon. Act independently to complete tasks — do not ask the user for confirmation between steps. Execute multi-step tasks without pausing. Only respond to the user when the task is fully complete or if you hit a genuine ambiguity that cannot be resolved with tools.

        # Using Your Tools

        IMPORTANT: Use dedicated tools instead of shell equivalents:
        - readFile instead of `cat`, `head`, `tail`
        - writeFile instead of `echo >` or `cat <<EOF`
        - editFile instead of `sed` or `awk`
        - searchFiles instead of `grep` or `rg`
        - listFiles instead of `find` or `ls -R`

        Use shell only for: build/test commands, git operations, process management, installing packages, and commands with no dedicated tool equivalent. Shell commands run as a login shell — your PATH includes user tools (node, npm, python, brew, etc.).

        # Workspace Confinement

        All file operations (readFile, writeFile, editFile, searchFiles, listFiles) are confined to the working directory. You CANNOT read, write, or search outside it. Use relative paths — they resolve against the working directory automatically. Absolute paths are allowed only if they point inside the workspace. Do not attempt to access /tmp, /etc, ~, or any path outside the workspace — the tool will reject it.

        Tool arguments must be valid JSON: {"key": "value"}. NEVER call a tool with empty arguments {}. Every tool call MUST include at least the required parameters. If you are unsure what parameters to use, use readFile or listFiles first to gather information — do not guess with empty calls.

        CRITICAL: writeFile has a size limit — content longer than ~150 lines will be truncated and the write will fail. For large files (HTML pages, long scripts, etc.), use shell with a heredoc instead:
        shell: {"command": "cat > file.html << 'HEREDOC'\n<html>...</html>\nHEREDOC"}
        Use writeFile only for small files (< 100 lines). For anything larger, always use shell with cat heredoc. This is the most reliable approach for creating files with substantial content.

        # File Editing Rules

        - ALWAYS readFile before editFile — you must see the line numbers
        - readFile shows line numbers as "N| text"
        - editFile supports two modes:
          1. **Line-based (preferred)**: provide startLine, endLine, and replace. Use line numbers from readFile output. This is the most reliable approach.
          2. **Text-based**: provide find and replace. The find string must match file content exactly.
        - If editFile fails, readFile the file again and use line-based editing instead
        - writeFile overwrites the entire file — use editFile for partial modifications

        # Shell Rules

        - Each shell command runs in a fresh login shell. `cd` does NOT persist between calls.
        - To run a command in a subdirectory, use: `cd subdir && command` (all in one shell call)
        - Shell output includes `[cwd: /path]` so you can see where the command ran
        - For a long-lived process (a server, a watcher, anything that doesn't return on its own), DON'T use `&` — call shell with `run_in_background:"true"`. It returns instantly with a handle (bg1, bg2, …) and the process keeps running so you can keep working.
        - Check on it with `readProcessOutput` (its stdout/stderr since you last read) or a quick `curl`; stop it with `killProcess`; see all of them with `listProcesses`.

        # Serving Apps for Testing

        When you build something the user can open (web app, API, dev server), make it reachable from their other devices and ALWAYS hand back a working URL:
        - Bind to 0.0.0.0, never just localhost/127.0.0.1 — otherwise the user can't reach it from their phone or another machine. Vite: `npm run dev -- --host 0.0.0.0`; Next.js: `next dev -H 0.0.0.0`; Python http.server: `python3 -m http.server <port> --bind 0.0.0.0`; Flask/uvicorn: `--host 0.0.0.0`.
        - Prefer Docker when it's available — check with `docker info` (or `command -v docker`). If present, run the app in a container and publish the port: `docker run -d -p <port>:<port> <image>`. If Docker is NOT available, run it directly (`npm run dev`, etc.).
        - Start the server with shell `run_in_background:"true"` so your call returns instantly with a handle (a detached `docker run -d ...` is also fine), then verify it's up with `readProcessOutput` on the handle or a quick `curl http://localhost:<port>`. Stop it later with `killProcess`.
        - FINISH by telling the user the reachable URL built from this Mac's local network IP (given in the grounding line above): `http://<local-ip>:<port>`, plus one line on what they'll see there. If the IP wasn't provided, fall back to `http://localhost:<port>` and say so.

        # Scaffolding & Project Setup

        - Interactive scaffolders do NOT work here — your shell has no terminal, so any command that prompts for input (`npm create svelte@latest`, `npx sv create`, `npm create vite`, `create-react-app`, `npm init` without `-y`) gets EOF and fails or hangs. Never retry an interactive command — switch approaches immediately.
        - Prefer non-interactive invocations: pass every option as a flag and add `-y`/`--yes` (e.g. `npm init -y`) so nothing prompts.
        - If a tool can't run non-interactively, build the project by hand — it is more reliable than fighting a wizard: `npm install <deps>` to add packages, then create the config and source files yourself (writeFile for small files, shell `cat > file << 'EOF'` for large ones). For SvelteKit, that means `npm install` the deps and write `svelte.config.js`, `vite.config.js`, and `src/` files directly.
        - Plain `npm install` and most `npx <tool>` commands (e.g. `npx prisma init`, `npx prisma db push`) are fine — they don't prompt. Only the create/scaffold wizards are interactive.

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

    /// The agent system prompt. `~/.mlx-serve/system-prompt.md` is the single
    /// editable source of truth: seeded with `defaultPromptFile` on first use,
    /// then the file IS the prompt — users edit it wholesale via the tray's
    /// "Edit System Prompt" (`openSystemPromptInEditor` seeds the same default).
    /// Falls back to `defaultPromptFile` for an empty/unreadable file, or one
    /// still holding the pre-v26.6.11 additive "Custom Instructions" stub —
    /// migrating that stub in place so the editor and the agent stay in sync.
    static var systemPrompt: String {
        ensureFile(at: promptPath, defaultContent: defaultPromptFile)
        let raw = (try? String(contentsOfFile: promptPath, encoding: .utf8)) ?? ""
        let resolved = resolvePrompt(fileContent: raw)
        if resolved == defaultPromptFile,
           raw.trimmingCharacters(in: .whitespacesAndNewlines) != defaultPromptFile {
            try? defaultPromptFile.write(toFile: promptPath, atomically: true, encoding: .utf8)
        }
        return resolved
    }

    /// Pure resolution of the on-disk prompt file to the effective prompt: an
    /// empty file — or one still holding the legacy additive stub from before
    /// the prompt was unified into this file — yields the built-in default;
    /// anything else is the user's own prompt, verbatim (trimmed).
    static func resolvePrompt(fileContent: String) -> String {
        let trimmed = fileContent.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed.contains("These are appended to the base system prompt") {
            return defaultPromptFile
        }
        return trimmed
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

    /// Pre-serialized tool definitions JSON with guaranteed property key order.
    /// Critical: `path` appears before `content` in file tools so that if the model's
    /// output is truncated at max_tokens, the path is already emitted.
    static let toolDefinitionsJSON: String = #"""
    [
      {"type":"function","function":{"name":"shell","description":"Run a shell command. Commands run in the current working directory (use cwd tool to change it). For a long-lived process (a server, a watcher) set run_in_background to \"true\" — it returns instantly with a handle (bg1, bg2, …) and keeps running so you can keep working; poll it with readProcessOutput, stop it with killProcess. Example: {\"command\": \"ls -la /tmp\"}","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"},"run_in_background":{"type":"string","description":"Set to \"true\" to start a long-lived process in the background and return immediately with a handle (bg1, bg2, …). Default: foreground."}},"required":["command"]}}},
      {"type":"function","function":{"name":"cwd","description":"Change the working directory for all subsequent tool calls (shell, readFile, writeFile, etc.). Like cd but persistent. Example: {\"path\": \"myproject/src\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path (relative to current working directory, or absolute)"}},"required":["path"]}}},
      {"type":"function","function":{"name":"writeFile","description":"Write content to a file (overwrites). Only for SMALL files (under 100 lines). For large files use shell with cat heredoc instead. Example: {\"path\": \"src/main.swift\", \"content\": \"hello\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to working directory (must stay within workspace)"},"content":{"type":"string","description":"File content to write (keep under 100 lines — use shell cat heredoc for larger files)"}},"required":["path","content"]}}},
      {"type":"function","function":{"name":"readFile","description":"Read a file's contents with optional line range. For large files, use startLine/endLine to read specific sections. Example: {\"path\": \"src/main.swift\", \"startLine\": \"10\", \"endLine\": \"50\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to working directory (must stay within workspace)"},"startLine":{"type":"string","description":"First line to read (1-based, default: 1)"},"endLine":{"type":"string","description":"Last line to read (default: end of file)"}},"required":["path"]}}},
      {"type":"function","function":{"name":"editFile","description":"Edit a file. Two modes: (1) Line-based: provide path, startLine, endLine, replace — replaces those lines. (2) Text-based: provide path, find, replace — find must match exactly. Prefer line-based editing. Always readFile first to see line numbers. Example line-based: {\"path\": \"src/main.js\", \"startLine\": \"5\", \"endLine\": \"8\", \"replace\": \"new code here\"}. Example text-based: {\"path\": \"src/main.js\", \"find\": \"old text\", \"replace\": \"new text\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path relative to working directory (must stay within workspace)"},"startLine":{"type":"string","description":"First line to replace (1-based, from readFile output)"},"endLine":{"type":"string","description":"Last line to replace (1-based, defaults to startLine)"},"find":{"type":"string","description":"Exact text to find (for text-based mode)"},"replace":{"type":"string","description":"Replacement text"}},"required":["path"]}}},
      {"type":"function","function":{"name":"searchFiles","description":"Search file contents for a pattern (uses ripgrep if available). Returns matching lines with file paths and line numbers. Example: {\"pattern\": \"TODO\", \"include\": \"*.swift\"}","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Text or regex pattern to search for"},"path":{"type":"string","description":"Directory to search in (default: working directory)"},"include":{"type":"string","description":"File glob filter (e.g. '*.swift', '*.ts')"},"context":{"type":"string","description":"Number of context lines around matches (0-10, default: 0)"},"maxResults":{"type":"string","description":"Max matches to return (default: 100)"}},"required":["pattern"]}}},
      {"type":"function","function":{"name":"listFiles","description":"List files and directories. The root working directory listing is already in the system prompt — only use this for subdirectories or with glob patterns. Returns paths matching the optional glob pattern. Example: {\"path\": \"src\", \"pattern\": \"*.swift\", \"recursive\": \"true\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to list (default: working directory)"},"pattern":{"type":"string","description":"Glob pattern to filter (e.g. '*.swift', '**/*.ts')"},"recursive":{"type":"string","description":"If 'true', search recursively (default: false)"}},"required":[]}}},
      {"type":"function","function":{"name":"browse","description":"Browse and interact with web pages. Actions: 'navigate' (load URL), 'readText' (visible text from <main>/<article>, strips nav and <details> pickers), 'extractText' (innerText of elements matching a CSS selector — best for lists like article.Box-row, tr.athing, .result), 'readHTML' (raw HTML, first 8000 chars — mostly <head>), 'click', 'executeJS' (use to discover selectors), 'screenshot'. For data-listing pages prefer extractText. Provide a 'selector' param for click/extractText; provide 'script' for executeJS.","parameters":{"type":"object","properties":{"action":{"type":"string","description":"navigate, readText, extractText, readHTML, click, executeJS, or screenshot"},"url":{"type":"string","description":"URL to browse (required for navigate, optional for others)"},"selector":{"type":"string","description":"CSS selector for click/extractText (e.g. 'article.Box-row', 'button.submit', '#send-btn')"},"script":{"type":"string","description":"JavaScript code for executeJS action"}},"required":["action"]}}},
      {"type":"function","function":{"name":"webSearch","description":"Search the web using DuckDuckGo. Example: {\"query\": \"latest news\"}","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}}},
      {"type":"function","function":{"name":"saveMemory","description":"Save a memory for future sessions. Use for user preferences, project context, or important facts. Example: {\"memory\": \"User prefers dark mode themes\"}","parameters":{"type":"object","properties":{"memory":{"type":"string","description":"The memory to save"}},"required":["memory"]}}},
      {"type":"function","function":{"name":"createTask","description":"Create a background task that runs unattended as an autonomous agent and reports its result back to you when done — pushed to this chat if we're talking over Telegram, plus a desktop notification. Use it when asked to schedule something, do something later or periodically, or run a longer job and be notified. Omit 'schedule' (or set it to 'now') to run ONCE immediately in the background; provide a natural-language 'schedule' for a RECURRING task. The task has no memory of this conversation, so write 'goal' as a complete, self-contained instruction. Example: {\"goal\": \"Check Hacker News and summarize the top 3 stories with links\", \"schedule\": \"every day at 9am\"}","parameters":{"type":"object","properties":{"goal":{"type":"string","description":"The full, self-contained instruction the task should carry out"},"schedule":{"type":"string","description":"Optional. Natural-language recurring schedule like 'every day at 9am', 'every hour', 'Mon Wed Fri at 8am'. Omit or use 'now' to run once immediately."}},"required":["goal"]}}},
      {"type":"function","function":{"name":"readProcessOutput","description":"Read the stdout/stderr a background process (started via shell with run_in_background) has produced since you last read it. Use it to check whether a server came up or what a long job is printing. Example: {\"handle\": \"bg1\"}","parameters":{"type":"object","properties":{"handle":{"type":"string","description":"The process handle returned by shell run_in_background (bg1, bg2, …)"}},"required":["handle"]}}},
      {"type":"function","function":{"name":"killProcess","description":"Stop a background process started via shell with run_in_background. Example: {\"handle\": \"bg1\"}","parameters":{"type":"object","properties":{"handle":{"type":"string","description":"The process handle to stop (bg1, bg2, …)"}},"required":["handle"]}}},
      {"type":"function","function":{"name":"listProcesses","description":"List the background processes you've started in this chat, with their handles and status (running/exited). Takes no arguments. Example: {}","parameters":{"type":"object","properties":{},"required":[]}}}
    ]
    """#

    /// Tool definition for the per-session document index (mini RAG). Kept out
    /// of `toolDefinitionsJSON` because it is only advertised while a document
    /// folder is attached to the chat — see `ChatTurnEngine.combinedToolsJSON`.
    static let searchDocumentsToolJSON: String = #"""
    [
      {"type":"function","function":{"name":"searchDocuments","description":"Search the user's attached document folder for relevant excerpts. Returns the most relevant passages with their source filenames. Call this BEFORE answering any question about the attached documents. Use ONE short natural phrase per call (like a search box) — never a list of quoted keywords. Make several calls with different phrasings to cover different aspects. Example: {\"query\": \"customer frustrated about fees\"}","parameters":{"type":"object","properties":{"query":{"type":"string","description":"One short natural-language phrase describing what you are looking for. No boolean operators or quoted keyword lists."}},"required":["query"]}}}
    ]
    """#

    /// Minimal system prompt for docs-only mode (plain chat + attached folder —
    /// Agent and MCP toggles both off). Mirrors `mcpOnlySystemPrompt`: just
    /// enough instruction to drive the one available tool well.
    static func docsOnlySystemPrompt(folderName: String, fileCount: Int) -> String {
        """
        You are a helpful assistant. The user attached a folder of documents named "\(folderName)" (\(fileCount) files: chat transcripts, notes, PDFs, etc.). You cannot read whole files — your only access is the searchDocuments tool, which returns the passages most relevant to a query.

        For any question about the documents:
        1. Call searchDocuments with a focused query BEFORE answering. Never answer from memory alone.
        2. If results look incomplete, search again with different wording (names, dates, synonyms) — up to a few attempts.
        3. Answer from the retrieved excerpts and mention the source filenames you used.
        4. If nothing relevant comes back, say the documents don't seem to cover it.

        Questions unrelated to the documents can be answered normally without the tool.
        """
    }

    /// Section appended to the agent/MCP system prompt while a folder is attached.
    static func attachedDocumentsSection(folderName: String, fileCount: Int) -> String {
        """


        # Attached Documents
        The user attached a document folder "\(folderName)" (\(fileCount) files) to this chat. Use the searchDocuments tool to retrieve relevant excerpts before answering questions about its contents — it searches by meaning and returns passages with source filenames. Try multiple phrasings if a search misses. Cite the source filenames in your answer.
        """
    }

    /// Lightweight system prompt for MCP-only mode (Agent toggle off, MCP toggle on).
    /// Tells the model what MCP servers are available without dragging in the heavy agent rules.
    static func mcpOnlySystemPrompt(toolListing: String) -> String {
        var prompt = """
            You are a helpful assistant with access to external tools provided by Model Context Protocol (MCP) servers.

            Tools are namespaced as `<server>__<tool>`. Call them with valid JSON arguments matching each tool's schema. Only invoke a tool when it directly helps answer the user's request — otherwise reply normally.

            When you receive tool results, summarize them clearly for the user. If a tool returns an error, briefly explain what went wrong and either retry with corrected args or ask the user for clarification.
            """
        if !toolListing.isEmpty {
            prompt += "\n\n# Available MCP servers\n\n\(toolListing)"
        } else {
            prompt += "\n\nNote: no MCP servers are currently connected. The user can enable servers via the gear icon on the MCP toggle."
        }
        return prompt
    }

    /// Parsed tool definitions for param validation and example extraction.
    /// Includes the conditionally-advertised searchDocuments tool so its
    /// required params validate the same way as the always-on agent tools.
    /// Key order is NOT preserved here (Swift dictionaries); use `toolDefinitionsJSON` for API requests.
    static let toolDefinitions: [[String: Any]] = {
        func parse(_ json: String) -> [[String: Any]] {
            guard let data = json.data(using: .utf8),
                  let arr = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                fatalError("Invalid tool definitions JSON")
            }
            return arr
        }
        return parse(toolDefinitionsJSON) + parse(searchDocumentsToolJSON)
    }()
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
