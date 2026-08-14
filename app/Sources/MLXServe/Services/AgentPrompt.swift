import Foundation
import AppKit

enum AgentPrompt {

    static let skillManager = SkillManager()

    /// A concrete per-response output-budget note appended to the agent system
    /// prompt so the model self-limits BEFORE hitting the real ceiling. A vague
    /// "~200 lines" gets ignored (a model wrote ~750 lines straight past it); a
    /// real number is stronger — and it must be the EFFECTIVE budget, not a flat
    /// 16384. The model can emit at most ~2/5 of context per response (the rest
    /// holds prompt + history — mirrors AgentEngine's generation reservation),
    /// and no more than the `max_tokens` cap when one is set. `maxTokens <= 0`
    /// means "Auto" (no cap → bounded only by context). On a small-context /
    /// low-RAM machine the context term wins, which is the whole point of doing
    /// this dynamically: a flat 16384 would lie to the model about its real room.
    /// Stable within a session, so it rides in the volatile tail without
    /// disturbing the KV prefix.
    /// Below this effective budget the warning is emitted; at or above it the
    /// section is omitted entirely. A single one-shot file write measured
    /// live runs 8–10.7K tokens, so under ~12K a routine write genuinely
    /// risks being cut mid-call; above it the warning has no job to do — and
    /// an honest "you have ~419430 tokens per response" on a roomy machine
    /// reads as an invitation to one-shot a whole website in one multi-minute
    /// tool call (live 2026-07-03, Qwen3.6-27B). The chunking convention
    /// itself lives in the writeFile tool description, which no longer has a
    /// giant budget number contradicting it.
    static let outputBudgetGuidanceThreshold = 12288

    static func outputBudgetGuidance(maxTokens: Int, contextLength: Int) -> String {
        let contextBudget = contextLength > 0 ? max(256, contextLength * 2 / 5) : 8192
        let cap = maxTokens > 0 ? maxTokens : Int.max
        let effective = min(cap, contextBudget)
        guard effective < Self.outputBudgetGuidanceThreshold else { return "" }
        let safeTokens = max(256, min(effective / 2, 2048))
        let safeLines = max(20, min(safeTokens / 10, 200))
        return "\n\n# Output budget\n"
            + "This machine gives you about \(effective) tokens per response. Exceed it and the response is cut off mid-write: a tool call in progress is LOST (the file is NOT written) and the turn can end unfinished. Keep any single writeFile/editFile content under ~\(safeLines) lines (~\(safeTokens) tokens). For a larger file, write the first chunk, then call writeFile again with append:\"true\" for each remaining chunk — a shell heredoc has the same cap, so chunking is the only fix."
    }

    /// The music engine the Music window is set to. The `generate_music`
    /// schema is static, but the two engines take DIFFERENT inputs: Music 3
    /// is lyric-conditioned (a call with no lyrics fails before the model
    /// loads) and the server 400s bpm/keyscale/time_signature/vocal_language,
    /// while ACE-Step reads all of them. The model sees only the schema, so
    /// the contract rides the prompt — DERIVED from the preset's own flags,
    /// so a third engine gets a correct line without new prose.
    static func musicEngineNote(_ model: MusicModelPreset) -> String {
        var note = "\n\n# Music engine\nThe Music window is set to “\(model.name)”, so that is what `generate_music` runs. "
        if model.requiresLyrics {
            note += "It requires `lyrics` — a call without them fails. Write original words with section tags ([verse], [chorus]) on their own lines; for an instrumental send the tags with no words under them. "
        } else {
            note += "`lyrics` are optional — omit them for an instrumental. "
        }
        if model.supportsMusicalMeta {
            note += "It reads `bpm`, `keyscale`, `time_signature` and `vocal_language`; set them when the user names a tempo, key or language."
        } else {
            note += "It ignores tempo, key, meter and language arguments — put those in the `prompt` instead."
        }
        return note
    }

    private static let mlxServeDir = NSString(string: "~/.mlx-serve").expandingTildeInPath
    private static let promptPath = (mlxServeDir as NSString).appendingPathComponent("system-prompt.md")
    private static let memoryPath = (mlxServeDir as NSString).appendingPathComponent("memory.md")

    static let defaultPromptFile = """
        You are an autonomous agent. Finish the task yourself — don't ask for confirmation between steps. Reply to the user only when the task is done, or when you hit an ambiguity no tool can resolve.

        # Tools

        Prefer the dedicated tools over shell equivalents: readFile (not cat/head/tail), writeFile (not echo), editFile (not sed/awk), searchFiles (not grep/rg), listFiles (not find/ls -R). Use shell for build/test, git, installing packages, process management, and anything with no dedicated tool. Where shell commands actually run (host or sandbox) is stated in the Execution environment section below — use only commands that exist there.

        Tool arguments must be valid JSON, e.g. {"command": "ls -la"}. NEVER call a tool with empty {} — always include the required parameters; if unsure, gather context with readFile/listFiles first.

        # Files

        - File tools are confined to the working directory: use relative paths; paths outside it are rejected.
        - ALWAYS readFile before editFile (you need the line numbers it shows). editFile is line-based (startLine/endLine/replace — preferred) or text-based (find/replace — must match exactly); writeFile overwrites the whole file.
        - A whole file must fit in one response (it's part of your output), so a very large writeFile can get cut off mid-write — for a big file, write it in chunks: a first writeFile, then writeFile with append:"true" for each remaining chunk. If an output budget is stated at the end of this prompt, keep each call within it.

        # Shell

        - `cd` does NOT persist between calls — chain it (`cd dir && cmd`). Output is prefixed with [cwd: …].
        - For a long-lived process (server, watcher — anything that won't return on its own), set run_in_background:"true" (or just append `&`). It returns instantly with a handle (bg1, bg2, …) and keeps running so you can continue. Inspect it with readProcessOutput or curl, stop it with killProcess, list them with listProcesses.
        - Interactive scaffolders fail here (no TTY): `npm create …`, `npx sv create`, `create-react-app`, and `npm init` without `-y` hit EOF. Use non-interactive flags (`-y`/`--yes`) or build the project by hand (`npm install <deps>`, then write the config/source files yourself). Plain `npm install` and most `npx` commands are fine.

        # Serving apps

        If you start something the user can open (web app, dev server), bind to 0.0.0.0 (never just localhost), run it in the background, verify with curl, and finish by handing back the URL — use the URL form stated in the Execution environment section below.

        # Style

        Be concise; lead with actions. When a tool fails, read the error, fix your parameters, and try a different approach — never repeat the same failing call. When done, briefly summarize what changed and what to verify. Use saveMemory for durable user preferences or project facts.
        """

    /// Sandbox-aware "where do shell commands actually run" section, appended
    /// to the agent system prompt at request time (ChatTurnEngine reads the
    /// live sandbox setting). The base prompt file stays OS-NEUTRAL so one
    /// user-editable prompt serves both environments — without this split a
    /// macOS-flavored prompt sends `brew`/`open` into the Linux guest, and a
    /// Linux-flavored one sends `apt-get` at the host.
    static func executionEnvironmentSection(sandboxed: Bool) -> String {
        if sandboxed {
            return "\n\n# Execution environment\n"
                + "Shell commands run inside an isolated Linux VM (Debian, aarch64, GNU userland) — NOT on the host Mac. "
                + "python3/pip, node/npm, git, and curl are available; macOS-only commands (brew, open, pbcopy, osascript, defaults) are NOT — use Linux equivalents (apt-get to add packages). "
                + "The working folder is mounted at /workspace inside the VM; the file tools (readFile/writeFile/editFile/…) see the same files from the host side, so use them interchangeably with shell paths. "
                + "The VM normally has outbound network access (unless disabled in Settings — if a download fails with a network error, say so instead of retrying), and any TCP port a server listens on inside the VM is automatically reachable on the host at localhost with the SAME port — e.g. an Express app on 8080 is live at http://localhost:8080 for the user. Hand back exactly http://localhost:<port> — NEVER a LAN IP, <local-ip>, or guest IP URL: the mapping answers only on the user's own Mac, and other devices cannot reach the sandbox. "
                + "run_in_background starts the process inside the VM and returns a handle (bg1, bg2, …) you poll with readProcessOutput and stop with killProcess, exactly as on the host (its output is also appended to a log file inside the guest). "
                + "Exported shell variables persist between calls."
        }
        return "\n\n# Execution environment\n"
            + "Shell commands run directly on this Mac (macOS, zsh login shell, BSD userland; node, npm, python, git, and brew are on PATH). "
            + "Each call is a fresh login shell. "
            + "A server bound to 0.0.0.0 is reachable from the user's other devices too — hand back http://<local-ip>:<port> (the IP is in the grounding line above; if it's missing, use http://localhost:<port> and say so)."
    }

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
    /// anything else is the user's own prompt, verbatim (trimmed). A file
    /// holding a superseded shipped default is handled by the existing
    /// "Update System Prompt" menu flow (`isPromptOutdated`), not migrated here.
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

    // MARK: - Update to latest built-in default

    /// True when the on-disk prompt is a real prompt that differs from the latest
    /// built-in default — i.e. there's a newer default the user could pull in
    /// (or they have a customized prompt). Drives the "Update System Prompt" menu
    /// item's enabled state.
    static func isSystemPromptOutdated() -> Bool {
        isPromptOutdated(fileContent: try? String(contentsOfFile: promptPath, encoding: .utf8))
    }

    /// Pure decision behind `isSystemPromptOutdated`. Missing / empty / the legacy
    /// stub all resolve to the default (nothing to update); anything else that
    /// doesn't equal the default counts as outdated. Testable without the file.
    static func isPromptOutdated(fileContent: String?) -> Bool {
        guard let fileContent else { return false }
        return resolvePrompt(fileContent: fileContent) != defaultPromptFile
    }

    /// Backup filename for the user's current prompt, stamped so repeated updates
    /// never clobber an earlier backup. Pure + testable.
    static func promptBackupFileName(stamp: String) -> String {
        "system-prompt.backup-\(stamp).md"
    }

    /// Back up the current on-disk prompt, then overwrite it with the latest
    /// built-in default. Returns the backup path (nil when there was nothing to
    /// back up). The prompt is read live on every turn, so the next request picks
    /// up the new content automatically — no reload plumbing needed.
    @discardableResult
    static func updateSystemPromptToDefault() -> String? {
        let existing = (try? String(contentsOfFile: promptPath, encoding: .utf8)) ?? ""
        var backupPath: String? = nil
        if !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let f = DateFormatter()
            f.dateFormat = "yyyyMMdd-HHmmss"
            let path = (mlxServeDir as NSString).appendingPathComponent(promptBackupFileName(stamp: f.string(from: Date())))
            try? existing.write(toFile: path, atomically: true, encoding: .utf8)
            backupPath = path
        }
        try? FileManager.default.createDirectory(atPath: mlxServeDir, withIntermediateDirectories: true)
        try? defaultPromptFile.write(toFile: promptPath, atomically: true, encoding: .utf8)
        return backupPath
    }

    /// Menu action for "Update System Prompt": confirm (warn it's destructive),
    /// back up, overwrite with the latest default, then report where the backup
    /// went. No-ops with a friendly note when already up to date.
    @MainActor
    static func runSystemPromptUpdateFlow() {
        guard isSystemPromptOutdated() else {
            let a = NSAlert()
            a.messageText = "System prompt is up to date"
            a.informativeText = "Your system prompt already matches the latest built-in default."
            a.runModal()
            return
        }
        let confirm = NSAlert()
        confirm.alertStyle = .warning
        confirm.messageText = "Replace your system prompt with the latest default?"
        confirm.informativeText = "This overwrites ~/.mlx-serve/system-prompt.md with the latest built-in prompt. Your current prompt is backed up first so you can restore it."
        confirm.addButton(withTitle: "Update")
        confirm.addButton(withTitle: "Cancel")
        guard confirm.runModal() == .alertFirstButtonReturn else { return }

        let backup = updateSystemPromptToDefault()
        let done = NSAlert()
        done.messageText = "System prompt updated"
        done.informativeText = backup.map { "Updated to the latest default.\nYour previous prompt was saved to:\n\($0)" }
            ?? "Updated to the latest default."
        if backup != nil {
            done.addButton(withTitle: "Reveal Backup")
            done.addButton(withTitle: "OK")
        }
        let resp = done.runModal()
        if let backup, resp == .alertFirstButtonReturn {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: backup)])
        }
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
      {"type":"function","function":{"name":"writeFile","description":"Write a file. EVERY call needs BOTH \"path\" (where to write) and \"content\" (what to write) — emit \"path\" FIRST. A call with \"content\" but no \"path\" is REJECTED, so never omit it. Example: {\"path\": \"game/index.html\", \"content\": \"<!DOCTYPE html>...\"}. Overwrites by default; pass append:true to add to the end instead (creates the file if missing). Content rides in this one response, so for a large file write a first chunk (~200 lines), then call again with append:true per remaining chunk so nothing is cut off mid-write.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"REQUIRED — emit it FIRST, before content. Destination path relative to the working directory (e.g. \"game/index.html\"). A writeFile with no path FAILS; always set it."},"content":{"type":"string","description":"REQUIRED. The file body for this call (keep it within your per-response output budget; split a large file across multiple append:true calls). The body ALWAYS goes here — never in path or append."},"append":{"type":"boolean","description":"Append flag (a boolean, NOT text — emit bareword true/false, never \"true\"). true = add to the END of the file (creates it if missing); omit or false = overwrite. The file body ALWAYS goes in \"content\" — never here. Use append to write a large file across several calls."}},"required":["path","content"]}}},
      {"type":"function","function":{"name":"readFile","description":"Read a file's contents with optional line range. For large files, use startLine/endLine to read specific sections. Example: {\"path\": \"src/main.swift\", \"startLine\": \"10\", \"endLine\": \"50\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path, relative to the working directory"},"startLine":{"type":"string","description":"First line to read (1-based, default: 1)"},"endLine":{"type":"string","description":"Last line to read (default: end of file)"}},"required":["path"]}}},
      {"type":"function","function":{"name":"editFile","description":"Edit a file. Two modes: (1) Line-based: provide path, startLine, endLine, replace — replaces those lines. (2) Text-based: provide path, find, replace — find must match exactly. Prefer line-based editing. Always readFile first to see line numbers. Example line-based: {\"path\": \"src/main.js\", \"startLine\": \"5\", \"endLine\": \"8\", \"replace\": \"new code here\"}. Example text-based: {\"path\": \"src/main.js\", \"find\": \"old text\", \"replace\": \"new text\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path, relative to the working directory"},"startLine":{"type":"string","description":"First line to replace (1-based, from readFile output)"},"endLine":{"type":"string","description":"Last line to replace (1-based, defaults to startLine)"},"find":{"type":"string","description":"Exact text to find (for text-based mode)"},"replace":{"type":"string","description":"Replacement text"}},"required":["path"]}}},
      {"type":"function","function":{"name":"searchFiles","description":"Search file contents for a pattern (uses ripgrep if available). Returns matching lines with file paths and line numbers. Example: {\"pattern\": \"TODO\", \"include\": \"*.swift\"}","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Text or regex pattern to search for"},"path":{"type":"string","description":"Directory to search in (default: working directory)"},"include":{"type":"string","description":"File glob filter (e.g. '*.swift', '*.ts')"},"context":{"type":"string","description":"Number of context lines around matches (0-10, default: 0)"},"maxResults":{"type":"string","description":"Max matches to return (default: 100)"}},"required":["pattern"]}}},
      {"type":"function","function":{"name":"listFiles","description":"List files and directories. The root working directory listing is already in the system prompt — only use this for subdirectories or with glob patterns. Returns paths matching the optional glob pattern. Example: {\"path\": \"src\", \"pattern\": \"*.swift\", \"recursive\": \"true\"}","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory to list (default: working directory)"},"pattern":{"type":"string","description":"Glob pattern to filter (e.g. '*.swift', '**/*.ts')"},"recursive":{"type":"string","description":"If 'true', search recursively (default: false)"}},"required":[]}}},
      {"type":"function","function":{"name":"browse","description":"Browse web pages. Actions: navigate (load a URL), readText (visible text), extractText (innerText of a CSS selector — best for lists), readHTML, click, executeJS (to discover selectors), screenshot. Give 'selector' for click/extractText, 'script' for executeJS. Example: {\"action\": \"navigate\", \"url\": \"https://example.com\"}","parameters":{"type":"object","properties":{"action":{"type":"string","description":"navigate, readText, extractText, readHTML, click, executeJS, or screenshot"},"url":{"type":"string","description":"URL to browse (required for navigate, optional for others)"},"selector":{"type":"string","description":"CSS selector for click/extractText (e.g. 'article.Box-row', 'button.submit', '#send-btn')"},"script":{"type":"string","description":"JavaScript code for executeJS action"}},"required":["action"]}}},
      {"type":"function","function":{"name":"webSearch","description":"Search the web using DuckDuckGo. Example: {\"query\": \"latest news\"}","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}}},
      {"type":"function","function":{"name":"saveMemory","description":"Save a memory for future sessions. Use for user preferences, project context, or important facts. Example: {\"memory\": \"User prefers dark mode themes\"}","parameters":{"type":"object","properties":{"memory":{"type":"string","description":"The memory to save"}},"required":["memory"]}}},
      {"type":"function","function":{"name":"createTask","description":"Create a background task that runs unattended as an autonomous agent and reports its result back to you when done — pushed to this chat if we're talking over Telegram, plus a desktop notification. Use it when asked to schedule something, do something later or periodically, or run a longer job and be notified. Omit 'schedule' (or set it to 'now') to run ONCE immediately in the background; provide a natural-language 'schedule' for a RECURRING task. The task has no memory of this conversation, so write 'goal' as a complete, self-contained instruction. Example: {\"goal\": \"Check Hacker News and summarize the top 3 stories with links\", \"schedule\": \"every day at 9am\"}","parameters":{"type":"object","properties":{"goal":{"type":"string","description":"The full, self-contained instruction the task should carry out"},"schedule":{"type":"string","description":"Optional. Natural-language recurring schedule like 'every day at 9am', 'every hour', 'Mon Wed Fri at 8am'. Omit or use 'now' to run once immediately."}},"required":["goal"]}}},
      {"type":"function","function":{"name":"readProcessOutput","description":"Read the stdout/stderr a background process (started via shell with run_in_background) has produced since you last read it. Use it to check whether a server came up or what a long job is printing. Example: {\"handle\": \"bg1\"}","parameters":{"type":"object","properties":{"handle":{"type":"string","description":"The process handle returned by shell run_in_background (bg1, bg2, …)"}},"required":["handle"]}}},
      {"type":"function","function":{"name":"killProcess","description":"Stop a background process started via shell with run_in_background. Example: {\"handle\": \"bg1\"}","parameters":{"type":"object","properties":{"handle":{"type":"string","description":"The process handle to stop (bg1, bg2, …)"}},"required":["handle"]}}},
      {"type":"function","function":{"name":"listProcesses","description":"List the background processes you've started in this chat, with their handles and status (running/exited). Takes no arguments. Example: {}","parameters":{"type":"object","properties":{},"required":[]}}},
      {"type":"function","function":{"name":"generate_image","description":"Draw an image from a text prompt and show it INLINE in this chat. Call it only when the user wants a picture PRODUCED — a question that merely mentions images (how image generation works, which model to use) is answered in text with NO tool call. Chat images are quick previews at fast settings; the Image window is where full-quality work happens. This is a HEAVY one-shot (it loads a multi-GB diffusion model and can take a while), so make at most ONE media generation per user message. The model must already be downloaded; if it isn't you'll get a message telling the user to open the Image window once. The result is displayed automatically — reply with one short sentence, and never invent a file name or a markdown image link. Example: {\"prompt\": \"a red fox in the snow at golden hour, cinematic\"}","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"What to draw — a vivid, detailed description of the image."},"size":{"type":"string","description":"Optional shape: WIDTHxHEIGHT like \"1344x768\", or a plain ratio like \"16:9\" or \"9:16\". Snapped to the nearest size the model was trained on. Omit unless the user asked for a shape."}},"required":["prompt"]}}},
      {"type":"function","function":{"name":"generate_speech","description":"Speak a line of text out loud with the neural voice model and attach the clip to this chat. Use it when the user wants something SAID or narrated as an audio file — not for ordinary answers, which are always plain text. For music or a song use generate_music instead. This is a HEAVY one-shot (a voice model loads first), so make at most ONE media generation per user message. The clip is attached and playable automatically — reply with one short sentence and never invent a file name. Example: {\"text\": \"Your coffee is ready.\"}","parameters":{"type":"object","properties":{"text":{"type":"string","description":"Exactly the words to speak. Write them out in full — no markdown, no stage directions."},"speed":{"type":"string","description":"Optional speaking rate, 0.5 (slow) to 2.0 (fast). Omit for normal."}},"required":["text"]}}},
      {"type":"function","function":{"name":"generate_music","description":"Compose a piece of music from a style description and attach the track to this chat. Describe genre, mood and instrumentation in the prompt; supply lyrics only if the user wants singing (omit them for an instrumental). For a spoken line use generate_speech instead. This is a HEAVY one-shot (a multi-GB model loads and a 30s track takes a while), so make at most ONE media generation per user message. The track is attached and playable automatically — reply with one short sentence and never invent a file name. Example: {\"prompt\": \"warm lo-fi hip hop with a mellow Rhodes piano, dusty vinyl texture and soft boom-bap drums\", \"duration_seconds\": \"30\", \"bpm\": \"85\"}","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"The style: genre, mood, instrumentation, tempo feel. Not lyrics."},"lyrics":{"type":"string","description":"Optional words to sing, with [Verse] / [Chorus] section tags. Omit for an instrumental."},"duration_seconds":{"type":"string","description":"Track length in seconds, 10 to 600. SIZE IT TO THE LYRICS — whatever does not fit is cut off mid-song: about 4 seconds per sung line plus 15 for intro and outro, so a 24-line song needs ~120, not 30. Omit only for an instrumental (30s preview)."},"bpm":{"type":"string","description":"Optional tempo in beats per minute, 30 to 300 (85 hip-hop, 120 pop/house, 128 EDM, 174 drum & bass). Omit to let the model choose."},"keyscale":{"type":"string","description":"Optional musical key as note + major/minor, e.g. \"A minor\", \"C major\", \"Eb major\" (notes C C# D Eb E F F# G Ab A Bb B). Omit to let the model choose."},"time_signature":{"type":"string","description":"Optional meter, one of \"4/4\", \"3/4\", \"2/4\", \"6/8\". Omit to let the model choose."},"vocal_language":{"type":"string","description":"Optional language for the SUNG lyrics, e.g. \"English\", \"Spanish\", \"Japanese\". Only meaningful with lyrics. Omit unless the user asked."}},"required":["prompt"]}}},
      {"type":"function","function":{"name":"generate_video","description":"Generate a short video clip from a text prompt and attach it to this chat. Call it only when the user wants a clip PRODUCED. This is the SLOWEST tool here — a couple of seconds of video takes MINUTES on one GPU and blocks the chat while it runs — so make at most ONE media generation per user message and keep the clip short. The model must already be downloaded; if it isn't you'll get a message telling the user to open the Video window once. The clip is attached and playable automatically — reply with one short sentence and never invent a file name. Example: {\"prompt\": \"a timelapse of clouds rolling over a mountain range at sunrise\", \"seconds\": \"2\"}","parameters":{"type":"object","properties":{"prompt":{"type":"string","description":"What the clip should show — subject, motion, camera, lighting."},"seconds":{"type":"string","description":"Optional clip length in seconds, up to 4. Omit for 2. Every extra second costs another minute of generation."},"size":{"type":"string","description":"Optional shape: WIDTHxHEIGHT like \"704x480\", or a plain ratio like \"16:9\" or \"9:16\". Snapped to the nearest size the model was trained on."}},"required":["prompt"]}}}
    ]
    """#

    /// `toolDefinitionsJSON` filtered to the tools an agent may use.
    ///
    /// The filter is LINE-BASED on the literal above, deliberately: the key order
    /// inside each definition is load-bearing (`path` before `content`, so a
    /// truncated call still carries the path), and a JSONSerialization round-trip
    /// would silently reorder it. One tool per line is what makes that safe —
    /// keep it that way.
    ///
    /// An empty set yields `[]`, which `ChatTurnEngine.combinedToolsJSON` drops,
    /// so "no tools" means no `tools` field on the request rather than an empty
    /// array the model has to interpret.
    static func toolDefinitionsJSON(allowing allowed: Set<AgentToolKind>) -> String {
        let kept = toolDefinitionsJSON
            .components(separatedBy: "\n")
            .compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("{") else { return nil }   // skips [ and ]
                guard let name = toolName(inDefinitionLine: trimmed),
                      let kind = AgentToolKind(rawValue: name),
                      allowed.contains(kind) else { return nil }
                // Drop the separator comma; it's re-added by the join below.
                return trimmed.hasSuffix(",") ? String(trimmed.dropLast()) : trimmed
            }
        guard !kept.isEmpty else { return "[]" }
        return "[\n  " + kept.joined(separator: ",\n  ") + "\n]"
    }

    /// The `"name":"…"` of a single tool-definition line. The function name is the
    /// first such key on the line (parameter properties come later), so a plain
    /// scan is enough — and stays order-preserving, unlike a parse.
    static func toolName(inDefinitionLine line: String) -> String? {
        guard let key = line.range(of: "\"name\":\"") else { return nil }
        let rest = line[key.upperBound...]
        guard let end = rest.firstIndex(of: "\"") else { return nil }
        return String(rest[rest.startIndex..<end])
    }

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

    init(skillsDir: String? = nil) {
        self.skillsDir = skillsDir ?? NSString(string: "~/.mlx-serve/skills").expandingTildeInPath
        seedBuiltinSkills()
        reload()
    }

    /// Absolute path of the skills directory — used by the "Agent → Open Skills
    /// Folder" menu item (accessing the shared manager also triggers seeding).
    var skillsDirectory: String { skillsDir }

    /// Seed the skills the app ships with. Per FILE against a ledger of names
    /// already seeded (`.seeded` in the folder), not directory existence: the
    /// old gate was "does the dir exist", so a built-in added later would have
    /// reached brand-new installs only. Deleting a seeded skill still sticks —
    /// its name stays in the ledger. A pre-ledger folder is migrated by taking
    /// what is already ON DISK as seeded, so an example the user deleted back
    /// then is not resurrected.
    private func seedBuiltinSkills() {
        let fm = FileManager.default
        let firstRun = !fm.fileExists(atPath: skillsDir)
        if firstRun {
            guard (try? fm.createDirectory(atPath: skillsDir, withIntermediateDirectories: true)) != nil else { return }
        }
        let ledger = (skillsDir as NSString).appendingPathComponent(".seeded")
        var seeded: Set<String>
        if let text = try? String(contentsOfFile: ledger, encoding: .utf8) {
            seeded = Set(text.components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty })
        } else {
            seeded = Set((try? fm.contentsOfDirectory(atPath: skillsDir)) ?? [])
            // Pre-ledger folder: the old scheme already made its call on the
            // review example (seeded on first run, possibly deleted since).
            if !firstRun { seeded.insert("review.md") }
        }
        var changed = false
        for skill in Self.builtinSkills where !seeded.contains(skill.file) {
            let path = (skillsDir as NSString).appendingPathComponent(skill.file)
            if !fm.fileExists(atPath: path) {
                try? skill.body.write(toFile: path, atomically: true, encoding: .utf8)
            }
            seeded.insert(skill.file)
            changed = true
        }
        if changed {
            try? seeded.sorted().joined(separator: "\n").write(toFile: ledger, atomically: true, encoding: .utf8)
        }
    }

    /// The skills the app ships with, seeded into the folder so they are
    /// discoverable and editable like any user skill.
    static var builtinSkills: [(file: String, body: String)] {
        [("review.md", defaultSkillFile), ("music3.md", musicSkillFile)]
    }

    /// The example skill shipped on first run. Genuinely useful (a focused,
    /// read-only code review), and doubles as a format reference for users
    /// writing their own.
    static let defaultSkillFile = """
    ---
    name: review
    description: Review the current changes or a file for bugs and improvements
    trigger: review, code review, review this
    ---
    When the user asks you to review code:

    1. Decide what to review. Default to the working changes — run `git diff` and
       `git diff --staged`. If they name a file or folder, read that instead. Read
       enough surrounding code to judge the change in context; don't review a diff
       in isolation.

    2. Report findings grouped by severity, most important first:
       - Bugs / correctness — logic errors, unhandled edge cases, nil/force-unwraps,
         off-by-ones, races, resource leaks.
       - Risks — security holes, data loss, performance cliffs.
       - Cleanups — naming, duplication, dead code, simpler equivalents.
       For each finding cite file:line and give a concrete fix.

    3. Be specific and honest. Say "this looks good" only when it genuinely does.
       This is a read-only review — don't change code unless the user explicitly
       asks you to apply a fix.

    (This is an example skill that ships with the app — edit it, or add your own
    .md files in this folder. Each skill needs frontmatter with `name`,
    `description`, and `trigger` (comma-separated phrases that activate it when they
    appear in your message). Everything below the frontmatter is injected into the
    system prompt when a trigger matches.)
    """

    /// Music skill: MiniMax Music 3 takes a structured CAPTION plus separate
    /// lyrics, and a one-line "lo-fi beat" prompt leaves every arrangement
    /// decision to the model. A condensed version of MiniMax's own
    /// music-caption-rewriter — the three caption blocks and the lyric-tag
    /// contract, no template library.
    static let musicSkillFile = """
    ---
    name: music3
    description: Write a MiniMax Music 3 caption plus lyrics, then generate the track
    trigger: song, music, lyrics, soundtrack, jingle
    ---
    When the user asks for a song or a piece of music, write the two things the
    music model conditions on, then call `generate_music` once with them.

    This caption format is what MiniMax Music 3 was trained on. ACE-Step reads
    the same text happily, but takes tempo, key and meter from its own
    arguments instead.

    ## prompt — the caption

    Three plain-prose blocks in this order, about 120-250 words total. Plain
    text, no markdown, and never any lyrics.

    Global Metadata
    Genre and subgenre, tempo feel, the emotional arc from first bar to last,
    and the production profile: soundstage, warmth, dynamics.

    Vocal Details
    Lead voice — gender, timbre, register, delivery. Harmony and backing
    vocals, and restrained vocal effects. For an instrumental, say it is
    instrumental and name the instrument carrying the melody instead.

    Arrangement
    The song as a timeline: what enters, exits or intensifies in each section,
    how the groove develops, how sections transition. Concrete musical changes,
    not a list of gear.

    ## lyrics — the words

    Original words you write, unless the user gives you theirs. 
    Section tags go on their own line, above the
    lines they cover: [intro] [verse] [pre-chorus] [chorus] [bridge]
    [instrumental] [solo] [outro].

    The caption never contains lyrics, and the lyrics never contain style notes.

    ## Rules

    - The "Music engine" section of your instructions names the engine that
      will actually run, whether `lyrics` are required, and whether tempo, key
      and meter arguments are read. It wins over anything here.
    - Do not invent an exact BPM or key the user did not ask for — describe the
      tempo instead ("slow, unhurried", "driving four-on-the-floor").
    - Size `duration_seconds` to the lyrics: about 4 seconds a sung line plus 15
      for intro and outro, so a 24-line song needs ~120, not the 30-second
      default — whatever does not fit is cut off mid-song. On Music 3 the
      duration is an upper bound, so rounding up costs nothing.
    - One track per message. Always show the user what you used: the caption in
      its own fenced code block, then the lyrics in a second one, so either can
      be copied straight into the Music window's Style and Lyrics fields.
    - No `generate_music` tool in this chat? Write the same two code blocks and
      tell the user to switch Tools on (or paste them into the Music window) to
      render the track. Do not pretend to have made audio.
    """

    /// Whole-word/phrase trigger match: `keyword` must appear in `text` bounded
    /// by non-alphanumeric characters or the string edges — a plain
    /// `text.contains(keyword)` fired on fragments inside unrelated words
    /// ("ui" in "build", "review" in "preview"), injecting whole skill bodies
    /// on ordinary turns (issue #92). Multi-word phrases ("code review") and
    /// non-alphanumeric edges ("/plan", "requirements.txt") still match.
    /// Both arguments are already lowercased.
    static func triggerMatches(_ text: String, _ keyword: String) -> Bool {
        guard !keyword.isEmpty else { return false }
        func isAlnum(_ c: Character) -> Bool { c.isLetter || c.isNumber }
        var searchStart = text.startIndex
        while let range = text.range(of: keyword, range: searchStart..<text.endIndex) {
            let beforeOK = range.lowerBound == text.startIndex
                || !isAlnum(text[text.index(before: range.lowerBound)])
            let afterOK = range.upperBound == text.endIndex
                || !isAlnum(text[range.upperBound])
            if beforeOK && afterOK { return true }
            searchStart = text.index(after: range.lowerBound)
        }
        return false
    }

    /// Every skill, for the composer's "/" menu.
    var summaries: [SkillSummary] {
        reloadIfNeeded()
        return skills.map { SkillSummary(name: $0.name, description: $0.description) }
    }

    /// The body block for a skill invoked by name, or "" — the composer's
    /// `/name` path, which works in every chat mode (the user asked for it by
    /// name, so it does not wait for the agent loop's trigger matching).
    func invokedSkill(for userMessage: String) -> String {
        reloadIfNeeded()
        guard let name = SlashCommands.invokedSkillName(in: userMessage),
              let skill = skills.first(where: { $0.name.lowercased() == name }) else { return "" }
        return "\n\n## Skill: \(skill.name)\n\(skill.body)"
    }

    /// Returns skill index (always) + matching skill bodies (when triggered,
    /// or when the message invokes one by name).
    func matchingSkills(for userMessage: String) -> String {
        reloadIfNeeded()
        guard !skills.isEmpty else { return "" }

        let lower = userMessage.lowercased()
        var result = "\nAvailable skills: " + skills.map { "\($0.name) (\($0.description))" }.joined(separator: ", ")

        let invoked = SlashCommands.invokedSkillName(in: userMessage)
        let matched = skills.filter { skill in
            skill.name.lowercased() == invoked
                || skill.triggers.contains { Self.triggerMatches(lower, $0) }
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
