import Foundation

/// Which pre-send nudge to show when the user's message looks like it needs a
/// mode that's currently off. Drives the confirmation dialog in ChatView.
enum IntentPrompt: Identifiable, Equatable {
    case agent
    case mcp
    var id: Int { self == .agent ? 0 : 1 }
}

/// Lightweight, string-based detection of "this message wants a capability that
/// isn't enabled" — used to nudge the user to turn on Agent Mode or MCP before
/// sending. Deliberately a simple keyword/phrase matcher (no model call): it
/// errs toward NOT firing on ordinary chat ("write a poem", "explain X") and
/// only flags requests tied to real tool use (files, shell, web) or to a named,
/// currently-enabled MCP server. Pure → unit-tested in ComposerInputTests.
enum ComposerIntent {

    /// Action verbs that, paired with a tech/file object below, indicate a
    /// build/do task rather than a content request ("write a poem").
    private static let actionVerbs = [
        "create", "make", "build", "generate", "write", "code", "implement",
        "scaffold", "deploy", "add", "edit", "modify", "update", "refactor",
        "rename", "delete", "remove", "fix", "set up",
    ]

    /// Objects that turn an action verb into a tool task. "poem"/"story"/"essay"
    /// are intentionally absent so creative requests don't trip the nudge.
    private static let techObjects = [
        "file", "folder", "directory", "website", "web page", "webpage",
        "web app", "html", "css", "javascript", " js ", "python", "script",
        "program", " app", "application", "server", "api", "component",
        "function", "class", "game", "dashboard", "landing page", "project",
        "repo", "repository", "config", "readme", "code", "bug", "test",
        ".html", ".js", ".py", ".css", ".ts", ".swift", ".json", ".txt", ".md",
    ]

    /// Phrases that signal tool use on their own, regardless of object.
    private static let strongSignals = [
        "run the", "run it", "run this", "run a ", "execute ", "install ",
        "npm ", "npx ", "pip ", "brew ", "git clone", "git ", "mkdir",
        "curl ", "wget ", "download ", "search the web", "google ",
        "browse ", "scrape ", "look it up", "list the files", "list files",
        "read the file", "open the file", "open the website", "save it to",
        "clone the repo", "in the terminal", "command line", "start the server",
        "start a server", "spin up", "the filesystem", "on disk",
    ]

    /// True when the message reads like an agentic task (files / shell / web)
    /// rather than a question or a content request.
    static func wantsAgent(_ text: String) -> Bool {
        let t = text.lowercased()
        if strongSignals.contains(where: { t.contains($0) }) { return true }
        let hasVerb = actionVerbs.contains { v in
            t.hasPrefix(v + " ") || t.contains(" " + v + " ") || t.contains(v + "s ")
        }
        guard hasVerb else { return false }
        return techObjects.contains { t.contains($0) }
    }

    /// True when the message references an enabled MCP server by name, or the
    /// literal word "mcp". `serverNames` are the user's currently-enabled MCP
    /// server ids; matching is case-insensitive.
    static func wantsMCP(_ text: String, serverNames: [String]) -> Bool {
        let t = text.lowercased()
        if t.contains("mcp") { return true }
        return serverNames.contains { name in
            let n = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            return !n.isEmpty && t.contains(n)
        }
    }
}

/// Per-session record of which nudges the user has already declined ("Send
/// anyway"), so we stop re-prompting that suggestion for that chat. Keyed by
/// session id because ChatDetailView is reused across tabs (a plain Bool would
/// leak one tab's choice into the others — same class as SessionToolAllowList).
struct SessionIntentSuppression: Equatable {
    private var agent: Set<UUID> = []
    private var mcp: Set<UUID> = []

    mutating func suppress(_ prompt: IntentPrompt, for id: UUID) {
        switch prompt {
        case .agent: agent.insert(id)
        case .mcp: mcp.insert(id)
        }
    }

    func isSuppressed(_ prompt: IntentPrompt, for id: UUID) -> Bool {
        switch prompt {
        case .agent: return agent.contains(id)
        case .mcp: return mcp.contains(id)
        }
    }
}
