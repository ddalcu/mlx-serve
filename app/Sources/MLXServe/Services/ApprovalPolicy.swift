import Foundation

/// What to do with a tool call during an unattended run.
enum ApprovalDecision: Equatable {
    case allow
    case deny(reason: String)
    case ask(reason: String)
}

/// Pure decision layer for unattended task runs. Given a task's autonomy level and
/// a tool call, decides whether to auto-allow, auto-deny, or pause and ask the user.
///
/// This is the *approval* layer only. Hard filesystem confinement is independently
/// enforced by `ToolExecutor.resolveAndConfine` — a buggy `allow` here still can't
/// write outside the workspace. Yolo is unrestricted HERE (auto-allow everything)
/// but no longer unconfined THERE: since 2026-07-20 every run has a mandatory
/// working directory (yolo anchors at the default agent workspace —
/// `TaskScheduler.workDir`), and file tools refuse to run without one.
///
/// Kept `nonisolated`/pure so the whole autonomy matrix is unit-testable without a
/// running server, model, or main actor.
enum ApprovalPolicy {

    /// Tools that only read/observe — safe to auto-allow at every autonomy level.
    static let readOnlyTools: Set<String> = [
        "readFile", "searchFiles", "listFiles", "browse", "webSearch", "cwd",
    ]

    /// Tools that write to a path argument and can therefore be confined to a folder.
    static let pathWriteTools: Set<String> = ["writeFile", "editFile"]

    /// MCP tools are namespaced `<server>__<tool>`. They reach external services and
    /// can't be path-confined, so they're treated like `shell`: allowed under
    /// fullAuto/yolo, but they pause for approval at the lower levels.
    static func isMCPTool(_ name: String) -> Bool { name.contains("__") }

    static func decide(tool: String,
                       autonomy: TaskAutonomy,
                       arguments: [String: String],
                       rawArguments: String,
                       workingDirectory: String?) -> ApprovalDecision {
        switch autonomy {
        case .yolo:
            // Unrestricted by definition.
            return .allow

        case .readOnly:
            return readOnlyTools.contains(tool)
                ? .allow
                : .ask(reason: "Read-only task — “\(tool)” can change things, so it needs your OK.")

        case .workspace:
            if readOnlyTools.contains(tool) { return .allow }
            if pathWriteTools.contains(tool) {
                return isConfined(arguments["path"], to: workingDirectory)
                    ? .allow
                    : .ask(reason: "“\(tool)” would write outside the task's folder.")
            }
            // shell / saveMemory / unknown can act outside the folder — confirm.
            return .ask(reason: "“\(tool)” can act outside the task's folder, so it needs your OK.")

        case .fullAuto:
            // `createTask` only schedules a background run (itself governed by its
            // own autonomy when it executes) — benign like `saveMemory`, and the
            // whole point of unattended/Telegram agents, so it auto-allows here.
            // `computer` drives the sandbox desktop: not read-only, not
            // path-confinable — the shell bucket (issue: it must never fall
            // into the "Unrecognized tool" ask under fullAuto).
            if readOnlyTools.contains(tool) || tool == "shell" || tool == "saveMemory"
                || tool == "createTask" || tool == "computer" || isMCPTool(tool) {
                return .allow
            }
            if pathWriteTools.contains(tool) {
                return isConfined(arguments["path"], to: workingDirectory)
                    ? .allow
                    : .ask(reason: "“\(tool)” would write outside the task's folder.")
            }
            return .ask(reason: "Unrecognized tool “\(tool)” — needs your OK.")
        }
    }

    /// True if `path` resolves inside `workingDirectory`. Mirrors the containment
    /// check in `ToolExecutor.resolveAndConfine` (kept independent so this stays a
    /// pure function). A nil path means "no path-based escape" (allow); a nil
    /// working directory means we can't prove confinement (deny confinement).
    static func isConfined(_ path: String?, to workingDirectory: String?) -> Bool {
        guard let path, !path.isEmpty else { return true }
        guard let wd = workingDirectory else { return false }

        let resolved: String
        if path.hasPrefix("/") || path.hasPrefix("~") {
            resolved = (path as NSString).expandingTildeInPath
        } else {
            resolved = (wd as NSString).appendingPathComponent(path)
        }
        let normalized = (resolved as NSString).standardizingPath
        let normalizedWd = (wd as NSString).standardizingPath
        return normalized == normalizedWd || normalized.hasPrefix(normalizedWd + "/")
    }
}
