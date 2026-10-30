import Foundation

/// Copy-and-paste instructions for a coding agent (pi, Claude Code, Codex...) to install a community plugin the
/// same safe way the app does: from source, with the repo's pinned Zig, after reading what it builds.
/// Pure text, so it is testable and the wording lives in one place.
enum PluginAgentInstructions {
    /// For one catalog entry.
    static func text(for entry: PluginCatalog.Entry, binaryPath: String, runningVersion: String?) -> String {
        body(plugin: entry.name, header: header(for: entry), binaryPath: binaryPath, runningVersion: runningVersion,
             requirement: entry.requiresMlxServe)
    }

    /// For any plugin: the agent picks one from the repo's `plugins.json`.
    static func generic(binaryPath: String, runningVersion: String?) -> String {
        body(plugin: "<name>",
             header: "Install a plugin for mlx-serve on this Mac. The list of plugins is plugins.json in the repo below; if I did not name one, show it to me and ask which.",
             binaryPath: binaryPath, runningVersion: runningVersion, requirement: "")
    }

    private static func header(for e: PluginCatalog.Entry) -> String {
        var lines = ["Install the \"\(e.name)\" plugin (\(e.kind) \(e.version)) for mlx-serve on this Mac."]
        if !e.summary.isEmpty { lines.append(e.summary) }
        if !e.models.isEmpty { lines.append("It serves: " + e.models.joined(separator: "; ") + ".") }
        return lines.joined(separator: "\n")
    }

    private static func body(plugin: String, header: String, binaryPath: String, runningVersion: String?, requirement: String) -> String {
        let dir = (binaryPath as NSString).deletingLastPathComponent
        let onPath = binaryPath.hasPrefix("/")
            ? "mlx-serve must be on PATH for the scripts. Mine is \(binaryPath); run `export PATH=\"\(dir):$PATH\"` if `mlx-serve --version` is not found."
            : "mlx-serve must be on PATH for the scripts (`mlx-serve --version` must work)."
        var version = "My mlx-serve is \(runningVersion ?? "unknown")."
        if !requirement.isEmpty { version += " This plugin needs \(requirement)." }
        return """
        \(header)

        Source: \(PluginCatalog.repoURL) (branch \(PluginCatalog.ref)). Rules: compile from source, never download or run a prebuilt dylib, use only the Zig that the repo's scripts/setup.sh stages (no system Zig), and do not edit .zig-toolchain/ or .kit/.

        \(version) \(onPath)

        Steps:
        1. Clone: `git clone --depth 1 \(PluginCatalog.repoURL) ~/.mlx-serve/plugin-src` (if that folder exists, `git -C ~/.mlx-serve/plugin-src pull --ff-only` instead). Work in it.
        2. Read plugins/\(plugin)/plugin.json, plugins/\(plugin)/README.md and the plugin's *.zig and build.zig. A plugin is native code that runs with my rights: tell me in a few lines what it does, and stop and ask me before building if it touches the network, files outside ~/.mlx-serve or the model folders, or starts other processes.
        3. `scripts/setup.sh` (stages the pinned Zig and kit; the first run downloads them).
        4. `scripts/install.sh \(plugin) --yes` (runs the plugin's tests, compiles it, then runs `mlx-serve plugin install`). To try it first without touching my setup, use `scripts/install.sh \(plugin) --home ~/tmp/h` and serve with `HOME=~/tmp/h mlx-serve --model <pack dir>`.
        5. Plugins load only when ~/.mlx-serve/plugins.json has "enabled": true and "disabled" does not list \(plugin). Edit it keeping any other keys (or switch it on in the app: Models > Plugins).
        6. Restart mlx-serve (quit and reopen the app, or restart the server). Confirm the server log has `[plugin] \(plugin) serves <model_type>` and that /v1/models shows "plugin": "\(plugin)" for a pack it serves.

        If a step fails, show me the error and stop. "built against another MLX" means the plugin kit and my mlx-serve differ: do not work around it. Do not run two servers that each load a large model at once.
        """
    }
}
