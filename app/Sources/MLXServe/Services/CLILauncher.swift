import Foundation
import SwiftUI

/// Detects third-party CLI coding agents installed on PATH and launches them
/// configured to talk to the local mlx-serve instance.
///
/// Detection uses a login zsh shell so nvm / asdf / pyenv / Homebrew paths are
/// resolved the same way the user's terminal sees them. Results are cached on
/// the main actor and refreshed on demand.
@MainActor
final class CLILauncher: ObservableObject {
    /// What the user has right now.
    @Published private(set) var available: [LauncherCLI] = []
    /// Keep rescanning off the initial launch path until we actually have a result.
    @Published private(set) var hasScanned = false

    private nonisolated static let candidates: [LauncherCLI] = [
        .claudeCode,
        .pi,
        .omp,
        .opencode,
        .codex,
        .hermes,
        .aider,
    ]

    /// Stable id list — pinned against the MAS instructions panel's tabs
    /// (same CLIs, same order) by `CLISetupInstructionsTests`.
    nonisolated static var candidateIds: [String] { candidates.map(\.id) }

    init() {
        Task { await refresh() }
    }

    /// Re-scan PATH. Cheap — one `command -v` sweep in a single shell invocation.
    func refresh() async {
        // The App Store build cannot detect or launch other apps (App Review
        // 2.5.2) — it has no host shell to scan PATH with, either. Stay empty.
        guard BuildFeatures.current.cliLauncher else {
            self.hasScanned = true
            return
        }
        let found = await Self.detectInstalled()
        self.available = found
        self.hasScanned = true
    }

    /// Resolve installed binaries by running a single `command -v` sweep inside
    /// an **interactive** login zsh so user-specific PATH additions (nvm,
    /// Homebrew, ~/.local/bin, ~/.opencode/bin) are honored.
    ///
    /// Both `-i` and `-l` matter: a login shell only sources `.zprofile`/
    /// `.zlogin`, while most users put PATH mutations in `.zshrc` — which is
    /// sourced only by interactive shells. When the app is launched from
    /// Finder/LaunchServices the child process starts with a near-empty
    /// environment, so without `-i` we'd see none of the user's tools.
    ///
    /// Output is keyed (`name=path`) instead of positional so any stray
    /// stdout from `.zshrc` (e.g. `pyenv init`, version managers) can't
    /// misalign parsing.
    private static func detectInstalled() async -> [LauncherCLI] {
        let names = candidates.map { $0.binaryName }
        let script = names.map { "printf '%s=%s\\n' \($0) \"$(command -v \($0) 2>/dev/null)\"" }.joined(separator: "; ")

        let output = await Task.detached { () -> String in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/zsh")
            proc.arguments = ["-i", "-l", "-c", script]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = Pipe()
            do {
                try proc.run()
                proc.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                return String(data: data, encoding: .utf8) ?? ""
            } catch {
                return ""
            }
        }.value

        var resolvedByName: [String: String] = [:]
        for line in output.split(separator: "\n", omittingEmptySubsequences: true) {
            guard let eq = line.firstIndex(of: "=") else { continue }
            let key = String(line[..<eq])
            let value = String(line[line.index(after: eq)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else { continue }
            // Only keep keys we actually asked about — guards against any
            // stray `foo=bar` lines printed from user rc files.
            if names.contains(key) { resolvedByName[key] = value }
        }

        var result: [LauncherCLI] = []
        let home = NSHomeDirectory()
        for cli in candidates {
            var path = resolvedByName[cli.binaryName]
            if path == nil || !FileManager.default.isExecutableFile(atPath: path!) {
                // Not on PATH — a desktop app may bundle it (codex inside
                // ChatGPT.app/Codex.app).
                path = cli.fallbackPaths
                    .map { $0.replacingOccurrences(of: "$HOME", with: home) }
                    .first { FileManager.default.isExecutableFile(atPath: $0) }
            }
            guard let resolved = path,
                  FileManager.default.isExecutableFile(atPath: resolved) else { continue }
            var updated = cli
            updated.resolvedPath = resolved
            result.append(updated)
        }
        return result
    }

    /// Launch a CLI with a folder picker for its working directory.
    func launchWithPicker(_ cli: LauncherCLI, baseURL: String, servedModelId: String,
                          budget: AgentBudget.Budget, entries: [AgentModelEntry]) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true // show the "New Folder" button
        panel.prompt = "Open"
        panel.message = "Select or create a working directory"
        let defaultWS = NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: defaultWS, withIntermediateDirectories: true)
        panel.directoryURL = URL(fileURLWithPath: defaultWS)
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        launch(cli, baseURL: baseURL, servedModelId: servedModelId,
               budget: budget, entries: entries, workingDirectory: url.path)
    }

    /// Write a shell script that sets the right env vars / config for the given
    /// CLI, then hand it to Terminal.app via NSWorkspace.
    func launch(_ cli: LauncherCLI, baseURL: String, servedModelId: String,
                budget: AgentBudget.Budget, entries: [AgentModelEntry],
                workingDirectory: String?) {
        // pi and opencode both need their config files written before launch.
        // The budget travels with them: neither CLI reads `/v1/models` on its
        // own (pi's live list rides the extension we write), so the numbers
        // baked here ARE the contexts they believe the models have. `entries`
        // is the full chat-capable registry — the in-agent /model switch list.
        cli.prepareConfig?(baseURL, servedModelId, budget, entries)

        let cdLine = workingDirectory.map { "cd '\($0)'" } ?? ""
        let script = cli.scriptBody(baseURL, servedModelId, cdLine, budget, entries)
        let fullScript = "#!/bin/zsh -l\n\(script)\n"

        let filename = "mlx-launch-\(cli.id).command"
        let path = NSTemporaryDirectory() + filename
        try? fullScript.write(toFile: path, atomically: true, encoding: String.Encoding.utf8)
        try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path)
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }
}

/// One row in the launcher dropdown. `resolvedPath` is filled in after detection.
struct LauncherCLI: Identifiable, Equatable {
    let id: String
    let displayName: String
    let binaryName: String
    let iconSystemName: String?
    let useClaudeIcon: Bool
    /// Optional side-effect invoked before the terminal script runs (e.g. write
    /// pi's `models.json` into its dedicated `~/.mlx-serve/pi/` config dir).
    /// `entries` = the chat-capable registry snapshot, for configs that bake
    /// the full model list (aider's metadata file).
    let prepareConfig: (@Sendable (_ baseURL: String, _ servedModelId: String,
                                  _ budget: AgentBudget.Budget,
                                  _ entries: [AgentModelEntry]) -> Void)?
    /// Absolute paths probed when the binary is NOT on PATH — for CLIs a
    /// desktop app bundles (codex inside ChatGPT.app/Codex.app). `~` is not
    /// expanded here; entries may start with `$HOME`, expanded at probe time.
    var fallbackPaths: [String] = []
    /// Shell body that sets env vars and execs the CLI. Does NOT include the
    /// shebang. `entries` = the chat-capable registry snapshot (opencode bakes
    /// it into its inline config; pi/Claude Code ignore it — pi's list is
    /// fetched live by its extension, Claude Code has no per-model config).
    let scriptBody: (_ baseURL: String, _ servedModelId: String, _ cdLine: String,
                     _ budget: AgentBudget.Budget, _ entries: [AgentModelEntry]) -> String
    var resolvedPath: String = ""

    static func == (lhs: LauncherCLI, rhs: LauncherCLI) -> Bool { lhs.id == rhs.id }
}

extension LauncherCLI {

    /// Claude Code — Anthropic Messages API route. Uses env vars so the CLI
    /// talks to our `/v1/messages` endpoint with no code changes upstream.
    static let claudeCode = LauncherCLI(
        id: "claude",
        displayName: "Claude Code",
        binaryName: "claude",
        iconSystemName: nil,
        useClaudeIcon: true,
        prepareConfig: nil,
        scriptBody: { baseURL, model, cdLine, budget, _ in
            """
            \(AgentConfigs.claudeCodeExports(baseURL: baseURL, model: model, budget: budget))
            \(cdLine)
            claude --model \(model)
            """
        }
    )

    /// pi (https://github.com/earendil-works/pi, pi.dev) — OpenAI-compatible.
    /// Needs a `models.json` naming our provider; ours lives in a dedicated
    /// config dir selected via `PI_CODING_AGENT_DIR`.
    static let pi = LauncherCLI(
        id: "pi",
        displayName: "pi",
        binaryName: "pi",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        // A dedicated config dir (via PI_CODING_AGENT_DIR) rather than the real
        // ~/.pi/agent: writing models.json there would DESTROY any providers
        // the user already configured. Same isolation move as OPENCODE_CONFIG,
        // and the same dir the MAS instructions panel tells the user to create
        // (CLISetupInstructionsTests pins the two against each other).
        prepareConfig: { baseURL, model, budget, _ in
            let dir = NSString(string: "~/.mlx-serve/pi").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let config = AgentConfigs.piModelsJSON(baseURL: baseURL, model: model, budget: budget)
            let path = (dir as NSString).appendingPathComponent("models.json")
            try? config.write(toFile: path, atomically: true, encoding: .utf8)
            // Global context file — pi injects it into every session's system
            // prompt (same builder the sandbox registry materializes in-guest).
            try? AgentConfigs.piAgentsMD(budget: budget)
                .write(toFile: (dir as NSString).appendingPathComponent("AGENTS.md"),
                       atomically: true, encoding: .utf8)
            // Live /model list: pi auto-discovers <agentDir>/extensions/*.js;
            // this one fetches /v1/models at session start and registers every
            // chat-capable model (LAN @peer entries included) on the `mlx`
            // provider. models.json above stays the offline fallback.
            let extDir = (dir as NSString).appendingPathComponent("extensions")
            try? FileManager.default.createDirectory(atPath: extDir, withIntermediateDirectories: true)
            try? AgentConfigs.piModelsExtensionJS(baseURL: baseURL)
                .write(toFile: (extDir as NSString).appendingPathComponent("mlx-models.js"),
                       atomically: true, encoding: .utf8)
        },
        scriptBody: { _, model, cdLine, _, _ in
            """
            export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi"
            \(cdLine)
            pi --provider mlx --model \(model)
            """
        }
    )

    /// oh-my-pi (https://github.com/can1357/oh-my-pi) — pi fork with its own
    /// config tree: models.yml (YAML) under the agent dir. Dedicated dir,
    /// never the user's real ~/.omp/agent — the same non-clobber move as pi.
    /// The dir env var is still pi's spelling (PI_CODING_AGENT_DIR; measured
    /// on omp v17 — the OMP_ rename reached only its help text), so the
    /// script exports both spellings.
    static let omp = LauncherCLI(
        id: "omp",
        displayName: "oh-my-pi",
        binaryName: "omp",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget, entries in
            let dir = NSString(string: "~/.mlx-serve/omp").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? AgentConfigs.ompModelsYML(baseURL: baseURL, defaultModel: model, entries: entries)
                .write(toFile: (dir as NSString).appendingPathComponent("models.yml"),
                       atomically: true, encoding: .utf8)
        },
        scriptBody: { _, model, cdLine, _, _ in
            """
            export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
            export OMP_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
            \(cdLine)
            omp --model mlx/\(model)
            """
        }
    )

    /// codex (https://github.com/openai/codex) — Responses-wire only; config
    /// rides a dedicated CODEX_HOME (the dir must exist before codex runs).
    /// The script resolves the binary itself (PATH, then the desktop app's
    /// bundled CLI) so a ChatGPT.app-only install still launches.
    static let codex = LauncherCLI(
        id: "codex",
        displayName: "Codex",
        binaryName: "codex",
        iconSystemName: "chevron.left.forwardslash.chevron.right",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget, _ in
            let dir = NSString(string: "~/.mlx-serve/codex").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? AgentConfigs.codexConfigTOML(baseURL: baseURL, model: model, budget: budget)
                .write(toFile: (dir as NSString).appendingPathComponent("config.toml"),
                       atomically: true, encoding: .utf8)
        },
        fallbackPaths: [
            "/Applications/ChatGPT.app/Contents/Resources/codex",
            "/Applications/Codex.app/Contents/Resources/codex",
            "$HOME/Applications/ChatGPT.app/Contents/Resources/codex",
            "$HOME/Applications/Codex.app/Contents/Resources/codex",
        ],
        scriptBody: { _, _, cdLine, _, _ in
            """
            export CODEX_HOME="$HOME/.mlx-serve/codex"
            \(AgentConfigs.codexBinResolver)
            \(cdLine)
            "$CODEX_BIN"
            """
        }
    )

    /// hermes (Nous Research) — whole config tree rides HERMES_HOME
    /// (hermes_constants.py), so the sandbox's config.yaml + .env pair lands
    /// in a dedicated host dir instead of the user's real ~/.hermes.
    static let hermes = LauncherCLI(
        id: "hermes",
        displayName: "hermes",
        binaryName: "hermes",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget, _ in
            let dir = NSString(string: "~/.mlx-serve/hermes").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? AgentConfigs.hermesConfigYAML(baseURL: baseURL, apiKey: "mlx-serve",
                                               model: model, budget: budget, entries: [])
                .write(toFile: (dir as NSString).appendingPathComponent("config.yaml"),
                       atomically: true, encoding: .utf8)
            try? AgentConfigs.hermesEnvFile(baseURL: baseURL)
                .write(toFile: (dir as NSString).appendingPathComponent(".env"),
                       atomically: true, encoding: .utf8)
        },
        scriptBody: { _, _, cdLine, _, _ in
            """
            export HERMES_HOME="$HOME/.mlx-serve/hermes"
            \(cdLine)
            hermes
            """
        }
    )

    /// aider (https://aider.chat) — pure env vars plus a litellm metadata
    /// file carrying the real per-model context windows.
    static let aider = LauncherCLI(
        id: "aider",
        displayName: "aider",
        binaryName: "aider",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model, budget, entries in
            let dir = NSString(string: "~/.mlx-serve/aider").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            try? AgentConfigs.aiderModelMetadataJSON(model: model, budget: budget, entries: entries)
                .write(toFile: (dir as NSString).appendingPathComponent("model-metadata.json"),
                       atomically: true, encoding: .utf8)
        },
        scriptBody: { baseURL, model, cdLine, _, _ in
            """
            export OPENAI_API_BASE='\(baseURL)/v1'
            export OPENAI_API_KEY=mlx-serve
            \(cdLine)
            aider --model openai/\(model) --weak-model openai/\(model) --model-metadata-file ~/.mlx-serve/aider/model-metadata.json
            """
        }
    )

    /// opencode (https://opencode.ai) — registers a custom OpenAI-compatible
    /// provider via the inline `OPENCODE_CONFIG_CONTENT` env var, which MERGES
    /// over the user's global/project config (their settings and plugins keep
    /// working) and needs no file writes at all. Same block as the MAS
    /// instructions panel (CLISetupInstructionsTests pins the two together).
    static let opencode = LauncherCLI(
        id: "opencode",
        displayName: "OpenCode",
        binaryName: "opencode",
        iconSystemName: "chevron.left.forwardslash.chevron.right",
        useClaudeIcon: false,
        prepareConfig: nil,
        scriptBody: { baseURL, model, cdLine, budget, entries in
            // Full chat-capable list — opencode's in-session /models picker
            // shows exactly what this config declares. The served model is
            // force-included (with ITS budget) so `--model mlx/<id>` always
            // resolves, even on an empty registry snapshot.
            var list = entries
            if !list.contains(where: { $0.id == model }) {
                list.insert(AgentModelEntry(id: model, budget: budget, vision: false), at: 0)
            }
            return """
            export OPENCODE_CONFIG_CONTENT='\(AgentConfigs.opencodeJSON(baseURL: baseURL, defaultModel: model, entries: list))'
            \(cdLine)
            opencode --model mlx/\(model)
            """
        }
    )
}

// MARK: - UI

/// Launcher button for the menu bar: one `Menu` with the detected host CLIs
/// (launched in Terminal.app against the local server) plus the sandboxed
/// agents (pi/hermes INSIDE the guest VM — routed to the Sandbox window via
/// `openSandboxAgent`). The sandbox rows are always present, so the button no
/// longer hides when no host CLI is installed — running an agent needs
/// nothing on the host anymore.
@MainActor
struct CLILauncherButton: View {
    let baseURL: String
    let servedModelId: String
    /// The running server's EFFECTIVE context (`/v1/models` meta.context_length).
    /// pi and opencode budget their own `max_tokens` against the number we write
    /// into their config, so passing nil here silently caps every agent session
    /// at the conservative fallback. See `AgentBudget`.
    let serverContextLength: Int?
    /// Full registry snapshot (`ServerManager.allModels`) — the chat-capable
    /// subset becomes each CLI's in-agent /model switch list.
    let models: [ModelInfo]
    let isEnabled: Bool
    /// Tray → Sandbox window hand-off (agent id): the tray can't drive the
    /// window's state directly, so this posts the launch request and opens
    /// the window; the window focuses a running session or starts one.
    let openSandboxAgent: (String) -> Void

    @StateObject private var detector = CLILauncher()

    var body: some View {
        Group {
            if !detector.hasScanned {
                // Still scanning — reserve the space with a placeholder so the
                // footer doesn't reflow when scan finishes a moment later.
                Color.clear.frame(width: 0, height: 0)
            } else {
                Menu {
                    CLILauncherMenuItems(detector: detector, baseURL: baseURL,
                                         servedModelId: servedModelId,
                                         serverContextLength: serverContextLength,
                                         models: models,
                                         openSandboxAgent: openSandboxAgent)
                } label: {
                    HStack(spacing: TrayFooterMetrics.iconSpacing) {
                        Image(systemName: "terminal")
                        Text("Code")
                    }
                    .frame(maxWidth: .infinity)
                }
                // Standard bordered-button chrome so the menu is visually
                // identical to its sibling Chat/Tasks buttons — the previous
                // hand-rolled stroke + material background rendered as an
                // odd-one-out outlined pill in the tray footer.
                .menuStyle(.button)
                .buttonStyle(.bordered)
                .menuIndicator(.hidden)
                .disabled(!isEnabled)
                .help("Launch a coding agent — on this Mac (\(detector.available.isEmpty ? "none detected" : detector.available.map(\.displayName).joined(separator: ", "))) or inside the sandbox (pi, hermes)")
            }
        }
        .task { await detector.refresh() }
    }
}

/// The launcher dropdown's BODY — the detected host CLIs plus the sandboxed
/// agents. Shared by the tray's Code button and the chat empty state's Code
/// Launcher chip: one list, so the two surfaces cannot drift (a CLI the tray
/// gains that the chip never offers is the silent-hole class).
@MainActor
struct CLILauncherMenuItems: View {
    @ObservedObject var detector: CLILauncher
    let baseURL: String
    let servedModelId: String
    let serverContextLength: Int?
    let models: [ModelInfo]
    let openSandboxAgent: (String) -> Void

    private var budget: AgentBudget.Budget { AgentBudget.forServerContext(serverContextLength) }
    private var entries: [AgentModelEntry] { AgentModelEntry.chatEntries(from: models) }

    var body: some View {
        if !detector.available.isEmpty {
            Section("On this Mac") {
                ForEach(detector.available) { cli in
                    Button {
                        detector.launchWithPicker(cli, baseURL: baseURL, servedModelId: servedModelId,
                                                  budget: budget, entries: entries)
                    } label: {
                        Label(cli.displayName, systemImage: cli.iconSystemName ?? "terminal")
                    }
                }
            }
        }
        Section("In the sandbox") {
            ForEach(SandboxAgentRegistry.all) { spec in
                Button {
                    openSandboxAgent(spec.id)
                } label: {
                    Label("\(spec.displayName) in Sandbox", systemImage: "shippingbox")
                }
            }
        }
    }
}
