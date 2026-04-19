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

    private static let candidates: [LauncherCLI] = [
        .claudeCode,
        .pi,
        .opencode,
    ]

    init() {
        Task { await refresh() }
    }

    /// Re-scan PATH. Cheap — three `which` calls in a single shell invocation.
    func refresh() async {
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
        for cli in candidates {
            guard let path = resolvedByName[cli.binaryName],
                  FileManager.default.isExecutableFile(atPath: path) else { continue }
            var updated = cli
            updated.resolvedPath = path
            result.append(updated)
        }
        return result
    }

    /// Launch a CLI with a folder picker for its working directory.
    func launchWithPicker(_ cli: LauncherCLI, baseURL: String, servedModelId: String) {
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
        guard panel.runModal() == .OK, let url = panel.url else { return }
        launch(cli, baseURL: baseURL, servedModelId: servedModelId, workingDirectory: url.path)
    }

    /// Write a shell script that sets the right env vars / config for the given
    /// CLI, then hand it to Terminal.app via NSWorkspace.
    func launch(_ cli: LauncherCLI, baseURL: String, servedModelId: String, workingDirectory: String?) {
        // pi and opencode both need their config files written before launch.
        cli.prepareConfig?(baseURL, servedModelId)

        let cdLine = workingDirectory.map { "cd '\($0)'" } ?? ""
        let script = cli.scriptBody(baseURL, servedModelId, cdLine)
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
    /// `~/.pi/agent/models.json`).
    let prepareConfig: (@Sendable (_ baseURL: String, _ servedModelId: String) -> Void)?
    /// Shell body that sets env vars and execs the CLI. Does NOT include the shebang.
    let scriptBody: (_ baseURL: String, _ servedModelId: String, _ cdLine: String) -> String
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
        scriptBody: { baseURL, model, cdLine in
            """
            export ANTHROPIC_BASE_URL='\(baseURL)'
            export ANTHROPIC_API_KEY=
            export ANTHROPIC_AUTH_TOKEN=mlx-serve
            export CLAUDE_CODE_ATTRIBUTION_HEADER=0
            export ANTHROPIC_DEFAULT_OPUS_MODEL=\(model)
            export ANTHROPIC_DEFAULT_SONNET_MODEL=\(model)
            export ANTHROPIC_DEFAULT_HAIKU_MODEL=\(model)
            export CLAUDE_CODE_SUBAGENT_MODEL=\(model)
            \(cdLine)
            claude --model \(model)
            """
        }
    )

    /// pi (https://github.com/badlogic/pi-mono) — OpenAI-compatible. Needs a
    /// `~/.pi/agent/models.json` entry naming our provider.
    static let pi = LauncherCLI(
        id: "pi",
        displayName: "pi",
        binaryName: "pi",
        iconSystemName: "terminal",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model in
            let dir = NSString(string: "~/.pi/agent").expandingTildeInPath
            try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
            let config = """
            {
              "providers": {
                "mlx": {
                  "baseUrl": "\(baseURL)/v1",
                  "api": "openai-completions",
                  "apiKey": "mlx-serve",
                  "compat": {
                    "supportsDeveloperRole": false,
                    "supportsReasoningEffort": false,
                    "maxTokensField": "max_tokens",
                    "thinkingFormat": "qwen"
                  },
                  "models": [
                    {"id": "\(model)", "name": "mlx-\(model)", "input": ["text"],
                     "contextWindow": 32768, "maxTokens": 8192, "reasoning": true}
                  ]
                }
              }
            }
            """
            let path = (dir as NSString).appendingPathComponent("models.json")
            try? config.write(toFile: path, atomically: true, encoding: .utf8)
        },
        scriptBody: { _, model, cdLine in
            """
            \(cdLine)
            pi --provider mlx --model \(model)
            """
        }
    )

    /// opencode (https://opencode.ai) — registers a custom OpenAI-compatible
    /// provider via a dedicated `OPENCODE_CONFIG` file so the user's main
    /// `~/.config/opencode/opencode.json` is left untouched.
    static let opencode = LauncherCLI(
        id: "opencode",
        displayName: "OpenCode",
        binaryName: "opencode",
        iconSystemName: "chevron.left.forwardslash.chevron.right",
        useClaudeIcon: false,
        prepareConfig: { baseURL, model in
            let config = """
            {
              "$schema": "https://opencode.ai/config.json",
              "provider": {
                "mlx": {
                  "npm": "@ai-sdk/openai-compatible",
                  "name": "MLX Serve (local)",
                  "options": { "baseURL": "\(baseURL)/v1" },
                  "models": {
                    "\(model)": { "name": "\(model) (mlx-serve)" }
                  }
                }
              }
            }
            """
            let path = NSTemporaryDirectory() + "mlx-opencode-config.json"
            try? config.write(toFile: path, atomically: true, encoding: String.Encoding.utf8)
        },
        scriptBody: { _, model, cdLine in
            let configPath = NSTemporaryDirectory() + "mlx-opencode-config.json"
            return """
            export OPENCODE_CONFIG='\(configPath)'
            \(cdLine)
            opencode --model mlx/\(model)
            """
        }
    )
}

// MARK: - UI

/// Launcher button for the menu bar. Renders either:
///   * nothing (when we've scanned and found no CLIs installed — the user
///     doesn't care about a feature they can't use),
///   * a single bordered button for one CLI (skips the extra click),
///   * a `Menu` dropdown for 2+ CLIs.
@MainActor
struct CLILauncherButton: View {
    let baseURL: String
    let servedModelId: String
    let isEnabled: Bool

    @StateObject private var detector = CLILauncher()

    var body: some View {
        Group {
            if !detector.hasScanned {
                // Still scanning — reserve the space with a placeholder so the
                // footer doesn't reflow when scan finishes a moment later.
                Color.clear.frame(width: 0, height: 0)
            } else if detector.available.isEmpty {
                // None installed — hide entirely.
                EmptyView()
            } else if detector.available.count == 1, let only = detector.available.first {
                Button {
                    detector.launchWithPicker(only, baseURL: baseURL, servedModelId: servedModelId)
                } label: {
                    label(for: only)
                }
                .buttonStyle(.bordered)
                .disabled(!isEnabled)
                .help("Launch \(only.displayName)")
            } else {
                Menu {
                    ForEach(detector.available) { cli in
                        Button {
                            detector.launchWithPicker(cli, baseURL: baseURL, servedModelId: servedModelId)
                        } label: {
                            Label(cli.displayName, systemImage: cli.iconSystemName ?? "terminal")
                        }
                    }
                } label: {
                    Image(systemName: "terminal")
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(
                    RoundedRectangle(cornerRadius: 5)
                        .strokeBorder(.tertiary, lineWidth: 0.5)
                        .background(RoundedRectangle(cornerRadius: 5).fill(.regularMaterial))
                )
                .disabled(!isEnabled)
                .help("Launch coding agent (\(detector.available.map(\.displayName).joined(separator: ", ")))")
            }
        }
        .task { await detector.refresh() }
    }

    @ViewBuilder
    private func label(for cli: LauncherCLI) -> some View {
        if cli.useClaudeIcon {
            ClaudeIcon(size: 12).foregroundStyle(.white)
        } else if let name = cli.iconSystemName {
            Image(systemName: name)
        } else {
            Text(cli.displayName)
        }
    }
}
