import Foundation

/// One config file materialized HOST-SIDE into the guest rootfs before a CLI
/// session starts (virtiofs makes it visible in-guest immediately, booted or
/// not). Contents carry the literal `__MLX_HOST__` placeholder — the guest's
/// address for the host (the NAT gateway) is only knowable in-guest, so the
/// bootstrap script resolves and substitutes it at run time.
struct SandboxAgentFile: Equatable {
    let guestPath: String
    let content: String
}

/// One agent CLI the sandbox can host in the embedded terminal (issue #89
/// follow-up). Mirrors `LauncherCLI`'s registry shape: adding an agent is one
/// more row, not a new flow.
struct SandboxAgentSpec: Identifiable, Equatable {
    let id: String
    let displayName: String
    /// Probed in-guest by the bootstrap; the install runs only when absent
    /// (the rootfs is writable + persistent, so one install survives reboots).
    let binaryName: String
    /// Pinned install command, run INSIDE the guest on first use. Progress
    /// streams straight into the terminal — no app-side progress UI.
    let installScript: String
    /// Config files for a session against the local server. `apiKey` nil =
    /// no `--api-key` configured (guest→host arrives NON-loopback, so a real
    /// key must travel when one is set). `entries` is the chat-capable
    /// registry snapshot — the in-agent /model switch list (pi ignores it:
    /// its extension fetches the list live, in-guest).
    let configFiles: (_ model: String, _ serverPort: UInt16,
                      _ budget: AgentBudget.Budget, _ apiKey: String?,
                      _ entries: [AgentModelEntry]) -> [SandboxAgentFile]
    /// The line the bootstrap `exec`s once configs are patched + the CLI exists.
    let launchCommand: (_ model: String) -> String

    static func == (lhs: SandboxAgentSpec, rhs: SandboxAgentSpec) -> Bool { lhs.id == rhs.id }
}

extension SandboxAgentSpec {

    /// pi (pi.dev) — OpenAI-compatible via a models.json. Same generator as
    /// the host launcher + MAS instructions (`AgentConfigs.piModelsJSON`), so
    /// the three surfaces can never drift apart. In the guest, HOME=/root and
    /// the whole VM is ours — pi reads its real `~/.pi/agent`, no
    /// `PI_CODING_AGENT_DIR` isolation dance needed.
    static let pi = SandboxAgentSpec(
        id: "pi",
        displayName: "pi",
        binaryName: "pi",
        installScript: "npm i -g @earendil-works/pi-coding-agent@0.80.10",
        configFiles: { model, serverPort, budget, apiKey, _ in
            let base = "http://\(SandboxAgentRegistry.hostPlaceholder):\(serverPort)"
            let key = apiKey ?? "mlx-serve"
            return [SandboxAgentFile(
                guestPath: "/root/.pi/agent/models.json",
                content: AgentConfigs.piModelsJSON(
                    baseURL: base, model: model, budget: budget, apiKey: key)),
             SandboxAgentFile(
                guestPath: "/root/.pi/agent/AGENTS.md",
                content: AgentConfigs.piAgentsMD(budget: budget)),
             // Live /model list: pi auto-discovers <agentDir>/extensions;
             // the extension fetches the guest's OWN view of /v1/models
             // (keyless LAN-share mode filters it to shared models — exactly
             // what the guest may use). Content is entry-independent, so the
             // bootstrap's dummy-args path enumeration always includes it.
             SandboxAgentFile(
                guestPath: "/root/.pi/agent/extensions/mlx-models.js",
                content: AgentConfigs.piModelsExtensionJS(baseURL: base, apiKey: key))]
        },
        launchCommand: { model in "pi --provider mlx --model \(VzGuest.shellQuote(model))" }
    )

    /// hermes (Nous Research). The config mirrors EXACTLY what `hermes
    /// setup`'s custom-endpoint flow saves (verified against hermes_cli
    /// source: `_model_flow_custom` + `_save_custom_provider` +
    /// `_save_model_choice`): model name under `default` (the docs' `model:`
    /// key is wrong — it left "Active provider: none" and re-ran the wizard),
    /// `api_mode: chat_completions` (our server IS /chat/completions — no
    /// heuristics), and context_length in the `custom_providers` entry.
    ///
    /// The `.env` is the wizard KILL SWITCH: `_has_any_provider_configured()`
    /// gates the first-run setup wizard, and `OPENAI_BASE_URL` in
    /// `~/.hermes/.env` alone satisfies it — without this file every fresh
    /// session opened into "How would you like to set up Hermes?".
    static let hermes = SandboxAgentSpec(
        id: "hermes",
        displayName: "hermes",
        binaryName: "hermes",
        installScript: "curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash "
            + "|| pip install hermes-agent",
        configFiles: { model, serverPort, budget, apiKey, entries in
            let base = "http://\(SandboxAgentRegistry.hostPlaceholder):\(serverPort)"
            let key = apiKey ?? "mlx-serve"
            return [
                SandboxAgentFile(
                    guestPath: "/root/.hermes/config.yaml",
                    content: AgentConfigs.hermesConfigYAML(
                        baseURL: base, apiKey: key, model: model,
                        budget: budget, entries: entries)),
                SandboxAgentFile(
                    guestPath: "/root/.hermes/.env",
                    content: AgentConfigs.hermesEnvFile(baseURL: base, apiKey: key)),
            ]
        },
        launchCommand: { _ in "hermes" }
    )
}

/// Registry + the per-session bootstrap builder. The bootstrap is what the
/// embedded terminal's ssh session actually runs: resolve the host's address,
/// patch the configs, install on first use, hand the tty to the agent.
enum SandboxAgentRegistry {

    /// Literal placeholder written into config files host-side; the bootstrap
    /// seds it to the NAT gateway in-guest.
    static let hostPlaceholder = "__MLX_HOST__"

    static let all: [SandboxAgentSpec] = [.pi, .hermes]

    static let pi = SandboxAgentSpec.pi
    static let hermes = SandboxAgentSpec.hermes

    /// Guest path of the per-agent bootstrap (same rootfs-injection precedent
    /// as `/.vz-init`).
    static func bootstrapPath(agentId: String) -> String { "/.vz-bootstrap-\(agentId)" }

    /// The session bootstrap. Runs under `ssh -t` as a plain (non-login)
    /// command, so it exports its own PATH rather than relying on `.profile`.
    static func bootstrapScript(for spec: SandboxAgentSpec, model: String) -> String {
        // Dummy args — only the guest PATHS matter here, and every spec's
        // path set is argument-independent (pinned by the registry tests).
        let configPaths = spec.configFiles(model, 0, AgentBudget.fallback, nil, [])
            .map { VzGuest.shellQuote($0.guestPath) }
            .joined(separator: " ")
        return """
        #!/bin/sh
        # mlx-serve sandbox bootstrap for \(spec.displayName) — regenerated at each session start
        export PATH="$PATH:/root/.local/bin"
        GW=$(ip route 2>/dev/null | awk '/^default/{print $3; exit}')
        if [ -z "$GW" ]; then
          echo "mlx-serve: cannot resolve the host gateway (is guest networking on?)" >&2
          exit 1
        fi
        for f in \(configPaths); do
          [ -f "$f" ] && sed -i "s/__MLX_HOST__/$GW/g" "$f"
        done
        if ! command -v \(spec.binaryName) >/dev/null 2>&1; then
          echo "Installing \(spec.displayName) (first run — output streams below)…"
          \(spec.installScript) || { echo "mlx-serve: \(spec.displayName) install failed" >&2; exit 1; }
        fi
        cd /workspace 2>/dev/null || echo "mlx-serve: /workspace share missing — starting in $HOME" >&2
        exec \(spec.launchCommand(model))
        """
    }

    /// Write a session's config files + bootstrap into the rootfs dir
    /// HOST-SIDE (virtiofs makes them visible in-guest immediately, so this
    /// works on an already-booted guest too). Returns the bootstrap's GUEST
    /// path — what the ssh session runs. Rewritten at each session start.
    @discardableResult
    static func materialize(spec: SandboxAgentSpec, model: String, serverPort: UInt16,
                            budget: AgentBudget.Budget, apiKey: String?,
                            entries: [AgentModelEntry], rootfsDir: String) throws -> String {
        let fm = FileManager.default
        for file in spec.configFiles(model, serverPort, budget, apiKey, entries) {
            let host = rootfsDir + file.guestPath
            try fm.createDirectory(atPath: (host as NSString).deletingLastPathComponent,
                                   withIntermediateDirectories: true)
            try file.content.write(toFile: host, atomically: true, encoding: .utf8)
        }
        let bootstrap = bootstrapPath(agentId: spec.id)
        let bootstrapHost = rootfsDir + bootstrap
        try bootstrapScript(for: spec, model: model)
            .write(toFile: bootstrapHost, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: bootstrapHost)
        return bootstrap
    }
}

/// Blocking gates checked BEFORE a session opens — each one an alert with the
/// fix named, never a silent failure or a hung terminal.
enum SandboxCliPreflight {

    /// The guest reaches the host at the NAT gateway address, which is
    /// non-loopback traffic to the server — a loopback-bound server is
    /// unreachable from inside the VM.
    static func serverBindReachableFromGuest(host: String) -> Bool {
        let h = host.trimmingCharacters(in: .whitespacesAndNewlines)
        return h.isEmpty || h == "0.0.0.0" || h == "::"
    }

    static func issues(sandboxEnabled: Bool, networkOn: Bool,
                       serverRunning: Bool, serverHost: String) -> [String] {
        var out: [String] = []
        if !sandboxEnabled {
            out.append("the Agent Sandbox is off — turn it on in Settings → Agent Sandbox")
        }
        if !networkOn {
            out.append("guest networking is off — ssh, first-run installs, and reaching the model all need it (Settings → Agent Sandbox → Network)")
        }
        if !serverRunning {
            out.append("the server isn't running — load a model first; the sandboxed agent talks to it")
        } else if !serverBindReachableFromGuest(host: serverHost) {
            out.append("the server is bound to \(serverHost.trimmingCharacters(in: .whitespacesAndNewlines)) — the guest can only reach it when the server listens on 0.0.0.0 (Settings → Server → Host)")
        }
        return out
    }
}
