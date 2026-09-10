import XCTest
@testable import MLXCore

/// The sandboxed-agent registry (issue #89 follow-up): pi + hermes rows, the
/// per-session bootstrap script, and the preflight gates. All pure — no VM.
final class SandboxAgentRegistryTests: XCTestCase {

    private let budget = AgentBudget.Budget(context: 65536, output: 16384)

    // MARK: registry shape

    func testRegistryShipsPiAndHermesWithUniqueIdsAndBootstrapPaths() {
        let ids = SandboxAgentRegistry.all.map(\.id)
        XCTAssertEqual(ids, ["pi", "hermes"], "v1 agents — add rows, don't reorder")
        XCTAssertEqual(Set(ids).count, ids.count)
        let paths = ids.map(SandboxAgentRegistry.bootstrapPath(agentId:))
        XCTAssertEqual(Set(paths).count, paths.count, "each agent gets its own /.vz-bootstrap-<id>")
        for p in paths { XCTAssertTrue(p.hasPrefix("/.vz-bootstrap-"), p) }
    }

    // MARK: pi row — config cross-pinned against AgentConfigs (the launcher's source of truth)

    func testPiConfigIsAgentConfigsPiModelsJSONAtTheHostPlaceholder() {
        let files = SandboxAgentRegistry.pi.configFiles("gemma-4-12b", 11234, budget, nil, [])
        XCTAssertEqual(files.count, 3, "models.json + AGENTS.md + the live-list extension")
        XCTAssertEqual(files[0].guestPath, "/root/.pi/agent/models.json",
                       "guest HOME is /root; pi reads ~/.pi/agent — no PI_CODING_AGENT_DIR isolation needed in a VM that is ALL ours")
        // Same generator the host launcher + MAS instructions use — one shape,
        // pinned in one place (the CLISetupInstructions cross-pin pattern).
        XCTAssertEqual(files[0].content,
                       AgentConfigs.piModelsJSON(baseURL: "http://__MLX_HOST__:11234",
                                                 model: "gemma-4-12b", budget: budget))
        XCTAssertTrue(files[0].content.contains("\"contextWindow\": 65536"),
                      "budget must ride the config verbatim (never hardcoded)")
        // The global context file pi injects into every session's system
        // prompt — the chunked-write convention (2026-07-20 mega-write loop).
        // Same builder the host launcher writes, so the surfaces can't drift.
        XCTAssertEqual(files[1].guestPath, "/root/.pi/agent/AGENTS.md")
        XCTAssertEqual(files[1].content, AgentConfigs.piAgentsMD(budget: budget))
    }

    func testPiConfigCarriesTheRealApiKeyWhenOneIsSet() {
        // Guest→host traffic arrives at the server NON-loopback (via the NAT
        // gateway), so a configured --api-key must actually be sent.
        let files = SandboxAgentRegistry.pi.configFiles("m", 8080, budget, "sekrit-123", [])
        XCTAssertTrue(files[0].content.contains("\"apiKey\": \"sekrit-123\""), files[0].content)
        // The live-list extension fetches /v1/models non-loopback too.
        XCTAssertTrue(files[2].content.contains("sekrit-123"), files[2].content)
        // No key set → the placeholder value the host launcher uses.
        let noKey = SandboxAgentRegistry.pi.configFiles("m", 8080, budget, nil, [])
        XCTAssertTrue(noKey[0].content.contains("\"apiKey\": \"mlx-serve\""), noKey[0].content)
    }

    func testPiInstallIsPinnedNpmGlobal() {
        let install = SandboxAgentRegistry.pi.installScript
        XCTAssertTrue(install.contains("npm i -g @earendil-works/pi-coding-agent@"),
                      "install must be pinned — an unpinned global install drifts under us: \(install)")
        XCTAssertEqual(SandboxAgentRegistry.pi.binaryName, "pi")
        XCTAssertTrue(SandboxAgentRegistry.pi.launchCommand("my-model").contains("--provider mlx"),
                      "launch must select the provider our models.json declares")
        XCTAssertTrue(SandboxAgentRegistry.pi.launchCommand("my-model").contains("my-model"))
    }

    // MARK: hermes row

    func testHermesConfigMirrorsTheSetupWizardsCustomEndpointSave() {
        // Contract verified against hermes_cli source (model_setup_flows.py
        // `_model_flow_custom` + main.py `_save_custom_provider`): the model
        // name key is `default` (NOT `model` — the docs' key), api_mode picks
        // the /chat/completions path, and context_length rides the
        // custom_providers entry. Writing exactly what the wizard saves means
        // the first run starts CONFIGURED instead of launching the wizard.
        let files = SandboxAgentRegistry.hermes.configFiles("qwen3.6-27b", 11234, budget, "k1", [])
        XCTAssertEqual(files.count, 2)
        XCTAssertEqual(files[0].guestPath, "/root/.hermes/config.yaml")
        let yaml = files[0].content
        XCTAssertTrue(yaml.contains("default: \"qwen3.6-27b\""), yaml)
        XCTAssertTrue(yaml.contains("provider: custom"), yaml)
        XCTAssertTrue(yaml.contains("base_url: \"http://__MLX_HOST__:11234/v1\""),
                      "hermes wants an OpenAI-style /v1 base: \(yaml)")
        XCTAssertTrue(yaml.contains("api_mode: chat_completions"),
                      "auto-detect heuristics are for unknown servers; ours IS /chat/completions: \(yaml)")
        XCTAssertTrue(yaml.contains("api_key: \"k1\""), yaml)
        XCTAssertTrue(yaml.contains("custom_providers:"), yaml)
        XCTAssertTrue(yaml.contains("context_length: 65536"),
                      "budget must ride the config verbatim (never hardcoded): \(yaml)")
    }

    func testHermesEnvFileDefusesTheFirstRunWizard() {
        // `_has_any_provider_configured()` gates the first-run setup wizard,
        // and OPENAI_BASE_URL in ~/.hermes/.env ALONE satisfies it (hermes's
        // own comment: local models often don't require an API key). Without
        // this file the wizard hijacks every fresh session (live 2026-07-19).
        let files = SandboxAgentRegistry.hermes.configFiles("m", 8080, budget, nil, [])
        let env = files.first { $0.guestPath == "/root/.hermes/.env" }
        XCTAssertNotNil(env, "\(files.map(\.guestPath))")
        XCTAssertTrue(env!.content.contains("OPENAI_BASE_URL=http://__MLX_HOST__:8080/v1"), env!.content)
        XCTAssertTrue(env!.content.contains("OPENAI_API_KEY=mlx-serve"),
                      "no key configured → the placeholder value: \(env!.content)")
        // The bootstrap seds EVERY config file — the .env must be in the loop
        // or the placeholder leaks into hermes's endpoint.
        let script = SandboxAgentRegistry.bootstrapScript(for: .hermes, model: "m")
        XCTAssertTrue(script.contains("/root/.hermes/.env"), script)
        XCTAssertTrue(script.contains("/root/.hermes/config.yaml"), script)
    }

    func testHermesInstallFallsBackToPip() {
        // hermes's installer is the discovery item of this feature — the row
        // isolates it: official install.sh first, pip fallback (the guest
        // image ships python3 + unblocked pip).
        let install = SandboxAgentRegistry.hermes.installScript
        XCTAssertTrue(install.contains("hermes-agent.nousresearch.com/install.sh"), install)
        XCTAssertTrue(install.contains("pip install hermes-agent"), install)
        XCTAssertEqual(SandboxAgentRegistry.hermes.binaryName, "hermes")
    }

    // MARK: pi live-list extension (in-guest /model switching)

    func testPiConfigShipsTheLiveModelListExtension() {
        // Third file: pi's extension fetches /v1/models IN-GUEST at session
        // start (the guest's own view — keyless LAN-share mode filters it to
        // shared models, which is exactly what the guest may use). Same
        // generator as the host launcher; content is entry-independent (the
        // list is live), so the bootstrap's dummy-args path enumeration
        // always includes it.
        let files = SandboxAgentRegistry.pi.configFiles("gemma-4-12b", 11234, budget, "k7", [])
        XCTAssertEqual(files.count, 3, "models.json + AGENTS.md + the live-list extension")
        let ext = files[2]
        XCTAssertEqual(ext.guestPath, "/root/.pi/agent/extensions/mlx-models.js",
                       "pi scans <agentDir>/extensions — guest agent dir is /root/.pi/agent")
        XCTAssertEqual(ext.content,
                       AgentConfigs.piModelsExtensionJS(baseURL: "http://__MLX_HOST__:11234",
                                                        apiKey: "k7"))
        // The bootstrap seds EVERY config file — the extension bakes the
        // placeholder base URL and must be in the loop.
        let script = SandboxAgentRegistry.bootstrapScript(for: .pi, model: "m")
        XCTAssertTrue(script.contains("/root/.pi/agent/extensions/mlx-models.js"), script)
    }

    func testHermesConfigFilesCarryTheModelListEntries() {
        let entries = [
            AgentModelEntry(id: "qwen3.6-27b", budget: AgentBudget.forServerContext(94729), vision: false),
            AgentModelEntry(id: "small-gguf", budget: AgentBudget.forServerContext(32768), vision: false),
        ]
        let files = SandboxAgentRegistry.hermes.configFiles("qwen3.6-27b", 11234, budget, nil, entries)
        let yaml = files[0].content
        XCTAssertTrue(yaml.contains("\"small-gguf\":"),
                      "every chat entry must be switchable via hermes /model: \(yaml)")
        XCTAssertTrue(yaml.contains("context_length: 32768"), yaml)
    }

    // MARK: bootstrap script

    func testBootstrapResolvesGatewaySubstitutesConfigsInstallsAndExecs() {
        let s = SandboxAgentRegistry.bootstrapScript(for: .pi, model: "gemma-4-12b")
        XCTAssertTrue(s.hasPrefix("#!/bin/sh\n"))
        // 1. The host's address inside the guest is the NAT default gateway —
        // resolved at RUN time (it can differ per boot), never baked host-side.
        XCTAssertTrue(s.contains("ip route"), s)
        XCTAssertTrue(s.contains("/^default/"), s)
        // 2. The placeholder is patched into every config file in-guest.
        XCTAssertTrue(s.contains("sed -i \"s/__MLX_HOST__/$GW/g\""), s)
        XCTAssertTrue(s.contains("/root/.pi/agent/models.json"), s)
        // 3. Install only when absent (first run streams into the terminal).
        XCTAssertTrue(s.contains("command -v pi >/dev/null"), s)
        XCTAssertTrue(s.contains(SandboxAgentRegistry.pi.installScript), s)
        // 4. Hand the tty to the agent — the bootstrap must not linger.
        XCTAssertTrue(s.contains("exec pi --provider mlx --model 'gemma-4-12b'"), s)
        // 5. A gateway resolve failure is loud, not a hung TUI.
        XCTAssertTrue(s.contains("exit 1"), s)
        // 6. Installer-managed bins must be on PATH before the probe.
        XCTAssertTrue(s.range(of: "/root/.local/bin")!.lowerBound
                      < s.range(of: "command -v")!.lowerBound, s)
    }

    // ssh drops the bootstrap in /root, and nothing cd'd before the exec —
    // live 2026-07-20 a pi session built its whole project in /root/fps-game,
    // invisible on the host, while the configured workspace share sat empty.
    // Every agent session starts in the folder the user picked (`/workspace`
    // for the default, `/projects/<slug>` for a hot-mounted one); a missing
    // share stays loud but non-fatal.
    func testBootstrapStartsAgentSessionsInTheirWorkspace() {
        for spec in SandboxAgentRegistry.all {
            let s = SandboxAgentRegistry.bootstrapScript(for: spec, model: "m", cwd: "/projects/app-1a2b3c")
            let cd = s.range(of: "cd '/projects/app-1a2b3c'")
            let ex = s.range(of: "exec ")
            XCTAssertNotNil(cd, "\(spec.id): must start in the picked workspace, not /root")
            XCTAssertNotNil(ex, spec.id)
            if let cd, let ex {
                XCTAssertLessThan(cd.lowerBound, ex.lowerBound,
                                  "\(spec.id): the cd must precede the agent exec")
            }
            XCTAssertTrue(s.contains("share missing"),
                          "\(spec.id): a missing share must be loud, never a silent /root session")
        }
    }

    func testBootstrapQuotesTheWorkspacePath() {
        // A folder name with an apostrophe must stay ONE quoted `cd` — a split
        // word here is the ShellSentinel-desync class, and the session would
        // start in $HOME with the agent none the wiser.
        let s = SandboxAgentRegistry.bootstrapScript(for: .pi, model: "m", cwd: "/projects/dave's-app")
        XCTAssertTrue(s.contains("cd '/projects/dave'\\''s-app' 2>/dev/null"), s)
    }

    func testBootstrapQuotesTheModelId() {
        // Model ids come from configs/user pulls — a space or quote must not
        // split the exec line.
        let s = SandboxAgentRegistry.bootstrapScript(for: .pi, model: "it's odd")
        XCTAssertTrue(s.contains("exec pi --provider mlx --model 'it'\\''s odd'"), s)
    }

    // MARK: materialization into the rootfs (host-side)

    func testMaterializeWritesConfigsAndExecutableBootstrapIntoRootfs() throws {
        let tmp = NSTemporaryDirectory() + "sbx-agent-mat-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: tmp) }

        let bootstrap = try SandboxAgentRegistry.materialize(
            spec: .pi, model: "gemma-4", serverPort: 11234,
            budget: budget, apiKey: "k9", entries: [], rootfsDir: tmp)

        XCTAssertEqual(bootstrap, "/.vz-bootstrap-pi", "returns the GUEST path the ssh session runs")
        // Config landed at the guest-absolute path under the rootfs, verbatim.
        let config = try String(contentsOfFile: tmp + "/root/.pi/agent/models.json", encoding: .utf8)
        XCTAssertEqual(config, AgentConfigs.piModelsJSON(baseURL: "http://__MLX_HOST__:11234",
                                                         model: "gemma-4", budget: budget, apiKey: "k9"))
        // Bootstrap exists and is executable (the ssh session `sh`s it, but
        // 0755 keeps it runnable by hand for debugging).
        let bsHost = tmp + "/.vz-bootstrap-pi"
        let script = try String(contentsOfFile: bsHost, encoding: .utf8)
        XCTAssertEqual(script, SandboxAgentRegistry.bootstrapScript(for: .pi, model: "gemma-4"))
        let perms = try FileManager.default.attributesOfItem(atPath: bsHost)[.posixPermissions] as? Int
        XCTAssertEqual(perms, 0o755)
    }

    // MARK: preflight

    func testServerBindReachableFromGuestAcceptsWildcardsOnly() {
        // The guest reaches the host via the NAT gateway address — a server
        // bound to loopback is unreachable from inside the VM.
        XCTAssertTrue(SandboxCliPreflight.serverBindReachableFromGuest(host: "0.0.0.0"))
        XCTAssertTrue(SandboxCliPreflight.serverBindReachableFromGuest(host: "::"))
        XCTAssertTrue(SandboxCliPreflight.serverBindReachableFromGuest(host: ""),
                      "empty Settings field launches with the 0.0.0.0 default")
        XCTAssertTrue(SandboxCliPreflight.serverBindReachableFromGuest(host: " 0.0.0.0 "))
        XCTAssertFalse(SandboxCliPreflight.serverBindReachableFromGuest(host: "127.0.0.1"))
        XCTAssertFalse(SandboxCliPreflight.serverBindReachableFromGuest(host: "::1"))
        XCTAssertFalse(SandboxCliPreflight.serverBindReachableFromGuest(host: "localhost"))
    }

    func testPreflightIssuesNameEveryBlockingGateDistinctly() {
        // All gates broken → each broken gate named, ACTIONABLE and distinct
        // (bind-address advice is suppressed while the server is down — it
        // would be noise next to "not running").
        let issues = SandboxCliPreflight.issues(sandboxEnabled: false, networkOn: false,
                                                serverRunning: false, serverHost: "127.0.0.1")
        XCTAssertEqual(issues.count, 3, "\(issues)")
        XCTAssertTrue(issues.contains { $0.contains("Agent Sandbox") })
        XCTAssertTrue(issues.contains { $0.lowercased().contains("network") })
        XCTAssertTrue(issues.contains { $0.lowercased().contains("server") && $0.lowercased().contains("running") })
        // Running but loopback-bound → ONLY the bind issue, naming the fix.
        let bind = SandboxCliPreflight.issues(sandboxEnabled: true, networkOn: true,
                                              serverRunning: true, serverHost: "127.0.0.1")
        XCTAssertEqual(bind.count, 1, "\(bind)")
        XCTAssertTrue(bind[0].contains("0.0.0.0"), bind[0])
        // All good → no issues.
        XCTAssertTrue(SandboxCliPreflight.issues(sandboxEnabled: true, networkOn: true,
                                                 serverRunning: true, serverHost: "0.0.0.0").isEmpty)
        // A stopped server must not ALSO complain about its bind address.
        let stopped = SandboxCliPreflight.issues(sandboxEnabled: true, networkOn: true,
                                                 serverRunning: false, serverHost: "127.0.0.1")
        XCTAssertEqual(stopped.count, 1, "bind-address advice is noise while the server is down: \(stopped)")
    }
}
