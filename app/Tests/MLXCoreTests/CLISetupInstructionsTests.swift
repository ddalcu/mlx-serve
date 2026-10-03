import XCTest
@testable import MLXCore

final class CLISetupInstructionsTests: XCTestCase {

    private let budget = AgentBudget.Budget(context: 90112, output: 16384)
    private var tabs: [CLISetupInstructions.Tab] {
        CLISetupInstructions.tabs(baseURL: "http://localhost:11234",
                                  servedModelId: "gemma-4-e4b-it-4bit",
                                  budget: budget)
    }

    func testOpenCodeProbeStatusesAndUncheckedLegacy() {
        let version = OpenCodeVersion(generation: .v2, version: "2.0.20")
        let vectors: [(String, Bool, OpenCodeProbe)] = [
            ("MLXOCV=missing\n", true, .missing),
            ("banner 18.2.0\nMLXOCV=missing\n", true, .missing),
            ("MLXOCV=127 \n", true, .versionFailed(output: "\n")),
            ("MLXOCV=1 failed\n", true, .versionFailed(output: "failed\n")),
            ("2.0.20", true, .versionFailed(output: "2.0.20")),
            ("MLXOCV=0", true, .versionFailed(output: "MLXOCV=0")),
            ("MLXOCV=x out", true, .versionFailed(output: "MLXOCV=x out")),
            ("MLXOCV=0 dev\n", true, .unparsed(output: "dev\n")),
            ("MLXOCV=0 2.0.20\n", false, .versionFailed(output: "MLXOCV=0 2.0.20\n")),
            ("MLXOCV=missing\n", false, .versionFailed(output: "MLXOCV=missing\n")),
            ("MLXOCV=missing\nMLXOCV=0 2.0.20\n", true, .ok(version)),
            ("MLXOCV=0 2.0.20\nMLXOCV=missing\n", true, .missing),
        ]
        for (capture, shellOK, expected) in vectors {
            XCTAssertEqual(classifyOpenCodeProbe(capture, shellOK: shellOK), expected)
        }
        for token in ["1.18.34", "2.0.20", "3.0.0", "10.2.0"] {
            XCTAssertEqual(classifyOpenCodeProbe("banner node 18.2.0\nMLXOCV=0 opencode v\(token)\nbuild x\n", shellOK: true),
                           .ok(OpenCodeVersion(generation: token.hasPrefix("1.") ? .v1 : .v2, version: token)))
        }
        XCTAssertEqual(resolveOpencode2Bin(detected: version, legacyInstalled: nil), "opencode")
        XCTAssertNil(resolveOpencode2Bin(detected: nil, legacyInstalled: nil))
        XCTAssertEqual(decideOpenCodeLaunch(forcedV2: true, detection: OpenCodeDetection(
            probe: .ok(version), legacyOpencode2Installed: nil)), .v2(version: version, binary: "opencode"))
        XCTAssertEqual(decideOpenCodeLaunch(forcedV2: true, detection: OpenCodeDetection(
            probe: .missing, legacyOpencode2Installed: nil)), .shellFailed)
        XCTAssertEqual(decideOpenCodeLaunch(forcedV2: true, detection: OpenCodeDetection(
            probe: .missing, legacyOpencode2Installed: false)), .noV2Binary)
    }

    func testOpenCodeManualTabExplainsItsGeneration() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "opencode" })
        XCTAssertTrue(tab.installHint.contains("1.x"))
        XCTAssertTrue(tab.installHint.contains("2.x+"))
        XCTAssertTrue(tab.installHint.contains("OpenCode 2 tab"))
    }

    func testProductionOpenCodeProbeCountsShellsAndSkipsUnusedLegacy() throws {
        let scratch = ProcessInfo.processInfo.environment["TMPDIR"]
            .flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("tmp", isDirectory: true)
        let root = scratch.appendingPathComponent("pr723-swift-fixture-" + UUID().uuidString)
        let fm = FileManager.default
        try fm.createDirectory(at: root.appendingPathComponent("bin"), withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let oldHome = getenv("HOME").map { String(cString: $0) }
        let oldZdotdir = getenv("ZDOTDIR").map { String(cString: $0) }
        // /etc/zshrc sources /etc/zshrc_$TERM_PROGRAM; Terminal.app's session save prints after the marker.
        let oldTermProgram = getenv("TERM_PROGRAM").map { String(cString: $0) }
        setenv("HOME", root.path, 1)
        setenv("ZDOTDIR", root.path, 1)
        unsetenv("TERM_PROGRAM")
        defer {
            if let oldHome { setenv("HOME", oldHome, 1) } else { unsetenv("HOME") }
            if let oldZdotdir { setenv("ZDOTDIR", oldZdotdir, 1) } else { unsetenv("ZDOTDIR") }
            if let oldTermProgram { setenv("TERM_PROGRAM", oldTermProgram, 1) }
        }
        try "export HOME=\(CLILauncher.shellSingleQuoted(root.path))\n".write(
            to: root.appendingPathComponent(".zshenv"), atomically: true, encoding: .utf8)
        try "export PATH=\(CLILauncher.shellSingleQuoted(root.appendingPathComponent("bin").path)):/usr/bin:/bin\n".write(
            to: root.appendingPathComponent(".zprofile"), atomically: true, encoding: .utf8)
        try """
        print shell >> "$HOME/shells"
        print 'banner node 18.2.0'
        command() {
            if [[ "$*" == *opencode2* ]]; then print legacy >> "$HOME/legacy"; fi
            builtin command "$@"
        }
        """.write(to: root.appendingPathComponent(".zshrc"), atomically: true, encoding: .utf8)
        let vectors: [(String?, Bool, Int, Int, OpenCodeProbe)] = [
            ("print 2.0.20", false, 1, 0, .ok(OpenCodeVersion(generation: .v2, version: "2.0.20"))),
            ("print 3.0.0", true, 1, 0, .ok(OpenCodeVersion(generation: .v2, version: "3.0.0"))),
            ("print 1.18.34", true, 2, 1, .ok(OpenCodeVersion(generation: .v1, version: "1.18.34"))),
            ("exit 127", false, 1, 0, .versionFailed(output: "\n")),
            ("exit 127", true, 2, 1, .versionFailed(output: "\n")),
            (nil, false, 1, 0, .missing),
            (nil, true, 2, 1, .missing),
        ]
        for (body, forcedV2, shells, legacyChecks, expected) in vectors {
            for file in ["shells", "legacy", "bin/opencode"] {
                try? fm.removeItem(at: root.appendingPathComponent(file))
            }
            if let body {
                let executable = root.appendingPathComponent("bin/opencode")
                try ("#!/bin/zsh -f\n" + body + "\n").write(to: executable, atomically: true, encoding: .utf8)
                try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
            }
            let detection = CLILauncher.probeOpenCode(forcedV2: forcedV2)
            XCTAssertEqual(detection.probe, expected)
            func count(_ name: String) -> Int {
                (try? String(contentsOf: root.appendingPathComponent(name), encoding: .utf8))?
                    .split(separator: "\n").count ?? 0
            }
            XCTAssertEqual(count("shells"), shells, "forcedV2=\(forcedV2), body=\(body ?? "missing")")
            XCTAssertEqual(count("legacy"), legacyChecks)
            if legacyChecks == 0 {
                XCTAssertNil(detection.legacyOpencode2Installed)
            } else {
                XCTAssertEqual(detection.legacyOpencode2Installed, false)
            }
        }
        let legacy = root.appendingPathComponent("bin/opencode2")
        try "#!/bin/zsh -f\nexit 0\n".write(to: legacy, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: legacy.path)
        let fallback = CLILauncher.probeOpenCode(forcedV2: true)
        XCTAssertEqual(fallback.legacyOpencode2Installed, true)
        XCTAssertEqual(decideOpenCodeLaunch(forcedV2: true, detection: fallback),
                       .v2(version: nil, binary: "opencode2"))
        try "exit 1\n".write(to: root.appendingPathComponent(".zshrc"), atomically: true, encoding: .utf8)
        let unanswered = CLILauncher.probeOpenCode(forcedV2: true)
        XCTAssertNil(unanswered.legacyOpencode2Installed)
        XCTAssertEqual(decideOpenCodeLaunch(forcedV2: true, detection: unanswered), .shellFailed)
    }

    func testZCodeConfigForceIncludesTheServedModelWithItsBudget() throws {
        let served = "served/model\"with-quote"
        let other = AgentModelEntry(id: "other", budget: AgentBudget.Budget(context: 8192, output: 4096), vision: true)
        let json = AgentConfigs.zcodeProviderJSON(baseURL: "http://localhost:11234", model: served,
                                                 budget: budget, entries: [other])
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let config = try XCTUnwrap(root["config"] as? [String: Any])
        let selection = try XCTUnwrap(config["defaultModelSelection"] as? [String: Any])
        XCTAssertEqual(selection["modelId"] as? String, served)
        let providerRules = try XCTUnwrap((config["providerConfigRules"] as? [String: Any])?["providerRules"] as? [[String: Any]])
        let provider = try XCTUnwrap(providerRules.first)
        XCTAssertEqual(provider["providerName"] as? String, "mlx-serve")
        let providerConfig = try XCTUnwrap(provider["config"] as? [String: Any])
        XCTAssertEqual(providerConfig["personalModelIds"] as? [String], [served, "other"])
        XCTAssertEqual((providerConfig["api"] as? [String: Any])?["baseUrl"] as? String, "http://localhost:11234/v1")
        let rules = try XCTUnwrap((config["modelConfigRules"] as? [String: Any])?["providerModelRules"] as? [[String: Any]])
        let first = try XCTUnwrap(rules.first?["config"] as? [String: Any])
        XCTAssertEqual(rules.first?["modelId"] as? String, served)
        XCTAssertEqual((first["properties"] as? [String: Any])?["contextWindow"] as? Int, budget.context)
    }

    func testZCodeGuardsAMissingBinaryOnlyInTheLauncherScript() throws {
        let script = LauncherCLI.zcode.scriptBody("http://localhost:11234", "m", "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains("exit 127"), script)
        XCTAssertTrue(script.contains("zcode \"$@\""), script)
        let tab = try XCTUnwrap(tabs.first { $0.id == "zcode" })
        // Pasted into the user's own interactive shell: an `exit` closes it.
        XCTAssertFalse(tab.command.contains("exit"), tab.command)
        XCTAssertTrue(tab.command.contains("ZCODE_PERSONAL_PROVIDER_CONFIG_FILE"), tab.command)
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/zcode/provider_config.json <<'EOF'"), tab.command)
    }

    func testTabsHaveStableIdsInLauncherOrder() {
        XCTAssertEqual(tabs.map(\.id),
                       ["claude", "pi", "omp", "opencode", "opencode2", "codex", "hermes", "aider", "zcode"],
                       "same CLIs, same order as the DMG launcher dropdown")
        for tab in tabs {
            XCTAssertFalse(tab.command.isEmpty, tab.id)
            XCTAssertFalse(tab.installHint.isEmpty, tab.id)
        }
    }

    /// The two surfaces must offer the SAME CLIs in the SAME order — a CLI the
    /// launcher gains that the panel never shows is the silent-hole class.
    func testPanelAndLauncherOfferTheSameCLIs() {
        XCTAssertEqual(tabs.map(\.id), CLILauncher.candidateIds)
    }

    func testPlainShellLauncherNeedsNoServerAndKeepsTheShellAlive() {
        let shell = LauncherCLI.shell
        XCTAssertFalse(shell.requiresServer)
        let script = shell.scriptBody("http://localhost:11234", "m1", "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains("cd '/tmp'"), script)
        // Without an exec the script would run to the end and the row would
        // close the moment it opened.
        XCTAssertTrue(script.contains("exec"), script)
    }

    /// The plain shell rides beside the detected CLIs but is not a candidate:
    /// there is no binary to probe and no instructions tab for it.
    func testShellIsOfferedButNotDetected() {
        XCTAssertFalse(CLILauncher.candidateIds.contains("shell"))
        XCTAssertEqual(CLILauncher.offered(detected: [.claudeCode]).map(\.id),
                       ["claude", "shell"])
    }

    func testEveryOtherLauncherStillRequiresTheServer() {
        for cli in [LauncherCLI.claudeCode, .pi, .omp, .opencode, .opencode2, .codex, .hermes, .aider, .zcode] {
            XCTAssertTrue(cli.requiresServer, cli.id)
        }
    }

    func testClaudeTabExportsTheEnvAndLaunches() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "claude" })
        // Verbatim reuse of the launcher's env block — the drift guard.
        XCTAssertTrue(tab.command.contains(AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)))
        XCTAssertTrue(tab.command.contains("claude --model gemma-4-e4b-it-4bit"))
    }

    /// pi has no env-var/flag route for a custom base URL — a models.json is
    /// required — but `PI_CODING_AGENT_DIR` relocates the whole config dir. We
    /// use a dedicated dir so the instructions NEVER overwrite a user's real
    /// `~/.pi/agent/models.json` (a `cat >` there would destroy any providers
    /// they already configured).
    func testPiTabWritesAnIsolatedConfigDirNeverTheUsersRealOne() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "pi" })
        XCTAssertTrue(tab.command.contains("mkdir -p ~/.mlx-serve/pi"))
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/pi/models.json <<'EOF'"),
                      "heredoc must be quoted or the shell expands the JSON's contents")
        XCTAssertTrue(tab.command.contains(#"export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi""#))
        // The embedded config is the launcher's builder output, byte for byte.
        XCTAssertTrue(tab.command.contains(AgentConfigs.piModelsJSON(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)))
        XCTAssertTrue(tab.command.contains("pi --provider mlx --model gemma-4-e4b-it-4bit"))
        // The budget the server advertised travels into the user's config.
        XCTAssertTrue(tab.command.contains("\"contextWindow\": 90112"))
        // The non-clobber guarantee itself.
        XCTAssertFalse(tab.command.contains("~/.pi"), "must never touch the user's real pi config")
    }

    /// The DMG one-click launcher must make the same non-clobber move: its
    /// script exports PI_CODING_AGENT_DIR at the SAME dir the instructions use,
    /// or the two surfaces configure two different pis.
    func testDMGLauncherUsesTheSameIsolatedPiConfigDir() {
        let script = LauncherCLI.pi.scriptBody("http://localhost:11234", "gemma-4-e4b-it-4bit",
                                               "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains(#"export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi""#), script)
        XCTAssertTrue(script.contains("pi --provider mlx --model gemma-4-e4b-it-4bit"))
    }

    /// opencode needs NO file at all: `OPENCODE_CONFIG_CONTENT` carries the
    /// config inline and MERGES over the user's global/project config (docs:
    /// "Configuration files are merged together, not replaced"), so their own
    /// settings and plugins keep working with our provider added on top.
    func testOpencodeTabInlinesTheConfigWithNoFileWrites() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "opencode" })
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)
        XCTAssertTrue(tab.command.contains("export OPENCODE_CONFIG_CONTENT='\(json)'"))
        XCTAssertTrue(tab.command.contains("opencode --model mlx/gemma-4-e4b-it-4bit"))
        // No file mechanism left — nothing to create, nothing to clobber.
        XCTAssertFalse(tab.command.contains("cat >"), tab.command)
        XCTAssertFalse(tab.command.contains("opencode.json"), tab.command)
        // The inline export is single-quoted; a quote INSIDE the JSON would
        // truncate it silently in the user's shell.
        XCTAssertFalse(json.contains("'"), "opencodeJSON must stay single-quote-free")
    }

    func testOpencode2TabUsesDedicatedXdgConfigAndRegistersThePlugin() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "opencode2" })
        XCTAssertTrue(tab.command.contains(#"export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2""#))
        XCTAssertTrue(tab.command.contains(#""$oc_bin" --standalone"#), tab.command)
        XCTAssertFalse(tab.command.contains("--model"), tab.command)
        XCTAssertTrue(tab.command.contains(#""model": "mlx/gemma-4-e4b-it-4bit""#))
        XCTAssertTrue(tab.installHint.contains("curl -fsSL https://opencode.ai/install | bash"))
        XCTAssertTrue(tab.installHint.contains("opencode 2.x+"))
        XCTAssertTrue(tab.command.contains("./plugins/mlx-serve"))
        XCTAssertTrue(tab.command.contains("http://localhost:11234/metrics.json"))
        XCTAssertFalse(tab.command.contains("~/.config/opencode"), "must never write the user's real opencode config")
        // The v2 binary is version-resolved: Homebrew ships 2.x as `opencode`.
        XCTAssertTrue(tab.command.contains("grep -qE"), tab.command)
        XCTAssertTrue(tab.command.contains("oc_bin=opencode"), tab.command)
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", defaultModel: "gemma-4-e4b-it-4bit",
            entries: [AgentModelEntry(id: "gemma-4-e4b-it-4bit", budget: budget, vision: false)],
            pinModel: true, compaction: true)
        XCTAssertTrue(tab.command.contains("export OPENCODE_CONFIG_CONTENT='\(json)'"))
    }

    func testDMGLauncherOpencode2MatchesTheTab() throws {
        XCTAssertNotNil(LauncherCLI.opencode2.prepareConfig)
        let script = LauncherCLI.opencode2.scriptBody("http://localhost:11234",
                                                     "gemma-4-e4b-it-4bit", "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains(#"export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2""#), script)
        XCTAssertTrue(script.contains(#""$oc_bin" --standalone"#), script)
        XCTAssertFalse(script.contains("opencode2 --model"), script)
        XCTAssertTrue(script.contains(#""model": "mlx/gemma-4-e4b-it-4bit""#), script)
        XCTAssertTrue(script.contains("exit 127"), script)
        // Resolver order like `launch opencode2`: v2 `opencode`, then legacy, else refuse.
        let lines = script.split(separator: "\n").map(String.init)
        let assign = try XCTUnwrap(lines.first { $0.hasPrefix("oc_bin=") })
        let check = try XCTUnwrap(lines.first { $0.contains("grep -qE") })
        XCTAssertLessThan(lines.firstIndex(of: assign)!, lines.firstIndex(of: check)!,
                          "oc_bin must default to the legacy binary and be replaced only by a v2 version check")
    }

    // MARK: - OpenCode version routing (twin of Zig's launch.zig detection)

    /// Same table as Zig's `parseOpencodeVersion` unit test.
    func testOpencodeVersionParserTable() {
        for input in ["1.18.34", "v1.18.34", "opencode 1.18.34\n"] {
            let got = try? XCTUnwrap(parseOpencodeVersion(input))
            XCTAssertEqual(got?.generation, .v1, input)
            XCTAssertEqual(got?.version, "1.18.34", input)
        }
        let later: [(String, String)] = [
            ("2.0.20", "2.0.20"),
            ("v2.0.20", "2.0.20"),
            ("opencode v2.0.20", "2.0.20"),
            ("3.0.0", "3.0.0"),
            ("2.0.20-nightly", "2.0.20"),
        ]
        for (input, version) in later {
            let got = try? XCTUnwrap(parseOpencodeVersion(input))
            XCTAssertEqual(got?.generation, .v2, input)
            XCTAssertEqual(got?.version, version, input)
        }
        for input in ["", "dev", "0.14.0", "OpenCode — canary build"] {
            XCTAssertNil(parseOpencodeVersion(input), input)
        }
    }

    /// Same marker extraction as Zig: rc banners cannot pose as the version.
    func testExtractMarkedVersionIsolatesTheVersionSubshell() {
        let m = extractMarkedVersion("Welcome! node 18.2.0\nMLXOCV=0 opencode v2.0.20\n")
        XCTAssertEqual(m?.rc, 0)
        let v = m.flatMap { parseOpencodeVersion($0.out) }
        XCTAssertEqual(v?.generation, .v2)
        XCTAssertEqual(v?.version, "2.0.20")
        XCTAssertEqual(extractMarkedVersion("MLXOCV=127 opencode: boom\n")?.rc, 127)
        XCTAssertEqual(extractMarkedVersion("b\nMLXOCV=0 open 2.0.20\nbuild x\n")?.out.contains("build x"), true)
        XCTAssertNil(extractMarkedVersion("opencode v2.0.20"))
        XCTAssertNil(extractMarkedVersion("MLXOCV=0"))
        XCTAssertNil(extractMarkedVersion("MLXOCV=x out"))
    }

    /// The same table as Zig's `resolveOpencode2Bin` unit test.
    func testOpencode2ResolverPrefersTheV2OpencodeThenTheLegacyBinary() {
        let v2 = OpenCodeVersion(generation: .v2, version: "2.0.20")
        let v3 = OpenCodeVersion(generation: .v2, version: "3.0.0")
        let v1 = OpenCodeVersion(generation: .v1, version: "1.18.34")
        XCTAssertEqual(resolveOpencode2Bin(detected: v2, legacyInstalled: true), "opencode")
        XCTAssertEqual(resolveOpencode2Bin(detected: v3, legacyInstalled: false), "opencode")
        XCTAssertEqual(resolveOpencode2Bin(detected: v1, legacyInstalled: true), "opencode2")
        XCTAssertEqual(resolveOpencode2Bin(detected: nil, legacyInstalled: true), "opencode2")
        XCTAssertNil(resolveOpencode2Bin(detected: v1, legacyInstalled: false))
        XCTAssertNil(resolveOpencode2Bin(detected: nil, legacyInstalled: false))
    }

    /// Routing notices and refuse-on-failure for `launch opencode`.
    @MainActor
    func testLaunchOpencodeRoutesOnTheDetectedVersion() throws {
        let detection = OpenCodeDetection(probe: .ok(OpenCodeVersion(generation: .v2, version: "2.0.20")),
                                          legacyOpencode2Installed: false)
        let routed = try XCTUnwrap(CLILauncher.launchCommand(
            .opencode, baseURL: "http://localhost:11234", servedModelId: "m1",
            budget: budget, entries: [], workingDirectory: "/tmp",
            opencodeDetection: detection).args.last)
        let script = try String(contentsOfFile: routed, encoding: .utf8)
        XCTAssertTrue(script.contains("detected OpenCode 2.0.20; using the v2 integration."), script)
        XCTAssertTrue(script.contains("\nopencode --standalone "), script)
        XCTAssertTrue(script.contains(#"export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2""#), script)
        XCTAssertTrue(script.contains(#""model": "mlx/m1""#), script)
        XCTAssertFalse(script.contains("--model mlx/"), script)

        let v1 = OpenCodeDetection(probe: .ok(OpenCodeVersion(generation: .v1, version: "1.18.34")),
                                   legacyOpencode2Installed: true)
        let old = try String(contentsOfFile: try XCTUnwrap(CLILauncher.launchCommand(
            .opencode, baseURL: "http://localhost:11234", servedModelId: "m1",
            budget: budget, entries: [], workingDirectory: "/tmp",
            opencodeDetection: v1).args.last), encoding: .utf8)
        XCTAssertTrue(old.contains("detected OpenCode 1.18.34; using the v1 integration."), old)
        XCTAssertTrue(old.contains("opencode --model mlx/m1"), old)
        XCTAssertFalse(old.contains("XDG_CONFIG_HOME"), old)

        let missing = OpenCodeDetection(probe: .missing, legacyOpencode2Installed: false)
        let none = try String(contentsOfFile: try XCTUnwrap(CLILauncher.launchCommand(
            .opencode, baseURL: "http://localhost:11234", servedModelId: "m1",
            budget: budget, entries: [], workingDirectory: "/tmp",
            opencodeDetection: missing).args.last), encoding: .utf8)
        XCTAssertTrue(none.contains("OpenCode is not installed or is not available on PATH."), none)
        XCTAssertTrue(none.contains("exit 1"), none)

        // A shell that cannot start is not reported as "not installed".
        let unrunnable = try String(contentsOfFile: try XCTUnwrap(CLILauncher.launchCommand(
            .opencode, baseURL: "http://localhost:11234", servedModelId: "m1",
            budget: budget, entries: [], workingDirectory: "/tmp",
            opencodeDetection: OpenCodeDetection(probe: .shellUnrunnable,
                                                 legacyOpencode2Installed: false)).args.last), encoding: .utf8)
        XCTAssertTrue(unrunnable.contains("could not run the login shell"), unrunnable)
        XCTAssertFalse(unrunnable.contains("not installed"), unrunnable)

        // A `--version` failure quotes the CLI's own output and refuses to guess.
        let failed = OpenCodeDetection(probe: .versionFailed(output: "boom's bad"), legacyOpencode2Installed: true)
        let bad = try String(contentsOfFile: try XCTUnwrap(CLILauncher.launchCommand(
            .opencode, baseURL: "http://localhost:11234", servedModelId: "m1",
            budget: budget, entries: [], workingDirectory: "/tmp",
            opencodeDetection: failed).args.last), encoding: .utf8)
        XCTAssertTrue(bad.contains("could not determine the installed OpenCode version"), bad)
        XCTAssertTrue(bad.contains(#"printf '%s\n' 'boom'\''s bad'"#), bad)
        XCTAssertTrue(bad.contains("supports OpenCode 1.x (the v1 integration) and 2.x or newer"), bad)
        XCTAssertFalse(bad.contains("opencode --model mlx/m1"), "a detection failure must not launch a fallback profile")
    }

    /// `launch opencode2` resolver order; a v1 `opencode` never starts under v2 config.
    @MainActor
    func testLaunchOpencode2AliasResolvesTheV2Binary() throws {
        func script(_ detection: OpenCodeDetection) throws -> String {
            let path = try XCTUnwrap(CLILauncher.launchCommand(
                .opencode2, baseURL: "http://localhost:11234", servedModelId: "m1",
                budget: budget, entries: [], workingDirectory: "/tmp",
                opencodeDetection: detection).args.last)
            return try String(contentsOfFile: path, encoding: .utf8)
        }
        let v2 = try script(OpenCodeDetection(
            probe: .ok(OpenCodeVersion(generation: .v2, version: "2.0.20")), legacyOpencode2Installed: true))
        XCTAssertTrue(v2.contains("detected OpenCode 2.0.20; using the v2 integration."), v2)
        XCTAssertTrue(v2.contains("\nopencode --standalone "), v2)
        XCTAssertFalse(v2.contains("opencode2 --standalone"), v2)

        let legacy = try script(OpenCodeDetection(
            probe: .ok(OpenCodeVersion(generation: .v1, version: "1.18.34")), legacyOpencode2Installed: true))
        XCTAssertTrue(legacy.contains("\nopencode2 --standalone "), legacy)

        let refused = try script(OpenCodeDetection(probe: .missing, legacyOpencode2Installed: false))
        XCTAssertTrue(refused.contains("no OpenCode v2 binary found"), refused)
        XCTAssertFalse(refused.contains("--standalone"), refused)
    }

    func testOpencode2CliJsonMergeKeepsThemeAndReplacesMlxServe() throws {
        let existing = """
        {"theme":"nord","plugins":[{"package":"other-plugin","options":{"a":1}},{"package":"./plugins/mlx-serve","options":{"metricsUrl":"http://old:1/metrics.json"}}]}
        """
        let json = AgentConfigs.opencode2CliJSON(existing: existing, baseURL: "http://127.0.0.1:11234")
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(obj["theme"] as? String, "nord")
        let plugins = try XCTUnwrap(obj["plugins"] as? [Any]).compactMap { $0 as? [String: Any] }
        XCTAssertEqual(plugins.count, 2)
        let other = plugins.first { ($0["package"] as? String) == "other-plugin" }
        XCTAssertNotNil(other)
        XCTAssertEqual((other?["options"] as? [String: Any])?["a"] as? Int, 1)
        let mlx = plugins.filter { ($0["package"] as? String) == "./plugins/mlx-serve" }
        XCTAssertEqual(mlx.count, 1)
        let opts = try XCTUnwrap(mlx[0]["options"] as? [String: Any])
        XCTAssertEqual(opts["metricsUrl"] as? String, "http://127.0.0.1:11234/metrics.json")
        XCTAssertNil(opts["metricsToken"])

        let remote = AgentConfigs.opencode2CliJSON(existing: "{}", baseURL: "http://10.0.0.2:11234")
        let remoteObj = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(remote.utf8)) as? [String: Any])
        let remotePlugins = try XCTUnwrap(remoteObj["plugins"] as? [Any]).compactMap { $0 as? [String: Any] }
        let remoteOpts = try XCTUnwrap(remotePlugins[0]["options"] as? [String: Any])
        XCTAssertEqual(remoteOpts["metricsToken"] as? String, "mlx-serve")
    }

    /// The DMG one-click launcher makes the same move: inline env var in the
    /// script, no prepareConfig side-effect writing temp files.
    func testDMGLauncherInlinesTheOpencodeConfigToo() {
        XCTAssertNil(LauncherCLI.opencode.prepareConfig,
                     "no file writes — the config rides OPENCODE_CONFIG_CONTENT")
        let script = LauncherCLI.opencode.scriptBody("http://localhost:11234",
                                                     "gemma-4-e4b-it-4bit", "cd '/tmp'", budget, [])
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)
        XCTAssertTrue(script.contains("export OPENCODE_CONFIG_CONTENT='\(json)'"), script)
        XCTAssertTrue(script.contains("opencode --model mlx/gemma-4-e4b-it-4bit"))
    }

    /// omp (oh-my-pi) is a pi fork with its own config tree: models.yml (YAML,
    /// not models.json) under the agent dir. The env read is still pi's
    /// PI_CODING_AGENT_DIR spelling (measured on omp v17 — the changelog's
    /// OMP_ rename reached only its help text), so both spellings are
    /// exported. Same isolation move as pi — never the user's real ~/.omp.
    func testOmpTabWritesAnIsolatedConfigDirNeverTheUsersRealOne() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "omp" })
        XCTAssertTrue(tab.command.contains("mkdir -p ~/.mlx-serve/omp"))
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/omp/models.yml <<'EOF'"))
        XCTAssertTrue(tab.command.contains(#"export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp""#))
        XCTAssertTrue(tab.command.contains(#"export OMP_CODING_AGENT_DIR="$HOME/.mlx-serve/omp""#))
        XCTAssertTrue(tab.command.contains(AgentConfigs.ompModelsYML(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)))
        XCTAssertTrue(tab.command.contains("omp --model mlx/gemma-4-e4b-it-4bit"))
        XCTAssertTrue(tab.command.contains("contextWindow: 90112"))
        XCTAssertFalse(tab.command.contains("~/.omp"), "must never touch the user's real omp config")
    }

    func testDMGLauncherUsesTheSameIsolatedOmpConfigDir() {
        let script = LauncherCLI.omp.scriptBody("http://localhost:11234", "gemma-4-e4b-it-4bit",
                                                "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains(#"export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp""#), script)
        XCTAssertTrue(script.contains("omp --model mlx/gemma-4-e4b-it-4bit"))
    }

    /// omp's models.yml is a STATIC chat-capable list — deliberately NOT
    /// omp's openai-models-list discovery, which would put every media model
    /// in the coding agent's picker at omp's 128k default context. Each entry
    /// carries its own budget.
    func testOmpConfigBakesTheChatEntriesStatically() {
        let entries = [
            AgentModelEntry(id: "m1", budget: .init(context: 4096, output: 1024), vision: false),
            AgentModelEntry(id: "m2", budget: .init(context: 262144, output: 65536), vision: true),
        ]
        let yml = AgentConfigs.ompModelsYML(
            baseURL: "http://localhost:11234", defaultModel: "m1", entries: entries)
        XCTAssertFalse(yml.contains("discovery"), yml)
        XCTAssertTrue(yml.contains("baseUrl: http://localhost:11234/v1"), yml)
        XCTAssertTrue(yml.contains("api: openai-completions"), yml)
        XCTAssertTrue(yml.contains("contextWindow: 4096"), yml)
        XCTAssertTrue(yml.contains("contextWindow: 262144"), yml)
        XCTAssertTrue(yml.contains("maxTokens: 65536"), yml)
        XCTAssertTrue(yml.contains("input: [text, image]"), yml)
        XCTAssertTrue(yml.contains("thinkingFormat: qwen"), yml)
    }

    /// codex only speaks the Responses wire API (WireApi has one variant) and
    /// honors CODEX_HOME for its whole config tree — dedicated dir, keyless
    /// provider (no env_key: the loopback server ignores keys).
    func testCodexTabWritesAnIsolatedCodexHome() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "codex" })
        XCTAssertTrue(tab.command.contains("mkdir -p ~/.mlx-serve/codex"))
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/codex/config.toml <<'EOF'"))
        XCTAssertTrue(tab.command.contains(#"export CODEX_HOME="$HOME/.mlx-serve/codex""#))
        XCTAssertTrue(tab.command.contains(AgentConfigs.codexConfigTOML(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)))
        XCTAssertFalse(tab.command.contains("~/.codex"), "must never touch the user's real codex config")
    }

    /// The ChatGPT desktop app (codex's rebranded app; bundle id
    /// com.openai.codex, shipped as ChatGPT.app or Codex.app) bundles the
    /// codex CLI at Contents/Resources/codex — a desktop-app-only user has a
    /// working binary that is NOT on PATH. Both launch surfaces resolve it
    /// through the same shell snippet, and refuse with the install hint
    /// instead of exec'ing an empty string.
    func testCodexLaunchFallsBackToTheDesktopAppBundledBinary() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "codex" })
        let script = LauncherCLI.codex.scriptBody("http://localhost:11234", "m1",
                                                  "", budget, [])
        for surface in [tab.command, script] {
            XCTAssertTrue(surface.contains(AgentConfigs.codexBinResolver), surface)
            XCTAssertTrue(surface.contains("\"$CODEX_BIN\""), surface)
            XCTAssertFalse(surface.contains("\ncodex\n"), "bare codex would miss the bundled binary")
        }
        XCTAssertTrue(AgentConfigs.codexBinResolver.contains("/Applications/ChatGPT.app"))
        XCTAssertTrue(AgentConfigs.codexBinResolver.contains("/Applications/Codex.app"))
        XCTAssertTrue(AgentConfigs.codexBinResolver.contains("$HOME/Applications"))
        XCTAssertTrue(AgentConfigs.codexBinResolver.contains("Contents/Resources/codex"))
    }

    /// Detection must also SHOW the codex row for a desktop-app-only user:
    /// the `command -v` sweep can't see inside an app bundle, so codex
    /// declares the bundle paths as detection fallbacks.
    func testCodexDetectionProbesTheAppBundles() {
        XCTAssertEqual(LauncherCLI.codex.fallbackPaths.count, 4)
        XCTAssertTrue(LauncherCLI.codex.fallbackPaths.contains(
            "/Applications/ChatGPT.app/Contents/Resources/codex"))
        for cli in [LauncherCLI.claudeCode, .pi, .omp, .opencode, .opencode2, .hermes, .aider, .zcode] {
            XCTAssertTrue(cli.fallbackPaths.isEmpty, cli.id)
        }
    }

    func testCodexConfigTargetsOurResponsesAPIAndCarriesTheContext() {
        let toml = AgentConfigs.codexConfigTOML(
            baseURL: "http://localhost:11234", model: "m1", budget: budget)
        XCTAssertTrue(toml.contains(#"wire_api = "responses""#), toml)
        XCTAssertTrue(toml.contains(#"base_url = "http://localhost:11234/v1""#), toml)
        XCTAssertTrue(toml.contains("model_context_window = \(budget.context)"), toml)
        XCTAssertTrue(toml.contains(#"model = "m1""#), toml)
        XCTAssertTrue(toml.contains(#"model_provider = "mlx""#), toml)
        XCTAssertFalse(toml.contains("env_key"), "keyless — loopback is exempt from --api-key")
    }

    /// hermes reads its whole tree from HERMES_HOME (hermes_constants.py) —
    /// the same config.yaml + .env pair the sandbox materializes in-guest,
    /// relocated to a dedicated dir on the host.
    func testHermesTabWritesAnIsolatedHermesHome() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "hermes" })
        XCTAssertTrue(tab.command.contains("mkdir -p ~/.mlx-serve/hermes"))
        XCTAssertTrue(tab.command.contains(#"export HERMES_HOME="$HOME/.mlx-serve/hermes""#))
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/hermes/config.yaml <<'EOF'"))
        // The .env is the first-run wizard kill switch (OPENAI_BASE_URL set).
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/hermes/.env <<'ENVEOF'"))
        XCTAssertTrue(tab.command.contains("OPENAI_BASE_URL=http://localhost:11234/v1"))
        XCTAssertFalse(tab.command.contains("~/.hermes"), "must never touch the user's real hermes config")
    }

    func testDMGLauncherUsesTheSameIsolatedHermesHome() {
        let script = LauncherCLI.hermes.scriptBody("http://localhost:11234", "gemma-4-e4b-it-4bit",
                                                   "cd '/tmp'", budget, [])
        XCTAssertTrue(script.contains(#"export HERMES_HOME="$HOME/.mlx-serve/hermes""#), script)
    }

    /// aider is pure env vars (OPENAI_API_BASE) plus a litellm metadata file
    /// that tells it the real context window — without it aider assumes its
    /// own defaults for unknown openai/<id> models.
    func testAiderTabExportsEnvAndWritesTheMetadataFile() throws {
        let tab = try XCTUnwrap(tabs.first { $0.id == "aider" })
        XCTAssertTrue(tab.command.contains("export OPENAI_API_BASE='http://localhost:11234/v1'"))
        XCTAssertTrue(tab.command.contains("cat > ~/.mlx-serve/aider/model-metadata.json <<'EOF'"))
        XCTAssertTrue(tab.command.contains("aider --model openai/gemma-4-e4b-it-4bit"))
        XCTAssertTrue(tab.command.contains("--model-metadata-file ~/.mlx-serve/aider/model-metadata.json"))
        XCTAssertTrue(tab.command.contains("\"max_input_tokens\": 90112"))
    }

    func testAiderMetadataDeclaresEveryChatEntryWithItsOwnBudget() throws {
        let entries = [
            AgentModelEntry(id: "m1", budget: .init(context: 4096, output: 1024), vision: false),
            AgentModelEntry(id: "m2", budget: .init(context: 262144, output: 65536), vision: true),
        ]
        let json = AgentConfigs.aiderModelMetadataJSON(
            model: "m1", budget: .init(context: 4096, output: 1024), entries: entries)
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(json.utf8)) as? [String: [String: Any]])
        XCTAssertEqual(obj["openai/m1"]?["max_input_tokens"] as? Int, 4096)
        XCTAssertEqual(obj["openai/m2"]?["max_input_tokens"] as? Int, 262144)
        XCTAssertEqual(obj["openai/m2"]?["max_output_tokens"] as? Int, 65536)
        XCTAssertEqual(obj["openai/m1"]?["litellm_provider"] as? String, "openai")
    }

    /// A heredoc body containing its own delimiter line would truncate the
    /// config silently — assert the builders never emit one.
    func testHeredocBodiesNeverContainTheDelimiterLine() {
        for tab in tabs where tab.command.contains("<<'EOF'") {
            let body = tab.command
                .components(separatedBy: "<<'EOF'\n")[1]
                .components(separatedBy: "\nEOF")[0]
            XCTAssertFalse(body.split(separator: "\n").contains("EOF"), tab.id)
        }
    }

    /// The tray shows the one-click launcher where it can (DMG) and the
    /// instructions panel where it can't (MAS) — never both, never neither.
    func testInstructionsPanelReplacesTheLauncherExactlyWhereLaunchingIsGone() {
        XCTAssertTrue(CLISetupInstructions.replacesLauncher(features: .mas))
        XCTAssertFalse(CLISetupInstructions.replacesLauncher(features: .developerID))
    }
}
