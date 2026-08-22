import XCTest
@testable import MLXCore

final class CLISetupInstructionsTests: XCTestCase {

    private let budget = AgentBudget.Budget(context: 90112, output: 16384)
    private var tabs: [CLISetupInstructions.Tab] {
        CLISetupInstructions.tabs(baseURL: "http://localhost:11234",
                                  servedModelId: "gemma-4-e4b-it-4bit",
                                  budget: budget)
    }

    func testTabsHaveStableIdsInLauncherOrder() {
        XCTAssertEqual(tabs.map(\.id),
                       ["claude", "pi", "omp", "opencode", "codex", "hermes", "aider"],
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
        for cli in [LauncherCLI.claudeCode, .pi, .omp, .opencode, .hermes, .aider] {
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
