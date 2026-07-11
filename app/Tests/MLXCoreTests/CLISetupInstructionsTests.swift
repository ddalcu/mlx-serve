import XCTest
@testable import MLXCore

final class CLISetupInstructionsTests: XCTestCase {

    private let budget = AgentBudget.Budget(context: 90112, output: 16384)
    private var tabs: [CLISetupInstructions.Tab] {
        CLISetupInstructions.tabs(baseURL: "http://localhost:11234",
                                  servedModelId: "gemma-4-e4b-it-4bit",
                                  budget: budget)
    }

    func testThreeTabsWithStableIdsInLauncherOrder() {
        XCTAssertEqual(tabs.map(\.id), ["claude", "pi", "opencode"],
                       "same CLIs, same order as the DMG launcher dropdown")
        for tab in tabs {
            XCTAssertFalse(tab.command.isEmpty, tab.id)
            XCTAssertFalse(tab.installHint.isEmpty, tab.id)
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
                                               "cd '/tmp'", budget)
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
                                                     "gemma-4-e4b-it-4bit", "cd '/tmp'", budget)
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", model: "gemma-4-e4b-it-4bit", budget: budget)
        XCTAssertTrue(script.contains("export OPENCODE_CONFIG_CONTENT='\(json)'"), script)
        XCTAssertTrue(script.contains("opencode --model mlx/gemma-4-e4b-it-4bit"))
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
