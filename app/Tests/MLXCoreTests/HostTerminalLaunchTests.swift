import XCTest
@testable import MLXCore

/// Host CLIs (Claude Code, pi, opencode, …) open in an embedded terminal row,
/// not in Terminal.app: the launcher hands back a command the terminal can
/// spawn — a login+interactive zsh running the same script it used to hand to
/// Terminal.app, so PATH resolves exactly as detection saw it.
final class HostTerminalLaunchTests: XCTestCase {

    @MainActor
    func testLaunchCommandIsAnInteractiveLoginShellOverTheWrittenScript() throws {
        let cmd = CLILauncher.launchCommand(.claudeCode, baseURL: "http://127.0.0.1:8080",
                                            servedModelId: "m", budget: AgentBudget.fallback,
                                            entries: [], workingDirectory: "/tmp/work dir")
        XCTAssertEqual(cmd.executable, "/bin/zsh")
        XCTAssertEqual(Array(cmd.args.prefix(2)), ["-l", "-i"], "rc files are where PATH lives")
        let path = try XCTUnwrap(cmd.args.last)
        let script = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(script.contains("cd '/tmp/work dir'"), script)
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: path))
    }

    func testHostRowsAreTheirOwnKind() {
        var m = TerminalSessionList()
        let a = m.addPreparing(label: "Claude Code", agentId: nil, workspace: "/a", kind: .host)
        XCTAssertEqual(m.session(a)?.kind, .host)
        let b = m.addPreparing(label: "pi", agentId: "pi", workspace: "/a")
        XCTAssertEqual(m.session(b)?.kind, .sandbox, "sandbox is the default kind")
    }
}
