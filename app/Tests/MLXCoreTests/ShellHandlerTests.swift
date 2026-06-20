import XCTest
@testable import MLXCore

// Regression tests for the agentic shell tool. The original handler had no
// stdin redirection and a kill path that blocked on `readDataToEndOfFile`, so
// an interactive scaffolder (`npx sv create`) could hang the agent for 100s+.
final class ShellHandlerTests: XCTestCase {

    func testEchoSucceeds() async throws {
        let out = try await ShellHandler().execute(parameters: ["command": "echo hello"], workingDirectory: nil)
        XCTAssertTrue(out.contains("hello"), out)
        XCTAssertFalse(out.contains("timed out"), out)
        XCTAssertFalse(out.contains("exit code"), out)
    }

    func testChildStdinIsEmpty() async throws {
        // Contract: the child's stdin is /dev/null, so a command that consumes
        // stdin sees zero bytes (and an interactive prompt would hit EOF).
        let out = try await ShellHandler().execute(parameters: ["command": "wc -c"], workingDirectory: nil)
        // `wc -c` over /dev/null prints 0.
        XCTAssertTrue(out.contains("0"), out)
        XCTAssertFalse(out.contains("timed out"), out)
    }

    func testTimeoutKillsHangingCommandQuickly() async throws {
        let start = Date()
        let out = try await ShellHandler(timeoutSeconds: 2).execute(parameters: ["command": "sleep 20"], workingDirectory: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(out.contains("timed out"), out)
        // Must be bounded by the timeout, not the 20s sleep.
        XCTAssertLessThan(elapsed, 10, "shell did not honor the timeout (elapsed \(elapsed)s)")
    }

    func testNonZeroExitReported() async throws {
        let out = try await ShellHandler().execute(parameters: ["command": "exit 7"], workingDirectory: nil)
        XCTAssertTrue(out.contains("[exit code: 7]"), out)
    }

    func testRunsInWorkingDirectory() async throws {
        let tmp = NSTemporaryDirectory()
        let out = try await ShellHandler().execute(parameters: ["command": "pwd"], workingDirectory: tmp)
        let resolved = (tmp as NSString).standardizingPath
        XCTAssertTrue(out.contains(resolved) || out.contains(tmp), out)
    }

    func testMissingCommandThrows() async {
        do {
            _ = try await ShellHandler().execute(parameters: [:], workingDirectory: nil)
            XCTFail("expected missingParameter")
        } catch {
            // expected
        }
    }

    // MARK: - Background execution (run_in_background) + adopt backstop

    /// run_in_background returns immediately with a handle while the process
    /// keeps running — the canonical "serve this folder" fix.
    @MainActor
    func testRunInBackgroundReturnsImmediatelyAndStaysAlive() async throws {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let start = Date()
        let out = try await ShellHandler(registry: reg).execute(
            parameters: ["command": "sleep 5", "run_in_background": "true"], workingDirectory: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 2, "background start must return promptly (elapsed \(elapsed)s)")
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"), "background process must keep running")
    }

    /// Without a registry the flag degrades to a graceful error — no crash, no
    /// orphaned process.
    func testRunInBackgroundWithoutRegistryIsGraceful() async throws {
        let out = try await ShellHandler().execute(
            parameters: ["command": "sleep 1", "run_in_background": "true"], workingDirectory: nil)
        XCTAssertTrue(out.lowercased().contains("background"), out)
        XCTAssertFalse(out.contains("bg1"), out)
    }

    /// Backstop: a foreground command still alive at the timeout is ADOPTED as a
    /// managed background process — reported as such, NOT killed.
    @MainActor
    func testForegroundTimeoutAdoptsInsteadOfKilling() async throws {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let out = try await ShellHandler(timeoutSeconds: 1, registry: reg).execute(
            parameters: ["command": "sleep 30"], workingDirectory: nil)
        XCTAssertTrue(out.contains("managed in the background"), out)
        XCTAssertTrue(out.contains("NOT killed"), out)
        XCTAssertFalse(out.contains("timed out"), out)
        XCTAssertEqual(reg.list(sessionId: nil).count, 1, "the live process must be adopted")
        let handle = reg.list(sessionId: nil)[0].handle
        XCTAssertTrue(reg.isAlive(handle: handle), "adopted process must still be alive")
    }

    // MARK: - Model-agnostic flag + `&` handling (no prompt-following required)

    func testIsTruthyFlag() {
        for yes in ["true", "True", "TRUE", "1", "yes", "Y", "on", " true "] {
            XCTAssertTrue(ShellHandler.isTruthyFlag(yes), "\(yes) should be truthy")
        }
        for no in ["false", "0", "no", "", "  ", "tru"] {
            XCTAssertFalse(ShellHandler.isTruthyFlag(no), "\(no) should be falsy")
        }
        XCTAssertFalse(ShellHandler.isTruthyFlag(nil))
    }

    /// Models send `run_in_background: 1` (stringified to "1") as often as
    /// `"true"`. It must still background — not fall into the foreground/timeout
    /// path. Regression for the live "simmering 20s" stuck card.
    @MainActor
    func testNumericTruthyFlagBackgrounds() async throws {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let start = Date()
        let out = try await ShellHandler(registry: reg).execute(
            parameters: ["command": "sleep 30", "run_in_background": "1"], workingDirectory: nil)
        XCTAssertLessThan(Date().timeIntervalSince(start), 3, "run_in_background:1 must background, not block")
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"))
    }

    func testHasTrailingBackgroundOperator() {
        XCTAssertTrue(ShellHandler.hasTrailingBackgroundOperator("cmd &"))
        XCTAssertTrue(ShellHandler.hasTrailingBackgroundOperator("cmd &   "))
        XCTAssertFalse(ShellHandler.hasTrailingBackgroundOperator("cmd"))
        XCTAssertFalse(ShellHandler.hasTrailingBackgroundOperator("a && b"), "&& is logical-AND, not background")
        XCTAssertFalse(ShellHandler.hasTrailingBackgroundOperator("a & b"), "inner & is not a trailing operator")
    }

    func testStripTrailingBackgroundOperator() {
        XCTAssertEqual(ShellHandler.stripTrailingBackgroundOperator("python3 -m http.server 8080 &"),
                       "python3 -m http.server 8080")
        XCTAssertEqual(ShellHandler.stripTrailingBackgroundOperator("  cmd &  "), "cmd")
        XCTAssertEqual(ShellHandler.stripTrailingBackgroundOperator("cmd"), "cmd")
        XCTAssertEqual(ShellHandler.stripTrailingBackgroundOperator("a && b"), "a && b")
        XCTAssertEqual(ShellHandler.stripTrailingBackgroundOperator("a & b"), "a & b")
    }

    /// A small model that writes `… &` with NO run_in_background flag must still
    /// be auto-tracked — and the tracked pid must be the LIVE process (strip), not
    /// an instantly-exited backgrounding shell.
    @MainActor
    func testTrailingAmpersandAutoRoutesToLiveBackgroundProcess() async throws {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let start = Date()
        let out = try await ShellHandler(registry: reg).execute(
            parameters: ["command": "sleep 30 &"], workingDirectory: nil)
        XCTAssertLessThan(Date().timeIntervalSince(start), 3, "auto-background should return promptly")
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"),
                      "tracked pid must be the live process, not an exited backgrounding shell")
    }

    /// A capable model that sets the flag AND redundantly adds `&` must not orphan
    /// the process either.
    @MainActor
    func testRedundantAmpersandWithFlagStaysTracked() async throws {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let out = try await ShellHandler(registry: reg).execute(
            parameters: ["command": "sleep 30 &", "run_in_background": "true"], workingDirectory: nil)
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"), "redundant & must not orphan the tracked process")
    }
}
