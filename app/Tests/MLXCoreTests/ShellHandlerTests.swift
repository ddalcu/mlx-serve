import XCTest
@testable import MLXCore

// Regression tests for the agentic shell tool. The original handler had no
// stdin redirection and a kill path that blocked on `readDataToEndOfFile`, so
// an interactive scaffolder (`npx sv create`) could hang the agent for 100s+.
final class ShellHandlerTests: XCTestCase {

    func testEchoSucceeds() async throws {
        let out = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: ["command": "echo hello"], workingDirectory: nil)
        XCTAssertTrue(out.contains("hello"), out)
        XCTAssertFalse(out.contains("timed out"), out)
        XCTAssertFalse(out.contains("exit code"), out)
    }

    func testChildStdinIsEmpty() async throws {
        // Contract: the child's stdin is /dev/null, so a command that consumes
        // stdin sees zero bytes (and an interactive prompt would hit EOF).
        let out = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: ["command": "wc -c"], workingDirectory: nil)
        // `wc -c` over /dev/null prints 0.
        XCTAssertTrue(out.contains("0"), out)
        XCTAssertFalse(out.contains("timed out"), out)
    }

    func testTimeoutKillsHangingCommandQuickly() async throws {
        let start = Date()
        let out = try await ShellHandler(timeoutSeconds: 2, hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: ["command": "sleep 20"], workingDirectory: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(out.contains("timed out"), out)
        // Must be bounded by the timeout, not the 20s sleep.
        XCTAssertLessThan(elapsed, 10, "shell did not honor the timeout (elapsed \(elapsed)s)")
    }

    func testNonZeroExitReported() async throws {
        let out = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: ["command": "exit 7"], workingDirectory: nil)
        XCTAssertTrue(out.contains("[exit code: 7]"), out)
    }

    func testRunsInWorkingDirectory() async throws {
        let tmp = NSTemporaryDirectory()
        let out = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: ["command": "pwd"], workingDirectory: tmp)
        let resolved = (tmp as NSString).standardizingPath
        XCTAssertTrue(out.contains(resolved) || out.contains(tmp), out)
    }

    func testMissingCommandThrows() async {
        do {
            _ = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(parameters: [:], workingDirectory: nil)
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
        let out = try await ShellHandler(registry: reg, hostShellAllowed: true, sandboxEnabled: { false }).execute(
            parameters: ["command": "sleep 5", "run_in_background": "true"], workingDirectory: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 2, "background start must return promptly (elapsed \(elapsed)s)")
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"), "background process must keep running")
    }

    /// Without a registry the flag degrades to a graceful error — no crash, no
    /// orphaned process.
    func testRunInBackgroundWithoutRegistryIsGraceful() async throws {
        let out = try await ShellHandler(hostShellAllowed: true, sandboxEnabled: { false }).execute(
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
        let out = try await ShellHandler(timeoutSeconds: 1, registry: reg, hostShellAllowed: true, sandboxEnabled: { false }).execute(
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
        let out = try await ShellHandler(registry: reg, hostShellAllowed: true, sandboxEnabled: { false }).execute(
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
        let out = try await ShellHandler(registry: reg, hostShellAllowed: true, sandboxEnabled: { false }).execute(
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
        let out = try await ShellHandler(registry: reg, hostShellAllowed: true, sandboxEnabled: { false }).execute(
            parameters: ["command": "sleep 30 &", "run_in_background": "true"], workingDirectory: nil)
        XCTAssertTrue(out.contains("bg1"), out)
        XCTAssertTrue(reg.isAlive(handle: "bg1"), "redundant & must not orphan the tracked process")
    }

    // MARK: - Sandbox routing (background must never escape to the host)

    /// With the Agent Sandbox ON, a background-flagged command must NEVER reach
    /// the host ProcessRegistry — that executes on the host and defeats the
    /// isolation promise (the agent prompt actively suggests run_in_background).
    /// The App Store build has no host shell, so EVERY command routes into the
    /// guest regardless of the user's sandbox toggle.
    func testRouteWithoutHostShellForcesGuest() {
        for wantsBackground in [true, false] {
            let route = ShellHandler.route(sandboxEnabled: false, wantsBackground: wantsBackground,
                                           hasTrailingAmp: false, hasRegistry: true,
                                           hostShellAllowed: false)
            XCTAssertEqual(route, wantsBackground ? .sandboxBackground : .sandboxForeground,
                           "no host shell must never route to a host process")
        }
    }

    func testRouteSandboxOnNeverYieldsHostBackground() {
        for amp in [false, true] {
            for hasRegistry in [false, true] {
                XCTAssertEqual(
                    ShellHandler.route(sandboxEnabled: true, wantsBackground: true,
                                       hasTrailingAmp: amp, hasRegistry: hasRegistry),
                    .sandboxBackground,
                    "flagged background with sandbox on must run inside the guest (amp=\(amp) registry=\(hasRegistry))")
            }
        }
        // A bare trailing `&` routes to the managed guest background path, same
        // as it routes to the managed HOST background path when the sandbox is
        // off. Under vsock a foreground `cmd &` would otherwise WAIT on the
        // orphan's inherited stdout pipe until the timeout kills the call —
        // the legacy console shell returned promptly, so this only became
        // visible with the exec transport.
        XCTAssertEqual(ShellHandler.route(sandboxEnabled: true, wantsBackground: false,
                                          hasTrailingAmp: true, hasRegistry: true),
                       .sandboxBackground)
        XCTAssertEqual(ShellHandler.route(sandboxEnabled: true, wantsBackground: false,
                                          hasTrailingAmp: false, hasRegistry: true),
                       .sandboxForeground)
    }

    /// Sandbox off keeps today's host behavior exactly.
    func testRouteSandboxOffPreservesHostBehavior() {
        // Developer ID: a host shell exists. (The App Store build has none — see
        // testRouteWithoutHostShellForcesGuest.)
        func route(sandboxEnabled: Bool, wantsBackground: Bool, hasTrailingAmp: Bool, hasRegistry: Bool) -> ShellHandler.ShellRoute {
            ShellHandler.route(sandboxEnabled: sandboxEnabled, wantsBackground: wantsBackground,
                               hasTrailingAmp: hasTrailingAmp, hasRegistry: hasRegistry, hostShellAllowed: true)
        }
        XCTAssertEqual(route(sandboxEnabled: false, wantsBackground: true,
                                          hasTrailingAmp: false, hasRegistry: true),
                       .hostBackground)
        XCTAssertEqual(route(sandboxEnabled: false, wantsBackground: false,
                                          hasTrailingAmp: true, hasRegistry: true),
                       .hostBackground)
        XCTAssertEqual(route(sandboxEnabled: false, wantsBackground: true,
                                          hasTrailingAmp: false, hasRegistry: false),
                       .hostBackgroundUnavailable,
                       "explicit flag with no registry keeps the graceful error")
        XCTAssertEqual(route(sandboxEnabled: false, wantsBackground: false,
                                          hasTrailingAmp: true, hasRegistry: false),
                       .hostForeground,
                       "bare & with no registry runs foreground, as before")
        XCTAssertEqual(route(sandboxEnabled: false, wantsBackground: false,
                                          hasTrailingAmp: false, hasRegistry: true),
                       .hostForeground)
    }

    func testSandboxBackgroundCommandWrapsDetachedWithLogAndEchoesPid() {
        let wrapped = ShellHandler.sandboxBackgroundCommand("python3 -m http.server 8080",
                                                            logPath: "/tmp/mlx-bg-1.log")
        XCTAssertEqual(wrapped,
            "(python3 -m http.server 8080) </dev/null >>/tmp/mlx-bg-1.log 2>&1 & echo __CTN_BGPID=$!")
    }

    func testSandboxBackgroundLogPathIsUniquePerInvocation() {
        let a = ShellHandler.sandboxBackgroundLogPath(now: Date(timeIntervalSince1970: 1))
        let b = ShellHandler.sandboxBackgroundLogPath(now: Date(timeIntervalSince1970: 2))
        XCTAssertTrue(a.hasPrefix("/tmp/mlx-bg-"), a)
        XCTAssertTrue(a.hasSuffix(".log"), a)
        XCTAssertNotEqual(a, b, "each invocation needs its own guest log file")
    }

    func testParseSandboxBackgroundPID() {
        // The marker line the wrapper echoes, wrapped in the completed-message shape.
        XCTAssertEqual(ShellHandler.parseSandboxBackgroundPID("[cwd: /w]\n__CTN_BGPID=4242\n"), 4242)
        XCTAssertEqual(ShellHandler.parseSandboxBackgroundPID("__CTN_BGPID=7"), 7)
        XCTAssertNil(ShellHandler.parseSandboxBackgroundPID("no marker here"))
        XCTAssertNil(ShellHandler.parseSandboxBackgroundPID("__CTN_BGPID=\n"), "empty pid → nil")
    }

    /// The guest-backed registration seam: given the parsed guest pid it registers
    /// a SANDBOX process, surfaces its handle on the handleBox (drives the card's
    /// running badge + kill X), and the message names the handle + log — no live
    /// VM required.
    @MainActor
    func testRegisterSandboxBackgroundSetsHandleBoxAndRegisters() async {
        let reg = ProcessRegistry(); defer { reg.killAll() }
        let box = ProcessHandleBox()
        let handler = ShellHandler(registry: reg, handleBox: box, hostShellAllowed: true, sandboxEnabled: { false })
        let msg = await handler.registerSandboxBackground(
            command: "python3 -m http.server 8080", guestPID: 4242,
            logPath: "/tmp/mlx-bg-1.log", cwd: "/work")
        XCTAssertEqual(box.handle, "bg1", "the guest bg process must be surfaced on the handleBox")
        XCTAssertTrue(msg.contains("bg1"), msg)
        XCTAssertTrue(msg.contains("/tmp/mlx-bg-1.log"), msg)
        XCTAssertTrue(msg.contains("readProcessOutput") && msg.contains("killProcess"), msg)
        let entry = reg.list(sessionId: nil).first
        XCTAssertEqual(entry?.pid, 4242, "the guest pid must be tracked")
        XCTAssertTrue(entry?.isSandboxed ?? false, "the entry must be a sandbox (guest-backed) process")
        XCTAssertTrue(reg.isAlive(handle: "bg1"))
    }

    /// With no registry (older call sites / unit tests) the sandbox background
    /// path degrades to the log-only message — no handle, no crash.
    func testRegisterSandboxBackgroundWithoutRegistryFallsBack() async {
        let box = ProcessHandleBox()
        let handler = ShellHandler(handleBox: box, hostShellAllowed: true, sandboxEnabled: { false })
        let msg = await handler.registerSandboxBackground(
            command: "srv", guestPID: 5, logPath: "/tmp/x.log", cwd: "/work")
        XCTAssertNil(box.handle)
        XCTAssertTrue(msg.contains("/tmp/x.log"), msg)
        XCTAssertFalse(msg.contains("readProcessOutput"), "no handle → no poll/kill guidance: \(msg)")
    }
}
