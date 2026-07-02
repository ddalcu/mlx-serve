import XCTest
import Combine
@testable import MLXCore

/// Pure-logic tests for the sandbox manager's host↔guest mapping and provisioning
/// decisions. No guest, no VM — safe in CI.
final class AgentSandboxTests: XCTestCase {

    // MARK: image ref -> cache dir name

    func testImageDirNameSanitizesRefs() {
        XCTAssertEqual(AgentSandbox.imageDirName("nikolaik/python-nodejs"), "nikolaik_python-nodejs")
        XCTAssertEqual(AgentSandbox.imageDirName("alpine:3.20"), "alpine_3.20")
        XCTAssertEqual(AgentSandbox.imageDirName("ghcr.io/acme/img:tag"), "ghcr.io_acme_img_tag")
    }

    func testGuestArchIsArm64() {
        // The HVF guest on Apple Silicon is arm64; pulling amd64 boots to an
        // ENOEXEC kernel panic ("No working init"). Regression guard for that.
        XCTAssertEqual(AgentSandbox.guestArch, "arm64")
        XCTAssertTrue(AgentSandbox.archMarkerName().contains("arm64"),
                      "the cache marker must encode the arch so a wrong-arch cache is invalidated")
    }

    func testImageDirNameNeverEscapesWithDotsOrSlashes() {
        // Must not produce path traversal into a parent dir.
        let name = AgentSandbox.imageDirName("../../etc/passwd")
        XCTAssertFalse(name.contains("/"))
        XCTAssertFalse(name.contains(".."))
    }

    // MARK: guest background-process control (kill / log tail)

    func testGuestKillCommandTargetsPidWithGraceKill() {
        let cmd = AgentSandbox.guestKillCommand(pid: 4242)
        XCTAssertTrue(cmd.contains("kill -TERM 4242"), cmd)
        XCTAssertTrue(cmd.contains("kill -KILL 4242"), "must escalate to SIGKILL after a grace: \(cmd)")
        XCTAssertTrue(cmd.contains("&"), "the grace kill must be backgrounded so the exec returns: \(cmd)")
    }

    func testGuestReadLogCommandTailsQuotedPath() {
        let cmd = AgentSandbox.guestReadLogCommand(logPath: "/tmp/mlx-bg-1.log")
        XCTAssertTrue(cmd.contains("tail -c"), cmd)
        XCTAssertTrue(cmd.contains("'/tmp/mlx-bg-1.log'"), "path must be shell-quoted: \(cmd)")
    }

    /// With no live guest booted, the guest-kill / log-tail helpers are safe
    /// no-ops (they must never boot a guest just to kill/read).
    func testGuestControlHelpersAreNoopWithoutLiveGuest() async {
        AgentSandbox.shared.killGuestProcess(pid: 12345) // must not crash / boot
        let log = await AgentSandbox.shared.tailGuestLog(logPath: "/tmp/nope.log")
        XCTAssertNil(log, "no live guest → nil log (never boots one)")
    }

    // MARK: fallback shared root (no working directory supplied)

    func testFallbackSharedRootIsTheSessionWorkspaceNeverHomeOrCwd() {
        // When no working directory is supplied (e.g. the Sandbox Terminal
        // booting the guest before any agent command), the guest must share the
        // app's default session workspace — NOT the user's home folder (the old
        // libcontain-era fallback) and NOT the process cwd (`/` for a
        // Finder-launched app). Mounting either rw into the guest defeats the
        // point of the sandbox.
        let root = AgentSandbox.fallbackSharedRoot()
        XCTAssertEqual(root, ChatSession.defaultWorkingDirectory)
        XCTAssertTrue(root.hasSuffix(".mlx-serve/workspace"))
        XCTAssertNotEqual(root, FileManager.default.homeDirectoryForCurrentUser.path)
        XCTAssertNotEqual(root, "/")
    }

    // MARK: host cwd -> guest /workspace path

    func testGuestPathAtRoot() {
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/proj", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
        XCTAssertTrue(r.mapped)
    }

    func testGuestPathForSubdir() {
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/proj/src/app", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace/src/app")
        XCTAssertTrue(r.mapped)
    }

    func testGuestPathOutsideRootFallsBackToWorkspaceRoot() {
        let r = AgentSandbox.guestPath(hostPath: "/tmp/elsewhere", sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
        XCTAssertFalse(r.mapped, "a cwd outside the shared root isn't visible in the guest")
    }

    func testGuestPathNilHostFallsBackToWorkspaceRoot() {
        let r = AgentSandbox.guestPath(hostPath: nil, sharedRoot: "/Users/d/proj")
        XCTAssertEqual(r.path, "/workspace")
    }

    func testGuestPathDoesNotPartialSegmentMatch() {
        // "/Users/d/project2" must NOT be treated as under "/Users/d/proj".
        let r = AgentSandbox.guestPath(hostPath: "/Users/d/project2", sharedRoot: "/Users/d/proj")
        XCTAssertFalse(r.mapped)
        XCTAssertEqual(r.path, "/workspace")
    }

    // MARK: workspace remount when the session working folder moves

    func testNeedsRemountOnlyWhenCwdLeavesTheSharedRoot() {
        // Same root / subdir / no cwd → the live guest keeps its share.
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: nil),
                       "no cwd supplied → keep the current share")
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/proj"))
        XCTAssertFalse(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/proj/src/deep"))
        // The user switched the session's working folder → /workspace must
        // follow it (guest reboots with the new share; boot is sub-second).
        XCTAssertTrue(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/other"))
        XCTAssertTrue(AgentSandbox.needsRemount(sharedRoot: "/a/proj", requestedCwd: "/a/project2"),
                      "sibling sharing a path prefix is NOT under the root")
    }

    // MARK: command wrapping (cd into mapped dir, then run)

    func testWrapCommandCdsIntoGuestPath() {
        let wrapped = AgentSandbox.wrap(command: "ls -la", guestCwd: "/workspace/src")
        XCTAssertTrue(wrapped.hasPrefix("cd '/workspace/src'"), "must cd into the mapped dir first")
        XCTAssertTrue(wrapped.contains("ls -la"))
    }

    func testWrapEscapesApostropheInGuestCwd() {
        // An apostrophe in the working-directory name must not unbalance the
        // single-quoting — an unescaped `'` desyncs the ShellSentinel framing
        // and every subsequent command times out.
        let wrapped = AgentSandbox.wrap(command: "ls", guestCwd: "/workspace/it's here")
        XCTAssertTrue(wrapped.contains("cd '/workspace/it'\\''s here'"),
                      "guest cwd must be POSIX single-quote escaped: \(wrapped)")
        // Balanced: quote count net-zero — dash must parse it as one word.
        // '/workspace/it'  \'  's here'  → 6 quotes + 1 escaped = odd raw count
        // is fine; the load-bearing check is the '\'' escape sequence above and
        // that no bare `'s ` (unescaped apostrophe run) survives.
        XCTAssertFalse(wrapped.contains("cd '/workspace/it's here'"),
                       "raw unescaped interpolation must be gone: \(wrapped)")
    }

    // MARK: teardown must not block the caller (main-thread call sites)

    func testTeardownDetachesGuestSynchronouslyAndShutsDownOffThread() {
        // configure()/teardown() run on the main thread (Settings didSet fires
        // per keystroke; Sandbox Terminal Stop button). VzGuest.shutdown blocks
        // up to 10s on a semaphore — teardown must detach the guest reference
        // synchronously and push the blocking stop to a background queue.
        let sandbox = AgentSandbox.shared
        sandbox._testInstallGuest(VzGuest())
        XCTAssertTrue(sandbox._testHasGuest)

        let shutdownRan = expectation(description: "blocking shutdown ran off-thread")
        let start = Date()
        sandbox.teardown(shutdownBlocking: { _ in
            Thread.sleep(forTimeInterval: 2) // simulate the VZ stop wait
            shutdownRan.fulfill()
        })
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 1.0, "teardown blocked the caller for \(elapsed)s")
        XCTAssertFalse(sandbox._testHasGuest, "guest must be detached synchronously so a fresh boot is safe")
        wait(for: [shutdownRan], timeout: 5)
    }

    // MARK: transcript store (tray redraw churn)

    func testTranscriptStoreCapsAt400Entries() {
        let store = SandboxTranscript()
        for i in 0..<405 {
            store.append(AgentSandbox.Entry(source: .agent, command: "cmd\(i)", output: "",
                                            exitCode: 0, at: Date()))
        }
        XCTAssertEqual(store.entries.count, 400)
        XCTAssertEqual(store.entries.first?.command, "cmd5", "cap must drop the OLDEST entries")
        XCTAssertEqual(store.entries.last?.command, "cmd404")
    }

    func testTranscriptAppendsDoNotPublishThroughAgentSandbox() {
        // The tray menu observes AgentSandbox.shared; per-command transcript
        // appends must publish ONLY through the nested transcript store, or
        // every append re-renders the whole tray (MenuBarExtra churn class).
        let sandbox = AgentSandbox.shared
        var sandboxFired = 0
        var storeFired = 0
        var subs = Set<AnyCancellable>()
        sandbox.objectWillChange.sink { _ in sandboxFired += 1 }.store(in: &subs)
        sandbox.transcriptStore.objectWillChange.sink { _ in storeFired += 1 }.store(in: &subs)

        sandbox.transcriptStore.append(AgentSandbox.Entry(source: .user, command: "echo hi",
                                                          output: "hi", exitCode: 0, at: Date()))
        XCTAssertEqual(storeFired, 1, "the terminal's store must publish the append")
        XCTAssertEqual(sandboxFired, 0, "AgentSandbox itself must NOT publish transcript appends")
    }
}
