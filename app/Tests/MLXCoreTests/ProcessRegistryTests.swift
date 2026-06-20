import XCTest
import Darwin
@testable import MLXCore

/// The @MainActor registry that owns agent-spawned background processes. These
/// spawn real short-lived OS processes; every test reaps via killAll().
@MainActor
final class ProcessRegistryTests: XCTestCase {

    /// Poll a condition for up to `timeout` seconds, yielding to let the
    /// background readability/termination handlers run.
    private func waitUntil(_ timeout: Double = 3.0, _ cond: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if cond() { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func testStartRegistersAndReturnsRunningHandle() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let p = reg.start(command: "sleep 5", workingDirectory: nil, sessionId: nil)
        XCTAssertEqual(p.handle, "bg1")
        XCTAssertEqual(p.status, .running)
        XCTAssertTrue(reg.isAlive(handle: "bg1"))
        XCTAssertEqual(reg.list(sessionId: nil).count, 1)
    }

    func testReadOutputCapturesStdout() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let p = reg.start(command: "echo captured-line; exec sleep 5", workingDirectory: nil, sessionId: nil)
        await waitUntil { p.output.snapshot().contains("captured-line") }
        let out = reg.readOutput(handle: p.handle) ?? ""
        XCTAssertTrue(out.contains("captured-line"), "stdout not captured: \(out)")
    }

    func testKillFlipsStatusAndAliveness() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let p = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: nil)
        XCTAssertTrue(reg.isAlive(handle: p.handle))
        reg.kill(handle: p.handle)
        // kill() flips status synchronously — no need to wait for the OS.
        XCTAssertFalse(reg.isAlive(handle: p.handle))
        XCTAssertEqual(p.status, .killed)
    }

    func testListFiltersBySession() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let a = UUID(); let b = UUID()
        _ = reg.start(command: "sleep 5", workingDirectory: nil, sessionId: a)
        _ = reg.start(command: "sleep 5", workingDirectory: nil, sessionId: a)
        _ = reg.start(command: "sleep 5", workingDirectory: nil, sessionId: b)
        XCTAssertEqual(reg.list(sessionId: a).count, 2)
        XCTAssertEqual(reg.list(sessionId: b).count, 1)
        XCTAssertEqual(reg.list(sessionId: nil).count, 3, "nil session lists all")
    }

    func testKillSessionScopesToThatSession() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let a = UUID(); let b = UUID()
        let pa = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: a)
        let pb = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: b)
        reg.killSession(a)
        XCTAssertFalse(reg.isAlive(handle: pa.handle))
        XCTAssertTrue(reg.isAlive(handle: pb.handle), "other session must be untouched")
    }

    func testKillAllStopsEverything() async {
        let reg = ProcessRegistry()
        let p1 = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: nil)
        let p2 = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: nil)
        reg.killAll()
        XCTAssertFalse(reg.isAlive(handle: p1.handle))
        XCTAssertFalse(reg.isAlive(handle: p2.handle))
    }

    func testUnknownHandleIsGraceful() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        XCTAssertNil(reg.readOutput(handle: "nope"))
        XCTAssertFalse(reg.isAlive(handle: "nope"))
        reg.kill(handle: "nope") // must not crash
    }

    /// Killing a tracked process must take down the children it spawned — the
    /// `npm run dev → node/esbuild` / `cd x && server` case — not just the leader.
    func testKillTakesDownChildProcesses() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        let pidFile = NSTemporaryDirectory() + "mlxtest_child_\(UUID().uuidString).pid"
        // Leader (zsh) backgrounds a child sleep, records its pid, then waits —
        // so the leader stays alive and the child is a live descendant at kill time.
        let p = reg.start(command: "sleep 300 & echo $! > \(pidFile); wait",
                          workingDirectory: nil, sessionId: nil)
        await waitUntil(5) {
            guard let s = try? String(contentsOfFile: pidFile) else { return false }
            return !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        let childPid = Int32((try? String(contentsOfFile: pidFile))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "") ?? 0
        XCTAssertGreaterThan(childPid, 0, "failed to capture the child pid")
        XCTAssertEqual(Darwin.kill(childPid, 0), 0, "child should be alive before kill")

        reg.kill(handle: p.handle)
        await waitUntil(5) { Darwin.kill(childPid, 0) != 0 }
        XCTAssertNotEqual(Darwin.kill(childPid, 0), 0, "child must be killed along with its parent")
        try? FileManager.default.removeItem(atPath: pidFile)
    }

    func testAdoptedProcessIsManagedAndAlive() async {
        let reg = ProcessRegistry()
        defer { reg.killAll() }
        // Build a live process the way the foreground backstop would hand it over.
        let proc = ProcessRegistry.makeProcess(command: "sleep 30", workingDirectory: nil)
        try? proc.run()
        let p = reg.adopt(process: proc, command: "sleep 30", workingDirectory: nil,
                          sessionId: nil, priorOutput: "earlier output\n")
        XCTAssertEqual(p.status, .running)
        XCTAssertTrue(reg.isAlive(handle: p.handle))
        XCTAssertTrue((reg.readOutput(handle: p.handle) ?? "").contains("earlier output"),
                      "priorOutput must seed the buffer")
    }
}
