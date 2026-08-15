import XCTest
import Foundation
import System
import MCP
@testable import MLXCore

/// Regression for a live CPU leak (2026-08-15): `MCPManager`'s stdio connect race
/// (`connectOrFailFast`) resumes its outer continuation when it loses to the death-watcher or the
/// hard-cap timeout, but historically never told the ABANDONED `client.connect(transport:)` Task to
/// stop — that Task's `withCheckedThrowingContinuation` is only ever unstuck by an explicit
/// `Client.disconnect()` call (confirmed against `swift-sdk`'s `Client.swift`), and nothing called
/// it on the losing paths. Symptom: a live MLX Core session sat at ~160% average CPU over 23h;
/// `sample <pid>` showed two threads permanently parked inside `Client.connect(transport:)` — one
/// per failed/timed-out MCP server connect attempt, each leaking forever.
///
/// This test forces the hard-cap path (a real child process that accepts the stdio pipes but never
/// speaks MCP, so the death-watcher never fires) and asserts the abandoned Path A task actually
/// settles afterward, instead of running forever.
final class MCPConnectLeakTests: XCTestCase {

    func testAbandonedConnectSettlesAfterLosingTheRace() async throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", "sleep 60"]
        let stdin = Pipe(), stdout = Pipe()
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = Pipe()
        try process.run()
        defer { if process.isRunning { process.terminate() } }

        let transport = StdioTransport(
            input: FileDescriptor(rawValue: stdout.fileHandleForReading.fileDescriptor),
            output: FileDescriptor(rawValue: stdin.fileHandleForWriting.fileDescriptor))
        let client = Client(name: "mcp-leak-test", version: "1.0.0")
        let stderrBox = StderrBox()

        let settled = expectation(description: "abandoned connect() settles")
        let cap: TimeInterval = 0.3

        let started = Date()
        do {
            try await MCPManager.connectOrFailFast(
                client: client, transport: transport, child: process, stderrBox: stderrBox,
                hardCapSeconds: cap, onPathASettled: { settled.fulfill() })
            XCTFail("connectOrFailFast should throw when the hard cap expires")
        } catch {
            // Expected — the hard cap fired because the child never speaks MCP.
        }
        let raceElapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(raceElapsed, cap + 2, "the race itself should end at the hard cap, got \(raceElapsed)s")

        // Before the fix: this times out, because nothing ever calls client.disconnect() on the
        // abandoned Path A task, so its continuation never resumes and it runs forever.
        await fulfillment(of: [settled], timeout: 5)
    }
}