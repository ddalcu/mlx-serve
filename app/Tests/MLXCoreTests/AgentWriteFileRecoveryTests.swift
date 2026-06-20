import XCTest
@testable import MLXCore

/// Integration test of the REAL tool-execution path (`AgentEngine.executeToolCall`)
/// for the Gemma 4 12B append-jam observed live (2026-06-20): the model crams the
/// whole file body into the `append` value with NO `content` key —
/// `{"append":"true,\n<body>","path":"jfk.txt"}` — which left `content` "missing",
/// fired a bogus "your output got cut off" error, and looped the agent until the
/// stuck-detector gave up. These exercise the full dispatch (canonical name →
/// effectiveTool → normalizeWriteFileArgs → validate → handler), not just the
/// pure helper, so they prove the recovery is actually wired into execution.
@MainActor
final class AgentWriteFileRecoveryTests: XCTestCase {

    private func tempDir() throws -> String {
        let d = (NSTemporaryDirectory() as NSString).appendingPathComponent("awfr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    private func read(_ dir: String, _ name: String) throws -> String {
        try String(contentsOfFile: (dir as NSString).appendingPathComponent(name), encoding: .utf8)
    }

    func testGemma12BAppendJamIsRecoveredEndToEnd() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let mem = AgentMemory()
        let rep = AgentEngine.RepetitionTracker()

        // Turn 1 succeeded in the live session — seed the file.
        let seed = APIClient.ToolCall(id: "1", name: "writeFile",
            arguments: ["path": "jfk.txt", "content": "Chapter 1.\n"], rawArguments: "")
        _ = await AgentEngine.executeToolCall(seed, workingDirectory: &wd, repetition: rep,
                                              iteration: 0, agentMemory: mem)

        // The exact append-jam shape captured from the failed session.
        let jam = APIClient.ToolCall(id: "2", name: "writeFile",
            arguments: ["append": "true,\nThe motorcade moved forward, a steady\npulse of movement.",
                        "path": "jfk.txt"], rawArguments: "")
        let result = await AgentEngine.executeToolCall(jam, workingDirectory: &wd, repetition: rep,
                                                       iteration: 1, agentMemory: mem)

        XCTAssertFalse(result.output.lowercased().contains("error"),
                       "append-jam must be recovered and written, not error: \(result.output)")
        XCTAssertEqual(try read(dir, "jfk.txt"),
                       "Chapter 1.\nThe motorcade moved forward, a steady\npulse of movement.")
    }

    func testGenuinelyMissingContentGivesHonestErrorNotTruncation() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let mem = AgentMemory()
        let rep = AgentEngine.RepetitionTracker()

        // Clean boolean append, no content → genuinely missing (not recoverable).
        let tc = APIClient.ToolCall(id: "1", name: "writeFile",
            arguments: ["path": "x.txt", "append": "true"], rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd, repetition: rep,
                                                       iteration: 0, agentMemory: mem)

        XCTAssertTrue(result.output.lowercased().contains("content"),
                      "error must name the content parameter: \(result.output)")
        // The user's complaint: it must NOT falsely claim truncation when the
        // token cap was never hit (real truncation is handled upstream).
        XCTAssertFalse(result.output.lowercased().contains("cut off"),
                       "must not misdiagnose missing content as a truncation: \(result.output)")
        XCTAssertFalse(result.output.lowercased().contains("truncated"),
                       "must not misdiagnose missing content as a truncation: \(result.output)")
    }
}
