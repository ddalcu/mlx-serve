import XCTest
@testable import MLXCore

/// Regression tests for the editFile endLine-without-startLine loop observed
/// live (2026-07-02, alex.html session): after a legitimate "Pattern not found",
/// the model switched to line-based editing but emitted `{path, endLine, replace}`
/// with NO `startLine` — twelve near-identical retries across three user turns,
/// never recovering, until the stuck-detector killed each turn. Two app bugs
/// compounded: (1) `AgentEngine.toolExample` extracts the example by the literal
/// marker "Example: ", but editFile's description says "Example line-based: ",
/// so every editFile error shipped a literal `Example: {}` — a null steer
/// (contrast writeFile, whose real example recovered the model in one shot);
/// (2) the missing-param error never said WHAT was missing relative to the keys
/// the model actually sent.
@MainActor
final class EditFileRecoveryTests: XCTestCase {

    private func tempDir() throws -> String {
        let d = (NSTemporaryDirectory() as NSString).appendingPathComponent("efr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    private func read(_ dir: String, _ name: String) throws -> String {
        try String(contentsOfFile: (dir as NSString).appendingPathComponent(name), encoding: .utf8)
    }

    // MARK: toolExample extraction

    func testToolExampleForEditFileIsARealExample() {
        let example = AgentEngine.toolExample(for: "editFile")
        XCTAssertNotEqual(example, "{}", "editFile must never steer with an empty example")
        XCTAssertTrue(example.contains("startLine"),
                      "the example must show line-based mode: \(example)")
    }

    /// Class guard: every tool whose description carries an example must yield a
    /// real one — a description wording drift ("Example line-based: ") must never
    /// silently degrade the error steer to `Example: {}` again. No-arg tools
    /// (empty properties schema, e.g. listProcesses) legitimately example as {}.
    func testEveryToolDescriptionExampleIsExtractable() {
        for def in AgentPrompt.toolDefinitions {
            guard let fn = def["function"] as? [String: Any],
                  let name = fn["name"] as? String,
                  let desc = fn["description"] as? String,
                  desc.contains("Example") else { continue }
            let example = AgentEngine.toolExample(for: name)
            XCTAssertTrue(example.hasPrefix("{"),
                          "\(name): extracted example should start at the JSON: \(example)")
            if let params = fn["parameters"] as? [String: Any],
               let props = params["properties"] as? [String: Any], props.isEmpty {
                continue // no-arg tool: {} is the correct example
            }
            XCTAssertNotEqual(example, "{}",
                              "\(name): example in description was not extractable")
        }
    }

    // MARK: endLine-without-startLine steering (exact captured shape)

    func testEndLineWithoutStartLineSteersToStartLineEndToEnd() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let mem = AgentMemory()
        let rep = AgentEngine.RepetitionTracker()

        try "line1\nline2\nline3\n".write(
            toFile: (dir as NSString).appendingPathComponent("alex.html"),
            atomically: true, encoding: .utf8)

        // The exact captured shape: endLine + path + replace, no startLine, no find.
        let tc = APIClient.ToolCall(id: "1", name: "editFile",
            arguments: ["path": "alex.html",
                        "endLine": "65",
                        "replace": "<h2 class=\"highlight\">The Modern Identity</h2>"],
            rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd, repetition: rep,
                                                       iteration: 0, agentMemory: mem)

        XCTAssertFalse(result.output.contains("Example: {}"),
                       "must never ship an empty example: \(result.output)")
        XCTAssertTrue(result.output.contains("but no startLine"),
                      "must name the missing key relative to what was sent: \(result.output)")
        XCTAssertTrue(result.output.contains("\"startLine\""),
                      "must include a concrete JSON example with startLine: \(result.output)")
    }

    // MARK: dirty line-number values (same class as the writeFile append flag)

    func testDirtyStartLineStillPerformsLineBasedEdit() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let mem = AgentMemory()
        let rep = AgentEngine.RepetitionTracker()

        try "line1\nline2\nline3\n".write(
            toFile: (dir as NSString).appendingPathComponent("f.txt"),
            atomically: true, encoding: .utf8)

        // A trailing comma on the value must not demote a line-based edit into
        // "find/startLine missing" (the appendFlagIsTrue lesson: never gate a
        // mode on an exact string parse of a weak model's value).
        let tc = APIClient.ToolCall(id: "1", name: "editFile",
            arguments: ["path": "f.txt", "startLine": "2,", "endLine": "2", "replace": "LINE2"],
            rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd, repetition: rep,
                                                       iteration: 0, agentMemory: mem)

        XCTAssertFalse(result.output.lowercased().contains("error"),
                       "dirty startLine must still edit: \(result.output)")
        XCTAssertEqual(try read(dir, "f.txt"), "line1\nLINE2\nline3\n")
    }

    func testDirtyEndLineFallsBackSafelyNotSilentlyToStartLine() async throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let mem = AgentMemory()
        let rep = AgentEngine.RepetitionTracker()

        try "line1\nline2\nline3\nline4\n".write(
            toFile: (dir as NSString).appendingPathComponent("f.txt"),
            atomically: true, encoding: .utf8)

        // endLine "3," must parse as 3 — not fail Int() and silently collapse the
        // range to startLine..startLine (which would leave line 3 in place).
        let tc = APIClient.ToolCall(id: "1", name: "editFile",
            arguments: ["path": "f.txt", "startLine": "2", "endLine": "3,", "replace": "X"],
            rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd, repetition: rep,
                                                       iteration: 0, agentMemory: mem)

        XCTAssertFalse(result.output.lowercased().contains("error"), result.output)
        XCTAssertEqual(try read(dir, "f.txt"), "line1\nX\nline4\n")
    }
}
