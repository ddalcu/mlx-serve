import XCTest
@testable import MLXCore

/// readFile and editFile count lines the way an editor does: LF, CRLF and CR each end one (#736).
final class FileToolLineBreakTests: XCTestCase {
    private let gate = FileToolSandboxGate(sandboxEnabled: { false }, pinnedWorkspace: { (nil, nil) }, ensureMounted: { _ in })

    private func workspace(_ content: String) throws -> String {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("ftlb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try Data(content.utf8).write(to: URL(fileURLWithPath: dir + "/f.txt"))
        return dir
    }

    func testReadFileNumbersCrAndCrlfLines() async throws {
        let dir = try workspace("first\rsecond\r\nthird\n\nfifth")
        let read = ReadFileHandler(gate: gate)
        let all = try await read.execute(parameters: ["path": "f.txt"], workingDirectory: dir)
        XCTAssertEqual(all, "1| first\n2| second\n3| third\n4| \n5| fifth")
        let one = try await read.execute(parameters: ["path": "f.txt", "startLine": "2", "endLine": "2"], workingDirectory: dir)
        XCTAssertEqual(one, "2| second")
    }

    func testEditFileByLineKeepsTheFilesLineBreaks() async throws {
        for (before, after) in [("a\rb\rc", "a\rB\rc"), ("a\r\nb\r\nc\r\n", "a\r\nB\r\nC\r\n")] {
            let dir = try workspace(before)
            let edit = EditFileHandler(gate: gate)
            let endLine = before.contains("\r\n") ? "3" : "2"
            let replace = before.contains("\r\n") ? "B\nC" : "B"
            _ = try await edit.execute(parameters: ["path": "f.txt", "startLine": "2", "endLine": endLine, "replace": replace], workingDirectory: dir)
            XCTAssertEqual(try String(contentsOfFile: dir + "/f.txt", encoding: .utf8), after)
        }
    }
}
