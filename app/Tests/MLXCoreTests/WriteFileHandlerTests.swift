import XCTest
@testable import MLXCore

// The agent's writeFile tool. A whole file rides inside one tool-call response,
// so a verbose small model can blow past its token budget mid-file and the
// (truncated) call gets dropped. The robust mitigation is to keep each write
// small and grow large files incrementally via `append:true` — these tests pin
// that append contract.
final class WriteFileHandlerTests: XCTestCase {

    private func makeTempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("wfh-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    private func read(_ dir: String, _ name: String) throws -> String {
        try String(contentsOfFile: (dir as NSString).appendingPathComponent(name), encoding: .utf8)
    }

    func testWriteThenAppendConcatenatesChunks() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "doc.html", "content": "<!DOCTYPE html>\n<body>\n"], workingDirectory: dir)
        let out = try await WriteFileHandler().execute(
            parameters: ["path": "doc.html", "content": "</body>\n", "append": "true"], workingDirectory: dir)
        XCTAssertTrue(out.lowercased().contains("append"), "append result should say so: \(out)")
        XCTAssertEqual(try read(dir, "doc.html"), "<!DOCTYPE html>\n<body>\n</body>\n")
    }

    func testAppendToMissingFileCreatesIt() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "new.txt", "content": "hello", "append": "true"], workingDirectory: dir)
        XCTAssertEqual(try read(dir, "new.txt"), "hello")
    }

    func testDefaultIsOverwriteNotAppend() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "first"], workingDirectory: dir)
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "second"], workingDirectory: dir)
        XCTAssertEqual(try read(dir, "f.txt"), "second")
    }

    func testAppendFalseStringOverwrites() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "first"], workingDirectory: dir)
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "second", "append": "false"], workingDirectory: dir)
        XCTAssertEqual(try read(dir, "f.txt"), "second")
    }

    // ── Content-jammed-into-append recovery ────────────────────────────────
    // Gemma 4 12B (live, 2026-06-20) crams the whole file body into the
    // `append` value with NO `content` key — `{"append":"true,\n<body>",...}` —
    // so the call had no content and looped on a misleading "truncated" error.
    // normalizeWriteFileArgs splits the leading boolean from the body.

    func testNormalizeRecoversBodyJammedIntoAppend() {
        let args = ["append": "true,\nThe motorcade moved forward, a steady\npulse of movement.", "path": "jfk.txt"]
        let out = AgentEngine.normalizeWriteFileArgs(args)
        XCTAssertEqual(out["append"], "true")
        XCTAssertEqual(out["content"], "The motorcade moved forward, a steady\npulse of movement.")
        XCTAssertEqual(out["path"], "jfk.txt")
    }

    func testNormalizeHandlesNewlineWithoutComma() {
        let out = AgentEngine.normalizeWriteFileArgs(["append": "true\nbody text here", "path": "x"])
        XCTAssertEqual(out["append"], "true")
        XCTAssertEqual(out["content"], "body text here")
    }

    func testNormalizeLeavesCleanBooleanAlone() {
        // append is a clean flag and content is genuinely missing — must NOT
        // fabricate content, so a real missing-content error still surfaces.
        let out = AgentEngine.normalizeWriteFileArgs(["append": "true", "path": "x"])
        XCTAssertNil(out["content"])
        XCTAssertEqual(out["append"], "true")
    }

    func testNormalizeDoesNotClobberExistingContent() {
        let out = AgentEngine.normalizeWriteFileArgs(["append": "true,junk", "content": "real body", "path": "x"])
        XCTAssertEqual(out["content"], "real body")
        XCTAssertEqual(out["append"], "true,junk")
    }

    func testNormalizeIgnoresNonBooleanAppend() {
        // "truely" is not a boolean keyword followed by a separator → leave it.
        let out = AgentEngine.normalizeWriteFileArgs(["append": "truely yours", "path": "x"])
        XCTAssertNil(out["content"])
        XCTAssertEqual(out["append"], "truely yours")
    }

    // ── Dirty append-flag value ────────────────────────────────────────────
    // gemma-4-12b (live, 2026-06-20) emits the flag with a trailing comma:
    // `{"append":"true,","content":"<body>"}`. The exact `== "true"` match
    // failed → it OVERWROTE and shrank the file instead of appending.

    func testAppendWithTrailingCommaStillAppends() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "first\n"], workingDirectory: dir)
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "f.txt", "content": "second\n", "append": "true,"], workingDirectory: dir)
        XCTAssertEqual(try read(dir, "f.txt"), "first\nsecond\n", "append:\"true,\" must append, not overwrite")
    }

    func testAppendFlagIsTrueTolerant() {
        for v in ["true", "true,", "True", " true ", "TRUE,", "true,\n", "1", "yes"] {
            XCTAssertTrue(WriteFileHandler.appendFlagIsTrue(v), "\(v.debugDescription) should be append=true")
        }
        for v in ["false", "", "false,", "no", "0", "truely", "truthy"] {
            XCTAssertFalse(WriteFileHandler.appendFlagIsTrue(v), "\(v.debugDescription) should be append=false")
        }
        XCTAssertFalse(WriteFileHandler.appendFlagIsTrue(nil))
    }

    func testNormalizedAppendActuallyWritesContent() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        _ = try await WriteFileHandler().execute(
            parameters: ["path": "j.txt", "content": "Chapter 1.\n"], workingDirectory: dir)
        let jammed = ["append": "true,\nChapter 2.\n", "path": "j.txt"]
        let normalized = AgentEngine.normalizeWriteFileArgs(jammed)
        _ = try await WriteFileHandler().execute(parameters: normalized, workingDirectory: dir)
        XCTAssertEqual(try read(dir, "j.txt"), "Chapter 1.\nChapter 2.\n")
    }
}
