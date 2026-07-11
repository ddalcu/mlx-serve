import XCTest
@testable import MLXCore

/// `searchFiles` used to shell out to `rg` (only at two hardcoded Homebrew
/// paths) and fall back to `grep -rn`. Three consequences the agent sees:
///
///  1. `context` is honored by the `rg` branch and SILENTLY DROPPED by the
///     `grep` branch — so the same tool call returns different output on two
///     machines depending on whether Homebrew ripgrep happens to be installed.
///  2. `grep -rn` descends into `.git`, flooding results with packfile noise.
///  3. `grep -rn` prints `Binary file <p> matches` for binaries.
///
///
/// These tests pin the behavior of the pure in-process searcher: no subprocess,
/// no host binaries, identical results everywhere.
final class FileSearcherTests: XCTestCase {

    /// Fixture tree:
    ///   src/a.swift      — NEEDLE on line 5, with 4 lines above/below
    ///   src/b.txt        — NEEDLE once
    ///   .git/config      — NEEDLE (must never be searched)
    ///   bin/blob.dat     — NUL bytes + NEEDLE (must never be searched)
    private func makeTree() throws -> String {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("filesearcher-\(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        let fm = FileManager.default
        for sub in ["src", ".git", "bin"] {
            try fm.createDirectory(at: root.appendingPathComponent(sub), withIntermediateDirectories: true)
        }

        let swiftLines = (1...9).map { $0 == 5 ? "let x = NEEDLE" : "line\($0)" }
        try swiftLines.joined(separator: "\n").write(
            to: root.appendingPathComponent("src/a.swift"), atomically: true, encoding: .utf8)

        try "nothing\nNEEDLE here\n".write(
            to: root.appendingPathComponent("src/b.txt"), atomically: true, encoding: .utf8)

        // `foo(bar` is not a valid regex — an agent searching for it literally
        // used to get a regex-compile error out of rg/grep.
        try "call foo(bar unbalanced\n".write(
            to: root.appendingPathComponent("src/c.txt"), atomically: true, encoding: .utf8)

        try "[core]\n\tNEEDLE = true\n".write(
            to: root.appendingPathComponent(".git/config"), atomically: true, encoding: .utf8)

        var blob = Data([0x00, 0x01, 0x02, 0x00])
        blob.append("NEEDLE".data(using: .utf8)!)
        try blob.write(to: root.appendingPathComponent("bin/blob.dat"))

        addTeardownBlock { try? fm.removeItem(at: root) }
        return root.path
    }

    private func search(_ root: String, _ params: [String: String]) async throws -> String {
        var p = params
        p["pattern"] = p["pattern"] ?? "NEEDLE"
        return try await SearchFilesHandler().execute(parameters: p, workingDirectory: root)
    }

    // MARK: - The three bugs

    func testContextLinesAreHonored() async throws {
        let root = try makeTree()
        let out = try await search(root, ["path": "src", "context": "2"])
        XCTAssertTrue(out.contains("let x = NEEDLE"), out)
        XCTAssertTrue(out.contains("line3"), "context lines above the match are missing:\n\(out)")
        XCTAssertTrue(out.contains("line7"), "context lines below the match are missing:\n\(out)")
        XCTAssertFalse(out.contains("line1"), "context should stop 2 lines out:\n\(out)")
    }

    func testGitDirectoryIsNeverSearched() async throws {
        let root = try makeTree()
        let out = try await search(root, [:])
        XCTAssertTrue(out.contains("a.swift"), out)
        XCTAssertFalse(out.contains(".git"), "results leaked .git contents:\n\(out)")
    }

    func testBinaryFilesAreSkipped() async throws {
        let root = try makeTree()
        let out = try await search(root, [:])
        XCTAssertFalse(out.contains("blob.dat"), "binary file was searched:\n\(out)")
        XCTAssertFalse(out.lowercased().contains("binary file"), "grep's binary notice leaked:\n\(out)")
    }

    // MARK: - Contract the old shell-out also had (must not regress)

    func testIncludeGlobFiltersByFilenameAtAnyDepth() async throws {
        let root = try makeTree()
        let out = try await search(root, ["include": "*.swift"])
        XCTAssertTrue(out.contains("a.swift"), out)
        XCTAssertFalse(out.contains("b.txt"), out)
    }

    func testOutputIsPathColonLineColonText() async throws {
        let root = try makeTree()
        let out = try await search(root, ["path": "src/b.txt"])
        // ripgrep `-n --no-heading` shape: <path>:<line>:<text>
        let line = try XCTUnwrap(out.split(separator: "\n").first.map(String.init), out)
        XCTAssertTrue(line.hasSuffix(":2:NEEDLE here"), "unexpected result shape: \(line)")
    }

    /// Files inside one directory must be VISITED alphabetically, not just
    /// rendered sorted — `maxResults` truncation happens in visit order, so a
    /// reverse-alphabetical walk truncates away the files an agent (and the
    /// docs) expect to see first.
    func testTruncationKeepsTheAlphabeticallyFirstFile() async throws {
        let root = try makeTree()
        let many = URL(fileURLWithPath: root).appendingPathComponent("many")
        try FileManager.default.createDirectory(at: many, withIntermediateDirectories: true)
        for name in ["aaa.txt", "mmm.txt", "zzz.txt"] {
            try "NEEDLE\n".write(
                to: many.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        let lines = FileSearcher.search(.init(
            root: many.path, pattern: "NEEDLE", maxResults: 1))
        XCTAssertEqual(lines.count, 1)
        XCTAssertTrue(lines[0].path.hasSuffix("/aaa.txt"),
                      "expected the alphabetically first match, got \(lines[0].path)")
    }

    func testMaxResultsTruncates() async throws {
        let root = try makeTree()
        let out = try await search(root, ["maxResults": "1"])
        XCTAssertEqual(out.split(separator: "\n").count, 1, out)
    }

    func testNoMatchesReturnsCleanMessageNotAnError() async throws {
        let root = try makeTree()
        let out = try await search(root, ["pattern": "ZZZ_ABSENT_ZZZ"])
        XCTAssertFalse(out.contains("exit code"), "a no-match search must not surface a shell exit code:\n\(out)")
        XCTAssertTrue(out.lowercased().contains("no match"), out)
    }

    func testPathOutsideWorkspaceIsRejected() async throws {
        let root = try makeTree()
        do {
            _ = try await search(root, ["path": "/etc"])
            XCTFail("expected confinement error")
        } catch {
            XCTAssertTrue("\(error)".contains("outside the workspace"), "\(error)")
        }
    }

    func testInvalidRegexFallsBackToLiteralSearch() async throws {
        let root = try makeTree()
        // `foo(bar` has an unbalanced group — invalid as a regex. Fall back to
        // a literal search rather than failing the tool call.
        let out = try await search(root, ["pattern": "foo(bar"])
        XCTAssertTrue(out.contains("c.txt"), out)
        XCTAssertTrue(out.contains("call foo(bar unbalanced"), out)
    }

    func testValidRegexIsTreatedAsARegex() async throws {
        let root = try makeTree()
        let out = try await search(root, ["pattern": "^line[37]$"])
        XCTAssertTrue(out.contains("line3"), out)
        XCTAssertTrue(out.contains("line7"), out)
        XCTAssertFalse(out.contains("line4"), out)
    }
}

/// Characterization of the glob semantics `ListFilesHandler` has always had,
/// pinned here because `FileSearcher.include` now shares the implementation.
final class GlobTests: XCTestCase {
    func testStarDoesNotCrossPathSeparators() {
        XCTAssertTrue(Glob.matches("a.swift", pattern: "*.swift"))
        XCTAssertFalse(Glob.matches("src/a.swift", pattern: "*.swift"))
    }

    func testDoubleStarCrossesPathSeparators() {
        XCTAssertTrue(Glob.matches("src/deep/a.ts", pattern: "**/*.ts"))
        XCTAssertTrue(Glob.matches("a.ts", pattern: "**/*.ts"))
    }

    func testQuestionMarkMatchesOneNonSeparator() {
        XCTAssertTrue(Glob.matches("a1.txt", pattern: "a?.txt"))
        XCTAssertFalse(Glob.matches("a/1.txt", pattern: "a?1.txt"))
    }

    func testDotIsLiteral() {
        XCTAssertFalse(Glob.matches("axswift", pattern: "*.swift"))
    }
}
