import XCTest
@testable import MLXCore

/// The enlarge preview's before/after slider. An upscale records nothing about
/// the photo it came from — and that photo can move, be deleted, or be an
/// earlier result already in the Trash — so the comparison keeps its own copy
/// of the exact picture SeedVR2 was handed.
final class RestoreComparisonTests: XCTestCase {

    private var root: String = ""

    override func setUpWithError() throws {
        root = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("restore-comparison-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: root)
    }

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// A result in a day folder under `root`, laid out the way `RestoreService`
    /// writes one.
    private func makeResult(day: String = "2026-09-13", name: String) throws -> String {
        let dir = (root as NSString).appendingPathComponent(day)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent(name)
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: URL(fileURLWithPath: path))
        return path
    }

    func testTheInputIsKeptInAHiddenFolderBesideItsResult() {
        // Keyed by the result's own filename, so a result always finds its
        // input and two results can never share one.
        XCTAssertEqual(
            RestoreComparison.inputPath(forResult: "/u/upscales/2026-09-13/2026-09-13_10-00-00_cat_upscaled.png"),
            "/u/upscales/2026-09-13/.before/2026-09-13_10-00-00_cat_upscaled.png")
    }

    func testASavedInputComesBackByteForByte() throws {
        let result = try makeResult(name: "2026-09-13_10-00-00_cat_upscaled.png")
        let sent = Data((0..<256).map { UInt8($0) })
        XCTAssertTrue(RestoreComparison.saveInput(sent, forResult: result))
        let input = try XCTUnwrap(RestoreComparison.existingInput(forResult: result))
        // Byte-identical, never re-encoded: SeedVR2 removes compression
        // artefacts, so a lossy "before" would credit it with repairing damage
        // the comparison itself did.
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: input)), sent)
    }

    func testAResultMadeBeforeComparisonsExistedHasNoInput() throws {
        // The preview falls back to the plain picture for these — the upscales
        // folder already holds results nobody saved an input for.
        let result = try makeResult(name: "2026-08-21_09-00-00_old_upscaled.png")
        XCTAssertNil(RestoreComparison.existingInput(forResult: result))
    }

    func testTheStoredInputIsNeverListedAsAResultOfItsOwn() throws {
        // The strip is rebuilt from a scan of the upscales folder, and the
        // input carries the SAME filename as its result. A scan that ever
        // descends into `.before/` lists every enlarge twice, the second copy
        // a "before" passing itself off as a result.
        let result = try makeResult(name: "2026-09-13_10-00-00_cat_upscaled.png")
        XCTAssertTrue(RestoreComparison.saveInput(Data([1, 2, 3]), forResult: result))
        XCTAssertEqual(RestoreService.recentPaths(root: root), [result])
    }

    func testRemovingTheInputLeavesTheResultAlone() throws {
        let result = try makeResult(name: "2026-09-13_10-00-00_cat_upscaled.png")
        XCTAssertTrue(RestoreComparison.saveInput(Data([1, 2, 3]), forResult: result))
        RestoreComparison.removeInput(forResult: result)
        XCTAssertNil(RestoreComparison.existingInput(forResult: result))
        XCTAssertTrue(FileManager.default.fileExists(atPath: result))
    }

    func testTheDividerFollowsThePointerAndStaysOnThePicture() {
        XCTAssertEqual(RestoreComparison.fraction(x: 150, width: 300), 0.5)
        XCTAssertEqual(RestoreComparison.fraction(x: 0, width: 300), 0)
        // A drag past either edge parks the divider ON that edge instead of
        // losing it off the picture.
        XCTAssertEqual(RestoreComparison.fraction(x: -40, width: 300), 0)
        XCTAssertEqual(RestoreComparison.fraction(x: 900, width: 300), 1)
        // A frame with no width yet (the first layout pass) must not divide
        // by zero.
        XCTAssertEqual(RestoreComparison.fraction(x: 10, width: 0), 0.5)
    }

    // MARK: - Wiring

    func testTheServiceKeepsWhatItSentAndOnlyAfterTheResultLanded() throws {
        let text = try source("Sources/MLXServe/Services/RestoreService.swift")
        let write = try XCTUnwrap(text.range(of: "try png.write(to:"), "result write not found")
        let save = try XCTUnwrap(
            text.range(of: "RestoreComparison.saveInput(prepared.data, forResult: outputPath)"),
            "the restore must keep the bytes it sent as the comparison's before")
        // After, never before: a result write that throws must not leave an
        // input behind with nothing to compare it to.
        XCTAssertLessThan(write.lowerBound, save.lowerBound)
    }

    func testThePreviewComparesAndTrashTakesTheInputWithTheResult() throws {
        let text = try source("Sources/MLXServe/Views/ImageGenView.swift")
        XCTAssertTrue(text.contains("BeforeAfterSlider("),
                      "an enlarged result with a stored input is shown as a comparison")
        // Inside Move to Trash, and only once the trash itself succeeded.
        let start = try XCTUnwrap(text.range(of: "private func moveToTrash"))
        let end = try XCTUnwrap(text.range(of: "\n    }\n", range: start.upperBound..<text.endIndex))
        let body = text[start.upperBound..<end.lowerBound]
        let trashed = try XCTUnwrap(body.range(of: "trashItem("))
        let removed = try XCTUnwrap(body.range(of: "RestoreComparison.removeInput(forResult: item.path)"),
                                    "trashing an enlarged result must not leave its input behind")
        XCTAssertLessThan(trashed.lowerBound, removed.lowerBound)
    }
}
