import XCTest
@testable import MLXCore

/// Auto-detection of OTHER local-inference tools' model folders.
///
/// LM Studio's folder was detected from the start; every other tool's was not,
/// so a library that MLX Core can already read sat invisible until someone
/// spent their single "Custom folder" slot on it. Two of them are real here:
/// MTPLX (`~/.mtplx/models`, flat `Org--Name` dirs) and Osaurus
/// (`~/MLXModels`, plain `org/repo`) — both layouts the Zig discovery has
/// always understood, so this is a configuration gap, not an engine one.
///
/// Detection is INJECTED rather than read from the real home directory: a test
/// that passes only on a machine with MTPLX installed is not a test.
final class ToolModelRootsTests: XCTestCase {

    private var defaults: UserDefaults!
    private let suite = "ToolModelRootsTests"
    private var scratch: String!

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removePersistentDomain(forName: suite)
        defaults = UserDefaults(suiteName: suite)
        scratch = NSTemporaryDirectory() + "ToolModelRoots-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: scratch, withIntermediateDirectories: true)
    }

    override func tearDown() {
        UserDefaults.standard.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(atPath: scratch)
        super.tearDown()
    }

    /// Make `<scratch>/<rel>` and hand back its standardized path, so the
    /// expectation compares the same spelling `existingDirectory` produces.
    @discardableResult
    private func makeHomeDir(_ rel: String) -> String {
        let p = (scratch as NSString).appendingPathComponent(rel)
        try? FileManager.default.createDirectory(atPath: p, withIntermediateDirectories: true)
        return URL(fileURLWithPath: p).standardizedFileURL.path
    }

    // MARK: - Detection

    /// The two canonical folders, found where their own tools put them.
    func testCanonicalToolFoldersAreDetectedUnderHome() {
        let mtplx = makeHomeDir(".mtplx/models")
        let osaurus = makeHomeDir("MLXModels")
        let lms = makeHomeDir(".lmstudio/models")

        let detected = ToolModelRoots.detected(home: scratch)

        XCTAssertEqual(detected.mtplx, mtplx)
        XCTAssertEqual(detected.osaurus, osaurus)
        XCTAssertEqual(detected.lmStudio, lms)
    }

    /// A tool that is not installed contributes nothing. Existence-gating is
    /// the whole safety story: an absent folder can never widen the scan.
    func testAbsentToolFoldersAreNil() {
        let detected = ToolModelRoots.detected(home: scratch)
        XCTAssertNil(detected.mtplx)
        XCTAssertNil(detected.osaurus)
        XCTAssertNil(detected.lmStudio)
        XCTAssertTrue(detected.ordered.isEmpty)
    }

    /// A FILE at the canonical path is not a folder, and must not be offered
    /// to `--model-dir` (the server exits on a root it cannot open).
    func testAFileAtTheCanonicalPathIsNotARoot() {
        let dir = (scratch as NSString).appendingPathComponent(".mtplx")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: (dir as NSString).appendingPathComponent("models"), contents: Data())
        XCTAssertNil(ToolModelRoots.detected(home: scratch).mtplx)
    }

    // MARK: - How they reach the server

    /// Every detected folder becomes a scanned root. Without this the model is
    /// listed nowhere: the picker reads `readRoots` and the server reads the
    /// same list as `--model-dir` flags.
    func testDetectedFoldersAreScannedAfterTheOwnedFolders() {
        let mtplx = makeHomeDir(".mtplx/models")
        let osaurus = makeHomeDir("MLXModels")
        let lms = makeHomeDir(".lmstudio/models")
        let dest = makeHomeDir("dest")

        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dest
        let scanned = roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: lms, mtplx: mtplx, osaurus: osaurus))

        XCTAssertEqual(scanned.first, dest, "the folder we write into holds the live copy and is first-wins")
        XCTAssertTrue(scanned.contains(mtplx))
        XCTAssertTrue(scanned.contains(osaurus))
        XCTAssertTrue(scanned.contains(lms))
        XCTAssertEqual(scanned, roots.readRoots(toolRoots: ToolModelRoots(lmStudio: lms, mtplx: mtplx, osaurus: osaurus)),
                       "the app's reads and the server's scan are one list")
    }

    /// These are other tools' trees. The app may serve out of them and must
    /// never delete into them — the same rule LM Studio's folder already had.
    func testDetectedFoldersAreNeverOwned() {
        let mtplx = makeHomeDir(".mtplx/models")
        let osaurus = makeHomeDir("MLXModels")
        let roots = ModelRoots(defaults: defaults)
        _ = roots.scanRoots(toolRoots: ToolModelRoots(mtplx: mtplx, osaurus: osaurus))
        XCTAssertFalse(roots.ownedRoots.contains(mtplx))
        XCTAssertFalse(roots.ownedRoots.contains(osaurus))
    }

    /// `main.zig` EXITS on one `--model-dir` past its cap, so the client's list
    /// is bounded by the server's own number however many tools are installed.
    func testDetectedFoldersCannotPushPastTheServerCap() {
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = makeHomeDir("dest")
        roots.customRoot = makeHomeDir("custom")
        let tools = ToolModelRoots(lmStudio: makeHomeDir(".lmstudio/models"),
                                   mtplx: makeHomeDir(".mtplx/models"),
                                   osaurus: makeHomeDir("MLXModels"))
        XCTAssertLessThanOrEqual(roots.scanRoots(toolRoots: tools).count, ModelRoots.serverRootLimit)
    }

    /// One folder reached twice is one `--model-dir`. Someone whose download
    /// destination IS `~/MLXModels` must not get it scanned as both.
    func testAFolderReachedTwiceIsScannedOnce() {
        let shared = makeHomeDir("MLXModels")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = shared
        let scanned = roots.scanRoots(toolRoots: ToolModelRoots(osaurus: shared))
        XCTAssertEqual(scanned.filter { $0 == shared }.count, 1)
    }
}

/// The picker enumerates folders SEPARATELY from `scanRoots`, so a root added
/// to the server's `--model-dir` list and nowhere else is served but invisible
/// — the mirror image of the bug that created `ModelRoots` in the first place
/// (listed by the picker, absent from `/v1/models`). These pin both halves.
final class ToolFolderListingTests: XCTestCase {

    private var scratch: String!

    override func setUp() {
        super.setUp()
        scratch = NSTemporaryDirectory() + "ToolFolderListing-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: scratch, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: scratch)
        super.tearDown()
    }

    private func writeModel(_ rel: String, modelType: String = "qwen3") -> String {
        let dir = (scratch as NSString).appendingPathComponent(rel)
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try? #"{"model_type":"\#(modelType)"}"#.write(toFile: (dir as NSString).appendingPathComponent("config.json"),
                                                     atomically: true, encoding: .utf8)
        // A config alone is not a model: `makeLocalModels` requires weights, so
        // an interrupted download never becomes a picker entry.
        FileManager.default.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"),
                                       contents: Data("weights".utf8))
        return dir
    }

    /// MTPLX writes one FLAT `Org--Name` dir per model; Osaurus writes plain
    /// `org/repo`. `dualLayoutModels` reads both, which is why neither needs an
    /// engine change — only a folder nobody was looking in.
    func testBothToolLayoutsEnumerate() {
        _ = writeModel("mtplx/Youssofal--Some-Model")
        _ = writeModel("osaurus/SomeOrg/Some-Repo")

        let flat = DownloadManager.dualLayoutModels(
            atRoot: (scratch as NSString).appendingPathComponent("mtplx"),
            idPrefix: "tool:", source: .toolFolder)
        let nested = DownloadManager.dualLayoutModels(
            atRoot: (scratch as NSString).appendingPathComponent("osaurus"),
            idPrefix: "tool:", source: .toolFolder)

        XCTAssertEqual(flat.map(\.name), ["Youssofal--Some-Model"])
        XCTAssertEqual(nested.map(\.name), ["SomeOrg/Some-Repo"])
        XCTAssertTrue((flat + nested).allSatisfy { $0.source == .toolFolder })
    }

    /// Another tool's tree is never ours to delete, and the badge that replaces
    /// the trash has to SAY something — a nil reason renders a blank badge.
    func testToolFolderModelsAreReadOnlyAndExplainWhy() throws {
        _ = writeModel("mtplx/Org--M")
        let model = try XCTUnwrap(DownloadManager.dualLayoutModels(
            atRoot: (scratch as NSString).appendingPathComponent("mtplx"),
            idPrefix: "tool:", source: .toolFolder).first)
        XCTAssertFalse(model.isDeletable)
        XCTAssertNotNil(model.externalReadOnlyReason)
    }

    /// A new source that no picker renders is a model you cannot select. Every
    /// case must be in the browser's display order, and its two title tables
    /// must agree — they are separate switches that have drifted before.
    func testEverySourceIsRenderableAndTitledConsistently() {
        for source in LocalModelSource.allCases {
            XCTAssertTrue(ModelBrowserUse.sourceOrder.contains(source),
                          "\(source) has no group in the model browser")
            XCTAssertFalse(source.sectionTitle.isEmpty)
        }
        XCTAssertTrue(ModelBrowserUse.sourceOrder.contains(.toolFolder))
    }

    /// LM Studio's group was titled "Other Discovered Models" back when it was
    /// the only non-owned folder. Now that it isn't, the generic title belongs
    /// to the generic bucket and LM Studio gets named.
    func testTheGenericTitleBelongsToTheGenericBucket() {
        XCTAssertEqual(LocalModelSource.toolFolder.sectionTitle, "Other Discovered Models")
        XCTAssertEqual(LocalModelSource.lmStudio.sectionTitle, "LM Studio Models")
        XCTAssertEqual(ModelBrowserUse.groupTitle(.toolFolder), LocalModelSource.toolFolder.sectionTitle)
        XCTAssertEqual(ModelBrowserUse.groupTitle(.lmStudio), LocalModelSource.lmStudio.sectionTitle)
    }

    /// The tray picker hardcodes its section headings rather than reading
    /// `sectionTitle`, so it is the one that silently drops a new source.
    func testTheTrayPickerRendersEverySource() throws {
        let src = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/StatusMenuView.swift")
        let text = try String(contentsOf: src, encoding: .utf8)
        for source in LocalModelSource.allCases {
            XCTAssertTrue(text.contains("$0.source == .\(source.rawValue)"),
                          "StatusMenuView never filters for .\(source.rawValue), so those models are unpickable")
        }
    }
}
