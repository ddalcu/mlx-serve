import XCTest
@testable import MLXCore

/// Where models live, and which of those folders the SERVER is told about.
///
/// Two folder settings existed before this: the hardcoded `~/.mlx-serve/models`
/// download destination, and a "Custom folder" that only ever fed the app's own
/// scan. The server takes `--model-dir`, so anything in the custom folder was
/// listed by the picker and absent from `/v1/models`.
final class ModelRootsTests: XCTestCase {

    private var defaults: UserDefaults!
    private let suite = "ModelRootsTests"

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removePersistentDomain(forName: suite)
        defaults = UserDefaults(suiteName: suite)
    }

    override func tearDown() {
        UserDefaults.standard.removePersistentDomain(forName: suite)
        super.tearDown()
    }

    private func tempDir(_ name: String) -> String {
        let p = NSTemporaryDirectory() + "ModelRootsTests-\(name)-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: p, withIntermediateDirectories: true)
        return p
    }

    /// A model in the custom scan folder (or LM Studio's) is served by the
    /// server, so the app's "is it on disk?" reads must find it there too —
    /// reading only `ownedRoots` is how a pack in the custom folder got a
    /// Download bar over a copy already being served. Writes/deletes stay on
    /// `ownedRoots`, so the custom folder must NOT appear there.
    func testReadRootsCoverEveryServedFolderButOwnedRootsStayNarrow() {
        let dest = tempDir("dest"), custom = tempDir("custom"), lms = tempDir("lms")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dest
        roots.customRoot = custom
        let read = roots.readRoots(toolRoots: ToolModelRoots(lmStudio: lms))
        XCTAssertEqual(read, roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: lms)))
        XCTAssertEqual(read.first, dest)
        XCTAssertTrue(read.contains(custom))
        XCTAssertTrue(read.contains(lms))
        XCTAssertFalse(roots.ownedRoots.contains(custom))
        XCTAssertFalse(roots.ownedRoots.contains(lms))
    }

    // MARK: - The download destination

    /// With nothing configured the destination is exactly what it has always
    /// been. This is the compatibility assertion: every existing install has no
    /// value stored, and must keep downloading to the same place.
    func testUnsetDefaultFolderIsTheHistoricalPath() {
        let roots = ModelRoots(defaults: defaults)
        XCTAssertEqual(roots.downloadRoot, NSString(string: "~/.mlx-serve/models").expandingTildeInPath)
        XCTAssertNil(roots.configuredDownloadRoot)
    }

    func testChoosingADefaultFolderMovesTheDownloadDestination() {
        let dir = tempDir("dest")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dir
        XCTAssertEqual(roots.downloadRoot, dir)
        // It survives a relaunch.
        XCTAssertEqual(ModelRoots(defaults: defaults).downloadRoot, dir)
        // Clearing returns to the built-in path rather than to nothing.
        roots.configuredDownloadRoot = nil
        XCTAssertEqual(roots.downloadRoot, NSString(string: "~/.mlx-serve/models").expandingTildeInPath)
    }

    /// A path that has stopped existing must not silently become the download
    /// destination — the folder was on a drive that is no longer plugged in,
    /// and creating it would write GBs into a mountpoint stub. The configured
    /// value is KEPT so Settings can still show it and the user can see why.
    func testAMissingDefaultFolderFallsBackButIsNotForgotten() {
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = "/Volumes/Gone/models"
        XCTAssertEqual(roots.downloadRoot, NSString(string: "~/.mlx-serve/models").expandingTildeInPath)
        XCTAssertEqual(roots.configuredDownloadRoot, "/Volumes/Gone/models")
        XCTAssertTrue(roots.downloadRootIsUnavailable)
    }

    // MARK: - What the server is told to scan

    /// Every folder we know about, download destination FIRST — that ordering
    /// is what decides a duplicate id server-side, and the folder we write into
    /// is the one whose copy must win.
    func testScanRootsLeadWithTheDownloadDestination() {
        let dest = tempDir("dest")
        let custom = tempDir("custom")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dest
        roots.customRoot = custom
        let all = roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: nil))
        XCTAssertEqual(all.first, dest)
        XCTAssertTrue(all.contains(custom))
    }

    /// The historical root does not stop being scanned when the destination
    /// moves. Everything downloaded before the change lives there, and dropping
    /// it would make a user's whole library vanish from the picker.
    ///
    /// `scanRoots` only emits folders that EXIST, so this rule is invisible on
    /// a machine that has never downloaded anything — which is every CI runner,
    /// and where this asserted a developer's own disk rather than the code
    /// (`NSHomeDirectory()` ignores a `HOME` override, so the folder cannot be
    /// faked away either). Create it, and take it back if it wasn't already
    /// there.
    func testTheBuiltInFolderIsStillScannedAfterTheDestinationMoves() {
        let builtIn = ModelRoots.builtInRoot
        let preexisting = FileManager.default.fileExists(atPath: builtIn)
        if !preexisting {
            try? FileManager.default.createDirectory(atPath: builtIn, withIntermediateDirectories: true)
        }
        defer { if !preexisting { try? FileManager.default.removeItem(atPath: builtIn) } }

        let dest = tempDir("dest")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dest
        let all = roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: nil))
        XCTAssertEqual(all.first, dest)
        XCTAssertTrue(all.contains(builtIn),
                      "models downloaded before the move must stay visible")
    }

    /// The roots the APP ITSELF reads and deletes from — destination first,
    /// built-in second. Unlike `scanRoots` this is not existence-filtered (a
    /// read against a missing folder just finds nothing) and never includes
    /// LM Studio / custom folders, which are other tools' trees the app must
    /// not delete into.
    func testOwnedRootsKeepTheBuiltInFolderAfterTheDestinationMoves() {
        var roots = ModelRoots(defaults: defaults)
        XCTAssertEqual(roots.ownedRoots, [ModelRoots.builtInRoot])

        let dest = tempDir("dest")
        roots.configuredDownloadRoot = dest
        XCTAssertEqual(roots.ownedRoots, [dest, ModelRoots.builtInRoot],
                       "the pre-move library must stay readable, destination's copy winning")
    }

    /// De-duped, and never a folder that isn't there: each entry becomes a
    /// `--model-dir`, and the server logs a warning per unopenable one.
    func testScanRootsAreDedupedAndExist() {
        let dest = tempDir("dest")
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = dest
        roots.customRoot = dest + "/"          // same folder, trailing slash
        let all = roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: dest))  // and again as LM Studio
        XCTAssertEqual(all.filter { $0 == dest }.count, 1)
        XCTAssertFalse(all.contains("/Volumes/Gone/models"))
        for r in all {
            var isDir: ObjCBool = false
            XCTAssertTrue(FileManager.default.fileExists(atPath: r, isDirectory: &isDir) && isDir.boolValue)
        }
    }

    /// The server's cap is 8 folders and it EXITS on the ninth, so the client
    /// must never build a list that long.
    func testScanRootsStayInsideTheServersOwnCap() {
        var roots = ModelRoots(defaults: defaults)
        roots.configuredDownloadRoot = tempDir("dest")
        roots.customRoot = tempDir("custom")
        XCTAssertLessThanOrEqual(roots.scanRoots(toolRoots: ToolModelRoots(lmStudio: tempDir("lms"))).count, ModelRoots.serverRootLimit)
        XCTAssertEqual(ModelRoots.serverRootLimit, 8)
    }

    // MARK: - The launch flags

    /// One `--model-dir` per folder. Before this the server was handed exactly
    /// one and everything else was invisible to `/v1/models`.
    func testEveryScanRootBecomesItsOwnModelDirFlag() {
        let a = tempDir("a"), b = tempDir("b")
        let args = ServerOptions().toCLIArgs(modelDirs: [a, b])
        let dirFlags = args.enumerated().filter { $0.element == "--model-dir" }.map { args[$0.offset + 1] }
        XCTAssertEqual(dirFlags, [a, b])
    }

    /// The single-dir call site keeps working unchanged — it is what every
    /// non-app launcher and every existing test passes.
    func testTheSingleDirectoryFormIsUnchanged() {
        let a = tempDir("a")
        let args = ServerOptions().toCLIArgs(modelDirOverride: a)
        XCTAssertEqual(args.filter { $0 == "--model-dir" }.count, 1)
        XCTAssertTrue(args.contains(a))
        // No folders at all emits no flag, rather than an empty one the arg
        // loop would reject.
        XCTAssertFalse(ServerOptions().toCLIArgs(modelDirs: []).contains("--model-dir"))
        XCTAssertFalse(ServerOptions().toCLIArgs(modelDirs: [""]).contains("--model-dir"))
    }

    // MARK: - Launch wiring

    /// A model outside every library folder still gets its own `--model-dir`,
    /// so it reaches `/v1/models` and can be hot-swapped.
    ///
    /// This used to be an either/or: `discoveryModelDir` returned the library
    /// root OR the outside model's parent, never both, so selecting a model in
    /// the custom folder made the whole `~/.mlx-serve/models` library vanish
    /// from `/v1/models`, and selecting a library model made the custom one
    /// unreachable (hot-switch 404, falling back to a full restart).
    func testAModelOutsideEveryLibraryFolderStillGetsScanned() {
        let lib = tempDir("lib")
        let elsewhere = tempDir("elsewhere")
        let model = elsewhere + "/org/some-model"
        try? FileManager.default.createDirectory(atPath: model, withIntermediateDirectories: true)

        let dirs = ServerManager.launchModelDirs(selectedModel: model, roots: [lib])
        XCTAssertTrue(dirs.contains(lib), "the library must not drop out")
        XCTAssertTrue(dirs.contains(elsewhere + "/org"), "the outside model's own folder must be scanned")
    }

    /// A model already covered by a library folder adds nothing — otherwise
    /// every launch would carry a redundant flag for its own org subfolder,
    /// and two roots holding the same id is what the server has to de-dup.
    ///
    /// Ported from `ServerDiscoveryModelDirTests`, which pinned the predecessor
    /// `discoveryModelDir` — deleted with it, since a function nothing calls
    /// keeps passing forever.
    func testAModelInsideALibraryFolderAddsNoExtraFlag() {
        let lib = tempDir("lib")
        let model = lib + "/org/some-model"
        try? FileManager.default.createDirectory(atPath: model, withIntermediateDirectories: true)
        XCTAssertEqual(ServerManager.launchModelDirs(selectedModel: model, roots: [lib]), [lib])
        // A DIFFERENT org under the same library is still covered — the whole
        // root is scanned two levels deep, which is the fix that made
        // `/v1/models` match the app's own dropdown.
        let other = lib + "/other-org/another"
        try? FileManager.default.createDirectory(atPath: other, withIntermediateDirectories: true)
        XCTAssertEqual(ServerManager.launchModelDirs(selectedModel: other, roots: [lib]), [lib])
        // A trailing slash on the root must not defeat the under-root check.
        XCTAssertEqual(ServerManager.launchModelDirs(selectedModel: model, roots: [lib + "/"]), [lib + "/"])
        // No selection at all is just the library.
        XCTAssertEqual(ServerManager.launchModelDirs(selectedModel: "", roots: [lib]), [lib])
    }

    /// An HF-cache GGUF quant is a `<quant>.gguf` SYMLINK to an extensionless
    /// blob (`blobs/<sha256>`). The server routes GGUF by the `.gguf`
    /// extension, so resolving the leaf symlink hands it the blob path, which
    /// falls through to the MLX directory loader and dies `error: NotDir`
    /// (issue #158). The launch path must keep the symlink's own name; the
    /// filesystem follows the link at open time.
    func testAnHFCacheGgufSymlinkKeepsItsExtensionInTheLaunchPath() {
        let repo = tempDir("hub") + "/models--unsloth--Qwen3.5-4B-GGUF"
        let blobs = repo + "/blobs"
        let snap = repo + "/snapshots/abc123"
        let fm = FileManager.default
        try? fm.createDirectory(atPath: blobs, withIntermediateDirectories: true)
        try? fm.createDirectory(atPath: snap, withIntermediateDirectories: true)
        let blob = blobs + "/00fe7986ff5f6b46"
        fm.createFile(atPath: blob, contents: Data("GGUF".utf8))
        let link = snap + "/Qwen3.5-4B-Q4_K_M.gguf"
        try? fm.createSymbolicLink(atPath: link, withDestinationPath: "../../blobs/00fe7986ff5f6b46")

        XCTAssertEqual(ServerManager.launchModelPath(link), link)
        // A safetensors model dir (the MLX case) passes through unchanged too.
        let dir = tempDir("lib") + "/org/model"
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        XCTAssertEqual(ServerManager.launchModelPath(dir), dir)
    }

    /// The extra folder can never push the list past what the server accepts —
    /// `main.zig` EXITS on the ninth `--model-dir`, so an over-long list is a
    /// server that does not start.
    func testTheOutsideModelCannotPushPastTheServerCap() {
        let roots = (0..<ModelRoots.serverRootLimit).map { tempDir("r\($0)") }
        let outside = tempDir("outside") + "/org/m"
        let dirs = ServerManager.launchModelDirs(selectedModel: outside, roots: roots)
        XCTAssertEqual(dirs.count, ModelRoots.serverRootLimit)
    }

    /// Nothing may rebuild a single-root `--model-dir` behind the shared
    /// builder's back: that is exactly how the custom folder came to be listed
    /// by the picker and invisible to the server. Needles split so this test's
    /// own source can't satisfy the scan.
    func testNoLaunchSiteHardcodesASingleModelDir() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let sources = root.appendingPathComponent("Sources/MLXServe")
        let files = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" } ?? []
        XCTAssertFalse(files.isEmpty, "found no sources to scan")

        let flagLiteral = "\"--model" + "-dir\""
        for f in files {
            let text = try String(contentsOf: f, encoding: .utf8)
            // Only `toCLIArgs` may spell the flag.
            if text.contains(flagLiteral) {
                XCTAssertEqual(f.lastPathComponent, "ServerOptions.swift",
                               "\(f.lastPathComponent) builds --model-dir itself")
            }
            // And nothing may BUILD the historical path itself. Scanning for
            // the bare string would flag prose (two panes name the folder in
            // their download instructions, correctly); what must not exist is
            // a second construction of the value, which is how `ServerManager`
            // and `DownloadManager` came to hold separate copies of it.
            let construction = "NSString(string: \"~/.mlx-serve/mod" + "els\")"
            if text.contains(construction) {
                XCTAssertEqual(f.lastPathComponent, "ModelRoots.swift",
                               "\(f.lastPathComponent) builds the models root itself")
            }
        }
    }

    // MARK: - Build gating

    /// Under MAS the server helper is signed `com.apple.security.inherit`,
    /// which inherits the app's CONTAINER but not its security-scoped grants —
    /// so the app could pick a folder the server then cannot read the weights
    /// out of. The setting is Developer-ID-only until that has an answer; the
    /// control is hidden on the same flag that hides it.
    func testTheDefaultFolderSettingIsDeveloperIdOnly() {
        XCTAssertTrue(BuildFeatures.developerID.customModelFolders)
        XCTAssertFalse(BuildFeatures.mas.customModelFolders)
    }

    /// ...and the gate is on the ROOTS, not just on the UI: a value left in
    /// defaults by a Developer ID build (same user, same defaults domain, an
    /// App Store install later) must not move the destination there.
    func testAStoredFolderIsIgnoredWhereTheFeatureIsOff() {
        var roots = ModelRoots(defaults: defaults, allowCustomFolders: true)
        let dir = tempDir("dest")
        roots.configuredDownloadRoot = dir
        XCTAssertEqual(roots.downloadRoot, dir)

        let gated = ModelRoots(defaults: defaults, allowCustomFolders: false)
        XCTAssertEqual(gated.downloadRoot, NSString(string: "~/.mlx-serve/models").expandingTildeInPath)
        XCTAssertFalse(gated.scanRoots(toolRoots: ToolModelRoots(lmStudio: nil)).contains(dir))
    }
}
