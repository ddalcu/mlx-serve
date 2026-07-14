import XCTest
@testable import MLXCore

/// GGUF repos ship MANY quants in ONE folder (`Qwen3.5-4B-Q4_K_M.gguf`,
/// `-IQ4_NL.gguf`, `-Q8_0.gguf`, …). Three bugs followed from treating a repo
/// as a single model:
///
/// 1. `existingModelDir` gated on `config.json`, which a GGUF-only download
///    never has — so `isReady`'s GGUF fast-path underneath it was dead code and
///    a downloaded quant stopped resolving the moment the in-memory download
///    row went away (app relaunch).
/// 2. Discovery picked the alphabetically-smallest `.gguf` and dropped the
///    rest, so the tray picker could only ever offer ONE quant per repo.
/// 3. Deleting (or cancelling) one quant removed the whole folder, taking every
///    sibling quant with it.
///
/// These pin the per-quant model: one `LocalModel` per `.gguf`, per-file
/// deletes, and a Discover menu that keeps offering the quants you don't have.
final class GgufQuantTests: XCTestCase {
    private var tempRoot: String!

    override func setUpWithError() throws {
        tempRoot = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-gguf-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: tempRoot)
    }

    /// A GGUF quant folder as `downloadGguf` leaves it: the `.gguf` files only —
    /// no config.json, no tokenizer, no safetensors.
    @discardableResult
    private func makeGgufRepo(_ repoId: String, files: [String]) throws -> String {
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        for f in files {
            let path = (dir as NSString).appendingPathComponent(f)
            // Non-trivial size: `isReady`/discovery ignore sub-1MB stubs.
            let data = Data(count: 2_000_000)
            try data.write(to: URL(fileURLWithPath: path))
        }
        return dir
    }

    // MARK: - Disk resolution

    func testExistingModelDirResolvesGgufOnlyFolder() throws {
        // No config.json — a GGUF download writes exactly one file. Before the
        // fix this returned nil, so `isReady` was false for a model that was
        // fully on disk and the Discover row offered "Download" again.
        let dir = try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: ["Qwen3.5-4B-Q4_K_M.gguf"])
        XCTAssertEqual(DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "unsloth/Qwen3.5-4B-GGUF"), dir)
    }

    func testExistingModelDirIgnoresMmprojOnlyFolder() throws {
        // An mmproj sidecar is a CLIP encoder, not a language model — a folder
        // holding only that is not a resolvable model.
        try makeGgufRepo("acme/vision-bits", files: ["mmproj-F16.gguf"])
        XCTAssertNil(DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "acme/vision-bits"))
    }

    func testDownloadedGgufFilesListsEveryQuantSorted() throws {
        try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: [
            "Qwen3.5-4B-Q8_0.gguf",
            "Qwen3.5-4B-Q4_K_M.gguf",
            "mmproj-F16.gguf",          // sidecar, never a quant
        ])
        // A half-finished transfer must not read as an on-disk quant.
        let partial = (DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: "unsloth/Qwen3.5-4B-GGUF") as NSString)
            .appendingPathComponent("Qwen3.5-4B-IQ4_NL.gguf.partial")
        try Data(count: 1_000).write(to: URL(fileURLWithPath: partial))

        XCTAssertEqual(
            DownloadManager.downloadedGgufFiles(rootDir: tempRoot, repoId: "unsloth/Qwen3.5-4B-GGUF"),
            ["Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B-Q8_0.gguf"]
        )
    }

    func testDownloadedGgufFilesEmptyForSafetensorsRepo() {
        XCTAssertEqual(DownloadManager.downloadedGgufFiles(rootDir: tempRoot, repoId: "nobody/missing"), [])
    }

    /// A GGUF folder can ship non-LLM `.gguf` SIDECARS beside the quants — the
    /// mmproj CLIP projection, and (live, in `~/.mlx-serve/models/qwen3-tts-0.6b`)
    /// a 341 MB speech-tokenizer file. Listing every non-mmproj `.gguf` as a quant
    /// would offer `qwen3-tts-tokenizer-f16` as a chat model you could select and
    /// fail to load. Discovery's old "pick the alphabetically-smallest" rule hid
    /// this by accident; enumerating them all exposes it.
    func testTokenizerSidecarIsNotAQuant() throws {
        XCTAssertFalse(DownloadManager.isSupportedGguf("qwen3-tts-tokenizer-f16.gguf"))
        XCTAssertFalse(DownloadManager.isSupportedGguf("mmproj-F16.gguf"))
        XCTAssertTrue(DownloadManager.isSupportedGguf("qwen3-tts-0.6b-f16.gguf"))
        XCTAssertTrue(DownloadManager.isSupportedGguf("Qwen3.5-4B-Q4_K_M.gguf"))

        let dir = try makeGgufRepo("local/qwen3-tts-0.6b", files: [
            "qwen3-tts-0.6b-f16.gguf",
            "qwen3-tts-tokenizer-f16.gguf",
        ])
        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "qwen3-tts-0.6b", idKey: "qwen3-tts-0.6b", source: .mlxServe
        )
        XCTAssertEqual(models.map(\.quantFile), ["qwen3-tts-0.6b-f16.gguf"],
                       "the tokenizer sidecar is not a selectable model")
    }

    // MARK: - Discovery: one model per quant

    func testDiscoveryEmitsOneModelPerQuant() throws {
        let dir = try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: [
            "Qwen3.5-4B-Q4_K_M.gguf",
            "Qwen3.5-4B-Q8_0.gguf",
            "mmproj-F16.gguf",
        ])
        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "unsloth/Qwen3.5-4B-GGUF",
            idKey: "unsloth/Qwen3.5-4B-GGUF", source: .mlxServe
        )

        XCTAssertEqual(models.count, 2, "each quant is separately selectable; the mmproj sidecar is not a model")
        // Each points at its OWN file — that path is what the server loads, so
        // picking a quant in the tray needs no server-side change.
        XCTAssertEqual(models.map(\.path).sorted(), [
            (dir as NSString).appendingPathComponent("Qwen3.5-4B-Q4_K_M.gguf"),
            (dir as NSString).appendingPathComponent("Qwen3.5-4B-Q8_0.gguf"),
        ])
        XCTAssertEqual(Set(models.map(\.id)).count, 2, "ids must be unique or SwiftUI collapses the rows")
        XCTAssertEqual(models.map(\.quantFile).compactMap { $0 }.sorted(),
                       ["Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B-Q8_0.gguf"])
        XCTAssertTrue(models.allSatisfy { $0.isChatPickable })
    }

    func testDiscoveryLabelsEachQuantDistinctly() throws {
        let dir = try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: [
            "Qwen3.5-4B-Q4_K_M.gguf",
            "Qwen3.5-4B-Q8_0.gguf",
        ])
        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "unsloth/Qwen3.5-4B-GGUF",
            idKey: "unsloth/Qwen3.5-4B-GGUF", source: .mlxServe
        ).sorted { $0.displayLabel < $1.displayLabel }

        XCTAssertEqual(models.map(\.displayLabel), [
            "unsloth/Qwen3.5-4B-GGUF · Q4_K_M",
            "unsloth/Qwen3.5-4B-GGUF · Q8_0",
        ])
        // `name` stays the repo name — filters/grouping key off it.
        XCTAssertTrue(models.allSatisfy { $0.name == "unsloth/Qwen3.5-4B-GGUF" })
    }

    func testSafetensorsDiscoveryStillReturnsExactlyOneModel() throws {
        let dir = (tempRoot as NSString).appendingPathComponent("mlx-community/demo-4bit")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        for (f, body) in [
            ("config.json", "{\"model_type\":\"qwen3\"}"),
            ("tokenizer.json", "{}"),
        ] {
            try body.write(toFile: (dir as NSString).appendingPathComponent(f), atomically: true, encoding: .utf8)
        }
        try Data(count: 2_000_000).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("model.safetensors")))

        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "mlx-community/demo-4bit",
            idKey: "mlx-community/demo-4bit", source: .mlxServe
        )
        XCTAssertEqual(models.count, 1)
        XCTAssertNil(models[0].quantFile, "an MLX checkpoint has no per-file quant variant")
        XCTAssertEqual(models[0].displayLabel, "mlx-community/demo-4bit", "no suffix when there's no quant file")
    }

    // MARK: - Picker labels

    /// macOS `.menu` Pickers key the checkmark by item TITLE. Two quants of one
    /// repo share a `name`, so the title must come from `displayLabel` — else
    /// both rows render selected and the user can't tell which quant is loaded.
    func testDuplicateNamesIsComputedOnDisplayLabelSoQuantsStayDistinct() {
        let a = LocalModel(id: "mlxServe:r#a.gguf", name: "unsloth/Q-GGUF", path: "/m/a.gguf",
                           sizeFormatted: "2 GB", modelType: "gguf", source: .mlxServe, kind: .base,
                           quantFile: "Q-Q4_K_M.gguf")
        let b = LocalModel(id: "mlxServe:r#b.gguf", name: "unsloth/Q-GGUF", path: "/m/b.gguf",
                           sizeFormatted: "4 GB", modelType: "gguf", source: .mlxServe, kind: .base,
                           quantFile: "Q-Q8_0.gguf")
        XCTAssertTrue(LocalModel.duplicateNames(in: [a, b]).isEmpty,
                      "distinct quant labels are already unique — no engine suffix needed")

        // A real collision (same repo name, one GGUF one MLX) still gets tagged.
        let mlx = LocalModel(id: "mlxServe:x", name: "demo", path: "/m/demo",
                             sizeFormatted: "4 GB", modelType: "qwen3", source: .mlxServe, kind: .base)
        let gguf = LocalModel(id: "mlxServe:y#demo.gguf", name: "demo", path: "/m/demo.gguf",
                              sizeFormatted: "4 GB", modelType: "gguf", source: .mlxServe, kind: .base)
        XCTAssertEqual(LocalModel.duplicateNames(in: [mlx, gguf]), ["demo"])
    }

    // MARK: - Per-quant delete

    func testDeletingOneQuantKeepsItsSiblings() throws {
        let dir = try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: [
            "Qwen3.5-4B-Q4_K_M.gguf",
            "Qwen3.5-4B-Q8_0.gguf",
        ])
        let victim = (dir as NSString).appendingPathComponent("Qwen3.5-4B-Q4_K_M.gguf")
        let survivor = (dir as NSString).appendingPathComponent("Qwen3.5-4B-Q8_0.gguf")

        XCTAssertTrue(DownloadManager.removeGgufQuant(at: victim, roots: [tempRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: victim))
        XCTAssertTrue(FileManager.default.fileExists(atPath: survivor), "a sibling quant must survive")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir), "the folder still holds a quant")
    }

    func testDeletingTheLastQuantRemovesTheFolderAndPrunesTheAuthorDir() throws {
        let dir = try makeGgufRepo("unsloth/Qwen3.5-4B-GGUF", files: ["Qwen3.5-4B-Q4_K_M.gguf"])
        let only = (dir as NSString).appendingPathComponent("Qwen3.5-4B-Q4_K_M.gguf")

        XCTAssertTrue(DownloadManager.removeGgufQuant(at: only, roots: [tempRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir), "empty repo folder goes with the last quant")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: (tempRoot as NSString).appendingPathComponent("unsloth")),
            "the now-empty author dir is pruned"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempRoot), "never climb past a scan root")
    }

    func testDeletingTheLastQuantKeepsAnMmprojSidecarsFolderIntact() throws {
        // A vision GGUF folder holds the LLM quant + its mmproj encoder. Deleting
        // the only LLM quant leaves nothing loadable, so the folder (sidecar and
        // all) goes — an orphan mmproj is dead weight.
        let dir = try makeGgufRepo("acme/vl-gguf", files: ["vl-Q4_K_M.gguf", "mmproj-F16.gguf"])
        XCTAssertTrue(DownloadManager.removeGgufQuant(
            at: (dir as NSString).appendingPathComponent("vl-Q4_K_M.gguf"), roots: [tempRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir))
    }

    // MARK: - Discover menu

    func testQuantMenuSplitsOnDiskFromAvailable() {
        let menu = GgufQuantMenuModel.build(
            remote: ["Q-IQ4_NL.gguf", "Q-Q4_K_M.gguf", "Q-Q8_0.gguf", "mmproj-F16.gguf"],
            onDisk: ["Q-Q4_K_M.gguf"]
        )
        XCTAssertEqual(menu.onDisk.map(\.filename), ["Q-Q4_K_M.gguf"])
        XCTAssertEqual(menu.onDisk.map(\.label), ["Q4_K_M"])
        XCTAssertEqual(menu.available.map(\.filename), ["Q-IQ4_NL.gguf", "Q-Q8_0.gguf"],
                       "mmproj sidecars are not offerable quants")
    }

    func testQuantMenuKeepsAnOnDiskQuantTheRepoNoLongerLists() {
        // The repo re-quantized and dropped a file we already have. It's still
        // on disk and still usable — never silently hide it.
        let menu = GgufQuantMenuModel.build(remote: ["Q-Q8_0.gguf"], onDisk: ["Q-Q2_K.gguf"])
        XCTAssertEqual(menu.onDisk.map(\.filename), ["Q-Q2_K.gguf"])
        XCTAssertEqual(menu.available.map(\.filename), ["Q-Q8_0.gguf"])
    }

    /// The whole point of the fix: a repo with one quant downloaded still shows
    /// the OTHER quants as downloadable. Before, the row collapsed to
    /// "✓ On disk" + trash and there was no way back to the menu.
    func testDownloadedQuantDoesNotHideTheRest() {
        let menu = GgufQuantMenuModel.build(
            remote: ["Q-Q4_K_M.gguf", "Q-Q8_0.gguf"], onDisk: ["Q-Q4_K_M.gguf"]
        )
        XCTAssertFalse(menu.available.isEmpty, "a downloaded quant must not close the door on the others")
    }

    func testQuantMenuButtonLabelStates() {
        XCTAssertEqual(GgufQuantMenuModel.buttonLabel(onDisk: [], failed: false, hasPartial: false), "Download")
        XCTAssertEqual(GgufQuantMenuModel.buttonLabel(onDisk: [], failed: false, hasPartial: true), "Resume")
        XCTAssertEqual(GgufQuantMenuModel.buttonLabel(onDisk: [], failed: true, hasPartial: false), "Retry")
        // One quant: name it, so the row says WHICH quant you have.
        XCTAssertEqual(
            GgufQuantMenuModel.buttonLabel(onDisk: [.init(filename: "Q-Q4_K_M.gguf", label: "Q4_K_M")],
                                           failed: false, hasPartial: false),
            "✓ Q4_K_M"
        )
        XCTAssertEqual(
            GgufQuantMenuModel.buttonLabel(onDisk: [.init(filename: "a.gguf", label: "Q4_K_M"),
                                                    .init(filename: "b.gguf", label: "Q8_0")],
                                           failed: false, hasPartial: false),
            "✓ 2 on disk"
        )
        // A failed retry of a SECOND quant must not erase the fact that the
        // first one is sitting on disk and usable.
        XCTAssertEqual(
            GgufQuantMenuModel.buttonLabel(onDisk: [.init(filename: "a.gguf", label: "Q4_K_M")],
                                           failed: true, hasPartial: true),
            "✓ Q4_K_M"
        )
    }
}
