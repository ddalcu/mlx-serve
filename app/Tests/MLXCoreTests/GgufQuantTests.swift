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

    /// A sharded/subfoldered GGUF repo as HF lays out a large quant and
    /// `downloadGguf` writes it: `<model>/<quant>/<quant>-NNNNN-of-MMMMM.gguf`.
    /// `quants` maps a quant subfolder name → its shard basenames.
    @discardableResult
    private func makeShardedGgufRepo(_ repoId: String, quants: [String: [String]]) throws -> String {
        let dir = DownloadManager.newLayoutDir(rootDir: tempRoot, repoId: repoId)
        for (sub, shards) in quants {
            let subDir = (dir as NSString).appendingPathComponent(sub)
            try FileManager.default.createDirectory(atPath: subDir, withIntermediateDirectories: true)
            for f in shards {
                let path = (subDir as NSString).appendingPathComponent(f)
                try Data(count: 2_000_000).write(to: URL(fileURLWithPath: path))
            }
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
        // The MTP draft head (antirez/deepseek-v4-gguf) is a speculative-decode
        // dependency, not a selectable chat quant — it was leaking into the
        // Discover dropdown as a bogus "Q4K" entry.
        XCTAssertFalse(DownloadManager.isSupportedGguf("DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"))
        XCTAssertTrue(DownloadManager.isSupportedGguf("qwen3-tts-0.6b-f16.gguf"))
        XCTAssertTrue(DownloadManager.isSupportedGguf("Qwen3.5-4B-Q4_K_M.gguf"))
        // A real chat quant whose long scheme name merely contains "mtp" is fine.
        XCTAssertTrue(DownloadManager.isSupportedGguf("DeepSeek-V4-Flash-IQ2XXS-chat-v2.gguf"))

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

    // MARK: - Sharded repos on disk

    func testDownloadedGgufPathsFindsEveryShard() throws {
        try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
        ])
        XCTAssertEqual(
            DownloadManager.downloadedGgufPaths(rootDir: tempRoot, repoId: "vcruz305/Hy3-GGUF"),
            ["Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf"],
            "shards live in a subfolder — the walk must recurse and return repo-relative paths"
        )
    }

    func testExistingModelDirResolvesShardedFolder() throws {
        let dir = try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
        ])
        XCTAssertEqual(DownloadManager.existingModelDir(rootDir: tempRoot, repoId: "vcruz305/Hy3-GGUF"), dir,
                       "a folder whose only .gguf files are nested shards still holds a model")
    }

    func testShardedDiscoveryEmitsOneModelPointingAtThePrimaryShard() throws {
        let dir = try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
            "Hy3-IQ2_M": ["Hy3-IQ2_M-00001-of-00003.gguf", "Hy3-IQ2_M-00002-of-00003.gguf", "Hy3-IQ2_M-00003-of-00003.gguf"],
        ])
        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "vcruz305/Hy3-GGUF",
            idKey: "vcruz305/Hy3-GGUF", source: .mlxServe
        ).sorted { $0.displayLabel < $1.displayLabel }

        XCTAssertEqual(models.count, 2, "one model per shard GROUP, not per shard file")
        XCTAssertEqual(models.map(\.displayLabel), [
            "vcruz305/Hy3-GGUF · IQ1_M",
            "vcruz305/Hy3-GGUF · IQ2_M",
        ])
        // path = the primary (-00001) shard — llama.cpp auto-loads the rest.
        XCTAssertEqual(models[0].path, (dir as NSString).appendingPathComponent("Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf"))
        XCTAssertEqual(models[1].path, (dir as NSString).appendingPathComponent("Hy3-IQ2_M/Hy3-IQ2_M-00001-of-00003.gguf"))
        XCTAssertTrue(models.allSatisfy { $0.isChatPickable })
        XCTAssertEqual(Set(models.map(\.id)).count, 2, "ids stay unique across sharded quants")
    }

    func testDeletingAShardedQuantRemovesEveryShardAndTheSubfolder() throws {
        let dir = try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
            "Hy3-IQ2_M": ["Hy3-IQ2_M-00001-of-00002.gguf", "Hy3-IQ2_M-00002-of-00002.gguf"],
        ])
        let primary = (dir as NSString).appendingPathComponent("Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf")

        XCTAssertTrue(DownloadManager.removeGgufQuant(at: primary, roots: [tempRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("Hy3-IQ1_M")),
                       "the whole quant subfolder (every shard) is gone")
        XCTAssertTrue(FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("Hy3-IQ2_M")),
                      "a sibling quant subfolder survives")
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir))
    }

    /// The discovery walk calls `makeLocalModels` on an AUTHOR dir first and
    /// recurses into children only when it returns []. A sharded model's shards
    /// live TWO levels below the author (`author/model/quant/shard`), so the
    /// author-level call must find nothing — otherwise every sharded repo under
    /// one author collapses into a single bogus author-named model and the real
    /// per-repo entries are never produced.
    func testMakeLocalModelsOnAnAuthorDirFindsNothingSoDiscoveryRecurses() throws {
        try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
        ])
        // A second sharded repo under the SAME author — must not merge with the first.
        try makeShardedGgufRepo("vcruz305/Other-GGUF", quants: [
            "Other-Q4_K_M": ["Other-Q4_K_M-00001-of-00002.gguf", "Other-Q4_K_M-00002-of-00002.gguf"],
        ])
        let author = (tempRoot as NSString).appendingPathComponent("vcruz305")
        let atAuthor = DownloadManager.makeLocalModels(
            atDir: author, displayName: "vcruz305", idKey: "vcruz305", source: .mlxServe)
        XCTAssertTrue(atAuthor.isEmpty, "an author dir is not a model — recurse into it, never merge its repos")
    }

    /// An interrupted split download (some shards present) is not a loadable
    /// model — it must not appear in the tray; it stays a resumable partial.
    func testMakeLocalModelsSkipsIncompleteShardedQuant() throws {
        let dir = try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf"],   // only 1 of 2
        ])
        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "vcruz305/Hy3-GGUF", idKey: "vcruz305/Hy3-GGUF", source: .mlxServe)
        XCTAssertTrue(models.isEmpty, "an incomplete split isn't a loadable model")
    }

    func testDeletingTheLastShardedQuantRemovesTheRepoAndPrunesTheAuthorDir() throws {
        let dir = try makeShardedGgufRepo("vcruz305/Hy3-GGUF", quants: [
            "Hy3-IQ1_M": ["Hy3-IQ1_M-00001-of-00002.gguf", "Hy3-IQ1_M-00002-of-00002.gguf"],
        ])
        let primary = (dir as NSString).appendingPathComponent("Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf")

        XCTAssertTrue(DownloadManager.removeGgufQuant(at: primary, roots: [tempRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir), "empty repo folder goes with the last quant")
        XCTAssertFalse(FileManager.default.fileExists(atPath: (tempRoot as NSString).appendingPathComponent("vcruz305")),
                       "the now-empty author dir is pruned")
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempRoot), "never climb past a scan root")
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

    // MARK: - MTP auto-download resolver

    func testMtpSidecarPathFindsTheDraftHead() {
        // Real antirez/deepseek-v4-gguf tree: the MTP draft head is fetched
        // alongside whichever chat quant the user picks.
        let files = [
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
            "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf",
            "imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat",
            "README.md",
        ]
        XCTAssertEqual(DownloadManager.mtpSidecarPath(in: files), "DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf")
    }

    func testMtpSidecarPathNilWhenNoDraftHead() {
        XCTAssertNil(DownloadManager.mtpSidecarPath(in: ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf", "README.md"]))
    }

    // MARK: - Sharded / subfoldered quants (a quant = a shard group)
    //
    // Large GGUFs (anything HF splits over ~50 GB) ship each quant as a
    // SUBFOLDER of split shards: `Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf`,
    // `…-00002-of-00002.gguf`. The single-file assumption dropped all of them.
    // A quant is now a shard group: primary = the `-00001` shard (the path the
    // server loads), completeness = present-shard-count == MMMMM.

    func testGroupQuantsFoldsShardsIntoOneQuant() {
        let quants = GgufQuant.groupQuants([
            "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
            "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
        ])
        XCTAssertEqual(quants.count, 1, "the two shards are ONE quant")
        XCTAssertEqual(quants[0].label, "IQ1_M")
        XCTAssertEqual(quants[0].filename, "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
                       "primary = the -00001 shard (the path the server loads)")
        XCTAssertEqual(quants[0].shards.count, 2)
        XCTAssertEqual(quants[0].allFiles.sorted(), [
            "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
            "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
        ])
        XCTAssertTrue(quants[0].isComplete, "both shards present ⇒ complete")
    }

    func testGroupQuantsSeparatesDistinctSubfolderedQuants() {
        let quants = GgufQuant.groupQuants([
            "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
            "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
            "Hy3-IQ2_M/Hy3-IQ2_M-00001-of-00003.gguf",
            "Hy3-IQ2_M/Hy3-IQ2_M-00002-of-00003.gguf",
            "Hy3-IQ2_M/Hy3-IQ2_M-00003-of-00003.gguf",
        ]).sorted { $0.label < $1.label }
        XCTAssertEqual(quants.map(\.label), ["IQ1_M", "IQ2_M"])
        XCTAssertEqual(quants[0].shards.count, 2)
        XCTAssertEqual(quants[1].shards.count, 3)
    }

    // MARK: - Label disambiguation (colliding quant tokens)
    //
    // antirez/deepseek-v4-gguf encodes tier (Flash vs Pro), quant scheme, AND an
    // imatrix flag in each filename, but `quantLabel` keeps only the quant token
    // — so four distinct IQ2XXS files all showed as "IQ2XXS" in the dropdown with
    // no way to tell them apart. When labels collide, the distinguishing filename
    // tokens (tier / imatrix prioritized) are appended.

    func testGroupQuantsDisambiguatesFourWayCollision() {
        let quants = GgufQuant.groupQuants([
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
            "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf",
            "DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct-imatrix.gguf",
            "DeepSeek-V4-Pro-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-Instruct.gguf",
        ])
        XCTAssertEqual(quants.count, 4)
        XCTAssertEqual(Set(quants.map(\.label)), [
            "IQ2XXS · Flash · imatrix",
            "IQ2XXS · Flash",
            "IQ2XXS · Pro · imatrix",
            "IQ2XXS · Pro",
        ])
    }

    func testGroupQuantsImatrixOnlyDifference() {
        // The base (no imatrix) keeps the plain label; only the imatrix one is tagged.
        let quants = GgufQuant.groupQuants([
            "Model-IQ2XXS-imatrix.gguf",
            "Model-IQ2XXS.gguf",
        ])
        XCTAssertEqual(Set(quants.map(\.label)), ["IQ2XXS · imatrix", "IQ2XXS"])
    }

    func testGroupQuantsNonCollidingLabelsStayPlain() {
        let quants = GgufQuant.groupQuants([
            "Model-Q4_K_M.gguf",
            "Model-Q8_0.gguf",
        ])
        XCTAssertEqual(quants.map(\.label), ["Q4_K_M", "Q8_0"], "unique labels get no suffix")
    }

    func testGroupQuantsCollisionWithNoTierOrQualityMarkerFallsBackToUniqueToken() {
        // Two files whose only difference is a non-tier, non-imatrix token still
        // get distinguished by that token.
        let quants = GgufQuant.groupQuants([
            "Model-Q4_K_M-chat.gguf",
            "Model-Q4_K_M-instruct.gguf",
        ])
        XCTAssertEqual(Set(quants.map(\.label)), ["Q4_K_M · chat", "Q4_K_M · instruct"])
    }

    func testGroupQuantsSingleFileUnchanged() {
        // A non-split quant is a group of one with an EMPTY shards list (the
        // single-file callers — the existing menu tests — depend on this).
        let quants = GgufQuant.groupQuants([
            "Qwen3.5-4B-Q8_0.gguf",
            "Qwen3.5-4B-Q4_K_M.gguf",
            "mmproj-F16.gguf",          // sidecar, dropped
        ])
        XCTAssertEqual(quants.map(\.filename), ["Qwen3.5-4B-Q4_K_M.gguf", "Qwen3.5-4B-Q8_0.gguf"])
        XCTAssertTrue(quants.allSatisfy { $0.shards.isEmpty }, "single-file ⇒ empty shard list")
        XCTAssertEqual(quants.map(\.allFiles), [["Qwen3.5-4B-Q4_K_M.gguf"], ["Qwen3.5-4B-Q8_0.gguf"]])
        XCTAssertTrue(quants.allSatisfy { $0.isComplete })
    }

    func testShardCountParsesTotal() {
        XCTAssertEqual(GgufQuant.shardCount(forName: "Hy3-IQ1_M-00001-of-00002.gguf"), 2)
        XCTAssertEqual(GgufQuant.shardCount(forName: "Hy3-IQ2_M/Hy3-IQ2_M-00003-of-00007.gguf"), 7)
        XCTAssertNil(GgufQuant.shardCount(forName: "Qwen3.5-4B-Q4_K_M.gguf"), "non-split ⇒ nil")
    }

    /// A partial group (1 of 2 shards on disk) is NOT on-disk in the menu —
    /// it's an interrupted download that should read as available/resume.
    func testPartialShardGroupIsNotOnDisk() {
        let menu = GgufQuantMenuModel.build(
            remote: [
                "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
                "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
            ],
            onDisk: ["Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf"]   // only 1 of 2
        )
        XCTAssertTrue(menu.onDisk.isEmpty, "an incomplete shard group isn't on disk")
        XCTAssertEqual(menu.available.map(\.label), ["IQ1_M"], "it's offered as available (resume)")
    }

    func testCompleteShardGroupIsOnDisk() {
        let menu = GgufQuantMenuModel.build(
            remote: [
                "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
                "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
                "Hy3-IQ2_M/Hy3-IQ2_M-00001-of-00002.gguf",
                "Hy3-IQ2_M/Hy3-IQ2_M-00002-of-00002.gguf",
            ],
            onDisk: [
                "Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf",
                "Hy3-IQ1_M/Hy3-IQ1_M-00002-of-00002.gguf",
            ]
        )
        XCTAssertEqual(menu.onDisk.map(\.label), ["IQ1_M"])
        XCTAssertEqual(menu.available.map(\.label), ["IQ2_M"], "the other quant is still downloadable")
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
