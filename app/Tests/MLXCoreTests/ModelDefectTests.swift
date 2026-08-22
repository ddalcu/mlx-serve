import XCTest
@testable import MLXCore

/// A directory holding a `config.json` and no usable weights is not a model.
///
/// Two of them sit in Osaurus' folder on the reporter's Mac. One had no
/// `.safetensors` at all and was silently DROPPED — invisible in the app while
/// the server happily registered it and would have died on load. The other
/// ships a 48 KB stub `.safetensors`, which passed the file-exists check and so
/// was offered as a real, selectable model.
///
/// Both are the same defect wearing different clothes, and hiding either one is
/// wrong: these folders are junk taking up a name in your library, and the only
/// useful thing the app can do is SAY SO and offer to delete them.
final class ModelDefectTests: XCTestCase {

    private var scratch: String!

    override func setUp() {
        super.setUp()
        scratch = NSTemporaryDirectory() + "ModelDefect-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: scratch, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: scratch)
        super.tearDown()
    }

    @discardableResult
    private func write(_ rel: String, bytes: Int) -> String {
        let p = (scratch as NSString).appendingPathComponent(rel)
        try? FileManager.default.createDirectory(atPath: (p as NSString).deletingLastPathComponent,
                                                 withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: p, contents: Data(count: bytes))
        return p
    }

    private func makeDir(_ rel: String) -> String {
        let dir = (scratch as NSString).appendingPathComponent(rel)
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        write("\(rel)/config.json", bytes: 0)
        try? #"{"model_type":"qwen3"}"#.write(toFile: (dir as NSString).appendingPathComponent("config.json"),
                                              atomically: true, encoding: .utf8)
        return dir
    }

    private func models(_ dir: String) -> [LocalModel] {
        DownloadManager.makeLocalModels(atDir: dir, displayName: "M", idKey: "M", source: .osaurus)
    }

    /// The healthy case must keep working: real weight bytes, no defect, and
    /// still selectable. A completeness check that flags good models is worse
    /// than no check at all.
    func testARealCheckpointHasNoDefect() throws {
        let dir = makeDir("good")
        write("good/model.safetensors", bytes: 4_000_000)
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertNil(m.defect)
        XCTAssertTrue(m.isChatPickable)
    }

    /// A FLUX.2 klein pack (mflux layout) keeps every weight under
    /// `transformer/`, `text_encoder/`, `vae/` and nothing at the root. It is a
    /// working image model, not an orphan.
    func testAPackWithWeightsOnlyInSubdirsHasNoDefect() throws {
        let dir = makeDir("klein")
        try? #"{"model_type":"flux2"}"#.write(toFile: (dir as NSString).appendingPathComponent("config.json"),
                                               atomically: true, encoding: .utf8)
        write("klein/transformer/diffusion_pytorch_model.safetensors", bytes: 4_000_000)
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertNil(m.defect, "weights in a subdir are still weights")
    }

    /// `LFM2.5-8B-A1B-MXFP8`: config and tokenizer, no weights at all. Was
    /// dropped from the list entirely — you could not see it, so you could not
    /// delete it.
    func testAConfigWithNoWeightsIsSurfacedAsAnOrphan() throws {
        let dir = makeDir("orphan")
        let m = try XCTUnwrap(models(dir).first, "a weightless dir must still be LISTED, not dropped")
        XCTAssertEqual(m.defect, .missingWeights)
    }

    /// `Hy3-preview-JANGTQ`: a 48 KB stub `.safetensors` satisfied the
    /// file-exists check, so it was offered as a real model.
    func testAStubWeightFileIsNotAModel() throws {
        let dir = makeDir("stub")
        write("stub/jangtq_runtime.safetensors", bytes: 48 * 1024)
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertEqual(m.defect, .missingWeights,
                       "a dir whose entire weight payload is 48 KB cannot be a checkpoint")
    }

    /// An index names its shards, so a missing one is EXACT — no size guess.
    func testAnIndexNamingAMissingShardIsIncomplete() throws {
        let dir = makeDir("sharded")
        write("sharded/model-00001-of-00002.safetensors", bytes: 4_000_000)
        try #"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
            .write(toFile: (dir as NSString).appendingPathComponent("model.safetensors.index.json"),
                   atomically: true, encoding: .utf8)
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertEqual(m.defect, .missingShards)
    }

    /// All shards present is a complete model even though each is small.
    func testAnIndexWithEveryShardPresentIsClean() throws {
        let dir = makeDir("whole")
        write("whole/model-00001-of-00002.safetensors", bytes: 4_000_000)
        write("whole/model-00002-of-00002.safetensors", bytes: 4_000_000)
        try #"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
            .write(toFile: (dir as NSString).appendingPathComponent("model.safetensors.index.json"),
                   atomically: true, encoding: .utf8)
        XCTAssertNil(try XCTUnwrap(models(dir).first).defect)
    }

    /// An interrupted download leaves `.partial` behind. It outranks the weight
    /// check: the weights are thin because the transfer stopped, and the fix is
    /// resume-or-delete, not "this folder is junk".
    func testAnInterruptedDownloadIsReportedAsSuch() throws {
        let dir = makeDir("partial")
        write("partial/model.safetensors.partial", bytes: 1024)
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertEqual(m.defect, .interruptedDownload)
    }

    /// A `.partial` under a live transfer is progress, not an interruption:
    /// the listing drops that defect for the download's destination dir only.
    func testALiveDownloadIsNotReportedAsInterrupted() throws {
        let dir = makeDir("live")
        write("live/model.safetensors.partial", bytes: 1024)
        let other = makeDir("stale")
        write("stale/model.safetensors.partial", bytes: 1024)
        let all = models(dir) + models(other)
        let fixed = DownloadManager.clearingInFlightDefects(all, activeDirs: [dir])
        XCTAssertNil(fixed.first { $0.path == dir }?.defect)
        XCTAssertEqual(fixed.first { $0.path == other }?.defect, .interruptedDownload)
    }

    /// A defective folder must never reach a picker, and must always be
    /// deletable — including inside another tool's tree, which is the one place
    /// the read-only rule is wrong: nobody wants to keep a broken folder, and
    /// the app that owns it is not showing it either.
    func testDefectiveModelsAreUnpickableAndDeletableAnywhere() throws {
        let dir = makeDir("orphan2")
        let m = try XCTUnwrap(models(dir).first)
        XCTAssertFalse(m.isChatPickable)
        XCTAssertTrue(m.isDeletable, "junk in a foreign tree is still junk")
        XCTAssertNil(m.externalReadOnlyReason, "a deletable row must not also claim to be read-only")
        XCTAssertFalse(try XCTUnwrap(m.defect).label.isEmpty)
        XCTAssertFalse(try XCTUnwrap(m.defect).explanation.isEmpty)
    }

    /// Defects get their OWN group, ahead of nothing and behind everything —
    /// they are not a tool's models and must not pad a tool's section.
    func testDefectsGroupSeparatelyFromTheirSource() {
        let dir = makeDir("orphan3")
        let good = makeDir("good3")
        write("good3/model.safetensors", bytes: 4_000_000)
        let all = models(dir) + models(good)
        let groups = ModelBrowserUse.groupedBySource(all, filter: "")

        let defective = groups.first { $0.title == ModelBrowserUse.defectGroupTitle }
        XCTAssertEqual(defective?.models.count, 1)
        XCTAssertEqual(defective?.models.first?.defect, .missingWeights)
        XCTAssertEqual(groups.last?.title, ModelBrowserUse.defectGroupTitle,
                       "broken folders sort last — they are not what you came to pick")
        XCTAssertTrue(groups.contains { $0.title == LocalModelSource.osaurus.sectionTitle
                                        && $0.models.count == 1 })
    }
}

/// Deleting a broken folder that lives in ANOTHER tool's tree.
///
/// `deleteModel` bounds its removal by a root list, and that list was
/// `ownedRoots` — the download destination plus the built-in root. A trash
/// button offered on a folder outside both, backed by a call that stops at
/// both, is a button that does nothing: the exact failure this whole change
/// exists to stop shipping.
final class DefectDeletionTests: XCTestCase {

    private var scratch: String!

    override func setUp() {
        super.setUp()
        scratch = NSTemporaryDirectory() + "DefectDelete-\(UUID().uuidString)"
        try? FileManager.default.createDirectory(atPath: scratch, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(atPath: scratch)
        super.tearDown()
    }

    /// The removal itself, at the layer that actually touches disk.
    func testAFolderOutsideEveryOwnedRootIsRemoved() {
        let foreignRoot = (scratch as NSString).appendingPathComponent("MLXModels")
        let dir = (foreignRoot as NSString).appendingPathComponent("OsaurusAI/Broken")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: (dir as NSString).appendingPathComponent("config.json"),
                                       contents: Data("{}".utf8))

        XCTAssertTrue(DownloadManager.removeModelFiles(at: dir, roots: [foreignRoot]))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir))
    }

    /// The root list is a GUARD, not a scope: whatever is in it must survive.
    /// A foreign root absent from the list is one this call would delete.
    func testTheRootItselfIsNeverRemoved() {
        let foreignRoot = (scratch as NSString).appendingPathComponent("MLXModels")
        try? FileManager.default.createDirectory(atPath: foreignRoot, withIntermediateDirectories: true)

        XCTAssertFalse(DownloadManager.removeModelFiles(at: foreignRoot, roots: [foreignRoot]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: foreignRoot))
    }
}
