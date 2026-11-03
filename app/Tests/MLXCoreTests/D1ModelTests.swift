import XCTest
@testable import MLXCore

/// A D1 pack's root config.json says `lfm2_vl`, the same as a chat LFM2.5-VL; its `auto_map` names the card's own
/// decision class, which is what makes it a decision model. Server twin: the d1 plugin's `claims`.
final class D1ModelTests: XCTestCase {

    private func makeDir(config: String) throws -> (root: String, dir: String) {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "d1-\(UUID().uuidString)"
        let dir = (root as NSString).appendingPathComponent("LiquidAI/d1-3B")
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let files = ["config.json": config, "tokenizer.json": "{}"]
        for (name, body) in files {
            fm.createFile(atPath: (dir as NSString).appendingPathComponent(name), contents: Data(body.utf8))
        }
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("model.safetensors"),
                      contents: Data(count: Int(DownloadManager.minimumWeightBytes) + 1))
        return (root, dir)
    }

    func testADPackIsListedAsD1AndGetsTheDecisionsButton() throws {
        let (root, dir) = try makeDir(config: #"{"model_type": "lfm2_vl", "auto_map": {"AutoModel": "modeling_d1.D1Model"}}"#)
        defer { try? FileManager.default.removeItem(atPath: root) }

        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "LiquidAI/d1-3B", idKey: "LiquidAI/d1-3B", source: .mlxServe)
        let m = try XCTUnwrap(models.first)
        XCTAssertEqual(m.modelType, "d1")
        XCTAssertNil(m.defect)
        XCTAssertTrue(m.isSupportedArchitecture, "must not badge Unsupported")
        XCTAssertFalse(m.isChatPickable, "a decision model is not a chat model")
        XCTAssertTrue(isDecisionModelType(m.modelType), "gets the Decisions Use button")
    }

    func testAPlainLFM2VLPackStaysAChatModel() throws {
        let (root, dir) = try makeDir(config: #"{"model_type": "lfm2_vl"}"#)
        defer { try? FileManager.default.removeItem(atPath: root) }

        let models = DownloadManager.makeLocalModels(
            atDir: dir, displayName: "LiquidAI/LFM2.5-VL-3B", idKey: "LiquidAI/LFM2.5-VL-3B", source: .mlxServe)
        let m = try XCTUnwrap(models.first)
        XCTAssertEqual(m.modelType, "lfm2_vl")
        XCTAssertFalse(isDecisionModelType(m.modelType))
    }
}
