import XCTest
@testable import MLXCore

/// Pixal3D ships TRELLIS.2's pipeline.json, name and all; only its flows' own config (`image_attn_mode: proj`)
/// tells it apart. Twin of `model_discovery.peekPixal3dPipeline` and its completion marker.
final class Pixal3DDiscoveryTests: XCTestCase {
    func testAPixal3DPackIsToldFromTrellis2ByItsFlowsAndRoutesToThe3DPane() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("pixal3d-\(UUID().uuidString)")
        let ckpts = dir.appendingPathComponent("ckpts")
        try fm.createDirectory(at: ckpts, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        try #"{"name":"Trellis2ImageTo3DPipeline","args":{"models":{"sparse_structure_flow_model":"ckpts/ss_flow"}}}"#
            .write(to: dir.appendingPathComponent("pipeline.json"), atomically: true, encoding: .utf8)
        try #"{"args":{"image_attn_mode":"proj"}}"#
            .write(to: ckpts.appendingPathComponent("ss_flow.json"), atomically: true, encoding: .utf8)

        XCTAssertEqual(DownloadManager.markerModelType(inDir: dir.path), "pixal3d")
        XCTAssertTrue(DownloadManager.holdsWeightLayout(dir.path))
        XCTAssertTrue(isMediaModelType("pixal3d"))
        XCTAssertEqual(MediaModality(modelType: "pixal3d"), .mesh)
        // The texture decoder is the completion marker, as on the server.
        XCTAssertFalse(DownloadManager.holdsCompleteMediaPack(dir.path))
        try Data().write(to: ckpts.appendingPathComponent("tex_dec_next_dc_f16c32_fp16.safetensors"))
        XCTAssertTrue(DownloadManager.holdsCompleteMediaPack(dir.path))

        // The same pipeline.json over cross-attention flows is TRELLIS.2.
        try #"{"args":{"image_attn_mode":"cross"}}"#
            .write(to: ckpts.appendingPathComponent("ss_flow.json"), atomically: true, encoding: .utf8)
        XCTAssertEqual(DownloadManager.markerModelType(inDir: dir.path), "trellis2")
    }
}
