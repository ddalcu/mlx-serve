import XCTest
@testable import MLXCore

/// A DeepSeek-V4.1 pack's Engram tables stay on disk, so the pair's estimate leaves them out.
final class ResidentSizeTests: XCTestCase {
    func testEngramTensorsComeOffThePacksSize() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("resident-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let header = #"{"language_model.layers.1.engram.embed.weight":{"dtype":"U8","shape":[600],"data_offsets":[0,600]},"language_model.layers.1.attn.wkv.weight":{"dtype":"U8","shape":[400],"data_offsets":[600,1000]}}"#
        var data = Data()
        var len = UInt64(header.utf8.count)
        data.append(Data(bytes: &len, count: 8))
        data.append(Data(header.utf8))
        data.append(Data(count: 1000))
        try data.write(to: dir.appendingPathComponent("model-00001-of-00001.safetensors"))
        XCTAssertEqual(ResidentSize.engramBytes(inDir: dir.path), 600)
        let m = LocalModel(id: "x", name: "V41", path: dir.path, sizeFormatted: "", modelType: "deepseek_v41",
                           source: .mlxServe, kind: .base, sizeBytes: Int64(data.count))
        XCTAssertEqual(ResidentSize.bytes(of: m), Int64(data.count) - 600)
    }
}
