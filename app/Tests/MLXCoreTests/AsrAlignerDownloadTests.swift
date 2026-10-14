import XCTest
@testable import MLXCore

/// A Qwen3-ASR download brings the forced aligner the server times words with, and the diarizer it labels speakers with.
final class AsrAlignerDownloadTests: XCTestCase {
    private func pack(_ config: String?) throws -> String {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).path
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        if let config { try config.write(toFile: (dir as NSString).appendingPathComponent("config.json"), atomically: true, encoding: .utf8) }
        return dir
    }

    /// Both packs say `qwen3_asr` at the root; only the aligner's thinker tells them apart.
    func testTheThinkerTellsTheAsrFromItsAligner() throws {
        let asr = try pack(#"{"model_type":"qwen3_asr","thinker_config":{"model_type":"qwen3_asr"}}"#)
        let aligner = try pack(#"{"model_type":"qwen3_asr","thinker_config":{"model_type":"qwen3_forced_aligner"}}"#)
        XCTAssertEqual(DownloadManager.qwen3AsrThinker(asr), "qwen3_asr")
        XCTAssertNil(DownloadManager.qwen3AsrThinker(try pack(#"{"model_type":"qwen3"}"#)))
        XCTAssertNil(DownloadManager.qwen3AsrThinker(try pack(nil)))
        // The scan lists the aligner under its own type, which no chat picker offers.
        XCTAssertNil(DownloadManager.markerModelType(inDir: asr))
        XCTAssertEqual(DownloadManager.markerModelType(inDir: aligner), "qwen3_forced_aligner")
        XCTAssertEqual(DownloadManager.alignerRepo, "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
        XCTAssertEqual(DownloadManager.diarizerRepo, "mlx-community/Nemotron-3-Diarization-8bit")
    }
}
