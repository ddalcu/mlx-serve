import XCTest
@testable import MLXCore

/// Speech-to-text runs on our server (Qwen3-ASR): which pack it uses, and what it reads back.
final class VoiceASRTests: XCTestCase {

    private func pack(_ type: String, downloading: Bool = false, defect: ModelDefect? = nil) -> LocalModel {
        var m = LocalModel(id: type, name: type, path: "/m/\(type)", sizeFormatted: "1 GB", modelType: type, source: .mlxServe, kind: .base)
        m.isDownloading = downloading
        m.defect = defect
        return m
    }

    func testOnlyAWholeQwen3ASRTranscribes() {
        XCTAssertEqual(VoiceASR.modelPath([pack("qwen3_forced_aligner"), pack("qwen3_asr")]), "/m/qwen3_asr")
        XCTAssertNil(VoiceASR.modelPath([pack("qwen3_asr", downloading: true)]))
        XCTAssertNil(VoiceASR.modelPath([pack("qwen3_asr", defect: .interruptedDownload)]))
        XCTAssertNil(VoiceASR.modelPath([pack("lfm3_audio")]), "the speech model is Qwen3-ASR, whatever else hears")
    }

    func testTheTranscriptIsTheResponsesTextTrimmed() {
        XCTAssertEqual(VoiceASR.text(fromResponse: Data(#"{"text":"  Hey Loki, what time is it? "}"#.utf8)), "Hey Loki, what time is it?")
        XCTAssertNil(VoiceASR.text(fromResponse: Data(#"{"text":"  "}"#.utf8)), "silence is no transcript")
        XCTAssertNil(VoiceASR.text(fromResponse: Data("oops".utf8)))
    }
}
