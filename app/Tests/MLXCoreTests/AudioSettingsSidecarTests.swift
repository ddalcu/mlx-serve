import XCTest
@testable import MLXCore

/// The `<clip>.txt` settings sidecar written beside each generated voice clip.
final class AudioSettingsSidecarTests: XCTestCase {
    func testVoiceSidecarDocumentsTextAndVoiceParams() {
        let req = AudioGenRequest(
            model: .qwen3TTS06B8bit,
            text: "  Hello there.  ",
            refAudioPath: "/clips/my-voice.wav",
            refText: "sample transcript",
            speed: 1.1,
            temperature: 0.8
        )
        let txt = AudioGenService.settingsText(req, modelName: "qwen3-tts")
        XCTAssertTrue(txt.contains("model: qwen3-tts"))
        XCTAssertTrue(txt.contains("speed: 1.10"))
        XCTAssertTrue(txt.contains("temperature: 0.80"))
        XCTAssertTrue(txt.contains("reference_voice: my-voice.wav"), "basename only, not full path")
        XCTAssertTrue(txt.contains("reference_transcript: sample transcript"))
        XCTAssertTrue(txt.contains("# Text\nHello there."), "text trimmed")
        XCTAssertEqual(AudioGenService.sidecarPath(forWav: "/a/clip.wav"), "/a/clip.txt")
    }

    func testVoiceSidecarOmitsAbsentReference() {
        let req = AudioGenRequest(model: .qwen3TTS06B8bit, text: "hi")
        let txt = AudioGenService.settingsText(req, modelName: "m")
        XCTAssertFalse(txt.contains("reference_voice:"))
        XCTAssertFalse(txt.contains("reference_transcript:"))
    }
}
