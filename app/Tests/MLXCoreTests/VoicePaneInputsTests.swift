import XCTest
@testable import MLXCore

/// A reference clip only reaches a model that can clone a voice. The pane
/// hides the control, but the pane is not the only door: the chat tool builds
/// its own `AudioGenRequest` from the same saved path, so the REQUEST is where
/// the rule has to hold.
final class VoicePaneInputsTests: XCTestCase {

    private func request(_ model: AudioModelPreset, path: String?) -> AudioGenRequest {
        AudioGenRequest(model: model, text: "hello", refAudioPath: path)
    }

    func testACloningModelSendsTheClip() {
        let req = request(.qwen3TTS06B8bit, path: "/tmp/voice.wav")
        XCTAssertEqual(AudioGenRequest.clonableReference(req), "/tmp/voice.wav")
    }

    /// Kokoro speaks in its own built-in voices; `ref_audio` is a named 400
    /// server-side, so a clip left behind by a model switch must not travel.
    func testAModelThatCannotCloneSendsNothing() {
        let req = request(.kokoro82M, path: "/tmp/voice.wav")
        XCTAssertNil(AudioGenRequest.clonableReference(req))
    }

    func testNoClipIsNoClip() {
        XCTAssertNil(AudioGenRequest.clonableReference(request(.qwen3TTS06B8bit, path: nil)))
        XCTAssertNil(AudioGenRequest.clonableReference(request(.qwen3TTS06B8bit, path: "")))
    }

    /// The pane's own gate reads the same capability, so the control and the
    /// request cannot disagree about whether a clip is wanted.
    func testThePaneAsksForAClipOnExactlyThoseModels() {
        XCTAssertTrue(VoiceGenInputs.showsReference(.qwen3TTS06B8bit))
        XCTAssertFalse(VoiceGenInputs.showsReference(.kokoro82M))
    }
}
