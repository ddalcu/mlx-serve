import XCTest
@testable import MLXCore

/// The Voice pane asks for a reference clip only where one changes the result.
/// A model that speaks in its own built-in voices takes no `ref_audio` (the
/// server answers a named 400), so the control would be asking for a file it
/// then has nowhere to send.
final class VoicePaneInputsTests: XCTestCase {

    func testACloningModelIsOfferedAReferenceVoice() {
        XCTAssertTrue(VoiceGenInputs.showsReference(AudioModelPreset.qwen3TTS06B8bit))
    }

    func testAModelThatCannotCloneIsNotAskedForOne() {
        XCTAssertFalse(VoiceGenInputs.showsReference(AudioModelPreset.kokoro82M))
    }
}
