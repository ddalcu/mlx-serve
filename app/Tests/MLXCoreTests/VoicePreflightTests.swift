import XCTest
@testable import MLXCore

/// Voice needs two things: the microphone and our own speech-to-text model (Qwen3-ASR). Apple's dictation is gone.
final class VoicePreflightTests: XCTestCase {

    func testTheMicComesFirstThenTheSpeechModel() {
        XCTAssertNil(VoicePreflight.firstIssue(.init(micAuthorized: true, asrReady: true)))
        XCTAssertEqual(VoicePreflight.firstIssue(.init(micAuthorized: false, asrReady: false)), .microphoneDenied)
        XCTAssertEqual(VoicePreflight.firstIssue(.init(micAuthorized: true, asrReady: false)), .asrMissing)
    }

    /// The chat window with a model that hears the recording asks for no words, so it needs no Qwen3-ASR.
    func testNoSpeechModelIsNeededWhereNoWordsAreAsked() {
        XCTAssertTrue(VoicePreflight.asrReady(wantsWords: false, onDisk: false))
        XCTAssertFalse(VoicePreflight.asrReady(wantsWords: true, onDisk: false))
        XCTAssertTrue(VoicePreflight.asrReady(wantsWords: true, onDisk: true))
    }

    func testTheMissingSpeechModelOffersItsDownloadNotASettingsPane() {
        XCTAssertNil(VoicePreflight.settingsURLString(for: .asrMissing))
        XCTAssertTrue(VoicePreflight.actionLabel(for: .asrMissing).contains("Download"))
        XCTAssertTrue(VoicePreflight.detail(for: .asrMissing).contains("Qwen3-ASR"))
        let mic = try? XCTUnwrap(VoicePreflight.settingsURLString(for: .microphoneDenied))
        XCTAssertTrue(mic?.contains("Privacy_Microphone") == true)
    }
}
