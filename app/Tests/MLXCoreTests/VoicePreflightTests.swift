import XCTest
@testable import MLXCore

final class VoicePreflightTests: XCTestCase {

    private func snap(mic: Bool = true, speech: Bool = true, onDevice: Bool = true,
                      locale: String = "en_US") -> VoicePreflight.Snapshot {
        .init(micAuthorized: mic, speechAuthorized: speech, onDeviceAvailable: onDevice, locale: locale)
    }

    func testNoIssueWhenEverythingIsReady() {
        XCTAssertNil(VoicePreflight.firstIssue(snap()))
    }

    func testIssueOrderingMicThenSpeechThenModel() {
        // Mic wins even if everything else is also missing.
        XCTAssertEqual(VoicePreflight.firstIssue(snap(mic: false, speech: false, onDevice: false)),
                       .microphoneDenied)
        // Mic ok → speech is next.
        XCTAssertEqual(VoicePreflight.firstIssue(snap(mic: true, speech: false, onDevice: false)),
                       .speechDenied)
        // Mic + speech ok → the on-device model (the user's actual case).
        XCTAssertEqual(VoicePreflight.firstIssue(snap(mic: true, speech: true, onDevice: false, locale: "fr_FR")),
                       .dictationUnavailable(locale: "fr_FR"))
    }

    func testEveryIssueHasNonEmptyCopy() {
        let issues: [VoicePreflight.Issue] = [
            .microphoneDenied, .speechDenied, .dictationUnavailable(locale: "en_US"),
        ]
        for issue in issues {
            XCTAssertFalse(VoicePreflight.shortMessage(for: issue).isEmpty)
            XCTAssertFalse(VoicePreflight.detail(for: issue).isEmpty)
            XCTAssertFalse(VoicePreflight.actionLabel(for: issue).isEmpty)
        }
    }

    func testDictationDetailNamesTheLocaleAndDictation() {
        let detail = VoicePreflight.detail(for: .dictationUnavailable(locale: "fr_FR"))
        XCTAssertTrue(detail.contains("fr_FR"))
        XCTAssertTrue(detail.contains("Dictation"))
    }

    func testSettingsURLsAreValidAndPaneSpecific() throws {
        let mic = VoicePreflight.settingsURLString(for: .microphoneDenied)
        let speech = VoicePreflight.settingsURLString(for: .speechDenied)
        let dict = VoicePreflight.settingsURLString(for: .dictationUnavailable(locale: "en_US"))
        // Each must parse as a URL (NSWorkspace.open needs a valid URL).
        XCTAssertNotNil(URL(string: mic))
        XCTAssertNotNil(URL(string: speech))
        XCTAssertNotNil(URL(string: dict))
        // …and point at the right pane.
        XCTAssertTrue(mic.contains("Privacy_Microphone"))
        XCTAssertTrue(speech.contains("Privacy_SpeechRecognition"))
        XCTAssertTrue(dict.contains("keyboard"))
        XCTAssertTrue(mic.hasPrefix("x-apple.systempreferences:"))
    }
}
