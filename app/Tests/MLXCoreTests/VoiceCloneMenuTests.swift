import XCTest

@testable import MLXCore

/// The voice picker (tray panel + voice overlay) presents the cloned voice as
/// a first-class selectable voice: when a clone clip is set, cloning is
/// enabled, and the Qwen3-TTS model is downloaded, the picker shows the clip
/// — not the Apple fallback voice that would only speak on synthesis failure.
/// Pure decisions live in `VoiceCloneMenuModel`.
final class VoiceCloneMenuTests: XCTestCase {

    // MARK: - When does the clone actually speak?

    func testCloneActiveNeedsClipEnabledAndModel() {
        XCTAssertTrue(VoiceCloneMenuModel.cloneIsActive(
            clipPath: "/x/clip.wav", cloneEnabled: true, ttsModelDownloaded: true))
    }

    func testCloneInactiveWithoutClip() {
        XCTAssertFalse(VoiceCloneMenuModel.cloneIsActive(
            clipPath: "", cloneEnabled: true, ttsModelDownloaded: true))
    }

    func testCloneInactiveWhenUserPickedSystemVoice() {
        XCTAssertFalse(VoiceCloneMenuModel.cloneIsActive(
            clipPath: "/x/clip.wav", cloneEnabled: false, ttsModelDownloaded: true))
    }

    func testCloneInactiveWithoutTTSModel() {
        // No Qwen3-TTS on disk → every sentence would fall back to the system
        // voice; the picker must not claim the clone is speaking.
        XCTAssertFalse(VoiceCloneMenuModel.cloneIsActive(
            clipPath: "/x/clip.wav", cloneEnabled: true, ttsModelDownloaded: false))
    }

    // MARK: - Collapsed picker label

    func testCollapsedLabelShowsClipLabelWhenCloneActive() {
        XCTAssertEqual(
            VoiceCloneMenuModel.collapsedLabel(
                clipPath: "/x/clip.wav", cloneEnabled: true, ttsModelDownloaded: true,
                cloneLabel: "morgan.mp3", systemVoiceName: "Jamie"),
            "morgan.mp3")
    }

    func testCollapsedLabelFallsBackToMyVoiceWithoutALabel() {
        XCTAssertEqual(
            VoiceCloneMenuModel.collapsedLabel(
                clipPath: "/x/clip.wav", cloneEnabled: true, ttsModelDownloaded: true,
                cloneLabel: "", systemVoiceName: "Jamie"),
            "My voice")
    }

    func testCollapsedLabelShowsSystemVoiceWhenCloneCannotSpeak() {
        // Clip set but TTS model missing → Jamie is what will actually speak.
        XCTAssertEqual(
            VoiceCloneMenuModel.collapsedLabel(
                clipPath: "/x/clip.wav", cloneEnabled: true, ttsModelDownloaded: false,
                cloneLabel: "morgan.mp3", systemVoiceName: "Jamie"),
            "Jamie")
        XCTAssertEqual(
            VoiceCloneMenuModel.collapsedLabel(
                clipPath: "", cloneEnabled: true, ttsModelDownloaded: true,
                cloneLabel: "", systemVoiceName: "Jamie"),
            "Jamie")
    }

    // MARK: - Menu rows

    func testCloneItemTitleUsesLabel() {
        XCTAssertEqual(VoiceCloneMenuModel.cloneItemTitle(label: "morgan.mp3"),
                       "My voice — morgan.mp3")
        XCTAssertEqual(VoiceCloneMenuModel.cloneItemTitle(label: ""), "My voice")
    }

    func testUnavailableReasonOnlyWhenTTSModelMissing() {
        XCTAssertNil(VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: true))
        let reason = VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: false)
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason?.contains("Audio") == true,
                      "reason should point at the Audio tile for the download")
    }
}

/// The two new ServerOptions fields must survive round-trip and default
/// correctly when decoding a blob written before they existed (the tolerant-
/// decode contract every ServerOptions field follows).
final class VoiceCloneOptionsCodingTests: XCTestCase {

    func testLegacyBlobDefaultsToCloneEnabledWithNoLabel() throws {
        let o = try JSONDecoder().decode(ServerOptions.self, from: Data("{}".utf8))
        XCTAssertTrue(o.voiceCloneEnabled,
                      "a clip set before the toggle existed must keep cloning")
        XCTAssertEqual(o.voiceCloneLabel, "")
    }

    func testRoundTripPreservesToggleAndLabel() throws {
        var o = ServerOptions()
        o.voiceCloneEnabled = false
        o.voiceCloneLabel = "morgan.mp3"
        let back = try JSONDecoder().decode(ServerOptions.self,
                                            from: JSONEncoder().encode(o))
        XCTAssertFalse(back.voiceCloneEnabled)
        XCTAssertEqual(back.voiceCloneLabel, "morgan.mp3")
    }
}
