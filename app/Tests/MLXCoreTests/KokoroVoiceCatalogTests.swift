import XCTest
@testable import MLXCore

final class KokoroVoiceCatalogTests: XCTestCase {

    func testDisplayNameDecodesTheLanguageGenderPrefix() {
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "af_bella"), "Bella — American English, female")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "am_michael"), "Michael — American English, male")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "bf_emma"), "Emma — British English, female")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "zm_yunjian"), "Yunjian — Chinese, male")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "jf_gongitsune"), "Gongitsune — Japanese, female")
    }

    func testUnparseableIdFallsBackToTheRawIdRatherThanBlank() {
        // An id we cannot decode must stay SELECTABLE — the wire value is still
        // valid server-side, so hiding it or rendering "" would strand a voice.
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "weird"), "weird")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: "xx_thing"), "xx_thing")
        XCTAssertEqual(KokoroVoiceCatalog.displayName(for: ""), "")
    }

    func testBlendDisplayNameJoinsShortNames() {
        XCTAssertEqual(KokoroVoiceCatalog.blendDisplayName(for: "af_bella"), "Bella")
        XCTAssertEqual(KokoroVoiceCatalog.blendDisplayName(for: "af_bella,af_sky"), "Bella + Sky")
        // Whitespace around the separator is what a person actually types.
        XCTAssertEqual(KokoroVoiceCatalog.blendDisplayName(for: "af_bella , af_sky"), "Bella + Sky")
        XCTAssertEqual(KokoroVoiceCatalog.blendDisplayName(for: ""), "Kokoro")
        XCTAssertEqual(KokoroVoiceCatalog.blendDisplayName(for: " , "), "Kokoro")
    }

    func testIsBlend() {
        XCTAssertFalse(KokoroVoiceCatalog.isBlend("af_bella"))
        XCTAssertFalse(KokoroVoiceCatalog.isBlend("af_bella,"))
        XCTAssertTrue(KokoroVoiceCatalog.isBlend("af_bella,af_sky"))
        XCTAssertTrue(KokoroVoiceCatalog.isBlend("af_bella, af_sky, am_puck"))
    }

    func testGroupingCoversEveryPublishedVoiceExactlyOnce() {
        let groups = KokoroVoiceCatalog.grouped()
        let flat = groups.flatMap { $0.voices }
        // No voice may be dropped by the grouping and none duplicated — the menu
        // IS this list, so a missed prefix silently removes voices from the UI
        // while they still work over HTTP.
        XCTAssertEqual(Set(flat), Set(AudioModelPreset.kokoroVoices))
        XCTAssertEqual(flat.count, AudioModelPreset.kokoroVoices.count)
        XCTAssertEqual(flat.count, 54, "the published catalog is 54 voices")
        XCTAssertNil(groups.first { $0.language == "Other" },
                     "every published voice should parse into a real language group")
    }

    func testGroupingIsOrderedEnglishFirst() {
        let groups = KokoroVoiceCatalog.grouped()
        XCTAssertEqual(groups.first?.language, "American English")
        XCTAssertEqual(groups.dropFirst().first?.language, "British English")
    }

    func testGroupingSendsUnparseableIdsToOtherInsteadOfDroppingThem() {
        let groups = KokoroVoiceCatalog.grouped(["af_bella", "mystery", "qq_nope"])
        let flat = groups.flatMap { $0.voices }
        XCTAssertEqual(Set(flat), ["af_bella", "mystery", "qq_nope"])
        XCTAssertEqual(groups.last?.language, "Other")
    }

    func testPreviewSentenceNamesTheVoiceAndStaysShort() {
        let single = KokoroVoiceCatalog.previewSentence(for: "af_bella")
        XCTAssertTrue(single.contains("Bella"), "a run of previews must stay distinguishable")
        XCTAssertLessThan(single.count, 60, "a preview is synthesized on demand — keep it quick")

        let blend = KokoroVoiceCatalog.previewSentence(for: "af_bella,af_sky")
        XCTAssertTrue(blend.contains("Bella + Sky"))
        XCTAssertTrue(blend.contains("blend"))
    }

    func testEveryPublishedVoiceProducesANonEmptyDisplayNameAndPreview() {
        for id in AudioModelPreset.kokoroVoices {
            XCTAssertFalse(KokoroVoiceCatalog.displayName(for: id).isEmpty, "\(id)")
            XCTAssertNotEqual(KokoroVoiceCatalog.displayName(for: id), id,
                              "\(id) fell through to the raw-id fallback")
            XCTAssertFalse(KokoroVoiceCatalog.previewSentence(for: id).isEmpty, "\(id)")
        }
    }
}

/// The tray's engine-aware decisions. These are what stop the collapsed label
/// from disagreeing with what will actually speak.
final class VoiceMenuEngineTests: XCTestCase {

    func testKokoroIsActiveOnlyWhenSelectedANDDownloaded() {
        // Without the checkpoint every sentence silently falls back to the
        // system voice, so ticking it in the menu would be a lie.
        XCTAssertTrue(VoiceCloneMenuModel.kokoroIsActive(engine: .kokoro, kokoroDownloaded: true))
        XCTAssertFalse(VoiceCloneMenuModel.kokoroIsActive(engine: .kokoro, kokoroDownloaded: false))
        XCTAssertFalse(VoiceCloneMenuModel.kokoroIsActive(engine: .clone, kokoroDownloaded: true))
        XCTAssertFalse(VoiceCloneMenuModel.kokoroIsActive(engine: .system, kokoroDownloaded: true))
    }

    func testCollapsedLabelNamesTheKokoroVoiceWhenKokoroSpeaks() {
        let label = VoiceCloneMenuModel.collapsedLabel(
            engine: .kokoro, clipPath: "/clips/me.wav", cloneEnabled: true,
            ttsModelDownloaded: true, kokoroDownloaded: true,
            kokoroVoice: "af_bella", cloneLabel: "morgan.mp3", systemVoiceName: "Samantha")
        XCTAssertEqual(label, "Bella", "a clip being present must not outrank the chosen engine")
    }

    func testCollapsedLabelShowsABlendAsABlend() {
        let label = VoiceCloneMenuModel.collapsedLabel(
            engine: .kokoro, clipPath: "", cloneEnabled: false,
            ttsModelDownloaded: false, kokoroDownloaded: true,
            kokoroVoice: "af_bella,af_sky", cloneLabel: "", systemVoiceName: "Samantha")
        XCTAssertEqual(label, "Bella + Sk…", "clamped to the tray's 10-char width budget")
    }

    func testCollapsedLabelFallsBackToTheSystemVoiceWhenKokoroIsNotDownloaded() {
        let label = VoiceCloneMenuModel.collapsedLabel(
            engine: .kokoro, clipPath: "", cloneEnabled: false,
            ttsModelDownloaded: false, kokoroDownloaded: false,
            kokoroVoice: "af_bella", cloneLabel: "", systemVoiceName: "Samantha")
        XCTAssertEqual(label, "Samantha", "the label must name what will really speak")
    }

    func testCollapsedLabelStillHonoursTheCloneEngine() {
        let label = VoiceCloneMenuModel.collapsedLabel(
            engine: .clone, clipPath: "/clips/me.wav", cloneEnabled: true,
            ttsModelDownloaded: true, kokoroDownloaded: true,
            kokoroVoice: "af_bella", cloneLabel: "morgan.mp3", systemVoiceName: "Samantha")
        XCTAssertEqual(label, "morgan.mp3", "exactly at the cap is NOT truncated")
    }

    func testCollapsedLabelUsesTheSystemVoiceForTheSystemEngine() {
        let label = VoiceCloneMenuModel.collapsedLabel(
            engine: .system, clipPath: "/clips/me.wav", cloneEnabled: true,
            ttsModelDownloaded: true, kokoroDownloaded: true,
            kokoroVoice: "af_bella", cloneLabel: "morgan.mp3", systemVoiceName: "Samantha")
        XCTAssertEqual(label, "Samantha")
    }

    /// The weights are published now, so the dead row points at the DOWNLOAD in
    /// Settings ▸ Voice — not at a local conversion script the user can't run.
    func testKokoroUnavailableReasonPointsAtTheDownload() {
        XCTAssertNil(VoiceCloneMenuModel.kokoroUnavailableReason(kokoroDownloaded: true))
        let reason = VoiceCloneMenuModel.kokoroUnavailableReason(kokoroDownloaded: false)
        XCTAssertTrue(reason?.lowercased().contains("download") == true,
                      "a dead row must say what to DO about it")
    }
}
