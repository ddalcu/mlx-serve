import XCTest
@testable import MLXCore

/// The clone path loads whichever Qwen3-TTS preset the Audio pane is configured
/// for (`AudioGenSettings.resolvedModel`), which DEFAULTS to the 0.6B repo. A
/// machine holding only the 1.7B variants therefore had `resolveModelDir` return
/// nil, `synthesize` return nil, and every cloned sentence fall back to the
/// system voice — silently, with the picker still showing the clip as active.
/// Live 2026-07-26: an uploaded wav "wasn't piping through Qwen3-TTS".
///
/// So the model the voice path uses is RESOLVED against the disk: the configured
/// preset when it's there, else any other cloning-capable one that is.
final class VoiceCloneModelResolutionTests: XCTestCase {

    private let configured = AudioModelPreset.qwen3TTS06B8bit
    private let other = AudioModelPreset.qwen3TTS17B8bit

    func testTheConfiguredModelWinsWhenItIsDownloaded() {
        let picked = VoiceCloneMenuModel.resolvedCloneModel(configured: configured,
                                                            isDownloaded: { _ in true })
        XCTAssertEqual(picked?.id, configured.id, "never second-guess an explicit choice")
    }

    func testAnotherDownloadedCloningModelIsUsedWhenTheConfiguredOneIsMissing() {
        let picked = VoiceCloneMenuModel.resolvedCloneModel(
            configured: configured,
            isDownloaded: { $0.id == other.id })
        XCTAssertEqual(picked?.id, other.id,
                       "a 1.7B on disk should speak rather than falling back to the system voice")
    }

    func testNothingDownloadedResolvesToNilSoTheUiCanSaySo() {
        XCTAssertNil(VoiceCloneMenuModel.resolvedCloneModel(configured: configured,
                                                            isDownloaded: { _ in false }))
    }

    func testOnlyCloningCapableModelsAreCandidates() {
        // Kokoro cannot clone — sending it `ref_audio` is a named 400 server-side,
        // so it must never stand in for a missing Qwen3-TTS.
        let picked = VoiceCloneMenuModel.resolvedCloneModel(
            configured: configured,
            isDownloaded: { $0.id == AudioModelPreset.kokoro82M.id })
        XCTAssertNil(picked)
        XCTAssertTrue(AudioModelPreset.all.allSatisfy(\.supportsCloning),
                      "the candidate list is the cloning-capable catalog")
    }

    func testAvailabilityAgreesWithWhatWouldActuallyLoad() {
        // The picker's enabled/disabled state and the synthesizer must read the
        // same answer, or the UI ticks a voice that can't speak.
        let none = VoiceCloneMenuModel.cloneAvailable(configured: configured,
                                                      isDownloaded: { _ in false })
        let some = VoiceCloneMenuModel.cloneAvailable(configured: configured,
                                                      isDownloaded: { $0.id == self.other.id })
        XCTAssertFalse(none)
        XCTAssertTrue(some)
    }
}
