import XCTest
@testable import MLXCore

/// Voice-mode endpointing math (pure) + the launch-time audio-stack laziness.
///
/// Live failure 2026-07-05, two bugs from one session:
/// 1. Fixed-threshold VAD wedge: with continuous ambient noise (GPU fans on the
///    built-in mic) above the fixed 0.015 RMS threshold, `lastVoiceAt` refreshed
///    forever, the silence endpoint never fired, and the utterance never
///    finalized — voice mode transcribed the user ("run-on" partials gluing
///    multiple sentences) but never submitted a turn to the LLM.
/// 2. Mic prompt at app launch: the eagerly-built voice stack (AVSpeechSynthesizer
///    + chime/cue AVAudioEngines inside VoiceModeController, which the tray
///    forces at launch) makes CoreAudio run its voice-isolation evaluation,
///    which consults the mic TCC service → permission dialog at launch instead
///    of at voice-mode enable.
final class VoiceActivityTests: XCTestCase {

    // MARK: - AmbientFloor (minimum-statistics noise tracker)

    func testFloorIsZeroBeforeAnyInput() {
        let f = AmbientFloor()
        XCTAssertEqual(f.floor, 0)
    }

    func testFloorTracksConstantAmbientNoise() {
        var f = AmbientFloor()
        for _ in 0..<20 { _ = f.ingest(0.05) }
        XCTAssertEqual(f.floor, 0.05, accuracy: 1e-6)
    }

    func testSpeechBurstsDoNotRaiseTheFloor() {
        // Fan at 0.05 with loud speech bursts — the windowed MINIMUM keeps
        // reading the between-words dips, so the floor stays at ambient.
        var f = AmbientFloor()
        for i in 0..<40 { _ = f.ingest(i % 4 == 0 ? 0.05 : 0.35) }
        XCTAssertEqual(f.floor, 0.05, accuracy: 1e-6)
    }

    func testFloorRisesWhenNoiseStartsMidSession() {
        // Quiet room → fan turns on: once the window slides past the quiet
        // samples, the floor converges up to the new ambient level.
        var f = AmbientFloor(size: 10)
        for _ in 0..<10 { _ = f.ingest(0.002) }
        XCTAssertEqual(f.floor, 0.002, accuracy: 1e-6)
        for _ in 0..<10 { _ = f.ingest(0.06) }   // fan fills the whole window
        XCTAssertEqual(f.floor, 0.06, accuracy: 1e-6)
    }

    func testFloorDropsImmediatelyWhenNoiseStops() {
        var f = AmbientFloor(size: 10)
        for _ in 0..<10 { _ = f.ingest(0.06) }
        _ = f.ingest(0.001)                      // fan off: min drops on the next sample
        XCTAssertEqual(f.floor, 0.001, accuracy: 1e-6)
    }

    // MARK: - Speech threshold (relative to the floor, never below the legacy min)

    func testQuietRoomKeepsLegacyAbsoluteThreshold() {
        // Near-silent ambient → the max() keeps the old 0.015 sensitivity, so
        // quiet-room behavior is byte-identical to the fixed threshold.
        XCTAssertEqual(VoiceActivity.speechThreshold(floor: 0.001), 0.015, accuracy: 1e-6)
        XCTAssertEqual(VoiceActivity.speechThreshold(floor: 0), 0.015, accuracy: 1e-6)
    }

    func testNoisyRoomScalesThresholdAboveTheFloor() {
        // THE 2026-07-05 WEDGE: fans at 0.05 RMS sat above the fixed 0.015
        // threshold, so silence never accrued. Relative thresholding puts the
        // bar above ambient: the fan reads as silence, speech still clears it.
        let t = VoiceActivity.speechThreshold(floor: 0.05)
        XCTAssertFalse(0.05 > t, "ambient noise itself must read as silence")
        XCTAssertTrue(0.2 > t, "normal speech must still clear the bar")
    }

    // MARK: - Finalize decision (silence OR transcript-stall backstop)

    func testSilenceTimeoutFinalizes() {
        XCTAssertTrue(VoiceActivity.shouldFinalize(
            silenceElapsed: 1.2, silenceTimeout: 1.1,
            hasTranscript: false, transcriptStallElapsed: 0.1, stallTimeout: 2.0))
    }

    func testTranscriptStallFinalizesEvenWhileRMSSaysSpeech() {
        // Backstop for anything the energy VAD misclassifies as endless speech:
        // recognition stopped producing new words 2s ago and we HAVE words →
        // the user is done talking, finalize regardless of RMS.
        XCTAssertTrue(VoiceActivity.shouldFinalize(
            silenceElapsed: 0.0, silenceTimeout: 1.1,
            hasTranscript: true, transcriptStallElapsed: 2.1, stallTimeout: 2.0))
    }

    func testStallWithoutTranscriptDoesNotFinalize() {
        // No words yet → nothing to submit; the stall path must not fire on
        // ambient noise that never transcribed to anything.
        XCTAssertFalse(VoiceActivity.shouldFinalize(
            silenceElapsed: 0.0, silenceTimeout: 1.1,
            hasTranscript: false, transcriptStallElapsed: 5.0, stallTimeout: 2.0))
    }

    func testMidUtteranceKeepsListening() {
        XCTAssertFalse(VoiceActivity.shouldFinalize(
            silenceElapsed: 0.4, silenceTimeout: 1.1,
            hasTranscript: true, transcriptStallElapsed: 0.3, stallTimeout: 2.0))
    }

    // MARK: - Lazy audio stack (mic prompt must wait for voice-mode enable)

    func testWakeChimeBuildsNoEngineAtInit() {
        XCTAssertNil(SystemWakeChime().engine,
                     "chime engine at init = CoreAudio bring-up at launch = mic prompt at launch")
    }

    func testLoadingCueBuildsNoEngineAtInitOrStop() {
        let cue = SystemLoadingCue()
        XCTAssertNil(cue.engine)
        cue.stop()   // stop before any start must stay a no-op, not build audio
        XCTAssertNil(cue.engine)
    }

    @MainActor
    func testSystemSynthesizerBuildsNoAVSpeechSynthesizerAtInit() {
        let synth = SystemSpeechSynthesizer()
        XCTAssertNil(synth.synthStorage,
                     "AVSpeechSynthesizer at init is what consulted the mic TCC service at launch")
        // Idle-path operations must not force it either.
        synth.voiceIdentifier = "com.apple.voice.compact.en-US.Samantha"
        synth.stop()
        synth.enqueue("   ")   // whitespace-only → early return
        XCTAssertNil(synth.synthStorage)
    }
}
