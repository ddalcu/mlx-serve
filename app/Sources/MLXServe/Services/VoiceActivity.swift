import Foundation

/// Minimum-statistics ambient-noise tracker for the voice-mode VAD: the floor
/// is the MINIMUM mic RMS over a sliding window (~8 s at the tap's ~11
/// buffers/s). Speech has between-word dips that keep the window minimum at
/// ambient level, so the floor tracks the room (fans, AC, hum) without being
/// dragged up by talking — and recovers within one window when the noise
/// changes mid-session. Pure value type; pinned by `VoiceActivityTests`.
struct AmbientFloor {
    private var window: [Float]
    private var next = 0
    private var count = 0

    /// Default 96 samples ≈ 8 s of 4096-frame buffers at 44.1/48 kHz.
    init(size: Int = 96) {
        window = [Float](repeating: 0, count: max(1, size))
    }

    /// Record one RMS sample and return the updated floor.
    mutating func ingest(_ rms: Float) -> Float {
        window[next] = rms
        next = (next + 1) % window.count
        count = min(count + 1, window.count)
        return floor
    }

    /// Minimum over the filled window; 0 before any input.
    var floor: Float {
        count == 0 ? 0 : window[0..<count].min() ?? 0
    }
}

/// Endpointing decisions for voice mode, kept pure for tests.
///
/// Live failure 2026-07-05: the fixed 0.015 RMS threshold sat BELOW the
/// ambient level of a fan-spinning MacBook mic, so every buffer counted as
/// "speech", the 1.1 s silence endpoint never fired, and utterances never
/// finalized — voice mode transcribed the user but never submitted a turn.
/// Two layers fix the class:
/// 1. The speech threshold is RELATIVE to the tracked ambient floor
///    (`floorFactor`× above it), floored at the legacy absolute minimum so
///    quiet-room sensitivity is unchanged.
/// 2. A transcript-stall backstop: when recognition has words and hasn't
///    produced a NEW word for `stallTimeout`, the user is done talking —
///    finalize regardless of what the energy VAD thinks.
enum VoiceActivity {
    /// Legacy absolute threshold — the quiet-room sensitivity bar.
    static let minSpeechThreshold: Float = 0.015
    /// Speech must be this many times louder than the ambient floor.
    static let floorFactor: Float = 2.0

    static func speechThreshold(floor: Float,
                                minThreshold: Float = minSpeechThreshold,
                                factor: Float = floorFactor) -> Float {
        max(minThreshold, floor * factor)
    }

    /// End the utterance? Either the mic went quiet for `silenceTimeout`, or
    /// recognition stalled with words in hand (`stallTimeout`, the noisy-room
    /// backstop). No-transcript stalls never finalize — ambient noise that
    /// transcribes to nothing has nothing to submit.
    static func shouldFinalize(silenceElapsed: TimeInterval, silenceTimeout: TimeInterval,
                               hasTranscript: Bool, transcriptStallElapsed: TimeInterval,
                               stallTimeout: TimeInterval) -> Bool {
        if silenceElapsed >= silenceTimeout { return true }
        return hasTranscript && transcriptStallElapsed >= stallTimeout
    }
}
