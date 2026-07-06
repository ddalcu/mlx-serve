import Foundation
import AVFoundation

/// Text-to-speech for voice mode: sentences are enqueued as they stream in and
/// spoken back-to-back; `stop()` clears the queue for barge-in. The controller
/// only depends on this protocol, so it can be unit-tested with a fake.
@MainActor
protocol SpeechSynthesizing: AnyObject {
    /// True while at least one utterance is queued or being spoken.
    var isSpeaking: Bool { get }
    /// The `AVSpeechSynthesisVoice` identifier to speak with; nil = system default.
    var voiceIdentifier: String? { get set }
    /// Fired (on the main actor) when the last queued utterance finishes and the
    /// queue is empty — the controller uses this to reopen the mic.
    var onQueueDrained: (() -> Void)? { get set }
    /// Queue one chunk (typically a sentence) to be spoken after the current one.
    func enqueue(_ text: String)
    /// Cancel everything immediately (barge-in / mode close).
    func stop()
    /// Voice mode is closing for good: stop AND release any resident audio
    /// resources (e.g. unload a server-side TTS model). Distinct from `stop()`,
    /// which also fires on barge-in mid-session where resources should stay
    /// warm. Default implementation just stops.
    func shutdown()
}

extension SpeechSynthesizing {
    func shutdown() { stop() }
}

/// `AVSpeechSynthesizer`-backed implementation. Tracks an explicit pending count
/// instead of relying on `AVSpeechSynthesizer.isSpeaking`, so `isSpeaking` flips
/// synchronously on `enqueue`/`stop` and the drain callback is race-free.
///
/// The `AVSpeechSynthesizer` itself is created lazily on the FIRST utterance —
/// never at init. This object is constructed inside `VoiceModeController`,
/// which the menu-bar tray forces at APP LAUNCH, and instantiating
/// AVSpeechSynthesizer brings up CoreAudio's voice-processing ("chat flavor")
/// machinery, which consults the microphone TCC service → the app prompted
/// for the mic at launch instead of at voice-mode enable (live 2026-07-05;
/// tccd AUTHREQ 1.9 s after launch). Laziness pinned by
/// `VoiceActivityTests.testSystemSynthesizerBuildsNoAVSpeechSynthesizerAtInit`.
@MainActor
final class SystemSpeechSynthesizer: NSObject, SpeechSynthesizing {
    /// Lazy backing store — internal (not private) so tests can assert init
    /// builds nothing. Access through `synth` everywhere except `stop()`,
    /// which must never FORCE creation just to stop nothing.
    private(set) var synthStorage: AVSpeechSynthesizer?
    private var synth: AVSpeechSynthesizer {
        if let s = synthStorage { return s }
        let s = AVSpeechSynthesizer()
        s.delegate = self
        synthStorage = s
        return s
    }
    private var pending = 0

    /// Optional explicit voice; nil → the system default for the user's language.
    var voice: AVSpeechSynthesisVoice?

    /// Identifier-based accessor for `voice` (satisfies `SpeechSynthesizing`).
    var voiceIdentifier: String? {
        get { voice?.identifier }
        set { voice = newValue.flatMap { AVSpeechSynthesisVoice(identifier: $0) } }
    }
    /// 0…1 multiplier mapped onto AVSpeech's rate range; 1.0 == default rate.
    var rateScale: Float = 1.0

    var onQueueDrained: (() -> Void)?
    var isSpeaking: Bool { pending > 0 }

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let u = AVSpeechUtterance(string: trimmed)
        u.voice = voice
        u.rate = min(AVSpeechUtteranceMaximumSpeechRate,
                     max(AVSpeechUtteranceMinimumSpeechRate,
                         AVSpeechUtteranceDefaultSpeechRate * rateScale))
        pending += 1
        synth.speak(u)
    }

    func stop() {
        pending = 0
        synthStorage?.stopSpeaking(at: .immediate)
    }

    private func finishedOne() {
        guard pending > 0 else { return }
        pending -= 1
        if pending == 0 { onQueueDrained?() }
    }
}

extension SystemSpeechSynthesizer: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finishedOne() }
    }
    // didCancel fires for each queued utterance when we stop(); pending is already
    // zeroed in stop(), and finishedOne() guards on pending > 0, so these are no-ops.
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self.finishedOne() }
    }
}
