import Foundation
import AVFoundation
@preconcurrency import AVFAudio

/// Speech-to-text for voice mode and dictation, on our own server: an energy VAD cuts the mic into utterances
/// (silence ends one), and each utterance is transcribed by Qwen3-ASR (`VoiceASR`). Publishes a 0…1 mic `level` for
/// the orb. The controller depends only on this protocol so it can be faked in tests.
@MainActor
protocol SpeechRecognizing: AnyObject {
    var level: Float { get }

    var onSpeechStarted: (() -> Void)? { get set }
    var onFinalTranscript: ((String) -> Void)? { get set }
    /// The mic heard sound but the speech model found no words in it (a cough, a door).
    var onUnrecognizedSpeech: (() -> Void)? { get set }
    var onError: ((String) -> Void)? { get set }

    /// Request microphone permission. Safe to call repeatedly.
    func requestAuthorization() async -> Bool
    /// What voice needs, read before opening the mic (prompting for the mic if undecided). A protocol requirement so
    /// the concrete recognizer's override is dynamically dispatched; fakes fall back to the extension default.
    func preflight() async -> VoicePreflight.Snapshot
    /// Open the mic and begin listening. Throws if the engine can't start.
    func start() throws
    /// Stop the mic; a transcription still in flight is dropped.
    func stop()
    /// The audio of the utterance `onFinalTranscript` just finalized, 16 kHz mono (native voice sends it).
    var utterancePCM: [Float] { get }
    /// false: the utterance is the turn as it is (a model that hears it), finalized with no transcript.
    var wantsWords: (() -> Bool)? { get set }
}

extension SpeechRecognizing {
    var utterancePCM: [Float] { [] }
    func preflight() async -> VoicePreflight.Snapshot {
        VoicePreflight.Snapshot(micAuthorized: await requestAuthorization(), asrReady: true)
    }
}

@MainActor
func makeSpeechRecognizer() -> any SpeechRecognizing {
    ServerSpeechRecognizer()
}

/// `SpeechRecognizing` over the mic's input tap: RMS metering, the adaptive VAD, and one `VoiceASR` request per
/// utterance.
@MainActor
final class ServerSpeechRecognizer: SpeechRecognizing {
    let engine = AVAudioEngine()

    private(set) var level: Float = 0
    private(set) var utterancePCM: [Float] = []

    var onSpeechStarted: (() -> Void)?
    var onFinalTranscript: ((String) -> Void)?
    var onUnrecognizedSpeech: (() -> Void)?
    var onError: ((String) -> Void)?
    var wantsWords: (() -> Bool)?

    /// Quiet-room speech bar (absolute RMS). The EFFECTIVE threshold is `VoiceActivity.speechThreshold` — this floored
    /// value scaled above the tracked ambient level, so fan/AC noise reads as silence.
    var voiceThreshold: Float = VoiceActivity.minSpeechThreshold
    /// Silence this long after speech ends an utterance.
    var silenceTimeout: TimeInterval = 1.1
    /// The longest utterance: past it the turn ends even mid-speech.
    var maxUtterance: TimeInterval = 30
    /// With no words asked for, an utterance needs this much speech-level audio to be a turn: a cough or a door is
    /// shorter, and there is no transcript to tell it from words.
    static let minVoicedSeconds: TimeInterval = 0.3
    private var voicedSeconds: TimeInterval = 0

    private let transcribe: @MainActor ([Float]) async -> String?
    private let ready: @MainActor () -> Bool
    private var speaking = false
    private var speechStartedAt = Date()
    private var lastVoiceAt = Date()
    private var silenceTimer: Timer?
    private var ambient = AmbientFloor()
    private var utterance = UtteranceAudio()
    /// An utterance is with the speech model; the mic's VAD waits for its words.
    private var transcribing = false
    /// Bumped by `stop()`: a transcript arriving after it belongs to a session that is gone.
    private var generation = 0

    init(transcribe: (@MainActor ([Float]) async -> String?)? = nil, ready: (@MainActor () -> Bool)? = nil) {
        self.transcribe = transcribe ?? { await VoiceASR.transcribe($0) }
        self.ready = ready ?? { VoiceASR.ready }
    }

    func requestAuthorization() async -> Bool { await AudioRecorder.requestPermission() }

    func preflight() async -> VoicePreflight.Snapshot {
        VoicePreflight.Snapshot(micAuthorized: await AudioRecorder.requestPermission(),
                                asrReady: VoicePreflight.asrReady(wantsWords: wantsWords?() ?? true, onDisk: ready()))
    }

    func start() throws {
        // Idempotent: `AVAudioEngine` raises (and aborts the app) if a second tap is installed on a bus that already
        // has one, so a stray double-`start()` must never reach `installTap` with the old tap in place.
        stop()

        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(domain: "SpeechRecognizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No usable audio input device."])
        }

        speaking = false
        lastVoiceAt = Date()
        ambient = AmbientFloor()
        utterance = UtteranceAudio()

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            let rms = Self.rms(of: buffer)
            let heard = Self.mono16k(buffer)
            let seconds = Double(buffer.frameLength) / buffer.format.sampleRate
            DispatchQueue.main.async {
                self.utterance.append(heard)
                self.noteAudio(rms: rms, seconds: seconds)
            }
        }
        engine.prepare()
        try engine.start()

        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkSilence() }
        }
    }

    func stop() {
        generation += 1
        transcribing = false
        silenceTimer?.invalidate(); silenceTimer = nil
        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning { engine.stop() }
        speaking = false
        level = 0
    }

    // MARK: VAD + endpointing (main actor)

    private func noteAudio(rms: Float, seconds: TimeInterval) {
        level = min(1, rms * 6)
        let floor = ambient.ingest(rms)
        guard !transcribing, rms > VoiceActivity.speechThreshold(floor: floor, minThreshold: voiceThreshold) else { return }
        lastVoiceAt = Date()
        if !speaking {
            speaking = true
            speechStartedAt = lastVoiceAt
            voicedSeconds = 0
            utterance.markSpeechStart()
            onSpeechStarted?()
        }
        voicedSeconds += seconds
    }

    private func checkSilence() {
        guard speaking else { return }
        let now = Date()
        guard VoiceActivity.shouldFinalize(silenceElapsed: now.timeIntervalSince(lastVoiceAt), silenceTimeout: silenceTimeout,
                                           speechElapsed: now.timeIntervalSince(speechStartedAt), maxUtterance: maxUtterance)
        else { return }
        speaking = false
        let pcm = utterance.take()
        if wantsWords?() == false {
            guard voicedSeconds >= Self.minVoicedSeconds else { onUnrecognizedSpeech?(); return }
            utterancePCM = pcm
            onFinalTranscript?("")
            return
        }
        transcribing = true
        let gen = generation
        Task { [weak self, transcribe] in
            let text = await transcribe(pcm)
            guard let self, gen == self.generation else { return }
            self.transcribing = false
            if let text {
                self.utterancePCM = pcm
                self.onFinalTranscript?(text)
            } else {
                self.onUnrecognizedSpeech?()
            }
        }
    }

    static func mono16k(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let ch = buffer.floatChannelData, buffer.frameLength > 0 else { return [] }
        let n = Int(buffer.frameLength)
        let channels = (0..<Int(buffer.format.channelCount)).map { Array(UnsafeBufferPointer(start: ch[$0], count: n)) }
        return UtteranceAudio.downmix16k(channels: channels, sampleRate: buffer.format.sampleRate)
    }

    static func rms(of buffer: AVAudioPCMBuffer) -> Float {
        guard let ch = buffer.floatChannelData, buffer.frameLength > 0 else { return 0 }
        let n = Int(buffer.frameLength)
        var sum: Float = 0
        for i in 0..<n { let s = ch[0][i]; sum += s * s }
        return (sum / Float(n)).squareRoot()
    }
}
