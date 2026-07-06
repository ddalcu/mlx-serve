import Foundation
import AVFoundation
import Speech

/// Speech-to-text for voice mode. Drives a hands-free turn: emits partial
/// transcripts as the user talks, fires `onFinalTranscript` once a silence
/// gap ends the utterance, and publishes a 0…1 mic `level` for the orb. The
/// controller depends only on this protocol so it can be faked in tests.
@MainActor
protocol SpeechRecognizing: AnyObject {
    var partialTranscript: String { get }
    var level: Float { get }

    var onSpeechStarted: (() -> Void)? { get set }
    var onPartialTranscript: ((String) -> Void)? { get set }
    var onFinalTranscript: ((String) -> Void)? { get set }
    /// A turn where the mic clearly heard speech (RMS crossed the threshold) but
    /// recognition produced NOTHING. The signature of on-device dictation being
    /// off / unavailable while the mic itself works — surfaced to the user as an
    /// actionable notice after a couple of consecutive occurrences.
    var onUnrecognizedSpeech: (() -> Void)? { get set }
    var onError: ((String) -> Void)? { get set }

    /// Request mic + speech-recognition permission. Safe to call repeatedly.
    func requestAuthorization() async -> Bool
    /// Granular prerequisite read (prompting for mic/speech if undecided) so the
    /// UI can show a precise, non-invasive "what's missing" notice before opening
    /// the mic. A protocol requirement (not just an extension) so the concrete
    /// recognizer's override is dynamically dispatched; fakes fall back to the
    /// extension default below.
    func preflight() async -> VoicePreflight.Snapshot
    /// Open the mic and begin recognizing. Throws if the engine can't start.
    func start() throws
    /// Stop the mic and tear down recognition.
    func stop()
}

extension SpeechRecognizing {
    /// Default: derive from the combined `requestAuthorization()` and assume the
    /// on-device model is present. The real `BaseSpeechRecognizer` overrides this
    /// with per-prerequisite truth; test fakes inherit this and stay green.
    func preflight() async -> VoicePreflight.Snapshot {
        let ok = await requestAuthorization()
        return VoicePreflight.Snapshot(micAuthorized: ok, speechAuthorized: ok,
                                       onDeviceAvailable: true, locale: Locale.current.identifier)
    }
}

/// Choose the speech-to-text backend.
///
/// IMPORTANT: macOS 26's `SpeechAnalyzer`/`SpeechTranscriber` requires its
/// on-device model assets to be fully provisioned. When they are not (e.g.
/// Apple Intelligence "assetIsNotReady" / the locale's model isn't installed),
/// `SpeechAnalyzer.start` *hard-traps* (SIGTRAP) deep inside the framework —
/// it's a `fatalError`, not a throwable error, so it takes the whole app down
/// and can't be caught. Until we can robustly gate on installed assets and
/// fall back without ever calling `start` on an unready analyzer, we use the
/// mature `SFSpeechRecognizer` path, which is reliable on macOS 14–26 and runs
/// on-device on Apple Silicon. `ModernSpeechRecognizer` is kept for that future
/// gated opt-in.
@MainActor
func makeSpeechRecognizer() -> any SpeechRecognizing {
    return LegacySpeechRecognizer()
}

// MARK: - Shared mic tap + silence endpointing

/// Owns the `AVAudioEngine` input tap, RMS level metering and the silence timer
/// that turns continuous recognition into discrete conversational turns.
/// Subclasses plug in a concrete recognizer via the `feed`/`reset`/`teardown`
/// hooks and keep `currentTranscript` up to date.
@MainActor
class BaseSpeechRecognizer: NSObject, SpeechRecognizing {
    let engine = AVAudioEngine()

    private(set) var partialTranscript = ""
    private(set) var level: Float = 0

    var onSpeechStarted: (() -> Void)?
    var onPartialTranscript: ((String) -> Void)?
    var onFinalTranscript: ((String) -> Void)?
    var onUnrecognizedSpeech: (() -> Void)?
    var onError: ((String) -> Void)?

    /// Quiet-room speech bar (absolute RMS). The EFFECTIVE threshold is
    /// `VoiceActivity.speechThreshold` — this floored value scaled above the
    /// tracked ambient level, so fan/AC noise reads as silence instead of
    /// pinning `lastVoiceAt` forever (the 2026-07-05 never-finalizes wedge).
    var voiceThreshold: Float = VoiceActivity.minSpeechThreshold
    /// Silence this long after speech ends a turn.
    var silenceTimeout: TimeInterval = 1.1
    /// Backstop: with words in hand and no NEW word for this long, finalize
    /// even if the energy VAD still reads "speech" (noise misclassification).
    var transcriptStallTimeout: TimeInterval = 2.0

    private var speaking = false
    private var lastVoiceAt = Date()
    private var silenceTimer: Timer?
    /// Sliding-minimum ambient tracker feeding the adaptive threshold.
    private var ambient = AmbientFloor()
    /// When recognition last produced DIFFERENT text (the stall clock).
    private var lastTranscriptChangeAt = Date()

    /// The transcript the silence finalizer will emit. Subclasses set this as
    /// recognition results arrive (and call `publishPartial`).
    var currentTranscript = ""

    // Hooks for subclasses ---------------------------------------------------
    /// Begin recognition against the engine's input format (already running).
    func prepareRecognition(inputFormat: AVAudioFormat) throws {}
    /// Feed one captured buffer to the recognizer.
    func feed(_ buffer: AVAudioPCMBuffer) {}
    /// Start a fresh utterance (called right after each finalize).
    func resetForNextUtterance() {}
    /// Release recognition resources.
    func teardownRecognition() {}

    func requestAuthorization() async -> Bool {
        let mic = await AudioRecorder.requestPermission()
        let speech = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        return mic && speech
    }

    /// Per-prerequisite read for the pre-flight notice: prompt for mic + speech
    /// (no-ops once decided), and check whether the on-device model is installed
    /// for the current locale (voice forces `requiresOnDeviceRecognition`).
    func preflight() async -> VoicePreflight.Snapshot {
        let mic = await AudioRecorder.requestPermission()
        let speech = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0 == .authorized) }
        }
        let onDevice = SFSpeechRecognizer(locale: Locale.current)?.supportsOnDeviceRecognition ?? false
        return VoicePreflight.Snapshot(micAuthorized: mic, speechAuthorized: speech,
                                       onDeviceAvailable: onDevice, locale: Locale.current.identifier)
    }

    func start() throws {
        // Idempotent start: tear down any existing tap/engine/recognition first.
        // `AVAudioEngine` raises (and aborts the app) if a second tap is installed
        // on a bus that already has one, so a stray double-`start()` — e.g. a
        // barge-in that reopens the mic while it's still live — must never reach
        // `installTap` with the old tap in place. `stop()` is a safe no-op when
        // nothing is running. Belt-and-suspenders with the controller's own
        // stop-before-restart; this guarantees it engine-side for every caller.
        stop()

        let input = engine.inputNode
        // (Voice-processing/echo-cancellation IO was here for talk-over barge-in;
        // the mic is off while the assistant speaks, so it isn't needed in v1 and
        // enabling it has caused input-format instability on some devices.)
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw NSError(domain: "SpeechRecognizer", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No usable audio input device."])
        }

        speaking = false
        currentTranscript = ""
        partialTranscript = ""
        lastVoiceAt = Date()
        lastTranscriptChangeAt = Date()
        ambient = AmbientFloor()

        try prepareRecognition(inputFormat: format)

        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            let rms = Self.rms(of: buffer)
            self.feed(buffer)
            DispatchQueue.main.async { self.noteAudio(rms: rms) }
        }
        engine.prepare()
        try engine.start()

        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkSilence() }
        }
    }

    func stop() {
        silenceTimer?.invalidate(); silenceTimer = nil
        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning { engine.stop() }
        teardownRecognition()
        speaking = false
        level = 0
    }

    /// Subclasses call this when a new (partial) transcript is available.
    func publishPartial(_ text: String) {
        if text != currentTranscript { lastTranscriptChangeAt = Date() }
        currentTranscript = text
        partialTranscript = text
        onPartialTranscript?(text)
    }

    // MARK: VAD + endpointing (main actor)

    private func noteAudio(rms: Float) {
        level = min(1, rms * 6)
        let floor = ambient.ingest(rms)
        if rms > VoiceActivity.speechThreshold(floor: floor, minThreshold: voiceThreshold) {
            lastVoiceAt = Date()
            if !speaking { speaking = true; onSpeechStarted?() }
        }
    }

    private func checkSilence() {
        guard speaking else { return }
        let text = currentTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard VoiceActivity.shouldFinalize(
            silenceElapsed: Date().timeIntervalSince(lastVoiceAt),
            silenceTimeout: silenceTimeout,
            hasTranscript: !text.isEmpty,
            transcriptStallElapsed: Date().timeIntervalSince(lastTranscriptChangeAt),
            stallTimeout: transcriptStallTimeout) else { return }
        speaking = false
        currentTranscript = ""
        partialTranscript = ""
        resetForNextUtterance()
        if !text.isEmpty {
            onFinalTranscript?(text)
        } else {
            // Heard speech, transcribed nothing → recognition isn't working
            // (on-device dictation off / unavailable). The controller decides
            // when to surface a notice (after a couple of these in a row).
            onUnrecognizedSpeech?()
        }
    }

    static func rms(of buffer: AVAudioPCMBuffer) -> Float {
        guard let ch = buffer.floatChannelData, buffer.frameLength > 0 else { return 0 }
        let n = Int(buffer.frameLength)
        var sum: Float = 0
        for i in 0..<n { let s = ch[0][i]; sum += s * s }
        return (sum / Float(n)).squareRoot()
    }
}

// MARK: - Legacy backend (SFSpeechRecognizer, macOS 14+)

@MainActor
final class LegacySpeechRecognizer: BaseSpeechRecognizer {
    private let recognizer = SFSpeechRecognizer(locale: Locale.current)
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    override func prepareRecognition(inputFormat: AVAudioFormat) throws {
        guard let recognizer, recognizer.isAvailable else {
            throw NSError(domain: "SpeechRecognizer", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Speech recognition unavailable for this locale."])
        }
        // Privacy gate: Voice mode is on-device only. If the on-device model isn't
        // installed for this locale, refuse rather than let SFSpeechRecognizer
        // stream audio to Apple's servers (its default fallback).
        if let msg = OnDeviceSpeech.unavailableMessage(
            supportsOnDevice: recognizer.supportsOnDeviceRecognition,
            locale: Locale.current.identifier) {
            throw NSError(domain: "SpeechRecognizer", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: msg])
        }
        startRequest()
    }

    private func startRequest() {
        guard let recognizer else { return }
        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        // Always on-device — `prepareRecognition` has already verified support, so
        // audio never leaves the Mac and recognition works offline.
        req.requiresOnDeviceRecognition = true
        request = req
        task = recognizer.recognitionTask(with: req) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result { self.publishPartial(result.bestTranscription.formattedString) }
                // Recognition errors here are routine (end-of-audio on reset,
                // no-speech timeouts) and must NOT flip voice mode into an error
                // state — the silence timer owns turn endpointing. Just log.
                if let error { NSLog("[voice] SF recognition note: %@", error.localizedDescription) }
            }
        }
    }

    override func feed(_ buffer: AVAudioPCMBuffer) {
        request?.append(buffer)
    }

    override func resetForNextUtterance() {
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        startRequest()
    }

    override func teardownRecognition() {
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
    }
}

// MARK: - Modern backend (SpeechTranscriber / SpeechAnalyzer, macOS 26+)

@available(macOS 26, *)
@MainActor
final class ModernSpeechRecognizer: BaseSpeechRecognizer {
    private let transcriber = SpeechTranscriber(locale: Locale.current,
                                                preset: .progressiveTranscription)
    private var analyzer: SpeechAnalyzer?
    private var inputContinuation: AsyncStream<AnalyzerInput>.Continuation?
    private var analyzerFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var resultsTask: Task<Void, Never>?
    private var runTask: Task<Void, Never>?

    private var finalizedText = ""
    private var volatileText = ""

    override func requestAuthorization() async -> Bool {
        guard await super.requestAuthorization() else { return false }
        // Ensure the on-device model for this locale is installed.
        do {
            if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                try await req.downloadAndInstall()
            }
            return true
        } catch {
            return false
        }
    }

    override func prepareRecognition(inputFormat: AVAudioFormat) throws {
        finalizedText = ""; volatileText = ""

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.analyzer = analyzer

        let (stream, cont) = AsyncStream<AnalyzerInput>.makeStream()
        inputContinuation = cont

        // Consume transcripts: volatile results refine the current word, final
        // results commit it. We keep a running transcript for the silence timer.
        resultsTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                for try await result in self.transcriber.results {
                    let piece = String(result.text.characters)
                    if result.isFinal {
                        self.finalizedText = (self.finalizedText + " " + piece)
                            .trimmingCharacters(in: .whitespaces)
                        self.volatileText = ""
                    } else {
                        self.volatileText = piece
                    }
                    let combined = (self.finalizedText + " " + self.volatileText)
                        .trimmingCharacters(in: .whitespaces)
                    self.publishPartial(combined)
                }
            } catch {
                self.onError?(error.localizedDescription)
            }
        }

        runTask = Task { [weak self] in
            do { try await analyzer.start(inputSequence: stream) }
            catch { await MainActor.run { self?.onError?(error.localizedDescription) } }
        }

        // Resolve the format the analyzer wants and build a converter if needed.
        Task { @MainActor [weak self] in
            guard let self else { return }
            let best = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [self.transcriber])
            self.analyzerFormat = best
            if let best, best != inputFormat {
                self.converter = AVAudioConverter(from: inputFormat, to: best)
            }
        }
    }

    override func feed(_ buffer: AVAudioPCMBuffer) {
        guard let cont = inputContinuation else { return }
        guard let target = analyzerFormat, let converter else {
            cont.yield(AnalyzerInput(buffer: buffer))   // formats match (or not resolved yet)
            return
        }
        let ratio = target.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        if err == nil, out.frameLength > 0 { cont.yield(AnalyzerInput(buffer: out)) }
    }

    override func resetForNextUtterance() {
        // Keep the analyzer streaming; just clear the accumulated turn transcript.
        finalizedText = ""
        volatileText = ""
    }

    override func teardownRecognition() {
        inputContinuation?.finish()
        inputContinuation = nil
        resultsTask?.cancel(); resultsTask = nil
        let analyzer = self.analyzer
        runTask = nil
        self.analyzer = nil
        converter = nil
        Task { try? await analyzer?.finalizeAndFinishThroughEndOfInput() }
    }
}
