import Foundation
import AVFoundation

/// Voice-mode speech router: when a "Voice clone clip" is set in Settings,
/// spoken answers are synthesized on the local server in the cloned voice
/// (Qwen3-TTS `/v1/audio/speech` with `ref_audio`); when no clip is set — or a
/// synthesis fails — the utterance is delegated to the wrapped
/// `SystemSpeechSynthesizer`, so there is never dead air.
///
/// Pipelined FIFO: sentence N plays while sentence N+1 synthesizes (the same
/// producer/consumer shape the chat's streamed-answer TTS uses). `stop()`
/// clears both pipelines (barge-in) but keeps the TTS model warm;
/// `shutdown()` additionally unloads it (voice mode closed). `onQueueDrained`
/// fires only when BOTH pipelines are idle.
///
/// The clip path, synthesis, and playback are injected as closures so the
/// routing decisions are unit-testable without audio hardware or a server
/// (`ClonedVoiceSynthesizerTests`).
@MainActor
final class ClonedVoiceSynthesizer: SpeechSynthesizing {
    /// One sentence + the clip path → WAV bytes; nil = synthesis failed and
    /// the utterance falls back to the system voice.
    typealias CloneSynth = (_ text: String, _ clipPath: String) async -> Data?
    /// Play one WAV clip to completion.
    typealias ClonePlay = (_ audio: Data) async -> Void

    private let system: any SpeechSynthesizing
    private let clipPath: () -> String?
    private let synthesizeClone: CloneSynth
    private let playClone: ClonePlay
    /// Silence the in-flight clone clip immediately (barge-in).
    private let stopClonePlayback: () -> Void
    /// Release the resident TTS model (voice mode closed). nil in tests.
    private let unloadClone: (() async -> Void)?

    /// Sentences awaiting clone synthesis (pipeline stage 1).
    private var texts: [String] = []
    /// Synthesized audio — or the fallback text of a failed synthesis —
    /// awaiting playback in submit order (pipeline stage 2).
    private enum Utterance { case clone(Data), fallback(String) }
    private var playQueue: [Utterance] = []
    private var synthPumping = false
    private var playPumping = false
    /// Bumped by `stop()`; in-flight pump iterations from the old turn observe
    /// the change and exit instead of speaking into the new one.
    private var generation = 0
    /// Parks the play pump while the SYSTEM synth speaks a fallback utterance,
    /// keeping mixed clone/fallback output strictly in submit order.
    private var systemDrainContinuation: CheckedContinuation<Void, Never>?

    var onQueueDrained: (() -> Void)?
    /// The system voice picker still applies to fallback utterances; the clone
    /// path has exactly one voice (the clip) and ignores it.
    var voiceIdentifier: String? {
        get { system.voiceIdentifier }
        set { system.voiceIdentifier = newValue }
    }
    var isSpeaking: Bool {
        !texts.isEmpty || !playQueue.isEmpty || synthPumping || playPumping || system.isSpeaking
    }

    init(system: any SpeechSynthesizing,
         clipPath: @escaping () -> String?,
         synthesizeClone: @escaping CloneSynth,
         playClone: @escaping ClonePlay,
         stopClonePlayback: @escaping () -> Void = {},
         unloadClone: (() async -> Void)? = nil) {
        self.system = system
        self.clipPath = clipPath
        self.synthesizeClone = synthesizeClone
        self.playClone = playClone
        self.stopClonePlayback = stopClonePlayback
        self.unloadClone = unloadClone
        system.onQueueDrained = { [weak self] in self?.systemFinished() }
    }

    /// Production wiring: TTS on the app's server (model kept resident across
    /// sentences), playback via `AVAudioPlayer`, clip path re-read per
    /// utterance from the persisted Settings blob so a change applies to the
    /// very next sentence.
    convenience init(server: ServerManager) {
        let tts = VoiceCloneTTS(server: server)
        let player = VoiceClonePlayer()
        self.init(
            system: SystemSpeechSynthesizer(),
            clipPath: {
                // The voice picker can switch back to a system voice without
                // deleting the clip — honor that toggle here.
                let o = ServerOptions.load()
                return (o.voiceCloneEnabled && !o.voiceClonePath.isEmpty) ? o.voiceClonePath : nil
            },
            synthesizeClone: { text, clip in await tts.synthesize(text: text, refClipPath: clip) },
            playClone: { data in await player.play(data) },
            stopClonePlayback: { player.stop() },
            unloadClone: { await tts.unload() }
        )
    }

    // MARK: - SpeechSynthesizing

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let clip = clipPath(), !clip.isEmpty else {
            system.enqueue(trimmed)     // no clone configured → system voice
            return
        }
        texts.append(trimmed)
        pumpSynth(clip: clip)
    }

    func stop() {
        generation += 1
        texts.removeAll()
        playQueue.removeAll()
        synthPumping = false
        playPumping = false
        system.stop()
        stopClonePlayback()
        // Unpark a play pump waiting on the system synth; it observes the
        // generation bump and exits.
        systemDrainContinuation?.resume()
        systemDrainContinuation = nil
    }

    /// Voice mode closed — stop everything and release the resident TTS model.
    func shutdown() {
        stop()
        if let unloadClone { Task { await unloadClone() } }
    }

    // MARK: - Pipelines

    /// Stage 1: synthesize queued sentences in order, handing each result to
    /// the play queue as soon as it's ready (playback overlaps synthesis).
    private func pumpSynth(clip: String) {
        guard !synthPumping else { return }
        synthPumping = true
        let gen = generation
        Task { [weak self] in
            while let self, self.generation == gen, !self.texts.isEmpty {
                let text = self.texts.removeFirst()
                let audio = await self.synthesizeClone(text, clip)
                guard self.generation == gen else { return }
                self.playQueue.append(audio.map { .clone($0) } ?? .fallback(text))
                self.pumpPlay()
            }
            guard let self, self.generation == gen else { return }
            self.synthPumping = false
            self.maybeDrained()
        }
    }

    /// Stage 2: play results in submit order. A failed synthesis is spoken by
    /// the system synth AT ITS TURN (never reordered, never dropped).
    private func pumpPlay() {
        guard !playPumping else { return }
        playPumping = true
        let gen = generation
        Task { [weak self] in
            while let self, self.generation == gen, !self.playQueue.isEmpty {
                switch self.playQueue.removeFirst() {
                case .clone(let audio):
                    await self.playClone(audio)
                case .fallback(let text):
                    await self.speakViaSystem(text)
                }
            }
            guard let self, self.generation == gen else { return }
            self.playPumping = false
            self.maybeDrained()
        }
    }

    /// Speak one fallback utterance through the system synthesizer and wait
    /// for its queue to drain, so the play pump can't start the next clone
    /// clip over the top of it.
    private func speakViaSystem(_ text: String) async {
        await withCheckedContinuation { cont in
            systemDrainContinuation = cont
            system.enqueue(text)
        }
    }

    private func systemFinished() {
        if let cont = systemDrainContinuation {
            systemDrainContinuation = nil
            cont.resume()
            return
        }
        maybeDrained()  // pure-system path (no clip configured)
    }

    private func maybeDrained() {
        guard !isSpeaking else { return }
        onQueueDrained?()
    }
}

// MARK: - Clip persistence (pure path contract, testable)

/// Where the normalized voice-clone clip lives. The OS temp dir (where
/// `AudioReference` writes) gets swept, so Settings copies the clip to a
/// stable location that survives relaunch.
enum VoiceCloneClipStore {
    static var directory: String {
        NSString(string: "~/.mlx-serve/voice-clips").expandingTildeInPath
    }

    /// The single global clip path (re-recording overwrites it).
    static var destinationPath: String {
        (directory as NSString).appendingPathComponent("voice-clone.wav")
    }

    /// Copy a normalized clip into the stable location and return the path to
    /// persist. Copy failure falls back to the source path rather than losing
    /// the clip.
    static func persist(_ url: URL) -> String {
        try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        let dest = destinationPath
        try? FileManager.default.removeItem(atPath: dest)
        do {
            try FileManager.default.copyItem(at: url, to: URL(fileURLWithPath: dest))
            return dest
        } catch {
            return url.path
        }
    }
}

// MARK: - Production audio helpers (not unit-tested — pure I/O)

/// Synthesizes one sentence at a time on the native server's
/// `/v1/audio/speech` (Qwen3-TTS, zero-shot cloning via `ref_audio`). The TTS
/// model is loaded once and kept resident across sentences — per-sentence
/// load/unload would stall the pipeline — and released by `unload()` when
/// voice mode closes.
@MainActor
final class VoiceCloneTTS {
    private let server: ServerManager
    private let api = APIClient()
    private var loadedModelId: String?
    private var loadedDir: String?

    init(server: ServerManager) { self.server = server }

    func synthesize(text: String, refClipPath: String?) async -> Data? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let preset = AudioGenSettings.load().resolvedModel
        guard let dir = ServerManager.resolveModelDir(repo: preset.repo) else { return nil }
        do {
            let port = try await server.ensureRunning(forGenModelDir: dir)
            if loadedModelId == nil || loadedDir != dir {
                let info = try await server.loadModel(id: dir)
                loadedModelId = info.name
                loadedDir = dir
            }
            var json: [String: Any] = ["model": loadedModelId ?? dir, "input": trimmed]
            if let refClipPath, let data = try? Data(contentsOf: URL(fileURLWithPath: refClipPath)) {
                json["ref_audio"] = data.base64EncodedString()
            }
            var wav: Data?
            for try await ev in api.streamGeneration(port: port, path: "/v1/audio/speech", json: json) {
                if ev["type"] as? String == "complete", let b64 = ev["data"] as? String {
                    wav = Data(base64Encoded: b64)
                }
            }
            return wav
        } catch {
            return nil
        }
    }

    func unload() async {
        if let id = loadedModelId { try? await server.unloadModel(id: id) }
        loadedModelId = nil
        loadedDir = nil
    }
}

/// Plays one WAV clip to completion (serial playback — voice mode speaks one
/// sentence at a time). Bridges `AVAudioPlayerDelegate` to `async`.
final class VoiceClonePlayer: NSObject, AVAudioPlayerDelegate {
    private var player: AVAudioPlayer?
    private var continuation: CheckedContinuation<Void, Never>?

    func play(_ data: Data) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            do {
                let p = try AVAudioPlayer(data: data)
                p.delegate = self
                self.player = p
                self.continuation = cont
                if !p.play() { self.finish() }
            } catch {
                cont.resume()
            }
        }
    }

    func stop() {
        player?.stop()
        finish()
    }

    private func finish() {
        continuation?.resume()
        continuation = nil
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        finish()
    }
}
