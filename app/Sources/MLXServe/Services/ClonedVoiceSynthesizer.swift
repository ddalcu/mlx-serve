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
/// Which neural voice an utterance should use, or nil for the system voice.
///
/// The two arms are DIFFERENT BACKENDS with disjoint controls, which is why
/// this is a sum type and not a pair of optional strings: Qwen3-TTS clones from
/// a clip and has no voice list, Kokoro names a built-in voice (or a
/// comma-separated blend) and cannot clone. Sending the wrong field is a named
/// 400 server-side.
enum NeuralVoice: Equatable, Sendable {
    case clone(clipPath: String)
    case kokoro(voice: String)
}

final class ClonedVoiceSynthesizer: SpeechSynthesizing {
    /// One sentence + the clip path → WAV bytes; nil = synthesis failed and
    /// the utterance falls back to the system voice.
    typealias CloneSynth = (_ text: String, _ voice: NeuralVoice) async -> Data?
    /// Play one WAV clip to completion.
    typealias ClonePlay = (_ audio: Data) async -> Void

    private let system: any SpeechSynthesizing
    private let voice: () -> NeuralVoice?
    private let synthesizeClone: CloneSynth
    private let playClone: ClonePlay
    /// Silence the in-flight clone clip immediately (barge-in).
    private let stopClonePlayback: () -> Void
    /// Release the resident TTS model (voice mode closed). nil in tests.
    private let unloadClone: (() async -> Void)?
    /// Pre-embed the clone reference clip (voice mode opened). nil = no-op.
    private let prewarmClone: ((NeuralVoice) async -> Void)?

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
         voice: @escaping () -> NeuralVoice?,
         synthesizeClone: @escaping CloneSynth,
         playClone: @escaping ClonePlay,
         stopClonePlayback: @escaping () -> Void = {},
         unloadClone: (() async -> Void)? = nil,
         prewarmClone: ((NeuralVoice) async -> Void)? = nil) {
        self.system = system
        self.voice = voice
        self.synthesizeClone = synthesizeClone
        self.playClone = playClone
        self.stopClonePlayback = stopClonePlayback
        self.unloadClone = unloadClone
        self.prewarmClone = prewarmClone
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
            voice: {
                // Re-read per utterance so a Settings change — or the active
                // agent's own voice, which is preferred here — applies to the
                // very next sentence, with no restart.
                ActiveAgentVoice.currentNeuralVoice(options: ServerOptions.load())
            },
            synthesizeClone: { text, sel in await tts.synthesize(text: text, voice: sel) },
            playClone: { data in await player.play(data) },
            stopClonePlayback: { player.stop() },
            unloadClone: { await tts.unload() },
            prewarmClone: { sel in await tts.prewarm(voice: sel) }
        )
    }

    /// Voice mode opened: pre-embed the clone reference clip so the FIRST
    /// sentence skips the cold speaker-encoder forward (the only per-session
    /// cost the content-keyed server cache cannot hide). Only the clone arm
    /// has anything to warm — Kokoro voices are a table lookup, the system
    /// voice needs nothing — and failure is fine: the first sentence simply
    /// pays what it always paid.
    func prewarm() {
        guard let prewarmClone, case let sel? = voice(), case .clone = sel else { return }
        Task { await prewarmClone(sel) }
    }

    // MARK: - SpeechSynthesizing

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let sel = voice() else {
            system.enqueue(trimmed)     // no neural voice configured → system
            return
        }
        texts.append(trimmed)
        pumpSynth(voice: sel)
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
    private func pumpSynth(voice sel: NeuralVoice) {
        guard !synthPumping else { return }
        synthPumping = true
        let gen = generation
        Task { [weak self] in
            while let self, self.generation == gen, !self.texts.isEmpty {
                let text = self.texts.removeFirst()
                let audio = await self.synthesizeClone(text, sel)
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

    func synthesize(text: String, voice sel: NeuralVoice) async -> Data? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        // The engine choice picks the MODEL too — Kokoro is its own checkpoint,
        // not a mode of the Qwen3-TTS one.
        // The clone model is resolved against the DISK, not just read from the
        // Audio pane: the configured default is the 0.6B repo, so a machine with
        // only the 1.7B variants used to resolve nothing and fall back silently to
        // the system voice.
        let preset: AudioModelPreset? = switch sel {
        case .kokoro: .kokoro82M
        case .clone: VoiceCloneMenuModel.resolvedCloneModel()
        }
        guard let preset, let dir = ServerManager.resolveModelDir(repo: preset.repo) else { return nil }
        do {
            let port = try await server.ensureRunning(forGenModelDir: dir)
            if loadedModelId == nil || loadedDir != dir {
                let info = try await server.loadModel(id: dir)
                loadedModelId = info.name
                loadedDir = dir
            }
            let json = VoiceCloneTTS.requestBody(model: loadedModelId ?? dir, text: trimmed, voice: sel)
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

    /// Build the `/v1/audio/speech` body. PURE and static so the field choice
    /// is testable: the fakes in `ClonedVoiceSynthesizerTests` stub out this
    /// whole class, so swapping `voice` for `ref_audio` here passed every
    /// routing test while being a guaranteed 400 in production.
    ///
    /// Sends ONLY the field the chosen backend accepts — the other one is a
    /// named 400 server-side, not an ignored extra.
    static func requestBody(model: String, text: String, voice sel: NeuralVoice) -> [String: Any] {
        var json: [String: Any] = ["model": model, "input": text]
        switch sel {
        case .kokoro(let v):
            json["voice"] = v
        case .clone(let clip):
            if let data = try? Data(contentsOf: URL(fileURLWithPath: clip)) {
                json["ref_audio"] = data.base64EncodedString()
            }
        }
        return json
    }

    func unload() async {
        if let id = loadedModelId { try? await server.unloadModel(id: id) }
        loadedModelId = nil
        loadedDir = nil
    }

    /// Pre-warm the server's speaker-embedding cache for the clip
    /// (docs/qwentts-cache.md): loads the TTS model if needed (voice mode is
    /// about to speak through it anyway) and POSTs `warm_only` so the first
    /// sentence's synthesis starts from a cache hit. Best-effort.
    func prewarm(voice sel: NeuralVoice) async {
        guard case .clone = sel else { return }
        let preset: AudioModelPreset? = VoiceCloneMenuModel.resolvedCloneModel()
        guard let preset, let dir = ServerManager.resolveModelDir(repo: preset.repo) else { return }
        do {
            let port = try await server.ensureRunning(forGenModelDir: dir)
            if loadedModelId == nil || loadedDir != dir {
                let info = try await server.loadModel(id: dir)
                loadedModelId = info.name
                loadedDir = dir
            }
            guard let json = VoiceCloneTTS.warmBody(model: loadedModelId ?? dir, voice: sel) else { return }
            var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/audio/speech")!)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try? JSONSerialization.data(withJSONObject: json)
            _ = try? await URLSession.shared.data(for: req)
        } catch {
            // First sentence pays the cold embed — exactly the pre-warm-less behavior.
        }
    }

    /// Build the `warm_only` body. PURE and static (the requestBody pattern)
    /// so the field choice is testable: `warm_only` + `ref_audio`, never
    /// `input` — a body with text would synthesize audio nobody asked for.
    /// nil when the clip can't be read (nothing to warm with).
    static func warmBody(model: String, voice sel: NeuralVoice) -> [String: Any]? {
        guard case let .clone(clip) = sel,
              let data = try? Data(contentsOf: URL(fileURLWithPath: clip)) else { return nil }
        return ["model": model, "warm_only": true, "ref_audio": data.base64EncodedString()]
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
