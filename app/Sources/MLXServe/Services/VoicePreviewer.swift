import Foundation

/// Plays a short sample of a Kokoro voice so the picker is auditionable rather
/// than a list of names.
///
/// Previews are SUPERSEDED, not queued: clicking through five voices must play
/// the fifth, not all five in a row. Each request takes a generation number and
/// a late-arriving result whose generation is stale is dropped on the floor
/// (the `ClonedVoiceSynthesizer` barge-in pattern).
///
/// The synthesis + playback seams are injected so `VoicePreviewerTests` can
/// drive the ordering rules without a server or an audio device.
@MainActor
final class VoicePreviewer: ObservableObject {
    /// The spec currently being synthesized or played; nil when idle. Drives the
    /// spinner, and is a spec rather than a Bool so the row that was CLICKED is
    /// the one that shows activity.
    @Published private(set) var active: String?
    /// Set when a preview could not be produced (model missing, server down).
    @Published private(set) var error: String?

    typealias Synthesize = (_ text: String, _ voice: String) async -> Data?
    typealias Play = (Data) async -> Void

    private var synthesize: Synthesize
    private var play: Play
    private var stopPlayback: () -> Void
    private var generation = 0
    private var attached = false
    /// Handle on the in-flight preview. Exposed so tests can await completion
    /// deterministically instead of yielding a fixed number of times.
    private(set) var inFlight: Task<Void, Never>?

    /// Un-attached: every preview reports the "not available" error until
    /// `attach(server:)` runs. A view holds this as `@StateObject`, which cannot
    /// see the environment at init — hence attaching in `.onAppear`, the same
    /// per-call server injection `AudioGenService` uses.
    init(synthesize: @escaping Synthesize = { _, _ in nil },
         play: @escaping Play = { _ in },
         stopPlayback: @escaping () -> Void = {}) {
        self.synthesize = synthesize
        self.play = play
        self.stopPlayback = stopPlayback
    }

    /// Bind to the app's server. Idempotent — `.onAppear` fires more than once
    /// and rebuilding the TTS client would drop the resident model, making every
    /// preview pay a reload.
    func attach(server: ServerManager) {
        guard !attached else { return }
        attached = true
        let tts = VoiceCloneTTS(server: server)
        let player = VoiceClonePlayer()
        synthesize = { text, voice in
            await tts.synthesize(text: text, voice: .kokoro(voice: voice))
        }
        play = { data in await player.play(data) }
        stopPlayback = { player.stop() }
    }

    /// Audition `spec` (a voice id, or a comma-separated blend).
    func preview(_ spec: String) {
        let voice = spec.trimmingCharacters(in: .whitespaces)
        guard !voice.isEmpty else { return }

        generation += 1
        let gen = generation
        stopPlayback()          // cut whatever is sounding now
        error = nil
        active = voice

        inFlight = Task { [weak self] in
            guard let self else { return }
            let text = KokoroVoiceCatalog.previewSentence(for: voice)
            let data = await self.synthesize(text, voice)
            // A newer preview started while this one was synthesizing — drop it
            // rather than talking over the voice the user actually wants.
            guard self.generation == gen else { return }
            guard let data else {
                self.error = "Couldn't play a preview. The Kokoro model may not be downloaded yet."
                self.active = nil
                return
            }
            await self.play(data)
            guard self.generation == gen else { return }
            self.active = nil
        }
    }

    /// Play a stored clip AS IS — the file the user uploaded or picked.
    ///
    /// Deliberately not a synthesis: auditioning a reference clip is "is this the
    /// right recording?", which the raw file answers instantly and with no model
    /// downloaded. Same generation/supersede rules as `preview` so clicking
    /// through clips plays the last one, not all of them.
    func playClip(path: String) {
        let path = path.trimmingCharacters(in: .whitespaces)
        guard !path.isEmpty else { return }

        generation += 1
        let gen = generation
        stopPlayback()
        error = nil
        active = path

        inFlight = Task { [weak self] in
            guard let self else { return }
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)), !data.isEmpty else {
                self.error = "Couldn't read that clip — it may have been moved or deleted."
                self.active = nil
                return
            }
            guard self.generation == gen else { return }
            await self.play(data)
            guard self.generation == gen else { return }
            self.active = nil
        }
    }

    /// Stop anything sounding and abandon in-flight work.
    func stop() {
        generation += 1
        stopPlayback()
        active = nil
    }

    func isPreviewing(_ spec: String) -> Bool {
        active == spec.trimmingCharacters(in: .whitespaces)
    }
}
