import Foundation
import Combine
import AVFoundation

/// One turn of the avatar conversation, kept for on-screen history and to give
/// the model context on the next turn.
struct AvatarTurn: Equatable {
    enum Role: String { case user, assistant }
    let role: Role
    var text: String
}

/// The avatar "talk-back" loop (plan §7): a spoken user turn (mic) or typed text
/// runs a plain-chat turn against the local LLM under the persona's system
/// prompt; the streamed answer is split into COMPLETE sentences by
/// `SentenceStreamer` and each sentence is synthesized (cloned voice via
/// `ref_audio`) and played back-to-back WHILE the next sentence is still
/// decoding. This is the only genuinely new engineering — a small FIFO queue so
/// TTS jobs interleave with LLM decode without either starving the other.
///
/// The engine is **window-independent** and runs its OWN generation (never the
/// shared `ChatTurnEngine`), so opening the avatar never fights the chat window.
/// Everything that isn't audio/mic is injected behind closures so the whole
/// orchestration is unit-testable with fakes:
/// - `Responder` — user text → a stream of answer deltas (production: the LLM).
/// - `Synthesizer` — one sentence → audio bytes (production: Qwen3-TTS).
/// - `Player` — play one clip to completion (production: `AVAudioPlayer`).
@MainActor
final class AvatarEngine: ObservableObject {

    enum State: String, Equatable {
        case idle        // nothing happening
        case listening   // mic open, waiting for the user to speak
        case thinking    // turn submitted, waiting for the first audio clip
        case speaking    // playing the answer, clip by clip
    }

    // MARK: - Injected collaborators

    /// user text (+ system prompt + prior turns) → a stream of answer deltas.
    typealias Responder = (_ system: String, _ history: [AvatarTurn], _ user: String)
        -> AsyncThrowingStream<String, Error>
    /// one sentence (+ the persona's clone clip path) → audio bytes (nil = skip).
    typealias Synthesizer = (_ text: String, _ voiceClipPath: String?) async -> Data?
    /// play one clip to completion.
    typealias Player = (_ audio: Data) async -> Void
    /// the user's utterance → retrieved document excerpts to ground the answer
    /// in (nil / empty = nothing to inject). Injected as a seam so tests use a
    /// fake; when nil, production retrieves from the persona's own
    /// `documentIndex`.
    typealias Retriever = (_ query: String) async -> String?

    private let respond: Responder
    private let synthesize: Synthesizer
    private let play: Player
    /// Injected retrieval seam (tests). nil → retrieve from `documentIndex`.
    private let retrieve: Retriever?
    /// Optional mic backend — the SAME service the voice assistant uses
    /// (`SpeechRecognizing`). nil in tests (no audio) and when a mic is absent.
    private let recognizer: (any SpeechRecognizing)?
    /// Release resident audio resources on `stop()` (production: unload the TTS
    /// model + stop the player). nil in tests.
    private let teardown: (() async -> Void)?
    /// Builds a document index for a persona's folder — the SAME mini-RAG the
    /// chat's "attach a folder" uses (GPU embeddings, lexical fallback). nil in
    /// tests, which set `documentIndex` directly.
    typealias IndexBuilder = (_ folder: URL) -> DocumentIndex
    private let buildIndex: IndexBuilder?

    /// Held-back-fragment threshold for `SentenceStreamer` — merges tiny
    /// fragments ("Ok.") into the next sentence so playback isn't machine-gunned.
    let minChars: Int

    // MARK: - Published state (observed by the view)

    @Published private(set) var state: State = .idle
    /// The answer as it streams — the on-screen subtitle.
    @Published private(set) var subtitle: String = ""
    /// Spoken conversation so far (drives context + an optional transcript view).
    @Published private(set) var transcript: [AvatarTurn] = []
    /// True while the mic is actively capturing.
    @Published private(set) var micActive = false
    /// Normalized [0,1] speech amplitude of the CURRENTLY PLAYING clip,
    /// attack/release-smoothed (SkeletalAnimator) — drives the avatar's jaw
    /// (P3.4). 0 while not speaking.
    @Published private(set) var speechAmplitude: Double = 0
    /// The active persona. Editing it (via the window) takes effect on the next
    /// turn; `voiceClipPath`/`systemPrompt` are read at submit time. Changing the
    /// document folder (re)builds the RAG index.
    @Published var persona: AvatarPersona {
        didSet { reconcileDocumentIndex(from: oldValue) }
    }
    /// The persona's document index (mini-RAG), rebuilt when the folder changes.
    /// `nil` → no retrieval. Published so the window can show an indexing chip;
    /// tests set it directly (via `DocumentIndex.finishForTesting`).
    @Published var documentIndex: DocumentIndex?

    /// The in-flight turn's task (producer). Exposed for tests to await.
    private(set) var turnTask: Task<Void, Never>?
    /// The in-flight turn's clip consumer.
    private var consumerTask: Task<Void, Never>?
    /// A continuous-listen session (mic reopens after each turn) is active.
    private var listenSession = false

    // MARK: - Init

    init(persona: AvatarPersona,
         minChars: Int = 16,
         respond: @escaping Responder,
         synthesize: @escaping Synthesizer,
         play: @escaping Player,
         retrieve: Retriever? = nil,
         recognizer: (any SpeechRecognizing)? = nil,
         teardown: (() async -> Void)? = nil,
         buildIndex: IndexBuilder? = nil) {
        self.persona = persona
        self.minChars = minChars
        self.respond = respond
        self.synthesize = synthesize
        self.play = play
        self.retrieve = retrieve
        self.recognizer = recognizer
        self.teardown = teardown
        self.buildIndex = buildIndex
        wireRecognizer()
        // `didSet` doesn't fire for the init assignment — build the index for the
        // starting persona's folder explicitly.
        reconcileDocumentIndex(from: nil)
    }

    /// Production wiring: LLM answers via `APIClient`, TTS via the native server,
    /// playback via `AVAudioPlayer`, mic via the shared speech recognizer.
    convenience init(appState: AppState) {
        let voice = AvatarVoice(server: appState.server)
        let player = AvatarClipPlayer()
        self.init(
            persona: AvatarPersonaStore.load().selectedPersona,
            respond: { system, history, user in
                AvatarEngine.streamReply(appState: appState, system: system,
                                         history: history, user: user)
            },
            synthesize: { text, clip in await voice.synthesize(text: text, refClipPath: clip) },
            play: { data in await player.play(data) },
            recognizer: makeSpeechRecognizer(),
            teardown: { await voice.unload(); player.stop() },
            buildIndex: { folder in
                // Same GPU-embedding mini-RAG as the chat's "attach a folder";
                // server down → lexical-only retrieval.
                DocumentIndex(folderURL: folder,
                              embedderProvider: ServerEmbedding.autoProvider(port: appState.server.port))
            }
        )
        // P3.4: the playing clip's metered envelope drives the jaw.
        player.onAmplitude = { [weak self] amp in
            Task { @MainActor [weak self] in self?.ingestSpeechAmplitude(amp) }
        }
    }

    /// Smooth (fast attack / slow release) and publish the speech envelope.
    func ingestSpeechAmplitude(_ raw: Double) {
        speechAmplitude = raw <= 0 && speechAmplitude < 0.02
            ? 0
            : SkeletalAnimator.smoothedAmplitude(previous: speechAmplitude, target: raw)
    }

    // MARK: - Turn submission

    /// Can a new turn start right now? Only from a quiescent state — the engine
    /// runs ONE turn at a time (mirrors `ChatTurnEngine`), so a submit while
    /// thinking/speaking is silently ignored.
    var canSubmit: Bool { state == .idle || state == .listening }

    /// Submit a user utterance (typed, or a finalized mic transcript). Sets the
    /// visible `.thinking` transition synchronously so the caller/tests see it
    /// immediately, then runs the streamed answer + speech pipeline in the
    /// background.
    func submit(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard canSubmit, !trimmed.isEmpty else { return }
        stopMicForTurn()                 // mic off while the model thinks/speaks
        let prior = transcript           // context is everything BEFORE this turn
        transcript.append(AvatarTurn(role: .user, text: trimmed))
        subtitle = ""
        state = .thinking
        turnTask = Task { [weak self] in await self?.runTurn(user: trimmed, prior: prior) }
    }

    /// The streamed-answer + sentence-TTS pipeline. Producer (this method) reads
    /// LLM deltas and yields COMPLETE sentences into a FIFO channel; the consumer
    /// synthesizes + plays each in order. The two run concurrently, so sentence N
    /// is spoken while sentence N+1 is still decoding.
    private func runTurn(user: String, prior: [AvatarTurn]) async {
        // Ground the answer in the persona's documents (mini-RAG) when attached.
        // Retrieval happens during `.thinking`, before the first token.
        let system = await augmentedSystemPrompt(for: user)
        let clip = persona.voiceClipPath

        let (sentences, sink) = AsyncStream<String>.makeStream()
        consumerTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for await sentence in sentences {
                if Task.isCancelled { break }
                let audio = await self.synthesize(sentence, clip)
                guard let audio else { continue }
                if self.state == .thinking { self.state = .speaking }   // first real clip
                // P3 seam: this clip's amplitude envelope would drive the jaw
                // skinner here (drive a bone open/close from `audio`'s RMS).
                await self.play(audio)
            }
        }

        var streamer = SentenceStreamer(minChars: minChars)
        var full = ""
        do {
            for try await delta in respond(system, prior, user) {
                try Task.checkCancellation()
                full += delta
                subtitle = full
                for sentence in streamer.feed(delta) { sink.yield(sentence) }
            }
        } catch is CancellationError {
            // Stopped by the user — fall through to drain what's queued and reset.
        } catch {
            if full.isEmpty { subtitle = "(couldn't reach the model)" }
        }
        if let tail = streamer.flush() { sink.yield(tail) }
        sink.finish()
        await consumerTask?.value
        consumerTask = nil

        if !full.isEmpty { transcript.append(AvatarTurn(role: .assistant, text: full)) }
        finishTurn()
        turnTask = nil
    }

    /// Cancel the in-flight turn and any playback, release audio resources, and
    /// drop back to idle (leaving the mic off).
    func stop() {
        turnTask?.cancel(); turnTask = nil
        consumerTask?.cancel(); consumerTask = nil
        stopListening()
        if let teardown { Task { await teardown() } }
        state = .idle
    }

    // MARK: - Mic (production only; recognizer is nil in tests)

    /// Toggle a continuous-listen session: the mic reopens automatically after
    /// each spoken answer so the conversation flows hands-free.
    func toggleListening() {
        if listenSession { stopListening() } else { startListening() }
    }

    /// Request mic + speech permission (no-op once granted) and then start the
    /// listen session. The view's mic button uses this so the first tap prompts
    /// instead of silently failing to open the mic.
    func requestMicThenListen() async {
        guard let recognizer else { return }
        _ = await recognizer.requestAuthorization()
        startListening()
    }

    func startListening() {
        guard let recognizer else { return }
        listenSession = true
        do {
            try recognizer.start()
            micActive = true
            if state == .idle { state = .listening }
        } catch {
            listenSession = false
            micActive = false
        }
    }

    func stopListening() {
        listenSession = false
        recognizer?.stop()
        micActive = false
        if state == .listening { state = .idle }
    }

    /// Turn the mic off for the duration of a turn WITHOUT ending the listen
    /// session, so it can reopen when the answer finishes.
    private func stopMicForTurn() {
        recognizer?.stop()
        micActive = false
    }

    private func wireRecognizer() {
        recognizer?.onFinalTranscript = { [weak self] text in self?.submit(text) }
    }

    /// End-of-turn state: reopen the mic if we're in a listen session, else idle.
    private func finishTurn() {
        if listenSession, let recognizer {
            do { try recognizer.start(); micActive = true; state = .listening }
            catch { listenSession = false; micActive = false; state = .idle }
        } else {
            state = .idle
        }
    }

    // MARK: - Document RAG (mini-RAG grounding)

    /// The persona prompt, plus retrieved document excerpts when a folder is
    /// attached and indexed. No retriever/index / not ready / no matches → the
    /// prompt is returned unchanged.
    private func augmentedSystemPrompt(for query: String) async -> String {
        guard let context = await retrieveContext(for: query),
              !context.isEmpty else { return persona.systemPrompt }
        return persona.systemPrompt + "\n\nReference material:\n" + context
    }

    /// Retrieve grounding excerpts for `query`: the injected seam wins (tests);
    /// otherwise the persona's own `documentIndex` (production). nil when there's
    /// nothing to inject (no folder, still indexing, no matches).
    private func retrieveContext(for query: String) async -> String? {
        if let retrieve { return await retrieve(query) }
        guard let index = documentIndex, case .ready = index.state else { return nil }
        let excerpts = await index.search(query: query, topK: 4)
        return excerpts.isEmpty ? nil : excerpts
    }

    /// (Re)build the index when the persona's document folder changes. Clearing
    /// the folder drops the index; setting one starts indexing immediately.
    private func reconcileDocumentIndex(from old: AvatarPersona?) {
        guard persona.docFolderPath != old?.docFolderPath else { return }
        documentIndex?.cancel()
        documentIndex = nil
        guard let path = persona.docFolderPath, !path.isEmpty, let buildIndex else { return }
        let index = buildIndex(URL(fileURLWithPath: path))
        documentIndex = index
        index.startIndexing()
    }

    // MARK: - Production LLM stream

    /// Build the plain-chat request (persona system prompt + prior turns + the
    /// new user turn) and map its content deltas into a plain string stream.
    /// Thinking/tool events are ignored — the avatar is plain chat.
    static func streamReply(appState: AppState, system: String,
                            history: [AvatarTurn], user: String) -> AsyncThrowingStream<String, Error> {
        var messages: [[String: Any]] = [["role": "system", "content": spokenStyle(system)]]
        for turn in history { messages.append(["role": turn.role.rawValue, "content": turn.text]) }
        messages.append(["role": "user", "content": user])

        let api = APIClient()
        let sse = api.streamChat(
            port: appState.server.port,
            messages: messages,
            maxTokens: appState.maxTokens,
            temperature: appState.serverOptions.defaultTemperature,
            enableThinking: false,
            defaults: APIClient.RequestDefaults.from(appState.serverOptions),
            modelId: appState.server.modelInfo?.name
        )
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in sse {
                        if case .content(let text) = event { continuation.yield(text) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Reinforce spoken-answer style on top of whatever persona the user wrote,
    /// so the avatar never reads Markdown/URLs aloud even with a terse prompt.
    private static func spokenStyle(_ persona: String) -> String {
        persona + "\n\nKeep every reply short and speakable: plain sentences, no Markdown, no URLs, no lists."
    }
}

// MARK: - Production audio helpers (not unit-tested — pure I/O)

/// Synthesizes one sentence at a time on the native server's `/v1/audio/speech`
/// (Qwen3-TTS, zero-shot cloning via `ref_audio`). The TTS model is loaded once
/// and kept resident across sentences — per-sentence load/unload would stall the
/// pipeline — and released by `unload()` when the conversation stops.
@MainActor
final class AvatarVoice {
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

/// Plays one WAV clip to completion (serial playback — the avatar speaks one
/// sentence at a time). Bridges `AVAudioPlayerDelegate` to `async`, and meters
/// the playing clip's amplitude at ~30 Hz (P3.4: the envelope drives the jaw).
final class AvatarClipPlayer: NSObject, AVAudioPlayerDelegate {
    private var player: AVAudioPlayer?
    private var continuation: CheckedContinuation<Void, Never>?
    private var meterTimer: Timer?
    /// Normalized [0,1] amplitude while a clip plays; a final 0 at clip end.
    var onAmplitude: ((Double) -> Void)?

    func play(_ data: Data) async {
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            do {
                let p = try AVAudioPlayer(data: data)
                p.delegate = self
                p.isMeteringEnabled = true
                self.player = p
                self.continuation = cont
                if p.play() { self.startMetering(p) } else { self.finish() }
            } catch {
                cont.resume()
            }
        }
    }

    func stop() {
        player?.stop()
        finish()
    }

    private func startMetering(_ p: AVAudioPlayer) {
        meterTimer?.invalidate()
        let timer = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self, weak p] _ in
            guard let self, let p, p.isPlaying else { return }
            p.updateMeters()
            // averagePower dB [-50, 0] → [0, 1].
            let db = Double(p.averagePower(forChannel: 0))
            self.onAmplitude?(min(max((db + 50.0) / 50.0, 0.0), 1.0))
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    private func finish() {
        meterTimer?.invalidate()
        meterTimer = nil
        onAmplitude?(0)
        continuation?.resume()
        continuation = nil
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        finish()
    }
}
