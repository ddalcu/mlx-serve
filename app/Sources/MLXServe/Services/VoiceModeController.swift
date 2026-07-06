import Foundation
import Combine

/// Drives one hands-free voice conversation: mic → transcript → submit the turn
/// to the shared `ChatTurnEngine` → observe the streamed answer via Combine →
/// speak it sentence-by-sentence → reopen the mic. Speech I/O is injected behind
/// protocols so the whole orchestration is unit-testable with fakes; the turn
/// runner is injected behind `TurnRunning` so tests can use a fake instead of a
/// live server.
///
/// The controller is **self-driving and window-independent**: once `bind(appState:)`
/// has wired it to the app, it submits turns and observes the active session with
/// no SwiftUI view in the loop. The in-window orb and the menu-bar tray panel are
/// just two views of the same controller.
@MainActor
final class VoiceModeController: ObservableObject {
    @Published private(set) var state: VoiceTurnState = .idle
    @Published private(set) var partialTranscript = ""
    @Published private(set) var level: Float = 0
    @Published private(set) var isMuted = false
    /// True from a successful `begin()` until `end()`. The tray toggle binds to
    /// this; it stays true through a recoverable error so the panel can show it.
    @Published private(set) var isActive = false

    /// Set when enabling Voice hit a missing prerequisite (mic / speech
    /// permission, or the on-device dictation model). Drives a non-invasive
    /// notice + "Open Settings" button in the voice panel; cleared on a clean
    /// start or on `end()`. nil = nothing to nag about.
    @Published private(set) var setupIssue: VoicePreflight.Issue?

    /// Consecutive turns where the mic heard speech but recognition returned
    /// nothing — the runtime signature of on-device dictation being off (the
    /// pre-flight can't see this; `supportsOnDeviceRecognition` stays true).
    private var unrecognizedSpeechStreak = 0
    /// True once recognition has produced ANY text this session (partial or
    /// final). That's positive proof on-device dictation is installed + on, so
    /// the "dictation unavailable" notice is then categorically wrong — later
    /// empty endpoints are just noise (a cough, the assistant's own TTS bleeding
    /// into the mic) and must never surface it. The empty-speech streak only
    /// diagnoses a genuinely-dead recognizer, i.e. before any text appears.
    private var hasRecognizedSpeech = false
    /// Surface the notice after this many empty-speech turns in a row — one stray
    /// noise shouldn't nag; two in a row means recognition really isn't working.
    private static let unrecognizedSpeechLimit = 2

    /// Hands-free agent runs: defaults ON for the tray voice assistant (it's
    /// hands-free by design). When OFF, each tool call surfaces a `pendingApproval`
    /// card in the panel/orb and the turn waits until the user allows or denies.
    @Published var autoApproveTools = true

    /// Voice-scoped Think / Agent / MCP toggles. Seeded from the chat session the
    /// user launches voice mode in (`ChatView.startVoiceMode`), then independent of
    /// the chat window's toolbar so talking and typing can run in different modes.
    @Published var agentMode = false
    @Published var enableThinking = false
    @Published var mcpMode = false

    /// Wake-word gate. When true (the default), the assistant ignores everything
    /// it hears until an utterance opens with the wake phrase ("Hey Loki"); the
    /// rest of that utterance becomes the query. A bare wake phrase arms the
    /// *next* utterance, so you can pause after "Hey Loki" and then speak the
    /// command. Persisted across launches.
    @Published var requireWakeWord: Bool {
        didSet {
            UserDefaults.standard.set(requireWakeWord, forKey: Self.wakeWordDefaultsKey)
            if !requireWakeWord { disarmFollowUp() }
        }
    }

    /// Set after a bare wake phrase ("Hey Loki" with no command); the next
    /// utterance is taken as the query without needing the wake word again.
    /// Surfaced as a "Go ahead…" hint in the tray/orb.
    @Published private(set) var awaitingWakeQuery = false

    /// The phrase to listen for. Fixed to "Hey Loki" today; a stored property so
    /// tests can override it and a future settings field can change it.
    var wakePhrase: String = WakeWord.defaultPhrase

    /// Seconds of continued listening after the assistant answers (or after a
    /// bare wake phrase) during which a follow-up needs no wake word — roughly
    /// Google Assistant's "continued conversation" window (Alexa's follow-up
    /// mode is ~5 s). Each completed turn reopens it, so a back-and-forth flows
    /// without repeating "Hey Loki"; after it elapses in silence the wake word
    /// is required again. Tunable; a future settings field can expose it.
    var followUpWindow: TimeInterval = 8

    private let followUpTimer: FollowUpTimer
    private static let wakeWordDefaultsKey = "voiceModeRequireWakeWord"

    /// A tool call awaiting the user's decision (agent mode, auto-approve off).
    /// Surfaced in the tray panel and the in-window orb; resolved via `resolve(_:)`.
    @Published var pendingApproval: ToolApprovalRequest?

    /// Voices offered in the picker (best quality first) and the chosen one.
    @Published private(set) var availableVoices: [VoiceOption] = []
    @Published private(set) var selectedVoiceId: String?

    /// The shared generation engine. Wired by `bind(appState:)` in production;
    /// set directly with a fake in tests.
    var runner: (any TurnRunning)?
    /// Resolves the chat session a new turn runs against (creating one if needed)
    /// plus the app-level MCP flag and working directory for the turn config.
    /// Wired by `bind(appState:)`; set directly in tests.
    var turnContext: (() -> (sessionId: UUID, workingDirectory: String?))?

    private let recognizer: any SpeechRecognizing
    private let synthesizer: any SpeechSynthesizing
    private let loadingCue: any LoadingCue
    private let chime: WakeChime
    private var chunker = SentenceChunker()
    private var streamComplete = false
    private var previewing = false
    private var resumeListeningAfterPreview = false
    private var cancellables = Set<AnyCancellable>()
    private var isBound = false

    private static let voiceDefaultsKey = "voiceModeVoiceId"
    /// Which assistant message we're currently voicing — when it changes (e.g. a
    /// new message in an agent loop) we restart sentence chunking for it.
    private var speakingMessageId: AnyHashable?

    init(recognizer: any SpeechRecognizing,
         synthesizer: any SpeechSynthesizing,
         voices: [VoiceOption] = VoiceCatalog.systemVoices(),
         loadingCue: any LoadingCue = SystemLoadingCue(),
         followUpTimer: FollowUpTimer = RealFollowUpTimer(),
         chime: WakeChime = SystemWakeChime()) {
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.loadingCue = loadingCue
        self.followUpTimer = followUpTimer
        self.chime = chime
        // Default ON: the tray assistant is hands-free and listens continuously,
        // so without a wake word it would answer ambient conversation. Setting it
        // in the initializer doesn't trip `didSet`, so this costs no UserDefaults
        // write on launch.
        self.requireWakeWord = (UserDefaults.standard.object(forKey: Self.wakeWordDefaultsKey) as? Bool) ?? true
        wire()
        setUpVoices(voices)
    }

    /// Production wiring: the system synthesizer is wrapped in the voice-clone
    /// router — when a clone clip is set in Settings, answers are spoken in the
    /// cloned voice via the server's TTS; otherwise (or on a failed synthesis)
    /// the wrapped system voice speaks. The controller only sees the protocol.
    convenience init(server: ServerManager) {
        self.init(recognizer: makeSpeechRecognizer(),
                  synthesizer: ClonedVoiceSynthesizer(server: server))
    }

    /// Wire the controller to the app: use its `ChatTurnEngine` as the runner and
    /// subscribe to the active session so streamed answers are spoken with no
    /// window open. Idempotent — safe to call once from `AppState`.
    func bind(appState: AppState) {
        guard !isBound else { return }
        isBound = true
        runner = appState.chatEngine
        turnContext = { [weak appState] in
            guard let appState else { return (UUID(), nil) }
            let sid = appState.activeChatId ?? appState.newChatSession()
            let wd = appState.chatSessions.first { $0.id == sid }?.workingDirectory
            return (sid, wd)
        }

        // Feed the active session's trailing assistant message into the
        // synthesizer whenever the conversation changes OR generation flips.
        // Replaces the chat view's `onChange(voiceFeedKey)` / `onChange(isGenerating)`
        // and, crucially, works with no chat window open. `observeAssistant`
        // itself no-ops unless we're mid-turn (thinking/speaking), so this is a
        // cheap pass-through when voice is idle or a plain text turn is running.
        Publishers.CombineLatest(appState.$chatSessions, appState.chatEngine.$isGenerating)
            .sink { [weak self, weak appState] sessions, generating in
                guard let self, let appState else { return }
                guard let session = sessions.first(where: { $0.id == appState.activeChatId }),
                      let last = Self.messageToVoice(in: session.messages) else { return }
                self.observeAssistant(messageId: last.id, content: last.content, generating: generating)
            }
            .store(in: &cancellables)

        // Auto-shutdown: when the server leaves the running state (user hit
        // Stop in the tray, a model switch is restarting it, or it crashed),
        // close voice mode so the toggle can't sit "on" over a dead server.
        // `removeDuplicates` collapses the unrelated status churn (.starting →
        // .running, /props ticks) to just the running-flag flips, and the
        // method itself no-ops unless voice is active, so the immediate replay
        // of the current status on subscribe costs nothing.
        appState.server.$status
            .map { $0 == .running }
            .removeDuplicates()
            .sink { [weak self] running in self?.serverStatusChanged(serverRunning: running) }
            .store(in: &cancellables)
    }

    /// Build the picker list and choose the active voice: a persisted choice if it
    /// still exists, otherwise the highest-quality voice for the user's language
    /// (so we don't default to the robotic compact voice).
    private func setUpVoices(_ all: [VoiceOption]) {
        let prefix = Locale.current.language.languageCode?.identifier ?? "en"
        availableVoices = VoiceCatalog.options(from: all, preferredLanguagePrefix: prefix)
        let saved = UserDefaults.standard.string(forKey: Self.voiceDefaultsKey)
        let chosen = (saved != nil && availableVoices.contains { $0.id == saved }) ? saved
            : VoiceCatalog.defaultVoiceId(from: all, preferredLanguagePrefix: prefix)
        selectedVoiceId = chosen
        synthesizer.voiceIdentifier = chosen
    }

    /// Pick a voice: applies immediately, persists, and (when idle/listening)
    /// speaks a short sample so the change is audible.
    func selectVoice(_ id: String?) {
        selectedVoiceId = id
        synthesizer.voiceIdentifier = id
        UserDefaults.standard.set(id, forKey: Self.voiceDefaultsKey)
        if state == .listening || state == .idle { previewVoice() }
    }

    /// Speak a short sample in the current voice without disturbing the turn
    /// machine; pauses the mic so the sample isn't transcribed back in.
    func previewVoice() {
        guard state == .listening || state == .idle else { return }
        resumeListeningAfterPreview = (state == .listening) && !isMuted
        recognizer.stop()
        previewing = true
        synthesizer.enqueue("Hi, this is how I'll sound.")
    }

    private func wire() {
        recognizer.onSpeechStarted = { [weak self] in self?.handleSpeechStarted() }
        recognizer.onPartialTranscript = { [weak self] t in
            guard let self else { return }
            self.partialTranscript = t
            // Any partial text is proof on-device dictation is working — clear
            // the empty-speech streak and latch it so noise can't false-trigger.
            if !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                self.hasRecognizedSpeech = true
                self.unrecognizedSpeechStreak = 0
            }
        }
        recognizer.onFinalTranscript = { [weak self] t in self?.handleFinalTranscript(t) }
        recognizer.onUnrecognizedSpeech = { [weak self] in self?.handleUnrecognizedSpeech() }
        recognizer.onError = { [weak self] m in self?.send(.failed(m)) }
        synthesizer.onQueueDrained = { [weak self] in self?.handleQueueDrained() }
    }

    // MARK: Public lifecycle

    /// Authorize, open the mic and enter the listening state. Returns false if
    /// permission was denied or the mic couldn't start.
    func begin() async -> Bool {
        // Non-invasive pre-flight: check the three prerequisites (mic + speech
        // permission, on-device dictation model) and, if one is missing, surface
        // a precise notice in the panel instead of opening a dead mic. No modal.
        let snapshot = await recognizer.preflight()
        if let issue = VoicePreflight.firstIssue(snapshot) {
            setupIssue = issue
            send(.failed(VoicePreflight.shortMessage(for: issue)))
            return false
        }
        setupIssue = nil
        unrecognizedSpeechStreak = 0
        hasRecognizedSpeech = false
        send(.start)
        isActive = true
        disarmFollowUp()              // a fresh session always starts behind the wake word
        startListening()
        return state == .listening
    }

    /// The mic heard speech but recognition produced nothing. A one-off is just
    /// noise; a couple in a row means on-device dictation is off/unavailable — so
    /// we stop the dead session and show the actionable notice (with the same
    /// "Open Keyboard Settings" affordance as the pre-flight).
    private func handleUnrecognizedSpeech() {
        guard isActive else { return }
        // Recognition already produced text this session → dictation is provably
        // installed + on. This empty endpoint is just noise; never claim
        // "dictation unavailable" and never stop a working session over it.
        if hasRecognizedSpeech { return }
        unrecognizedSpeechStreak += 1
        guard unrecognizedSpeechStreak >= Self.unrecognizedSpeechLimit else { return }
        end()
        setupIssue = .dictationUnavailable(locale: Locale.current.identifier)
    }

    /// Close voice mode and release all audio resources.
    func end() {
        // Don't strand a suspended approval continuation — deny it so the agent
        // loop can unwind cleanly instead of hanging on an awaited continuation.
        if pendingApproval != nil { resolve(.deny) }
        send(.stop)
        recognizer.stop()
        synthesizer.shutdown()   // stop + release the clone TTS model, if loaded
        resetTurn()
        partialTranscript = ""
        level = 0
        isMuted = false
        isActive = false
        setupIssue = nil
        disarmFollowUp()
    }

    /// React to the server's run state. The hands-free assistant is useless
    /// without a running server to generate answers, so when the server leaves
    /// the running state — the user hit Stop, a model switch is restarting it,
    /// or it crashed — we close voice mode. Otherwise the tray toggle keeps
    /// showing "on" against a dead server with the mic still hot, listening into
    /// a void. No-op when voice isn't active or the server is up.
    func serverStatusChanged(serverRunning: Bool) {
        guard !serverRunning, isActive else { return }
        end()
    }

    /// Whether there's an in-flight answer to cut off — the assistant is
    /// generating or speaking (or the user is mid-utterance). Mirrors `bargeIn`'s
    /// guard so the tray "Stop" control is enabled exactly when pressing it does
    /// something.
    var canInterrupt: Bool {
        state == .speaking || state == .thinking || state == .recognizing
    }

    /// Cut the assistant off and listen again — the tray "Stop" control, an orb
    /// tap, or a detected talk-over. Unlike a TTS-only interrupt this also stops
    /// the underlying turn runner, so a long stream or an agent loop can't keep
    /// producing more text to speak after you've decided to move on. The chat
    /// session is left intact (use "New" to also clear context).
    func bargeIn() {
        guard canInterrupt else { return }
        // Flip to listening *first*: `runner?.stop()` synchronously publishes
        // `isGenerating = false`, which re-enters our Combine sink. With the
        // state already `.listening`, `observeAssistant` no-ops there instead of
        // flushing one last sentence into a synthesizer we're about to silence.
        send(.bargeIn)
        runner?.stop()
        synthesizer.stop()
        resetTurn()
        // Release the mic before reopening it. When barge-in fires from
        // `.recognizing` (the user made a noise that tripped VAD but never
        // finalized — e.g. just tapping Stop), the recognizer is *still live*:
        // `handleFinalTranscript`'s `recognizer.stop()` never ran. Calling
        // `startListening()` → `recognizer.start()` on the running AVAudioEngine
        // re-installs the input tap, which floods/wedges the main thread and
        // freezes the tray ("buttons dead, dropdown still works"). Stopping first
        // gives a clean stop→start cycle; it's a harmless no-op when the mic was
        // already off (barge-in from `.speaking`/`.thinking`).
        recognizer.stop()
        startListening()
    }

    /// Stop/resume capturing without leaving voice mode.
    func toggleMute() {
        isMuted.toggle()
        if isMuted { recognizer.stop() }
        else if state == .listening { startListening() }
    }

    // MARK: Turn submission + approval

    /// Submit the finalized user transcript to the shared engine. Always uses a
    /// `voiceStyle` config so spoken answers stay short and Markdown-free; agent
    /// Agent, thinking, and MCP all come from the voice-scoped toggles.
    private func submitTurn(_ text: String) {
        guard let runner, let ctx = turnContext?() else { return }
        let config = ChatTurnEngine.TurnConfig(
            agentMode: agentMode,
            mcpMode: mcpMode,
            enableThinking: enableThinking,
            voiceStyle: true,
            workingDirectory: ctx.workingDirectory
        )
        runner.runTurn(sessionId: ctx.sessionId, userText: text, images: nil, audio: nil,
                       config: config, approval: { [weak self] tc in
            await self?.approvalDecision(for: tc) ?? false
        })
    }

    /// The agent loop's approval gate. Auto-approves while `autoApproveTools` is
    /// on (hands-free); otherwise surfaces a `pendingApproval` and suspends until
    /// the panel/orb calls `resolve(_:)`.
    func approvalDecision(for tc: APIClient.ToolCall) async -> Bool {
        if autoApproveTools { return true }
        let choice: ToolApprovalChoice = await withCheckedContinuation { cont in
            pendingApproval = ToolApprovalRequest(
                toolName: tc.name,
                arguments: tc.arguments,
                rawArguments: tc.rawArguments,
                continuation: cont
            )
        }
        return choice == .allow
    }

    /// Resume a pending approval with the user's choice. `allowAll` flips
    /// `autoApproveTools` so the rest of the hands-free turn doesn't block again.
    func resolve(_ choice: ToolApprovalChoice, allowAll: Bool = false) {
        guard let req = pendingApproval else { return }
        if allowAll { autoApproveTools = true }
        req.continuation.resume(returning: choice)
        pendingApproval = nil
    }

    /// Fed via Combine whenever the active assistant message changes. `content`
    /// is the visible answer only — reasoning lives in a separate field upstream
    /// and never reaches here, so the thinking trace is never spoken. The turn
    /// ends when `generating` goes false (covers multi-message agent loops, not
    /// just a single message's streaming flag).
    func observeAssistant(messageId: AnyHashable?, content: String, generating: Bool) {
        guard state == .thinking || state == .speaking else { return }

        if messageId != speakingMessageId {
            // Switching messages (e.g. agent loop): speak the previous message's
            // buffered tail before chunking the new one from scratch.
            if speakingMessageId != nil {
                for sentence in chunker.flush() { synthesizer.enqueue(sentence) }
            }
            speakingMessageId = messageId
            chunker = SentenceChunker()
        }
        if !content.isEmpty, state == .thinking { send(.responseStarted) }

        let clean = SpeechSanitizer.spokenText(from: content)
        for sentence in chunker.ingest(fullText: clean) { synthesizer.enqueue(sentence) }

        if !generating {
            for sentence in chunker.flush() { synthesizer.enqueue(sentence) }
            streamComplete = true
            maybeFinishTurn()
        }
        level = recognizer.level
        updateLoadingTone()   // speaking now → silence the cue
    }

    // MARK: Recognizer events

    private func handleSpeechStarted() {
        // The user is responding inside the follow-up window — freeze the disarm
        // timer so it can't expire mid-sentence and drop the finalized utterance.
        if awaitingWakeQuery { followUpTimer.cancel() }
        if state == .listening { send(.speechStarted) }
    }

    private func handleFinalTranscript(_ text: String) {
        guard state == .listening || state == .recognizing else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        unrecognizedSpeechStreak = 0   // recognition is working — clear the streak
        hasRecognizedSpeech = true     // …and latch it: never nag "unavailable" again

        guard let query = wakeWordGate(trimmed) else {
            // Ambient speech we weren't addressed in, or a bare wake phrase that
            // only arms the next utterance → drop it and keep listening. The
            // recognizer already recycled itself for the next utterance.
            partialTranscript = ""
            send(.utteranceDismissed)
            return
        }

        disarmFollowUp()                  // entering a turn closes any open follow-up window
        recognizer.stop()                 // mic off while the model thinks/speaks
        partialTranscript = ""
        resetTurn()
        send(.transcriptFinalized)
        submitTurn(query)
    }

    /// Decide what (if anything) to submit for a finalized utterance under the
    /// wake-word policy. Returns the query to send, or nil when the utterance
    /// should be ignored — either no wake word was present, or it was a bare wake
    /// phrase that only arms the next utterance. Mutates `awaitingWakeQuery`.
    func wakeWordGate(_ utterance: String) -> String? {
        guard requireWakeWord else { return utterance }     // always-on mode
        if awaitingWakeQuery {                              // bare wake phrase last time
            awaitingWakeQuery = false
            return utterance
        }
        guard let remainder = WakeWord.strip(utterance, phrase: wakePhrase) else {
            return nil                                      // not addressed to the assistant
        }
        chime.play()                                        // heard "Hey Loki" → audible acknowledgement
        if remainder.isEmpty {                             // bare "Hey Loki" → open the window
            armFollowUp()
            return nil
        }
        return remainder
    }

    /// Whether a session message should be read aloud. We voice only genuine
    /// assistant answers — never the agent loop's tool-call / result summary
    /// cards (`isAgentSummary`), which would otherwise be spoken verbatim as
    /// "shell command date" / "shell, arrow, Thu Jun 4…".
    nonisolated static func isSpeakable(_ message: ChatMessage) -> Bool {
        message.role == .assistant && !message.isAgentSummary
    }

    /// The message (if any) the synthesizer should be tracking right now: only
    /// the *very last* message in the session, and only if it's speakable.
    ///
    /// Crucially we do NOT scan backward for the last speakable message — at the
    /// start of a new turn the last entry is the user's question (or a tool
    /// summary mid-loop), and reaching back would re-voice the *previous* turn's
    /// answer (the "it repeats the time before answering" bug). Returning nil
    /// there is correct: there's nothing new to speak until the assistant's
    /// reply lands as the last message.
    nonisolated static func messageToVoice(in messages: [ChatMessage]) -> ChatMessage? {
        guard let last = messages.last, isSpeakable(last) else { return nil }
        return last
    }

    // MARK: Follow-up window

    /// Open the follow-up window: accept the next utterance as a query with no
    /// wake word, but only for `followUpWindow` seconds of silence, after which
    /// the wake word is required again. No-op when the wake word is disabled
    /// (always-on already accepts everything).
    private func armFollowUp() {
        guard requireWakeWord else { return }
        awaitingWakeQuery = true
        followUpTimer.schedule(after: followUpWindow) { [weak self] in
            self?.followUpExpired()
        }
    }

    /// Close the follow-up window immediately and cancel its timer.
    private func disarmFollowUp() {
        awaitingWakeQuery = false
        followUpTimer.cancel()
    }

    /// The follow-up window elapsed with no speech → require the wake word again.
    /// Internal (not private) so the timer wiring is unit-testable.
    func followUpExpired() {
        awaitingWakeQuery = false
    }

    // MARK: Synthesizer events

    private func handleQueueDrained() {
        if previewing {
            previewing = false
            if resumeListeningAfterPreview { startListening() }
            updateLoadingTone()
            return
        }
        if streamComplete { finishTurn() }   // → listening (send updates the cue)
        else { updateLoadingTone() }         // tool-wait gap: still generating → cue resumes
    }

    private func maybeFinishTurn() {
        // Nothing queued (e.g. an empty answer) → finish now; otherwise the drain
        // callback closes the turn once the last utterance finishes.
        if streamComplete && !synthesizer.isSpeaking { finishTurn() }
    }

    private func finishTurn() {
        guard state == .speaking || state == .thinking else { return }
        resetTurn()
        send(.turnFinished)
        startListening()
        armFollowUp()                 // continued conversation: next utterance needs no wake word
    }

    // MARK: Presentation

    /// Listening-state prompt under the current wake-word policy. Each surface
    /// phrases the muted case its own way, so this covers only the unmuted
    /// states: "Go ahead…" once a bare wake phrase armed the next utterance,
    /// "Say "Hey Loki"…" while waiting for the wake word, plain "Listening…" when
    /// the wake word is off.
    var listeningPrompt: String {
        if awaitingWakeQuery { return "Go ahead…" }
        if requireWakeWord { return "Say “\(wakePhraseDisplay)”…" }
        return "Listening…"
    }

    /// The wake phrase title-cased for display ("hey loki" → "Hey Loki").
    var wakePhraseDisplay: String {
        wakePhrase.split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    // MARK: Helpers

    private func resetTurn() {
        streamComplete = false
        speakingMessageId = nil
        chunker = SentenceChunker()
    }

    private func startListening() {
        guard !isMuted else { return }
        do { try recognizer.start() }
        catch { send(.failed(error.localizedDescription)) }
    }

    private func send(_ event: VoiceTurnEvent) {
        state = VoiceTurnMachine.reduce(state, on: event)
        updateLoadingTone()
    }

    /// Play the loading cue while the model is busy but nothing is being spoken:
    /// generating the first tokens (thinking), or a mid-answer gap waiting on more
    /// content / a tool call. Silent while the synthesizer is actually speaking,
    /// while listening, and when idle.
    private func updateLoadingTone() {
        let busy = state == .thinking || (state == .speaking && !synthesizer.isSpeaking)
        if busy { loadingCue.start() } else { loadingCue.stop() }
    }
}
