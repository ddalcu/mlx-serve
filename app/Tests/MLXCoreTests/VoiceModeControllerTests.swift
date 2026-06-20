import XCTest
@testable import MLXCore

/// Drives the controller through a full turn with fake speech I/O: a finalized
/// transcript is submitted to a fake `TurnRunning`, the streamed answer is
/// sanitized + chunked + enqueued to TTS, and the mic reopens when the queue
/// drains. Also covers empty answers, barge-in, multi-message agent loops,
/// approval routing, and closing.
@MainActor
final class VoiceModeControllerTests: XCTestCase {

    // MARK: Fakes

    final class FakeRecognizer: SpeechRecognizing {
        var partialTranscript = ""
        var level: Float = 0
        var onSpeechStarted: (() -> Void)?
        var onPartialTranscript: ((String) -> Void)?
        var onFinalTranscript: ((String) -> Void)?
        var onUnrecognizedSpeech: (() -> Void)?
        var onError: ((String) -> Void)?
        var authorized = true
        /// When set, `preflight()` returns this verbatim so tests can exercise the
        /// dictation-model-missing path; otherwise it derives from `authorized`.
        var preflightSnapshot: VoicePreflight.Snapshot?
        private(set) var startCount = 0
        private(set) var stopCount = 0
        /// Ordered log of mic lifecycle calls, so tests can assert a *clean*
        /// stop→start restart (a `start` while already started double-installs the
        /// real `AVAudioEngine` tap and wedges the UI).
        private(set) var events: [String] = []
        func requestAuthorization() async -> Bool { authorized }
        func preflight() async -> VoicePreflight.Snapshot {
            preflightSnapshot ?? VoicePreflight.Snapshot(
                micAuthorized: authorized, speechAuthorized: authorized,
                onDeviceAvailable: true, locale: "en_US")
        }
        func start() throws { startCount += 1; events.append("start") }
        func stop() { stopCount += 1; events.append("stop") }
    }

    final class FakeSynth: SpeechSynthesizing {
        private(set) var enqueued: [String] = []
        private(set) var stopCount = 0
        private var pending = 0
        var voiceIdentifier: String?
        var isSpeaking: Bool { pending > 0 }
        var onQueueDrained: (() -> Void)?
        func enqueue(_ text: String) { enqueued.append(text); pending += 1 }
        func stop() { stopCount += 1; pending = 0 }
        func simulateDrain() { pending = 0; onQueueDrained?() }
    }

    final class FakeCue: LoadingCue {
        private(set) var active = false
        private(set) var startCount = 0
        func start() { if !active { startCount += 1 }; active = true }
        func stop() { active = false }
    }

    /// Deterministic follow-up timer — records the scheduled window and fires on
    /// command instead of waiting wall-clock.
    @MainActor
    final class FakeFollowUpTimer: FollowUpTimer {
        private(set) var scheduledSeconds: TimeInterval?
        private(set) var cancelCount = 0
        private var action: (() -> Void)?
        var isScheduled: Bool { action != nil }
        func schedule(after seconds: TimeInterval, _ action: @escaping () -> Void) {
            scheduledSeconds = seconds
            self.action = action
        }
        func cancel() { if action != nil { cancelCount += 1 }; action = nil }
        func fire() { let a = action; action = nil; a?() }
    }

    /// Records chime plays instead of emitting real audio.
    final class FakeWakeChime: WakeChime {
        private(set) var playCount = 0
        func play() { playCount += 1 }
    }

    /// Stands in for `ChatTurnEngine` — records every submitted turn and captures
    /// the approval closure so the suspend/resolve flow can be exercised directly.
    @MainActor
    final class FakeRunner: TurnRunning {
        struct Call {
            let sessionId: UUID
            let userText: String
            let config: ChatTurnEngine.TurnConfig
        }
        var isGenerating = false
        private(set) var calls: [Call] = []
        private(set) var lastApproval: ((APIClient.ToolCall) async -> Bool)?
        private(set) var stopCount = 0
        func runTurn(sessionId: UUID, userText: String, images: [ChatImage]?, audio: [ChatAudio]?,
                     config: ChatTurnEngine.TurnConfig,
                     approval: @escaping (APIClient.ToolCall) async -> Bool) {
            calls.append(Call(sessionId: sessionId, userText: userText, config: config))
            lastApproval = approval
        }
        func stop() { stopCount += 1 }
    }

    private static let testVoices = [
        VoiceOption(id: "compact", name: "Sam", language: "en-US", quality: 1),
        VoiceOption(id: "premium", name: "Ava", language: "en-US", quality: 3),
    ]

    private func makeWithCue() -> (VoiceModeController, FakeRecognizer, FakeSynth, FakeCue) {
        let rec = FakeRecognizer(); let syn = FakeSynth(); let cue = FakeCue()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn,
                                    voices: Self.testVoices, loadingCue: cue, chime: FakeWakeChime())
        c.requireWakeWord = false   // these tests exercise the turn pipeline, not the wake gate
        return (c, rec, syn, cue)
    }

    private func make() -> (VoiceModeController, FakeRecognizer, FakeSynth) {
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, chime: FakeWakeChime())
        c.requireWakeWord = false
        return (c, rec, syn)
    }

    /// A controller wired to a fake runner + a fixed session id, mimicking what
    /// `bind(appState:)` does in production.
    private func makeRunnable() -> (VoiceModeController, FakeRecognizer, FakeSynth, FakeRunner, UUID) {
        let rec = FakeRecognizer(); let syn = FakeSynth(); let runner = FakeRunner()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices, chime: FakeWakeChime())
        c.requireWakeWord = false
        let sid = UUID()
        c.runner = runner
        c.turnContext = { (sid, nil) }
        return (c, rec, syn, runner, sid)
    }

    /// A wake-word-on controller with an injected fake timer, for exercising the
    /// follow-up ("continued conversation") window deterministically.
    private func makeTimed() -> (VoiceModeController, FakeRecognizer, FakeSynth, FakeRunner, FakeFollowUpTimer) {
        let rec = FakeRecognizer(); let syn = FakeSynth(); let runner = FakeRunner(); let timer = FakeFollowUpTimer()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices,
                                    followUpTimer: timer, chime: FakeWakeChime())
        c.requireWakeWord = true
        c.runner = runner
        c.turnContext = { (UUID(), nil) }
        return (c, rec, syn, runner, timer)
    }

    /// A wake-word-on controller that exposes its fake chime, for asserting the
    /// audible "Hey Loki" acknowledgement.
    private func makeChimed() -> (VoiceModeController, FakeRecognizer, FakeRunner, FakeWakeChime) {
        let rec = FakeRecognizer(); let syn = FakeSynth(); let runner = FakeRunner(); let chime = FakeWakeChime()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices, chime: chime)
        c.requireWakeWord = true
        c.runner = runner
        c.turnContext = { (UUID(), nil) }
        return (c, rec, runner, chime)
    }

    private static func toolCall(_ name: String) -> APIClient.ToolCall {
        APIClient.ToolCall(id: "1", name: name, arguments: ["command": "ls"], rawArguments: "{\"command\":\"ls\"}")
    }

    /// begin() + simulate the user finishing an utterance, landing in `.thinking`.
    private func beginAndSubmit(_ c: VoiceModeController, _ rec: FakeRecognizer,
                                _ phrase: String = "hi") async {
        _ = await c.begin()
        rec.onFinalTranscript?(phrase)
    }

    // MARK: Tests

    func testBeginOpensListeningAndStartsMic() async {
        let (c, rec, _) = make()
        let ok = await c.begin()
        XCTAssertTrue(ok)
        XCTAssertEqual(c.state, .listening)
        XCTAssertEqual(rec.startCount, 1)
        XCTAssertNil(c.setupIssue, "a clean start clears any setup notice")
    }

    func testBeginSurfacesMicrophoneIssueWhenPermissionDenied() async {
        let (c, rec, _) = make()
        rec.authorized = false
        let ok = await c.begin()
        XCTAssertFalse(ok)
        XCTAssertEqual(c.setupIssue, .microphoneDenied, "denied permission → non-invasive notice, not a dead mic")
        XCTAssertEqual(rec.startCount, 0, "mic must NOT open when a prerequisite is missing")
    }

    func testBeginSurfacesDictationIssueWhenOnDeviceModelMissing() async {
        // Pre-flight catches a model that was never downloaded.
        let (c, rec, _) = make()
        rec.preflightSnapshot = VoicePreflight.Snapshot(
            micAuthorized: true, speechAuthorized: true, onDeviceAvailable: false, locale: "en_US")
        let ok = await c.begin()
        XCTAssertFalse(ok)
        XCTAssertEqual(c.setupIssue, .dictationUnavailable(locale: "en_US"))
        XCTAssertEqual(rec.startCount, 0)
    }

    /// The Dictation-switch-OFF case the user hit: pre-flight passes (model is
    /// installed → `supportsOnDeviceRecognition` true), the mic hears speech, but
    /// recognition returns nothing. Two empty-speech turns → surface the notice.
    func testRepeatedUnrecognizedSpeechSurfacesDictationNotice() async {
        let (c, rec, _) = make()
        _ = await c.begin()
        XCTAssertEqual(c.state, .listening)
        XCTAssertNil(c.setupIssue)

        rec.onUnrecognizedSpeech?()            // one stray empty turn → no nag yet
        XCTAssertNil(c.setupIssue)
        XCTAssertTrue(c.isActive)

        rec.onUnrecognizedSpeech?()            // second in a row → notice + stop
        if case .dictationUnavailable = c.setupIssue {} else {
            XCTFail("expected dictationUnavailable, got \(String(describing: c.setupIssue))")
        }
        XCTAssertFalse(c.isActive, "a dead recognizer should stop, not keep a hot mic")
    }

    /// A successful transcription clears the empty-speech streak so an earlier
    /// stray miss can't combine with a much-later one to false-trigger.
    func testSuccessfulTranscriptResetsUnrecognizedStreak() async {
        let (c, rec, _, _, _) = makeRunnable()
        _ = await c.begin()
        rec.onUnrecognizedSpeech?()            // 1
        rec.onFinalTranscript?("hello there")  // success → reset
        rec.onUnrecognizedSpeech?()            // 1 again, not 2
        XCTAssertNil(c.setupIssue)
    }

    /// The false-positive the user hit: dictation has been transcribing fine,
    /// then a couple of empty endpoints land back-to-back — a cough, a door, or
    /// the assistant's own TTS bleeding into the mic crossing the VAD threshold
    /// with no words. Once recognition has produced text this session, on-device
    /// dictation is provably installed + on, so "dictation unavailable" is wrong
    /// and must NEVER surface — and a benign noise burst must not kill the
    /// working session. (Old behavior: streak went 1→2 and false-tripped.)
    func testUnrecognizedSpeechNeverNagsAfterSuccessfulTranscript() async {
        let (c, rec, _, _, _) = makeRunnable()
        _ = await c.begin()
        rec.onFinalTranscript?("hello there")  // proves on-device dictation works
        rec.onUnrecognizedSpeech?()            // noise burst 1
        rec.onUnrecognizedSpeech?()            // noise burst 2 — old streak would trip here
        XCTAssertNil(c.setupIssue, "recognition already worked — never claim dictation is unavailable")
        XCTAssertTrue(c.isActive, "a benign noise burst must not stop a working session")
    }

    /// Even a partial transcript is proof dictation works — words were detected.
    /// Words mid-utterance followed by empty endpoints must not surface the
    /// notice either.
    func testPartialTranscriptAloneSuppressesDictationNotice() async {
        let (c, rec, _) = make()
        _ = await c.begin()
        rec.onPartialTranscript?("hey lo")     // words detected → dictation is working
        rec.onUnrecognizedSpeech?()            // 1
        rec.onUnrecognizedSpeech?()            // 2 — would have tripped the streak
        XCTAssertNil(c.setupIssue)
        XCTAssertTrue(c.isActive)
    }

    func testEndClearsSetupIssue() async {
        let (c, rec, _) = make()
        rec.authorized = false
        _ = await c.begin()
        XCTAssertNotNil(c.setupIssue)
        c.end()
        XCTAssertNil(c.setupIssue)
    }

    func testBeginFailsWhenPermissionDenied() async {
        let (c, rec, _) = make()
        rec.authorized = false
        let ok = await c.begin()
        XCTAssertFalse(ok)
        // The pre-flight now reports the precise missing prerequisite (here, mic)
        // rather than a vague combined string.
        XCTAssertEqual(c.state, .error(VoicePreflight.shortMessage(for: .microphoneDenied)))
    }

    func testFinalTranscriptSubmitsVoiceTurnAndEntersThinking() async {
        let (c, rec, _, runner, sid) = makeRunnable()
        _ = await c.begin()
        rec.onFinalTranscript?("  hello there  ")
        XCTAssertEqual(c.state, .thinking)
        XCTAssertGreaterThanOrEqual(rec.stopCount, 1)        // mic off while thinking
        // The turn was handed to the runner exactly once, with the trimmed text,
        // the active session id, and a voice-style config.
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertEqual(runner.calls.first?.userText, "hello there")
        XCTAssertEqual(runner.calls.first?.sessionId, sid)
        XCTAssertEqual(runner.calls.first?.config.voiceStyle, true)
    }

    func testVoiceTurnConfigUsesVoiceScopedToggles() async {
        let (c, rec, _, runner, _) = makeRunnable()
        c.agentMode = true
        c.enableThinking = true
        c.mcpMode = true   // voice-scoped now (seeded from the launching chat session)
        _ = await c.begin()
        rec.onFinalTranscript?("do a thing")
        let config = runner.calls.first?.config
        XCTAssertEqual(config?.agentMode, true)
        XCTAssertEqual(config?.enableThinking, true)
        XCTAssertEqual(config?.mcpMode, true, "voice MCP comes from the controller, not the app-global")
        XCTAssertEqual(config?.voiceStyle, true)
    }

    func testStreamingAnswerIsSanitizedChunkedAndSpoken() async {
        let (c, rec, syn) = make()
        await beginAndSubmit(c, rec)

        c.observeAssistant(messageId: "A", content: "**Hello** there. How are you", generating: true)
        XCTAssertEqual(c.state, .speaking)
        XCTAssertEqual(syn.enqueued, ["Hello there."])       // markdown stripped, first sentence spoken

        c.observeAssistant(messageId: "A", content: "**Hello** there. How are you?", generating: false)
        XCTAssertEqual(syn.enqueued, ["Hello there.", "How are you?"])  // remainder flushed
        XCTAssertEqual(c.state, .speaking)                   // still speaking until queue drains

        syn.simulateDrain()
        XCTAssertEqual(c.state, .listening)                  // mic reopened for next turn
    }

    func testReasoningIsNeverSpoken() async {
        // The controller only ever receives visible `content`; thinking text is a
        // separate field upstream and never routed here.
        let (c, rec, syn) = make()
        await beginAndSubmit(c, rec)
        c.observeAssistant(messageId: "A", content: "Answer is two. ", generating: false)
        XCTAssertEqual(syn.enqueued, ["Answer is two."])
    }

    func testEmptyAnswerReopensListeningWithoutSpeaking() async {
        let (c, rec, syn) = make()
        await beginAndSubmit(c, rec)
        c.observeAssistant(messageId: "A", content: "", generating: false)
        XCTAssertTrue(syn.enqueued.isEmpty)
        XCTAssertEqual(c.state, .listening)
    }

    func testBargeInStopsSpeakingAndListens() async {
        let (c, rec, syn) = make()
        await beginAndSubmit(c, rec)
        c.observeAssistant(messageId: "A", content: "One. Two. ", generating: true)
        XCTAssertEqual(c.state, .speaking)
        c.bargeIn()
        XCTAssertEqual(syn.stopCount, 1)
        XCTAssertEqual(c.state, .listening)
    }

    func testBargeInWhileRecognizingReleasesMicBeforeReopening() async {
        // Repro for the tray freeze: toggle voice on → make a noise (VAD trips
        // `.speechStarted`) → state `.recognizing` with the mic STILL live (no
        // final transcript, so nothing stopped it) → click Stop. Barge-in must
        // release the mic *before* reopening it; restarting a live recognizer
        // re-installs the `AVAudioEngine` input tap and wedges the main thread
        // (the "tray buttons dead, dropdown still works" freeze). No LLM involved.
        let (c, rec, syn) = make()
        _ = await c.begin()                 // → .listening, mic started
        rec.onSpeechStarted?()              // a noise trips VAD
        XCTAssertEqual(c.state, .recognizing)
        XCTAssertTrue(c.canInterrupt)       // Stop is enabled

        let startsBefore = rec.startCount
        let stopsBefore = rec.stopCount
        c.bargeIn()                         // user clicks Stop

        XCTAssertEqual(c.state, .listening)
        XCTAssertEqual(syn.stopCount, 1)
        // The live mic must be torn down and reopened — a clean stop→start cycle,
        // never a bare second start on the running engine.
        XCTAssertEqual(rec.stopCount, stopsBefore + 1, "mic must be released on barge-in")
        XCTAssertEqual(rec.startCount, startsBefore + 1, "mic must be reopened to keep listening")
        XCTAssertEqual(Array(rec.events.suffix(2)), ["stop", "start"],
                       "restart must stop the live mic first, then start")
    }

    func testStopWhileSpeakingHaltsGenerationAndSpeechThenListens() async {
        // The tray "Stop" control: the model is reading a long answer aloud and
        // the user wants to move on. Cutting it off must halt the underlying
        // generation too (not just the TTS) so a long stream / agent loop can't
        // keep feeding new sentences, and the mic must reopen.
        let (c, rec, syn, runner, _) = makeRunnable()
        _ = await c.begin()
        rec.onFinalTranscript?("tell me a long story")
        c.observeAssistant(messageId: "A", content: "Once upon a time. ", generating: true)
        XCTAssertEqual(c.state, .speaking)

        c.bargeIn()

        XCTAssertEqual(runner.stopCount, 1)     // generation halted
        XCTAssertEqual(syn.stopCount, 1)        // speech silenced
        XCTAssertEqual(c.state, .listening)     // mic reopened
    }

    func testStopWhileThinkingHaltsGenerationAndListens() async {
        // Same control, but pressed before any audio — the model is still
        // generating (state `.thinking`). Must stop generation and return to
        // listening rather than hang on "Thinking…".
        let (c, rec, _, runner, _) = makeRunnable()
        _ = await c.begin()
        rec.onFinalTranscript?("think very hard about this")
        XCTAssertEqual(c.state, .thinking)

        c.bargeIn()

        XCTAssertEqual(runner.stopCount, 1)
        XCTAssertEqual(c.state, .listening)
    }

    func testCanInterruptReflectsInFlightAnswer() async {
        let (c, rec, _, _, _) = makeRunnable()
        _ = await c.begin()
        XCTAssertFalse(c.canInterrupt)          // listening — nothing to stop
        rec.onFinalTranscript?("hi")
        XCTAssertTrue(c.canInterrupt)           // thinking — a turn to cut off
        c.bargeIn()
        XCTAssertFalse(c.canInterrupt)          // back to listening
    }

    func testAgentLoopMultiMessageDoesNotFinishEarly() async {
        // An agent turn appends several assistant messages; an intermediate
        // message finishing must NOT end the turn — only `generating` going false
        // does. Each new message is chunked fresh.
        let (c, rec, syn) = make()
        await beginAndSubmit(c, rec, "list my files")

        c.observeAssistant(messageId: "tool1", content: "", generating: true)        // tool-call placeholder
        XCTAssertEqual(c.state, .thinking)                                           // nothing spoken yet
        c.observeAssistant(messageId: "tool1", content: "Checking. ", generating: true)
        XCTAssertEqual(syn.enqueued, [])                                             // last sentence buffered (may grow)
        XCTAssertEqual(c.state, .speaking)

        // New message → previous tail is flushed, new message chunked fresh.
        c.observeAssistant(messageId: "final", content: "You have three files. ", generating: true)
        XCTAssertEqual(syn.enqueued, ["Checking."])
        XCTAssertEqual(c.state, .speaking)                                           // still going, not finished

        c.observeAssistant(messageId: "final", content: "You have three files. ", generating: false)
        XCTAssertEqual(syn.enqueued, ["Checking.", "You have three files."])         // flushed, no double-speak
        XCTAssertEqual(c.state, .speaking)                                           // waiting on drain
        syn.simulateDrain()
        XCTAssertEqual(c.state, .listening)
    }

    func testEndReturnsToIdleAndStopsEverything() async {
        let (c, rec, syn) = make()
        _ = await c.begin()
        c.end()
        XCTAssertEqual(c.state, .idle)
        XCTAssertGreaterThanOrEqual(rec.stopCount, 1)
        XCTAssertGreaterThanOrEqual(syn.stopCount, 1)
    }

    // MARK: Wake word

    func testWakeWordDefaultsOn() {
        // A fresh controller with no stored preference requires the wake word.
        UserDefaults.standard.removeObject(forKey: "voiceModeRequireWakeWord")
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices)
        XCTAssertTrue(c.requireWakeWord)
    }

    func testWakeWordIgnoresUtteranceWithoutPhrase() async {
        let (c, rec, _, runner, _) = makeRunnable()
        c.requireWakeWord = true
        _ = await c.begin()
        let stopsAfterBegin = rec.stopCount
        rec.onFinalTranscript?("what's the weather today")
        XCTAssertTrue(runner.calls.isEmpty)                   // not addressed to the assistant
        XCTAssertEqual(c.state, .listening)                  // kept listening
        XCTAssertEqual(rec.stopCount, stopsAfterBegin)       // mic was never cut
    }

    func testWakeWordPrefixSubmitsStrippedQuery() async {
        let (c, rec, _, runner, sid) = makeRunnable()
        c.requireWakeWord = true
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki, what's the weather?")
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertEqual(runner.calls.first?.userText, "what's the weather?")   // wake word stripped
        XCTAssertEqual(runner.calls.first?.sessionId, sid)
        XCTAssertEqual(c.state, .thinking)
    }

    func testBareWakePhraseArmsNextUtterance() async {
        let (c, rec, _, runner, _) = makeRunnable()
        c.requireWakeWord = true
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                   // just the wake word, no command
        XCTAssertTrue(runner.calls.isEmpty)                  // nothing submitted yet
        XCTAssertTrue(c.awaitingWakeQuery)
        XCTAssertEqual(c.state, .listening)
        rec.onFinalTranscript?("what time is it")            // no wake word needed now
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertEqual(runner.calls.first?.userText, "what time is it")
        XCTAssertFalse(c.awaitingWakeQuery)
        XCTAssertEqual(c.state, .thinking)
    }

    func testWakeWordOffSubmitsEverything() async {
        let (c, rec, _, runner, _) = makeRunnable()
        c.requireWakeWord = false
        _ = await c.begin()
        rec.onFinalTranscript?("just do it")
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertEqual(runner.calls.first?.userText, "just do it")
    }

    func testTurningOffWakeWordClearsArming() async {
        let (c, rec, _, _, _) = makeRunnable()
        c.requireWakeWord = true
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")
        XCTAssertTrue(c.awaitingWakeQuery)
        c.requireWakeWord = false                            // toggled off mid-session
        XCTAssertFalse(c.awaitingWakeQuery)                  // no stale arm left behind
    }

    // MARK: Wake chime

    func testWakeWordPlaysChime() async {
        let (c, rec, _, chime) = makeChimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki, what's the weather?")   // inline wake word
        XCTAssertEqual(chime.playCount, 1)
    }

    func testBareWakeWordPlaysChime() async {
        let (c, rec, _, chime) = makeChimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                        // bare wake word
        XCTAssertEqual(chime.playCount, 1)
    }

    func testNonWakeUtteranceDoesNotChime() async {
        let (c, rec, _, chime) = makeChimed()
        _ = await c.begin()
        rec.onFinalTranscript?("what's the weather today")        // not addressed → no chime
        XCTAssertEqual(chime.playCount, 0)
    }

    func testFollowUpUtteranceDoesNotChime() async {
        // Speaking inside the follow-up window needs no wake word, so it must not
        // re-chime — only an actual "Hey Loki" does.
        let (c, rec, _, chime) = makeChimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                        // chime #1
        XCTAssertEqual(chime.playCount, 1)
        rec.onFinalTranscript?("what time is it")                 // follow-up, no wake word
        XCTAssertEqual(chime.playCount, 1)                        // still 1
    }

    // MARK: Spoken-message filtering

    func testIsSpeakableSkipsAgentSummaries() {
        let answer = ChatMessage(role: .assistant, content: "It's sunny.")
        XCTAssertTrue(VoiceModeController.isSpeakable(answer))

        var summary = ChatMessage(role: .assistant, content: "**shell**(command: date)")
        summary.isAgentSummary = true
        XCTAssertFalse(VoiceModeController.isSpeakable(summary))   // never read tool cards aloud

        let userMsg = ChatMessage(role: .user, content: "hi")
        XCTAssertFalse(VoiceModeController.isSpeakable(userMsg))
        let toolMsg = ChatMessage(role: .system, content: "Thu Jun 4 …")
        XCTAssertFalse(VoiceModeController.isSpeakable(toolMsg))
    }

    func testMessageToVoiceDoesNotRevoicePreviousAnswer() {
        // New turn just started: previous answer is followed by the new user
        // question. The synthesizer must find NOTHING to speak — otherwise it
        // re-reads the prior answer ("it repeats the time before answering").
        let prev = ChatMessage(role: .assistant, content: "It is 3:20 PM.")
        let newUser = ChatMessage(role: .user, content: "what's the weather")
        XCTAssertNil(VoiceModeController.messageToVoice(in: [prev, newUser]))
    }

    func testMessageToVoiceSkipsTrailingToolSummary() {
        // Mid agent loop the last entry is a tool-call/result card — skip it so
        // it isn't read aloud, and don't reach back to an earlier answer either.
        let answer = ChatMessage(role: .assistant, content: "Working on it.")
        var summary = ChatMessage(role: .assistant, content: "**shell** → Thu Jun 4 …")
        summary.isAgentSummary = true
        XCTAssertNil(VoiceModeController.messageToVoice(in: [answer, summary]))
    }

    func testMessageToVoiceReturnsTrailingAnswer() {
        let answer = ChatMessage(role: .assistant, content: "It's sunny.")
        let msgs = [ChatMessage(role: .user, content: "weather?"), answer]
        XCTAssertEqual(VoiceModeController.messageToVoice(in: msgs)?.content, "It's sunny.")
    }

    // MARK: Follow-up window (continued conversation)

    func testCompletedTurnOpensFollowUpWindow() async {
        let (c, rec, syn, runner, timer) = makeTimed()
        _ = await c.begin()
        // First turn still needs the wake word.
        rec.onFinalTranscript?("Hey Loki, what's the weather?")
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertFalse(c.awaitingWakeQuery)                  // mid-turn, window closed
        // Assistant answers and the TTS queue drains → turn complete.
        c.observeAssistant(messageId: "A", content: "Sunny today.", generating: false)
        syn.simulateDrain()
        XCTAssertEqual(c.state, .listening)
        XCTAssertTrue(c.awaitingWakeQuery)                   // follow-up window now open
        XCTAssertTrue(timer.isScheduled)
        XCTAssertEqual(timer.scheduledSeconds, c.followUpWindow)
        // A follow-up needs no wake word.
        rec.onFinalTranscript?("and tomorrow?")
        XCTAssertEqual(runner.calls.count, 2)
        XCTAssertEqual(runner.calls.last?.userText, "and tomorrow?")
    }

    func testFollowUpWindowExpiryRequiresWakeWordAgain() async {
        let (c, rec, _, runner, timer) = makeTimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                   // bare phrase opens the window
        XCTAssertTrue(c.awaitingWakeQuery)
        XCTAssertTrue(timer.isScheduled)
        timer.fire()                                         // window elapses in silence
        XCTAssertFalse(c.awaitingWakeQuery)
        rec.onFinalTranscript?("what's the weather")         // no wake word → ignored again
        XCTAssertTrue(runner.calls.isEmpty)
        XCTAssertEqual(c.state, .listening)
    }

    func testSpeakingInsideWindowFreezesExpiryTimer() async {
        let (c, rec, _, runner, timer) = makeTimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                   // window open, timer armed
        XCTAssertTrue(timer.isScheduled)
        rec.onSpeechStarted?()                               // user begins the follow-up
        XCTAssertFalse(timer.isScheduled)                    // timer frozen so it can't expire mid-sentence
        XCTAssertTrue(c.awaitingWakeQuery)                   // still armed → utterance accepted
        rec.onFinalTranscript?("set a timer for five minutes")
        XCTAssertEqual(runner.calls.count, 1)
        XCTAssertEqual(runner.calls.first?.userText, "set a timer for five minutes")
    }

    func testSubmittingAQueryCancelsTheWindowTimer() async {
        let (c, rec, _, _, timer) = makeTimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")                   // arm
        XCTAssertTrue(timer.isScheduled)
        rec.onFinalTranscript?("do the thing")               // consumed as the query
        XCTAssertFalse(timer.isScheduled)                    // entering the turn closed the window
    }

    func testEndCancelsFollowUpWindow() async {
        let (c, rec, _, _, timer) = makeTimed()
        _ = await c.begin()
        rec.onFinalTranscript?("Hey Loki")
        XCTAssertTrue(timer.isScheduled)
        c.end()
        XCTAssertFalse(timer.isScheduled)
        XCTAssertFalse(c.awaitingWakeQuery)
    }

    // MARK: Server lifecycle

    func testServerStoppingEndsActiveVoice() async {
        // The tray toggle binds to `isActive`; stopping the server must flip it
        // off (and release the mic/synth) instead of leaving voice hot.
        let (c, rec, syn) = make()
        _ = await c.begin()
        XCTAssertTrue(c.isActive)
        c.serverStatusChanged(serverRunning: false)           // user hit Stop
        XCTAssertFalse(c.isActive)                            // toggle flips off
        XCTAssertEqual(c.state, .idle)
        XCTAssertGreaterThanOrEqual(rec.stopCount, 1)         // mic released
        XCTAssertGreaterThanOrEqual(syn.stopCount, 1)         // speech stopped
    }

    func testServerStoppingIsNoOpWhenVoiceInactive() {
        // Server stopping while voice was never started must not spin anything up.
        let (c, rec, _) = make()
        XCTAssertFalse(c.isActive)
        c.serverStatusChanged(serverRunning: false)
        XCTAssertFalse(c.isActive)
        XCTAssertEqual(c.state, .idle)
        XCTAssertEqual(rec.stopCount, 0)                      // mic never touched
    }

    func testServerRunningLeavesActiveVoiceUntouched() async {
        // A health tick reporting "running" must not interrupt an active session.
        let (c, _, _) = make()
        _ = await c.begin()
        c.serverStatusChanged(serverRunning: true)
        XCTAssertTrue(c.isActive)
        XCTAssertEqual(c.state, .listening)
    }

    func testUpdatesIgnoredWhenNotInAResponse() async {
        let (c, _, syn) = make()
        _ = await c.begin()
        c.observeAssistant(messageId: "A", content: "ghost. ", generating: true)
        XCTAssertTrue(syn.enqueued.isEmpty)
        XCTAssertEqual(c.state, .listening)
    }

    func testMuteStopsMicAndUnmuteRestarts() async {
        let (c, rec, _) = make()
        _ = await c.begin()
        let startsAfterBegin = rec.startCount
        c.toggleMute()
        XCTAssertTrue(c.isMuted)
        XCTAssertGreaterThanOrEqual(rec.stopCount, 1)
        c.toggleMute()
        XCTAssertFalse(c.isMuted)
        XCTAssertEqual(rec.startCount, startsAfterBegin + 1)  // restarted on unmute
    }

    // MARK: Voice selection

    func testDefaultsToHighestQualityVoice() {
        UserDefaults.standard.removeObject(forKey: "voiceModeVoiceId")
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices)
        XCTAssertEqual(c.selectedVoiceId, "premium")          // not the robotic compact voice
        XCTAssertEqual(syn.voiceIdentifier, "premium")
        XCTAssertEqual(c.availableVoices.first?.id, "premium")
    }

    func testSelectVoiceAppliesAndPersists() {
        UserDefaults.standard.removeObject(forKey: "voiceModeVoiceId")
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices)
        c.selectVoice("compact")
        XCTAssertEqual(c.selectedVoiceId, "compact")
        XCTAssertEqual(syn.voiceIdentifier, "compact")
        XCTAssertEqual(UserDefaults.standard.string(forKey: "voiceModeVoiceId"), "compact")
    }

    func testPreviewPausesMicAndResumesOnDrain() async {
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices)
        _ = await c.begin()                                   // listening
        let startsBefore = rec.startCount
        let stopsBefore = rec.stopCount
        c.previewVoice()
        XCTAssertEqual(rec.stopCount, stopsBefore + 1)        // mic paused for the sample
        XCTAssertFalse(syn.enqueued.isEmpty)
        syn.simulateDrain()
        XCTAssertEqual(rec.startCount, startsBefore + 1)      // mic resumed
        XCTAssertEqual(c.state, .listening)
    }

    // MARK: Loading cue

    func testLoadingCuePlaysWhileThinking() async {
        let (c, rec, _, cue) = makeWithCue()
        _ = await c.begin()
        XCTAssertFalse(cue.active)              // listening → quiet
        rec.onFinalTranscript?("hi")
        XCTAssertEqual(c.state, .thinking)
        XCTAssertTrue(cue.active)               // model is working, nothing spoken yet
    }

    func testLoadingCueStopsWhileSpeaking() async {
        let (c, rec, syn, cue) = makeWithCue()
        _ = await c.begin()
        rec.onFinalTranscript?("hi")
        c.observeAssistant(messageId: "A", content: "Here we go. More. ", generating: true)
        XCTAssertEqual(c.state, .speaking)
        XCTAssertTrue(syn.isSpeaking)
        XCTAssertFalse(cue.active)              // TTS is talking → no cue
    }

    func testLoadingCueResumesInToolWaitGap() async {
        let (c, rec, syn, cue) = makeWithCue()
        _ = await c.begin()
        rec.onFinalTranscript?("do a thing")
        c.observeAssistant(messageId: "A", content: "On it. Next. ", generating: true)
        XCTAssertFalse(cue.active)
        syn.simulateDrain()                     // queue empties but generation continues
        XCTAssertEqual(c.state, .speaking)
        XCTAssertTrue(cue.active)               // waiting on more / a tool call → cue resumes
    }

    func testLoadingCueOffAfterTurnAndOnEnd() async {
        let (c, rec, syn, cue) = makeWithCue()
        _ = await c.begin()
        rec.onFinalTranscript?("hi")
        c.observeAssistant(messageId: "A", content: "Done. ", generating: false)
        syn.simulateDrain()
        XCTAssertEqual(c.state, .listening)
        XCTAssertFalse(cue.active)
        c.end()
        XCTAssertFalse(cue.active)
    }

    func testSelectingVoiceWhileSpeakingDoesNotPreview() async {
        let rec = FakeRecognizer(); let syn = FakeSynth()
        let c = VoiceModeController(recognizer: rec, synthesizer: syn, voices: Self.testVoices)
        c.requireWakeWord = false
        _ = await c.begin()
        rec.onFinalTranscript?("hi")
        c.observeAssistant(messageId: "A", content: "Hello there. More text. ", generating: true)
        XCTAssertEqual(c.state, .speaking)
        let enqueuedBefore = syn.enqueued.count
        c.selectVoice("compact")
        XCTAssertEqual(syn.voiceIdentifier, "compact")
        XCTAssertEqual(syn.enqueued.count, enqueuedBefore)    // no preview interrupting the answer
    }

    // MARK: Tool approval routing

    func testAutoApproveReturnsTrueWithoutPrompting() async {
        let (c, _, _, _, _) = makeRunnable()
        c.autoApproveTools = true
        let allowed = await c.approvalDecision(for: Self.toolCall("shell"))
        XCTAssertTrue(allowed)
        XCTAssertNil(c.pendingApproval)                       // hands-free: no card shown
    }

    func testManualApprovalSuspendsThenAllows() async {
        let (c, _, _, _, _) = makeRunnable()
        c.autoApproveTools = false
        let decision = Task { await c.approvalDecision(for: Self.toolCall("writeFile")) }
        try? await Task.sleep(nanoseconds: 50_000_000)        // let the continuation register
        XCTAssertEqual(c.pendingApproval?.toolName, "writeFile")
        c.resolve(.allow)
        let allowed = await decision.value
        XCTAssertTrue(allowed)
        XCTAssertNil(c.pendingApproval)
    }

    func testManualApprovalDenyReturnsFalse() async {
        let (c, _, _, _, _) = makeRunnable()
        c.autoApproveTools = false
        let decision = Task { await c.approvalDecision(for: Self.toolCall("shell")) }
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertNotNil(c.pendingApproval)
        c.resolve(.deny)
        let allowed = await decision.value
        XCTAssertFalse(allowed)
        XCTAssertNil(c.pendingApproval)
    }

    func testAllowAllFlipsAutoApprove() async {
        let (c, _, _, _, _) = makeRunnable()
        c.autoApproveTools = false
        let decision = Task { await c.approvalDecision(for: Self.toolCall("shell")) }
        try? await Task.sleep(nanoseconds: 50_000_000)
        c.resolve(.allow, allowAll: true)
        let allowed = await decision.value
        XCTAssertTrue(allowed)
        XCTAssertTrue(c.autoApproveTools)                     // subsequent calls won't prompt
        // A follow-up call now auto-approves without a card.
        let next = await c.approvalDecision(for: Self.toolCall("editFile"))
        XCTAssertTrue(next)
        XCTAssertNil(c.pendingApproval)
    }

    func testEndDeniesPendingApproval() async {
        let (c, rec, _, _, _) = makeRunnable()
        c.autoApproveTools = false
        _ = await c.begin()
        rec.onFinalTranscript?("do a thing")
        let decision = Task { await c.approvalDecision(for: Self.toolCall("shell")) }
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertNotNil(c.pendingApproval)
        c.end()                                               // must not strand the continuation
        let allowed = await decision.value
        XCTAssertFalse(allowed)
        XCTAssertNil(c.pendingApproval)
        XCTAssertEqual(c.state, .idle)
    }
}
