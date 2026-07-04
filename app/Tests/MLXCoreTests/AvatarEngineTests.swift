import XCTest
import Combine
@testable import MLXCore

/// P4 avatar-loop unit tests. Cover the three pure pieces the plan calls out:
/// the persona model (Codable round-trip + defaults), the sentence→speech queue
/// (streamed deltas → the exact TTS-request texts, in order), and the
/// idle→thinking→speaking→idle state machine. NO audio / SceneKit here — the LLM
/// stream, the synthesizer, and the player are all injected as fakes.
@MainActor
final class AvatarEngineTests: XCTestCase {

    // MARK: - Fakes

    /// Records every TTS request text (in call order), every playback, and the
    /// system prompt the responder was called with (for the RAG tests).
    private final class Recorder {
        var synthTexts: [String] = []
        var playCount = 0
        var synthReturns: Data? = Data([1])
        var lastSystem: String?
    }

    private func makeEngine(minChars: Int = 0, deltas: [String],
                            retrieve: AvatarEngine.Retriever? = nil) -> (AvatarEngine, Recorder) {
        let rec = Recorder()
        let engine = AvatarEngine(
            persona: AvatarPersona(),
            minChars: minChars,
            respond: { system, _, _ in
                rec.lastSystem = system
                return AsyncThrowingStream { continuation in
                    Task {
                        for d in deltas { continuation.yield(d) }
                        continuation.finish()
                    }
                }
            },
            synthesize: { text, _ in rec.synthTexts.append(text); return rec.synthReturns },
            play: { _ in rec.playCount += 1 },
            retrieve: retrieve
        )
        return (engine, rec)
    }

    // MARK: - Persona model

    func testPersonaCodableRoundTrip() throws {
        var p = AvatarPersona()
        p.name = "Nova"
        p.systemPrompt = "Be brief and kind."
        p.voiceClipPath = "/tmp/voice.wav"
        p.glbPath = "/tmp/model.glb"
        p.docFolderPath = "/tmp/knowledge"
        let data = try JSONEncoder().encode(p)
        let decoded = try JSONDecoder().decode(AvatarPersona.self, from: data)
        XCTAssertEqual(decoded, p)
        XCTAssertEqual(decoded.docFolderPath, "/tmp/knowledge")
    }

    func testPersonaDecodesOlderBlobWithoutDocFolder() throws {
        // A persona persisted before docFolderPath shipped must still decode.
        let legacy = #"{"id":"\#(UUID().uuidString)","name":"Old","systemPrompt":"hi"}"#
        let decoded = try JSONDecoder().decode(AvatarPersona.self, from: Data(legacy.utf8))
        XCTAssertNil(decoded.docFolderPath)
        XCTAssertEqual(decoded.name, "Old")
    }

    func testPersonaDefaultsAreUsable() {
        let p = AvatarPersona()
        XCTAssertFalse(p.name.isEmpty)
        XCTAssertFalse(p.systemPrompt.isEmpty)
        XCTAssertNil(p.voiceClipPath)
        XCTAssertNil(p.glbPath)
    }

    func testPersonaStoreRoundTripAndSelection() throws {
        var store = AvatarPersonaStore()      // seeds one default persona
        let a = store.selectedPersona
        var b = AvatarPersona(); b.name = "B"
        store = store.upserting(b)
        XCTAssertEqual(store.selectedPersona.id, b.id, "upsert selects the new persona")
        XCTAssertEqual(store.personas.count, 2)

        let data = try JSONEncoder().encode(store)
        let decoded = try JSONDecoder().decode(AvatarPersonaStore.self, from: data)
        XCTAssertEqual(decoded, store)

        // Removing the selected persona re-points the selection.
        let afterRemove = decoded.removing(b.id)
        XCTAssertEqual(afterRemove.selectedPersona.id, a.id)
    }

    func testPersonaStoreNeverEmptyOrUnselected() {
        var s = AvatarPersonaStore()
        s.personas = []
        s.selectedId = nil
        let n = s.normalized()
        XCTAssertEqual(n.personas.count, 1)
        XCTAssertNotNil(n.selectedId)
        XCTAssertEqual(n.selectedPersona.id, n.personas.first?.id)
    }

    // MARK: - Sentence → speech queue

    func testStreamedDeltasProduceOrderedTTSRequests() async {
        // The classic SentenceStreamer trace, driven end-to-end through the engine.
        let (engine, rec) = makeEngine(
            minChars: 0,
            deltas: ["Hello wor", "ld. How are", " you? Fine.", ""])
        engine.submit("hi")
        await engine.turnTask?.value
        XCTAssertEqual(rec.synthTexts, ["Hello world.", "How are you?", "Fine."])
        XCTAssertEqual(rec.playCount, 3, "one clip played per synthesized sentence")
    }

    func testCoalescesShortFragmentsWithMinChars() async {
        // With a high minChars, tiny fragments merge into one spoken clip instead
        // of machine-gunning three.
        let (engine, rec) = makeEngine(minChars: 100, deltas: ["Hi. ", "Ok. ", "Bye. "])
        engine.submit("x")
        await engine.turnTask?.value
        XCTAssertEqual(rec.synthTexts, ["Hi. Ok. Bye."])
    }

    func testSilentSynthesisSkipsPlaybackButStillFinishes() async {
        let (engine, rec) = makeEngine(minChars: 0, deltas: ["One. ", "Two. "])
        rec.synthReturns = nil                 // synthesizer yields no audio
        engine.submit("hi")
        await engine.turnTask?.value
        XCTAssertEqual(rec.synthTexts, ["One.", "Two."])
        XCTAssertEqual(rec.playCount, 0)
        XCTAssertEqual(engine.state, .idle)    // still returns to idle cleanly
    }

    // MARK: - State machine

    func testStateMachineIdleThinkingSpeakingIdle() async {
        let (engine, _) = makeEngine(minChars: 0, deltas: ["Hello there. ", "Bye now. "])
        var states: [AvatarEngine.State] = []
        let cancellable = engine.$state.sink { states.append($0) }

        engine.submit("hi")
        XCTAssertEqual(engine.state, .thinking, "submit transitions to thinking synchronously")
        await engine.turnTask?.value
        cancellable.cancel()

        XCTAssertEqual(engine.state, .idle)
        XCTAssertEqual(states, [.idle, .thinking, .speaking, .idle],
                       "full lifecycle: idle → thinking → speaking → idle")
    }

    func testEmptyReplyEndsIdleWithoutSpeaking() async {
        let (engine, rec) = makeEngine(minChars: 0, deltas: [])
        engine.submit("hi")
        await engine.turnTask?.value
        XCTAssertTrue(rec.synthTexts.isEmpty)
        XCTAssertEqual(rec.playCount, 0)
        XCTAssertEqual(engine.state, .idle)
        XCTAssertEqual(engine.transcript.map(\.role), [.user],
                       "an empty answer is not recorded as an assistant turn")
    }

    func testTranscriptAccumulatesUserThenAssistant() async {
        let (engine, _) = makeEngine(minChars: 0, deltas: ["Hi there. "])
        engine.submit("hello")
        await engine.turnTask?.value
        XCTAssertEqual(engine.transcript.count, 2)
        XCTAssertEqual(engine.transcript[0].role, .user)
        XCTAssertEqual(engine.transcript[0].text, "hello")
        XCTAssertEqual(engine.transcript[1].role, .assistant)
        XCTAssertEqual(engine.transcript[1].text, "Hi there. ")
    }

    // MARK: - Persona RAG (retriever seam)

    func testRetrievedContextIsInjectedIntoSystemPrompt() async {
        var askedQuery: String?
        let (engine, rec) = makeEngine(minChars: 0, deltas: ["Answer. "], retrieve: { query in
            askedQuery = query
            return "[1] specs.md\nThe Nimbus 3000 kettle boils in ninety seconds."
        })
        engine.submit("how fast does the kettle boil")
        await engine.turnTask?.value
        let system = rec.lastSystem ?? ""
        XCTAssertEqual(askedQuery, "how fast does the kettle boil",
                       "the retriever is asked for the user's utterance")
        XCTAssertTrue(system.contains("Reference material:"),
                      "retrieved excerpts are injected under a Reference material block")
        XCTAssertTrue(system.contains("ninety seconds"),
                      "the retrieved text reaches the system prompt")
        XCTAssertTrue(system.contains(engine.persona.systemPrompt),
                      "the persona prompt is preserved alongside the retrieved context")
    }

    func testNoRetrieverLeavesSystemPromptUnchanged() async {
        let (engine, rec) = makeEngine(minChars: 0, deltas: ["Answer. "])   // no retriever
        engine.submit("anything")
        await engine.turnTask?.value
        XCTAssertEqual(rec.lastSystem, engine.persona.systemPrompt,
                       "without an attached folder the system prompt is just the persona")
    }

    func testRetrieverReturningNilLeavesSystemPromptUnchanged() async {
        // Empty folder / still-indexing / no matches → nil → no injection.
        let (engine, rec) = makeEngine(minChars: 0, deltas: ["Answer. "], retrieve: { _ in nil })
        engine.submit("anything")
        await engine.turnTask?.value
        XCTAssertEqual(rec.lastSystem, engine.persona.systemPrompt)
    }

    func testConcurrentSubmitIsIgnoredWhileBusy() async {
        let (engine, _) = makeEngine(minChars: 0, deltas: ["First answer. "])
        engine.submit("one")
        engine.submit("two")   // a turn is already in flight — ignored
        await engine.turnTask?.value
        let userTurns = engine.transcript.filter { $0.role == .user }.map(\.text)
        XCTAssertEqual(userTurns, ["one"])
    }
}
