import XCTest
@testable import MLXCore

/// Routing decisions of the voice-clone speech router (`ClonedVoiceSynthesizer`):
/// clip set → the clone pipeline synthesizes + plays; no clip → the wrapped
/// system synthesizer speaks; a failed synthesis falls back to the system voice
/// for THAT utterance (never dead air, never dropped) — and in every case the
/// drain callback fires so the controller can reopen the mic.
@MainActor
final class ClonedVoiceSynthesizerTests: XCTestCase {

    /// Minimal system-synth fake mirroring `SystemSpeechSynthesizer`'s contract:
    /// explicit pending count, drain callback when it hits zero. `autoFinish`
    /// completes each utterance synchronously (the common case in these tests).
    final class FakeSystemSynth: SpeechSynthesizing {
        var spoken: [String] = []
        var voiceIdentifier: String?
        var onQueueDrained: (() -> Void)?
        /// Test hook: observe each utterance as it arrives (ordering checks).
        var onSpoke: ((String) -> Void)?
        var stops = 0
        var autoFinish = true
        private var pending = 0
        var isSpeaking: Bool { pending > 0 }

        func enqueue(_ text: String) {
            spoken.append(text)
            onSpoke?(text)
            pending += 1
            if autoFinish { finishOne() }
        }

        func finishOne() {
            guard pending > 0 else { return }
            pending -= 1
            if pending == 0 { onQueueDrained?() }
        }

        func stop() { pending = 0; stops += 1 }
    }

    private func makeRouter(system: FakeSystemSynth,
                            clip: String?,
                            engine: VoiceEngine = .clone,
                            synth: @escaping ClonedVoiceSynthesizer.CloneSynth,
                            play: @escaping ClonedVoiceSynthesizer.ClonePlay = { _ in },
                            stopPlayback: @escaping () -> Void = {},
                            unload: (() async -> Void)? = nil) -> ClonedVoiceSynthesizer {
        ClonedVoiceSynthesizer(system: system,
                               voice: {
                                   guard let clip, !clip.isEmpty else { return nil }
                                   return engine == .kokoro ? .kokoro(voice: clip) : .clone(clipPath: clip)
                               },
                               synthesizeClone: synth,
                               playClone: play,
                               stopClonePlayback: stopPlayback,
                               unloadClone: unload)
    }

    // MARK: - Engine selection

    func testKokoroEngineSendsAVoiceNotAClip() async {
        // The two backends take DISJOINT controls: sending Kokoro a ref_audio
        // (or Qwen3-TTS a voice) is a named 400 server-side, so the engine
        // choice must decide which field travels — not both, not neither.
        let system = FakeSystemSynth()
        var seen: NeuralVoice?
        let router = makeRouter(system: system, clip: "af_bella,af_sky", engine: .kokoro,
                                synth: { text, sel in
                                    seen = sel
                                    return Data(text.utf8)
                                })
        let drained = expectation(description: "drained")
        router.onQueueDrained = { drained.fulfill() }
        router.enqueue("Hello.")
        await fulfillment(of: [drained], timeout: 2)

        XCTAssertEqual(seen, .kokoro(voice: "af_bella,af_sky"),
                       "a blended voice spec must reach the backend verbatim")
        XCTAssertEqual(system.spoken, [], "Kokoro must not fall back to the system voice")
    }

    func testNoVoiceConfiguredFallsBackToTheSystemVoice() async {
        let system = FakeSystemSynth()
        var called = false
        let router = makeRouter(system: system, clip: nil,
                                synth: { _, _ in called = true; return Data() })
        router.enqueue("Hello.")
        XCTAssertFalse(called, "with no voice selected nothing should hit the server")
        XCTAssertEqual(system.spoken, ["Hello."])
    }

    func testKokoroSynthesisFailureStillFallsBackPerUtterance() async {
        // The never-dead-air contract has to hold for the new engine too.
        let system = FakeSystemSynth()
        let router = makeRouter(system: system, clip: "af_heart", engine: .kokoro,
                                synth: { text, _ in text.contains("bad") ? nil : Data(text.utf8) })
        let drained = expectation(description: "drained")
        router.onQueueDrained = { drained.fulfill() }
        router.enqueue("good one")
        router.enqueue("bad one")
        await fulfillment(of: [drained], timeout: 2)
        XCTAssertEqual(system.spoken, ["bad one"],
                       "only the failed utterance falls back, and it is not dropped")
    }

    // MARK: - Request body

    func testKokoroBodyCarriesVoiceAndNeverRefAudio() {
        let body = VoiceCloneTTS.requestBody(model: "m", text: "hi", voice: .kokoro(voice: "af_bella,af_sky"))
        XCTAssertEqual(body["voice"] as? String, "af_bella,af_sky")
        XCTAssertNil(body["ref_audio"], "Kokoro cannot clone — ref_audio is a named 400, not an ignored extra")
        XCTAssertEqual(body["input"] as? String, "hi")
        XCTAssertEqual(body["model"] as? String, "m")
    }

    func testCloneBodyCarriesRefAudioAndNeverVoice() {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("clip-\(UUID().uuidString).wav")
        try? Data("RIFFfake".utf8).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let body = VoiceCloneTTS.requestBody(model: "m", text: "hi", voice: .clone(clipPath: tmp.path))
        XCTAssertNotNil(body["ref_audio"], "the clip must be base64'd into the body")
        XCTAssertNil(body["voice"], "Qwen3-TTS has no voice list")
    }

    func testCloneBodyWithAnUnreadableClipOmitsRefAudioRatherThanSendingJunk() {
        let body = VoiceCloneTTS.requestBody(model: "m", text: "hi", voice: .clone(clipPath: "/no/such/clip.wav"))
        XCTAssertNil(body["ref_audio"])
        XCTAssertEqual(body["input"] as? String, "hi")
    }

    // MARK: - Pre-warm (docs/qwentts-cache.md first-sentence lever)

    func testWarmBodyCarriesWarmOnlyAndRefAudioNeverInput() {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("clip-\(UUID().uuidString).wav")
        try? Data("RIFFfake".utf8).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let body = VoiceCloneTTS.warmBody(model: "m", voice: .clone(clipPath: tmp.path))
        XCTAssertNotNil(body)
        XCTAssertEqual(body?["warm_only"] as? Bool, true)
        XCTAssertNotNil(body?["ref_audio"], "the clip must be base64'd into the warm body")
        XCTAssertNil(body?["input"], "a warm body with text would synthesize audio nobody asked for")
        XCTAssertNil(body?["voice"])
    }

    func testWarmBodyIsNilForKokoroAndUnreadableClips() {
        XCTAssertNil(VoiceCloneTTS.warmBody(model: "m", voice: .kokoro(voice: "af_bella")),
                     "Kokoro voices are a table lookup — nothing to warm")
        XCTAssertNil(VoiceCloneTTS.warmBody(model: "m", voice: .clone(clipPath: "/no/such/clip.wav")))
    }

    func testPrewarmFiresOnlyForACloneVoice() async {
        let system = FakeSystemSynth()
        var warmed: [NeuralVoice] = []
        let fired = expectation(description: "warm fired")
        let clone = ClonedVoiceSynthesizer(
            system: system,
            voice: { .clone(clipPath: "/clips/me.wav") },
            synthesizeClone: { _, _ in nil },
            playClone: { _ in },
            prewarmClone: { sel in warmed.append(sel); fired.fulfill() })
        clone.prewarm()
        await fulfillment(of: [fired], timeout: 2)
        XCTAssertEqual(warmed, [.clone(clipPath: "/clips/me.wav")])

        // Kokoro / no voice: the closure must never fire.
        var wrongWarms = 0
        let kokoro = ClonedVoiceSynthesizer(
            system: FakeSystemSynth(),
            voice: { .kokoro(voice: "af_bella") },
            synthesizeClone: { _, _ in nil },
            playClone: { _ in },
            prewarmClone: { _ in wrongWarms += 1 })
        kokoro.prewarm()
        let none = ClonedVoiceSynthesizer(
            system: FakeSystemSynth(),
            voice: { nil },
            synthesizeClone: { _, _ in nil },
            playClone: { _ in },
            prewarmClone: { _ in wrongWarms += 1 })
        none.prewarm()
        // Give any stray Task a beat to run before asserting it didn't.
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(wrongWarms, 0, "only the clone arm has anything to warm")
    }

    // MARK: - Routing

    func testClipSetRoutesThroughClonePipelineInOrderAndDrains() async {
        let system = FakeSystemSynth()
        var synthesized: [String] = []
        var played: [String] = []
        let router = makeRouter(system: system, clip: "/clips/me.wav",
                                synth: { text, sel in
                                    XCTAssertEqual(sel, .clone(clipPath: "/clips/me.wav"))
                                    synthesized.append(text)
                                    return Data(text.utf8)
                                },
                                play: { data in played.append(String(decoding: data, as: UTF8.self)) })
        let drained = expectation(description: "queue drained")
        router.onQueueDrained = { drained.fulfill() }

        router.enqueue("First sentence.")
        router.enqueue("Second sentence.")
        XCTAssertTrue(router.isSpeaking, "queued work must read as speaking")

        await fulfillment(of: [drained], timeout: 5)
        XCTAssertEqual(synthesized, ["First sentence.", "Second sentence."])
        XCTAssertEqual(played, ["First sentence.", "Second sentence."], "clips play in submit order")
        XCTAssertTrue(system.spoken.isEmpty, "clone path must not reach the system voice")
        XCTAssertFalse(router.isSpeaking)
    }

    func testNoClipDelegatesToSystemVoice() async {
        let system = FakeSystemSynth()
        var synthCalls = 0
        let router = makeRouter(system: system, clip: nil,
                                synth: { _, _ in synthCalls += 1; return Data() })
        let drained = expectation(description: "queue drained")
        router.onQueueDrained = { drained.fulfill() }

        router.enqueue("Hello there.")

        await fulfillment(of: [drained], timeout: 5)
        XCTAssertEqual(system.spoken, ["Hello there."])
        XCTAssertEqual(synthCalls, 0, "no clip → the clone synth must never be called")
    }

    func testEmptyClipPathCountsAsNoClone() async {
        let system = FakeSystemSynth()
        let router = makeRouter(system: system, clip: "",
                                synth: { _, _ in XCTFail("clone synth called for empty clip"); return nil })
        let drained = expectation(description: "queue drained")
        router.onQueueDrained = { drained.fulfill() }
        router.enqueue("Hi.")
        await fulfillment(of: [drained], timeout: 5)
        XCTAssertEqual(system.spoken, ["Hi."])
    }

    // MARK: - Fallback (never dead air)

    func testSynthesisFailureFallsBackToSystemAndStillDrains() async {
        let system = FakeSystemSynth()
        var played: [String] = []
        let router = makeRouter(system: system, clip: "/clips/me.wav",
                                synth: { _, _ in nil },   // TTS down / model missing
                                play: { data in played.append(String(decoding: data, as: UTF8.self)) })
        let drained = expectation(description: "queue drained")
        router.onQueueDrained = { drained.fulfill() }

        router.enqueue("Still audible.")

        await fulfillment(of: [drained], timeout: 5)
        XCTAssertEqual(system.spoken, ["Still audible."], "failed synthesis must still be spoken")
        XCTAssertTrue(played.isEmpty)
        XCTAssertFalse(router.isSpeaking)
    }

    func testMixedFailureKeepsUtteranceOrder() async {
        let system = FakeSystemSynth()
        var played: [String] = []
        var order: [String] = []
        let router = makeRouter(system: system, clip: "/clips/me.wav",
                                synth: { text, _ in text.contains("fails") ? nil : Data(text.utf8) },
                                play: { data in
                                    let s = String(decoding: data, as: UTF8.self)
                                    played.append(s)
                                    order.append(s)
                                })
        system.onSpoke = { order.append($0) }
        let drained = expectation(description: "queue drained")
        router.onQueueDrained = { drained.fulfill() }

        router.enqueue("one ok")
        router.enqueue("two fails")
        router.enqueue("three ok")

        await fulfillment(of: [drained], timeout: 5)
        XCTAssertEqual(played, ["one ok", "three ok"])
        XCTAssertEqual(system.spoken, ["two fails"])
        XCTAssertEqual(order, ["one ok", "two fails", "three ok"],
                       "the fallback utterance speaks AT ITS TURN, not reordered")
    }

    // MARK: - Barge-in + shutdown

    func testStopClearsBothPipelinesAndSilencesPlayback() async {
        let system = FakeSystemSynth()
        var played = 0
        var playbackStops = 0
        let router = makeRouter(system: system, clip: "/clips/me.wav",
                                synth: { text, _ in
                                    try? await Task.sleep(nanoseconds: 100_000_000)
                                    return Data(text.utf8)
                                },
                                play: { _ in played += 1 },
                                stopPlayback: { playbackStops += 1 })
        router.enqueue("Interrupted sentence.")
        XCTAssertTrue(router.isSpeaking)

        router.stop()
        XCTAssertFalse(router.isSpeaking, "stop() must read idle immediately (barge-in)")
        XCTAssertEqual(playbackStops, 1)
        XCTAssertEqual(system.stops, 1)

        // The in-flight synthesis resolves after stop — it must be discarded.
        try? await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertEqual(played, 0, "a stopped turn's clips must never play")
    }

    func testShutdownUnloadsTheCloneModel() async {
        let system = FakeSystemSynth()
        let unloaded = expectation(description: "unload called")
        let router = makeRouter(system: system, clip: nil,
                                synth: { _, _ in nil },
                                unload: { unloaded.fulfill() })
        router.shutdown()
        await fulfillment(of: [unloaded], timeout: 5)
    }

    // MARK: - Clip persistence path contract

    func testClipStorePathShape() {
        let home = NSString(string: "~").expandingTildeInPath
        XCTAssertEqual(VoiceCloneClipStore.directory, home + "/.mlx-serve/voice-clips")
        XCTAssertEqual(VoiceCloneClipStore.destinationPath,
                       home + "/.mlx-serve/voice-clips/voice-clone.wav")
        XCTAssertTrue(VoiceCloneClipStore.destinationPath.hasSuffix(".wav"))
    }
}
