import XCTest
@testable import MLXCore

/// A model that hears the recording itself (LFM3-Audio / Omni): the model row's fact, the utterance audio the mic
/// keeps, the composer toggle's look, and LFM3 as a voice engine.
final class NativeVoiceTests: XCTestCase {

    func testAModelThatSpeaksSaysSoOnItsRow() {
        let row: [String: Any] = ["id": "lfm3", "capabilities": ["chat", "audio", "speech"],
                                  "input_modalities": ["text", "audio"], "output_modalities": ["text", "audio"], "meta": [:]]
        XCTAssertTrue(APIClient.parseModelInfo(row).speaks)
        XCTAssertFalse(APIClient.parseModelInfo(["id": "qwen", "capabilities": ["chat", "audio"], "meta": [:]]).speaks,
                       "hearing is not speaking: Gemma 4's audio input keeps the cascaded voice")
    }

    func testTheMicKeepsTheUtteranceAt16kMonoWithAPreRoll() {
        // 48 kHz stereo, left 1 and right 0: the downmix averages, the resample keeps a third of the frames.
        let mono = UtteranceAudio.downmix16k(channels: [[Float](repeating: 1, count: 4800), [Float](repeating: 0, count: 4800)], sampleRate: 48000)
        XCTAssertEqual(mono.count, 1600)
        XCTAssertEqual(mono[800], 0.5, accuracy: 1e-6)

        var u = UtteranceAudio(preRollSeconds: 0.25, maxSeconds: 2)
        u.append([Float](repeating: 0.1, count: 16000))   // a second of room tone
        u.markSpeechStart()
        u.append([Float](repeating: 0.9, count: 8000))    // half a second of speech
        let taken = u.take()
        XCTAssertEqual(taken.count, 4000 + 8000, "a quarter second before speech began, then the speech")
        XCTAssertEqual(taken.first ?? 0, 0.1, accuracy: 1e-6)
        XCTAssertEqual(taken.last ?? 0, 0.9, accuracy: 1e-6)
        XCTAssertTrue(u.take().isEmpty, "taking ends the utterance")

        u.markSpeechStart()
        u.append([Float](repeating: 0.5, count: 16000 * 5))
        XCTAssertEqual(u.take().count, 16000 * 2, "an utterance is capped, keeping its latest audio")
    }

    func testTheVoiceButtonLooksDifferentWhenTheModelHearsYouItself() {
        let native = VoiceToggleLook.for(hears: true)
        let cascaded = VoiceToggleLook.for(hears: false)
        XCTAssertNotEqual(native.symbol, cascaded.symbol)
        XCTAssertNotEqual(native.onTint, cascaded.onTint)
        XCTAssertEqual(cascaded.symbol, "waveform", "every other model keeps today's toggle")
    }

    // MARK: LFM3 as a voice engine (Settings ▸ Voice, an agent's voice)

    @MainActor func testLFM3IsAVoiceLikeAnyOther() throws {
        var o = ServerOptions()
        o.voiceEngine = .lfm3
        o.lfm3Voice = "british-english-male"
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: nil, options: o), .lfm3(voice: "british-english-male"))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: .lfm3("french-female"), options: ServerOptions()), .lfm3(voice: "french-female"))
        let json = VoiceCloneTTS.requestBody(model: "m", text: "Hi.", voice: .lfm3(voice: "french-female"))
        XCTAssertEqual(json["voice"] as? String, "french-female")
        XCTAssertNil(json["ref_audio"])
        let agent = AgentVoice.lfm3("german-male")
        XCTAssertEqual(try JSONDecoder().decode(AgentVoice.self, from: JSONEncoder().encode(agent)), agent)
    }

    /// The chat model speaks the answer when it can (no second model); else the LFM3 speech pack on disk.
    func testTheLFM3VoiceUsesTheChatModelWhenItSpeaks() {
        func pack(_ type: String, _ path: String, downloading: Bool = false) -> LocalModel {
            var m = LocalModel(id: path, name: path, path: path, sizeFormatted: "2 GB", modelType: type, source: .mlxServe, kind: .base)
            m.isDownloading = downloading
            return m
        }
        let models = [pack("lfm3", "/m/text"), pack("lfm3_omni", "/m/omni"), pack("lfm3_audio", "/m/audio")]
        XCTAssertEqual(LFM3Voice.modelPath(models, chatModelPath: "/m/omni", chatSpeaks: true), "/m/omni")
        XCTAssertEqual(LFM3Voice.modelPath(models, chatModelPath: "/m/text", chatSpeaks: false), "/m/audio", "the smaller speech pack first")
        XCTAssertNil(LFM3Voice.modelPath([pack("lfm3_audio", "/m/a", downloading: true)], chatModelPath: nil, chatSpeaks: false))
    }

    func testLFM3VoicesReadAsLanguageAndGender() {
        XCTAssertEqual(LFM3Voice.displayName("british-english-male"), "British English (male)")
        XCTAssertEqual(LFM3Voice.displayName("french-female"), "French (female)")
        XCTAssertEqual(LFM3Voice.voices([], current: "french-female"), ["french-female"], "no server: the pick stays listed")
    }

    func testAModelRowListsItsVoices() {
        let row: [String: Any] = ["id": "lfm3", "output_modalities": ["text", "audio"], "voices": ["american-english-female", "french-male"], "meta": [:]]
        XCTAssertEqual(APIClient.parseModelInfo(row).voices, ["american-english-female", "french-male"])
    }
}

