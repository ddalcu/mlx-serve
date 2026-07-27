import XCTest
@testable import MLXCore

/// `ClonedVoiceSynthesizer`'s production `voice:` closure re-reads the saved
/// settings once per utterance — that per-utterance re-read is the seam a
/// per-agent voice hangs on, so an agent's voice applies from the very next
/// sentence with no restart and no new plumbing. The decision itself is pure.
final class AgentVoiceOverrideTests: XCTestCase {

    private func options(_ mutate: (inout ServerOptions) -> Void) -> ServerOptions {
        var o = ServerOptions()
        mutate(&o)
        return o
    }

    func testNoAgentVoiceReproducesTodaysSettingsBehavior() {
        XCTAssertNil(ActiveAgentVoice.neuralVoice(
            agent: nil, options: options { $0.voiceEngine = .system }))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(
            agent: nil, options: options { $0.voiceEngine = .kokoro; $0.kokoroVoice = "af_sky" }),
            .kokoro(voice: "af_sky"))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(
            agent: nil, options: options { $0.voiceEngine = .clone; $0.voiceClonePath = "/c.wav" }),
            .clone(clipPath: "/c.wav"))
    }

    func testBlankGlobalKokoroVoiceStillFallsBackToTheDefaultVoice() {
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(
            agent: nil, options: options { $0.voiceEngine = .kokoro; $0.kokoroVoice = "  " }),
            .kokoro(voice: "af_heart"))
    }

    func testCloneWithNoClipFallsBackToTheSystemVoice() {
        XCTAssertNil(ActiveAgentVoice.neuralVoice(
            agent: nil, options: options { $0.voiceEngine = .clone; $0.voiceClonePath = "" }))
    }

    func testAnAgentVoiceWinsOverTheGlobalSetting() {
        let global = options { $0.voiceEngine = .system }
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: .kokoro("af_bella"), options: global),
                       .kokoro(voice: "af_bella"))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: .clone("/mine.wav"), options: global),
                       .clone(clipPath: "/mine.wav"))
    }

    func testAnAgentPinnedToTheSystemVoiceOverridesAGlobalNeuralVoice() {
        // .system means "speak with the Apple synthesizer" — nil here is what the
        // synthesizer reads as "no neural voice", so it must NOT fall through to
        // the global Kokoro setting.
        let global = options { $0.voiceEngine = .kokoro; $0.kokoroVoice = "af_sky" }
        XCTAssertNil(ActiveAgentVoice.neuralVoice(agent: .system("com.apple.x"), options: global))
    }

    func testAnAgentVoiceWithAnEmptyValueDefersToTheGlobalSetting() {
        // A half-saved agent voice must not produce silence.
        let global = options { $0.voiceEngine = .kokoro; $0.kokoroVoice = "af_sky" }
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: .kokoro("  "), options: global),
                       .kokoro(voice: "af_sky"))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: .clone(""), options: global),
                       .kokoro(voice: "af_sky"))
    }

    func testTheHolderIsReadableFromTheSynthesizersClosure() {
        // The closure is not main-actor bound, so the override lives behind a
        // lock rather than on AppState.
        ActiveAgentVoice.set(.kokoro("af_nicole"))
        XCTAssertEqual(ActiveAgentVoice.current, .kokoro("af_nicole"))
        ActiveAgentVoice.set(nil)
        XCTAssertNil(ActiveAgentVoice.current)
    }
}
