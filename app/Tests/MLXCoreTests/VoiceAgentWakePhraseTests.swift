import XCTest
@testable import MLXCore

/// Which wake phrase hands-free mode actually listens for.
///
/// Voice mode launched from a chat adopts that chat's agent — its persona, its
/// voice, its tools. The phrase has to come along: an agent that introduces
/// itself by name but only answers to the app's global "hey loki" is the
/// mismatch the user hears first, and the "Say “…”" hint would be telling them
/// to say the wrong thing.
final class VoiceAgentWakePhraseTests: XCTestCase {

    func testAgentPhraseWinsOverTheGlobalSetting() {
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: "hey chef", global: "hey loki"),
                       "hey chef")
    }

    func testNoAgentPhraseFallsBackToTheGlobalSetting() {
        // Most agents don't set one — they answer to the app's phrase.
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: nil, global: "hey loki"), "hey loki")
    }

    func testBlankOrWhitespaceAgentPhraseIsNotAPhrase() {
        // A half-saved field must not produce a never-matching gate that makes
        // voice mode look broken.
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: "", global: "hey loki"), "hey loki")
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: "   ", global: "hey loki"), "hey loki")
    }

    func testAgentPhraseIsNormalizedLikeTheSettingsField() {
        // Agents are edited by hand (and written by a model), so the raw value
        // carries punctuation and case exactly as the Settings field does.
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: "Hey, Chef!", global: "hey loki"),
                       "hey chef")
    }

    func testAnEmptyGlobalStillYieldsAUsablePhrase() {
        // Never return "" — `strip` would match everything and every stray
        // noise would count as a wake.
        XCTAssertEqual(WakeWord.activePhrase(agentPhrase: nil, global: ""), WakeWord.defaultPhrase)
    }

    func testTheResolvedPhraseActuallyStripsThatAgentsWake() {
        // End to end: the phrase this returns is the one `strip` is called with,
        // so it must recognize the agent's own wake and not the global one.
        let phrase = WakeWord.activePhrase(agentPhrase: "hey chef", global: "hey loki")
        XCTAssertEqual(WakeWord.strip("hey chef what's for dinner", phrase: phrase),
                       "what's for dinner")
        XCTAssertNil(WakeWord.strip("hey loki what's for dinner", phrase: phrase),
                     "the global phrase must not wake an agent with its own")
    }
}
