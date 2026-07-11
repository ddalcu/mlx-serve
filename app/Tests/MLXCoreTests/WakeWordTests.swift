import XCTest
@testable import MLXCore

/// Pins the wake-word detector: an utterance opens the assistant only when it
/// carries the wake phrase, the remaining query is returned verbatim, a bare
/// phrase yields an empty remainder (the controller treats that as "arm"), and
/// everyday speech is rejected.
final class WakeWordTests: XCTestCase {

    func testExactPhraseReturnsRemainder() {
        XCTAssertEqual(WakeWord.strip("Hey Loki, what's the weather?"), "what's the weather?")
    }

    func testCaseAndPunctuationInsensitive() {
        XCTAssertEqual(WakeWord.strip("hey   loki ... tell me a joke"), "tell me a joke")
        XCTAssertEqual(WakeWord.strip("HEY LOKI! status report"), "status report")
    }

    func testRemainderPreservesOriginalCasingAndPunctuation() {
        // The query keeps its real text (apostrophes, capitals), not a normalized form.
        XCTAssertEqual(WakeWord.strip("Hey Loki, What's the ETA?"), "What's the ETA?")
    }

    func testBareWakePhraseReturnsEmptyString() {
        XCTAssertEqual(WakeWord.strip("Hey Loki"), "")
        XCTAssertEqual(WakeWord.strip("hey loki."), "")
    }

    func testGreetingVariantsOpenTheAssistant() {
        XCTAssertEqual(WakeWord.strip("Okay Loki, go ahead"), "go ahead")
        XCTAssertEqual(WakeWord.strip("Hi Loki do the thing"), "do the thing")
    }

    func testBareNameWithoutGreetingMatchesExactName() {
        XCTAssertEqual(WakeWord.strip("Loki, what's up?"), "what's up?")
    }

    func testHomophoneMatchesOnlyWithGreeting() {
        // "low key" is a plausible mis-hearing of "Loki" — accepted after a greeting…
        XCTAssertEqual(WakeWord.strip("Hey low key, set a timer"), "set a timer")
        // …but NOT on its own, so everyday "lowkey I agree" doesn't wake it.
        XCTAssertNil(WakeWord.strip("lowkey I agree with that"))
        XCTAssertNil(WakeWord.strip("low key this is great"))
    }

    func testNoWakeWordReturnsNil() {
        XCTAssertNil(WakeWord.strip("what's the weather today"))
        XCTAssertNil(WakeWord.strip(""))
        XCTAssertNil(WakeWord.strip("   "))
    }

    // MARK: - Custom phrases (Settings ▸ Voice ▸ Wake phrase)

    func testCustomPhraseWakesAndTheOldNameStopsWaking() {
        XCTAssertEqual(WakeWord.strip("Hey Jarvis, lights on", phrase: "hey jarvis"), "lights on")
        // Greeting variants and the bare exact name work for custom phrases too.
        XCTAssertEqual(WakeWord.strip("Okay Jarvis go ahead", phrase: "hey jarvis"), "go ahead")
        XCTAssertEqual(WakeWord.strip("Jarvis, status?", phrase: "hey jarvis"), "status?")
        // Renaming the assistant retires the old name.
        XCTAssertNil(WakeWord.strip("Hey Loki, lights on", phrase: "hey jarvis"))
    }

    /// The Settings field stores whatever the user typed; consumers normalize.
    func testNormalizePhraseToleratesUserFormatting() {
        XCTAssertEqual(WakeWord.normalizePhrase("  Hey,  JARVIS! "), "hey jarvis")
        XCTAssertEqual(WakeWord.normalizePhrase("Hey Loki"), WakeWord.defaultPhrase)
        // Empty / punctuation-only input can't be a phrase → nil, so callers
        // fall back to the default instead of a never-matching gate.
        XCTAssertNil(WakeWord.normalizePhrase(""))
        XCTAssertNil(WakeWord.normalizePhrase("  …!  "))
    }

    func testDisplayAndAssistantNameDeriveFromThePhrase() {
        XCTAssertEqual(WakeWord.display("hey jarvis"), "Hey Jarvis")
        XCTAssertEqual(WakeWord.assistantName("hey jarvis"), "Jarvis")
        // Single-token phrase: the whole phrase is the name.
        XCTAssertEqual(WakeWord.assistantName("computer"), "Computer")
    }

    func testWakeWordMustBeAWholeWord() {
        // A token that merely starts with the name doesn't count.
        XCTAssertNil(WakeWord.strip("lokimon is a game"))
        XCTAssertNil(WakeWord.strip("hey lokira what's up"))
    }

    func testCustomPhrase() {
        XCTAssertEqual(WakeWord.strip("computer, run diagnostics", phrase: "computer"), "run diagnostics")
        XCTAssertNil(WakeWord.strip("hey loki run diagnostics", phrase: "computer"))
    }
}
