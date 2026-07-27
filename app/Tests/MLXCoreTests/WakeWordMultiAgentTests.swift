import XCTest
@testable import MLXCore

/// Hands-free voice listens for EVERY agent's phrase plus the global fallback,
/// so the single-phrase `strip` grew a sibling. The traps: a short phrase eating
/// a longer one that contains it ("hey loki" swallowing "hey loki coder"), and
/// two agents whose phrases can't be told apart — which has to be refused at
/// save time, not discovered by talking.
final class WakeWordMultiAgentTests: XCTestCase {

    private let loki = UUID()
    private let coder = UUID()
    private let chef = UUID()

    // MARK: - match

    func testMatchesAnAgentPhraseAndReturnsTheQueryVerbatim() {
        let m = WakeWord.match("Hey Chef, What's for dinner?",
                              phrases: [(chef, "hey chef"), (coder, "hey coder")])
        XCTAssertEqual(m?.id, chef)
        XCTAssertEqual(m?.query, "What's for dinner?")
    }

    func testLongestPhraseWinsSoAShortOneCannotEatIt() {
        // The live trap: "hey loki" is a prefix of "hey loki coder".
        let phrases = [(loki, "hey loki"), (coder, "hey loki coder")]
        let m = WakeWord.match("hey loki coder, run the tests", phrases: phrases)
        XCTAssertEqual(m?.id, coder, "the more specific phrase must win regardless of list order")
        XCTAssertEqual(m?.query, "run the tests")

        // Reversed input order must not change the answer.
        XCTAssertEqual(WakeWord.match("hey loki coder, run the tests",
                                      phrases: phrases.reversed())?.id, coder)
    }

    func testTheShorterPhraseStillMatchesItsOwnUtterance() {
        let phrases = [(loki, "hey loki"), (coder, "hey loki coder")]
        let m = WakeWord.match("hey loki, what time is it?", phrases: phrases)
        XCTAssertEqual(m?.id, loki)
        XCTAssertEqual(m?.query, "what time is it?")
    }

    func testOneAgentsPhraseNeverAnswersForAnother() {
        let m = WakeWord.match("hey chef, deploy the app",
                              phrases: [(coder, "hey coder"), (chef, "hey chef")])
        XCTAssertEqual(m?.id, chef, "no cross-agent eating")
    }

    func testAmbientSpeechMatchesNothing() {
        XCTAssertNil(WakeWord.match("the chef said the coder was late",
                                    phrases: [(coder, "hey coder"), (chef, "hey chef")]))
    }

    func testBarePhraseYieldsAnEmptyQuery() {
        let m = WakeWord.match("Hey Chef.", phrases: [(chef, "hey chef")])
        XCTAssertEqual(m?.id, chef)
        XCTAssertEqual(m?.query, "")
    }

    func testGreetingVariantsWorkForCustomNamesToo() {
        XCTAssertEqual(WakeWord.match("Okay Chef, go", phrases: [(chef, "hey chef")])?.query, "go")
        XCTAssertEqual(WakeWord.match("Chef, go", phrases: [(chef, "hey chef")])?.query, "go")
    }

    func testHomophonesStayLokiSpecific() {
        // "low key" is a Loki mis-hearing; a custom name gets no invented ones.
        XCTAssertEqual(WakeWord.match("hey low key, hello",
                                      phrases: [(loki, "hey loki")])?.id, loki)
        XCTAssertNil(WakeWord.match("hey chief, hello", phrases: [(chef, "hey chef")]),
                     "no homophone table for custom names")
    }

    func testEmptyPhraseListMatchesNothing() {
        XCTAssertNil(WakeWord.match("hey loki, hello", phrases: []))
    }

    func testBlankPhrasesAreIgnoredRatherThanMatchingEverything() {
        XCTAssertNil(WakeWord.match("hello there", phrases: [(chef, "   ")]))
    }

    // MARK: - collision (checked at save time in the Agents window)

    func testIdenticalAndGreetingEquivalentPhrasesCollide() {
        XCTAssertTrue(WakeWord.collides("hey chef", with: ["hey chef"]))
        XCTAssertTrue(WakeWord.collides("Hey, Chef!", with: ["hey chef"]),
                      "normalization first")
        // Greetings are universal, so two phrases sharing the NAME are the same gate.
        XCTAssertTrue(WakeWord.collides("ok chef", with: ["hey chef"]))
        XCTAssertTrue(WakeWord.collides("hey loki chef", with: ["hey chef"]))
    }

    func testDistinctNamesDoNotCollide() {
        XCTAssertFalse(WakeWord.collides("hey chef", with: ["hey loki", "hey coder"]))
    }

    func testABlankPhraseNeverCollides() {
        // Blank means "use the app phrase" — the picker shows that, not an error.
        XCTAssertFalse(WakeWord.collides("  ", with: ["hey loki"]))
    }

    // MARK: - strip is untouched

    func testSinglePhraseStripIsUnchanged() {
        XCTAssertEqual(WakeWord.strip("Hey Loki, what's the weather?"), "what's the weather?")
        XCTAssertEqual(WakeWord.strip("Hey Loki"), "")
        XCTAssertNil(WakeWord.strip("lowkey I agree with that"))
    }
}
