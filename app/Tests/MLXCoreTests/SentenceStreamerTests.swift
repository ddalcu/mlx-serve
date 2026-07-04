import XCTest
@testable import MLXCore

/// P4 avatar-loop groundwork: the sentence-level TTS pipeline feeds streaming
/// LLM deltas into `SentenceStreamer` and speaks each COMPLETE sentence while
/// the next one decodes. Pure + incremental, so it's fully unit-tested here.
final class SentenceStreamerTests: XCTestCase {

    func testEmitsSentencesAsDeltasComplete() {
        var s = SentenceStreamer()
        XCTAssertEqual(s.feed("Hello wor"), [])
        XCTAssertEqual(s.feed("ld. How are"), ["Hello world."])
        XCTAssertEqual(s.feed(" you? Fine"), ["How are you?"])
        XCTAssertEqual(s.flush(), "Fine")
    }

    func testDecimalsAndVersionsDoNotSplit() {
        var s = SentenceStreamer()
        // The final "." has no lookahead yet (could be "3." awaiting "14"), so
        // the last sentence arrives on flush — the honest streaming contract.
        XCTAssertEqual(s.feed("Pi is 3.14159 exactly. Version 2.5 shipped."),
                       ["Pi is 3.14159 exactly."])
        XCTAssertEqual(s.flush(), "Version 2.5 shipped.")
    }

    func testExclamationAndEllipsis() {
        var s = SentenceStreamer()
        XCTAssertEqual(s.feed("Wow! That's great… Right? Yes"),
                       ["Wow!", "That's great…", "Right?"])
        XCTAssertEqual(s.flush(), "Yes")
    }

    func testNewlineTerminatesASentenceFragment() {
        // Lists / headings without terminal punctuation still need speaking —
        // a newline closes the fragment.
        var s = SentenceStreamer()
        XCTAssertEqual(s.feed("First point\nSecond point\n"),
                       ["First point", "Second point"])
    }

    func testTinyFragmentsAreHeldUntilMinLength() {
        // "Ok." alone is held and merged with what follows (avoids machine-gun
        // TTS clips), but flush() always releases it.
        var s = SentenceStreamer(minChars: 8)
        XCTAssertEqual(s.feed("Ok. Let us begin now. "), ["Ok. Let us begin now."])
        var s2 = SentenceStreamer(minChars: 8)
        XCTAssertEqual(s2.feed("Ok."), [])
        XCTAssertEqual(s2.flush(), "Ok.")
    }

    func testMarkdownEmphasisAndCodeFencesAreStripped() {
        // The avatar speaks; asterisks and backticks are noise to a TTS voice.
        var s = SentenceStreamer()
        let out = s.feed("This is **bold** and `code`. Done.\n")
        XCTAssertEqual(out, ["This is bold and code.", "Done."])
    }

    func testFlushOnEmptyIsNil() {
        var s = SentenceStreamer()
        XCTAssertNil(s.flush())
        _ = s.feed("Hi there. ")
        XCTAssertNil(s.flush())
    }

    func testWhitespaceOnlyDeltasAreIgnored() {
        var s = SentenceStreamer()
        XCTAssertEqual(s.feed("   \n  "), [])
        XCTAssertNil(s.flush())
    }
}
