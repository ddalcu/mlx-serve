import XCTest
@testable import MLXCore

/// Pins streaming sentence segmentation: complete sentences are released as soon
/// as their terminator is followed by more text, the trailing partial is held
/// until `flush`, nothing is emitted twice, and common abbreviations don't cause
/// a false split.
final class SentenceChunkerTests: XCTestCase {

    func testEmitsCompleteSentenceFollowedBySpace() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "Hello there. "), ["Hello there."])
    }

    func testHoldsTrailingSentenceUntilFlush() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "Bye"), [])
        XCTAssertEqual(c.flush(), ["Bye"])
    }

    func testReleasesEarlierSentenceWhileBufferingPartial() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "Hello there. How are"), ["Hello there."])
        XCTAssertEqual(c.ingest(fullText: "Hello there. How are you?"), [])      // still trailing
        XCTAssertEqual(c.flush(), ["How are you?"])
    }

    func testNeverEmitsTheSameSentenceTwice() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "One. "), ["One."])
        XCTAssertEqual(c.ingest(fullText: "One. Two. "), ["Two."])
        XCTAssertEqual(c.ingest(fullText: "One. Two. Three. "), ["Three."])
    }

    func testMultipleSentencesInOneUpdate() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "One. Two! Three? "), ["One.", "Two!", "Three?"])
    }

    func testNewlineIsASentenceBoundary() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "Title\nBody"), ["Title"])
        XCTAssertEqual(c.flush(), ["Body"])
    }

    func testTrimsWhitespaceAroundEmittedSentences() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "  Spaced out.   Next part. "),
                       ["Spaced out.", "Next part."])
    }

    func testDoesNotSplitOnCommonAbbreviation() {
        var c = SentenceChunker()
        // "Dr." must not end the sentence; the real boundary is after "here."
        XCTAssertEqual(c.ingest(fullText: "Dr. Smith is here. "), ["Dr. Smith is here."])
    }

    func testDoesNotSplitOnSingleLetterInitial() {
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "John Q. Public arrived. "), ["John Q. Public arrived."])
    }

    func testFlushOnEmptyChunkerReturnsNothing() {
        var c = SentenceChunker()
        XCTAssertEqual(c.flush(), [])
    }

    func testIngestAfterFlushOfTheSameTextEmitsNothing() {
        // Voice mode re-ingests the finished answer whenever the session list
        // republishes while TTS is still speaking; a flush must not rewind the
        // chunker or the whole response is enqueued (and spoken) twice.
        var c = SentenceChunker()
        XCTAssertEqual(c.ingest(fullText: "One. Two"), ["One."])
        XCTAssertEqual(c.flush(), ["Two"])
        XCTAssertEqual(c.ingest(fullText: "One. Two"), [])
        XCTAssertEqual(c.flush(), [])
    }
}
