import XCTest
@testable import MLXCore

/// Reading a seed a user typed or PASTED.
///
/// The image and video panes rendered their seed as a bare `Stepper` whose
/// label was a static `Text` — so there was no text entry at all and the only
/// input was ±1 clicks. Reaching a pasted 7-digit seed was 3.8 million of them.
final class SeedTextTests: XCTestCase {

    private let videoRange = 0...Int.max
    private let imageRange = -1...Int.max

    func testPlainDigitsAreRead() {
        XCTAssertEqual(SeedText.parse("3847592", in: videoRange), 3847592)
        XCTAssertEqual(SeedText.parse("0", in: videoRange), 0)
        XCTAssertEqual(SeedText.parse("  42  ", in: videoRange), 42)
    }

    /// The whole point is pasting, and people paste seeds out of captions and
    /// filenames, not out of a numeric field. Being strict here would reject
    /// exactly the input this exists to accept.
    func testPastedJunkAroundTheNumberIsForgiven() {
        XCTAssertEqual(SeedText.parse("Seed: 3847592", in: videoRange), 3847592)
        XCTAssertEqual(SeedText.parse("3,847,592", in: videoRange), 3847592)
        XCTAssertEqual(SeedText.parse("seed=1234\n", in: videoRange), 1234)
        // A filename holds several numbers; the seed is the longest run, and
        // the resolution and the container's own "4" are not part of it.
        XCTAssertEqual(SeedText.parse("video_1234_768p.mp4", in: videoRange), 1234)
        XCTAssertEqual(SeedText.parse("h3_3847592_1344x768.mp4", in: videoRange), 3847592)
    }

    /// Nothing readable must leave the current value alone rather than becoming
    /// zero — clearing the field to type a new number is the normal way to use
    /// it, and a field that snaps to 0 mid-edit fights the user.
    func testUnreadableInputIsNilRatherThanZero() {
        XCTAssertNil(SeedText.parse("", in: videoRange))
        XCTAssertNil(SeedText.parse("   ", in: videoRange))
        XCTAssertNil(SeedText.parse("abc", in: videoRange))
        XCTAssertNil(SeedText.parse("-", in: imageRange))
        // Past what an Int can hold: refused, never silently truncated to some
        // other number the user did not ask for.
        XCTAssertNil(SeedText.parse("99999999999999999999999", in: videoRange))
    }

    /// `-1` means "random" in the image pane and means nothing in the video one
    /// (that backend has no random path and would send it verbatim), so the
    /// range decides whether the sign is even readable.
    func testTheSignIsOnlyReadWhereItMeansSomething() {
        XCTAssertEqual(SeedText.parse("-1", in: imageRange), -1)
        // Where the range has no negative half the `-` is just a character, the
        // same as it is inside a filename — so "-1" reads as 1 and the box then
        // SHOWS 1, which is the seed that will run. Nothing is hidden.
        XCTAssertEqual(SeedText.parse("-1", in: videoRange), 1)
        // A minus inside the text is not a sign — it is a filename separator.
        XCTAssertEqual(SeedText.parse("clip-1234", in: imageRange), 1234)
    }

    func testValuesAreClampedIntoTheRange() {
        XCTAssertEqual(SeedText.parse("-500", in: imageRange), -1)
        XCTAssertEqual(SeedText.parse("7", in: 0...5), 5)
    }

    /// What the field shows for the value it holds. `-1` reads as empty so the
    /// placeholder ("random") does the explaining — a literal "-1" in a seed box
    /// looks like a broken value.
    func testFormattingRoundTrips() {
        XCTAssertEqual(SeedText.format(3847592, in: videoRange), "3847592")
        XCTAssertEqual(SeedText.format(0, in: videoRange), "0")
        XCTAssertEqual(SeedText.format(-1, in: imageRange), "")
        // A -1 restored into a pane with no random path shows the seed that
        // will run, not a blank box that reads as random.
        XCTAssertEqual(SeedText.format(-1, in: videoRange), "0")
        for v in [0, 1, 42, 3847592, Int(UInt32.max)] {
            XCTAssertEqual(SeedText.parse(SeedText.format(v, in: videoRange), in: videoRange), v)
        }
    }

    /// The dice hands back a CONCRETE seed. Rolling the random sentinel would
    /// mean the dice sometimes rolls "surprise me", which is the one value it
    /// cannot be allowed to produce — the whole point is getting a number you
    /// can read off and paste back later.
    func testTheDiceNeverRollsTheRandomSentinel() {
        for _ in 0..<500 {
            let v = SeedText.randomSeed(in: imageRange)
            XCTAssertGreaterThanOrEqual(v, 0)
            XCTAssertTrue(imageRange.contains(v))
        }
    }

    /// Seeds are shared as 32-bit numbers, and a 19-digit one is unpasteable in
    /// practice even though `Int` would hold it.
    func testDiceRollsStayInTheRangeSeedsAreSharedIn() {
        for _ in 0..<500 {
            let v = SeedText.randomSeed(in: videoRange)
            XCTAssertTrue((0...Int(UInt32.max)).contains(v), "\(v) is outside the 32-bit seed range")
        }
        // A range tighter than the convention still binds.
        for _ in 0..<50 {
            XCTAssertTrue((0...5).contains(SeedText.randomSeed(in: 0...5)))
        }
    }

    /// A dice that returns the same number is not a dice. (Two identical draws
    /// in a row are legal; 200 are not.)
    func testTheDiceActuallyVaries() {
        let draws = Set((0..<200).map { _ in SeedText.randomSeed(in: videoRange) })
        XCTAssertGreaterThan(draws.count, 100)
    }

    /// Whatever it rolls has to survive the round trip through the box, or the
    /// user reads a number off the screen that is not the one that ran.
    func testADiceRollRoundTripsThroughTheField() {
        for _ in 0..<200 {
            let v = SeedText.randomSeed(in: videoRange)
            XCTAssertEqual(SeedText.parse(SeedText.format(v, in: videoRange), in: videoRange), v)
        }
    }

    /// Clearing the box must go back to random where random exists — the mirror
    /// of `-1` rendering blank. Without it the image pane would silently keep
    /// the last number, losing behaviour it already shipped with.
    func testAnEmptyBoxMeansRandomOnlyWhereRandomExists() {
        XCTAssertEqual(SeedText.emptyValue(in: imageRange), -1)
        XCTAssertNil(SeedText.emptyValue(in: videoRange))
        // Round trip through the blank: -1 formats to "", and "" reads back
        // as -1 rather than as a number the user never typed.
        XCTAssertEqual(SeedText.emptyValue(in: imageRange), SeedText.parse("-1", in: imageRange))
        XCTAssertNil(SeedText.parse(SeedText.format(-1, in: imageRange), in: imageRange))
    }
}
