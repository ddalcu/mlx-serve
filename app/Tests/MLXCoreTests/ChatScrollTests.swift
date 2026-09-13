import XCTest
import SwiftUI
@testable import MLXCore

/// The transcript's follow-the-newest-line behaviour, as a pure state machine.
///
/// It used to be three loosely-coupled hacks in `ChatView`: a preference key
/// reporting a 1pt anchor's position, an app-global `NSEvent` scroll-wheel
/// monitor, and an animated `scrollTo` fired on every streamed token. Each one
/// was wrong in a way the others hid, and none of them was testable. The rules
/// below are what that machinery was trying to express.
final class ChatScrollTests: XCTestCase {

    // MARK: - Opening a conversation

    func testATranscriptOpensOnTheNewestMessageWithoutBeingScrolledThere() {
        // Restoring a saved chat used to land at the OLDEST message, and was
        // fixed by scrolling to the end on appear. That jump is itself the
        // blank-transcript bug (C16): a lazy stack that has measured almost
        // none of its rows puts "the end" a few hundred points down, and the
        // jump lands the visible rectangle outside the rows that exist. The
        // view is laid out bottom-anchored instead, so opening at the newest
        // line costs no scroll at all.
        var s = ChatScrollState()
        XCTAssertEqual(s.handle(.transcriptShown), .none)
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testSwitchingSessionsRe_engagesFollowing() {
        // ChatDetailView is REUSED across tabs, so scroll state is per-view and
        // leaked between conversations: leaving tab A scrolled up made tab B
        // open unpinned.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 800))
        XCTAssertFalse(s.isPinnedToBottom)

        _ = s.handle(.transcriptShown)
        XCTAssertTrue(s.isPinnedToBottom)
    }

    // MARK: - Streaming while the user is at the bottom

    func testSittingAtTheBottomIssuesNoScrollAtAll() {
        // The smooth path: while pinned, the scroll view's own bottom size-change
        // anchor keeps the newest line in view, so a streamed token produces no
        // explicit scroll. The old code ran a 0.15s `withAnimation` scrollTo per
        // TOKEN — dozens of overlapping animations a second, which is the jank.
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 0)), .none)
    }

    func testDriftBelowTheFoldIsCorrectedWithoutAnimation() {
        // Belt and braces: if the anchor doesn't keep up (a row re-measures, an
        // image finishes decoding), snap back — never animated, or the snaps
        // queue up into the same animation storm.
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 18)),
                       .toBottom(animated: false))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testALargeJumpWhileNobodyIsTouchingItIsStillJustAContentChange() {
        // A tool-call card or a media progress card appearing adds hundreds of
        // points in one layout pass. Nobody scrolled, so we follow.
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 420)),
                       .toBottom(animated: false))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    // MARK: - The user takes over

    func testScrollingUpDisengages() {
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        _ = s.handle(.driverChanged(.user))
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 500)), .none)
        XCTAssertFalse(s.isPinnedToBottom)
    }

    func testReadingHistoryIsNeverDraggedBackDown() {
        // The whole point of disengaging. Once unpinned, content growth reports
        // an ever-larger distance and must produce nothing.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 500))
        _ = s.handle(.driverChanged(.idle))

        for distance in stride(from: CGFloat(500), through: 2000, by: 100) {
            XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: distance)), .none,
                           "content growth at \(distance)pt from the bottom must not scroll")
        }
        XCTAssertFalse(s.isPinnedToBottom)
    }

    func testAFlickThatStaysAtTheBottomDoesNotDisengage() {
        // The old monitor unpinned on ANY `scrollingDeltaY > 0` anywhere in the
        // app — a one-notch nudge, a horizontal swipe's vertical component, or a
        // scroll in a completely different window all switched auto-follow off.
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 9))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testRubberBandOverscrollCountsAsTheBottom() {
        // Flinging down hard overscrolls past the end, so the distance goes
        // NEGATIVE and the elastic settle reports upward movement. Reading that
        // as "the user scrolled up" is exactly why auto-follow refused to
        // re-engage after scrolling to the bottom.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 700))
        XCTAssertFalse(s.isPinnedToBottom)

        _ = s.handle(.geometryChanged(distanceFromBottom: -46))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testCatchingUpMidMomentumRe_engages() {
        // Re-engaging must not wait for the gesture to end — the distance is the
        // signal, whoever is moving it.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 900))
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 4)), .none)
        XCTAssertTrue(s.isPinnedToBottom)
    }

    // MARK: - Our own scrolls

    func testAScrollOfOursNeitherDisengagesNorFightsItself() {
        // Mid-flight of an animated jump the distance is large and shrinking.
        // Correcting there would cancel the animation; disengaging there would
        // switch auto-follow off every time we used it.
        var s = ChatScrollState()
        _ = s.handle(.userSentMessage)
        _ = s.handle(.driverChanged(.us))
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: 300)), .none)
        XCTAssertTrue(s.isPinnedToBottom)
    }

    // MARK: - Explicit user intent

    func testSendingAMessageAlwaysJumpsToIt() {
        // Sending from halfway up the history used to leave you there, watching
        // nothing: `isNearBottom` was false so no scroll fired for your own turn.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 900))
        _ = s.handle(.driverChanged(.idle))

        XCTAssertEqual(s.handle(.userSentMessage), .toBottom(animated: true))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testJumpToLatestRe_engagesFollowing() {
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 900))

        XCTAssertEqual(s.handle(.jumpTapped), .toBottom(animated: true))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    // MARK: - A row gets shorter

    /// The offset is a distance from the top of the transcript, not a hold on a
    /// piece of content: a row that shrinks above the reader slides everything
    /// they can see up by what it lost. The control they clicked sits at the
    /// BOTTOM of that row, so it — and everything below it — stays put exactly
    /// when the offset drops by what the row has lost.
    func testAShrinkingRowKeepsTheControlWhereItWas() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 5000, contentHeight: 7000)),
                       .toOffset(3000))
    }

    /// An animated fold loses its height over several frames; every one is
    /// measured against where the fold STARTED, never against the last frame.
    func testAnAnimatedShrinkTracksFrameByFrame() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 5000, contentHeight: 8500)), .toOffset(4500))
        // The correction landed; the offset now reads 4500. No new height, no action.
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 4500, contentHeight: 8500)), .none)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 4500, contentHeight: 7000)), .toOffset(3000))
    }

    /// A row taller than everything above it: the view stops at the top.
    func testTheCorrectedOffsetNeverGoesAboveTheTop() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 800, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 800, contentHeight: 2000)), .toOffset(0))
    }

    /// Once the fold is over, height changes are content again (a reply
    /// streaming in) and must not be corrected against a stale snapshot.
    func testAFinishedShrinkStopsCorrecting() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        _ = s.handle(.rowDidShrink)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9400)), .none)
    }

    func testAReaderWhoScrollsMidShrinkWins() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        _ = s.handle(.driverChanged(.user))
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 4200, contentHeight: 7000)), .none)
    }

    /// While following the end, the bottom anchor already has it.
    func testAShrinkingRowWhileFollowingTheEndIsLeftToTheAnchor() {
        var s = ChatScrollState()
        _ = s.handle(.transcriptShown)
        _ = s.handle(.contentGeometry(offsetY: 8000, contentHeight: 9000))
        _ = s.handle(.rowWillShrink)
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 8000, contentHeight: 7000)), .none)
        XCTAssertTrue(s.isPinnedToBottom)
    }

    /// The frame after a fold: content shorter than the stale offset, distance
    /// deeply negative — the rubber-band shape, without anyone having gone
    /// anywhere. Reading it as arrival re-engaged following, and the bottom
    /// anchor plus the correction rule then dragged the reader to the end,
    /// overriding every fold correction that was ever issued.
    func testAFoldOvershootIsNotArrivalAtTheBottom() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 30000, contentHeight: 31000))
        _ = s.handle(.rowWillShrink)
        XCTAssertEqual(s.handle(.geometryChanged(distanceFromBottom: -21000)), .none)
        XCTAssertFalse(s.isPinnedToBottom)
        // Once the fold is over, the rule is the rule again.
        _ = s.handle(.rowDidShrink)
        _ = s.handle(.geometryChanged(distanceFromBottom: -46))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testAHeightChangeOutsideAShrinkIsNotCorrected() {
        var s = unpinned()
        _ = s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9000))
        XCTAssertEqual(s.handle(.contentGeometry(offsetY: 5000, contentHeight: 9400)), .none)
    }

    /// Reading history with following switched off.
    private func unpinned() -> ChatScrollState {
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 900))
        _ = s.handle(.driverChanged(.idle))
        return s
    }

    // MARK: - Driver classification

    func testMomentumBelongsToTheUserAndAnimationBelongsToUs() {
        // `.decelerating` is the user's fling still carrying — treating it as
        // ours would let momentum sail past the bottom without re-engaging.
        XCTAssertEqual(ChatScrollDriver(.tracking), .user)
        XCTAssertEqual(ChatScrollDriver(.interacting), .user)
        XCTAssertEqual(ChatScrollDriver(.decelerating), .user)
        XCTAssertEqual(ChatScrollDriver(.animating), .us)
        XCTAssertEqual(ChatScrollDriver(.idle), .idle)
    }

    // MARK: - Geometry

    func testDistanceFromBottomIsZeroAtTheEnd() {
        // Fully scrolled: offset == contentHeight + bottomInset - containerHeight.
        let d = ChatScrollState.distanceFromBottom(
            contentHeight: 4000, offsetY: 3600, containerHeight: 400, bottomInset: 0)
        XCTAssertEqual(d, 0, accuracy: 0.001)
    }

    func testDistanceFromBottomAtTheTopIsTheWholeOverflow() {
        let d = ChatScrollState.distanceFromBottom(
            contentHeight: 1000, offsetY: 0, containerHeight: 400, bottomInset: 0)
        XCTAssertEqual(d, 600, accuracy: 0.001)
    }

    func testBottomInsetIsPartOfTheScrollableRange() {
        // A scroll-edge effect / safe-area inset extends how far down the view
        // can go; ignoring it reports a permanent residual distance and the
        // correction rule would then fire forever.
        let d = ChatScrollState.distanceFromBottom(
            contentHeight: 1000, offsetY: 620, containerHeight: 400, bottomInset: 20)
        XCTAssertEqual(d, 0, accuracy: 0.001)
    }

    func testContentShorterThanTheViewportIsAlwaysAtTheBottom() {
        let d = ChatScrollState.distanceFromBottom(
            contentHeight: 120, offsetY: 0, containerHeight: 400, bottomInset: 0)
        XCTAssertLessThanOrEqual(d, 0)
    }

    // MARK: - Constants

    func testToleranceIsForgivingEnoughToLandOnAndTightEnoughToCorrect() {
        // Re-engage tolerance is about two lines of text: land anywhere in that
        // band and following resumes. The correction slack is sub-point noise —
        // anything the eye can see as a clipped line gets snapped back.
        XCTAssertGreaterThanOrEqual(ChatScrollState.bottomTolerance, 24)
        XCTAssertLessThanOrEqual(ChatScrollState.bottomTolerance, 48)
        XCTAssertLessThan(ChatScrollState.correctionSlack, ChatScrollState.bottomTolerance)
    }
}
