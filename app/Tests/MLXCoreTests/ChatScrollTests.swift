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

    func testATranscriptOpensOnTheNewestMessage() {
        // Restoring a saved chat used to land at the OLDEST message: the scroll
        // view had no bottom anchor and nothing scrolled it, because the only
        // triggers were message-count and last-content changes that never fire
        // on a conversation that isn't moving.
        var s = ChatScrollState()
        XCTAssertEqual(s.handle(.transcriptShown), .toBottom(animated: false))
        XCTAssertTrue(s.isPinnedToBottom)
    }

    func testSwitchingSessionsReturnsToTheBottomAndRe_engages() {
        // ChatDetailView is REUSED across tabs, so scroll state is per-view and
        // leaked between conversations: leaving tab A scrolled up made tab B
        // open unpinned at an arbitrary offset.
        var s = ChatScrollState()
        _ = s.handle(.driverChanged(.user))
        _ = s.handle(.geometryChanged(distanceFromBottom: 800))
        XCTAssertFalse(s.isPinnedToBottom)

        XCTAssertEqual(s.handle(.transcriptShown), .toBottom(animated: false))
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
