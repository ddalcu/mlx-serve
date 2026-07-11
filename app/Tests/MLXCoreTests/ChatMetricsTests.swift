import XCTest
@testable import MLXCore

/// Pins the chat column's alignment contract. These numbers live in separate
/// views (transcript, context monitor, composer, bubbles) and drifted when
/// inlined — 16/12/12 gutters left the input pill and token bar 4pt left of
/// the message bubbles, with chip rows carrying +4 compensation paddings.
/// `ChatMetrics` is the single source of truth; this test keeps the
/// relationships from regressing.
final class ChatMetricsTests: XCTestCase {

    func testChatColumnGutter() {
        XCTAssertEqual(ChatMetrics.gutter, 16)
    }

    func testStatsCaptionAlignsWithBubbleTextColumn() {
        // The "N+M tokens" caption under a reply indents by the bubble's inner
        // padding so it lines up with the text, not the bubble edge.
        XCTAssertEqual(ChatMetrics.statsIndent, ChatMetrics.bubblePaddingH)
    }

    func testComposerControlsMatchPillRestHeight() {
        // Attach / mic / send frames equal the input pill's single-line height,
        // so a bottom-aligned HStack lines everything up with no nudge paddings.
        XCTAssertEqual(ChatMetrics.composerControlSize, ChatMetrics.composerMinHeight)
    }

    func testModePillGeometryIsPinnedAndShared() {
        // Think / Agent / MCP capsules draw from ONE geometry: an explicit
        // height and a fixed icon slot — SF symbols at the same point size
        // render different intrinsic sizes (brain vs wrench vs puzzle), so
        // padding-derived heights made the three pills subtly unequal.
        XCTAssertEqual(ChatMetrics.togglePillPaddingH, 8)
        XCTAssertEqual(ChatMetrics.togglePillHeight, 24)
        XCTAssertEqual(ChatMetrics.togglePillIconSize, 15)
        // The icon slot must fit inside the pill.
        XCTAssertLessThan(ChatMetrics.togglePillIconSize, ChatMetrics.togglePillHeight)
    }

    func testModePillClusterOwnsItsSpacing() {
        // The three pills ride ONE ToolbarItem with an explicit HStack gap, so
        // in-cluster spacing is ours — uniform on both sides of Agent — rather
        // than whatever the system puts between separate toolbar items.
        // 8 (down from 12): wide gaps push the cluster into toolbar
        // compression on narrow windows, truncating "MCP" to "…".
        XCTAssertEqual(ChatMetrics.togglePillSpacing, 8)
    }
}
