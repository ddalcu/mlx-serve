import XCTest
@testable import MLXCore

/// Reading measure for the chat column.
///
/// The measure used to be a fixed 740pt: on a fullscreen window that left
/// roughly 40-50% dead space on either side, because the panel next to the
/// session sidebar is wider than 740/0.8. It is now proportional — 80% of
/// the detail column's measured width, centered — so the column grows with
/// the window instead of stopping at a hardcoded cap. The transcript, the
/// composer, and the empty-state greeting all share the same measure: a
/// composer narrower or wider than the answer above it is the seam you
/// can't unsee.
final class ChatColumnMetricsTests: XCTestCase {

    /// Pinned so "make it a bit wider" is a decision someone makes on
    /// purpose. Below 0.7 it's back to visible dead space; above 0.9 prose
    /// runs close enough to the window edge to lose the "column" feel.
    func testTheMeasureIsProportionalAndInTheReadableBand() {
        XCTAssertGreaterThanOrEqual(ChatMetrics.contentWidthFraction, 0.7)
        XCTAssertLessThanOrEqual(ChatMetrics.contentWidthFraction, 0.9)
    }

    /// The fallback only covers the single frame before `ChatDetailView`
    /// measures its own width — it should still sit in a sane reading-width
    /// range so that frame doesn't flash something absurd.
    /// A wide column keeps prose off the edges; a narrow one (the desktop
    /// pane's chat column) wraps to the panel like the fixed modes do, or the
    /// composer's control row overflows the column on both sides.
    func testANarrowColumnUsesItsFullWidth() {
        UserDefaults.standard.removeObject(forKey: InterfacePrefKey.chatColumn)  // .wide
        XCTAssertEqual(ChatMetrics.proseWidth(panelWidth: 1000), 800)
        XCTAssertEqual(ChatMetrics.proseWidth(panelWidth: 480), 480)
        XCTAssertEqual(ChatMetrics.proseWidth(panelWidth: 0), ChatMetrics.contentFallbackWidth, "unmeasured → fallback")
        XCTAssertLessThan(ChatMetrics.proseWidth(panelWidth: ChatMetrics.narrowColumnWidth),
                          ChatMetrics.narrowColumnWidth, "past the seam the margin comes back")
    }

    /// A narrow column sheds the Start caption and the context pill; the
    /// unmeasured first frame is never compact.
    func testTheComposerCollapsesInANarrowColumn() {
        XCTAssertFalse(ChatMetrics.composerIsCompact(columnWidth: 0))
        XCTAssertFalse(ChatMetrics.composerIsCompact(columnWidth: 900))
        XCTAssertTrue(ChatMetrics.composerIsCompact(columnWidth: 400))
        XCTAssertLessThan(ChatMetrics.compactComposerWidth, ChatMetrics.narrowColumnWidth,
                          "compact kicks in only after the full-width measure already applies")
    }

    func testTheFallbackIsInTheReadableRange() {
        XCTAssertGreaterThanOrEqual(ChatMetrics.contentFallbackWidth, 640)
        XCTAssertLessThanOrEqual(ChatMetrics.contentFallbackWidth, 820)
    }

    // MARK: - Source audit

    /// One measure, applied by all three. A transcript capped without the
    /// composer (or the other way round) is exactly the misalignment this
    /// replaced.
    func testTranscriptComposerAndGreetingAllUseTheSharedMeasure() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let uses = source.components(separatedBy: "maxWidth: contentWidth").count - 1
        XCTAssertGreaterThanOrEqual(uses, 3, """
            The reading measure must be applied to the transcript, the composer \
            AND the empty state's greeting block — found \(uses) use(s).
            """)
    }
}
