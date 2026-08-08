import XCTest
@testable import MLXCore

/// The chat column's reading measure.
///
/// The transcript and the composer used to be full-width: on a 1400pt window a
/// line of prose ran the whole way across (~180 characters, roughly twice the
/// measure text is comfortable at), and the input pill grew with the window
/// until it read as a document, not a message field. Both are now capped at ONE
/// width and centred — the same number, because a composer narrower or wider
/// than the answers above it is the seam you can't unsee.
final class ChatColumnMetricsTests: XCTestCase {

    func testWideWindowsGetTheReadingMeasure() {
        XCTAssertEqual(ChatMetrics.columnWidth(available: 1600), ChatMetrics.contentMaxWidth)
        XCTAssertEqual(ChatMetrics.columnWidth(available: 1000), ChatMetrics.contentMaxWidth)
    }

    /// Below the cap the column is the window minus its gutters — never a fixed
    /// width that would clip, and never wider than what it is given.
    func testNarrowWindowsFallBackToTheAvailableWidth() {
        let narrow = ChatMetrics.columnWidth(available: 600)
        XCTAssertEqual(narrow, 600 - ChatMetrics.gutter * 2)
        XCTAssertLessThan(narrow, ChatMetrics.contentMaxWidth)
    }

    /// A window can be reported at zero during layout; a negative frame width is
    /// a crash in some SwiftUI containers and a silently invisible column in the
    /// rest.
    func testDegenerateWidthsStayPositive() {
        for available in [CGFloat(0), 1, 10, ChatMetrics.gutter * 2] {
            XCTAssertGreaterThan(ChatMetrics.columnWidth(available: available), 0,
                                 "available=\(available) produced a non-positive column")
        }
    }

    /// The measure is a reading width, not a hunch: ~70-90 characters at the
    /// chat's body size. Pinned so "make it a bit wider" is a decision someone
    /// makes on purpose.
    func testTheMeasureIsInTheReadableRange() {
        XCTAssertGreaterThanOrEqual(ChatMetrics.contentMaxWidth, 640)
        XCTAssertLessThanOrEqual(ChatMetrics.contentMaxWidth, 820)
    }

    // MARK: - Source audit

    /// One measure, applied by both. A transcript capped without the composer
    /// (or the other way round) is exactly the misalignment this replaced.
    func testTranscriptAndComposerBothUseTheSharedMeasure() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let uses = source.components(separatedBy: "ChatMetrics.contentMaxWidth").count - 1
        XCTAssertGreaterThanOrEqual(uses, 3, """
            The reading measure must be applied to the transcript, the composer \
            AND the empty state's greeting block — found \(uses) use(s).
            """)
    }
}
