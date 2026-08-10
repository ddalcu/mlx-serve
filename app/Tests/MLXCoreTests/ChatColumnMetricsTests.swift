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

    // `ChatMetrics.columnWidth(available:)` and its tests are gone: no view
    // ever called it (the transcript and composer cap with
    // `.frame(maxWidth: contentMaxWidth)` directly), so the tests pinned a
    // helper that existed only to be tested.

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
