import XCTest
@testable import MLXCore

/// The transcript's reading size is ONE number, and the typeface is the system
/// font.
final class TranscriptTypographyTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testTheTranscriptReadsAtSixteenPoints() {
        XCTAssertEqual(ChatMetrics.transcriptFontSize, 16)
    }

    /// Headings scale FROM the body size. As three literals (18/16/14) raising
    /// the body to 16 would have left an h3 smaller than the paragraph under
    /// it and an h2 identical to it — the hierarchy silently flattening
    /// rather than breaking.
    func testHeadingsStayAboveTheBodySize() {
        let base = ChatMetrics.transcriptFontSize
        let chat = try? source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertEqual(chat?.contains("level == 1 ? base + 5 : level == 2 ? base + 3 : base + 1"), true,
                       "heading sizes must derive from the body size, not be restated")
        for bump in [5, 3, 1] {
            XCTAssertGreaterThan(base + CGFloat(bump), base)
        }
    }

    /// Code is monospaced, and monospaced glyphs run wide — matching the prose
    /// size makes a fenced block look bigger than the sentence introducing it.
    func testCodeIsSmallerThanProse() {
        XCTAssertLessThan(ChatMetrics.transcriptCodeFontSize, ChatMetrics.transcriptFontSize)
    }

    /// Every transcript size reads the constants. A literal beside them is how
    /// four of five call sites get changed.
    func testNoTranscriptFontSizeIsHardCoded() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")
        // The renderer's own font construction only — other views legitimately
        // use small sizes for badges and captions.
        guard let start = chat.range(of: "private static func buildAttributedString") else {
            return XCTFail("the markdown renderer moved — update this audit")
        }
        let renderer = String(chat[start.upperBound...])
        for literal in ["systemFont(ofSize: 13", "monospacedSystemFont(ofSize: 12"] {
            XCTAssertFalse(renderer.contains(literal), """
                The transcript renderer still hard-codes `\(literal)`. Sizes come \
                from ChatMetrics.transcriptFontSize / .transcriptCodeFontSize.
                """)
        }
    }

    /// SF Pro is the macOS system font, so the app gets it by asking for the
    /// system font. Naming it explicitly would be the same typeface with none
    /// of the optical sizing or weight mapping — and would break the moment
    /// Apple ships a new system face.
    func testTheAppNeverNamesAFontFamilyByString() throws {
        for path in ["Sources/MLXServe/Views/ChatView.swift",
                     "Sources/MLXServe/Views/ChatMetrics.swift",
                     "Sources/MLXServe/Views/NewTaskSheet.swift",
                     "Sources/MLXServe/Views/TasksView.swift",
                     "Sources/MLXServe/Views/AgentsWindow.swift"] {
            // Comments explain why NOT to name a font, so scanning them finds
            // the ban in the prose that states it.
            let text = try source(path)
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
                .joined(separator: "\n")
            XCTAssertFalse(text.contains(".custom("), """
                \(path) names a font family by string. The system font IS SF Pro; \
                a named copy loses optical sizing and weight mapping.
                """)
            XCTAssertFalse(text.contains("NSFont(name:"), "\(path) constructs a font by name")
        }
    }
}
