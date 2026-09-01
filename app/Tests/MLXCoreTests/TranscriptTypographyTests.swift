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

    /// The default (Settings ▸ Interface ▸ Text Size = Default) reads at
    /// 14pt — bumped down from the original 16pt (2026-08-31, felt too large
    /// at normal reading distance); the size is a user setting now
    /// (`ChatTextSize`), and `.medium` is what nobody who never opens that
    /// picker gets.
    func testTheTranscriptReadsAtFourteenPointsByDefault() {
        XCTAssertEqual(ChatTextSize.medium.proseSize, 14)
        XCTAssertEqual(ChatMetrics.transcriptFontSize, ChatTextSize.medium.proseSize)
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

    /// The markdown render cache is keyed on the text-size setting: identical
    /// bytes rendered at Small and then Extra Large must come back at two
    /// sizes, not as a stale cache hit built at the old size.
    func testTheRenderedTranscriptFollowsTheTextSizeSetting() {
        let original = UserDefaults.standard.object(forKey: InterfacePrefKey.textSize)
        defer {
            if let original { UserDefaults.standard.set(original, forKey: InterfacePrefKey.textSize) }
            else { UserDefaults.standard.removeObject(forKey: InterfacePrefKey.textSize) }
        }
        UserDefaults.standard.set(ChatTextSize.small.rawValue, forKey: InterfacePrefKey.textSize)
        let small = MarkdownText.attributedString(for: "same bytes, two sizes")
        UserDefaults.standard.set(ChatTextSize.xlarge.rawValue, forKey: InterfacePrefKey.textSize)
        let large = MarkdownText.attributedString(for: "same bytes, two sizes")
        XCTAssertEqual((small.attribute(.font, at: 0, effectiveRange: nil) as? NSFont)?.pointSize,
                       ChatTextSize.small.proseSize)
        XCTAssertEqual((large.attribute(.font, at: 0, effectiveRange: nil) as? NSFont)?.pointSize,
                       ChatTextSize.xlarge.proseSize)
    }

    /// Prose wraps at `ChatMetrics.proseMeasure` (a paragraph tailIndent);
    /// code blocks — and tables, which share the no-indent default — keep the
    /// full column. The measure must never become a frame cap on wide content.
    func testProseWrapsEarlyWhileCodeKeepsTheFullColumn() {
        let rendered = MarkdownText.attributedString(for: "flowing prose\n\n```\nwide code\n```")
        let text = rendered.string as NSString
        let proseStyle = rendered.attribute(.paragraphStyle, at: text.range(of: "flowing").location,
                                            effectiveRange: nil) as? NSParagraphStyle
        let codeStyle = rendered.attribute(.paragraphStyle, at: text.range(of: "wide code").location,
                                           effectiveRange: nil) as? NSParagraphStyle
        XCTAssertEqual(proseStyle?.tailIndent, ChatMetrics.proseMeasure)
        // Code keeps its own trailing inset (≤ 0 = relative to the trailing
        // margin, full column) — a POSITIVE tailIndent is the prose measure
        // leaking onto wide content.
        XCTAssertLessThanOrEqual(codeStyle?.tailIndent ?? 0, 0, "code must not inherit the prose measure")
        XCTAssertEqual(proseStyle?.lineHeightMultiple, ChatMetrics.proseLineHeightMultiple)
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
