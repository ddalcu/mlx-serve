import XCTest
import AppKit
@testable import MLXCore

/// The code block renders through ONE attributed string, not one SwiftUI view
/// per line.
///
/// That is a performance decision with a correctness surface, so both are
/// pinned here. The old renderer built a `Text` for every line, so a 300-line
/// block put hundreds of nodes in SwiftUI's attribute graph — and a streaming
/// reply re-created all of them on every token. Measured live: 103% of one core
/// in the UI process, with `AG::Graph::propagate_dirty` the hottest symbol in
/// the sample and our own lexer nowhere in it. Colouring the whole block by
/// NSRange (which is what `SyntaxSpan`'s UTF-16 offsets exist for) makes it one
/// node.
final class CodeBlockTextTests: XCTestCase {

    private func kinds(in s: NSAttributedString) -> [(String, NSColor?)] {
        var out: [(String, NSColor?)] = []
        s.enumerateAttribute(.foregroundColor, in: NSRange(location: 0, length: s.length)) { v, r, _ in
            let text = (s.string as NSString).substring(with: r)
            out.append((text, v as? NSColor))
        }
        return out
    }

    // MARK: - Content is the source, verbatim

    func testTheStringIsTheCodeExactly() {
        let code = "// note\nfunc f(x: Int) -> String {\n    return \"a\\\"b\"  // t\n}"
        XCTAssertEqual(CodeBlockText.code(code, language: .swift).string, code,
                       "a renderer that drops or duplicates a character shows code nobody wrote")
        XCTAssertEqual(CodeBlockText.code(code, language: nil).string, code)
    }

    func testEmptyCodeIsAnEmptyString() {
        XCTAssertEqual(CodeBlockText.code("", language: .swift).string, "")
    }

    // MARK: - Colour lands on the right characters

    func testEverySpanColoursExactlyItsOwnText() {
        let code = "let x = \"hi\" // note"
        let s = CodeBlockText.code(code, language: .swift)
        let ns = code as NSString
        for span in SyntaxHighlighter.spans(code, language: .swift) {
            let range = NSRange(location: span.start, length: span.length)
            var covered = 0
            s.enumerateAttribute(.foregroundColor, in: range) { v, r, _ in
                XCTAssertEqual(v as? NSColor, CodeTheme.nsColor(for: span.kind),
                               "\(ns.substring(with: r)) should be \(span.kind)")
                covered += r.length
            }
            XCTAssertEqual(covered, span.length)
        }
    }

    func testAMultiLineCommentStaysColouredAcrossItsNewlines() {
        // Free with one attributed string, and the reason per-line splitting
        // existed at all — it had to re-emit the span on every row it crossed.
        let s = CodeBlockText.code("/* one\n   two */\nx", language: .cFamily)
        let comment = CodeTheme.nsColor(for: .comment)
        for i in 0..<16 {
            XCTAssertEqual(s.attribute(.foregroundColor, at: i, effectiveRange: nil) as? NSColor, comment)
        }
        XCTAssertEqual(s.attribute(.foregroundColor, at: 17, effectiveRange: nil) as? NSColor,
                       CodeTheme.nsColor(for: nil))
    }

    func testNonAsciiDoesNotShiftColourOffItsToken() {
        // The whole reason spans are UTF-16: a Character offset is short by one
        // for every emoji before it and mis-paints everything after.
        let code = "let s = \"🚀🚀\"\nlet k = 42"
        let s = CodeBlockText.code(code, language: .swift)
        let ns = code as NSString
        let numberRange = ns.range(of: "42")
        XCTAssertEqual(s.attribute(.foregroundColor, at: numberRange.location, effectiveRange: nil) as? NSColor,
                       CodeTheme.nsColor(for: .number))
    }

    func testNoLanguageLeavesEverythingPlain() {
        let s = CodeBlockText.code("func f() { return 1 }", language: nil)
        XCTAssertEqual(kinds(in: s).count, 1)
        XCTAssertEqual(kinds(in: s)[0].1, CodeTheme.nsColor(for: nil))
    }

    // MARK: - Size without laying out

    /// TextKit's own answer, which the arithmetic has to reproduce exactly.
    private func laidOutSize(_ s: NSAttributedString) -> NSSize {
        let storage = NSTextStorage(attributedString: s)
        let manager = NSLayoutManager()
        let container = NSTextContainer(size: NSSize(width: CGFloat.greatestFiniteMagnitude,
                                                     height: CGFloat.greatestFiniteMagnitude))
        container.lineFragmentPadding = 0
        manager.addTextContainer(container)
        storage.addLayoutManager(manager)
        manager.ensureLayout(for: container)
        let used = manager.usedRect(for: container)
        return NSSize(width: ceil(used.width), height: ceil(used.height))
    }

    /// Laying a 300-line block out to ask how tall it is costs 6.4 ms, and it
    /// happens on every streamed flush — for lines that are mostly off screen.
    /// A monospaced block that never wraps has a size that is pure arithmetic,
    /// so this is the check that the arithmetic IS the layout: if it drifts, the
    /// block clips its last lines or leaves a gap under them.
    func testTheComputedSizeIsExactlyWhatTextKitWouldLayOut() {
        for n in [1, 2, 7, 50, 150, 300] {
            let src = (0..<n).map { "    let value\($0) = compute(\($0), name: \"item-\($0)\")" }
                .joined(separator: "\n")
            let computed = try? XCTUnwrap(CodeBlockText.measuredSize(of: src))
            XCTAssertEqual(computed, laidOutSize(CodeBlockText.code(src, language: .swift)),
                           "\(n) lines")
        }
    }

    func testBlankLinesMeasureLikeTextKitToo() {
        for src in ["\n", "a\n\nb", "\n\n\n"] {
            XCTAssertEqual(CodeBlockText.measuredSize(of: src),
                           laidOutSize(CodeBlockText.code(src, language: .swift)),
                           src.debugDescription)
        }
        // Empty storage is sized from TextKit's extra line fragment, not from
        // the font's line height (14 vs 15), so it declines and gets measured —
        // which is free for an empty block.
        XCTAssertNil(CodeBlockText.measuredSize(of: ""))
    }

    /// The arithmetic is only exact while every glyph is one monospaced advance
    /// wide. A tab snaps to a tab stop and a wide glyph is not one advance, so
    /// those decline and the caller measures for real — a guessed width clips
    /// the right-hand end of the longest line.
    func testAnythingThatIsNotOneAdvancePerCharacterDeclines() {
        XCTAssertNil(CodeBlockText.measuredSize(of: "let a = 1\n\tlet b = 2"), "tab")
        XCTAssertNil(CodeBlockText.measuredSize(of: "let s = \"🚀\""), "emoji")
        XCTAssertNil(CodeBlockText.measuredSize(of: "let s = \"日本語\""), "wide glyphs")
        XCTAssertNotNil(CodeBlockText.measuredSize(of: "let a = 1"))
    }

    // MARK: - Incremental streaming update

    func testAnAppendedTailIsTheOnlyThingReportedAsChanged() {
        let a = CodeBlockText.code("let a = 1\nlet b = 2", language: .swift)
        let b = CodeBlockText.code("let a = 1\nlet b = 2\nlet c = 3", language: .swift)
        XCTAssertEqual(CodeBlockText.changedSuffix(from: a, to: b), a.length,
                       "a pure append must not re-lay out a single unchanged line")
    }

    func testIdenticalStringsReportNoChange() {
        let a = CodeBlockText.code("let a = 1", language: .swift)
        let b = CodeBlockText.code("let a = 1", language: .swift)
        XCTAssertEqual(CodeBlockText.changedSuffix(from: a, to: b), a.length)
    }

    /// The next token can RE-COLOUR text already on screen: an identifier is a
    /// call only once its `(` arrives, so `foo` is plain in one frame and a
    /// function in the next while the characters are byte-identical. A diff that
    /// compared only characters would report "nothing before the paren changed"
    /// and leave `foo` painted plain for the rest of the reply.
    func testRecolouringUnderIdenticalCharactersIsCaught() {
        let before = CodeBlockText.code("let x = foo", language: .swift)
        let after = CodeBlockText.code("let x = foo(", language: .swift)
        XCTAssertNotEqual(before.attribute(.foregroundColor, at: 8, effectiveRange: nil) as? NSColor,
                          after.attribute(.foregroundColor, at: 8, effectiveRange: nil) as? NSColor,
                          "precondition: the identifier really is re-coloured")

        let start = CodeBlockText.changedSuffix(from: before, to: after) ?? 0
        XCTAssertLessThanOrEqual(start, 8, "the changed range must reach back over the re-coloured identifier")

        let patched = NSMutableAttributedString(attributedString: before)
        patched.replaceCharacters(
            in: NSRange(location: start, length: patched.length - start),
            with: after.attributedSubstring(from: NSRange(location: start, length: after.length - start)))
        XCTAssertTrue(patched.isEqual(to: after), "the patched storage must equal a full rebuild")
    }

    func testShrinkingAndUnrelatedEditsStillPatchToTheRightResult() {
        for (before, after) in [("let a = 1\nlet b = 2", "let a = 1"),
                                ("func f() {}", "class C {}"),
                                ("", "let a = 1"),
                                ("let a = 1", "")] {
            let a = CodeBlockText.code(before, language: .swift)
            let b = CodeBlockText.code(after, language: .swift)
            let patched = NSMutableAttributedString(attributedString: a)
            let start = CodeBlockText.changedSuffix(from: a, to: b) ?? 0
            patched.replaceCharacters(
                in: NSRange(location: start, length: patched.length - start),
                with: b.attributedSubstring(from: NSRange(location: start, length: b.length - start)))
            XCTAssertTrue(patched.isEqual(to: b), "\(before.debugDescription) → \(after.debugDescription)")
        }
    }

    // MARK: - Class guard

    /// The defect was never in the lexer; it was one view per line. A `ForEach`
    /// over lines reintroduces it exactly, and nothing would fail — the block
    /// would render correctly and quietly cost a core again while streaming.
    func testTheBlockNeverBuildsAViewPerLine() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Sources/MLXServe/Views/CodeBlockView.swift"),
            encoding: .utf8)
        // Code only — the rule is explained in a comment in that file, and a
        // scan that matched its own explanation could never pass.
        let code = source.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")
        XCTAssertFalse(code.contains("ForEach"),
                       "a per-line ForEach puts one SwiftUI node per line back in the attribute graph")
    }
}
