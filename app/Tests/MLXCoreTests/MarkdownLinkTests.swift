import XCTest
@testable import MLXCore

/// URLs in assistant messages must be clickable regardless of how the model
/// dressed them up. CommonMark autolinks a bare `http://…`, but models love
/// wrapping URLs in backticks (`` `http://localhost:3000` ``) — a code span
/// that Foundation's markdown parser renders WITHOUT a `.link` attribute. Live
/// symptom: the URL flickers clickable mid-stream (closing backtick not yet
/// arrived → parsed as bare URL) then goes dead when the span completes.
final class MarkdownLinkTests: XCTestCase {

    /// All (.link value, covered text) pairs in the rendered string.
    private func links(in source: String) -> [(url: String, text: String)] {
        let rendered = MarkdownText.attributedString(for: source)
        var found: [(String, String)] = []
        rendered.enumerateAttribute(.link, in: NSRange(location: 0, length: rendered.length)) { value, range, _ in
            guard let value else { return }
            let url = (value as? URL)?.absoluteString ?? String(describing: value)
            found.append((url, (rendered.string as NSString).substring(with: range)))
        }
        return found
    }

    func testBacktickWrappedUrlIsClickable() {
        let found = links(in: "URL: `http://localhost:3000` (or use the LAN IP).")
        XCTAssertEqual(found.map(\.url), ["http://localhost:3000"])
    }

    func testBacktickWrappedUrlInListItemIsClickable() {
        // The exact shape from the live report: a bullet with two code-span URLs.
        let found = links(in: "- URL: `http://localhost:3000` (or `http://192.168.2.61:3000` if accessing from your local network).")
        XCTAssertEqual(found.map(\.url), ["http://localhost:3000", "http://192.168.2.61:3000"])
    }

    func testBareUrlStaysClickable() {
        let found = links(in: "Visit http://localhost:3000 now")
        XCTAssertEqual(found.map(\.url), ["http://localhost:3000"])
    }

    func testUrlInFencedCodeBlockIsClickable() {
        let found = links(in: "```\nserver listening at http://localhost:8080\n```")
        XCTAssertEqual(found.map(\.url), ["http://localhost:8080"])
    }

    func testExplicitMarkdownLinkIsUntouched() {
        let found = links(in: "[the app](http://localhost:3000) is up")
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found[0].url, "http://localhost:3000")
        XCTAssertEqual(found[0].text, "the app")
    }

    func testTrailingPunctuationExcludedFromLink() {
        let found = links(in: "Open (`http://localhost:3000`).")
        XCTAssertEqual(found.map(\.url), ["http://localhost:3000"])
        XCTAssertEqual(found.first?.text, "http://localhost:3000")
    }

    func testFileNamesAreNotLinkified() {
        XCTAssertTrue(links(in: "read `CHANGELOG.md` and run build.sh").isEmpty)
    }
}
