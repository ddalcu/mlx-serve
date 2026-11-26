import XCTest
@testable import MLXCore

/// The welcome's "What's new" panel: which release it asks GitHub for, and which lines of that
/// release's body (its CHANGELOG entry, plus whatever was pasted on the page) it shows.
final class ReleaseNotesTests: XCTestCase {
    func testAsksForTheRunningVersionsTag() {
        XCTAssertEqual(ReleaseNotes.url(forVersion: "26.11.4").absoluteString,
                       "https://api.github.com/repos/\(UpdateChecker.repo)/releases/tags/v26.11.4")
    }

    /// Bar: bullets lose their marker and keep their markdown, plain lines stay; headings, HTML
    /// (pasted screenshots), rules and blank lines go.
    func testKeepsBulletsAndPlainLinesOnly() {
        let body = """
        ##  Inkling 2 Mini

        - **Inkling 2 Mini.** Runs from its released checkpoint.

        - **Fixes.** `--os-reserve-gib` takes effect again.
        https://huggingface.co/ddalcu/Inkling-2-Mini-MLX-8bit

        <img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/a" />
        ---
        * Another bullet
        """
        XCTAssertEqual(ReleaseNotes.items(fromMarkdown: body), [
            "**Inkling 2 Mini.** Runs from its released checkpoint.",
            "**Fixes.** `--os-reserve-gib` takes effect again.",
            "https://huggingface.co/ddalcu/Inkling-2-Mini-MLX-8bit",
            "Another bullet",
        ])
    }

    func testReadsTheBodyAndPageOfARelease() throws {
        let json = #"{"tag_name":"v26.11.4","html_url":"https://github.com/ddalcu/mlx-serve/releases/tag/v26.11.4","body":"- **A.** b"}"#
        let notes = try XCTUnwrap(ReleaseNotes.parse(releaseJSON: Data(json.utf8)))
        XCTAssertEqual(notes.version, "26.11.4")
        XCTAssertEqual(notes.items, ["**A.** b"])
        XCTAssertEqual(notes.page?.absoluteString, "https://github.com/ddalcu/mlx-serve/releases/tag/v26.11.4")
    }

    func testAReleaseWithoutABodyHasNoItems() throws {
        let notes = try XCTUnwrap(ReleaseNotes.parse(releaseJSON: Data(#"{"tag_name":"v26.11.4"}"#.utf8)))
        XCTAssertEqual(notes.items, [])
        XCTAssertNil(ReleaseNotes.parse(releaseJSON: Data("not json".utf8)))
    }
}
