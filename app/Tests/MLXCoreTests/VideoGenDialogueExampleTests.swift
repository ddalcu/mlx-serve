import XCTest

/// LTX only generates speech when the prompt contains quoted dialogue (the
/// official prompting guide: put the spoken words between quotation marks,
/// with acting directions between phrases). Without it the soundtrack is
/// ambient noise — the exact "audio works but nobody talks" report. The
/// video pane's Examples menu and placeholder are the only prompting
/// guidance users see, so they must demonstrate the dialogue format.
///
/// MLXCore is an executable target (no @testable import), and these are UI
/// string constants — so this test reads the view source directly, the
/// InfoPlistTests pattern.
final class VideoGenDialogueExampleTests: XCTestCase {
    private func videoGenViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MLXCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // Tests → app root
            .appendingPathComponent("Sources/MLXServe/Views/VideoGenView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testExamplesIncludeQuotedDialoguePrompt() throws {
        let source = try videoGenViewSource()
        // An example body with spoken words in (escaped) quotes — the marker
        // of the dialogue prompt format. `says` + `\"` must appear in the
        // examplePrompts block.
        let examplesBlock = try XCTUnwrap(
            source.range(of: "examplePrompts").map { String(source[$0.lowerBound...]) },
            "VideoGenView lost its examplePrompts list"
        )
        XCTAssertTrue(
            examplesBlock.contains("says") && examplesBlock.contains("\\\""),
            "No example prompt demonstrates quoted dialogue — without one, users never learn the format that makes LTX characters speak"
        )
    }

    func testPlaceholderMentionsDialogueInQuotes() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(
            source.contains("dialogue in quotes"),
            "The prompt placeholder should tell users to put spoken dialogue in quotes — it's the only way LTX generates speech"
        )
    }
}
