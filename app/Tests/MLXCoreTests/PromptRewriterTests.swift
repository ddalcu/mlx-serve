import XCTest
@testable import MLXCore

/// Enhance on the Image and Video panes asks the chat model for the shape the
/// selected model was trained on: an H3 prompt that loses its labels is a
/// worse prompt than the one the user typed.
final class PromptRewriterTests: XCTestCase {

    func testVideoRewriteNamesTheFormatsLabelsInOrder() {
        for format in [VideoPromptFormat.h3Base, .h3Reference] {
            let r = PromptRewriter.video(text: "a dog runs", format: format, seconds: 5)
            let labels = H3PromptExamples.sections(for: format)
            XCTAssertTrue(r.system.contains(labels.joined(separator: ", ")), "\(format)")
            XCTAssertTrue(r.system.contains(H3PromptExamples.examples(for: format)[0].body))
            XCTAssertTrue(r.user.contains("a dog runs"))
        }
    }

    func testLtxRewriteIsProseWithQuotedDialogueAndNoLabels() {
        let r = PromptRewriter.video(text: "a dog runs", format: .ltx, seconds: 5)
        XCTAssertTrue(r.system.contains("double quotes"))
        XCTAssertFalse(r.system.contains("integrated_multimodal_description:"))
    }

    func testVideoRewriteCapsTheActionAtTheClipLength() {
        for seconds in [2, 8] {
            let r = PromptRewriter.video(text: "a dog runs", format: .ltx, seconds: seconds)
            XCTAssertTrue(r.user.contains("\(seconds) seconds"), "\(seconds)")
            XCTAssertTrue(r.user.contains("a dog runs"))
        }
    }

    func testImageRewriteShowsTheModesExamplesOnly() {
        let t2i = PromptRewriter.image(text: "a fox", editing: false, groups: ImagePromptExamples.textToImage)
        XCTAssertTrue(t2i.system.contains(ImagePromptExamples.textToImage[0].examples[0].body))
        XCTAssertFalse(t2i.system.contains("instruction"))

        let edit = PromptRewriter.image(text: "make it night", editing: true, groups: ImagePromptExamples.genericEdit)
        XCTAssertTrue(edit.system.contains(ImagePromptExamples.genericEdit[0].examples[0].body))
        XCTAssertTrue(edit.system.contains("instruction"))
        XCTAssertTrue(edit.user.contains("make it night"))
    }

    func testImageRewriteTakesTwoExamplesPerGroup() {
        let groups = ImagePromptExamples.mageFlowEdit
        let r = PromptRewriter.image(text: "x", editing: true, groups: groups)
        let g = groups[0].examples
        XCTAssertTrue(r.system.contains(g[1].body))
        XCTAssertFalse(r.system.contains(g[2].body))
    }

    func testCleanStripsFencesAndQuotes() {
        XCTAssertEqual(PromptRewriter.clean("```\n“A red fox.”\n```"), "A red fox.")
    }
}
