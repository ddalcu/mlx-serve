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
        let t2i = PromptRewriter.image(text: "a fox", editing: false, groups: ImagePromptExamples.textToImage, denseText: false)
        XCTAssertTrue(t2i.system.contains(ImagePromptExamples.textToImage[0].examples[0].body))
        XCTAssertFalse(t2i.system.contains("instruction"))

        let edit = PromptRewriter.image(text: "make it night", editing: true, groups: ImagePromptExamples.genericEdit, denseText: false)
        XCTAssertTrue(edit.system.contains(ImagePromptExamples.genericEdit[0].examples[0].body))
        XCTAssertTrue(edit.system.contains("instruction"))
        XCTAssertTrue(edit.user.contains("make it night"))
    }

    func testImageRewriteTakesTwoExamplesPerGroup() {
        let groups = ImagePromptExamples.mageFlowEdit
        let r = PromptRewriter.image(text: "x", editing: true, groups: groups, denseText: false)
        let g = groups[0].examples
        XCTAssertTrue(r.system.contains(g[1].body))
        XCTAssertFalse(r.system.contains(g[2].body))
    }

    /// A model that renders dense text gets a quoted LAYOUT, never "one or two
    /// sentences": a rewrite that drops the exact words is what comes back as
    /// gibberish lettering.
    func testDenseTextRewriteAsksForAQuotedLayout() {
        let groups = ImageModelPreset.qwenImage21_8bit.promptExamples(editing: false)
        let r = PromptRewriter.image(text: "an infographic about our app", editing: false, groups: groups, denseText: true)
        XCTAssertTrue(r.system.contains("double quotation marks"))
        XCTAssertTrue(r.system.contains("There is no other text in the image."))
        XCTAssertTrue(r.system.contains(ImagePromptExamples.infographics.examples[0].body))
        XCTAssertTrue(r.user.contains("an infographic about our app"))
    }

    /// Every model keeps the user's text word for word; only a dense-text model is asked to lay it out.
    func testOtherModelsKeepTheirTextWithoutALayout() {
        let r = PromptRewriter.image(text: "a sign reading \"OPEN\"", editing: false,
                                     groups: ImagePromptExamples.textToImage, denseText: false)
        XCTAssertTrue(r.system.contains("word for word"))
        XCTAssertFalse(r.system.contains("There is no other text in the image."))
    }

    func testCleanStripsFencesAndQuotes() {
        XCTAssertEqual(PromptRewriter.clean("```\n“A red fox.”\n```"), "A red fox.")
        XCTAssertEqual(PromptRewriter.clean("\"A red fox.\""), "A red fox.")
    }

    /// Only a WRAPPING pair is the model's decoration; a prompt that ends on quoted text keeps its closing quote.
    func testCleanKeepsTheQuotesOfTextInThePrompt() {
        let sign = "A bakery sign reading \"MORNING LOAF\""
        XCTAssertEqual(PromptRewriter.clean(sign), sign)
        let both = "\"SALE\" in red above \"NEW\""
        XCTAssertEqual(PromptRewriter.clean(both), both)
    }

    // MARK: - First frame

    private let png = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3])

    /// A clip with a first frame carries the picture and the rule that the prompt must describe it.
    func testAFirstFrameRidesTheRequestWithItsRule() {
        let clip = PromptRewriter.video(text: "they dance", format: .h3Base, seconds: 10, firstFrame: png)
        let plan = PromptRewriter.storyboard(idea: "they dance", format: .h3Base, totalSeconds: 30,
                                             shotSeconds: 5...10, firstFrame: png)
        for r in [clip, plan] {
            XCTAssertEqual(r.firstFrame, png)
            XCTAssertTrue(r.user(seeingImage: true).contains("attached picture"))
        }
        XCTAssertTrue(plan.user(seeingImage: true).contains("Shot 1 opens on it"))
        XCTAssertFalse(clip.user(seeingImage: true).contains("Shot 1"))
    }

    /// A model that cannot see the picture is never told about it.
    func testTheRuleStaysOutWhenTheModelCannotSeeThePicture() {
        let r = PromptRewriter.video(text: "they dance", format: .h3Base, seconds: 10, firstFrame: png)
        XCTAssertFalse(r.user(seeingImage: false).contains("attached picture"))
        XCTAssertEqual(r.user(seeingImage: false), PromptRewriter.video(text: "they dance", format: .h3Base, seconds: 10).user)
    }

    /// Reference pictures ride along in order, with the rule that ties each to its `<Picture N>`.
    func testReferencePicturesRideTheRequestInOrderWithTheirRule() {
        let jpg = Data([0xFF, 0xD8, 0xFF, 9])
        let r = PromptRewriter.video(text: "<Picture 1> meets <Picture 2>", format: .h3Reference, seconds: 10,
                                     references: [png, jpg])
        XCTAssertEqual(r.images, [png, jpg])
        XCTAssertTrue(r.user(seeingImage: true).contains("<Picture 2>"))
        XCTAssertFalse(r.user(seeingImage: true).contains("first frame"))
        XCTAssertEqual(r.user(seeingImage: false), r.user)
        // A first frame goes ahead of the references, each with its own rule.
        let both = PromptRewriter.video(text: "x", format: .h3Reference, seconds: 10, firstFrame: png, references: [jpg])
        XCTAssertEqual(both.images, [png, jpg])
        XCTAssertTrue(both.user(seeingImage: true).contains("attached picture"))
        XCTAssertTrue(both.user(seeingImage: true).contains("<Picture 1>"))
    }

    func testNoFirstFrameMeansNoPicture() {
        let r = PromptRewriter.video(text: "they dance", format: .h3Base, seconds: 10)
        XCTAssertNil(r.firstFrame)
        XCTAssertEqual(r.user(seeingImage: true), r.user)
    }
}
