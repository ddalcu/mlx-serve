import XCTest
@testable import MLXCore

/// A video model only generates SPEECH when the prompt asks for it in that
/// model's own notation, and the pane's Examples menu plus placeholder are the
/// only prompting guidance users see — so they must demonstrate it.
///
/// The two engines spell it differently, which is the whole reason this is a
/// per-format guard now:
///   LTX  — the spoken words between quotation marks, acting directions
///          between phrases (its official prompting guide). Without it the
///          soundtrack is ambient noise, the exact "audio works but nobody
///          talks" report.
///   H3   — `<d>[English] …</d>` spoken by a numbered speaker `(S1)`, inside
///          `integrated_multimodal_description:` / `detailed_description:`.
///
/// These strings used to live in `VideoGenView` and this test read the view
/// source; they moved to `H3PromptExamples` when the guidance became
/// per-backend, so it reads the API instead — a source scan would now pass
/// vacuously off unrelated text in the view.
final class VideoGenDialogueExampleTests: XCTestCase {

    func testLtxExamplesDemonstrateQuotedDialogue() {
        let spoken = H3PromptExamples.ltx.filter { $0.body.contains("says") && $0.body.contains("\"") }
        XCTAssertFalse(spoken.isEmpty,
                       "No LTX example demonstrates quoted dialogue — without one, users never learn the format that makes LTX characters speak")
    }

    func testLtxPlaceholderMentionsDialogueInQuotes() {
        XCTAssertTrue(H3PromptExamples.placeholder(for: .ltx).contains("dialogue in quotes"),
                      "The LTX placeholder should tell users to put spoken dialogue in quotes — it's the only way LTX generates speech")
    }

    func testH3ExamplesDemonstrateItsOwnDialogueMarkup() {
        // H3 does NOT take quoted dialogue: speech rides `<d>[Language] …</d>`
        // with a speaker id. An example rewritten into LTX-style quotes would
        // teach the wrong notation for the model it is shown under.
        let spoken = H3PromptExamples.h3Base.filter { $0.body.contains("<d>[English]") && $0.body.contains("(S1)") }
        XCTAssertFalse(spoken.isEmpty, "No H3 example demonstrates <d>[English] …</d> dialogue with a speaker id")
    }
}
