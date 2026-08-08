import XCTest
@testable import MLXCore

/// The composer's create mode: the chat surface driven directly, instead of
/// throwing the user out of the chat and into the Create pane's form.
final class ChatCreateModeTests: XCTestCase {

    /// Persisted as a raw string, like the per-chat tool switches: a mode
    /// retired in a later build must leave an unknown name behind, not fail the
    /// whole session's decode.
    func testUnknownAndAbsentModesDecodeToNothing() {
        XCTAssertNil(ChatCreateMode.from(nil))
        XCTAssertNil(ChatCreateMode.from("hologram"))
        XCTAssertEqual(ChatCreateMode.from("image"), .image)
    }

    /// Every mode points at a generator the app actually offers — the shared
    /// `GenExperiment` catalogue, so a mode can't name a page that doesn't exist.
    func testEveryModeMapsToAnOfferedGenerator() {
        for mode in ChatCreateMode.allCases {
            XCTAssertTrue(GenExperiment.allCases.contains(mode.experiment),
                          "\(mode) points at a generator that isn't in the catalogue")
        }
    }

    /// The banner shows the MODEL doing the work, not a paragraph explaining
    /// what the mode is — the model's name is the fact that changes with the
    /// asset type and the only one the user can act on. What the copy still owes
    /// is a placeholder that isn't "Ask me anything" and a button that doesn't
    /// say Send: nothing is being sent to anyone.
    func testEveryModeAsksForTheRightInputAndDoesNotSayMessage() {
        for mode in ChatCreateMode.allCases {
            XCTAssertFalse(mode.title.isEmpty)
            XCTAssertFalse(mode.placeholder.isEmpty)
            XCTAssertFalse(mode.placeholder.localizedCaseInsensitiveContains("ask me"),
                           "\(mode) must not reuse the chat placeholder")
            XCTAssertEqual(mode.actionVerb, "Generate")
        }
    }

    // MARK: - Pressing Generate without the model

    /// You may always TYPE. A missing model is answered after the fact — offer
    /// the download, hold the prompt, run it when the bytes land — rather than
    /// blocking the composer until the user goes and fetches something.
    func testAMissingModelOffersTheDownloadInsteadOfBlocking() {
        XCTAssertEqual(ChatCreateSend.decide(prompt: "a red fox", modelReady: false, busy: false),
                       .offerDownload)
        XCTAssertEqual(ChatCreateSend.decide(prompt: "a red fox", modelReady: true, busy: false),
                       .generate)
    }

    /// An empty prompt and an in-flight generation are both no-ops — never a
    /// download offer for a prompt that doesn't exist.
    func testNothingHappensWithoutAPromptOrWhileBusy() {
        XCTAssertEqual(ChatCreateSend.decide(prompt: "   ", modelReady: false, busy: false), .ignore)
        XCTAssertEqual(ChatCreateSend.decide(prompt: "", modelReady: true, busy: false), .ignore)
        XCTAssertEqual(ChatCreateSend.decide(prompt: "a red fox", modelReady: true, busy: true), .ignore)
        XCTAssertEqual(ChatCreateSend.decide(prompt: "a red fox", modelReady: false, busy: true), .ignore)
    }

    /// An attached image means different things per mode: a source to edit or
    /// animate, versus something the speech model has no use for.
    func testOnlyVisualModesTakeASourceImage() {
        XCTAssertTrue(ChatCreateMode.image.usesSourceImage)
        XCTAssertTrue(ChatCreateMode.video.usesSourceImage)
        XCTAssertFalse(ChatCreateMode.audio.usesSourceImage)
    }

    /// 3D is deliberately absent: the transcript has no mesh viewer, the same
    /// reason `GeneratedMediaHandoff` won't hand one off to a chat.
    func testThereIsNo3DModeBecauseTheTranscriptCannotShowOne() {
        XCTAssertFalse(ChatCreateMode.allCases.contains { $0.experiment == .model3d })
        XCTAssertNil(GeneratedMediaHandoff.kind(for: .model3d))
    }
}
