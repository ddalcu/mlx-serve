import XCTest
@testable import MLXCore

/// The two first-run decisions (`LaunchDecision`, `ChatGateState`) plus the
/// copy on the shared starter card. Everything here is the pure seam under a
/// surface that is otherwise untestable — an `NSHostingView` outside the Scene
/// graph, and a sheet over a window.
final class FirstRunFlowTests: XCTestCase {

    private func model(_ name: String, type: String = "gemma4_text", kind: ModelKind = .base) -> LocalModel {
        LocalModel(
            id: name,
            name: name,
            path: "/tmp/\(name)",
            sizeFormatted: "1 GB",
            modelType: type,
            source: .mlxServe,
            kind: kind
        )
    }

    // MARK: - LaunchDecision

    /// The default (key absent ⇒ false) is today's behaviour: the welcome
    /// window on every launch.
    func testLaunchShowsWelcomeWhenNotSuppressed() {
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: false, hasChatModels: true), .showWelcome)
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: false, hasChatModels: false), .showWelcome)
    }

    /// "Don't show this again" means it, in BOTH model states. A suppressed
    /// welcome that comes back on the one launch where the user has no models —
    /// exactly when they'd want to re-tick the box — is how the checkbox loses
    /// its meaning; the chat gate covers that case with the same starter card.
    func testSuppressedLaunchOpensChatEvenWithNoModels() {
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: true, hasChatModels: true), .openChat)
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: true, hasChatModels: false), .openChat)
    }

    // MARK: - ChatGateState

    func testGateHiddenWhenAChatModelExists() {
        let state = ChatGateState.resolve(localModels: [model("gemma-4-e4b")], activeDownload: nil)
        XCTAssertEqual(state, .hidden)
        XCTAssertFalse(state.isBlocking)
    }

    func testGateBlocksWithNothingDownloaded() {
        let state = ChatGateState.resolve(localModels: [], activeDownload: nil)
        XCTAssertEqual(state, .needsModel)
        XCTAssertTrue(state.isBlocking)
    }

    /// Models present but none chat-capable — someone whose only download is an
    /// image backend has a full models folder and still can't send a message.
    func testGateBlocksWhenOnlyNonChatModelsExist() {
        let media = model("Krea-2-Turbo", type: "krea")
        let drafter = model("gemma-4-12B-it-assistant", kind: .drafter)
        XCTAssertFalse(media.isChatPickable)
        XCTAssertFalse(drafter.isChatPickable)
        XCTAssertEqual(ChatGateState.resolve(localModels: [media, drafter], activeDownload: nil), .needsModel)
    }

    func testGateReportsTheTransferWhileItRuns() {
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: 0.42),
                       .downloading(progress: 0.42))
    }

    /// Progress is clamped — a transfer whose reported total is briefly wrong
    /// must not drive a bar past the end of its track.
    func testGateClampsProgress() {
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: 1.8), .downloading(progress: 1))
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: -0.5), .downloading(progress: 0))
    }

    /// A usable model wins over an in-flight transfer: downloading a SECOND
    /// model must never block a chat the user can already have.
    func testAnExistingModelBeatsAnInFlightDownload() {
        XCTAssertEqual(ChatGateState.resolve(localModels: [model("gemma-4-e4b")], activeDownload: 0.1),
                       .hidden)
    }

    /// A Mac with nothing downloaded can still chat on a peer's model — the
    /// tray already counts those (`trayHasNoUsableModels`), and a gate that
    /// didn't would lock the user out of a conversation they can have.
    func testALanPeersChatModelIsEnoughToNotBlock() {
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: nil, lanChatModelCount: 1),
                       .hidden)
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: 0.5, lanChatModelCount: 2),
                       .hidden)
        // No peers is the default, so the plain two-argument call still blocks.
        XCTAssertEqual(ChatGateState.resolve(localModels: [], activeDownload: nil, lanChatModelCount: 0),
                       .needsModel)
    }

    // MARK: - Starter card copy

    /// Every tier the RAM bands can produce gets a real lead sentence — never
    /// the generic fallback, which exists only so an unlisted pick can't render
    /// an empty line.
    func testEveryStarterTierHasItsOwnLeadSentence() {
        let GiB: UInt64 = 1_073_741_824
        for bytes: UInt64 in [8 * GiB, 16 * GiB, 32 * GiB, 128 * GiB] {
            let pick = RecommendedModelPick.starterPick(physicalMemoryBytes: bytes)
            let lead = RecommendedStarterCard.lead(for: pick)
            XCTAssertFalse(lead.hasPrefix("A local AI assistant"), "\(pick.id) fell through to the fallback lead")
            // Leads with WHAT it is, then the size — never the catalog's
            // comparative tagline, which means nothing on a one-model card.
            XCTAssertTrue(lead.contains(" · "), lead)
            XCTAssertFalse(lead.contains(pick.tagline), lead)
        }
    }

    func testStarterLeadNamesTheDownloadSize() {
        XCTAssertEqual(RecommendedStarterCard.lead(for: .gemmaE4B), "A fast, capable assistant · 4.8 GB")
    }

    /// A partial transfer says Resume, not Download — a user who quit mid-pull
    /// is not starting over, and telling them they are invites a cancel.
    func testStarterActionTitleFollowsTheTransferState() {
        XCTAssertEqual(RecommendedStarterCard.actionTitle(hasPartial: false, failed: false), "Download")
        XCTAssertEqual(RecommendedStarterCard.actionTitle(hasPartial: true, failed: false), "Resume Download")
        XCTAssertEqual(RecommendedStarterCard.actionTitle(hasPartial: true, failed: true), "Resume Download")
        XCTAssertEqual(RecommendedStarterCard.actionTitle(hasPartial: false, failed: true), "Try Again")
    }
}
