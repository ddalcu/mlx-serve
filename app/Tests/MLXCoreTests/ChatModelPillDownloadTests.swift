import XCTest
@testable import MLXCore

/// The chat model pill's progress hairline claims "a live transfer for the
/// model this chat is pointed at" — and for two rounds it did not check which
/// model. First it read `.values.first` over the unordered downloads dictionary,
/// i.e. ANY in-flight transfer: a 30 GB video pack from the Create pane rendered
/// as the chat model arriving, and while it ran `needsDownload` went false,
/// hiding the download-arrow affordance on a Mac with zero chat models. Media
/// repos are excluded now (`DownloadManager.mediaBundleRepos`) — but a CHAT
/// model fetched in the background still drew a bar under a completely
/// different, already-serving model.
///
/// The bar is keyed to the pill's own model now, the way every other download
/// surface keys its row (`downloads[pick.repoId]`). The one exception is the
/// reason it was put there: with nothing chat-pickable on disk, whatever is
/// arriving IS what the empty composer is waiting for.
@MainActor
final class ChatModelPillDownloadTests: XCTestCase {

    private func downloading() -> DownloadManager.DownloadState {
        DownloadManager.DownloadState(status: .downloading)
    }

    private let selected = "/Users/me/.mlx-serve/models/mlx-community/gemma-4-12b-it-4bit"

    func testAMediaBundleTransferNeverShowsOnTheChatPill() {
        let picked = ChatModelPill.chatDownload(
            in: ["Lightricks/LTX-2": downloading()],
            mediaRepos: ["Lightricks/LTX-2"],
            selectedModelPath: selected, hasChatModelOnDisk: true)
        XCTAssertNil(picked, "a media pack downloading is not the chat model arriving")
    }

    func testThePillsOwnModelArrivingShowsTheBar() {
        let picked = ChatModelPill.chatDownload(
            in: ["mlx-community/gemma-4-12b-it-4bit": downloading(),
                 "Lightricks/LTX-2": downloading()],
            mediaRepos: ["Lightricks/LTX-2"],
            selectedModelPath: selected, hasChatModelOnDisk: true)
        XCTAssertNotNil(picked, "the selected model's own transfer keeps the hairline")
    }

    func testAnotherModelDownloadingInTheBackgroundDrawsNothing() {
        // The bug: chatting on one model while a second downloads put a
        // progress bar under the model that was already answering.
        let picked = ChatModelPill.chatDownload(
            in: ["ddalcu/Qwen3.8-27B-MLX-Serve-4bit": downloading()],
            mediaRepos: [],
            selectedModelPath: selected, hasChatModelOnDisk: true)
        XCTAssertNil(picked, "someone else's transfer is not this chat's model arriving")
    }

    func testWithNothingOnDiskAnyChatTransferIsTheOneWeAreWaitingFor() {
        // First run: the composer can't answer at all, and the bar is the
        // reason why — this is what the hairline was added for.
        let picked = ChatModelPill.chatDownload(
            in: ["ddalcu/Qwen3.8-27B-MLX-Serve-4bit": downloading()],
            mediaRepos: [],
            selectedModelPath: "", hasChatModelOnDisk: false)
        XCTAssertNotNil(picked)
    }

    func testAGgufQuantMatchesItsReposTransfer() {
        // A GGUF selection points at the FILE inside the repo folder, so the
        // repo id is one level further up than for an MLX checkpoint.
        let picked = ChatModelPill.chatDownload(
            in: ["unsloth/Qwen3.5-4B-GGUF": downloading()],
            mediaRepos: [],
            selectedModelPath: "/Users/me/.mlx-serve/models/unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf",
            hasChatModelOnDisk: true)
        XCTAssertNotNil(picked)
    }

    func testCompletedTransfersDoNotShow() {
        var done = DownloadManager.DownloadState(status: .completed)
        done.progress = 1
        XCTAssertNil(ChatModelPill.chatDownload(in: ["a/b": done], mediaRepos: [],
                                                selectedModelPath: "/models/a/b",
                                                hasChatModelOnDisk: true))
    }

    func testStartBundleRecordsItsComponentRepos() {
        // The recording happens synchronously, before any transfer starts, so
        // the pill's filter is right from the first progress publish. Cancel
        // immediately after — the task hasn't run yet on this actor, so no
        // network transfer ever starts.
        let manager = DownloadManager(modelsRoot: NSTemporaryDirectory())
        let bundle = MediaBundle.flux(repo: "example/media-pack",
                                      displayName: "Test", sizeGB: 1)
        manager.startBundle(bundle) {}
        manager.cancelBundle(bundle)
        XCTAssertTrue(manager.mediaBundleRepos.contains("example/media-pack"),
                      "a bundle component must be filtered off the chat pill")
    }
}
