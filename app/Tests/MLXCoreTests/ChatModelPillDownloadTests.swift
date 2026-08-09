import XCTest
@testable import MLXCore

/// The chat model pill's progress hairline claims "a live transfer for the
/// model this chat is pointed at" — but it read `.values.first` over the
/// unordered downloads dictionary, i.e. ANY in-flight transfer: a 30 GB video
/// pack from the Create pane rendered as the chat model arriving, and while it
/// ran `needsDownload` went false, hiding the download-arrow affordance on a
/// Mac with zero chat models. The filter now excludes media-bundle repos
/// (`DownloadManager.mediaBundleRepos`, recorded by startBundle/startTurboLora).
@MainActor
final class ChatModelPillDownloadTests: XCTestCase {

    private func downloading() -> DownloadManager.DownloadState {
        DownloadManager.DownloadState(status: .downloading)
    }

    func testAMediaBundleTransferNeverShowsOnTheChatPill() {
        let picked = ChatModelPill.chatDownload(
            in: ["Lightricks/LTX-2": downloading()],
            mediaRepos: ["Lightricks/LTX-2"])
        XCTAssertNil(picked, "a media pack downloading is not the chat model arriving")
    }

    func testAChatModelTransferStillShows() {
        let picked = ChatModelPill.chatDownload(
            in: ["mlx-community/gemma-4-12b-it-4bit": downloading(),
                 "Lightricks/LTX-2": downloading()],
            mediaRepos: ["Lightricks/LTX-2"])
        XCTAssertNotNil(picked, "the chat model's own transfer keeps the hairline")
    }

    func testCompletedTransfersDoNotShow() {
        var done = DownloadManager.DownloadState(status: .completed)
        done.progress = 1
        XCTAssertNil(ChatModelPill.chatDownload(in: ["a/b": done], mediaRepos: []))
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
