import XCTest
@testable import MLXCore

/// Pins the pure decision behind `AppState.useModelAndAwaitReady` — the
/// Model Browser's "Use" button now starts/switches the server AND waits for
/// it to actually be serving before opening the Chat window, instead of just
/// setting `selectedModelPath` and leaving the user to press Start Server
/// themselves. The branch (start explicitly vs. await the `didSet`'s
/// fire-and-forget hot-switch/restart) is the part worth pinning without a
/// real `ServerManager`.
final class AppStateUseModelTests: XCTestCase {

    /// `selectedModelPath`'s `didSet` is a no-op against the server for
    /// `.stopped`/`.error` — the caller must start the server itself.
    func testStoppedOrErrorStartsExplicitly() {
        XCTAssertEqual(AppState.useModelStartAction(forStatusBefore: .stopped), .startExplicitly)
        XCTAssertEqual(AppState.useModelStartAction(forStatusBefore: .error("boom")), .startExplicitly)
    }

    /// `.running`/`.starting` already trigger a hot-switch or restart inside
    /// `didSet` — the caller only needs to wait for it.
    func testRunningOrStartingAwaitsThePendingSwitch() {
        XCTAssertEqual(AppState.useModelStartAction(forStatusBefore: .running), .awaitPendingSwitch)
        XCTAssertEqual(AppState.useModelStartAction(forStatusBefore: .starting), .awaitPendingSwitch)
    }

    /// The `didSet`'s own decision: a RUNNING server is hot-switched in place —
    /// no restart. The id it loads must be the model's ABSOLUTE PATH: registry
    /// ids are two-level `org/name`, so a dir basename 404s (register-by-path
    /// resolves either shape). The old `hotSwitchEnabled` gate is gone — it
    /// shipped default-off with no UI, so every picker change restarted the
    /// server for everyone, forever.
    func testRunningHotSwitchesInPlaceWithTheAbsolutePath() {
        let path = "/Users/me/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit"
        XCTAssertEqual(AppState.modelSwitchAction(forStatus: .running, path: path),
                       .hotSwitch(id: path))
    }

    /// Mid-boot the process is still loading the OLD pick — restart with the
    /// new one. Stopped/error stays untouched: explicit starts
    /// (`useModelAndAwaitReady`, the launch gate) own that.
    func testStartingRestartsAndStoppedIsLeftAlone() {
        XCTAssertEqual(AppState.modelSwitchAction(forStatus: .starting, path: "/x"), .restart)
        XCTAssertEqual(AppState.modelSwitchAction(forStatus: .stopped, path: "/x"), .leaveStopped)
        XCTAssertEqual(AppState.modelSwitchAction(forStatus: .error("boom"), path: "/x"), .leaveStopped)
    }
}
