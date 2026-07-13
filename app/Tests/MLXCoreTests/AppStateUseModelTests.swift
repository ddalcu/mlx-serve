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
}
