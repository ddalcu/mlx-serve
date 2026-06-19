import XCTest
@testable import MLXCore

/// Pure presentation seam for the tool-call card's kill X.
final class ProcessCardControlsTests: XCTestCase {

    func testAliveHandleYieldsButton() {
        let live: Set<String> = ["bg1"]
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1"]) { live.contains($0) }, ["bg1"])
    }

    func testDeadOrUnknownYieldsNone() {
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1", "bg2"]) { _ in false }, [])
    }

    func testNilHandlesYieldsNone() {
        XCTAssertEqual(ProcessCardControls.killable(handles: nil) { _ in true }, [])
    }

    func testMixedKeepsOnlyLive() {
        let live: Set<String> = ["bg2"]
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1", "bg2", "bg3"]) { live.contains($0) }, ["bg2"])
    }

    /// Handles persisted on a message survive a restart, but the registry isn't
    /// persisted — so on load nothing is alive → no kill buttons.
    @MainActor
    func testPersistedHandlesAfterRestartYieldNone() {
        let freshRegistry = ProcessRegistry()
        let out = ProcessCardControls.killable(handles: ["bg1", "bg2"], isAlive: freshRegistry.isAlive)
        XCTAssertEqual(out, [])
    }
}
