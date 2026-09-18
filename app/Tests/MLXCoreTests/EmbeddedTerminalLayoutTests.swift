import XCTest
@testable import MLXCore

/// Gutter balancing for the embedded sandbox terminal. SwiftTerm reserves the
/// scroller width on the RIGHT (its columns stop short of the right edge)
/// while drawing from x=0 on the left — which reads as lopsided margins. The
/// container insets the left by the same reservation so the two gutters match
/// (± up to one cell of column quantization, which every terminal app has).
final class EmbeddedTerminalLayoutTests: XCTestCase {

    func testTerminalFrameInsetsLeftByTheScrollerReservation() {
        let frame = EmbeddedTerminalLayout.terminalFrame(
            in: CGRect(x: 0, y: 0, width: 800, height: 600), scrollerReservation: 16)
        XCTAssertEqual(frame.minX, 16, "left gutter must mirror the right-side scroller strip")
        XCTAssertEqual(frame.width, 784, "the terminal keeps everything except the left inset — SwiftTerm carves the right strip out itself")
        XCTAssertEqual(frame.minY, 0)
        XCTAssertEqual(frame.height, 600, "full height — only horizontal gutters are balanced")
    }

    func testTerminalFrameNeverGoesNegativeOnTinyBounds() {
        let frame = EmbeddedTerminalLayout.terminalFrame(
            in: CGRect(x: 0, y: 0, width: 10, height: 5), scrollerReservation: 16)
        XCTAssertGreaterThanOrEqual(frame.width, 0)
        XCTAssertGreaterThanOrEqual(frame.height, 0)
    }

    func testExitCodeIsDecodedFromTheRawWaitStatus() {
        XCTAssertEqual(EmbeddedTerminalView.exitCode(waitStatus: 256), 1)
        XCTAssertEqual(EmbeddedTerminalView.exitCode(waitStatus: 0), 0)
        XCTAssertEqual(EmbeddedTerminalView.exitCode(waitStatus: 15), 128 + 15, "killed by SIGTERM")
        XCTAssertNil(EmbeddedTerminalView.exitCode(waitStatus: nil))
    }
}
