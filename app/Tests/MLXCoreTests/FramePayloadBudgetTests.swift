import XCTest
@testable import MLXCore

/// A whole clip rides back as ONE base64 response, so its raw RGB has a cap
/// (#283). The length ladder bills it, but the ladder alone is a gate you can
/// walk around: raising the window count AFTER choosing a length shortens the
/// ladder under a value that is already set, the slider hides that by pinning
/// to its right edge, and Generate would send it.
final class FramePayloadBudgetTests: XCTestCase {

    private let h3 = VideoModelPreset.minimaxH3

    func testChainedWindowsAreCountedAsTheFramesTheyDeliver() {
        // w windows joined end to end share their seam frames.
        XCTAssertEqual(VideoModelPreset.deliveredFrames(perWindow: 100, chainWindows: 1), 100)
        XCTAssertEqual(VideoModelPreset.deliveredFrames(perWindow: 100, chainWindows: 6), 595)
        XCTAssertEqual(VideoModelPreset.deliveredFrames(perWindow: 100, chainWindows: 0), 100,
                       "a nonsense window count is one window")
    }

    /// The reported case: 260 frames is fine alone and impossible six times
    /// over, and nothing about the canvas or the quality tier changed.
    func testALengthThatFitsAloneCanStopFittingWhenWindowsAreRaised() {
        XCTAssertTrue(h3.framePayloadFits(width: 1344, height: 768, numFrames: 260, chainWindows: 1))
        XCTAssertFalse(h3.framePayloadFits(width: 1344, height: 768, numFrames: 260, chainWindows: 6))
    }

    /// The gate and the ladder have to agree, or one of them is lying about
    /// what the request can carry.
    func testTheLadderOffersExactlyWhatTheGateAccepts() {
        for windows in 1...6 {
            let offered = h3.frameOptions(width: 1344, height: 768, chainWindows: windows)
            for n in offered where n != h3.frameOptions.first {
                XCTAssertTrue(h3.framePayloadFits(width: 1344, height: 768,
                                                  numFrames: n, chainWindows: windows),
                              "ladder offers \(n) at \(windows) windows but the gate refuses it")
            }
            for n in h3.frameOptions where !offered.contains(n) {
                XCTAssertFalse(h3.framePayloadFits(width: 1344, height: 768,
                                                   numFrames: n, chainWindows: windows),
                               "gate accepts \(n) at \(windows) windows but the ladder hides it")
            }
        }
    }

    /// The slider reads this list by index, so an empty one would leave it
    /// with nothing to land on. A nonsense canvas divides by a clamped 1
    /// rather than by zero, which makes it look free — the guard being tested
    /// here is that the answer is always usable, not what it contains.
    func testTheLadderIsNeverEmpty() {
        for (w, h) in [(0, 0), (1, 1), (1920, 1088), (99_999, 99_999)] {
            for windows in [1, 6] {
                XCTAssertFalse(h3.frameOptions(width: w, height: h, chainWindows: windows).isEmpty,
                               "\(w)x\(h) at \(windows) windows")
            }
        }
    }
}
