import XCTest
@testable import MLXCore

/// Pins the pure frame-timing math behind chat video attachments. Unlike
/// MiniMax-H3's reference-video sampling (`VideoGenService.refFrameTimes`,
/// fixed 24 fps, first-N-seconds only), a chat attachment samples evenly
/// across the WHOLE clip — a user asking "what happens in this video" wants
/// the gist of the whole thing, not just its opening beats.
final class VideoPreprocessorTests: XCTestCase {

    func testFrameTimesSpanTheWholeDurationInclusiveOfBothEnds() {
        let t = VideoPreprocessor.frameTimes(duration: 10, count: 5)
        XCTAssertEqual(t.count, 5)
        XCTAssertEqual(t[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(t[4], 10.0, accuracy: 1e-9) // last timestamp reaches the end, not short of it
        XCTAssertEqual(t[2], 5.0, accuracy: 1e-9) // evenly spaced midpoint
    }

    func testFrameTimesSingleFrameIsTheStart() {
        XCTAssertEqual(VideoPreprocessor.frameTimes(duration: 30, count: 1), [0.0])
    }

    func testFrameTimesEmptyOnInvalidInput() {
        XCTAssertTrue(VideoPreprocessor.frameTimes(duration: 0, count: 5).isEmpty)
        XCTAssertTrue(VideoPreprocessor.frameTimes(duration: 10, count: 0).isEmpty)
        XCTAssertTrue(VideoPreprocessor.frameTimes(duration: -1, count: 5).isEmpty)
    }
}
