import XCTest
@testable import MLXCore

/// The reference tile's hover preview is sized by arithmetic, not by the
/// layout proposal: an overlay is proposed the size of the tile under it, and
/// a fit image sized to that came out a third of the size on every picture.
final class ReferencePreviewTests: XCTestCase {

    // `accuracy` throughout: the fitted side is `size * (side / size)`, which
    // in floating point is 204.00000000000003, and a frame does not care.
    func testAWidePictureFillsTheWidthAndATallOneTheHeight() {
        let wide = VideoGenView.previewSize(for: CGSize(width: 3000, height: 2000), within: 204)
        XCTAssertEqual(wide?.width ?? 0, 204, accuracy: 0.001)
        XCTAssertEqual(wide?.height ?? 0, 136, accuracy: 0.001)
        let tall = VideoGenView.previewSize(for: CGSize(width: 1000, height: 3000), within: 204)
        XCTAssertEqual(tall?.height ?? 0, 204, accuracy: 0.001)
        XCTAssertEqual(tall?.width ?? 0, 68, accuracy: 0.001)
        let square = VideoGenView.previewSize(for: CGSize(width: 2048, height: 2048), within: 204)
        XCTAssertEqual(square?.width ?? 0, 204, accuracy: 0.001)
        XCTAssertEqual(square?.height ?? 0, 204, accuracy: 0.001)
    }

    /// Never upscaled: a picture smaller than the box is shown at its own
    /// size rather than blurred up to fill it.
    func testASmallPictureIsNotUpscaled() {
        let small = VideoGenView.previewSize(for: CGSize(width: 120, height: 90), within: 204)
        XCTAssertEqual(small?.width, 120)
        XCTAssertEqual(small?.height, 90)
    }

    func testANonSizeYieldsNothing() {
        XCTAssertNil(VideoGenView.previewSize(for: .zero, within: 204))
        XCTAssertNil(VideoGenView.previewSize(for: CGSize(width: 10, height: 0), within: 204))
        XCTAssertNil(VideoGenView.previewSize(for: CGSize(width: 10, height: 10), within: 0))
    }
}
