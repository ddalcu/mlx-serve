import XCTest
@testable import MLXCore

/// A starting frame is resized to the canvas by the server with no letterbox
/// and no crop, so a canvas that does not match its shape stretches the
/// picture. The model's shipped resolution list is a curated set, not its
/// limit, so the canvases come from the GRID.
final class AspectCanvasesTests: XCTestCase {

    /// LTX one-stage.
    private let ltx = ResolutionGrid(alignment: 32, minDim: 256, maxDim: 1920)

    func testEverySuggestionIsOnTheGridAndInsideTheRange() {
        for c in AspectCanvases.options(sourceWidth: 2048, sourceHeight: 1152, grid: ltx) {
            XCTAssertEqual(c.width % 32, 0, "\(c.id)")
            XCTAssertEqual(c.height % 32, 0, "\(c.id)")
            XCTAssertTrue((256...1920).contains(c.width), "\(c.id)")
            XCTAssertTrue((256...1920).contains(c.height), "\(c.id)")
        }
    }

    func testFiveAreOfferedLargestFirst() {
        let opts = AspectCanvases.options(sourceWidth: 2048, sourceHeight: 1152, grid: ltx)
        XCTAssertEqual(opts.count, 5)
        XCTAssertEqual(opts.map(\.area), opts.map(\.area).sorted(by: >))
        XCTAssertEqual(opts.first?.id, "1920x1088", "the largest 16:9-ish canvas the model samples")
    }

    /// The whole point of filtering before spreading: the smallest LEGAL 16:9
    /// canvas is 480 x 256, which is 5.5% off and visibly a different shape.
    func testAVisiblyDistortedCanvasIsNotOfferedEvenAsTheSmallest() {
        let opts = AspectCanvases.options(sourceWidth: 2048, sourceHeight: 1152, grid: ltx)
        XCTAssertFalse(opts.contains(AspectCanvas(width: 480, height: 256)))
        let ratio = 2048.0 / 1152.0
        for c in opts {
            let deviation = abs(Double(c.width) / Double(c.height) - ratio) / ratio
            XCTAssertLessThanOrEqual(deviation, AspectCanvases.maxDeviation, "\(c.id)")
        }
    }

    /// A shape no canvas in range can hold: 3840 x 400 is 9.6:1, and at the
    /// widest legal width the height lands under the 256 floor. Nothing is
    /// offered — the menu's disabled row says why.
    func testAnImpossibleShapeOffersNothing() {
        XCTAssertTrue(AspectCanvases.options(sourceWidth: 3840, sourceHeight: 400, grid: ltx).isEmpty)
    }

    /// Candidates exist but none within 3%: the closest is offered rather than
    /// an empty menu. Needs a grid small enough that no large canvas can get
    /// close — on the shipped grids a big enough canvas always can.
    func testWhenNothingIsFaithfulTheClosestIsOffered() {
        let tiny = ResolutionGrid(alignment: 64, minDim: 256, maxDim: 320)
        let opts = AspectCanvases.options(sourceWidth: 230, sourceHeight: 200, grid: tiny)
        XCTAssertEqual(opts, [AspectCanvas(width: 320, height: 256)],
                       "1.25:1 is 8.7% off 1.15:1, the least bad of the two candidates")
    }

    // MARK: - Source size

    func testTheSourcesOwnSizeIsOfferedWhenItFits() {
        let c = AspectCanvases.sourceSize(sourceWidth: 1000, sourceHeight: 700, grid: ltx)
        XCTAssertEqual(c, AspectCanvas(width: 992, height: 704), "nearest grid point, both sides")
    }

    func testASourceOutsideTheRangeHasNoOwnSize() {
        XCTAssertNil(AspectCanvases.sourceSize(sourceWidth: 4032, sourceHeight: 3024, grid: ltx),
                     "over the model's maximum side")
        XCTAssertNil(AspectCanvases.sourceSize(sourceWidth: 200, sourceHeight: 150, grid: ltx),
                     "under the model's minimum side")
    }

    // MARK: - Labels

    func testTheRatioIsNamedTheWayPeopleNameIt() {
        XCTAssertEqual(AspectCanvases.ratioLabel(width: 2048, height: 1152), "16:9")
        XCTAssertEqual(AspectCanvases.ratioLabel(width: 4032, height: 3024), "4:3")
        XCTAssertEqual(AspectCanvases.ratioLabel(width: 1080, height: 1080), "1:1")
    }

    /// The exact reduction is no help: 109:107 is a square nobody would read
    /// as one, and 3841 x 2159 reduces to itself.
    func testAnAwkwardRatioIsRoundedToTheShapeItIs() {
        XCTAssertEqual(AspectCanvases.ratioLabel(width: 1090, height: 1070), "1:1")
        XCTAssertEqual(AspectCanvases.ratioLabel(width: 3841, height: 2159), "16:9")
    }
}
