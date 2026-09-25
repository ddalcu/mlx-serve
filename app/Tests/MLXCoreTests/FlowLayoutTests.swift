import XCTest
@testable import MLXCore

/// Controls sat a whole grid column apart because `LazyVGrid` divides the
/// width into equal columns whether or not the cells need it. A flow packs
/// them against each other and wraps only when the next one does not fit.
final class FlowLayoutTests: XCTestCase {

    func testItemsShareARowWhileThereIsRoom() {
        XCTAssertEqual(FlowLayout.rows(widths: [100, 100, 100], maxWidth: 340, spacing: 10),
                       [[0, 1, 2]])
    }

    /// 100 + 10 + 100 + 10 + 100 = 320, so the third one does not fit in 300.
    func testTheItemThatDoesNotFitStartsTheNextRow() {
        XCTAssertEqual(FlowLayout.rows(widths: [100, 100, 100], maxWidth: 300, spacing: 10),
                       [[0, 1], [2]])
    }

    /// A control wider than the container keeps its own row rather than
    /// producing an empty one before it.
    func testAnOversizedItemStillGetsARow() {
        XCTAssertEqual(FlowLayout.rows(widths: [400, 50], maxWidth: 300, spacing: 10),
                       [[0], [1]])
    }

    func testNothingLaysOutAsNoRows() {
        XCTAssertEqual(FlowLayout.rows(widths: [], maxWidth: 300, spacing: 10), [])
    }

    /// The gap belongs BETWEEN items: three 100s fit in exactly 320, not 300.
    func testTheGapIsCountedOnlyBetweenItems() {
        XCTAssertEqual(FlowLayout.rows(widths: [100, 100, 100], maxWidth: 320, spacing: 10),
                       [[0, 1, 2]])
    }
}
