import XCTest
@testable import MLXCore

/// The report's headline shows the realistic decode peak and, beside it, the best
/// case ("Max") with its own context: the two peaks are usually at different rungs.
final class BenchmarkHighlightsTests: XCTestCase {

    private func row(_ target: Int, decode: Double, max: Double) -> BenchmarkRungTable.Row {
        BenchmarkRungTable.Row(targetTokens: target, promptTokens: target, prefillTps: 1000, decodeTps: decode,
                               ceilingDecodeTps: max, ttftMs: 100, contextUsed: true, samples: nil)
    }

    func testPeakDecodeAndMaxDecodeEachNameTheirOwnContext() {
        let rows = [row(512, decode: 234.1, max: 271.9), row(1024, decode: 223.6, max: 280.9), row(16384, decode: 186.9, max: 255.5)]
        XCTAssertEqual(BenchmarkHighlights.peakDecode(rows)?.targetTokens, 512)
        XCTAssertEqual(BenchmarkHighlights.maxDecode(rows)?.targetTokens, 1024)
        XCTAssertEqual(BenchmarkHighlights.maxDecode(rows)?.ceilingDecodeTps, 280.9)
    }

    func testNoMaxIsShownWhenNoRungMeasuredOne() {
        let rows = [row(512, decode: 234.1, max: 0), row(1024, decode: 223.6, max: 0)]
        XCTAssertNil(BenchmarkHighlights.maxDecode(rows))
        XCTAssertNotNil(BenchmarkHighlights.peakDecode(rows))
    }

    func testTheChartAndTableCallTheBestCaseMax() {
        let points = BenchmarkLadderChart.points(decode: [(512, 200)], ceiling: [(512, 250)])
        XCTAssertEqual(Set(points.map(\.series)), ["Decode", "Max"])
    }
}
