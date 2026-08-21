import XCTest
@testable import MLXCore

final class ThroughputStatsTests: XCTestCase {
    private let feed: [String: Any] = [
        "counters": ["generation_tokens_total": 1200, "prefill_tokens_total": 40000],
        "gauges": ["generation_tokens_live": 1250, "prefill_tokens_live": 8192, "requests_prefilling": 1],
        "histograms": [
            "decode_time_seconds": ["count": 4, "sum": 20.0],
            "prefill_time_seconds": ["count": 4, "sum": 100.0],
        ],
    ]

    func testParsesCountersGaugesAndHistogramSums() {
        let s = ThroughputSnapshot.parse(feed, at: 100)
        XCTAssertEqual(s.generatedTokens, 1200)
        XCTAssertEqual(s.prefillTokens, 40000)
        XCTAssertEqual(s.generatedLive, 1250)
        XCTAssertEqual(s.displayedTokens, 1250)   // live, not the finished-request counter
        XCTAssertEqual(s.prefillLive, 8192)
        XCTAssertTrue(s.prefilling)
        XCTAssertEqual(s.avgDecodeTPS ?? 0, 60.0, accuracy: 0.001)
        XCTAssertEqual(s.avgPrefillTPS ?? 0, 400.0, accuracy: 0.001)
    }

    func testMissingFeedIsAllZerosAndNoRates() {
        let s = ThroughputSnapshot.parse([:], at: 0)
        XCTAssertEqual(s.generatedTokens, 0)
        XCTAssertNil(s.generatedLive)
        XCTAssertNil(s.avgDecodeTPS)
        XCTAssertNil(s.avgPrefillTPS)
    }

    func testTokensFallBackToTheCounterWhenTheGaugeIsAbsent() {
        var old = feed
        old["gauges"] = ["requests_prefilling": 0]
        XCTAssertEqual(ThroughputSnapshot.parse(old, at: 0).displayedTokens, 1200)
    }

    func testLiveRateComesFromTheGaugeDelta() {
        let a = ThroughputSnapshot.parse(feed, at: 100)
        var later = feed
        later["gauges"] = ["generation_tokens_live": 1400, "prefill_tokens_live": 8192, "requests_prefilling": 0]
        let b = ThroughputSnapshot.parse(later, at: 103)
        XCTAssertEqual(b.decodeTPS(since: a) ?? 0, 50.0, accuracy: 0.001)
    }

    func testAResetGaugeAndATooShortIntervalYieldNoRate() {
        let a = ThroughputSnapshot.parse(feed, at: 100)
        var restarted = feed
        restarted["gauges"] = ["generation_tokens_live": 0]
        XCTAssertNil(ThroughputSnapshot.parse(restarted, at: 103).decodeTPS(since: a))
        XCTAssertNil(ThroughputSnapshot.parse(feed, at: 100.05).decodeTPS(since: a))
    }

    func testFormatting() {
        XCTAssertEqual(ThroughputSnapshot.formatTPS(nil), "—")
        XCTAssertEqual(ThroughputSnapshot.formatTPS(0), "—")
        XCTAssertEqual(ThroughputSnapshot.formatTPS(62.44), "62.4")
        XCTAssertEqual(ThroughputSnapshot.formatTPS(301.6), "302")
    }
}
