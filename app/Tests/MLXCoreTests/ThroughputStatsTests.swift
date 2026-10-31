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

    func testPrefillProgressIsTheTokensDoneOverTheOnesExpected() {
        func progress(live: Int, expected: Int, running: Int = 1) -> Double? {
            ThroughputSnapshot.parse(["gauges": ["prefill_tokens_live": live, "prefill_tokens_expected": expected,
                                                 "requests_prefilling": running]], at: 0).prefillProgress
        }
        XCTAssertEqual(progress(live: 8192, expected: 32768), 0.25)
        XCTAssertNil(progress(live: 8192, expected: 32768, running: 0), "no prefill runs: no bar")
        XCTAssertNil(progress(live: 0, expected: 0), "a server that did not say how long it is: no bar")
        XCTAssertEqual(progress(live: 40000, expected: 32768), 1, "never past full")
    }

    func testPerModelUsageRowsParseIntoCumulativeCounters() {
        let json: [String: Any] = [
            "gauges": ["process_start_time_seconds": 1_790_000_000],
            "usage": [
                ["model": "org/a", "requests": 3, "prompt_tokens": 900, "cached_tokens": 400, "generation_tokens": 70,
                 "prefill_ns": 2_000_000_000, "decode_ns": 1_000_000_000],
                ["model": "org/b", "requests": 1, "prompt_tokens": 10, "cached_tokens": 0, "generation_tokens": 5,
                 "prefill_ns": 1, "decode_ns": 2],
            ],
        ]
        let s = ThroughputSnapshot.parse(json, at: 0)
        XCTAssertEqual(s.processStart, 1_790_000_000)
        XCTAssertEqual(s.usage["org/a"], UsageCounters(requests: 3, promptTokens: 900, cachedTokens: 400, generatedTokens: 70,
                                                       prefillNs: 2_000_000_000, decodeNs: 1_000_000_000))
        XCTAssertEqual(s.usage["org/b"]?.generatedTokens, 5)
        XCTAssertTrue(ThroughputSnapshot.parse([:], at: 0).usage.isEmpty, "a server that predates the rows reads as none")
    }

    func testBusyMeansARequestIsRunningOrWaiting() {
        func busy(_ gauges: [String: Any]) -> Bool { ThroughputSnapshot.parse(["gauges": gauges], at: 0).busy }
        XCTAssertFalse(busy(["requests_running": 0, "requests_waiting": 0, "requests_prefilling": 0]))
        XCTAssertTrue(busy(["requests_running": 1]))
        XCTAssertTrue(busy(["requests_waiting": 2]))
        XCTAssertFalse(ThroughputSnapshot.parse([:], at: 0).busy)
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
