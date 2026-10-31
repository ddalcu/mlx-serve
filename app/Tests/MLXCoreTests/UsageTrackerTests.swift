import XCTest
@testable import MLXCore

/// The running tally: the server's counters in, a per-day history out, kept across restarts.
@MainActor
final class UsageTrackerTests: XCTestCase {
    private var path: String!

    override func setUp() {
        path = NSTemporaryDirectory() + "usage-tracker-\(UUID().uuidString).db"
    }

    override func tearDown() {
        for suffix in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: path + suffix) }
    }

    private func counters(requests: Int64 = 1, prompt: Int64 = 100, generated: Int64 = 10) -> UsageCounters {
        UsageCounters(requests: requests, promptTokens: prompt, generatedTokens: generated, prefillNs: 1_000_000_000, decodeNs: 1_000_000_000)
    }

    private func snapshot(_ usage: [String: UsageCounters], at: TimeInterval = 0, generatedLive: Int64? = nil,
                          generated: Int64 = 0) -> ThroughputSnapshot {
        var s = ThroughputSnapshot()
        s.processStart = 1000
        s.usage = usage
        s.takenAt = at
        s.generatedTokens = generated
        s.generatedLive = generatedLive
        return s
    }

    private func tracker(day: String = "2026-10-30") -> UsageTracker {
        UsageTracker(path: path, today: { day }, fetch: { nil })
    }

    func testPollsFastWhileARequestIsBeingServedAndSlowlyAtRest() {
        XCTAssertLessThan(UsageTracker.interval(busy: true), UsageTracker.interval(busy: false))
    }

    func testTheFirstLookOnlyBaselinesAndLaterOnesLandInTheSummary() {
        let t = tracker()
        t.ingest(snapshot(["org/a": counters()]))
        XCTAssertTrue(t.summary.rows.isEmpty)
        t.ingest(snapshot(["org/a": counters(requests: 3, prompt: 500, generated: 60)]))
        XCTAssertEqual(t.summary.rows, [UsageRow(day: "2026-10-30", model: "org/a",
                                                 counters: UsageCounters(requests: 2, promptTokens: 400, generatedTokens: 50,
                                                                         prefillNs: 0, decodeNs: 0))])
        XCTAssertEqual(t.summary.total.totalTokens, 450)
    }

    func testTheTallyOutlivesTheTracker() {
        do {
            let t = tracker()
            t.ingest(snapshot(["org/a": counters()]))
            t.ingest(snapshot(["org/a": counters(requests: 2, prompt: 300, generated: 30)]))
        }
        let relaunched = tracker()
        relaunched.load()
        XCTAssertEqual(relaunched.summary.total.totalTokens, 220)
    }

    func testResetForgetsOneModelOrEverythingAndStaysForgotten() {
        let t = tracker()
        t.ingest(snapshot(["org/a": counters(), "org/b": counters()]))
        t.ingest(snapshot(["org/a": counters(requests: 2), "org/b": counters(requests: 2)]))
        XCTAssertEqual(Set(t.summary.rows.map(\.model)), ["org/a", "org/b"])
        t.reset(model: "org/a")
        XCTAssertEqual(t.summary.rows.map(\.model), ["org/b"])
        t.reset(model: nil)
        XCTAssertTrue(t.summary.rows.isEmpty)
        let relaunched = tracker()
        relaunched.load()
        XCTAssertTrue(relaunched.summary.rows.isEmpty)
    }

    func testTheStripTotalGrowsWithTheTokensAReplyHasWrittenSoFar() {
        let t = tracker()
        t.ingest(snapshot(["org/a": counters()], at: 0, generatedLive: 0, generated: 0))
        t.ingest(snapshot(["org/a": counters(requests: 2, prompt: 200, generated: 20)], at: 1, generatedLive: 20, generated: 20))
        XCTAssertEqual(t.totalTokens, 110, "nothing in flight: the finished request's tokens")
        t.ingest(snapshot(["org/a": counters(requests: 2, prompt: 200, generated: 20)], at: 2, generatedLive: 55, generated: 20))
        XCTAssertEqual(t.totalTokens, 145, "35 more tokens are on their way")
    }

    func testRatesAreTheTokensMovedBetweenTwoLooks() {
        let t = tracker()
        t.ingest(snapshot([:], at: 0, generatedLive: 100, generated: 0))
        t.ingest(snapshot([:], at: 2, generatedLive: 210, generated: 0))
        XCTAssertEqual(t.decodeTPS ?? 0, 55, accuracy: 0.001)
        t.ingest(snapshot([:], at: 4, generatedLive: 210, generated: 210))
        XCTAssertNil(t.decodeTPS, "nothing moved: the figure goes quiet instead of showing the last burst")
    }

    func testAServerThatStopsAnsweringClearsTheLiveReading() {
        let t = tracker()
        t.ingest(snapshot([:], at: 0, generatedLive: 100, generated: 0))
        t.stop()
        XCTAssertNil(t.live)
        XCTAssertNil(t.decodeTPS)
    }
}
