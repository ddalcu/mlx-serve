import XCTest
@testable import MLXCore

/// What the dashboard counts: the server's cumulative per-model counters turned into additions per day.
final class UsageLedgerTests: XCTestCase {
    private func counters(requests: Int64 = 1, prompt: Int64 = 100, cached: Int64 = 0, generated: Int64 = 10) -> UsageCounters {
        UsageCounters(requests: requests, promptTokens: prompt, cachedTokens: cached, generatedTokens: generated,
                      prefillNs: 1_000_000_000, decodeNs: 2_000_000_000)
    }

    private func snapshot(start: Int64 = 1000, _ usage: [String: UsageCounters]) -> ThroughputSnapshot {
        var s = ThroughputSnapshot()
        s.processStart = start
        s.usage = usage
        return s
    }

    func testTheFirstSnapshotOnlySetsTheBaseline() {
        // A server already running when the app first looks has a past nobody here can date.
        var ledger = UsageLedger()
        XCTAssertTrue(ledger.ingest(snapshot(["org/a": counters()]), day: "2026-10-30").isEmpty)
    }

    func testLaterSnapshotsCountTheDifferencePerModelAndOmitWhatDidNotMove() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(["org/a": counters(), "org/b": counters()]), day: "2026-10-30")
        let deltas = ledger.ingest(snapshot(["org/a": counters(requests: 3, prompt: 500, cached: 200, generated: 60),
                                             "org/b": counters()]), day: "2026-10-31")
        XCTAssertEqual(deltas, [UsageDelta(day: "2026-10-31", model: "org/a",
                                           counters: UsageCounters(requests: 2, promptTokens: 400, cachedTokens: 200, generatedTokens: 50,
                                                                   prefillNs: 0, decodeNs: 0))])
    }

    func testANewServerProcessCountsEverythingItHasDone() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(start: 1000, ["org/a": counters(requests: 9, prompt: 9000, generated: 900)]), day: "2026-10-30")
        let deltas = ledger.ingest(snapshot(start: 2000, ["org/a": counters(requests: 1, prompt: 40, generated: 4)]), day: "2026-10-30")
        XCTAssertEqual(deltas.map(\.counters.promptTokens), [40], "a restart is not a negative number")
        XCTAssertEqual(deltas.map(\.counters.requests), [1])
    }

    func testAModelFirstSeenLaterCountsWhole() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(["org/a": counters()]), day: "2026-10-30")
        let deltas = ledger.ingest(snapshot(["org/a": counters(), "org/new": counters(requests: 2, prompt: 70, generated: 7)]), day: "2026-10-30")
        XCTAssertEqual(deltas.map(\.model), ["org/new"])
        XCTAssertEqual(deltas.first?.counters.promptTokens, 70)
    }

    func testACounterThatFellWithinOneProcessIsTakenAsAReset() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(["org/a": counters(requests: 5, prompt: 500, generated: 50)]), day: "2026-10-30")
        let deltas = ledger.ingest(snapshot(["org/a": counters(requests: 1, prompt: 20, generated: 2)]), day: "2026-10-30")
        XCTAssertEqual(deltas.first?.counters.promptTokens, 20)
    }

    func testSpeedsAreTheTokensOverTheTimeSpent() {
        let c = UsageCounters(requests: 2, promptTokens: 1000, cachedTokens: 600, generatedTokens: 90, prefillNs: 2_000_000_000, decodeNs: 3_000_000_000)
        XCTAssertEqual(c.decodeTPS ?? 0, 30, accuracy: 0.001)
        XCTAssertEqual(c.prefillTPS ?? 0, 200, accuracy: 0.001, "only the tokens that were computed, not the cached ones")
        XCTAssertNil(UsageCounters().decodeTPS)
        XCTAssertEqual(c.totalTokens, 1090)
    }

    func testEachLookCarriesItsOwnRatesAsItsPeaks() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(["org/a": counters()]), day: "2026-10-31")
        var now = counters(requests: 2, prompt: 2100, generated: 90)
        now.prefillNs += 1_000_000_000   // 2000 prompt tokens in one more second
        now.decodeNs += 2_000_000_000    // 80 generated tokens in two more seconds
        let peak = ledger.ingest(snapshot(["org/a": now]), day: "2026-10-31").first?.counters
        XCTAssertEqual(peak?.peakDecodeTPS ?? 0, 40, accuracy: 0.001)
        XCTAssertEqual(peak?.peakPrefillTPS ?? 0, 2000, accuracy: 0.001)
    }

    func testAStretchTooShortToTimeSetsNoPeak() {
        var ledger = UsageLedger()
        _ = ledger.ingest(snapshot(["org/a": counters()]), day: "2026-10-31")
        var now = counters(requests: 2, prompt: 110, generated: 13)
        now.prefillNs += 1_000
        now.decodeNs += 1_000
        let short = ledger.ingest(snapshot(["org/a": now]), day: "2026-10-31").first?.counters
        XCTAssertEqual(short?.peakDecodeTPS, 0)
        XCTAssertEqual(short?.peakPrefillTPS, 0)
    }

    func testAddingKeepsTheFasterPeak() {
        let slow = UsageCounters(requests: 1, peakDecodeTPS: 30, peakPrefillTPS: 900)
        let fast = UsageCounters(requests: 1, peakDecodeTPS: 50, peakPrefillTPS: 700)
        XCTAssertEqual(slow + fast, UsageCounters(requests: 2, peakDecodeTPS: 50, peakPrefillTPS: 900))
    }

    func testTheDayKeyIsTheLocalCalendarDay() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        let lateEvening = ISO8601DateFormatter().date(from: "2026-10-31T02:30:00Z")!   // still the 30th in New York
        XCTAssertEqual(UsageDay.key(lateEvening, calendar: calendar), "2026-10-30")
    }
}
