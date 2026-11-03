import XCTest
@testable import MLXCore

/// What the dashboard reads off the history: totals, who used what, and the run of days.
final class UsageSummaryTests: XCTestCase {
    private var calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        c.firstWeekday = 1
        return c
    }()

    private func row(_ day: String, _ model: String, prompt: Int64, generated: Int64, requests: Int64 = 1) -> UsageRow {
        UsageRow(day: day, model: model, counters: UsageCounters(requests: requests, promptTokens: prompt, generatedTokens: generated))
    }

    private var rows: [UsageRow] {
        [row("2026-10-27", "org/a", prompt: 100, generated: 10),
         row("2026-10-28", "org/a", prompt: 200, generated: 20),
         row("2026-10-28", "org/b", prompt: 1000, generated: 100, requests: 4),
         row("2026-10-30", "org/b", prompt: 50, generated: 5)]
    }

    func testTotalsAndPerModelOrderedByTokens() {
        let s = UsageSummary(rows: rows)
        XCTAssertEqual(s.total.totalTokens, 1485)
        XCTAssertEqual(s.total.requests, 7)
        XCTAssertEqual(s.byModel.map(\.model), ["org/b", "org/a"], "the busiest model first")
        XCTAssertEqual(s.byModel.first?.counters.totalTokens, 1155)
        XCTAssertEqual(s.byModel.first?.lastDay, "2026-10-30")
    }

    func testPerDayCanBeNarrowedToOneModel() {
        let s = UsageSummary(rows: rows)
        XCTAssertEqual(s.tokensByDay()["2026-10-28"], 1320)
        XCTAssertEqual(s.tokensByDay(model: "org/a")["2026-10-28"], 220)
        XCTAssertNil(s.tokensByDay(model: "org/a")["2026-10-30"])
    }

    func testBusiestDayAndFirstDay() {
        let s = UsageSummary(rows: rows)
        XCTAssertEqual(s.busiestDay?.day, "2026-10-28")
        XCTAssertEqual(s.busiestDay?.tokens, 1320)
        XCTAssertEqual(s.firstDay, "2026-10-27")
        XCTAssertNil(UsageSummary(rows: []).busiestDay)
    }

    func testTheCurrentStreakEndsTodayOrYesterdayAndBreaksOnAGap() {
        let s = UsageSummary(rows: rows)
        // 27, 28 used, 29 empty, 30 used.
        XCTAssertEqual(s.currentStreak(today: "2026-10-30", calendar: calendar), 1)
        XCTAssertEqual(s.currentStreak(today: "2026-10-31", calendar: calendar), 1, "today not over yet: yesterday still counts")
        XCTAssertEqual(s.currentStreak(today: "2026-11-02", calendar: calendar), 0, "two idle days end it")
        XCTAssertEqual(s.currentStreak(today: "2026-10-28", calendar: calendar), 2)
    }

    func testTheLastDaysEndTodayOldestFirstAcrossAMonthEnd() {
        XCTAssertEqual(UsageDay.lastDays(3, endingAt: "2026-11-01", calendar: calendar), ["2026-10-30", "2026-10-31", "2026-11-01"])
        XCTAssertTrue(UsageDay.lastDays(3, endingAt: "not a day", calendar: calendar).isEmpty)
    }

    func testDailyTokensCoverEveryDayOfTheWindowAndNarrowToOneModel() {
        let s = UsageSummary(rows: rows)
        XCTAssertEqual(s.dailyTokens(lastDays: 4, endingAt: "2026-10-30", calendar: calendar),
                       [.init(day: "2026-10-27", tokens: 110), .init(day: "2026-10-28", tokens: 1320),
                        .init(day: "2026-10-29", tokens: 0), .init(day: "2026-10-30", tokens: 55)])
        XCTAssertEqual(s.dailyTokens(lastDays: 2, endingAt: "2026-10-28", model: "org/a", calendar: calendar).map(\.tokens), [110, 220])
    }

    func testCountersOverTheLastDaysLeaveOutOlderOnes() {
        let s = UsageSummary(rows: rows)
        XCTAssertEqual(s.counters(lastDays: 3, endingAt: "2026-10-30", calendar: calendar).totalTokens, 1375, "28th and 30th only")
        XCTAssertEqual(s.counters(lastDays: 1, endingAt: "2026-10-30", calendar: calendar).requests, 1)
    }

    func testCompactNumbersTruncateInsteadOfOverstating() {
        XCTAssertEqual([0, 999, 1_000, 1_234, 9_999, 10_000, 123_456, 999_999, 1_000_000, 2_500_000_000].map(UsageFormat.compact),
                       ["0", "999", "1.0K", "1.2K", "9.9K", "10K", "123K", "999K", "1.0M", "2.5B"])
    }
}
