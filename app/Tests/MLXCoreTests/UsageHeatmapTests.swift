import XCTest
@testable import MLXCore

/// The green chart: a column per week, a cell per day, shaded by how much was used.
final class UsageHeatmapTests: XCTestCase {
    private var calendar: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        c.firstWeekday = 1   // weeks start on Sunday
        c.locale = Locale(identifier: "en_US")
        return c
    }()

    private func date(_ key: String) -> Date {
        UsageDay.date(key, calendar: calendar)!
    }

    func testTheLastColumnIsTheCurrentWeekAndEndsToday() {
        // Wednesday the 28th: that week is Sun 25 .. Sat 31, of which Thursday on has not happened.
        let weeks = UsageHeatmap.weeks(values: [:], today: date("2026-10-28"), weekCount: 3, calendar: calendar)
        XCTAssertEqual(weeks.count, 3)
        XCTAssertTrue(weeks.allSatisfy { $0.count == 7 })
        XCTAssertEqual(weeks[0].compactMap { $0?.day }.first, "2026-10-11")
        XCTAssertEqual(weeks[1].compactMap { $0?.day }.last, "2026-10-24")
        XCTAssertEqual(weeks[2].compactMap { $0?.day }, ["2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28"])
        XCTAssertEqual(weeks[2].suffix(3).map { $0 == nil }, [true, true, true], "days still to come are empty, not zero")
    }

    func testAWeekStartsOnTheCalendarsFirstWeekday() {
        calendar.firstWeekday = 2   // Monday
        let weeks = UsageHeatmap.weeks(values: [:], today: date("2026-10-28"), weekCount: 1, calendar: calendar)
        XCTAssertEqual(weeks[0].compactMap { $0?.day }.first, "2026-10-26")
    }

    func testCellsCarryTheirDaysValueAndNothingForAnIdleDay() {
        let weeks = UsageHeatmap.weeks(values: ["2026-10-27": 500], today: date("2026-10-28"), weekCount: 1, calendar: calendar)
        XCTAssertEqual(weeks[0][2]?.value, 500)
        XCTAssertEqual(weeks[0][1]?.value, 0)
        XCTAssertEqual(weeks[0][1]?.level, 0)
    }

    func testLevelsAreQuartilesOfTheDaysThatWereUsedSoOneHugeDayDoesNotFlattenTheRest() {
        let levels = UsageHeatmap.levels(for: [1, 2, 3, 4, 5, 6, 7, 8, 1_000_000])
        XCTAssertEqual(levels.level(0), 0, "an idle day is never shaded")
        // Against the busiest day alone, everything below the outlier would be level 1.
        XCTAssertEqual([1, 3, 5, 7, 1_000_000].map(levels.level), [1, 2, 3, 4, 4])
    }

    func testASingleUsedDayAndTiesAreTheDarkestAndIdleDaysDoNotCount() {
        XCTAssertEqual(UsageHeatmap.levels(for: [500]).level(500), 4)
        XCTAssertEqual(UsageHeatmap.levels(for: [0, 0, 7, 7]).level(7), 4)
        XCTAssertEqual(UsageHeatmap.levels(for: []).level(7), 0, "nothing to compare against: no shade")
    }

    func testAMonthNameSitsOverTheFirstColumnThatStartsInThatMonth() {
        // Sunday columns: Oct 4, 11, 18, 25, then Nov 1 and 8.
        let weeks = UsageHeatmap.weeks(values: [:], today: date("2026-11-12"), weekCount: 6, calendar: calendar)
        let labels = UsageHeatmap.monthLabels(weeks: weeks, calendar: calendar)
        XCTAssertEqual(labels.map(\.column), [0, 4])
        XCTAssertEqual(labels.map(\.month), [10, 11])
    }

    func testAFirstLabelTooCloseToTheNextOneIsLeftOut() {
        // Columns Oct 25 and Nov 1: two names one column apart would overprint.
        let weeks = UsageHeatmap.weeks(values: [:], today: date("2026-11-05"), weekCount: 2, calendar: calendar)
        XCTAssertEqual(UsageHeatmap.monthLabels(weeks: weeks, calendar: calendar).map(\.month), [11])
    }

    func testWeekdayNamesRunFromTheCalendarsFirstWeekday() {
        XCTAssertEqual(UsageHeatmap.weekdaySymbols(calendar: calendar), ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
        calendar.firstWeekday = 2
        XCTAssertEqual(UsageHeatmap.weekdaySymbols(calendar: calendar).prefix(2), ["Mon", "Tue"])
        XCTAssertEqual(UsageHeatmap.weekdaySymbols(calendar: calendar).last, "Sun")
    }
}
