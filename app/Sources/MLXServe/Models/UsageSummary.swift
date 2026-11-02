import Foundation

/// One model's counters on one local calendar day.
struct UsageRow: Equatable {
    var day: String
    var model: String
    var counters: UsageCounters
}

/// What the dashboard reads off the stored rows.
struct UsageSummary: Equatable {
    struct ModelTotal: Equatable {
        var model: String
        var counters: UsageCounters
        /// The latest day this model was used.
        var lastDay: String
    }

    struct DayTotal: Equatable {
        var day: String
        var tokens: Int64
    }

    var rows: [UsageRow] = []

    var total: UsageCounters { rows.reduce(UsageCounters()) { $0 + $1.counters } }

    var firstDay: String? { rows.map(\.day).min() }

    /// Each model's totals, the busiest first.
    var byModel: [ModelTotal] {
        Dictionary(grouping: rows, by: \.model)
            .map { model, rows in
                ModelTotal(model: model, counters: rows.reduce(UsageCounters()) { $0 + $1.counters },
                           lastDay: rows.map(\.day).max() ?? "")
            }
            .sorted { ($0.counters.totalTokens, $1.model) > ($1.counters.totalTokens, $0.model) }
    }

    /// Prompt plus generated tokens per day, for one model or (nil) all of them.
    func tokensByDay(model: String? = nil) -> [String: Int64] {
        rows.reduce(into: [:]) { out, row in
            if model == nil || row.model == model { out[row.day, default: 0] += row.counters.totalTokens }
        }
    }

    /// Tokens on each of the `count` days ending at `today`, idle days as zero.
    func dailyTokens(lastDays count: Int, endingAt today: String, model: String? = nil,
                     calendar: Calendar = .current) -> [DayTotal] {
        let byDay = tokensByDay(model: model)
        return UsageDay.lastDays(count, endingAt: today, calendar: calendar).map { DayTotal(day: $0, tokens: byDay[$0] ?? 0) }
    }

    /// What was used over the `count` days ending at `today`.
    func counters(lastDays count: Int, endingAt today: String, calendar: Calendar = .current) -> UsageCounters {
        let days = Set(UsageDay.lastDays(count, endingAt: today, calendar: calendar))
        return rows.filter { days.contains($0.day) }.reduce(UsageCounters()) { $0 + $1.counters }
    }

    /// The day with the most tokens; the earlier one when two tie.
    var busiestDay: DayTotal? {
        tokensByDay().map { DayTotal(day: $0.key, tokens: $0.value) }
            .max { ($0.tokens, $1.day) < ($1.tokens, $0.day) }
    }

    /// Days in a row up to today, or up to yesterday while today is still empty: the day is not over yet.
    func currentStreak(today: String, calendar: Calendar = .current) -> Int {
        let active = Set(rows.map(\.day))
        var cursor = UsageDay.date(today, calendar: calendar)
        if !active.contains(today) { cursor = cursor.flatMap { calendar.date(byAdding: .day, value: -1, to: $0) } }
        var streak = 0
        while let day = cursor, active.contains(UsageDay.key(day, calendar: calendar)) {
            streak += 1
            cursor = calendar.date(byAdding: .day, value: -1, to: day)
        }
        return streak
    }
}

/// The green chart: a column per week, a cell per day, shaded by how much was used.
enum UsageHeatmap {
    struct Cell: Equatable {
        var day: String
        var value: Int64
        /// 0 for an idle day, else 1...4.
        var level: Int
    }

    struct MonthLabel: Equatable {
        var column: Int
        var month: Int
    }

    /// Shades from how the USED days rank, so a single enormous day does not turn every other day pale.
    struct Levels {
        private let used: [Int64]

        fileprivate init(_ values: [Int64]) { used = values.filter { $0 > 0 }.sorted() }

        func level(_ value: Int64) -> Int {
            guard value > 0, !used.isEmpty else { return 0 }
            let through = used.filter { $0 <= value }.count   // days at or below this one
            return (4 * through + used.count - 1) / used.count
        }
    }

    static func levels(for values: [Int64]) -> Levels { Levels(values) }

    /// `weekCount` columns of seven days, the last one the week holding `today`. A day still to come is nil.
    static func weeks(values: [String: Int64], today: Date, weekCount: Int, calendar: Calendar = .current) -> [[Cell?]] {
        let todayStart = calendar.startOfDay(for: today)
        let sinceWeekStart = (calendar.component(.weekday, from: todayStart) - calendar.firstWeekday + 7) % 7
        let start = calendar.date(byAdding: .day, value: -((weekCount - 1) * 7 + sinceWeekStart), to: todayStart) ?? todayStart
        let days: [Cell?] = (0..<weekCount * 7).map { offset in
            guard let date = calendar.date(byAdding: .day, value: offset, to: start), date <= todayStart else { return nil }
            let key = UsageDay.key(date, calendar: calendar)
            return Cell(day: key, value: values[key] ?? 0, level: 0)
        }
        let levels = levels(for: days.compactMap { $0?.value })
        return stride(from: 0, to: days.count, by: 7).map { first in
            days[first..<first + 7].map { cell in cell.map { Cell(day: $0.day, value: $0.value, level: levels.level($0.value)) } }
        }
    }

    /// The short weekday names in the order the calendar's weeks run.
    static func weekdaySymbols(calendar: Calendar = .current) -> [String] {
        let names = calendar.shortWeekdaySymbols
        let first = calendar.firstWeekday - 1
        return Array(names[first...] + names[..<first])
    }

    /// A month's name goes over the first column that starts in it, unless the next name is too near to fit beside it.
    static func monthLabels(weeks: [[Cell?]], calendar: Calendar = .current) -> [MonthLabel] {
        var labels: [MonthLabel] = []
        var previous: Int?
        for (column, week) in weeks.enumerated() {
            guard let first = week.lazy.compactMap({ $0 }).first,
                  let date = UsageDay.date(first.day, calendar: calendar) else { continue }
            let month = calendar.component(.month, from: date)
            if month != previous { labels.append(MonthLabel(column: column, month: month)) }
            previous = month
        }
        if labels.count > 1, labels[1].column - labels[0].column < 2 { labels.removeFirst() }
        return labels
    }
}
