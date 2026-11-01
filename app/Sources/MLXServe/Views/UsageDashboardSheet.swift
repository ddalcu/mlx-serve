import SwiftUI

/// What the Reset button wipes.
enum UsageReset: Equatable {
    case all
    case model(String)

    var model: String? {
        if case .model(let id) = self { return id }
        return nil
    }
}

/// Everything the server has served, day by day and model by model: opened from the status strip's Total Tokens.
struct UsageDashboardSheet: View {
    @ObservedObject var usage: UsageTracker
    @Environment(\.dismiss) private var dismiss
    /// nil = every model.
    @State private var model: String?
    @State private var pendingReset: UsageReset?

    private var today: String { UsageDay.key(Date()) }

    /// What the filter row scopes: the tiles, the activity chart and the daily bars. The table always lists every model.
    private var scoped: UsageSummary {
        guard let model else { return usage.summary }
        return UsageSummary(rows: usage.summary.rows.filter { $0.model == model })
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            if let reset = pendingReset { resetBanner(reset) }
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 26) {
                    filterRow
                    tiles
                    activity
                    daily
                    if !usage.summary.rows.isEmpty { models }
                }
                .padding(24)
            }
        }
        // The activity chart is 53 columns wide and does not compress: the width is fixed to hold it beside a scroller.
        .frame(minWidth: 800, idealWidth: 800, maxWidth: 800, minHeight: 440, idealHeight: 700)
        .onAppear { usage.load() }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Usage").font(.app(.sectionTitle, weight: .semibold))
                Text(subtitle).font(.app(.explainer)).foregroundStyle(.secondary)
            }
            Spacer()
            Button("Reset…") { pendingReset = .all }.disabled(usage.summary.rows.isEmpty)
            Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    private var subtitle: String {
        guard let first = usage.summary.firstDay else {
            return L10n.text("Nothing counted yet. Tokens are tallied as the server finishes each request.")
        }
        return L10n.format("Tokens from every request the server finished, counted since %@.", UsageDay.text(first))
    }

    /// Two steps, in the sheet itself: a system alert here would compete with the window's single alert.
    private func resetBanner(_ reset: UsageReset) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            Text(message(for: reset)).font(.app(.explainer))
            Spacer(minLength: 12)
            Button("Cancel") { pendingReset = nil }
            Button("Reset") { confirm(reset) }.tint(.red)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 10)
        .background(Color.orange.opacity(0.12))
    }

    private func message(for reset: UsageReset) -> String {
        switch reset {
        case .all:
            return L10n.text("Erase the counts of every model and day? This cannot be undone.")
        case .model(let id):
            return L10n.format("Erase the counts of %@ on every day? This cannot be undone.", ModelDisplayName.pretty(id))
        }
    }

    private func confirm(_ reset: UsageReset) {
        usage.reset(model: reset.model)
        if reset.model == nil || reset.model == model { model = nil }
        pendingReset = nil
    }

    // MARK: - Filter and tiles

    private var filterRow: some View {
        HStack {
            Picker("Model", selection: $model) {
                Text("All models").tag(String?.none)
                ForEach(usage.summary.byModel, id: \.model) { entry in
                    Text(verbatim: ModelDisplayName.pretty(entry.model)).tag(Optional(entry.model))
                }
            }
            .labelsHidden()
            .fixedSize()
            Spacer()
        }
    }

    private var tiles: some View {
        let s = scoped
        let total = s.total
        let todays = s.counters(lastDays: 1, endingAt: today)
        let week = s.counters(lastDays: 7, endingAt: today)
        let cachedShare = total.promptTokens > 0 ? Int(Double(total.cachedTokens) / Double(total.promptTokens) * 100) : 0
        return LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 10)], spacing: 10) {
            UsageTile(label: "Total tokens", value: UsageFormat.compact(total.totalTokens), hero: true,
                      detail: L10n.format("%@ in · %@ out", UsageFormat.compact(total.promptTokens), UsageFormat.compact(total.generatedTokens)))
            UsageTile(label: "Today", value: UsageFormat.compact(todays.totalTokens),
                      detail: L10n.format("Requests: %lld", todays.requests))
            UsageTile(label: "Last 7 days", value: UsageFormat.compact(week.totalTokens),
                      detail: L10n.format("Requests: %lld", week.requests))
            UsageTile(label: "Requests", value: ThroughputSnapshot.formatTokens(total.requests),
                      detail: L10n.format("%lld%% of prompt cached", cachedShare))
            UsageSpeedTile(counters: total)
            UsageTile(label: "Days in a row", value: "\(s.currentStreak(today: today))",
                      detail: s.busiestDay.map { L10n.format("Busiest day: %@", UsageDay.text($0.day)) } ?? "")
        }
    }

    // MARK: - Charts

    private var activity: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Activity").font(.app(.rowTitle, weight: .semibold))
            UsageHeatmapView(weeks: UsageHeatmap.weeks(values: scoped.tokensByDay(), today: Date(), weekCount: 53))
        }
    }

    private var daily: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Last 30 days").font(.app(.rowTitle, weight: .semibold))
            UsageDailyBars(days: scoped.dailyTokens(lastDays: 30, endingAt: today))
        }
    }

    // MARK: - Per model

    private var models: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Models").font(.app(.rowTitle, weight: .semibold))
            UsageModelTable(models: usage.summary.byModel, totalTokens: usage.summary.total.totalTokens) {
                pendingReset = .model($0)
            }
            Text("Decode and Prefill are tokens per second, averaged over each model's requests.")
                .font(.app(.annotation)).foregroundStyle(.tertiary)
        }
    }
}

/// A headline number: the label above it, the figure in proportional digits, a line of context beneath.
private struct UsageTile: View {
    let label: String
    let value: String
    var hero = false
    let detail: String

    var body: some View {
        let valueStyle: Font.TextStyle = hero ? .largeTitle : .title2
        VStack(alignment: .leading, spacing: 4) {
            Text(L10n.text(label)).font(.app(.annotation)).foregroundStyle(.secondary)
            Text(verbatim: value).font(.app(valueStyle, weight: .semibold)).lineLimit(1).minimumScaleFactor(0.7)
            Text(verbatim: detail).font(.app(.annotation)).foregroundStyle(.tertiary).lineLimit(1)
        }
        .usageTile()
    }
}

/// Decode and prefill: the average over every request beside the fastest one.
private struct UsageSpeedTile: View {
    let counters: UsageCounters

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Speed (tok/s)").font(.app(.annotation)).foregroundStyle(.secondary)
            Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 2) {
                GridRow {
                    Color.clear.gridCellUnsizedAxes([.horizontal, .vertical])
                    Text("Average")
                    Text("Max")
                }
                .font(.app(.annotation))
                .foregroundStyle(.tertiary)
                row("Decode", average: counters.decodeTPS, peak: counters.peakDecodeTPS)
                row("Prefill", average: counters.prefillTPS, peak: counters.peakPrefillTPS)
            }
        }
        .usageTile()
    }

    private func row(_ label: LocalizedStringKey, average: Double?, peak: Double) -> some View {
        GridRow(alignment: .firstTextBaseline) {
            Text(label).font(.app(.annotation)).foregroundStyle(.secondary).gridColumnAlignment(.leading)
            Text(verbatim: ThroughputSnapshot.formatTPS(average))
            Text(verbatim: ThroughputSnapshot.formatTPS(peak))
        }
        .font(.app(.title3, weight: .semibold))
        .monospacedDigit()
    }
}

private extension View {
    /// A tile's ground: content from the top left, a faint fill and edge.
    func usageTile() -> some View {
        frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.04)))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.primary.opacity(0.08)))
            .accessibilityElement(children: .combine)
    }
}

/// The green chart: a column per week, a cell per day, one hue stepped by how the used days rank.
private struct UsageHeatmapView: View {
    let weeks: [[UsageHeatmap.Cell?]]

    private static let cell: CGFloat = 11
    private static let gap: CGFloat = 2
    private static var pitch: CGFloat { cell + gap }

    var body: some View {
        VStack(alignment: .trailing, spacing: 8) {
            HStack(alignment: .top, spacing: 6) {
                weekdayLabels
                VStack(alignment: .leading, spacing: 4) {
                    monthLabels
                    grid
                }
            }
            legend
        }
    }

    private var weekdayLabels: some View {
        let names = UsageHeatmap.weekdaySymbols()
        return VStack(alignment: .trailing, spacing: 0) {
            Color.clear.frame(height: 18)
            ForEach(0..<7, id: \.self) { row in
                Text(verbatim: row % 2 == 1 ? names[row] : "")
                    .font(.app(.annotation)).foregroundStyle(.tertiary)
                    .frame(height: Self.pitch)
            }
        }
        .frame(width: 30, alignment: .trailing)
    }

    private var monthLabels: some View {
        ZStack(alignment: .topLeading) {
            ForEach(UsageHeatmap.monthLabels(weeks: weeks), id: \.column) { label in
                Text(verbatim: Calendar.current.shortMonthSymbols[label.month - 1])
                    .font(.app(.annotation)).foregroundStyle(.secondary)
                    .fixedSize()
                    .offset(x: CGFloat(label.column) * Self.pitch)
            }
        }
        .frame(height: 14, alignment: .topLeading)
    }

    private var grid: some View {
        HStack(spacing: 0) {
            ForEach(weeks.indices, id: \.self) { column in
                VStack(spacing: 0) {
                    ForEach(0..<7, id: \.self) { row in square(weeks[column][row]) }
                }
            }
        }
    }

    /// The hit target is the whole pitch, so the gap between two days is not a dead strip for the tooltip.
    @ViewBuilder private func square(_ cell: UsageHeatmap.Cell?) -> some View {
        if let cell {
            RoundedRectangle(cornerRadius: 2)
                .fill(UsagePalette.color(level: cell.level))
                .padding(Self.gap / 2)
                .frame(width: Self.pitch, height: Self.pitch)
                .contentShape(Rectangle())
                .help(Self.tooltip(cell))
                .accessibilityElement()
                .accessibilityLabel(Self.tooltip(cell))
        } else {
            Color.clear.frame(width: Self.pitch, height: Self.pitch)
        }
    }

    private static func tooltip(_ cell: UsageHeatmap.Cell) -> String {
        cell.value > 0
            ? L10n.format("%@ · %@ tokens", UsageDay.text(cell.day), ThroughputSnapshot.formatTokens(cell.value))
            : L10n.format("No usage on %@", UsageDay.text(cell.day))
    }

    private var legend: some View {
        HStack(spacing: 4) {
            Text("Less")
            ForEach(0..<5, id: \.self) { level in
                RoundedRectangle(cornerRadius: 2).fill(UsagePalette.color(level: level)).frame(width: Self.cell, height: Self.cell)
            }
            Text("More")
        }
        .font(.app(.annotation))
        .foregroundStyle(.secondary)
    }
}

/// Tokens per day for the last month: one series, one colour, so the title is the legend.
private struct UsageDailyBars: View {
    let days: [UsageSummary.DayTotal]

    private static let plotHeight: CGFloat = 96

    var body: some View {
        let peak = days.map(\.tokens).max() ?? 0
        HStack(alignment: .top, spacing: 8) {
            yAxis(peak: peak)
            VStack(spacing: 4) {
                GeometryReader { geo in plot(peak: peak, width: geo.size.width) }
                    .frame(height: Self.plotHeight)
                xAxis
            }
        }
    }

    /// Top, middle and zero; with nothing to scale to, only the zero.
    private func yAxis(peak: Int64) -> some View {
        VStack(alignment: .trailing) {
            if peak >= 2 {
                Text(verbatim: UsageFormat.compact(peak))
                Spacer(minLength: 0)
                Text(verbatim: UsageFormat.compact(peak / 2))
            }
            Spacer(minLength: 0)
            Text(verbatim: "0")
        }
        .font(.app(.annotation)).foregroundStyle(.tertiary).monospacedDigit()
        .frame(width: 40, height: Self.plotHeight)
    }

    private func plot(peak: Int64, width: CGFloat) -> some View {
        let pitch = width / CGFloat(max(days.count, 1))
        let barWidth = min(24, pitch - 2)
        return ZStack(alignment: .bottom) {
            VStack(spacing: 0) {
                hairline
                Spacer(minLength: 0)
                hairline
                Spacer(minLength: 0)
                Rectangle().fill(Color.primary.opacity(0.18)).frame(height: 1)
            }
            HStack(spacing: 0) {
                ForEach(days, id: \.day) { day in
                    bar(day, peak: peak, width: barWidth).frame(width: pitch, height: Self.plotHeight, alignment: .bottom)
                }
            }
        }
    }

    private var hairline: some View { Rectangle().fill(Color.primary.opacity(0.07)).frame(height: 1) }

    /// 4pt rounded at the tip, square on the baseline; an idle day draws nothing but still answers the pointer.
    private func bar(_ day: UsageSummary.DayTotal, peak: Int64, width: CGFloat) -> some View {
        let height = peak > 0 && day.tokens > 0 ? max(2, Self.plotHeight * CGFloat(day.tokens) / CGFloat(peak)) : 0
        return UnevenRoundedRectangle(topLeadingRadius: 4, bottomLeadingRadius: 0, bottomTrailingRadius: 0, topTrailingRadius: 4)
            .fill(UsagePalette.series)
            .frame(width: width, height: height)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
            .contentShape(Rectangle())
            .help(day.tokens > 0
                  ? L10n.format("%@ · %@ tokens", UsageDay.text(day.day), ThroughputSnapshot.formatTokens(day.tokens))
                  : L10n.format("No usage on %@", UsageDay.text(day.day)))
    }

    private var xAxis: some View {
        GeometryReader { geo in
            let pitch = geo.size.width / CGFloat(max(days.count, 1))
            ZStack(alignment: .topLeading) {
                ForEach(Array(stride(from: 0, to: days.count, by: 7)), id: \.self) { index in
                    Text(verbatim: UsageDay.text(days[index].day, year: false))
                        .font(.app(.annotation)).foregroundStyle(.tertiary)
                        .fixedSize()
                        .offset(x: CGFloat(index) * pitch)
                }
            }
        }
        .frame(height: 14)
    }
}

/// Every model's totals as a table: past a handful of rows a chart stops being readable.
private struct UsageModelTable: View {
    let models: [UsageSummary.ModelTotal]
    let totalTokens: Int64
    let onReset: (String) -> Void

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 10) {
            GridRow {
                Text("Model").gridColumnAlignment(.leading)
                Text("Requests")
                Text("In")
                Text("Out")
                Text("Cached")
                Text("Decode")
                Text("Prefill")
                Color.clear.frame(width: 18, height: 1)
            }
            .font(.app(.annotation))
            .foregroundStyle(.secondary)
            Divider().gridCellUnsizedAxes(.horizontal)
            ForEach(models, id: \.model) { entry in row(entry) }
        }
    }

    private func row(_ entry: UsageSummary.ModelTotal) -> some View {
        let c = entry.counters
        let cached = c.promptTokens > 0 ? Int(Double(c.cachedTokens) / Double(c.promptTokens) * 100) : 0
        let share = totalTokens > 0 ? Double(c.totalTokens) / Double(totalTokens) : 0
        return GridRow {
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: ModelDisplayName.pretty(entry.model)).lineLimit(1).help(entry.model)
                Text(L10n.format("Last used %@ · %lld%% of all tokens", UsageDay.text(entry.lastDay), Int(share * 100)))
                    .font(.app(.annotation)).foregroundStyle(.tertiary)
                LoadBar(fraction: share, tint: UsagePalette.series, height: 4)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(verbatim: ThroughputSnapshot.formatTokens(c.requests))
            Text(verbatim: UsageFormat.compact(c.promptTokens))
            Text(verbatim: UsageFormat.compact(c.generatedTokens))
            Text(verbatim: "\(cached)%")
            Text(verbatim: ThroughputSnapshot.formatTPS(c.decodeTPS))
            Text(verbatim: ThroughputSnapshot.formatTPS(c.prefillTPS))
            Button { onReset(entry.model) } label: { Image(systemName: "arrow.counterclockwise") }
                .buttonStyle(.borderless)
                .help(L10n.format("Reset %@…", ModelDisplayName.pretty(entry.model)))
        }
        .monospacedDigit()
    }
}
