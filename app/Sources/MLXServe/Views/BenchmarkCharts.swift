import SwiftUI
import Charts

/// One point of a ladder chart: a rung and a measured rate.
struct BenchmarkChartPoint: Identifiable {
    var targetTokens: Int
    var series: String
    var value: Double
    var id: String { "\(series)-\(targetTokens)" }
}

/// Decode + speculation ceiling per rung. Log2 x axis, because the rungs are
/// powers of two and a linear axis puts three of four points in the first
/// eighth of the width.
struct BenchmarkLadderChart: View {
    let points: [BenchmarkChartPoint]

    static let decodeSeries = "Decode"
    static let ceilingSeries = "Ceiling"

    static func points(decode: [(Int, Double)], ceiling: [(Int, Double)]) -> [BenchmarkChartPoint] {
        decode.filter { $0.1 > 0 }.map { BenchmarkChartPoint(targetTokens: $0.0, series: decodeSeries, value: $0.1) }
        + ceiling.filter { $0.1 > 0 }.map { BenchmarkChartPoint(targetTokens: $0.0, series: ceilingSeries, value: $0.1) }
    }

    var body: some View {
        BenchmarkRateChart(points: points, color: .accentColor, showsCeiling: true)
    }
}

/// Prefill uses the same geometry as decode so the contexts line up.
struct BenchmarkPrefillChart: View {
    let points: [BenchmarkChartPoint]

    static func points(_ prefill: [(Int, Double)]) -> [BenchmarkChartPoint] {
        prefill.filter { $0.1 > 0 }.map { BenchmarkChartPoint(targetTokens: $0.0, series: "Prefill", value: $0.1) }
    }

    var body: some View {
        BenchmarkRateChart(points: points, color: .orange, showsCeiling: false)
    }
}

private struct BenchmarkRateChart: View {
    let points: [BenchmarkChartPoint]
    let color: Color
    let showsCeiling: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                legend(showsCeiling ? "Decode" : "Prefill", color: color)
                if showsCeiling { legend("Ceiling", color: .secondary, dashed: true) }
                Spacer()
                Text("tok/s").font(.app(.caption)).foregroundStyle(.secondary)
            }

            Chart {
                ForEach(points.filter { $0.series != BenchmarkLadderChart.ceilingSeries }) { p in
                    AreaMark(x: .value("Context", p.targetTokens), yStart: .value("Baseline", 0),
                             yEnd: .value("tok/s", p.value))
                        .foregroundStyle(LinearGradient(colors: [color.opacity(0.16), color.opacity(0.01)],
                                                        startPoint: .top, endPoint: .bottom))
                        .interpolationMethod(.monotone)
                }
                ForEach(points) { p in
                    let isCeiling = p.series == BenchmarkLadderChart.ceilingSeries
                    LineMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value),
                             series: .value("Series", p.series))
                        .foregroundStyle(isCeiling ? Color.secondary.opacity(0.65) : color)
                        .lineStyle(StrokeStyle(lineWidth: isCeiling ? 1.5 : 2.5, dash: isCeiling ? [4, 4] : []))
                        .interpolationMethod(.monotone)
                    PointMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value))
                        .foregroundStyle(isCeiling ? Color.secondary : color)
                        .symbolSize(isCeiling ? 16 : 30)
                }
            }
            .chartXScale(domain: rungDomain, type: .log)
            .chartXAxis {
                AxisMarks(values: BenchmarkSuite.ladder.map(\.targetTokens)) { value in
                    AxisValueLabel(centered: false, anchor: .top) {
                        if let tokens = value.as(Int.self) {
                            Text(BenchmarkSuite.title(forTarget: tokens)).font(.app(.caption))
                        }
                    }
                }
            }
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { _ in
                    AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                        .foregroundStyle(.primary.opacity(0.10))
                    AxisValueLabel(centered: false, anchor: .trailing).font(.app(.caption))
                }
            }
            .chartYScale(domain: .automatic(includesZero: true))
            .chartLegend(.hidden)
            .frame(height: 120)
            .overlay {
                if points.isEmpty {
                    Text("No measurements").font(.app(.callout)).foregroundStyle(.secondary)
                }
            }

            Text("Context length · tokens")
                .font(.app(.caption)).foregroundStyle(.secondary)
                .frame(maxWidth: .infinity)
        }
    }

    private func legend(_ title: String, color: Color, dashed: Bool = false) -> some View {
        HStack(spacing: 5) {
            HStack(spacing: 3) {
                Capsule().fill(color)
                if dashed { Capsule().fill(color) }
            }
            .frame(width: 16, height: 3)
            Text(L10n.text(title)).font(.app(.caption)).foregroundStyle(.secondary)
        }
    }
}

/// Include empty rungs so incomplete runs keep the same context scale.
private var rungDomain: ClosedRange<Double> {
    let targets = BenchmarkSuite.ladder.map { Double($0.targetTokens) }
    return (targets.min() ?? 512) * 0.85 ... (targets.max() ?? 16384) * 1.25
}

/// Highlights always name their context; peaks are not whole-run averages.
struct BenchmarkHighlights: View {
    let rows: [BenchmarkRungTable.Row]

    private var peakDecode: BenchmarkRungTable.Row? {
        rows.filter { $0.decodeTps > 0 }.max { $0.decodeTps < $1.decodeTps }
    }
    private var peakPrefill: BenchmarkRungTable.Row? {
        rows.filter { $0.prefillTps > 0 }.max { $0.prefillTps < $1.prefillTps }
    }
    private var fastestResponse: BenchmarkRungTable.Row? {
        rows.filter { $0.ttftMs > 0 }.min { $0.ttftMs < $1.ttftMs }
    }

    var body: some View {
        HStack(spacing: 16) {
            metric("Peak decode", icon: "waveform.path", color: .accentColor,
                   value: peakDecode?.decodeTps, decimals: 1, unit: "tok/s", row: peakDecode)
            metric("Peak prefill", icon: "bolt", color: .orange,
                   value: peakPrefill?.prefillTps, decimals: 0, unit: "tok/s", row: peakPrefill)
            metric("Fastest first token", icon: "stopwatch", color: .primary,
                   value: fastestResponse?.ttftMs, decimals: 0, unit: "ms", row: fastestResponse)
        }
    }

    private func metric(_ title: String, icon: String, color: Color, value: Double?,
                        decimals: Int, unit: String, row: BenchmarkRungTable.Row?) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(L10n.text(title), systemImage: icon)
                .font(.app(.callout, weight: .medium)).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 5) {
                Text(BenchmarkFormat.rate(value ?? 0, decimals: decimals))
                    .font(.app(.largeTitle, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(color)
                Text(unit).font(.app(.callout)).foregroundStyle(.secondary)
            }
            Group {
                if let row {
                    Text(L10n.format("At %@ context", BenchmarkSuite.title(forTarget: row.targetTokens)))
                } else {
                    Text("No measurements")
                }
            }
            .font(.app(.caption)).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(BenchSurface())
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(color.opacity(0.14), lineWidth: 0.5) }
        .accessibilityElement(children: .combine)
    }
}

/// The compact rung table shared by the result card and the sheet.
struct BenchmarkRungTable: View {
    struct Row: Identifiable {
        var targetTokens: Int
        var promptTokens: Int
        var prefillTps: Double
        var decodeTps: Double
        var ceilingDecodeTps: Double
        var ttftMs: Double
        var contextUsed: Bool?
        var samples: Int?
        var id: Int { targetTokens }
    }

    let rows: [Row]

    static func rows(_ rungs: [BenchmarkResult]) -> [Row] {
        rungs.map { r in
            Row(targetTokens: r.effectiveTargetTokens, promptTokens: r.promptTokens,
                prefillTps: r.prefillTps, decodeTps: r.decodeTps,
                ceilingDecodeTps: r.ceilingDecodeTps ?? 0, ttftMs: r.ttftMs,
                contextUsed: r.contextUsed, samples: nil)
        }
    }

    static func rows(_ family: BenchmarkStore.CellFamily) -> [Row] {
        family.rungs.map { r in
            Row(targetTokens: r.targetTokens, promptTokens: 0,
                prefillTps: r.prefillTps, decodeTps: r.decodeTps,
                ceilingDecodeTps: r.ceilingDecodeTps, ttftMs: r.ttftMs,
                contextUsed: nil, samples: r.sampleCount)
        }
    }

    private var showsSamples: Bool { rows.contains { $0.samples != nil } }
    private var showsPrompt: Bool { rows.contains { $0.promptTokens > 0 } }

    var body: some View {
        Grid(alignment: .trailing, horizontalSpacing: 16, verticalSpacing: 0) {
            GridRow {
                header("Context", unit: "tokens", alignment: .leading).gridColumnAlignment(.leading)
                if showsPrompt { header("Prompt", unit: "tokens") }
                header("Prefill", unit: "tok/s")
                header("Decode", unit: "tok/s")
                header("Ceiling", unit: "tok/s")
                header("First token", unit: "ms")
                if showsSamples { header("Samples", unit: "sessions") }
                else { header("Context", unit: "check") }
            }
            .padding(.bottom, 10)
            Divider().gridCellUnsizedAxes(.horizontal)
            ForEach(rows) { row in
                GridRow {
                    Text(BenchmarkSuite.title(forTarget: row.targetTokens))
                        .fontWeight(.medium)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .gridColumnAlignment(.leading)
                    if showsPrompt { cell(row.promptTokens > 0 ? "\(row.promptTokens)" : "—") }
                    cell(BenchmarkFormat.rate(row.prefillTps, decimals: 0))
                    cell(BenchmarkFormat.rate(row.decodeTps, decimals: 1))
                        .fontWeight(.semibold).foregroundStyle(Color.accentColor)
                    cell(BenchmarkFormat.rate(row.ceilingDecodeTps, decimals: 1))
                        .foregroundStyle(.secondary)
                    cell(BenchmarkFormat.rate(row.ttftMs, decimals: 0))
                    if showsSamples {
                        Text("\(row.samples ?? 0)")
                            .monospacedDigit()
                            .foregroundStyle((row.samples ?? 0) == 1 ? .orange : .secondary)
                            .help("Number of sessions behind this result")
                    } else {
                        Image(systemName: row.contextUsed == true ? "checkmark.circle.fill" : "minus")
                            .foregroundStyle(row.contextUsed == true ? .green : .secondary)
                            .accessibilityLabel(row.contextUsed == true ? "Context used" : "Context not verified")
                            .help(row.contextUsed == true
                                  ? "The answer used the constant planted in the context"
                                  : "The answer did not use the planted constant")
                    }
                }
                .padding(.vertical, 5)
                if row.id != rows.last?.id {
                    Divider().gridCellUnsizedAxes(.horizontal)
                }
            }
        }
        .font(.app(.callout))
        .textSelection(.enabled)
    }

    private func header(_ text: String, unit: String, alignment: HorizontalAlignment = .trailing) -> some View {
        VStack(alignment: alignment, spacing: 2) {
            Text(L10n.text(text)).font(.app(.caption, weight: .medium)).foregroundStyle(.secondary)
            Text(L10n.text(unit)).font(.app(.caption2)).foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: alignment == .leading ? .leading : .trailing)
    }

    private func cell(_ text: String) -> some View {
        Text(text).monospacedDigit()
    }
}

enum BenchmarkFormat {
    static func rate(_ value: Double, decimals: Int) -> String {
        guard value > 0 else { return "—" }
        return String(format: "%.\(decimals)f", value)
    }
}

/// Settings chips, as the tables draw them.
struct BenchmarkSettingsChips: View {
    let settings: [String: String]

    var body: some View {
        HStack(spacing: 4) {
            ForEach(BenchmarkSettings.summaryChips(settings), id: \.self) { chip in
                Text(L10n.text(chip))
                    .font(.app(.caption2, weight: .medium))
                    .padding(.horizontal, 5)
                    .padding(.vertical, 2)
                    .background(.quaternary.opacity(0.5), in: Capsule())
            }
        }
    }
}

/// One titled card. Cards rather than `GroupBox` so the title can carry an
/// icon and the fill can stay subtle.
struct BenchCard<Content: View>: View {
    let title: String
    let icon: String
    var footnote: String? = nil
    @ViewBuilder var content: Content

    init(_ title: String, icon: String, footnote: String? = nil,
         @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.footnote = footnote
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(L10n.text(title), systemImage: icon)
                .font(.app(.subheadline).weight(.semibold))
                .foregroundStyle(.primary)

            content

            if let footnote {
                Text(L10n.text(footnote))
                    .font(.app(.caption))
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(BenchSurface())
        .overlay {
            RoundedRectangle(cornerRadius: 12).strokeBorder(.primary.opacity(0.08), lineWidth: 0.5)
        }
    }
}

/// A label/value row with an optional explainer under the label.
struct BenchRow<Value: View>: View {
    let label: String
    var detail: String? = nil
    @ViewBuilder var value: Value

    init(_ label: String, detail: String? = nil, @ViewBuilder value: () -> Value) {
        self.label = label
        self.detail = detail
        self.value = value()
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.text(label))
                if let detail {
                    Text(L10n.text(detail))
                        .font(.app(.caption))
                        .foregroundStyle(.tertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            value
                .multilineTextAlignment(.trailing)
        }
    }
}

/// A restrained native surface that remains distinct from the window in either appearance.
struct BenchSurface: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color(nsColor: .controlBackgroundColor))
            .overlay {
                RoundedRectangle(cornerRadius: 12).fill(.primary.opacity(0.025))
            }
    }
}
