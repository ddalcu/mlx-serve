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
        Chart(points) { p in
            LineMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value))
                .foregroundStyle(by: .value("Series", p.series))
                .interpolationMethod(.monotone)
            PointMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value))
                .foregroundStyle(by: .value("Series", p.series))
        }
        .chartForegroundStyleScale([
            Self.decodeSeries: Color.accentColor,
            Self.ceilingSeries: Color.secondary,
        ])
        .chartXScale(domain: rungDomain, type: .log)
        .chartXAxis { rungAxis }
        .chartYAxisLabel("tok/s")
        .chartYScale(domain: .automatic(includesZero: true))
        .chartLegend(position: .top, alignment: .leading)
        .frame(height: 190)
    }
}

/// Prefill per rung on the same axis treatment.
struct BenchmarkPrefillChart: View {
    let points: [BenchmarkChartPoint]

    static func points(_ prefill: [(Int, Double)]) -> [BenchmarkChartPoint] {
        prefill.filter { $0.1 > 0 }.map { BenchmarkChartPoint(targetTokens: $0.0, series: "Prefill", value: $0.1) }
    }

    var body: some View {
        Chart(points) { p in
            LineMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value))
                .foregroundStyle(.orange)
                .interpolationMethod(.monotone)
            PointMark(x: .value("Context", p.targetTokens), y: .value("tok/s", p.value))
                .foregroundStyle(.orange)
        }
        .chartXScale(domain: rungDomain, type: .log)
        .chartXAxis { rungAxis }
        .chartYAxisLabel("prefill tok/s")
        .chartYScale(domain: .automatic(includesZero: true))
        .frame(height: 150)
    }
}

/// Room on both sides of the rungs so the first and last labels are not
/// clipped at the plot edge.
private var rungDomain: ClosedRange<Double> {
    let targets = BenchmarkSuite.ladder.map { Double($0.targetTokens) }
    return (targets.min() ?? 512) * 0.85 ... (targets.max() ?? 16384) * 1.25
}

/// Ticks at the ladder's rungs, labelled "512 / 4k / 8k / 16k".
private var rungAxis: some AxisContent {
    AxisMarks(values: BenchmarkSuite.ladder.map(\.targetTokens)) { value in
        AxisGridLine()
        AxisValueLabel {
            if let tokens = value.as(Int.self) {
                Text(BenchmarkSuite.title(forTarget: tokens))
            }
        }
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
        Grid(alignment: .trailing, horizontalSpacing: 14, verticalSpacing: 6) {
            GridRow {
                header("Rung").gridColumnAlignment(.leading)
                if showsPrompt { header("Prompt") }
                header("Prefill")
                header("Decode")
                header("Ceiling")
                header("TTFT")
                if showsSamples { header("n") } else { header("Ctx") }
            }
            ForEach(rows) { row in
                GridRow {
                    Text(BenchmarkSuite.title(forTarget: row.targetTokens))
                        .fontWeight(.medium)
                        .gridColumnAlignment(.leading)
                    if showsPrompt { cell(row.promptTokens > 0 ? "\(row.promptTokens)" : "—") }
                    cell(BenchmarkFormat.rate(row.prefillTps, decimals: 0))
                    cell(BenchmarkFormat.rate(row.decodeTps, decimals: 1)).fontWeight(.semibold)
                    cell(BenchmarkFormat.rate(row.ceilingDecodeTps, decimals: 1))
                    cell(BenchmarkFormat.rate(row.ttftMs, decimals: 0))
                    if showsSamples {
                        Text("\(row.samples ?? 0)")
                            .monospacedDigit()
                            .foregroundStyle((row.samples ?? 0) == 1 ? .orange : .secondary)
                    } else {
                        Image(systemName: row.contextUsed == true ? "checkmark" : "minus")
                            .foregroundStyle(row.contextUsed == true ? .green : .secondary)
                            .help(row.contextUsed == true
                                  ? "The answer used the constant planted in the context"
                                  : "The answer did not use the planted constant")
                    }
                }
            }
        }
        .font(.app(.callout))
    }

    private func header(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.app(.caption2, weight: .semibold))
            .tracking(0.5)
            .foregroundStyle(.tertiary)
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
                Text(chip)
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
            Label(title, systemImage: icon)
                .font(.app(.subheadline).weight(.semibold))
                .foregroundStyle(.secondary)

            content

            if let footnote {
                Text(footnote)
                    .font(.app(.caption))
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.28), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary, lineWidth: 0.5)
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
                Text(label)
                if let detail {
                    Text(detail)
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
