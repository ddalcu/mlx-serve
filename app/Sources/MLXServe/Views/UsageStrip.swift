import SwiftUI

/// The chat window's bottom-left corner: how hard the machine is working, how fast the server is going, and the
/// running token tally, which opens the usage dashboard. The gauges follow iTherm's status bar (the author's own
/// menu-bar thermal monitor): green, yellow from 60%, red from 85%.
struct UsageStrip: View {
    @ObservedObject var usage: UsageTracker
    let openDashboard: () -> Void
    @StateObject private var system = SystemLoadSampler()
    @State private var hoveringTotal = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Divider()
            VStack(alignment: .leading, spacing: 3) {
                gauges
                Divider().padding(.vertical, 3)
                SpeedRow(label: "Decode", live: usage.decodeTPS, average: usage.live?.avgDecodeTPS)
                SpeedRow(label: "Prefill", live: usage.prefillTPS, average: usage.live?.avgPrefillTPS)
                // Always laid out, drawn only while a prefill runs, so the rows below do not jump.
                LoadBar(fraction: usage.live?.prefillProgress ?? 0, tint: .accentColor, height: 3)
                    .opacity(usage.live?.prefillProgress == nil ? 0 : 1)
                totalButton
            }
            .font(.app(.annotation))
            .padding(.horizontal, ChatMetrics.sidebarGutter + 4)
            .padding(.top, 8)
            .padding(.bottom, 10)
        }
        .onAppear {
            system.start()
            usage.load()
        }
        .onDisappear { system.stop() }
    }

    @ViewBuilder private var gauges: some View {
        if let load = system.load {
            LoadRow(label: "GPU", percent: Double(load.gpu), value: Self.gpuValue(load))
            if let ane = load.ane { LoadRow(label: "ANE", percent: ane, value: "\(Int(ane))%") }
            LoadRow(label: "MEM", percent: load.memoryPercent, value: load.memoryLabel)
        }
    }

    static func gpuValue(_ load: SystemLoad) -> String {
        load.gpuTemperature.map { "\(load.gpu)% · \(Int($0))°" } ?? "\(load.gpu)%"
    }

    private var totalButton: some View {
        Button(action: openDashboard) {
            HStack(spacing: 4) {
                Text("Total Tokens").foregroundStyle(.secondary)
                Spacer(minLength: 4)
                Text(verbatim: UsageFormat.compact(usage.totalTokens)).monospacedDigit()
                Image(systemName: "chevron.right").imageScale(.small).foregroundStyle(.tertiary)
            }
            .font(.app(.annotation))
            .padding(.vertical, 3)
            .padding(.horizontal, 4)
            .background(RoundedRectangle(cornerRadius: 5).fill(SidebarRowStyle.fill(selected: false, hovering: hoveringTotal)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // The hover fill reaches into the gutter, so the words line up with the rows above.
        .padding(.horizontal, -4)
        .onHover { hoveringTotal = $0 }
        .help(L10n.format("%@ tokens since the last reset. Click to open the usage dashboard.",
                          ThroughputSnapshot.formatTokens(usage.totalTokens)))
    }
}

private struct LoadRow: View {
    let label: String
    let percent: Double
    let value: String

    var body: some View {
        HStack(spacing: 6) {
            Text(verbatim: label).foregroundStyle(.secondary).frame(width: 28, alignment: .leading)
            LoadBar(fraction: SystemLoad.fraction(percent), tint: Self.tint(SystemLoad.level(percent)), height: 5)
            Text(verbatim: value).monospacedDigit().frame(width: 70, alignment: .trailing)
        }
    }

    private static func tint(_ level: SystemLoad.Level) -> Color {
        switch level {
        case .normal: return .green
        case .busy: return .yellow
        case .saturated: return .red
        }
    }
}

/// Tokens per second: the live rate while tokens move, the session average (in a quieter ink) at rest.
private struct SpeedRow: View {
    let label: String
    let live: Double?
    let average: Double?

    var body: some View {
        let tps = live ?? average
        HStack {
            Text(L10n.text(label)).foregroundStyle(.secondary)
            Spacer(minLength: 4)
            Text(verbatim: tps.flatMap { $0 > 0 ? L10n.format("%@ tok/s", ThroughputSnapshot.formatTPS($0)) : nil } ?? "—")
                .monospacedDigit()
                .foregroundStyle(live != nil ? .primary : .tertiary)
        }
        .help(live == nil && average != nil ? L10n.text("Average since the server started") : "")
    }
}

/// A thin capsule bar: a quiet track and a fill that is `fraction` of it.
struct LoadBar: View {
    let fraction: Double
    let tint: Color
    var height: CGFloat = 5

    var body: some View {
        Capsule()
            .fill(UsagePalette.idle)
            .overlay(alignment: .leading) {
                GeometryReader { geo in
                    Capsule().fill(tint).frame(width: geo.size.width * min(max(fraction, 0), 1))
                }
            }
            .frame(height: height)
    }
}
