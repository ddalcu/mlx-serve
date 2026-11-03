import SwiftUI

/// The chat window's bottom-left corner: how hard the machine is working and how fast the server is going, which
/// opens the usage dashboard. The gauges follow iTherm's status bar (the author's own menu-bar thermal monitor):
/// green, yellow from 60%, red from 85%.
struct UsageStrip: View {
    @ObservedObject var usage: UsageTracker
    let openDashboard: () -> Void
    @StateObject private var system = SystemLoadSampler()
    @State private var hoveringSpeed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Divider()
            VStack(alignment: .leading, spacing: 3) {
                gauges
                Divider().padding(.vertical, 3)
                speedButton
                // Always laid out, drawn only while a prefill runs, so the strip does not jump.
                LoadBar(fraction: usage.live?.prefillProgress ?? 0, tint: .accentColor, height: 3)
                    .opacity(usage.live?.prefillProgress == nil ? 0 : 1)
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

    /// VoiceOver's reading of the speed line, whose numbers leave their unit to the tooltip.
    static func speedSpoken(decode: Double?, prefill: Double?) -> String {
        func part(_ name: String, _ tps: Double?) -> String {
            guard let tps, tps > 0 else { return L10n.format("%@ not measured yet", L10n.text(name)) }
            return L10n.format("%@ %@ tokens per second", L10n.text(name), ThroughputSnapshot.formatTPS(tps))
        }
        return part("Decode", decode) + ", " + part("Prefill", prefill)
    }

    private var speedButton: some View {
        Button(action: openDashboard) {
            HStack(spacing: 4) {
                SpeedRate(label: "Decode", live: usage.decodeTPS, average: usage.live?.avgDecodeTPS)
                Spacer(minLength: 8)
                SpeedRate(label: "Prefill", live: usage.prefillTPS, average: usage.live?.avgPrefillTPS)
                Image(systemName: "chevron.right").imageScale(.small).foregroundStyle(.tertiary)
            }
            .font(.app(.annotation))
            .padding(.vertical, 3)
            .padding(.horizontal, 4)
            .background(RoundedRectangle(cornerRadius: 5).fill(SidebarRowStyle.fill(selected: false, hovering: hoveringSpeed)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // The hover fill reaches into the gutter, so the words line up with the rows above.
        .padding(.horizontal, -4)
        .onHover { hoveringSpeed = $0 }
        .help(L10n.text("Tokens per second, dimmed when it is the average since the server started. Click to open the usage dashboard."))
        .accessibilityLabel(Self.speedSpoken(decode: usage.decodeTPS ?? usage.live?.avgDecodeTPS,
                                             prefill: usage.prefillTPS ?? usage.live?.avgPrefillTPS))
        .accessibilityHint(L10n.text("Opens the usage dashboard"))
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
private struct SpeedRate: View {
    let label: String
    let live: Double?
    let average: Double?

    var body: some View {
        HStack(spacing: 4) {
            Text(L10n.text(label)).foregroundStyle(.secondary)
            Text(verbatim: ThroughputSnapshot.formatTPS(live ?? average))
                .monospacedDigit()
                .foregroundStyle(live != nil ? .primary : .tertiary)
        }
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
