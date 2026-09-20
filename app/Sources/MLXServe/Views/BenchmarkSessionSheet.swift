import SwiftUI

/// The detail behind a History or Community row: charts, rung table, the
/// server settings that shaped the numbers, and the machine.
///
/// One view for both panes. A History row is one session (a date); a
/// Community row is a family of sessions (medians, n per rung, no date).
struct BenchmarkSessionSheet: View {

    enum Source {
        case session(BenchmarkSession)
        case family(BenchmarkStore.CellFamily)
    }

    let source: Source
    @Environment(\.dismiss) private var dismiss

    /// A History session not yet sent to the board gets its own Share here,
    /// so a run whose Share was skipped at the time is not lost to the board.
    @State private var shareState: ShareState = .idle
    private let client = BenchmarkCommunityClient()

    enum ShareState: Equatable { case idle, sending, sent, failed(String) }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header

            // Two columns: the facts on the left, the numbers on the right.
            // Stacked, the sheet ran taller than a laptop screen.
            HStack(alignment: .top, spacing: 16) {
                VStack(spacing: 16) {
                    BenchCard("Settings", icon: "slider.horizontal.3") { settingsGrid }
                    BenchCard("System", icon: "desktopcomputer") { systemGrid }
                }
                .frame(width: 300)

                VStack(spacing: 16) {
                    BenchCard("Decode", icon: "chart.xyaxis.line") {
                        BenchmarkLadderChart(points: ladderPoints)
                    }
                    BenchCard("Prefill", icon: "chart.line.uptrend.xyaxis") {
                        BenchmarkPrefillChart(points: prefillPoints)
                    }
                    BenchCard("Rungs", icon: "tablecells") {
                        BenchmarkRungTable(rows: rungRows)
                    }
                }
                .frame(maxWidth: .infinity)
            }

            HStack(spacing: 12) {
                shareControl
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 1000)
        .onAppear {
            if case .session(let s) = source, BenchmarkStore.isShared(s) { shareState = .sent }
        }
    }

    @ViewBuilder
    private var shareControl: some View {
        if case .session(let session) = source {
            switch shareState {
            case .idle:
                Button {
                    Task { await share(session) }
                } label: {
                    Label("Share to Community", systemImage: "square.and.arrow.up")
                }
                .disabled(session.rungs.allSatisfy { !$0.isPublishable })
            case .sending:
                ProgressView().controlSize(.small)
            case .sent:
                Label("Shared", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.octagon.fill")
                    .font(.caption).foregroundStyle(.red)
                Button("Try Again") { Task { await share(session) } }
                    .controlSize(.small)
            }
        }
    }

    private func share(_ session: BenchmarkSession) async {
        shareState = .sending
        let outcome = await client.submit(BenchmarkStore.unsent(session.rungs))
        BenchmarkStore.markShared(outcome.sentIds)
        shareState = outcome.error.map { .failed($0.localizedDescription) } ?? .sent
    }

    // MARK: - Pieces

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(modelId).font(.title3.weight(.semibold)).lineLimit(1).truncationMode(.middle)
            HStack(spacing: 8) {
                Text(hardware.displayName)
                Text("·")
                switch source {
                case .session(let s):
                    Text(s.date, format: .dateTime.year().month(.abbreviated).day().hour().minute())
                case .family(let f):
                    Text("\(f.sessionCount) session\(f.sessionCount == 1 ? "" : "s")")
                }
            }
            .font(.callout)
            .foregroundStyle(.secondary)
            BenchmarkSettingsChips(settings: settings)
            if case .session(let s) = source, let note = s.note {
                Text(note).font(.callout).foregroundStyle(.secondary)
            }
        }
    }

    private var settingsGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
            ForEach(BenchmarkSettings.labels.filter { settings[$0.key] != nil }, id: \.key) { entry in
                GridRow {
                    Text(entry.label).foregroundStyle(.secondary)
                    Text(settings[entry.key] ?? "").monospacedDigit()
                }
            }
            if settings.isEmpty {
                Text("Not recorded (run before settings capture).").foregroundStyle(.tertiary)
            }
        }
        .font(.callout)
    }

    private var systemGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
            GridRow { Text("Chip").foregroundStyle(.secondary); Text(hardware.chip) }
            GridRow { Text("GPU cores").foregroundStyle(.secondary); Text(hardware.gpuCores > 0 ? "\(hardware.gpuCores)" : "—") }
            GridRow { Text("Memory").foregroundStyle(.secondary); Text("\(hardware.ramGB) GB") }
            GridRow { Text("macOS").foregroundStyle(.secondary); Text(hardware.osVersion.isEmpty ? "—" : hardware.osVersion) }
            GridRow { Text("On battery").foregroundStyle(.secondary); Text(hardware.onBattery ? "yes" : "no") }
            GridRow { Text("Server").foregroundStyle(.secondary); Text(engineVersion) }
            if case .session(let s) = source {
                GridRow {
                    Text("Drift").foregroundStyle(.secondary)
                    Text(BenchmarkDrift.summary(first: s.driftBaselineTps, last: s.driftDecodeTps, percent: s.driftPercent))
                        .foregroundStyle(driftColor(s.driftPercent))
                        .help("The smallest rung's decode, measured again after the whole ladder. Beyond ±10% the run's numbers are a range, not figures.")
                }
            }
        }
        .font(.callout)
    }

    private func driftColor(_ percent: Double?) -> Color {
        switch BenchmarkDrift.verdict(percent: percent) {
        case .steady: return .green
        case .degraded, .improved: return .orange
        case .unknown: return .secondary
        }
    }

    // MARK: - Data

    private var modelId: String {
        switch source { case .session(let s): return s.modelId; case .family(let f): return f.modelId }
    }
    private var hardware: BenchmarkHardware {
        switch source { case .session(let s): return s.hardware; case .family(let f): return f.hardware }
    }
    private var settings: [String: String] {
        switch source { case .session(let s): return s.settings; case .family(let f): return f.settings }
    }
    private var engineVersion: String {
        switch source {
        case .session(let s): return s.engineVersion
        case .family(let f): return f.settings["version"] ?? "—"
        }
    }
    private var ladderPoints: [BenchmarkChartPoint] {
        switch source {
        case .session(let s):
            return BenchmarkLadderChart.points(
                decode: s.rungs.map { ($0.effectiveTargetTokens, $0.decodeTps) },
                ceiling: s.rungs.map { ($0.effectiveTargetTokens, $0.ceilingDecodeTps ?? 0) })
        case .family(let f):
            return BenchmarkLadderChart.points(
                decode: f.rungs.map { ($0.targetTokens, $0.decodeTps) },
                ceiling: f.rungs.map { ($0.targetTokens, $0.ceilingDecodeTps) })
        }
    }
    private var prefillPoints: [BenchmarkChartPoint] {
        switch source {
        case .session(let s): return BenchmarkPrefillChart.points(s.rungs.map { ($0.effectiveTargetTokens, $0.prefillTps) })
        case .family(let f): return BenchmarkPrefillChart.points(f.rungs.map { ($0.targetTokens, $0.prefillTps) })
        }
    }
    private var rungRows: [BenchmarkRungTable.Row] {
        switch source {
        case .session(let s): return BenchmarkRungTable.rows(s.rungs)
        case .family(let f): return BenchmarkRungTable.rows(f)
        }
    }
}
