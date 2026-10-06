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
    @State private var showingConfiguration = false
    private let client = BenchmarkCommunityClient()

    enum ShareState: Equatable { case idle, sending, sent, failed(String) }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    header
                    BenchmarkHighlights(rows: rungRows)

                    HStack(alignment: .top, spacing: 16) {
                        BenchCard("Decode", icon: "waveform.path") {
                            BenchmarkLadderChart(points: ladderPoints)
                        }
                        BenchCard("Prefill", icon: "bolt") {
                            BenchmarkPrefillChart(points: prefillPoints)
                        }
                    }

                    HStack(alignment: .top, spacing: 16) {
                        BenchCard("Results by context", icon: "tablecells") {
                            BenchmarkRungTable(rows: rungRows)
                            Text(resultsFootnote)
                                .font(.app(.caption))
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .frame(width: 580)

                        configuration
                    }
                }
                .padding(20)
            }

            HStack(spacing: 12) {
                shareControl
                Spacer()
                Button { dismiss() } label: {
                    Text("Done").font(.app(.body))
                        .frame(minWidth: 56)
                }
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 14)
            .background(.bar)
            .overlay(alignment: .top) { Divider() }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .frame(width: 1000, height: min(800, (NSScreen.main?.visibleFrame.height ?? 900) - 100))
        .onAppear {
            if case .session(let s) = source, BenchmarkStore.isShared(s) { shareState = .sent }
        }
    }

    private var resultsFootnote: String {
        switch source {
        case .session:
            return L10n.text("Server-measured timings. Context check indicates whether the answer used the planted constant.")
        case .family:
            return L10n.text("Median per context length. Samples is the number of sessions behind each result.")
        }
    }

    private var configuration: some View {
        BenchCard("Configuration", icon: "slider.horizontal.3") {
            systemGrid
            Divider()
            DisclosureGroup(isExpanded: $showingConfiguration) {
                settingsGrid
                    .padding(.top, 12)
            } label: {
                Text("Server settings")
                    .font(.app(.callout, weight: .medium))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                    .onTapGesture { showingConfiguration.toggle() }
            }
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
                    Label("Share to Community", systemImage: "square.and.arrow.up").font(.app(.body))
                }
                .disabled(session.rungs.allSatisfy { !$0.isPublishable })
            case .sending:
                ProgressView().controlSize(.small)
            case .sent:
                Label("Shared", systemImage: "checkmark.circle.fill").font(.app(.body)).foregroundStyle(.green)
            case .failed(let message):
                Label(message, systemImage: "exclamationmark.octagon.fill")
                    .font(.app(.caption)).foregroundStyle(.red)
                Button { Task { await share(session) } } label: { Text("Try Again")
                    .font(.app(.body)) }
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
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Benchmark report", systemImage: "chart.bar.xaxis")
                    .font(.app(.subheadline, weight: .medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Group {
                    switch source {
                    case .session(let s):
                        Text(s.date, format: .dateTime.year().month(.abbreviated).day().hour().minute())
                    case .family(let f):
                        Text("\(f.sessionCount) session\(f.sessionCount == 1 ? "" : "s")")
                    }
                }
                .font(.app(.callout))
                .foregroundStyle(.secondary)
            }

            Text(modelId)
                .font(.app(.title2, weight: .semibold))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 16) {
                Label(hardware.displayName, systemImage: "desktopcomputer")
                    .font(.app(.callout)).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                BenchmarkSettingsChips(settings: settings)
            }
            if case .session(let s) = source, let note = s.note {
                Text(note).font(.app(.callout)).foregroundStyle(.secondary)
            }
        }
    }

    private var settingsGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
            ForEach(BenchmarkSettings.labels.filter { settings[$0.key] != nil }, id: \.key) { entry in
                GridRow {
                    Text(L10n.text(entry.label)).foregroundStyle(.secondary)
                    Text(settings[entry.key] ?? "").monospacedDigit()
                }
            }
            if settings.isEmpty {
                Text("Not recorded (run before settings capture).").foregroundStyle(.tertiary)
            }
        }
        .font(.app(.callout))
    }

    private var systemGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
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
        .font(.app(.callout))
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
