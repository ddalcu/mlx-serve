import SwiftUI

/// Composer-row context gauge: a percentage and a fill ring, opening a popover
/// with the full breakdown.
///
/// Replaces the full-width bar that used to sit above the composer. That bar
/// spent a permanent strip of the window on a number that matters occasionally,
/// and still couldn't say what was IN the context. A pill costs one control's
/// width and puts the detail one click away.
struct ContextPill: View {
    let stats: ContextWindowStats
    /// Model the window belongs to — shown in the popover, because "78.8K" means
    /// nothing without knowing whose window it is (and LAN chats can be talking
    /// to another Mac's model entirely).
    let modelName: String?
    /// Decode speed of the last timed reply, from the SERVER's own `timings`
    /// (see `ContextWindowStats.speedText`). nil until a reply has been timed.
    let decodeSpeed: Double?
    /// True while this chat is streaming: the figure is moving, so it's marked
    /// live rather than reading as a settled total.
    let isLive: Bool

    @State private var showDetail = false

    private var tint: Color {
        switch stats.pressure {
        case .comfortable: .secondary
        case .warm: .orange
        case .tight: .orange
        case .over: .red
        }
    }

    var body: some View {
        Button { showDetail.toggle() } label: {
            HStack(spacing: 5) {
                Text(stats.percentText)
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(stats.pressure == .comfortable ? Color.secondary : tint)
                ring
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.secondary.opacity(0.12))
            .clipShape(Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .help("Context window — \(stats.percentText) of \(ContextWindowStats.compact(stats.contextLength)) tokens used. Click for the breakdown.")
        .popover(isPresented: $showDetail, arrowEdge: .top) {
            ContextWindowDetail(stats: stats, modelName: modelName, decodeSpeed: decodeSpeed)
        }
    }

    /// Fill ring rather than a bar: at pill size a 4pt-tall bar is unreadable,
    /// and a ring shows the same fraction in a square.
    private var ring: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.30), lineWidth: 2)
            Circle()
                .trim(from: 0, to: stats.barFraction)
                .stroke(tint, style: StrokeStyle(lineWidth: 2, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.15), value: stats.barFraction)
        }
        .frame(width: 12, height: 12)
        .opacity(isLive ? 0.85 : 1)
    }
}

/// The pill's popover: headline percentage, fill bar, and the exact figures.
struct ContextWindowDetail: View {
    let stats: ContextWindowStats
    let modelName: String?
    let decodeSpeed: Double?

    private var tint: Color {
        switch stats.pressure {
        case .comfortable: .green
        case .warm: .orange
        case .tight: .orange
        case .over: .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "brain")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text("Context window")
                        .font(.callout.weight(.semibold))
                    if let modelName, !modelName.isEmpty {
                        Text(modelName)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(12)

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline) {
                    Text(stats.percentText)
                        .font(.system(size: 26, weight: .semibold, design: .rounded))
                        .foregroundStyle(tint)
                    Spacer()
                    Text("\(ContextWindowStats.compact(stats.usedTokens)) / \(ContextWindowStats.compact(stats.contextLength))")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.secondary.opacity(0.18))
                        Capsule().fill(tint)
                            .frame(width: max(0, geo.size.width * stats.barFraction))
                    }
                }
                .frame(height: 6)

                if stats.fromRejectedRequest {
                    Text("Last request overflowed the context window.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            .padding(12)

            Divider()

            VStack(spacing: 6) {
                row(icon: "arrow.up", label: "Prompt", value: stats.promptTokens)
                row(icon: "sum", label: "Used", value: stats.usedTokens, emphasized: true)
                row(icon: "ruler", label: "Remaining", value: stats.remainingTokens)
            }
            .padding(12)

            // Its OWN section, below the divider: everything above counts
            // TOKENS IN THE WINDOW, and a throughput figure in that group would
            // read as one more thing filling the context. Absent until a reply
            // has been timed, rather than a placeholder dash.
            if let speed = ContextWindowStats.speedText(decodeSpeed) {
                Divider()
                textRow(icon: "gauge.with.needle", label: "Decode speed", value: speed)
                    .padding(12)
            }
        }
        .frame(width: 280)
    }

    /// Same layout as `row`, for a figure that is already formatted.
    private func textRow(icon: String, label: String, value: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .frame(width: 14)
            Text(label)
                .font(.callout)
            Spacer()
            Text(value)
                .font(.system(size: 12, design: .monospaced))
        }
    }

    private func row(icon: String, label: String, value: Int, emphasized: Bool = false) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .frame(width: 14)
            Text(label)
                .font(.callout)
            Spacer()
            // Exact figures here — the pill rounds, this is where you check.
            Text(value.formatted(.number.grouping(.automatic)))
                .font(.system(size: 12, design: .monospaced))
                .fontWeight(emphasized ? .semibold : .regular)
        }
    }
}
