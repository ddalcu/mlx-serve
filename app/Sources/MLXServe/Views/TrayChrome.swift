import SwiftUI

/// Shared visual system for the menu-bar panel.
enum TrayMetrics {
    /// Panel width. 340 rather than the old 320: the model picker's collapsed
    /// title is a full repo id ("mlx-community/gemma-4-12b-it-4bit") and every
    /// point of gutter came straight out of it.
    static let width: CGFloat = 340
    /// The one horizontal inset. Cards, headers, footer — everything.
    static let gutter: CGFloat = 14
    /// Between a section header and its card, and between rows inside a card.
    static let rowSpacing: CGFloat = 8
    /// Between sections. Whitespace is the grouping now, not hairlines.
    static let sectionSpacing: CGFloat = 12
    static let cardRadius: CGFloat = 10
    static let cardPadding: CGFloat = 10
}

// MARK: - Status chip

/// Presentation for the header's status chip. Pure so the "never render the raw
/// error message" rule is testable without a view: `ServerStatus.label`
/// interpolates the whole failure ("Error: MISSING WEIGHT: …"), which at this
/// width truncates to nothing useful. The full text stays one hover away and is
/// also shown verbatim in the error row under the server controls.
struct TrayStatusChipModel: Equatable {
    enum Tone: Equatable {
        case running, starting, stopped, error

        var color: Color {
            switch self {
            case .running:  .green
            case .starting: .orange
            case .stopped:  .secondary
            case .error:    .red
            }
        }
    }

    let label: String
    let tone: Tone

    init(status: ServerStatus) {
        switch status {
        case .running:  (label, tone) = ("Running", .running)
        case .starting: (label, tone) = ("Loading", .starting)
        case .stopped:  (label, tone) = ("Stopped", .stopped)
        case .error:    (label, tone) = ("Error", .error)
        }
    }
}

/// Dot + word in a capsule. Replaces the old pair of indicators (a bare dot in
/// the header AND a "Running" label down in the server row) with one thing that
/// says the state once, where the eye lands first.
struct TrayStatusChip: View {
    let status: ServerStatus

    var body: some View {
        let model = TrayStatusChipModel(status: status)
        return HStack(spacing: 5) {
            Circle()
                .fill(model.tone.color)
                .frame(width: 6, height: 6)
            Text(model.label)
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(Capsule().fill(Color.primary.opacity(0.06)))
        // The chip is a glance; the whole error text lives here.
        .help(status.label)
    }
}

// MARK: - Section chrome

/// An all-caps section label. Same treatment as the welcome screen's
/// "BEST MODELS FOR YOUR MAC", so a user moving between the two surfaces sees
/// one design, not two.
struct TraySectionHeader: View {
    let title: String
    /// Optional trailing text (a count, a size) — never a control; controls that
    /// belong to a section ride the card's own rows.
    var detail: String?

    var body: some View {
        HStack(spacing: 6) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .tracking(0.6)
            Spacer(minLength: 0)
            if let detail {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .foregroundStyle(.secondary)
    }
}

/// The grouping primitive. A subtle filled, hairline-stroked rounded rect —
/// exactly the welcome screen's card, at tray scale.
struct TrayCard<Content: View>: View {
    var padding: CGFloat = TrayMetrics.cardPadding
    var spacing: CGFloat = TrayMetrics.rowSpacing
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(padding)
        .background(
            RoundedRectangle(cornerRadius: TrayMetrics.cardRadius, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: TrayMetrics.cardRadius, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.07), lineWidth: 1)
        )
    }
}

/// A hairline used INSIDE a card to separate its rows. Inset from the card's
/// padding so it reads as a row separator rather than a section break.
struct TrayRowSeparator: View {
    var body: some View {
        Rectangle()
            .fill(Color.primary.opacity(0.08))
            .frame(height: 1)
            .padding(.vertical, 1)
    }
}

// MARK: - Rows

/// One feature row: symbol, title, optional subtitle, trailing control. Voice,
/// Quick Launcher and the Agent Sandbox badge are the same kind of thing — a
/// capability you switch on or step into — and rendering them three different
/// ways made the panel read as three panels.
struct TrayFeatureRow<Trailing: View>: View {
    let icon: String
    let title: String
    var subtitle: String?
    /// Tints the symbol when the feature is on — colour alone carries the state,
    /// as it does on the composer's mode discs.
    var isOn: Bool = false
    var tint: Color = .accentColor
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(isOn ? tint : Color.secondary)
                .frame(width: 17)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.subheadline.weight(.medium))
                if let subtitle {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 6)
            trailing
        }
    }
}

/// A disclosure header with an optional sibling action button (Endpoints +
/// Metrics, Download Models + Browse). The chevron and title are ONE button
/// with its own `contentShape` — macOS hit-tests only the chevron of a real
/// `DisclosureGroup` label — and the action sits beside it as a separate
/// target, never nested in the label.
struct TrayDisclosureHeader<Accessory: View>: View {
    let title: String
    @Binding var isExpanded: Bool
    @ViewBuilder var accessory: Accessory

    var body: some View {
        HStack(spacing: 6) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                        .frame(width: 9)
                    Text(title)
                        .font(.subheadline.weight(.medium))
                    Spacer(minLength: 0)
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            accessory
        }
    }
}

extension TrayDisclosureHeader where Accessory == EmptyView {
    init(title: String, isExpanded: Binding<Bool>) {
        self.init(title: title, isExpanded: isExpanded) { EmptyView() }
    }
}

/// A compact bordered accessory button (Browse, Metrics) — one style for the
/// small actions that ride a section header.
struct TrayAccessoryButton: View {
    let title: String
    let icon: String
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                Text(title)
            }
            .font(.caption)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .help(help)
    }
}

/// A media-generation tile: symbol over label, equal widths, no bezel. Four
/// `.bordered` pill buttons with `minimumScaleFactor(0.7)` were a width
/// overflow admitting itself — stacking the glyph gives each target twice the
/// area at the same footprint and lets the labels render at full size.
struct TrayTile: View {
    let icon: String
    let title: String
    let help: String
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 15, weight: .regular))
                    .foregroundStyle(Color.accentColor)
                Text(title)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.primary.opacity(hovering ? 0.10 : 0.05))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.07), lineWidth: 1)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // Enter/exit only — no continuous redraw, so the popover's hit-testing
        // stays alive (see the VoiceTrayPanel dot comment).
        .onHover { hovering = $0 }
        .help(help)
    }
}
