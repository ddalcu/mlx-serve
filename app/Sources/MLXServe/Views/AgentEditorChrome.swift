import SwiftUI

/// The Agents editor's geometry — one source of truth for numbers that must
/// agree across independently-written sections.
enum AgentEditorMetrics {
    /// Column edge → content.
    static let contentPadding: CGFloat = 24

    /// The reading measure (cf. `ChatMetrics.contentMaxWidth`).
    static let contentMaxWidth: CGFloat = 760

    /// Between one section and the next.
    static let sectionSpacing: CGFloat = 24
    /// Label → the thing it names. Strictly less than `sectionSpacing`.
    static let labelSpacing: CGFloat = 12

    /// Every card in the editor.
    static let cardRadius: CGFloat = 16
    static let cardPadding: CGFloat = 18
    /// Between rows inside one card.
    static let cardSpacing: CGFloat = 16

    /// A well is a surface sunk INTO a card — the prompt editor. Never rounder
    /// than the card holding it, or its corners bulge out of the outer shape.
    static let wellRadius: CGFloat = 12
    static let wellPadding: CGFloat = 10

    /// The prompt editor's floor — the one field you write paragraphs in.
    static let promptMinHeight: CGFloat = 132

    /// The symbol, and the card beside the name.
    static let avatarSize: CGFloat = 44
    static let avatarPadding: CGFloat = 12

    /// Sized, not stretched: a full-width button reads as a bar.
    static let primaryMaxWidth: CGFloat = 260
}

// MARK: - Surfaces

extension View {
    /// The editor's card surface: one fill, one hairline, one radius. Shared
    /// with the rest of the app's hand-built cards (`NewTaskSheet`) so a card
    /// here and a card there are the same object.
    func agentSurface(radius: CGFloat = AgentEditorMetrics.cardRadius) -> some View {
        background(RoundedRectangle(cornerRadius: radius, style: .continuous)
            .fill(Color.primary.opacity(0.05)))
            .overlay(RoundedRectangle(cornerRadius: radius, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }

    /// A surface sunk into a card — darker than what holds it, so the thing you
    /// type into reads as a recess rather than a second card.
    func agentWell() -> some View {
        background(RoundedRectangle(cornerRadius: AgentEditorMetrics.wellRadius,
                                    style: .continuous)
            .fill(Color.primary.opacity(0.04)))
            .overlay(RoundedRectangle(cornerRadius: AgentEditorMetrics.wellRadius,
                                      style: .continuous)
                .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
    }
}

// MARK: - Structure

/// A section: its title, then whatever it titles.
struct AgentSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: () -> Content

    init(_ title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: AgentEditorMetrics.labelSpacing) {
            Text(title)
                .font(.title3.weight(.semibold))
                .foregroundStyle(.primary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A card — the surface a group of related controls sits on.
struct AgentCard<Content: View>: View {
    var spacing: CGFloat = AgentEditorMetrics.cardSpacing
    @ViewBuilder let content: () -> Content

    init(spacing: CGFloat = AgentEditorMetrics.cardSpacing,
         @ViewBuilder content: @escaping () -> Content) {
        self.spacing = spacing
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: spacing) {
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(AgentEditorMetrics.cardPadding)
        .agentSurface()
    }
}

/// A labelled field: the label above, the field on its own card surface below.
struct AgentLabeledField<Content: View>: View {
    let label: String
    @ViewBuilder let content: () -> Content

    init(_ label: String, @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: AgentEditorMetrics.labelSpacing) {
            Text(label)
                .font(.headline)
                .foregroundStyle(.secondary)
            content()
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, AgentEditorMetrics.cardPadding)
                .padding(.vertical, AgentEditorMetrics.wellPadding)
                .agentSurface()
        }
    }
}

/// One row inside a card: what it is on the left, what it's set to on the
/// right, and the sentence explaining it underneath.
struct AgentEditorRow<Trailing: View>: View {
    let title: String
    var caption: String?
    /// Baseline for a row whose trailing side is a line of text or one control;
    /// `.center` when it is a STACK (a slider over its end labels), where
    /// baseline-aligning the title to the slider's first text lifts it clear
    /// off the row.
    var alignment: VerticalAlignment = .firstTextBaseline
    @ViewBuilder let trailing: () -> Trailing

    init(_ title: String, caption: String? = nil,
         alignment: VerticalAlignment = .firstTextBaseline,
         @ViewBuilder trailing: @escaping () -> Trailing) {
        self.title = title
        self.caption = caption
        self.alignment = alignment
        self.trailing = trailing
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: alignment, spacing: 12) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Spacer(minLength: 8)
                trailing()
            }
            if let caption {
                Text(caption)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

// MARK: - Controls

/// The editor's pill button — a secondary action that acts on the field beside
/// it ("Write it for me"), rather than on the agent.
struct AgentPillButton: View {
    let title: String
    let systemImage: String
    var isBusy = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                if isBusy {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: systemImage)
                        .font(.system(size: 11, weight: .semibold))
                }
                Text(title).font(.callout)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Capsule().fill(Color.primary.opacity(0.09)))
            .overlay(Capsule().strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}
