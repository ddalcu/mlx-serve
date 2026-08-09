import SwiftUI

/// The Agents editor's geometry — ONE source of truth for the numbers that have
/// to agree across a column of independently-written sections.
///
/// These used to belong to a grouped `Form`, which was the problem: a Form owns
/// the card radius, the row insets, the label typography and the section
/// spacing, so none of it could be specified, and it right-aligns a TextField's
/// text — which put the agent's own NAME hard against the trailing edge of its
/// field. Owning the numbers means they can now drift, so the relationships
/// between them are pinned by `AgentEditorLayoutTests` rather than left as
/// literals scattered through a view body.
///
/// Derived from the design frame (Figma `agents-config-panel`), retyped to
/// macOS control sizes: the frame is drawn in Inter at web scale, where 14pt
/// body reads as SF 13 — so the geometry is the frame's and the type is the
/// platform's semantic scale (`.title3` section titles, `.headline` labels,
/// `.subheadline` captions), which is also the only way this survives a
/// Dynamic Type change.
enum AgentEditorMetrics {
    /// Column edge → content. The frame's 32 at 861pt wide; the detail column
    /// here can be half that, where 32 is a sixth of the width.
    static let contentPadding: CGFloat = 24

    /// The reading measure, the same idea as `ChatMetrics.contentMaxWidth`: a
    /// form field run the whole way across a 1400pt window is a field you scan
    /// rather than read, and the frame's own content is 797pt wide.
    static let contentMaxWidth: CGFloat = 760

    /// Between one section and the next.
    static let sectionSpacing: CGFloat = 24
    /// Between a label (or a section title) and the thing it names. Strictly
    /// less than `sectionSpacing`, or the column reads as one undifferentiated
    /// list of settings — which is exactly what it did.
    static let labelSpacing: CGFloat = 12

    /// Every card in the editor: the surface a group of related controls sits
    /// on.
    static let cardRadius: CGFloat = 16
    static let cardPadding: CGFloat = 18
    /// Between rows inside one card.
    static let cardSpacing: CGFloat = 16

    /// A well is a surface sunk INTO a card — the prompt editor. Never rounder
    /// than the card holding it, or its corners bulge out of the outer shape.
    static let wellRadius: CGFloat = 12
    static let wellPadding: CGFloat = 10

    /// The prompt editor's floor. It is the one field you write paragraphs in.
    static let promptMinHeight: CGFloat = 132

    /// The symbol, and the card it sits in beside the name.
    static let avatarSize: CGFloat = 44
    static let avatarPadding: CGFloat = 12

    /// The primary action. Sized rather than stretched: a full-width button
    /// reads as a bar, and this one sits in the flow of the column.
    static let primaryMaxWidth: CGFloat = 260

    /// The column's actual width for a given available width — the measure, or
    /// what the window gives minus its padding when that is narrower. Never
    /// non-positive: a negative frame width is a crash in some SwiftUI
    /// containers and a silently invisible column in the rest, and layout does
    /// report zero.
    static func columnWidth(available: CGFloat) -> CGFloat {
        max(1, min(contentMaxWidth, available - contentPadding * 2))
    }
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
///
/// One builder, so two headings cannot stop matching. The title is `.title3`
/// weighted rather than `.headline`, which on macOS is the same 13pt as the
/// labels under it — a heading the same size as its content is not a heading.
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
///
/// The label sits OUTSIDE the field rather than beside it inside — a Form puts
/// it inside and then right-aligns the value, which is what had the agent's
/// name pinned to the trailing edge of a field with the word "Name" at the
/// other end of it.
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
///
/// The caption belongs to the ROW, not to the card — a card-level footnote
/// can't say which of three rows it is about.
struct AgentEditorRow<Trailing: View>: View {
    let title: String
    var caption: String?
    @ViewBuilder let trailing: () -> Trailing

    init(_ title: String, caption: String? = nil,
         @ViewBuilder trailing: @escaping () -> Trailing) {
        self.title = title
        self.caption = caption
        self.trailing = trailing
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
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
