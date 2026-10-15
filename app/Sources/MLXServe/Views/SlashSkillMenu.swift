import SwiftUI

/// The composer's "/" menu: the skills you can invoke by name, drawn as a
/// sibling directly above the input. Selection and filtering are the
/// composer's (`SlashCommands`); this only draws and reports a pick.
struct SlashSkillMenu: View {
    let matches: [SkillSummary]
    let selection: Int
    let onPick: (SkillSummary) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(matches.enumerated()), id: \.element.id) { index, skill in
                Button { onPick(skill) } label: {
                    HStack(spacing: 8) {
                        Text("/\(skill.name)")
                            .font(.app(.callout).weight(.medium))
                        Text(L10n.text(skill.description))
                            .font(.app(.caption))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    // The row is the target, not just its text.
                    .contentShape(Rectangle())
                    .background(index == selection ? Color.accentColor.opacity(0.18) : Color.clear)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.vertical, 4)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
        )
    }
}

extension View {
    /// The "/" menu hung 8pt above this view (the composer), over the transcript: inside the bar it grew the
    /// composer on every keystroke, and an NSPopover would take the keyboard from the field. An alignment
    /// guide does not reach the overlay through the `if`, so the menu sits at the bottom of a zero-height box.
    func slashSkillMenu(matches: [SkillSummary], selection: Int, onPick: @escaping (SkillSummary) -> Void) -> some View {
        overlay(alignment: .top) {
            VStack(spacing: 0) {
                if !matches.isEmpty {
                    SlashSkillMenu(matches: matches, selection: selection, onPick: onPick)
                        .shadow(color: .black.opacity(0.28), radius: 14, y: 4)
                }
            }
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, 8)
            .frame(height: 0, alignment: .bottom)
        }
    }
}
