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
                            .font(.callout.weight(.medium))
                        Text(skill.description)
                            .font(.caption)
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
