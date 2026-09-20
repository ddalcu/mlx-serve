import SwiftUI

/// The header of a section that folds: one label and one chevron in both
/// states, the chevron on the leading edge, and the WHOLE row is the control.
/// A disclosure you can only close by finding a 12-point glyph at the other
/// end of the row is a disclosure that stays open.
struct FoldingSectionHeader: View {
    let title: String
    @Binding var isExpanded: Bool

    var body: some View {
        Button {
            withAnimation { isExpanded.toggle() }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                Text(title)
                Spacer()
            }
            // A section heading like the others: a disclosure is still a
            // section, and it sits in their column.
            .font(.subheadline.weight(.semibold))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
    }
}
