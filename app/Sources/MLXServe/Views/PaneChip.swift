import SwiftUI

/// A menu wearing the chip. `.borderlessButton` hands the label to AppKit,
/// which keeps the text and throws the background away, and draws its own
/// indicator; `.button` renders the label as a real button, so the chip
/// survives and the chevron can live inside it.
struct PaneChipMenu: ViewModifier {
    func body(content: Content) -> some View {
        content
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
    }
}

/// The small grey chip the Create panes' secondary controls are drawn as: the
/// buttons above a text box, and the icon buttons that hang off a typed field.
/// One place, so a row of them cannot end up with two different greys.
struct PaneChip: ViewModifier {
    /// An icon-only chip hanging off a field: a square the height of the
    /// bezeled control beside it, rather than a shorter pill that reads as a
    /// different class of thing. The side is stated rather than taken from the
    /// row — `maxHeight: .infinity` inside an `HStack` leaves an icon hugged by
    /// its own background, since nothing proposes it the row's height. 24 is
    /// the bezel of a `.roundedBorder` field at `.body`, which is what these
    /// hang off — change that font and this follows it.
    static let side: CGFloat = 24

    var square = false
    /// An exact height, for a chip that has to line up with a bezeled field
    /// beside it rather than hug its own text.
    var height: CGFloat? = nil

    func body(content: Content) -> some View {
        if square {
            content
                .font(.caption)
                .foregroundStyle(.primary)
                .frame(width: Self.side, height: Self.side)
                .background(chipShape)
        } else if let height {
            content
                .font(.caption)
                .foregroundStyle(.primary)
                .padding(.horizontal, 9)
                .frame(height: height)
                .background(chipShape)
        } else {
            content
                .font(.caption)
                .foregroundStyle(.primary)
                .padding(.horizontal, 9)
                .padding(.vertical, 4)
                .background(chipShape)
        }
    }

    private var chipShape: some View {
        RoundedRectangle(cornerRadius: 6, style: .continuous)
            .fill(Color.primary.opacity(0.08))
    }
}
