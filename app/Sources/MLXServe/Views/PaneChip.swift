import SwiftUI

/// The small grey chip the Create panes' secondary controls are drawn as: the
/// buttons above a text box, and the icon buttons that hang off a typed field.
/// One place, so a row of them cannot end up with two different greys.
struct PaneChip: ViewModifier {
    /// An icon-only chip hanging off a field: a square the height of the
    /// bezeled control beside it, rather than a shorter pill that reads as a
    /// different class of thing. The side is stated rather than taken from the
    /// row — `maxHeight: .infinity` inside an `HStack` leaves an icon hugged by
    /// its own background, since nothing proposes it the row's height.
    static let side: CGFloat = 24

    var square = false

    func body(content: Content) -> some View {
        if square {
            content
                .font(.caption)
                .foregroundStyle(.primary)
                .frame(width: Self.side, height: Self.side)
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
