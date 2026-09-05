import SwiftUI

/// Wraps attachments into rows, each row packed against the TRAILING edge.
///
/// A plain `HStack` cannot do this: it never wraps, so six photos in a message
/// were squeezed into one strip until each was a sliver, and the strip then
/// spanned the whole column and left the text above it hanging at a different
/// edge. This is the one thing SwiftUI's stacks do not offer and `Layout` does.
///
/// Trailing, because these sit in the user's own turn, which is right-aligned:
/// a row of pictures packed to the left under a bubble packed to the right
/// reads as two unrelated things.
struct AttachmentFlowLayout: Layout {
    var spacing: CGFloat = 6

    /// Rows as (index range, total width). Computed once per pass and handed to
    /// `placeSubviews`, which would otherwise repeat the packing and could
    /// disagree with the height that was already reported.
    private func rows(for subviews: Subviews, maxWidth: CGFloat) -> [(items: [Int], width: CGFloat, height: CGFloat)] {
        var result: [(items: [Int], width: CGFloat, height: CGFloat)] = []
        var current: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0

        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let widthWithItem = current.isEmpty ? size.width : width + spacing + size.width
            // The first item on a row is placed even when it overflows: a row
            // has to hold something, and an over-wide attachment is clamped by
            // its own frame long before it reaches this.
            if !current.isEmpty && widthWithItem > maxWidth {
                result.append((current, width, height))
                current = [index]
                width = size.width
                height = size.height
            } else {
                current.append(index)
                width = widthWithItem
                height = max(height, size.height)
            }
        }
        if !current.isEmpty { result.append((current, width, height)) }
        return result
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard !subviews.isEmpty else { return .zero }
        let maxWidth = proposal.width ?? .infinity
        let packed = rows(for: subviews, maxWidth: maxWidth)
        let height = packed.reduce(0) { $0 + $1.height } + spacing * CGFloat(max(0, packed.count - 1))
        // The widest row, not the proposal: a single attachment must not claim
        // the full column just because it was offered.
        let width = packed.map(\.width).max() ?? 0
        return CGSize(width: min(width, maxWidth), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let packed = rows(for: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in packed {
            // Right-packed: the row starts wherever it has to in order to END
            // at the trailing edge.
            var x = bounds.maxX - row.width
            for index in row.items {
                let size = subviews[index].sizeThatFits(.unspecified)
                subviews[index].place(
                    at: CGPoint(x: x, y: y + (row.height - size.height) / 2),
                    proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }
}
