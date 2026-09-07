import SwiftUI

/// Wraps attachments into rows packed against the trailing edge (they sit in
/// the right-aligned user turn). An `HStack` never wraps.
struct AttachmentFlowLayout: Layout {
    var spacing: CGFloat = 6

    private func rows(for subviews: Subviews, maxWidth: CGFloat) -> [(items: [Int], width: CGFloat, height: CGFloat)] {
        var result: [(items: [Int], width: CGFloat, height: CGFloat)] = []
        var current: [Int] = []
        var width: CGFloat = 0
        var height: CGFloat = 0

        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let widthWithItem = current.isEmpty ? size.width : width + spacing + size.width
            // A row's first item is placed even when it overflows.
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
        // The widest row, not the proposal.
        let width = packed.map(\.width).max() ?? 0
        return CGSize(width: min(width, maxWidth), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let packed = rows(for: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in packed {
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
