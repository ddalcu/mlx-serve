import SwiftUI

/// Packs its subviews along a row at their own widths and wraps to the next
/// row when the next one does not fit.
///
/// A grid was the obvious container and the wrong one: `LazyVGrid` divides the
/// width into equal columns whether the cells need them or not, so a 64pt box
/// and a 210pt menu sat a column apart with nothing between them. Here the gap
/// is the gap.
struct FlowLayout: Layout {
    /// Between two items on a row.
    var spacing: CGFloat = 10
    /// Between rows.
    var rowSpacing: CGFloat = 10

    /// Which items land on which row. Greedy, and an item wider than the
    /// container takes a row of its own rather than opening an empty one.
    static func rows(widths: [CGFloat], maxWidth: CGFloat, spacing: CGFloat) -> [[Int]] {
        var rows: [[Int]] = []
        var row: [Int] = []
        var used: CGFloat = 0
        for (i, w) in widths.enumerated() {
            let needed = row.isEmpty ? w : used + spacing + w
            if !row.isEmpty && needed > maxWidth {
                rows.append(row)
                row = [i]
                used = w
            } else {
                row.append(i)
                used = needed
            }
        }
        if !row.isEmpty { rows.append(row) }
        return rows
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let maxWidth = proposal.width ?? .infinity
        let rows = Self.rows(widths: sizes.map(\.width), maxWidth: maxWidth, spacing: spacing)
        let width = rows.map { rowWidth($0, sizes) }.max() ?? 0
        let height = rows.reduce(0) { $0 + rowHeight($1, sizes) }
            + rowSpacing * CGFloat(max(0, rows.count - 1))
        // Never report more than we were offered: this is what keeps a long
        // menu label from setting the pane's minimum width. A subview wider
        // than the offer overflows its row instead of widening the column,
        // which is the safer of the two failures here (a column whose minimum
        // exceeds what the split view gives it slides under the sidebar).
        return CGSize(width: min(width, maxWidth), height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let sizes = subviews.map { $0.sizeThatFits(.unspecified) }
        let rows = Self.rows(widths: sizes.map(\.width), maxWidth: bounds.width, spacing: spacing)
        var y = bounds.minY
        for row in rows {
            let height = rowHeight(row, sizes)
            var x = bounds.minX
            for i in row {
                // Bottom-aligned: these are label-over-control cells, and a
                // taller neighbour must not lift the controls off one line.
                subviews[i].place(at: CGPoint(x: x, y: y + height - sizes[i].height),
                                  proposal: ProposedViewSize(sizes[i]))
                x += sizes[i].width + spacing
            }
            y += height + rowSpacing
        }
    }

    private func rowWidth(_ row: [Int], _ sizes: [CGSize]) -> CGFloat {
        row.reduce(0) { $0 + sizes[$1].width } + spacing * CGFloat(max(0, row.count - 1))
    }

    private func rowHeight(_ row: [Int], _ sizes: [CGSize]) -> CGFloat {
        row.map { sizes[$0].height }.max() ?? 0
    }
}
