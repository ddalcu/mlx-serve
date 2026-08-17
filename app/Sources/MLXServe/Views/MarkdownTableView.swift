import SwiftUI

/// A parsed markdown table as a real grid: semibold header row, one divider,
/// no vertical borders — the "minimal GFM" look chat UIs use, replacing the
/// monospaced space-padded text `MarkdownSegmenter`/`MarkdownText` used to
/// assemble inline. Columns take a proportional share of the available width
/// (`MarkdownTable.columnWidths`) so a short "Tip" column doesn't eat as much
/// room as a long "Explanation" one, and cells wrap instead of overflowing.
struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]
    let alignments: [TableAlignment]

    @Environment(\.colorScheme) private var colorScheme
    /// The table's laid-out width, measured once via `onGeometryChange`. Zero
    /// for the first frame, during which columns fall back to equal flexible
    /// widths rather than collapsing to zero.
    @State private var tableWidth: CGFloat = 0

    private var theme: LaTeXTheme { colorScheme == .dark ? .dark : .light }
    private var fractions: [CGFloat] { MarkdownTable.columnWidths(headers: headers, rows: rows) }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            row(headers, bold: true)
            Divider()
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                row(r, bold: false)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { tableWidth = $0 }
        .textSelection(.enabled)
    }

    private func row(_ cells: [String], bold: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(headers.indices, id: \.self) { j in
                cell(j < cells.count ? cells[j] : "", column: j, bold: bold)
            }
        }
        .padding(.vertical, 4)
    }

    private func alignment(_ column: Int) -> Alignment {
        guard column < alignments.count else { return .leading }
        switch alignments[column] {
        case .left: return .leading
        case .right: return .trailing
        case .center: return .center
        }
    }

    @ViewBuilder
    private func cell(_ text: String, column: Int, bold: Bool) -> some View {
        let attributed = MarkdownText.renderInline(text, theme: theme, weight: bold ? .semibold : .regular)
        let label = Text(AttributedString(attributed))
            .padding(.horizontal, 8)
        if tableWidth > 0, column < fractions.count {
            label.frame(width: tableWidth * fractions[column], alignment: alignment(column))
        } else {
            label.frame(maxWidth: .infinity, alignment: alignment(column))
        }
    }
}