import SwiftUI

/// Breathing room reserved after each cell's wrapped text, before the next
/// column starts — without it, a wrapped cell's last word and the next
/// column's first word read as run together. Also subtracted from natural
/// widths below, so the measured column already accounts for it.
private let tableCellTrailingPadding: CGFloat = 8

/// A parsed markdown table as a real grid: semibold header row, one divider,
/// no vertical borders — the "minimal GFM" look chat UIs use. Columns take
/// their *natural* width from content (`NSAttributedString.size()`, the
/// same rendering used to draw the cell) — a short "Tip" column stays tight,
/// it doesn't stretch to share the row with a long "Explanation" one. Only
/// when the sum of natural widths exceeds the reading column's width do
/// columns scale down together (and wrap) to fit — the table never forces
/// itself to the full measure the way the surrounding prose never does.
struct MarkdownTableView: View {
    let headers: [String]
    let rows: [[String]]
    let alignments: [TableAlignment]

    @Environment(\.colorScheme) private var colorScheme
    /// The reading column's measured width, via `onGeometryChange` below.
    /// Zero for the first frame, before which columns fall back to a
    /// nominal width rather than collapsing to zero.
    @State private var tableWidth: CGFloat = 0

    private var theme: LaTeXTheme { colorScheme == .dark ? .dark : .light }

    /// Each column's natural width: the widest of its header and data
    /// cells, as actually rendered (bold header, regular data), plus the
    /// same trailing padding the cell view reserves. Unbounded — scaling
    /// to the available width happens separately in `columnWidths`.
    private var naturalColumnWidths: [CGFloat] {
        headers.indices.map { column in
            var widest: CGFloat = MarkdownText.renderInline(headers[column], theme: theme, weight: .semibold).size().width
            for r in rows {
                let text = column < r.count ? r[column] : ""
                widest = max(widest, MarkdownText.renderInline(text, theme: theme, weight: .regular).size().width)
            }
            return ceil(widest) + tableCellTrailingPadding
        }
    }

    /// Per-column pixel widths: natural widths, tight to content, unless
    /// their sum overflows the measured reading column — then every column
    /// is scaled down by the same factor so the row fits (and wraps).
    private var columnWidths: [CGFloat] {
        guard tableWidth > 0 else {
            // Before the first geometry measurement, fall back to equal
            // nominal widths rather than 0 (which would collapse cells).
            let cols = headers.count
            guard cols > 0 else { return [] }
            return Array(repeating: CGFloat(80), count: cols)
        }
        return MarkdownTable.layout(natural: naturalColumnWidths, available: tableWidth)
    }

    var body: some View {
        let widths = columnWidths
        VStack(alignment: .leading, spacing: 0) {
            row(headers, bold: true, widths: widths)
            Divider()
                .frame(width: widths.reduce(0, +))
            ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                row(r, bold: false, widths: widths)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .onGeometryChange(for: CGFloat.self) { proxy in
            proxy.size.width
        } action: { tableWidth = $0 }
        .textSelection(.enabled)
    }

    private func row(_ cells: [String], bold: Bool, widths: [CGFloat]) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(Array(headers.enumerated()), id: \.offset) { column, _ in
                cell(column < cells.count ? cells[column] : "",
                     column: column, bold: bold,
                     width: widths[column])
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
    private func cell(_ text: String, column: Int, bold: Bool, width: CGFloat) -> some View {
        let a = alignment(column)
        MarkdownTableCell(
            attributed: MarkdownText.renderInline(
                text, theme: theme, weight: bold ? .semibold : .regular
            ),
            alignment: a,
            width: width
        )
        .frame(width: width, alignment: a)
    }
}

/// NSViewRepresentable wrapper around an NSTextView for reliable multiline
/// wrapping of attributed text inside a table cell. SwiftUI's Text does not
/// wrap AttributedString content even with .lineLimit(nil), so we use the
/// AppKit text view that natively supports it.
private struct MarkdownTableCell: NSViewRepresentable {
    let attributed: NSAttributedString
    let alignment: Alignment
    let width: CGFloat

    private var wrapWidth: CGFloat { max(width - tableCellTrailingPadding, 0) }

    /// `attributed` with the column's alignment baked in as a paragraph
    /// style. Computed fresh (not mutated onto a shared text storage) so
    /// both `makeNSView` and `updateNSView` apply alignment the same way —
    /// `setAttributedString` replaces the whole storage, so an attribute
    /// added beforehand (the previous bug) was always discarded.
    private var aligned: NSAttributedString {
        let pStyle = NSMutableParagraphStyle()
        switch alignment {
        case .leading: pStyle.alignment = .left
        case .trailing: pStyle.alignment = .right
        case .center: pStyle.alignment = .center
        default: pStyle.alignment = .left
        }
        let mutable = NSMutableAttributedString(attributedString: attributed)
        mutable.addAttribute(.paragraphStyle, value: pStyle, range: NSRange(location: 0, length: mutable.length))
        return mutable
    }

    func makeNSView(context: Context) -> MarkdownTableCellTextView {
        let tv = MarkdownTableCellTextView()
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.textContainerInset = .zero
        tv.textContainer?.lineFragmentPadding = 0
        tv.textContainer?.widthTracksTextView = false
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.autoresizingMask = [.width]
        tv.textColor = .labelColor
        tv.textContainer?.size = CGSize(width: wrapWidth, height: CGFloat.greatestFiniteMagnitude)
        tv.textStorage?.setAttributedString(aligned)
        return tv
    }

    func updateNSView(_ nsView: MarkdownTableCellTextView, context: Context) {
        let styled = aligned
        if nsView.textStorage?.isEqual(to: styled) == false {
            nsView.textStorage?.setAttributedString(styled)
            nsView.invalidateIntrinsicContentSize()
        }
        // Only touch the text container when the width actually changed —
        // NSTextContainer posts a geometry-change notification on every set,
        // which invalidates intrinsic content size and re-triggers SwiftUI
        // layout. Setting it unconditionally on every updateNSView call
        // creates an infinite layout feedback loop that pegs the main thread.
        // A width change can also change the line-wrap count, so the cached
        // height must be invalidated here too — otherwise a cell that wraps
        // to more lines as columns narrow keeps reporting its old (shorter)
        // height, and the next row draws on top of it.
        if nsView.textContainer?.size.width != wrapWidth {
            nsView.textContainer?.size = CGSize(width: wrapWidth, height: CGFloat.greatestFiniteMagnitude)
            nsView.invalidateIntrinsicContentSize()
        }
    }
}

/// NSTextView that reports its laid-out height as its intrinsic content size,
/// so embedding it in SwiftUI's layout system works without manual height
/// bindings.
private final class MarkdownTableCellTextView: NSTextView {
    private var cachedHeight: CGFloat?

    override var intrinsicContentSize: NSSize {
        if let cachedHeight { return NSSize(width: NSView.noIntrinsicMetric, height: cachedHeight) }
        guard let lm = layoutManager, let tc = textContainer else {
            return super.intrinsicContentSize
        }
        lm.ensureLayout(for: tc)
        let height = ceil(lm.usedRect(for: tc).height)
        cachedHeight = height
        return NSSize(width: NSView.noIntrinsicMetric, height: height)
    }

    override func invalidateIntrinsicContentSize() {
        cachedHeight = nil
        super.invalidateIntrinsicContentSize()
    }

    override func didChangeText() {
        super.didChangeText()
        invalidateIntrinsicContentSize()
    }
}
