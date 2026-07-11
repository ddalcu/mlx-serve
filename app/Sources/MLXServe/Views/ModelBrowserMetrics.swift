import Foundation

/// Layout decisions for the Discover table — ONE source of truth for the
/// column widths that must agree between `ColumnHeaderRow` and
/// `ModelBrowserRow` (they used to be duplicated magic numbers), plus the
/// width→column-visibility decision.
///
/// The pane measures its own width and WE decide what fits, instead of
/// letting SwiftUI starve the flexible Model column. Before this, the fixed
/// columns cost more than the old 720pt floor could spare, so names
/// truncated to "gem…" and chip text wrapped one character per line.
/// Relationships pinned by `ModelBrowserMetricsTests`.
enum ModelBrowserMetrics {
    // Fixed column widths. RAM is 120 so GGUF range strings
    // ("21.2–55.4 GB") stay on one line; Action is 120 for the on-disk
    // "✓ Use" + trash cell and the "Download ▾" GGUF menu.
    static let quantWidth: CGFloat = 54
    static let sizeWidth: CGFloat = 54
    static let pullsWidth: CGFloat = 64
    static let likesWidth: CGFloat = 50
    static let ramWidth: CGFloat = 120
    static let updatedWidth: CGFloat = 64
    static let actionWidth: CGFloat = 120

    static let columnSpacing: CGFloat = 8
    static let rowPaddingH: CGFloat = 12

    /// What the flexible Model column must keep for names to stay readable
    /// before a tier gives up a column…
    static let modelColumnMinWidth: CGFloat = 180
    /// …and the reduced floor the compact tier is allowed to squeeze to.
    static let compactModelColumnMinWidth: CGFloat = 140

    /// Sidebar widths — the `navigationSplitViewColumnWidth` arguments in
    /// `ModelBrowserView` read these so window math can't drift from them.
    static let sidebarMinWidth: CGFloat = 190
    static let sidebarIdealWidth: CGFloat = 205
    static let sidebarMaxWidth: CGFloat = 260

    /// First-open window width: full tier beside the sidebar. The old 900
    /// default left the detail ~695pt — squeezed from the first frame.
    static let defaultWindowWidth: CGFloat = 1000
    static let defaultWindowHeight: CGFloat = 640

    /// Window floor. The old code put `.frame(minWidth: 700)` on the whole
    /// window while the detail alone demanded 720 — an explicit outer frame
    /// CAPS the window's reported minimum, so the window shrank to 700 and
    /// the table clipped off the right edge. The floor must be derived:
    /// sidebar minimum + detail minimum.
    static var minWindowWidth: CGFloat { sidebarMinWidth + minDetailWidth }

    /// Column sets, widest to narrowest. Model, Quant, Size, RAM Est., and
    /// the action cell survive every tier — they're what you need to decide
    /// "can I run this and how do I get it".
    enum Tier: Equatable {
        /// Everything.
        case full
        /// Drops Likes + Updated.
        case medium
        /// Additionally drops Pulls.
        case compact

        var showsPulls: Bool { self != .compact }
        var showsLikes: Bool { self == .full }
        var showsUpdated: Bool { self == .full }
    }

    /// Fixed cost of a row at a tier: edge padding + every visible fixed
    /// column + one inter-column spacing each (the flexible Model column
    /// absorbs the remainder).
    static func fixedWidth(for tier: Tier) -> CGFloat {
        var columns: [CGFloat] = [quantWidth, sizeWidth, ramWidth, actionWidth]
        if tier.showsPulls { columns.append(pullsWidth) }
        if tier.showsLikes { columns.append(likesWidth) }
        if tier.showsUpdated { columns.append(updatedWidth) }
        return rowPaddingH * 2
            + columns.reduce(0, +)
            + CGFloat(columns.count) * columnSpacing
    }

    /// Width 0 = not measured yet → assume roomy (full columns).
    static func tier(forDetailWidth width: CGFloat) -> Tier {
        guard width > 0 else { return .full }
        if width >= fixedWidth(for: .full) + modelColumnMinWidth { return .full }
        if width >= fixedWidth(for: .medium) + modelColumnMinWidth { return .medium }
        return .compact
    }

    /// Detail pane floor (`.frame(minWidth:)`): the compact tier with its
    /// reduced Model minimum. Below this the action cell clips off the
    /// window edge, which is the broken state in the bug screenshot.
    static var minDetailWidth: CGFloat {
        fixedWidth(for: .compact) + compactModelColumnMinWidth
    }
}
