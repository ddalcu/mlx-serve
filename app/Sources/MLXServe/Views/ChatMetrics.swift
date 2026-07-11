import Foundation

/// Shared layout constants for the chat column — ONE source of truth for the
/// numbers that must agree across independent views. The transcript, context
/// monitor, and composer each pad themselves; when these were inlined they
/// drifted (16/12/12 gutters), leaving the input pill and token bar 4pt left
/// of the message bubbles while two chip rows carried secret +4 compensation
/// paddings. Relationships pinned by `ChatMetricsTests`.
enum ChatMetrics {
    /// Left/right inset of every full-width surface in the chat column:
    /// transcript content, context monitor, composer row.
    static let gutter: CGFloat = 16

    /// Inner padding + radius of a message bubble (and the tool-call card,
    /// which is styled as one).
    static let bubblePaddingH: CGFloat = 14
    static let bubblePaddingV: CGFloat = 10
    static let bubbleCornerRadius: CGFloat = 14

    /// Indent of the token-stats caption under an assistant reply so it
    /// aligns with the bubble's text column, not the bubble edge.
    static var statsIndent: CGFloat { bubblePaddingH }

    /// Single-line height of the composer's input pill — also the frame of
    /// every round control beside it (attach / mic / send), so a
    /// bottom-aligned HStack lines their centers up with the resting pill
    /// without per-view nudge paddings.
    static let composerMinHeight: CGFloat = 36
    static var composerControlSize: CGFloat { composerMinHeight }
    /// Visual diameter of the round control glyphs/backgrounds inside their
    /// `composerControlSize` frames (send symbol point size == attach circle).
    static let composerIconSize: CGFloat = 30

    /// Toolbar mode pills (Think / Agent / MCP) — one geometry so the three
    /// capsules render identically. Height and icon slot are EXPLICIT because
    /// SF symbols at the same point size have different intrinsic sizes
    /// (brain vs wrench vs puzzle); padding-derived heights made the pills
    /// subtly unequal. The pills ride ONE ToolbarItem with
    /// `togglePillSpacing` between them, so in-cluster gaps are ours and
    /// uniform — no item ever adds outer padding of its own (a stray
    /// leading/trailing pad on one item is what made the gaps uneven before).
    static let togglePillPaddingH: CGFloat = 8
    static let togglePillHeight: CGFloat = 24
    static let togglePillIconSize: CGFloat = 15
    static let togglePillSpacing: CGFloat = 8
}
