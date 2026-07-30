import Foundation
import SwiftUI

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

    /// Exact height of BOTH controls in the sidebar's bottom row (New Chat +
    /// the agent menu).
    ///
    /// They are `.plain` buttons drawing their own background — the same shape
    /// as the composer's discs — precisely so this number is the height. Don't
    /// put them back on `.buttonStyle(.bordered)`: a bordered control keeps its
    /// INTRINSIC size and merely centers inside whatever frame it's given, so
    /// its height can only be steered indirectly through the label, and a text
    /// label and a bare glyph never land on the same number. Measured through
    /// the accessibility API while both sat inside 28pt frames: New Chat 24pt,
    /// the menu 17pt.
    static let sidebarButtonHeight: CGFloat = 28
    static let sidebarButtonCornerRadius: CGFloat = 6

    // The Think / Agent / MCP capsules that used to live in the window toolbar
    // had their own `togglePill*` geometry here. They are icon-only composer
    // controls now and draw from `composerIconSize` / `composerControlSize` like
    // every other control in that row, so the separate metrics are gone rather
    // than left behind as a second, unused way to size a control.
}

extension View {
    /// The sidebar's bottom-row button chrome: one exact height, one fill, one
    /// radius — applied to both controls so neither can drift from the other.
    func sidebarActionButton() -> some View {
        frame(height: ChatMetrics.sidebarButtonHeight)
            .background(Color.secondary.opacity(0.15),
                        in: RoundedRectangle(cornerRadius: ChatMetrics.sidebarButtonCornerRadius))
    }
}
