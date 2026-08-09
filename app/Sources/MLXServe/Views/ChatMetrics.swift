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

    /// The reading measure — the width the transcript AND the composer are
    /// capped at, centred in whatever the window gives them.
    ///
    /// Both used to be full-width: on a 1400pt window a line of prose ran the
    /// whole way across (~180 characters, about twice what text is comfortable
    /// at) and the input pill grew with the window until it read as a document
    /// rather than a message field. It is ONE number for both, because a
    /// composer even slightly narrower or wider than the answers above it is a
    /// seam you can't unsee.
    static let contentMaxWidth: CGFloat = 740

    /// Between turns in the transcript. Wider than the old 12: with the column
    /// capped, vertical rhythm is what separates one turn from the next — the
    /// window's edges no longer do it.
    static let transcriptSpacing: CGFloat = 18

    /// The column's actual width for a given available width: the measure, or
    /// the window minus its gutters when that is narrower. Never non-positive —
    /// a negative frame width is a crash in some SwiftUI containers and a
    /// silently invisible column in the rest, and layout does report zero.
    static func columnWidth(available: CGFloat) -> CGFloat {
        max(1, min(contentMaxWidth, available - gutter * 2))
    }

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

    /// Panel edge → row edge. Every row in the sidebar reads it, so the
    /// destinations and the conversations are the same width by construction.
    ///
    /// They were not: the destinations are a `VStack` with this padding, while
    /// the conversations are `List` rows, and `listRowInsets` measures from the
    /// list's OWN content area — which a `.sidebar`-style list has already
    /// inset. The same "8" in both places therefore drew two different widths,
    /// with the chats pushed right by the list's built-in margin. The rows zero
    /// their `listRowInsets` and apply this instead, so both start from the
    /// panel edge.
    static let sidebarGutter: CGFloat = 8
    /// Row edge → label. The icon of a destination and the title of a chat both
    /// start here, which is what makes the column read as one list.
    static let sidebarRowInset: CGFloat = 8

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
