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

    /// Fraction of the detail column's measured width the reading measure
    /// takes — the transcript, composer, and empty-state greeting are all
    /// capped at this fraction, centred in whatever the panel gives them.
    /// 0.8, not 1.0: the window can be as wide as the user wants, but prose
    /// still shouldn't run edge to edge. Pinned by `ChatColumnMetricsTests`.
    static let contentWidthFraction: CGFloat = 0.8

    /// Reading width used for the single frame before `ChatDetailView` has
    /// measured its own column (`onGeometryChange` hasn't fired yet).
    static let contentFallbackWidth: CGFloat = 740

    /// Interface ▸ Compact mode — tighter vertical rhythm for more on screen.
    /// Read directly off UserDefaults (like the font-size constants below):
    /// this is a display density knob, not a launch flag, so it has no place
    /// on `ServerOptions`.
    static var compactMode: Bool { UserDefaults.standard.bool(forKey: InterfacePrefKey.compactMode) }

    /// Between turns. Small: every turn already carries its own trailing band
    /// (a reply's footer, a user turn's action row).
    static var transcriptSpacing: CGFloat { compactMode ? 6 : 10 }

    /// Inner padding + radius of a message bubble (and the tool-call card,
    /// which is styled as one).
    static let bubblePaddingH: CGFloat = 14
    /// Not a density knob: compact tightens the leading (`userLineSpacing`),
    /// and less padding reads as a rendering fault.
    static let bubblePaddingV: CGFloat = 10
    static let bubbleCornerRadius: CGFloat = 14

    /// Indent of the token-stats caption under an assistant reply so it
    /// aligns with the bubble's text column, not the bubble edge.
    static var statsIndent: CGFloat { bubblePaddingH }

    /// Half the action row's height, in compact where the row is not drawn.
    static var userBubbleBottomPadding: CGFloat { compactMode ? 10 : 0 }

    /// Extra air under a reply: the turn boundary is answer -> next question.
    static var assistantTurnBottomPadding: CGFloat { compactMode ? 4 : 10 }

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
    static let sidebarButtonHeight: CGFloat = 28
    static let sidebarButtonCornerRadius: CGFloat = 6

    /// Settings ▸ Interface ▸ Text Size. Absent = `.medium`, which is the same
    /// 14/13pt this shipped with before the setting existed.
    static var textSize: ChatTextSize {
        ChatTextSize(rawValue: UserDefaults.standard.string(forKey: InterfacePrefKey.textSize) ?? "") ?? .medium
    }
    /// The transcript's reading size.
    static var transcriptFontSize: CGFloat { textSize.proseSize }

    /// Leading. AppKit's default (~1.19 × size) is what made the transcript
    /// read as a wall next to every web chat (~1.6–1.75 of the font size);
    /// 1.4 × natural lands in that zone. Code gets less — a listing wants
    /// rows, not air.
    /// Baked into the render cache: `MarkdownText`'s key carries the density.
    static var proseLineHeightMultiple: CGFloat { compactMode ? 1.15 : 1.4 }
    static let codeLineHeightMultiple: CGFloat = 1.2

    /// The user's bubble is plain SwiftUI `Text`, where leading is extra points
    /// rather than a multiple. Tighter than the reply on purpose. Never below
    /// zero: negative leading overlaps lines.
    static var userLineSpacing: CGFloat {
        let base = (transcriptFontSize * 0.48 * 0.8).rounded()
        return compactMode ? max(0, base - 3.5) : base
    }

    /// Settings ▸ Interface ▸ Chat Column. Absent = `.wide`.
    static var chatColumn: ChatColumnWidth { ChatColumnWidth.current }

    /// One height for every attachment; width follows each picture's ratio so
    /// a row shares a baseline.
    static var attachmentHeight: CGFloat { compactMode ? 140 : 200 }
    static let attachmentSpacing: CGFloat = 6

    /// Height cap for generated media; image width follows its ratio
    /// (`ChatImagePreview.displaySize`), a video's box is fixed.
    static var generatedImageHeight: CGFloat { compactMode ? 220 : 300 }
    static var generatedVideoHeight: CGFloat { compactMode ? 200 : 260 }
    /// One width for every generated row, so stacked media share an edge.
    static let generatedMediaMaxWidth: CGFloat = 420

    /// Widest a user turn gets; a reply takes the column.
    static var userBubbleMaxWidth: CGFloat { chatColumn.userBubbleWidth }

    /// The reading column: the setting's points, or the window when narrower.
    /// `.wide` stays proportional (`contentWidthFraction`).
    static func proseWidth(panelWidth: CGFloat) -> CGFloat {
        guard panelWidth > 0 else { return contentFallbackWidth }
        guard let target = chatColumn.proseWidth else {
            return (panelWidth * contentWidthFraction).rounded()
        }
        return min(target, panelWidth)
    }
    /// Fenced/inline code inside the transcript. Monospaced digits and glyphs
    /// run wide, so matching the prose size makes code look larger than the
    /// sentence around it.
    static var transcriptCodeFontSize: CGFloat { textSize.codeSize }

    /// Panel edge → row edge. Every row in the sidebar reads it, so the
    /// destinations and the conversations are the same width by construction.
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
