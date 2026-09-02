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

    /// Between turns in the transcript.
    ///
    /// Small, because by now every turn carries its own trailing band: a reply
    /// has the footer with its timestamp and token counts, and your own turn
    /// has the action row. Both already put air under a message, so a wide gap
    /// on top of them reads as a hole rather than as rhythm. It was 18, from
    /// when the separation had nothing else to lean on.
    static var transcriptSpacing: CGFloat { compactMode ? 6 : 10 }

    /// Inner padding + radius of a message bubble (and the tool-call card,
    /// which is styled as one).
    static let bubblePaddingH: CGFloat = 14
    /// Not a density knob. Compact tightens the LEADING inside the bubble
    /// (`userLineSpacing`); taking the padding as well pulled the text against
    /// the bubble's own edge, which reads as a rendering fault rather than as
    /// density.
    static let bubblePaddingV: CGFloat = 10
    static let bubbleCornerRadius: CGFloat = 14

    /// Indent of the token-stats caption under an assistant reply so it
    /// aligns with the bubble's text column, not the bubble edge.
    static var statsIndent: CGFloat { bubblePaddingH }

    /// Air under your own turn in compact mode, where the action row is not
    /// drawn at all.
    ///
    /// Half the row's height, not all of it: the point of compact is to get the
    /// space back, but with nothing under the bubble at all your question sits
    /// flush against the reply to it. The row is an 18pt button plus its 2pt
    /// top padding.
    static var userBubbleBottomPadding: CGFloat { compactMode ? 10 : 0 }

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
    static let proseLineHeightMultiple: CGFloat = 1.4
    static let codeLineHeightMultiple: CGFloat = 1.2

    /// The user's bubble is plain SwiftUI `Text`, where leading is EXTRA
    /// points rather than a multiple.
    ///
    /// Tighter than the reply's rhythm on purpose. A restatement of
    /// `proseLineHeightMultiple` would be 0.48 × the size, but your own turn is
    /// a question you have already read, inside a filled bubble whose padding
    /// gives it air the reply gets from the page — so it can run denser without
    /// reading as cramped.
    ///
    /// Compact takes 3.5 points more. Never below zero: negative leading in
    /// `Text` overlaps the lines rather than tightening them, and at the
    /// smallest text size the base is only a few points to begin with.
    static var userLineSpacing: CGFloat {
        let base = (transcriptFontSize * 0.48 * 0.8).rounded()
        return compactMode ? max(0, base - 3.5) : base
    }

    /// Settings ▸ Interface ▸ Chat Column. Absent = `.wide`.
    static var chatColumn: ChatColumnWidth { ChatColumnWidth.current }

    /// One height for every attachment in a message, so a row of them shares a
    /// baseline whatever shape the pictures are.
    ///
    /// Height rather than a box: capping BOTH axes gave a portrait photo 225
    /// points of width and a landscape one 400, so a row had neither a common
    /// height nor a common width and read as a pile. Fixing the height and
    /// letting the width follow each picture's own ratio is what makes a row
    /// line up - a landscape shot simply takes twice the room of a portrait
    /// one, which is what it is worth.
    static var attachmentHeight: CGFloat { compactMode ? 140 : 200 }
    /// Between attachments, both ways.
    static let attachmentSpacing: CGFloat = 6

    /// Widest your own turn gets. The reply has no such cap - it is the page's
    /// main content and takes the column.
    ///
    /// A cap rather than the old bare `Spacer(minLength: 60)`, which was the
    /// only thing keeping a long question off the left edge: sixty points is
    /// seven percent of a narrow column and two percent of a wide one, so how
    /// far the bubble hung off the right depended on the window.
    static var userBubbleMaxWidth: CGFloat { chatColumn.userBubbleWidth }

    /// The reading column: the setting's width in points, or the window when
    /// that is narrower.
    ///
    /// A fixed width rather than a fraction of the window, so resizing spends
    /// the MARGINS and leaves the lines alone. Past the setting's own width
    /// there is nothing left to spend, so the window takes over and wraps them.
    ///
    /// `.wide` is the exception and stays proportional
    /// (`contentWidthFraction`): it is the setting for people who want the
    /// window to decide, and running text to the very edge of a wide window
    /// looks unfinished no matter how wide the window is.
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
