import SwiftUI
import AppKit

/// The app's type ladder: macOS's own text styles, each snapped to a whole even
/// point and never below `floor`.
///
/// Fixed on purpose, and the reason is measured rather than stylistic. macOS
/// has no dynamic type: `NSFont.preferredFont(forTextStyle:)` hands the same
/// numbers to every user, and `.dynamicTypeSize(_:)` does not move a semantic
/// font here at all — a `Text` at `.body` renders the same height under
/// `.xSmall` and under `.accessibility4`, because that environment is an iOS
/// one. Naming a system style therefore buys ONE VOCABULARY, not scaling, and
/// the numbers below are where that vocabulary comes from: each system size,
/// taken once, with the odd steps moved up a point and nothing under 12.
///
/// The floor is 12, not 10. Ten was the system's own `caption` size and it is
/// legal, but it is a size you *can* read rather than one you enjoy reading,
/// and with a floor that low the smallest step a view could name was the one
/// nobody wanted — settings prose ended up there by default. Raising the floor
/// is what stops that from being reachable by accident: the smallest step a
/// view can name is now one that reads.
///
/// Three steps collapse onto each other, and that is the price of even-only
/// with this floor: `body`/`headline` both land on 14, and `subheadline`,
/// `callout`, `footnote`, `caption` and `caption2` all land on 12. The names
/// still differ because the roles differ — a heading is still a heading when
/// the system's heading grows, and the role table below says which step each
/// role takes so the collapse is a deliberate choice rather than a drift.
enum AppType {
    /// No text in the app renders smaller than this.
    static let floor: CGFloat = 12

    /// Every step, with the macOS size it was derived from. The `system` column
    /// is what the ladder is anchored to: `SystemTypeTests` fails when macOS
    /// moves one, which is the signal to re-derive this table rather than let
    /// the app drift away from the platform on its own.
    static let table: [(style: Font.TextStyle, system: CGFloat, pointSize: CGFloat)] = [
        (.largeTitle,  26, 26),
        (.title,       22, 22),
        (.title2,      17, 18),
        (.title3,      15, 16),
        (.headline,    13, 14),
        (.body,        13, 14),
        (.callout,     12, 12),
        (.subheadline, 11, 12),
        (.footnote,    10, 12),
        (.caption,     10, 12),
        (.caption2,    10, 12),
    ]

    /// The point size a step renders at.
    static func pointSize(for style: Font.TextStyle) -> CGFloat {
        table.first { $0.style == style }?.pointSize ?? floor
    }

    /// May the app render text at this size? The two rules the ladder keeps:
    /// whole even points, and never under `floor`.
    static func isLegal(_ size: CGFloat) -> Bool {
        size >= floor && size.truncatingRemainder(dividingBy: 2) == 0
    }

    /// What a piece of text IS, which is what picks its step. Without this the
    /// ladder has a floor and a rule and no way to say "an explainer is not a
    /// caption", and every view picks the smallest step it can — which is how
    /// the settings prose ended up at 10pt in the first place.
    ///
    /// The steps, read as a scale rather than as a menu: page > section > row >
    /// supporting > annotation. `Font.app(_:)` takes the step directly; this is
    /// the sentence that says which one a given piece of copy wants.
    enum Role {
        /// A window's own name. Once per window.
        case pageTitle
        /// A pane or sheet's heading.
        case sectionTitle
        /// The name of one setting, row, model or item.
        case rowTitle
        /// A sentence under a row that says what it does. The one that was too
        /// small: prose the user reads on purpose, not a label.
        case explainer
        /// A value, a status, a cost, a count.
        case value
        /// A badge, a unit, a qualifier next to something bigger.
        case annotation

        /// The step this role takes.
        var step: Font.TextStyle {
            switch self {
            case .pageTitle:   return .largeTitle
            // 16, not `.headline`(14): seventeen sheet and section headings in
            // this app already sit at `.title3`, and a pane column's title in
            // the toolbar cannot render SMALLER than the section headings
            // underneath it. The role table follows the code, not the other
            // way round.
            case .sectionTitle: return .title3
            case .rowTitle:    return .body
            case .explainer:   return .callout
            case .value:       return .callout
            case .annotation:  return .footnote
            }
        }
    }
}

extension Font {
    /// The ladder, by role: the step says what the text IS, so two views that
    /// mean the same thing cannot pick two different sizes.
    static func app(_ role: AppType.Role, weight: Font.Weight? = nil, design: Font.Design? = nil) -> Font {
        .app(role.step, weight: weight, design: design)
    }
}

extension Font {
    /// The one way a view states a text size. `style` names the step and the
    /// number comes from `AppType`, so a size cannot be typed into a view and
    /// drift off the ladder — `SystemTypeTests` fails the build on a literal.
    ///
    /// The weight and design stay at the call site: they change how a step
    /// looks, never how big it is.
    static func app(
        _ style: Font.TextStyle,
        weight: Font.Weight? = nil,
        design: Font.Design? = nil
    ) -> Font {
        .system(size: AppType.pointSize(for: style), weight: weight, design: design)
    }
}

extension AppType {
    /// The AppKit font for a step, for the text SwiftUI does not draw: the log
    /// view, the status menu, a code block's text view. Same table, so the two
    /// halves of the app cannot drift apart.
    static func system(_ style: Font.TextStyle, weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.systemFont(ofSize: pointSize(for: style), weight: weight)
    }

    static func monospaced(_ style: Font.TextStyle, weight: NSFont.Weight = .regular) -> NSFont {
        NSFont.monospacedSystemFont(ofSize: pointSize(for: style), weight: weight)
    }
}
