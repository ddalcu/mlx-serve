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
/// taken once, with the odd steps moved up a point and nothing under 10.
///
/// Two steps collapse onto each other, and that is the price of even-only:
/// `body`/`headline` both land on 14, and `subheadline` joins `callout` at 12.
/// The names still differ because the roles differ — a heading is still a
/// heading when the system's heading grows.
enum AppType {
    /// No text in the app renders smaller than this.
    static let floor: CGFloat = 10

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
        (.footnote,    10, 10),
        (.caption,     10, 10),
        (.caption2,    10, 10),
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
