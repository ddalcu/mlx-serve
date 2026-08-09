import Foundation

/// Where the composer's text actually starts — and therefore where its
/// placeholder has to sit.
///
/// The placeholder is a SwiftUI `Text` laid over an NSTextView, so nothing
/// aligns them for you: the offset has to be reconstructed from the three
/// insets between the field's edge and the first glyph, and it was a literal
/// `9` against a real offset of `5 + 7 + 2 = 14`. The caret therefore sat five
/// points to the RIGHT of the placeholder it was standing in for — visible as
/// the cursor overlapping the first letter.
///
/// One source of truth for all three numbers, read by the editor AND the
/// overlay, because two of them are set on AppKit objects and the third on a
/// SwiftUI modifier: nothing in the type system relates them, so a literal in
/// either place is a drift waiting to happen.
enum ComposerTextMetrics {
    /// The field's own horizontal padding, outside the scroll view.
    static let fieldHorizontalPadding: CGFloat = 5
    /// `NSTextView.textContainerInset`.
    static let containerInsetWidth: CGFloat = 7
    static let containerInsetHeight: CGFloat = 8
    /// `NSTextContainer.lineFragmentPadding` — the last inset before the glyph,
    /// and the one that is easiest to forget because it is not a padding
    /// anybody wrote.
    static let lineFragmentPadding: CGFloat = 2

    /// Leading offset of the first typed character from the field's edge — the
    /// placeholder's leading padding, by construction rather than by eye.
    static var placeholderLeading: CGFloat {
        fieldHorizontalPadding + containerInsetWidth + lineFragmentPadding
    }

    /// Top offset of the first line. The overlay has no `fieldHorizontalPadding`
    /// equivalent vertically — the field's frame is the text view's.
    static var placeholderTop: CGFloat { containerInsetHeight }
}
