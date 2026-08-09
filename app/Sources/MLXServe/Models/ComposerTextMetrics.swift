import Foundation

/// Where the composer's text starts — and therefore where its placeholder has
/// to sit.
enum ComposerTextMetrics {
    /// The field's own padding, outside the scroll view.
    static let fieldHorizontalPadding: CGFloat = 5
    /// `NSTextView.textContainerInset`.
    static let containerInsetWidth: CGFloat = 7
    static let containerInsetHeight: CGFloat = 8
    /// `NSTextContainer.lineFragmentPadding` — the last inset before the glyph,
    /// and the easiest to forget because it is not a padding anybody wrote.
    static let lineFragmentPadding: CGFloat = 2

    static var placeholderLeading: CGFloat {
        fieldHorizontalPadding + containerInsetWidth + lineFragmentPadding
    }
    static var placeholderTop: CGFloat { containerInsetHeight }
}
