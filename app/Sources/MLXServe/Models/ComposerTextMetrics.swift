import Foundation

/// Where the composer's text starts — and therefore where its placeholder has
/// to sit.
///
/// The placeholder is a SwiftUI `Text` laid over an NSTextView, so nothing
/// aligns them: two of these are set on AppKit objects and one on a SwiftUI
/// modifier, and only arithmetic relates them. It was a literal 9 against a
/// real 14, so the caret overlapped its own placeholder.
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
