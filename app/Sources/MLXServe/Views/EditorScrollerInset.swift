import SwiftUI
import AppKit

/// The room a `TextEditor`'s own scroller takes from its text. An overlay
/// scroller floats over the text and takes none; a legacy one ("Show scroll
/// bars: Always", or a mouse attached) narrows the column the text wraps in.
enum EditorScrollerInset {
    static func trailing(for style: NSScroller.Style) -> CGFloat {
        style == .legacy ? NSScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy) : 0
    }
}

extension View {
    /// For text laid over a `TextEditor` (a placeholder): wraps where the
    /// editor's own text wraps, clear of its scroller.
    func clearOfEditorScroller() -> some View { modifier(ClearOfEditorScroller()) }
}

private struct ClearOfEditorScroller: ViewModifier {
    @State private var style = NSScroller.preferredScrollerStyle

    func body(content: Content) -> some View {
        content
            .padding(.trailing, EditorScrollerInset.trailing(for: style))
            .onReceive(NotificationCenter.default.publisher(for: NSScroller.preferredScrollerStyleDidChangeNotification)) { _ in
                style = NSScroller.preferredScrollerStyle
            }
    }
}
