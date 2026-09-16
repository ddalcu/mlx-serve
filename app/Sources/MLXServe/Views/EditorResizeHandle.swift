import SwiftUI
import AppKit

/// The drag strip under a resizable text box. One copy, because the two panes
/// that have one had the same gesture written out twice and only one of them
/// was ever fixed.
struct EditorResizeHandle: View {
    @Binding var height: Double
    /// Called when the drag ends, for the pane to persist its new height.
    let onCommit: () -> Void
    var help: String = "Drag to resize the box."

    /// `translation` is cumulative, so it is applied to the height the drag
    /// STARTED at; applying it to the live height compounds.
    @State private var heightAtDragStart: Double? = nil

    var body: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.35))
            .frame(width: 36, height: 4)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 3)
            .contentShape(Rectangle())
            .gesture(
                // GLOBAL, never the default `.local`: the handle sits under the
                // box it resizes, so growing the box moves the handle, and a
                // translation measured in a space that moved subtracts its own
                // effect — the box then tracks at half the cursor's speed and
                // re-solves every frame.
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { v in
                        let base = heightAtDragStart ?? height
                        if heightAtDragStart == nil { heightAtDragStart = base }
                        height = PromptEditorHeight.clamp(base + v.translation.height)
                    }
                    .onEnded { _ in
                        heightAtDragStart = nil
                        onCommit()
                    }
            )
            // `set()` rather than `push()`/`pop()`: a handle can be removed
            // while the pointer is inside it (the lyrics box goes away when
            // Instrumental is ticked), and the pop that would balance the push
            // never arrives — leaving the resize cursor stuck app-wide.
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.set() } else { NSCursor.arrow.set() }
            }
            .onDisappear { NSCursor.arrow.set() }
            .help(help)
    }
}
