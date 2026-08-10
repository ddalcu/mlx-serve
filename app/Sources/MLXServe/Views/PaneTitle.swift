import SwiftUI

/// A pane column's title, with its create control beside it, for the window
/// toolbar.
struct PaneTitleBar: View {
    let title: String
    let addHelp: String
    let add: () -> Void

    @State private var hovering = false

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.primary)
            Button(action: add) {
                Image(systemName: "plus")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.primary)
                    // A square target the glyph sits in the middle of, rather
                    // than the glyph's own bounds — a bare symbol is a few
                    // points across and awkward to hit.
                    .frame(width: 22, height: 22)
                    .background(
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(Color.primary.opacity(hovering ? 0.12 : 0)))
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .onHover { hovering = $0 }
            .help(addHelp)
        }
        // Breathing room from the column's leading edge; the toolbar gives none
        // once the shared background is off.
        .padding(.leading, 4)
    }
}

extension View {
    /// The pane title + create control, with the toolbar's own capsule
    /// suppressed where the platform draws one.
    @ViewBuilder
    func paneTitle(_ title: String, help: String = "New", add: @escaping () -> Void) -> some View {
        PaneTitleBar(title: title, addHelp: help, add: add)
    }
}
