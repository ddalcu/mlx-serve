import SwiftUI

/// A storyboard shot's own last frame, beside its prompt: the picture the shot
/// lands on and the next one opens on. Empty, the pane's frame well; filled, a
/// reference tile (remove badge, the whole picture and its name on hover).
struct ShotEndFrameWell: View {
    @Binding var url: URL?
    let height: CGFloat
    let choose: () -> Void

    static let width: CGFloat = 96
    /// Room the hover bubble takes, kept inside the card: it grows leftward from the well.
    private static let bubbleRoom: CGFloat = 220

    @State private var isTargeted = false
    @State private var container: CGRect = .zero

    var body: some View {
        Group {
            if let picked = url {
                RefTile(url: picked, promptMarker: "", kind: .image, container: container,
                        showsMarker: false, insert: { _ in choose() }) { url = nil }
            } else {
                Button(action: choose) {
                    MediaWellAction(title: "Last frame (optional)...", systemImage: "photo.on.rectangle.angled",
                                    layout: .stacked)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(L10n.text("The picture this shot ends on and the next one starts from. Empty, the next shot starts from this shot's own last frame."))
            }
        }
        .frame(width: Self.width, height: height)
        .background(MediaDropWellBackground(isTargeted: isTargeted))
        .onGeometryChange(for: CGRect.self) { proxy in
            let r = proxy.frame(in: .global)
            return CGRect(x: ((r.maxX - Self.bubbleRoom) / 8).rounded() * 8, y: 0, width: Self.bubbleRoom, height: 0)
        } action: { container = $0 }
        .mediaDrop(.image, isTargeted: $isTargeted) { urls in
            if let dropped = urls.first { url = dropped }
        }
    }
}
