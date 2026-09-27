import SwiftUI
import AppKit

/// The reference tiles the Image and Video panes share: a fixed grid of
/// square thumbnails, each with a remove badge, a caption naming it the way
/// the prompt does, and a hover bubble with the whole picture and filename.
enum RefTiles {
    /// Three per row at the pane's 340pt floor: 340 - 32 form gutters - 12
    /// `MediaDropModifier` padding - 24 well padding = 272, less two 8pt gaps,
    /// over three.
    static let side: CGFloat = 84
    static let spacing: CGFloat = 8

    /// The pane's floating-bubble surface, in one place: the prompt's format
    /// advice and a reference tile's filename are the same kind of thing said
    /// over the pointer, and two copies would drift. Static, so the tile —
    /// its own view — can draw it too.
    static func hoverBubble(_ text: String) -> some View {
        hoverBubble(text) { EmptyView() }
    }

    /// A picture's size fitted into a square of `side`, never upscaled: a
    /// small reference shown larger than it is would only be blurry. nil for
    /// a size that is not a size.
    static func previewSize(for size: CGSize, within side: CGFloat) -> CGSize? {
        guard size.width > 0, size.height > 0, side > 0 else { return nil }
        let scale = min(side / size.width, side / size.height, 1)
        return CGSize(width: size.width * scale, height: size.height * scale)
    }

    /// The same bubble with something ABOVE the sentence — a reference tile
    /// puts the whole picture there, uncropped, since the tile shows a square
    /// cut from it.
    static func hoverBubble<Above: View>(_ text: String,
                                         @ViewBuilder above: () -> Above) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            above()
            Text(text)
                .font(.app(.caption))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .background(Color(nsColor: .textBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .shadow(color: .black.opacity(0.18), radius: 3, y: 1)
    }
}

enum RefTileKind {
    case image
    case icon(String)

    var isImage: Bool { if case .image = self { return true } else { return false } }
}

/// Fixed cells: `adaptive(minimum:maximum:)` at ONE value packs 84pt
/// columns leading (a minimum alone stretches them). Its own view so the
/// rect the bubbles are clamped to is this view's state, not the pane's.
/// Identity is the FILE (every way in dedupes); the label is the position.
struct RefTileGrid: View {
    let urls: Binding<[URL]>
    /// What the prompt calls the tile at a position: the caption under it
    /// and the text a click drops into the prompt.
    let label: (Int) -> String
    let kind: RefTileKind
    let insert: (String) -> Void

    @State private var rect: CGRect = .zero

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: RefTiles.side,
                                               maximum: RefTiles.side),
                                     spacing: RefTiles.spacing, alignment: .top)],
                  alignment: .leading,
                  spacing: RefTiles.spacing) {
            ForEach(Array(urls.wrappedValue.enumerated()), id: \.element) { idx, url in
                RefTile(url: url, promptMarker: label(idx), kind: kind, container: rect,
                        insert: insert) {
                    urls.wrappedValue.removeAll { $0 == url }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        // Quantised to 8pt: only `minX` and `width` are read
        // (`HoverReveal.clampedX`), and an exact rect changes on every
        // frame of a drag.
        .onGeometryChange(for: CGRect.self) { proxy in
            let r = proxy.frame(in: .global)
            return CGRect(x: (r.minX / 8).rounded() * 8, y: 0,
                          width: (r.width / 8).rounded() * 8, height: 0)
        } action: { rect = $0 }
    }
}

/// One reference, in the shape of the chat composer's attachment chips.
/// Hover floats the picture and its filename in the pane's bubble (the
/// 84pt caption would be dots). The label is the POSITION, so removing
/// one renumbers the rest, which is what the prompt's name for it means.
struct RefTile: View {
    let url: URL
    /// What the prompt calls this reference.
    let promptMarker: String
    let kind: RefTileKind
    let container: CGRect
    let insert: (String) -> Void
    let remove: () -> Void

    /// Loaded once per tile, not in `body`: nine full-size photos re-read
    /// on every change in the pane is a resize that drags.
    @State private var image: NSImage?

    /// Half the tile, so a clip or a track reads as an icon, not a mark.
    private static let iconSize: CGFloat = 34
    /// Long enough for a filename on one or two lines, never wider than
    /// the grid it is kept inside.
    private var bubbleWidth: CGFloat {
        container.width > 0 ? min(220, container.width) : 220
    }

    var body: some View {
        VStack(spacing: 4) {
            ZStack(alignment: .topTrailing) {
                // `contentShape` beside `clipShape`: the clip cuts only the
                // DRAWING, and a `.fill` image's overflow still takes the
                // pointer — over the next tile's badge. The badge is a
                // SIBLING above the face, never a Button inside a Button.
                Button { insert(promptMarker) } label: {
                    face
                        .frame(width: RefTiles.side, height: RefTiles.side)
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                .buttonStyle(.plain)
                Button(action: remove) {
                    // The glyph sits on a photograph, so it carries its own
                    // erased ring instead of trusting what is behind it.
                    ZStack {
                        Circle()
                            .fill(Color(nsColor: .windowBackgroundColor))
                            .frame(width: 17, height: 17)
                        Image(systemName: "multiply.circle.fill")
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
                .padding(3)
            }
            Text(promptMarker)
                .font(.app(.caption2).monospacedDigit())
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(width: RefTiles.side)
        .hoverReveal(placement: .pointerClamped(width: bubbleWidth, container: container)) {
            RefTiles.hoverBubble(url.lastPathComponent) {
                // The whole picture, fitted. Sized from its own aspect:
                // an overlay is proposed the tile's 84pt, so
                // `maxWidth`/`maxHeight` would cap it there.
                if let image,
                   let fitted = RefTiles.previewSize(for: image.size,
                                                         within: bubbleWidth - 16) {
                    Image(nsImage: image)
                        .resizable()
                        .frame(width: fitted.width, height: fitted.height)
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .onAppear { if kind.isImage { image = NSImage(contentsOf: url) } }
    }

    @ViewBuilder
    private var face: some View {
        switch kind {
        case .image:
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                placeholder("photo")
            }
        case .icon(let name):
            placeholder(name)
        }
    }

    private func placeholder(_ name: String) -> some View {
        ZStack {
            Color.secondary.opacity(0.12)
            Image(systemName: name)
                .font(.system(size: Self.iconSize))
                .foregroundStyle(.secondary)
        }
    }
}
