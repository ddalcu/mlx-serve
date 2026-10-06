import AppKit
import ImageIO
import SwiftUI

/// Runs `work` on a background queue, even when awaited from the main actor.
func offMain<T>(_ work: @escaping () -> T) async -> T {
    await withCheckedContinuation { (cont: CheckedContinuation<T, Never>) in
        DispatchQueue.global(qos: .userInitiated).async { cont.resume(returning: work()) }
    }
}

/// Display pictures, downsampled off-main and shared across panes.
enum MediaImage {
    /// Cost = decoded bytes.
    private static let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>()
        c.totalCostLimit = 256 * 1024 * 1024
        return c
    }()

    static func key(url: URL, maxPixel: CGFloat) -> String {
        var st = stat()
        guard stat(url.path, &st) == 0 else { return "\(url.path)|missing|\(Int(maxPixel))" }
        let m = st.st_mtimespec
        return "\(url.path)|\(m.tv_sec).\(m.tv_nsec)|\(st.st_size)|\(Int(maxPixel))"
    }

    /// Decodes at display size with EXIF orientation applied; never upscales.
    static func downsample(url: URL, maxPixel: CGFloat) -> NSImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return thumbnail(from: src, maxPixel: maxPixel)
    }

    static func downsample(data: Data, maxPixel: CGFloat) -> NSImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return thumbnail(from: src, maxPixel: maxPixel)
    }

    private static func thumbnail(from src: CGImageSource, maxPixel: CGFloat) -> NSImage? {
        let options: [CFString: Any] = [
            kCGImageSourceThumbnailMaxPixelSize: max(1, maxPixel),
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }

    static func load(url: URL, maxPixel: CGFloat) async -> NSImage? {
        await cached(key(url: url, maxPixel: maxPixel)) { downsample(url: url, maxPixel: maxPixel) }
    }

    /// `id` names the bytes (a chat attachment's image id).
    static func load(data: Data, id: String, maxPixel: CGFloat) async -> NSImage? {
        await cached("data|\(id)|\(Int(maxPixel))") { downsample(data: data, maxPixel: maxPixel) }
    }

    private static func cached(_ key: String, _ decode: @escaping () -> NSImage?) async -> NSImage? {
        if let hit = cache.object(forKey: key as NSString) { return hit }
        let image = await offMain(decode)
        if let image {
            cache.setObject(image, forKey: key as NSString,
                            cost: Int(image.size.width * image.size.height * 4))
        }
        return image
    }
}

/// A file-backed picture, decoded at `maxPixel` off-main.
struct MediaImageView: View {
    let url: URL?
    var maxPixel: CGFloat = 512
    var contentMode: ContentMode = .fill

    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            } else {
                Color.clear
            }
        }
        .task(id: Slot(url: url, maxPixel: maxPixel)) {
            guard let url else { image = nil; return }
            let loaded = await MediaImage.load(url: url, maxPixel: maxPixel)
            if !Task.isCancelled { image = loaded }
        }
    }

    private struct Slot: Hashable {
        let url: URL?
        let maxPixel: CGFloat
    }
}
