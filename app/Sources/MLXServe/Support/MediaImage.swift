import AppKit
import ImageIO
import SwiftUI

/// Downsample display images off-main and share decoded thumbnails across panes.
/// Never construct full-size file-backed `NSImage`s from a view body.
enum MediaImage {
    /// Decoded images keyed by `key(url:maxPixel:)`. The cost limit is
    /// decoded bytes, so a shelf full of 4K inputs cannot grow it unbounded.
    private static let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>()
        c.totalCostLimit = 256 * 1024 * 1024
        return c
    }()

    /// Include inode and change time: an atomic replacement may preserve both
    /// modification time and byte count.
    static func key(url: URL, maxPixel: CGFloat) -> String {
        let suffix = "|\(Int(maxPixel))"
        var st = stat()
        guard stat(url.path, &st) == 0 else { return "\(url.path)|-1|-1\(suffix)" }
        let mtimeNs = st.st_mtimespec.tv_sec * 1_000_000_000 + st.st_mtimespec.tv_nsec
        return "\(url.path)|\(st.st_dev)|\(st.st_ino)|\(st.st_ctimespec.tv_sec)|\(st.st_ctimespec.tv_nsec)|\(mtimeNs)|\(st.st_size)\(suffix)"
    }

    static func cached(key: String) -> NSImage? {
        cache.object(forKey: key as NSString)
    }

    static func store(key: String, image: NSImage, cost: Int) {
        cache.setObject(image, forKey: key as NSString, cost: cost)
    }

    /// Approximate decoded bytes, for the cache cost.
    static func cost(_ image: NSImage) -> Int {
        Int(image.size.width * image.size.height * 4)
    }

    /// Decode directly at display size with EXIF orientation, not a full RGBA image.
    static func downsample(url: URL, maxPixel: CGFloat) -> NSImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return thumbnail(from: src, maxPixel: maxPixel)
    }

    /// The same for bytes already in memory — chat attachments carry JPEG
    /// bytes beside the message for the run's lifetime.
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

    /// Cache hits return immediately; misses decode off the caller's actor.
    static func load(url: URL, maxPixel: CGFloat) async -> NSImage? {
        let k = key(url: url, maxPixel: maxPixel)
        if let hit = cached(key: k) { return hit }
        let image = await offMain { downsample(url: url, maxPixel: maxPixel) }
        if let image { store(key: k, image: image, cost: cost(image)) }
        return image
    }

    /// Byte-backed chat attachments have a stable message image id.
    static func load(data: Data, id: String, maxPixel: CGFloat) async -> NSImage? {
        let k = "data|\(id)|\(Int(maxPixel))"
        if let hit = cached(key: k) { return hit }
        let image = await offMain { downsample(data: data, maxPixel: maxPixel) }
        if let image { store(key: k, image: image, cost: cost(image)) }
        return image
    }

    /// Force the decode onto a utility queue, including calls from the main actor.
    private static func offMain(_ work: @escaping () -> NSImage?) async -> NSImage? {
        await withCheckedContinuation { (cont: CheckedContinuation<NSImage?, Never>) in
            DispatchQueue.global(qos: .utility).async { cont.resume(returning: work()) }
        }
    }
}

/// A file-backed thumbnail whose decode is keyed by file and display size.
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
            image = nil
            guard let url else { return }
            let loaded = await MediaImage.load(url: url, maxPixel: maxPixel)
            // A cancelled decode may finish after a newer, cached selection.
            if !Task.isCancelled { image = loaded }
        }
    }

    private struct Slot: Hashable {
        let url: URL?
        let maxPixel: CGFloat
    }
}
