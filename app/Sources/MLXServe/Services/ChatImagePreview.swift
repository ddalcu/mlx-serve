import Foundation
import AppKit

/// Opens an inline chat image (generated or attached) in Preview on
/// double-click. `ChatImage` carries only JPEG bytes — no path — so the bytes
/// are staged to a deterministic temp file (keyed by the image id, reused
/// across clicks so re-opening doesn't litter temp) before the URL is handed to
/// the system.
enum ChatImagePreview {

    /// Attachment width at the shared height, from its own ratio, clamped so a
    /// panorama can share a row and a sliver stays visible. No size = square.
    static func displayWidth(for image: NSImage,
                             height: CGFloat = ChatMetrics.attachmentHeight,
                             maxWidth: CGFloat = ChatMetrics.userBubbleMaxWidth) -> CGFloat {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return height }
        let width = height * (size.width / size.height)
        return min(max(width, height * 0.35), maxWidth)
    }

    /// Exact box for a generated picture under both caps: the rounded corners
    /// clip the frame, so the frame must be the picture (no crop, no fit).
    static func displaySize(for image: NSImage,
                            maxHeight: CGFloat,
                            maxWidth: CGFloat) -> CGSize {
        let size = image.size
        guard size.width > 0, size.height > 0 else {
            return CGSize(width: maxHeight, height: maxHeight)
        }
        let ratio = size.width / size.height
        let width = min(maxHeight * ratio, maxWidth)
        return CGSize(width: width, height: width / ratio)
    }

    /// Directory the staged temp files live in.
    static var tempDir: String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("mlx-serve-chat-images")
    }

    /// Deterministic temp path for an image id (a `.jpg`).
    static func tempFileURL(for id: UUID) -> URL {
        URL(fileURLWithPath: (tempDir as NSString).appendingPathComponent("\(id.uuidString).jpg"))
    }

    /// Stage the image's JPEG bytes to its temp file and return the URL. Pure
    /// filesystem side effect (no NSWorkspace) → unit-testable.
    @discardableResult
    static func writeTempFile(_ image: ChatImage) throws -> URL {
        try FileManager.default.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
        let url = tempFileURL(for: image.id)
        try image.data.write(to: url)
        return url
    }

    /// Stage the image and open it in Preview (falling back to the default image
    /// viewer if Preview can't be resolved). Best-effort — a write/open failure
    /// is silently ignored.
    static func openInPreview(_ image: ChatImage) {
        guard let url = try? writeTempFile(image) else { return }
        if let preview = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Preview") {
            NSWorkspace.shared.open([url], withApplicationAt: preview,
                                    configuration: NSWorkspace.OpenConfiguration())
        } else {
            NSWorkspace.shared.open(url)
        }
    }
}
