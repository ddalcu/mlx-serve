import Foundation
import AppKit

/// Opens an inline chat image (generated or attached) in Preview on
/// double-click. `ChatImage` carries only JPEG bytes — no path — so the bytes
/// are staged to a deterministic temp file (keyed by the image id, reused
/// across clicks so re-opening doesn't litter temp) before the URL is handed to
/// the system.
enum ChatImagePreview {

    /// How wide an attachment draws, given that every attachment in a message
    /// is the same HEIGHT: its own aspect ratio, so a landscape shot takes
    /// twice the room of a portrait one and a row of them still lines up.
    ///
    /// Clamped at both ends. A panorama would otherwise be wider than the
    /// message it sits in and could never share a row with anything; a picture
    /// one pixel wide would vanish. A picture with no size at all (a decode
    /// that produced an empty representation) falls back to square rather than
    /// dividing by zero.
    static func displayWidth(for image: NSImage,
                             height: CGFloat = ChatMetrics.attachmentHeight,
                             maxWidth: CGFloat = ChatMetrics.userBubbleMaxWidth) -> CGFloat {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return height }
        let width = height * (size.width / size.height)
        return min(max(width, height * 0.35), maxWidth)
    }

    /// The exact box a GENERATED picture draws in: its own ratio, fitted under
    /// both caps.
    ///
    /// Exact, not `maxWidth`/`maxHeight` with `.fit`: a frame bigger than what
    /// it holds is what the rounded corners clip, so a square picture in a
    /// 420x300 frame had its corners rounded on the empty space beside it and
    /// stayed square itself. Attachments avoid this by filling their frame,
    /// which CROPS — acceptable for a photo in a row of thumbnails, not for the
    /// picture the model was asked to make.
    ///
    /// Width can run out first (a panorama), and then the height comes down
    /// with it rather than the picture stretching.
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
