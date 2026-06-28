import Foundation
import AppKit

/// Opens an inline chat image (generated or attached) in Preview on
/// double-click. `ChatImage` carries only JPEG bytes — no path — so the bytes
/// are staged to a deterministic temp file (keyed by the image id, reused
/// across clicks so re-opening doesn't litter temp) before the URL is handed to
/// the system.
enum ChatImagePreview {

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
