import Foundation
import AppKit

/// An image waiting in the composer, before the turn that sends it.
///
/// `original` is the file's OWN bytes, kept whenever the drop, the paste or the
/// picker handed us a file: those are what land in `attachments/`, so a PNG
/// screenshot stays a lossless PNG and a photo keeps its pixels. A paste of raw
/// pasteboard data has no file behind it and `original` is nil, so the decoded
/// image is encoded ONCE on send.
struct PendingImage: Identifiable {
    let id = UUID()
    let image: NSImage
    let original: Data?
    let filename: String?

    init(image: NSImage, original: Data? = nil, filename: String? = nil) {
        self.image = image
        self.original = original
        self.filename = filename
    }
}

/// Where a user's own attachments live, and what they are called there.
///
/// Uploads used to ride `chat-history.json` as base64. Measured on an ordinary
/// conversation, attachments were 97% of that file, so they live beside the
/// generated media instead: the history carries a PATH, the file carries the
/// picture. Generated media keeps its own root (`MediaStorage`) because it is
/// the user's OWN output, browsable and never deleted with a chat.
enum AttachmentStore {

    static let root: String = {
        let dir = NSString(string: "~/.mlx-serve/attachments").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }()

    // MARK: - Naming

    /// A filename the user chose, reduced to what is safe to put in a path.
    ///
    /// A separator is the dangerous character: an unfiltered `a/b.png` would
    /// name a file in a subdirectory that does not exist. Everything outside
    /// the allowed set collapses to `-`, and the result is capped so a
    /// pathological name cannot overflow the filesystem's own limit.
    static func sanitize(_ name: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_ "))
        let cleaned = String(name.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" })
            .trimmingCharacters(in: CharacterSet(charactersIn: "-. "))
        let collapsed = cleaned.isEmpty ? "image" : cleaned
        return String(collapsed.prefix(80))
    }

    /// `<uuid>_<name>.<ext>`. The uuid is the `ChatImage`'s own id, so a file in
    /// the folder names the record that points at it; it also makes collisions
    /// impossible, and two files called `screenshot.png` from two different
    /// folders are exactly what a flat store would otherwise clobber.
    static func filename(id: UUID, original: String?) -> String {
        let name = original.map { (($0 as NSString).deletingPathExtension) } ?? "pasted-image"
        let ext = original.flatMap { url -> String? in
            let e = (url as NSString).pathExtension.lowercased()
            return e.isEmpty ? nil : sanitize(e)
        } ?? "png"
        return "\(id.uuidString)_\(sanitize(name)).\(ext)"
    }

    /// What a dropped item is called.
    ///
    /// A drag that hands over a URL carries the name IN it; the provider's
    /// `suggestedName` is nil for exactly those providers (measured, a Finder
    /// drag of `01b At the gate.png`), so asking it first loses the name and
    /// the file lands as `pasted-image`.
    static func droppedName(url: URL?, suggested: String?) -> String? {
        url.map { $0.lastPathComponent } ?? suggested
    }

    // MARK: - Bytes

    /// PNG, not JPEG, for an image we have to encode ourselves.
    ///
    /// The no-file cases are dominated by screenshots, and JPEG is at its worst
    /// exactly there: it rings around text. Size stopped being the deciding
    /// factor once this left the history file.
    static func pngData(from image: NSImage) -> Data? {
        guard let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff) else { return nil }
        return bitmap.representation(using: .png, properties: [:])
    }

    /// The bytes to store for a pending attachment, and the name to store them
    /// under. Returns nil only when an image cannot be encoded at all.
    static func payload(for pending: PendingImage, id: UUID) -> (data: Data, name: String)? {
        if let original = pending.original, !original.isEmpty {
            // A capture can carry bytes and still have no name (a pasteboard
            // holding PNG, a drag out of a browser page), or a name with no
            // extension. Only the NAME falls back: the bytes are kept either
            // way, and the extension they imply beats a default of `png` that
            // would mislabel every JPEG.
            let base = pending.filename ?? "pasted-image"
            let name = (base as NSString).pathExtension.isEmpty ? "\(base).\(ext(for: original))" : base
            return (original, filename(id: id, original: name))
        }
        guard let png = pngData(from: pending.image) else { return nil }
        return (png, filename(id: id, original: nil))
    }

    /// Extension implied by the bytes themselves, for a file we were handed
    /// without a name (a browser drag, a pasteboard that carried PNG).
    static func ext(for data: Data) -> String {
        let b = [UInt8](data.prefix(4))
        if b.count >= 4, b[0] == 0x89, b[1] == 0x50, b[2] == 0x4E, b[3] == 0x47 { return "png" }
        if b.count >= 2, b[0] == 0xFF, b[1] == 0xD8 { return "jpg" }
        if b.count >= 3, b[0] == 0x47, b[1] == 0x49, b[2] == 0x46 { return "gif" }
        return "png"
    }

    // MARK: - Disk

    /// Writes the bytes and answers the path, or nil when the write failed.
    ///
    /// A failure is NOT fatal to the turn: the caller keeps the bytes in memory,
    /// so the model still sees the picture and the transcript still draws it.
    /// Only the next launch finds nothing, and says so.
    static func write(_ data: Data, named name: String, in root: String = AttachmentStore.root) -> String? {
        let path = (root as NSString).appendingPathComponent(name)
        do {
            try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
            try data.write(to: URL(fileURLWithPath: path), options: .atomic)
            return path
        } catch {
            return nil
        }
    }

    /// Whether a stored path really is one of ours.
    ///
    /// The delete path reads `ChatImage.path` out of saved history, so it is
    /// data, not a constant. Nothing outside this folder is ever removed.
    static func isInsideRoot(_ path: String, root: String = AttachmentStore.root) -> Bool {
        let p = (path as NSString).standardizingPath
        let r = (root as NSString).standardizingPath
        return p.hasPrefix(r + "/") && !p.hasSuffix("/")
    }

    /// The attachments a delete may actually remove.
    ///
    /// `ChatFork` COPIES messages into the new session, so two conversations can
    /// name the same file: deleting one must not take the picture out of the
    /// other. Everything a SURVIVING session still points at is spared.
    static func removablePaths(deleting ids: Set<UUID>,
                               in sessions: [ChatSession],
                               root: String = AttachmentStore.root) -> [String] {
        func paths(_ list: [ChatSession]) -> Set<String> {
            Set(list.flatMap(\.messages).flatMap { $0.images ?? [] }.compactMap(\.path))
        }
        let doomed = paths(sessions.filter { ids.contains($0.id) })
        let surviving = paths(sessions.filter { !ids.contains($0.id) })
        return doomed.subtracting(surviving)
            .filter { isInsideRoot($0, root: root) }
            .sorted()
    }

    static func remove(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }
}
