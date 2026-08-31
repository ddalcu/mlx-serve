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
    ///
    /// The extension describes the bytes being STORED, never the name they
    /// arrived under: an `IMG_1234.HEIC` that is converted on the way in is a
    /// JPEG once it lands, and a name saying otherwise is a file lying about
    /// itself to the only thing that ever reads it back.
    static func filename(id: UUID, name: String?, ext: String) -> String {
        let base = name.map { ($0 as NSString).deletingPathExtension } ?? "pasted-image"
        return "\(id.uuidString)_\(sanitize(base)).\(sanitize(ext))"
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

    /// PNG, not JPEG, for an image pasted with no file behind it.
    ///
    /// Those are dominated by screenshots, and JPEG is at its worst exactly
    /// there: it rings around text. Size stopped being the deciding factor once
    /// this left the history file.
    static func pngData(from image: NSImage) -> Data? {
        bitmap(from: image)?.representation(using: .png, properties: [:])
    }

    /// The encoding for a picture whose own format the server cannot read.
    ///
    /// JPEG, because what lands here is overwhelmingly a photo: HEIC is the
    /// iPhone default and is already lossy, so storing it losslessly preserves
    /// its compression artifacts at tens of MB for a 12 Mpx frame.
    ///
    /// 0.85 is the quality this app re-encoded EVERY attachment at before they
    /// moved to disk, so nothing here is worse than what shipped; it just
    /// applies to a far narrower set of inputs now. Higher settings buy nothing
    /// downstream — the vision tower smart-resizes to `ENGINE_MAX_PIXELS`
    /// (1536², about a fifth of a 12 Mpx frame) before it looks at anything —
    /// and measured on a real iPhone photo, q0.95 came out at twice the HEIC's
    /// own size, since HEVC intra beats JPEG at equal quality.
    ///
    /// Transparency is the exception, because JPEG has no alpha channel and
    /// would turn it black.
    static func transcoded(_ image: NSImage) -> (data: Data, ext: String)? {
        guard let bitmap = bitmap(from: image) else { return nil }
        if bitmap.hasAlpha, let png = bitmap.representation(using: .png, properties: [:]) {
            return (png, "png")
        }
        guard let jpeg = bitmap.representation(using: .jpeg,
                                               properties: [.compressionFactor: 0.85]) else { return nil }
        return (jpeg, "jpg")
    }

    /// The CGImage first, because `tiffRepresentation` is a re-encode and can
    /// hand back an alpha channel the picture never had - which decides JPEG
    /// vs PNG above.
    private static func bitmap(from image: NSImage) -> NSBitmapImageRep? {
        if let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            return NSBitmapImageRep(cgImage: cg)
        }
        guard let tiff = image.tiffRepresentation else { return nil }
        return NSBitmapImageRep(data: tiff)
    }

    /// The bytes to store for a pending attachment, and the name to store them
    /// under. Returns nil only when an image cannot be encoded at all.
    ///
    /// A file's own bytes are kept whenever the server can read them, which is
    /// what writing attachments out is for: a dropped PNG stays lossless and a
    /// photo keeps its pixels. Anything else is converted, because the
    /// alternative is an image that reaches the server and decodes to nothing.
    static func payload(for pending: PendingImage, id: UUID) -> (data: Data, name: String)? {
        if let original = pending.original, !original.isEmpty {
            if let ext = sniffedExt(for: original) {
                return (original, filename(id: id, name: pending.filename, ext: ext))
            }
            // A format only macOS can open: HEIC (the iPhone default), TIFF,
            // BMP, a camera raw, whatever the next system version adds. It is
            // already decoded - `pending.image` is what NSImage made of it - so
            // it is re-encoded from that rather than refused.
            if let out = transcoded(pending.image) {
                return (out.data, filename(id: id, name: pending.filename, ext: out.ext))
            }
        }
        guard let png = pngData(from: pending.image) else { return nil }
        return (png, filename(id: id, name: pending.filename, ext: "png"))
    }

    /// The format these bytes are in, or nil when it is not one the SERVER can
    /// read.
    ///
    /// These four are exactly what `decodeRgbOwned` handles: stb_image for
    /// PNG/JPEG/GIF, then libwebp. There is no HEIC decoder anywhere in the
    /// server and there is not going to be one.
    ///
    /// Answering nil rather than guessing is load-bearing twice over. The
    /// extension is the only record of what a stored file actually contains, and
    /// `payload` uses this same answer to decide whether the bytes can be kept
    /// at all. Guessing "png" put a HEIC under a .png name and sent it to a
    /// decoder that returns nothing for it, which drops the image out of the
    /// prompt without an error anywhere.
    static func sniffedExt(for data: Data) -> String? {
        let b = [UInt8](data.prefix(12))
        if b.count >= 4, b[0] == 0x89, b[1] == 0x50, b[2] == 0x4E, b[3] == 0x47 { return "png" }
        if b.count >= 3, b[0] == 0xFF, b[1] == 0xD8, b[2] == 0xFF { return "jpg" }
        if b.count >= 4, b[0] == 0x47, b[1] == 0x49, b[2] == 0x46, b[3] == 0x38 { return "gif" }
        // RIFF????WEBP - the four bytes between the tags are the container's
        // own size, so the second tag is what tells a WebP from a wave file.
        if b.count >= 12, b[0] == 0x52, b[1] == 0x49, b[2] == 0x46, b[3] == 0x46,
           b[8] == 0x57, b[9] == 0x45, b[10] == 0x42, b[11] == 0x50 { return "webp" }
        return nil
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
        let doomed = paths(in: sessions.filter { ids.contains($0.id) })
        let surviving = paths(in: sessions.filter { !ids.contains($0.id) })
        return orphans(doomed, surviving: surviving, root: root)
    }

    /// The attachments left behind by removing MESSAGES, against the sessions as
    /// they stand afterwards.
    ///
    /// Deleting a message takes its path out of the transcript, and the
    /// transcript is where `removablePaths(deleting:)` looks - so a file whose
    /// message is gone is not orphaned until the chat is deleted, it is
    /// orphaned for good. Same sparing rule: a fork may still hold a copy of the
    /// very message being deleted here.
    static func removablePaths(orphanedBy removed: [ChatMessage],
                               in sessions: [ChatSession],
                               root: String = AttachmentStore.root) -> [String] {
        let gone = Set(removed.flatMap { $0.images ?? [] }.compactMap(\.path))
        return orphans(gone, surviving: paths(in: sessions), root: root)
    }

    private static func paths(in sessions: [ChatSession]) -> Set<String> {
        Set(sessions.flatMap(\.messages).flatMap { $0.images ?? [] }.compactMap(\.path))
    }

    private static func orphans(_ gone: Set<String>,
                                surviving: Set<String>,
                                root: String) -> [String] {
        gone.subtracting(surviving)
            .filter { isInsideRoot($0, root: root) }
            .sorted()
    }

    static func remove(_ path: String) {
        try? FileManager.default.removeItem(atPath: path)
    }
}
