import AppKit
import Foundation

/// The people and pets the templates draw on: `~/.mlx-serve/cast/index.json`
/// plus one JPEG per member, all on this Mac. Memory is authoritative; a failed
/// write never loses the member the user just added.
@MainActor
final class CastStore: ObservableObject {
    @Published private(set) var members: [CastMember] = []

    let rootDir: URL
    private var indexURL: URL { rootDir.appendingPathComponent("index.json") }

    init(rootDir: URL? = nil) {
        self.rootDir = rootDir ?? URL(fileURLWithPath: NSString(string: "~/.mlx-serve/cast").expandingTildeInPath,
                                      isDirectory: true)
        if let data = try? Data(contentsOf: indexURL),
           let decoded = try? Self.decoder.decode([CastMember].self, from: data) {
            members = decoded
        }
    }

    func photoURL(_ m: CastMember) -> URL { rootDir.appendingPathComponent(m.photo) }

    /// Adds a member, keeping a re-encoded copy of `photo`.
    @discardableResult
    func add(_ member: CastMember, photo: URL) throws -> CastMember {
        var m = member
        m.photo = try importPhoto(photo, id: m.id)
        members.append(m)
        persist()
        return m
    }

    /// Saves an edit; a new `photo` replaces the old file.
    func update(_ member: CastMember, photo: URL? = nil) throws {
        guard let i = members.firstIndex(where: { $0.id == member.id }) else { return }
        var m = member
        if let photo { m.photo = try importPhoto(photo, id: m.id) }
        members[i] = m
        persist()
    }

    func delete(_ id: UUID) {
        guard let m = members.first(where: { $0.id == id }) else { return }
        try? FileManager.default.removeItem(at: photoURL(m))
        members.removeAll { $0.id == id }
        persist()
    }

    /// The photo as a JPEG at most 1536 px on its long side: H3 reads PNG and
    /// JPEG only (a HEIC from Photos is converted), and a reference rides every
    /// sampling step, so a 48 MP original buys nothing.
    private func importPhoto(_ url: URL, id: UUID) throws -> String {
        guard let image = NSImage(contentsOf: url),
              let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw CastError.unreadablePhoto(url.lastPathComponent)
        }
        let scale = min(1, 1536 / Double(max(cg.width, cg.height)))
        let w = max(1, Int(Double(cg.width) * scale)), h = max(1, Int(Double(cg.height) * scale))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw CastError.unreadablePhoto(url.lastPathComponent)
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let scaled = ctx.makeImage(),
              let jpeg = NSBitmapImageRep(cgImage: scaled).representation(using: .jpeg, properties: [.compressionFactor: 0.9])
        else { throw CastError.unreadablePhoto(url.lastPathComponent) }
        try FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        // A fresh name per import: the stage and the cast strip cache by URL.
        let name = "\(id.uuidString)-\(Int(Date().timeIntervalSince1970 * 1000)).jpg"
        try jpeg.write(to: rootDir.appendingPathComponent(name), options: .atomic)
        if let old = members.first(where: { $0.id == id }), !old.photo.isEmpty, old.photo != name {
            try? FileManager.default.removeItem(at: photoURL(old))
        }
        return name
    }

    private func persist() {
        try? FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        if let data = try? Self.encoder.encode(members) { try? data.write(to: indexURL, options: .atomic) }
    }

    enum CastError: LocalizedError {
        case unreadablePhoto(String)
        var errorDescription: String? {
            switch self {
            case .unreadablePhoto(let name): return L10n.format("Couldn't read %@ as a photo.", name)
            }
        }
    }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}
