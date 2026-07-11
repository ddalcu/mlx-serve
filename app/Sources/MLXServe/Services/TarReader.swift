import Foundation

/// In-process tar extractor for OCI image layers and the bundled guest rootfs.
///
/// Replaces a `/usr/bin/tar` subprocess: unreachable from inside the App
/// Sandbox container, and a host escape from the Agent Sandbox. Handles the
/// subset of the format Docker/OCI layers actually use — ustar + pax extended
/// headers + GNU long names — and refuses entries that would escape the
/// destination, because the images are untrusted.
///
/// Not handled, deliberately: device/fifo nodes (the guest mounts devtmpfs),
/// sparse files, and setuid/setgid bits (mirroring the old `tar` invocation,
/// which omitted `-p` since the host user can't restore them anyway).
enum TarReader {

    struct TarError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private static let blockSize = 512

    private enum EntryType {
        case file, directory, symlink, hardlink, skip
        case gnuLongName, gnuLongLink, paxNext, paxGlobal
    }

    /// Extract `data` (a plain, already-decompressed tar stream) into `dir`.
    static func extract(_ data: Data, into destination: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)
        // Resolve ONCE and derive every path from the result. `/var` symlinks to
        // `/private/var` on macOS, so a resolved root compared against an
        // unresolved candidate rejects every entry.
        let dir = destination.resolvingSymlinksInPath().standardizedFileURL
        let rootPath = dir.path

        var offset = 0
        // Carried across entries by the `L`/`K`/`x` metadata headers.
        var pendingName: String?
        var pendingLink: String?
        var pendingSize: Int?

        while offset + blockSize <= data.count {
            let header = data.subdata(in: offset ..< offset + blockSize)
            offset += blockSize

            // Two consecutive zero blocks terminate the archive; one is enough
            // to stop, since nothing valid follows.
            if header.allSatisfy({ $0 == 0 }) { break }

            guard let headerSize = octal(header, 124, 12) else {
                throw TarError(message: "tar: malformed size field")
            }
            let type = entryType(header[header.startIndex + 156])
            let isMeta: Bool
            switch type {
            case .gnuLongName, .gnuLongLink, .paxNext, .paxGlobal: isMeta = true
            default: isMeta = false
            }
            // A pax `size` record overrides the header field (files > 8 GB write
            // 0 there). Honoring it is not optional: get the body length wrong
            // and every subsequent header is read at the wrong offset.
            let size = isMeta ? headerSize : (pendingSize ?? headerSize)

            let bodyBlocks = (size + blockSize - 1) / blockSize * blockSize
            guard size >= 0, offset + bodyBlocks <= data.count else {
                throw TarError(message: "tar: truncated archive (entry needs \(bodyBlocks) bytes)")
            }
            let body = data.subdata(in: offset ..< offset + size)
            offset += bodyBlocks

            switch type {
            case .gnuLongName:
                pendingName = cString(body)
                continue
            case .gnuLongLink:
                pendingLink = cString(body)
                continue
            case .paxNext:
                let records = parsePax(body)
                if let p = records["path"] { pendingName = p }
                if let l = records["linkpath"] { pendingLink = l }
                if let s = records["size"], let n = Int(s) { pendingSize = n }
                continue
            case .paxGlobal, .skip:
                pendingName = nil; pendingLink = nil; pendingSize = nil
                continue
            case .file, .directory, .symlink, .hardlink:
                break
            }

            let rawName = pendingName ?? ustarName(header)
            let rawLink = pendingLink ?? string(header, 157, 100)
            pendingName = nil; pendingLink = nil; pendingSize = nil
            if rawName.isEmpty { continue }

            let target = try resolve(rawName, under: dir, rootPath: rootPath)
            let mode = octal(header, 100, 8).map { $0 & 0o777 } ?? 0o644

            switch type {
            case .directory:
                try fm.createDirectory(at: target, withIntermediateDirectories: true,
                                       attributes: [.posixPermissions: mode])

            case .file:
                try fm.createDirectory(at: target.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                try? fm.removeItem(at: target) // later layers replace earlier ones
                guard fm.createFile(atPath: target.path, contents: body,
                                    attributes: [.posixPermissions: mode]) else {
                    throw TarError(message: "tar: could not write \(rawName)")
                }

            case .symlink:
                try fm.createDirectory(at: target.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                try? fm.removeItem(at: target)
                try fm.createSymbolicLink(atPath: target.path, withDestinationPath: rawLink)

            case .hardlink:
                // The link target is archive-relative, so it must also be confined.
                let source = try resolve(rawLink, under: dir, rootPath: rootPath)
                try fm.createDirectory(at: target.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                try? fm.removeItem(at: target)
                // A layer may reference a file a *previous* layer supplied; if
                // the link fails, fall back to a copy rather than losing it.
                do { try fm.linkItem(at: source, to: target) }
                catch { try? fm.copyItem(at: source, to: target) }

            default:
                break
            }
        }
    }

    // MARK: - Path confinement

    /// Strips the leading `/` (tar's own convention) and rejects anything that
    /// still resolves outside `rootPath` — `../../etc/passwd`, or a `..` that
    /// only escapes after normalization.
    ///
    /// The lexical check alone is NOT enough. An archive can ship a symlink
    /// `evil -> /etc` and then a regular file `evil/passwd`: the second entry's
    /// path is lexically inside the destination, but writing it follows the
    /// symlink and lands in `/etc`. This is the classic tar-slip, and it is why
    /// every existing ancestor is checked for being a symlink before we write.
    private static func resolve(_ name: String, under dir: URL, rootPath: String) throws -> URL {
        var rel = name
        while rel.hasPrefix("/") { rel.removeFirst() }
        guard !rel.isEmpty else { throw TarError(message: "tar: empty entry name") }

        let candidate = dir.appendingPathComponent(rel).standardizedFileURL
        let path = candidate.path
        guard path == rootPath || path.hasPrefix(rootPath + "/") else {
            throw TarError(message: "tar: entry '\(name)' escapes the destination directory")
        }

        // Walk the components between the root and the entry; any that already
        // exists as a symlink could redirect the write outside the root.
        let relative = String(path.dropFirst(rootPath.count)).split(separator: "/").dropLast()
        var walk = URL(fileURLWithPath: rootPath)
        for component in relative {
            walk.appendPathComponent(String(component))
            let values = try? walk.resourceValues(forKeys: [.isSymbolicLinkKey])
            if values?.isSymbolicLink == true {
                throw TarError(message: "tar: entry '\(name)' escapes the destination directory via the symlink '\(component)'")
            }
        }
        return candidate
    }

    // MARK: - Header decoding

    private static func entryType(_ flag: UInt8) -> EntryType {
        switch flag {
        case UInt8(ascii: "0"), 0, UInt8(ascii: "7"): return .file
        case UInt8(ascii: "1"):                       return .hardlink
        case UInt8(ascii: "2"):                       return .symlink
        case UInt8(ascii: "5"):                       return .directory
        case UInt8(ascii: "L"):                       return .gnuLongName
        case UInt8(ascii: "K"):                       return .gnuLongLink
        case UInt8(ascii: "x"), UInt8(ascii: "X"):    return .paxNext
        case UInt8(ascii: "g"):                       return .paxGlobal
        default:                                      return .skip // char/block/fifo
        }
    }

    /// ustar splits long-ish names as `prefix/name`.
    private static func ustarName(_ header: Data) -> String {
        let name = string(header, 0, 100)
        let magic = string(header, 257, 6)
        guard magic.hasPrefix("ustar") else { return name }
        let prefix = string(header, 345, 155)
        return prefix.isEmpty ? name : prefix + "/" + name
    }

    private static func string(_ header: Data, _ offset: Int, _ length: Int) -> String {
        let start = header.startIndex + offset
        return cString(header.subdata(in: start ..< start + length))
    }

    private static func cString(_ data: Data) -> String {
        let bytes = data.prefix(while: { $0 != 0 })
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Octal ASCII, space/NUL padded. GNU sets the high bit of byte 0 to signal
    /// a base-256 big-endian integer instead (files > 8 GB, and large uid/gid).
    private static func octal(_ header: Data, _ offset: Int, _ length: Int) -> Int? {
        let start = header.startIndex + offset
        let field = header.subdata(in: start ..< start + length)
        guard let first = field.first else { return nil }

        if first & 0x80 != 0 {
            var value = Int(first & 0x7F)
            for byte in field.dropFirst() { value = value << 8 | Int(byte) }
            return value
        }
        let text = String(decoding: field.prefix(while: { $0 != 0 && $0 != UInt8(ascii: " ") }), as: UTF8.self)
        if text.isEmpty { return 0 }
        return Int(text, radix: 8)
    }

    /// pax records are `"<len> <key>=<value>\n"`, where `<len>` counts itself.
    static func parsePax(_ body: Data) -> [String: String] {
        var records: [String: String] = [:]
        var rest = Data(body)
        while let space = rest.firstIndex(of: UInt8(ascii: " ")) {
            let lenText = String(decoding: rest[rest.startIndex ..< space], as: UTF8.self)
            guard let len = Int(lenText), len > 0, rest.count >= len else { break }
            let record = rest[rest.startIndex ..< rest.index(rest.startIndex, offsetBy: len)]
            let payload = record[rest.index(after: space)...].dropLast() // trailing \n
            let text = String(decoding: payload, as: UTF8.self)
            if let eq = text.firstIndex(of: "=") {
                records[String(text[..<eq])] = String(text[text.index(after: eq)...])
            }
            rest = Data(rest[rest.index(rest.startIndex, offsetBy: len)...])
        }
        return records
    }
}
