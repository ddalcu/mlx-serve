import XCTest
@testable import MLXCore

/// `OCIClient.extractLayer` shelled out to `/usr/bin/tar`. That binary is not
/// reachable from inside the App Sandbox container, and spawning it is exactly
/// the kind of host escape the Agent Sandbox is supposed to prevent — so layer
/// unpacking moves in-process.
///
/// The reader must handle everything a Docker/OCI layer actually contains:
/// directories, regular files with their mode bits, symlinks, hardlinks, and
/// >100-character paths (which bsdtar emits as pax `path` records or GNU `L`
/// entries). It must also refuse `..` traversal — we unpack untrusted images.
final class TarReaderTests: XCTestCase {

    private func tempDir(_ label: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("tar-\(label)-\(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Build a real tarball with the system tar — a *fixture* tool, not a
    /// production dependency.
    private func makeFixtureTar() throws -> Data {
        let src = try tempDir("src")
        let fm = FileManager.default

        try fm.createDirectory(at: src.appendingPathComponent("dir/nested"), withIntermediateDirectories: true)
        try "hello\n".write(to: src.appendingPathComponent("dir/file.txt"), atomically: true, encoding: .utf8)

        try "#!/bin/sh\n".write(to: src.appendingPathComponent("run.sh"), atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: src.appendingPathComponent("run.sh").path)

        try fm.createSymbolicLink(atPath: src.appendingPathComponent("link.txt").path,
                                  withDestinationPath: "dir/file.txt")
        try fm.linkItem(at: src.appendingPathComponent("dir/file.txt"),
                        to: src.appendingPathComponent("hard.txt"))

        // >100 chars forces bsdtar out of the plain ustar name field.
        let long = "dir/nested/" + String(repeating: "abcdefghij", count: 12) + "/deep.txt"
        try fm.createDirectory(at: src.appendingPathComponent((long as NSString).deletingLastPathComponent),
                               withIntermediateDirectories: true)
        try "deep\n".write(to: src.appendingPathComponent(long), atomically: true, encoding: .utf8)

        let tarURL = try tempDir("out").appendingPathComponent("fixture.tar")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        p.arguments = ["-cf", tarURL.path, "-C", src.path, "."]
        try p.run()
        p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0, "fixture tar failed")
        return try Data(contentsOf: tarURL)
    }

    func testExtractsFilesDirectoriesAndModes() throws {
        let out = try tempDir("dst")
        try TarReader.extract(try makeFixtureTar(), into: out)
        let fm = FileManager.default

        XCTAssertTrue(fm.fileExists(atPath: out.appendingPathComponent("dir/nested").path))
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("dir/file.txt"), encoding: .utf8), "hello\n")

        let mode = try XCTUnwrap(
            fm.attributesOfItem(atPath: out.appendingPathComponent("run.sh").path)[.posixPermissions] as? NSNumber)
        XCTAssertEqual(mode.int16Value & 0o777, 0o755, "executable bit lost")
    }

    func testExtractsSymlinkAsSymlinkNotACopy() throws {
        let out = try tempDir("dst")
        try TarReader.extract(try makeFixtureTar(), into: out)
        let dest = try FileManager.default.destinationOfSymbolicLink(
            atPath: out.appendingPathComponent("link.txt").path)
        XCTAssertEqual(dest, "dir/file.txt")
    }

    func testExtractsHardlink() throws {
        let out = try tempDir("dst")
        try TarReader.extract(try makeFixtureTar(), into: out)
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("hard.txt"), encoding: .utf8), "hello\n")
    }

    func testExtractsPathsLongerThanTheUstarNameField() throws {
        let out = try tempDir("dst")
        try TarReader.extract(try makeFixtureTar(), into: out)
        let long = "dir/nested/" + String(repeating: "abcdefghij", count: 12) + "/deep.txt"
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent(long), encoding: .utf8), "deep\n")
    }

    func testLaterEntryOverwritesEarlierOne() throws {
        let out = try tempDir("dst")
        var tar = Data()
        tar.append(TarReaderTests.ustarEntry(name: "a.txt", body: "first\n"))
        tar.append(TarReaderTests.ustarEntry(name: "a.txt", body: "second\n"))
        tar.append(Data(repeating: 0, count: 1024))
        try TarReader.extract(tar, into: out)
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("a.txt"), encoding: .utf8), "second\n")
    }

    // MARK: - Hostile input

    func testRejectsParentDirectoryTraversal() throws {
        let out = try tempDir("dst")
        var tar = Data()
        tar.append(TarReaderTests.ustarEntry(name: "../escaped.txt", body: "pwned\n"))
        tar.append(Data(repeating: 0, count: 1024))

        XCTAssertThrowsError(try TarReader.extract(tar, into: out)) { error in
            XCTAssertTrue("\(error)".contains("escapes"), "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: out.deletingLastPathComponent().appendingPathComponent("escaped.txt").path))
    }

    func testRejectsAbsolutePathEntry() throws {
        let out = try tempDir("dst")
        var tar = Data()
        tar.append(TarReaderTests.ustarEntry(name: "/tmp/absolute-escape.txt", body: "pwned\n"))
        tar.append(Data(repeating: 0, count: 1024))
        // A leading `/` is stripped (tar's own convention), landing inside `out`.
        try TarReader.extract(tar, into: out)
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.appendingPathComponent("tmp/absolute-escape.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: "/tmp/absolute-escape.txt"))
    }

    func testTruncatedArchiveThrows() throws {
        let out = try tempDir("dst")
        let tar = TarReaderTests.ustarEntry(name: "a.txt", body: "hello\n").prefix(700)
        XCTAssertThrowsError(try TarReader.extract(Data(tar), into: out))
    }

    /// The classic tar-slip: a symlink entry, then a regular file *through* it.
    /// The second entry's path is lexically inside the destination, so a plain
    /// prefix check passes it — and the write lands wherever the symlink points.
    func testRejectsWriteThroughASymlinkPlantedByTheArchive() throws {
        let out = try tempDir("dst")
        let outsideDir = try tempDir("outside")
        let victim = outsideDir.appendingPathComponent("victim.txt")

        var tar = Data()
        tar.append(TarReaderTests.symlinkEntry(name: "evil", target: outsideDir.path))
        tar.append(TarReaderTests.ustarEntry(name: "evil/victim.txt", body: "pwned\n"))
        tar.append(Data(repeating: 0, count: 1024))

        XCTAssertThrowsError(try TarReader.extract(tar, into: out)) { error in
            XCTAssertTrue("\(error)".contains("escapes"), "\(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: victim.path),
                       "archive wrote through its own symlink to \(victim.path)")
    }

    /// GNU tar emits an `L` entry for long names instead of a pax `path` record.
    /// bsdtar never produces one, so the fixture archive can't cover this.
    func testGnuLongNameEntry() throws {
        let out = try tempDir("dst")
        let long = "a/" + String(repeating: "z", count: 120) + ".txt"
        var tar = Data()
        tar.append(TarReaderTests.gnuLongNameEntry(long))
        tar.append(TarReaderTests.ustarEntry(name: "ignored-placeholder", body: "gnu\n"))
        tar.append(Data(repeating: 0, count: 1024))
        try TarReader.extract(tar, into: out)
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent(long), encoding: .utf8), "gnu\n")
    }

    /// A pax `size` record overrides the header's size field. Get it wrong and
    /// every following header is read at the wrong offset — silent corruption,
    /// not an error.
    func testPaxSizeRecordOverridesHeaderSize() throws {
        let out = try tempDir("dst")
        // The body must span more than one block: with a single block, skipping
        // zero bytes lands the reader exactly on the next real header by luck,
        // and the test would pass even with pax `size` ignored.
        let body = String(repeating: "A", count: 1024)
        var tar = Data()
        tar.append(TarReaderTests.paxSizeEntry(name: "big.txt", realSize: 1024))
        tar.append(TarReaderTests.ustarEntry(name: "big.txt", body: body, declaredSize: 0))
        tar.append(TarReaderTests.ustarEntry(name: "after.txt", body: "next\n"))
        tar.append(Data(repeating: 0, count: 1024))
        try TarReader.extract(tar, into: out)

        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("big.txt"), encoding: .utf8), body)
        // Ignoring the pax size desyncs the stream: the reader would parse
        // big.txt's first body block as a header and never reach after.txt.
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("after.txt"), encoding: .utf8), "next\n")
    }

    // MARK: - OCIClient integration (the call site that used /usr/bin/tar)

    func testExtractLayerHandlesAGzippedTarball() throws {
        let out = try tempDir("dst")
        let staging = try tempDir("gz")
        let plain = staging.appendingPathComponent("layer.tar")
        try makeFixtureTar().write(to: plain)

        let gz = staging.appendingPathComponent("layer.tar.gz")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/gzip")
        p.arguments = ["-c", plain.path]
        let pipe = Pipe()
        p.standardOutput = pipe
        try p.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        try data.write(to: gz)

        try OCIClient.extractLayer(gz, into: out)
        XCTAssertEqual(try String(contentsOf: out.appendingPathComponent("dir/file.txt"), encoding: .utf8), "hello\n")
    }

    func testExtractLayerRejectsZstdRatherThanCorruptingTheRootfs() throws {
        let out = try tempDir("dst")
        let staging = try tempDir("zst")
        let file = staging.appendingPathComponent("layer.tar.zst")
        // zstd magic + junk. We must fail loudly, not treat it as a plain tar.
        try Data([0x28, 0xb5, 0x2f, 0xfd] + [UInt8](repeating: 0x41, count: 600)).write(to: file)

        XCTAssertThrowsError(try OCIClient.extractLayer(file, into: out)) { error in
            XCTAssertTrue("\(error)".contains("zstd"), "\(error)")
        }
    }

    // MARK: - Differential test against the system tar on a real image layer
    //
    // Fixtures prove the format handling; a production Debian rootfs proves the
    // reader on ~10k entries of real hardlinks, symlinks, long paths and modes.
    // Opt in:
    //   TARREADER_REAL_LAYER=/path/to/layer.tar.gz swift test --filter testRealDockerLayer

    private struct Entry: Equatable, Comparable {
        let path: String, kind: String, size: Int, link: String, executable: Bool
        static func < (a: Entry, b: Entry) -> Bool { a.path < b.path }
    }

    private func manifest(of root: URL) throws -> [Entry] {
        let fm = FileManager.default
        var out: [Entry] = []
        let walker = fm.enumerator(at: root, includingPropertiesForKeys: nil,
                                   options: [.producesRelativePathURLs])
        for case let url as URL in walker! {
            let p = url.relativePath
            let attrs = try fm.attributesOfItem(atPath: url.path) // lstat: does not follow symlinks
            let type = attrs[.type] as? FileAttributeType
            let mode = (attrs[.posixPermissions] as? NSNumber)?.intValue ?? 0
            if type == .typeSymbolicLink {
                out.append(Entry(path: p, kind: "link", size: 0,
                                 link: (try? fm.destinationOfSymbolicLink(atPath: url.path)) ?? "",
                                 executable: false))
            } else if type == .typeDirectory {
                out.append(Entry(path: p, kind: "dir", size: 0, link: "", executable: false))
            } else {
                out.append(Entry(path: p, kind: "file", size: (attrs[.size] as? Int) ?? -1,
                                 link: "", executable: mode & 0o111 != 0))
            }
        }
        return out.sorted()
    }

    func testRealDockerLayerMatchesSystemTar() throws {
        guard let layer = ProcessInfo.processInfo.environment["TARREADER_REAL_LAYER"] else {
            throw XCTSkip("set TARREADER_REAL_LAYER to a .tar.gz image layer")
        }
        let mine = try tempDir("mine")
        let theirs = try tempDir("theirs")

        try OCIClient.extractLayer(URL(fileURLWithPath: layer), into: mine)

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        p.arguments = ["-x", "-f", layer, "-C", theirs.path]
        p.standardError = FileHandle.nullDevice
        try p.run(); p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0)

        let a = try manifest(of: mine), b = try manifest(of: theirs)
        // Guard against a vacuous pass: two empty walks compare equal.
        XCTAssertGreaterThan(a.count, 1000, "expected a real rootfs, walked \(a.count) entries")
        XCTAssertEqual(a.count, b.count, "entry count differs (mine \(a.count) vs tar \(b.count))")

        let byPath = Dictionary(uniqueKeysWithValues: b.map { ($0.path, $0) })
        var mismatches: [String] = []
        for e in a {
            guard let t = byPath[e.path] else { mismatches.append("only in TarReader: \(e.path)"); continue }
            if e != t { mismatches.append("\(e.path): mine=\(e) tar=\(t)") }
        }
        for e in b where !a.contains(where: { $0.path == e.path }) {
            mismatches.append("only in system tar: \(e.path)")
        }
        XCTAssertTrue(mismatches.isEmpty, "\(mismatches.count) mismatches:\n" + mismatches.prefix(20).joined(separator: "\n"))
    }

    // MARK: - Hand-rolled headers (for cases the system tar won't emit)

    /// One 512-byte ustar header plus a NUL-padded body.
    /// `declaredSize` overrides the size field (to fake a pax-sized entry);
    /// `linkTarget` fills the linkname field.
    static func entry(name: String,
                      body: Data,
                      typeflag: UInt8,
                      declaredSize: Int? = nil,
                      linkTarget: String = "") -> Data {
        var header = [UInt8](repeating: 0, count: 512)
        func put(_ s: String, _ offset: Int, _ len: Int) {
            for (i, b) in Array(s.utf8).prefix(len - 1).enumerated() { header[offset + i] = b }
        }
        put(name, 0, 100)
        put("000644 ", 100, 8)
        put("000000 ", 108, 8)
        put("000000 ", 116, 8)
        put(String(format: "%011o ", declaredSize ?? body.count), 124, 12)
        put(String(format: "%011o ", 0), 136, 12)
        header[156] = typeflag
        put(linkTarget, 157, 100)
        put("ustar", 257, 6)
        header[263] = UInt8(ascii: "0"); header[264] = UInt8(ascii: "0")

        // Checksum: computed with the checksum field itself read as spaces.
        for i in 148..<156 { header[i] = UInt8(ascii: " ") }
        let sum = header.reduce(0) { $0 + Int($1) }
        put(String(format: "%06o", sum), 148, 8)
        header[154] = 0
        header[155] = UInt8(ascii: " ")

        var out = Data(header)
        out.append(body)
        let pad = (512 - body.count % 512) % 512
        out.append(Data(repeating: 0, count: pad))
        return out
    }

    static func ustarEntry(name: String, body: String, declaredSize: Int? = nil) -> Data {
        entry(name: name, body: Data(body.utf8), typeflag: UInt8(ascii: "0"), declaredSize: declaredSize)
    }

    static func symlinkEntry(name: String, target: String) -> Data {
        entry(name: name, body: Data(), typeflag: UInt8(ascii: "2"), linkTarget: target)
    }

    /// GNU `L`: the body is the real path of the *next* entry.
    static func gnuLongNameEntry(_ path: String) -> Data {
        var body = Data(path.utf8)
        body.append(0)
        return entry(name: "././@LongLink", body: body, typeflag: UInt8(ascii: "L"))
    }

    /// pax `x`: `"<len> size=<n>\n"`, where `<len>` counts itself.
    static func paxSizeEntry(name: String, realSize: Int) -> Data {
        func record(_ text: String) -> String {
            var len = text.utf8.count + 2 // " " + "\n"
            while String(len).utf8.count + text.utf8.count + 2 != len { len += 1 }
            return "\(len) \(text)\n"
        }
        let body = Data(record("size=\(realSize)").utf8)
        return entry(name: "PaxHeader/\(name)", body: body, typeflag: UInt8(ascii: "x"))
    }
}
