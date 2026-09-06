import Foundation

/// The case-sensitive volume the sandbox rootfs lives on.
///
/// A Debian userland unpacked onto the user's (case-insensitive) APFS volume
/// is corrupted by the first package with case-colliding paths: libwww-perl
/// installs `/usr/bin/HEAD` over coreutils' `head`, and the guest's `head`
/// becomes an HTTP client. So the images directory is the mountpoint of a
/// sparse bundle formatted "Case-sensitive APFS", attached with hdiutil at
/// the SAME path every provisioner already uses (`~/.mlx-serve/sandbox/images`)
/// — rootfs lookup, re-pull, reset, ssh injection and the desktop scripts
/// need no path changes.
///
/// Pure helpers + one injectable `Tools` seam (hdiutil, the mountpoint probe,
/// the build's permission), so the ordering and the fallbacks are unit-tested
/// without touching disks. The App Store build cannot run hdiutil and keeps
/// the loose directory, with one transcript line naming the caveat.
enum SandboxVolume {
    static let bundleName = "rootfs.sparsebundle"
    static let mountDirName = "images"
    static let movedAsideDirName = "images-old"
    static let sizeSpec = "64g"
    static let volumeName = "mlx-sandbox"
    static let filesystem = "Case-sensitive APFS"
    static let hdiutil = "/usr/bin/hdiutil"

    struct Paths: Equatable {
        let bundle: URL
        let mountpoint: URL
        let movedAside: URL
    }

    static func paths(root: URL) -> Paths {
        Paths(bundle: root.appendingPathComponent(bundleName),
              mountpoint: root.appendingPathComponent(mountDirName, isDirectory: true),
              movedAside: root.appendingPathComponent(movedAsideDirName, isDirectory: true))
    }

    /// `hdiutil create` — one bundle, sparse (grows with the rootfs, ~2 GB for
    /// the base image), so the 64 GB cap costs nothing up front.
    static func createArgs(bundle: URL) -> [String] {
        ["create", "-type", "SPARSEBUNDLE", "-fs", filesystem, "-size", sizeSpec,
         "-volname", volumeName, "-quiet", bundle.path]
    }

    /// `hdiutil attach` at our own mountpoint, invisible to Finder, with
    /// ownership ENFORCED: the default user mount ignores owners, every file
    /// then reads as the mounting user, and the guest's `_apt` (which drops
    /// root to download) cannot write its own 700 cache dir — apt fails with
    /// "Permission denied" on every package (live 2026-09-06).
    static func attachArgs(bundle: URL, mountpoint: URL) -> [String] {
        ["attach", "-nobrowse", "-noautoopen", "-owners", "on", "-mountpoint", mountpoint.path, bundle.path]
    }

    static func detachArgs(mountpoint: URL) -> [String] {
        ["detach", mountpoint.path, "-force"]
    }

    /// The mountpoint `hdiutil attach` reports: its last tab-separated column
    /// on the line that names an Apple_APFS (or plain mounted) volume. nil
    /// when nothing in the output looks mounted (the caller then probes the
    /// path itself — output shape drifts across macOS releases).
    static func parseAttach(_ output: String) -> String? {
        for line in output.split(separator: "\n").reversed() {
            let cols = line.split(separator: "\t", omittingEmptySubsequences: true)
                .map { $0.trimmingCharacters(in: .whitespaces) }
            guard let last = cols.last, last.hasPrefix("/"), cols.count >= 2 else { continue }
            // "/dev/disk5s1  Apple_APFS  /path" — the device column is first;
            // a line whose last column is a device path is not a mount.
            if last.hasPrefix("/dev/") { continue }
            return last
        }
        return nil
    }

    /// Is `path` the root of a mounted filesystem? `statfs` names the mount
    /// point of the filesystem a path lives on; equal (after symlink
    /// resolution) means the path IS one.
    static func isMountpoint(_ path: String) -> Bool {
        var st = statfs()
        guard statfs(path, &st) == 0 else { return false }
        let mounted = withUnsafePointer(to: &st.f_mntonname) { ptr -> String in
            ptr.withMemoryRebound(to: CChar.self, capacity: Int(MAXPATHLEN)) { String(cString: $0) }
        }
        let resolved = URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL.path
        return mounted == resolved || mounted == path
    }

    /// Does the filesystem at `dir` tell `a` from `A`? A probe file, not a
    /// `diskutil` parse: works on any mount, and it is the property we need.
    static func isCaseSensitive(_ dir: URL) -> Bool {
        let fm = FileManager.default
        let lower = dir.appendingPathComponent(".mlx-case-probe")
        let upper = dir.appendingPathComponent(".MLX-CASE-PROBE")
        defer { try? fm.removeItem(at: lower) }
        guard fm.createFile(atPath: lower.path, contents: Data()) else { return false }
        return !fm.fileExists(atPath: upper.path)
    }

    /// A loose `images/` from before the volume existed, worth moving aside
    /// (non-empty, not a mountpoint). Empty = just recreate on top.
    static func hasLooseContents(_ dir: URL, isMountpoint: (String) -> Bool) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue,
              !isMountpoint(dir.path) else { return false }
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return !entries.isEmpty
    }

    // MARK: Attach / detach

    struct Tools {
        /// Run hdiutil; returns exit status + merged output. Throws only when
        /// the tool cannot be launched.
        var run: (_ args: [String]) throws -> (status: Int32, output: String)
        var isMountpoint: (String) -> Bool
        /// The Developer ID build may run hdiutil; the App Store sandbox
        /// refuses it (and its mount would land outside the container).
        var allowed: Bool

        static var live: Tools {
            Tools(run: { args in try runHdiutil(args) },
                  isMountpoint: SandboxVolume.isMountpoint,
                  allowed: !BuildFeatures.current.isMAS)
        }
    }

    enum Outcome: Equatable {
        /// The volume is mounted at `images`. `created` = the bundle was made
        /// on this call; `movedAside` = a loose images dir went to images-old.
        case attached(created: Bool, movedAside: Bool)
        case alreadyAttached
        /// No volume: the loose directory serves, `reason` says why.
        case looseDirectory(reason: String)
    }

    /// Make `images` a mounted case-sensitive volume, creating the bundle on
    /// first use and moving a pre-volume loose directory aside (the re-pull
    /// then happens naturally: the arch marker is gone). Never throws: every
    /// failure degrades to the loose directory with a reason.
    static func ensureAttached(root: URL, tools: Tools = .live) -> Outcome {
        let p = paths(root: root)
        let fm = FileManager.default
        try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        func loose(_ reason: String) -> Outcome {
            try? fm.createDirectory(at: p.mountpoint, withIntermediateDirectories: true)
            return .looseDirectory(reason: reason)
        }
        guard tools.allowed else { return loose("this build cannot mount disk images") }
        if tools.isMountpoint(p.mountpoint.path) { return .alreadyAttached }

        var created = false
        if !fm.fileExists(atPath: p.bundle.path) {
            do {
                let r = try tools.run(createArgs(bundle: p.bundle))
                guard r.status == 0 else { return loose("hdiutil create failed: \(r.output.trimmingCharacters(in: .whitespacesAndNewlines))") }
                created = true
            } catch {
                return loose("hdiutil could not run: \(error.localizedDescription)")
            }
        }
        var movedAside = false
        if hasLooseContents(p.mountpoint, isMountpoint: tools.isMountpoint) {
            try? fm.removeItem(at: p.movedAside)
            do {
                try fm.moveItem(at: p.mountpoint, to: p.movedAside)
                movedAside = true
            } catch {
                return loose("could not move the old images directory aside: \(error.localizedDescription)")
            }
        }
        try? fm.createDirectory(at: p.mountpoint, withIntermediateDirectories: true)
        do {
            let r = try tools.run(attachArgs(bundle: p.bundle, mountpoint: p.mountpoint))
            guard r.status == 0 else { return loose("hdiutil attach failed: \(r.output.trimmingCharacters(in: .whitespacesAndNewlines))") }
        } catch {
            return loose("hdiutil could not run: \(error.localizedDescription)")
        }
        guard tools.isMountpoint(p.mountpoint.path) else {
            return loose("hdiutil attach returned but \(p.mountpoint.path) is not a mountpoint")
        }
        return .attached(created: created, movedAside: movedAside)
    }

    /// Unmount (if mounted). Failure is reported, not thrown: the caller
    /// (reset) deletes the bundle next, and a busy mount just stays.
    @discardableResult
    static func detach(root: URL, tools: Tools = .live) -> Bool {
        let p = paths(root: root)
        guard tools.allowed, tools.isMountpoint(p.mountpoint.path) else { return true }
        let r = try? tools.run(detachArgs(mountpoint: p.mountpoint))
        return r?.status == 0
    }

    /// The transcript line for an outcome (nil = nothing worth saying).
    static func transcriptLine(_ outcome: Outcome) -> String? {
        switch outcome {
        case .alreadyAttached: return nil
        case .attached(let created, let movedAside):
            var s = created ? "sandbox: created a case-sensitive volume for the rootfs (\(bundleName), sparse, up to \(sizeSpec))"
                            : "sandbox: rootfs volume attached (case-sensitive)"
            if movedAside { s += "; the old images directory moved to \(movedAsideDirName)/ and the base image re-pulls" }
            return s
        case .looseDirectory(let reason):
            return "sandbox: rootfs stays on the loose \(mountDirName)/ directory (\(reason)); on a case-insensitive volume a package with case-colliding paths (libwww-perl's HEAD vs head) corrupts the guest"
        }
    }

    private static func runHdiutil(_ args: [String]) throws -> (status: Int32, output: String) {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: hdiutil)
        proc.arguments = args
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        return (proc.terminationStatus, String(decoding: data, as: UTF8.self))
    }
}
