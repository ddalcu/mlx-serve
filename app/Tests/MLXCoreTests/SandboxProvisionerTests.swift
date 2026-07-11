import XCTest
@testable import MLXCore

/// The App Store build ships its guest inside the bundle; the Developer ID
/// build downloads it. `SandboxProvisioner` is the seam, and `BundledProvisioner`
/// unpacks `Contents/Resources/guest/` into the container using the same
/// in-process `TarReader` as the OCI puller — no `/usr/bin/tar`, no network.
final class SandboxProvisionerTests: XCTestCase {

    // MARK: selection

    func testDeveloperIDDownloadsAndAppStoreBundles() {
        XCTAssertEqual(SandboxProvisionerFactory.kind(ociPullAllowed: true), .downloading)
        XCTAssertEqual(SandboxProvisionerFactory.kind(ociPullAllowed: false), .bundled)
    }

    /// The factory's default must track the build: the store build's `ociPull`
    /// is false, so it can never choose the downloading arm.
    func testDefaultKindMatchesTheBuild() {
        #if MAS_BUILD
        XCTAssertEqual(SandboxProvisionerFactory.kind(), .bundled)
        #else
        XCTAssertEqual(SandboxProvisionerFactory.kind(), .downloading)
        #endif
    }

    // MARK: DownloadingProvisioner just delegates

    func testDownloadingProvisionerDelegatesToItsClosures() throws {
        var pulledImage: String?
        let provisioner = DownloadingProvisioner(
            fetchKernel: { "/cache/kernel" },
            pullRootfs: { image in pulledImage = image; return "/cache/rootfs/\(image)" })

        XCTAssertEqual(try provisioner.kernelPath(), "/cache/kernel")
        XCTAssertEqual(try provisioner.rootfsDir(image: "debian:slim"), "/cache/rootfs/debian:slim")
        XCTAssertEqual(pulledImage, "debian:slim")
    }

    // MARK: BundledProvisioner

    /// Build a fake `Contents/Resources/guest/` with a real gzipped tar rootfs
    /// and a kernel file, then unpack it exactly as the app would.
    private func makeBundle() throws -> (resources: URL, container: URL) {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bundle-\(UUID().uuidString)", isDirectory: true)
        let guest = base.appendingPathComponent("Resources/guest", isDirectory: true)
        let container = base.appendingPathComponent("Container", isDirectory: true)
        try FileManager.default.createDirectory(at: guest, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: base) }

        // Kernel: opaque bytes; the provisioner only checks it exists.
        try Data("fake-kernel".utf8).write(to: guest.appendingPathComponent("kernel"))

        // Rootfs: a minimal tree with an executable /bin/sh, tarred + gzipped by
        // the system tools (fixture tools, not production deps).
        let src = base.appendingPathComponent("src/bin", isDirectory: true)
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try Data("#!/bin/true\n".utf8).write(to: src.appendingPathComponent("sh"))
        try FileManager.default.setAttributes([.posixPermissions: 0o755],
                                              ofItemAtPath: src.appendingPathComponent("sh").path)
        try Data("hello\n".utf8).write(to: base.appendingPathComponent("src/motd"))

        let tar = base.appendingPathComponent("rootfs.tar")
        run("/usr/bin/tar", ["-cf", tar.path, "-C", base.appendingPathComponent("src").path, "."])
        run("/usr/bin/gzip", ["-f", tar.path]) // -> rootfs.tar.gz
        try FileManager.default.moveItem(at: base.appendingPathComponent("rootfs.tar.gz"),
                                         to: guest.appendingPathComponent("rootfs.tar.gz"))

        return (base.appendingPathComponent("Resources"), container)
    }

    private func run(_ path: String, _ args: [String]) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        p.standardError = FileHandle.nullDevice
        try? p.run(); p.waitUntilExit()
    }

    func testBundledProvisionerUnpacksTheRootfsIntoTheContainer() throws {
        let (resources, container) = try makeBundle()
        let provisioner = BundledProvisioner(resourcesURL: resources, containerRoot: container)

        XCTAssertEqual(try provisioner.kernelPath(), resources.appendingPathComponent("guest/kernel").path)

        let rootfs = try provisioner.rootfsDir(image: "bundled")
        let fm = FileManager.default
        XCTAssertTrue(fm.isExecutableFile(atPath: (rootfs as NSString).appendingPathComponent("bin/sh")))
        XCTAssertEqual(try String(contentsOfFile: (rootfs as NSString).appendingPathComponent("motd"), encoding: .utf8),
                       "hello\n")
    }

    /// The second call must reuse the unpack, not redo it — the version marker
    /// is what proves the first one finished.
    func testSecondCallReusesTheUnpack() throws {
        let (resources, container) = try makeBundle()
        let provisioner = BundledProvisioner(resourcesURL: resources, containerRoot: container)

        let first = try provisioner.rootfsDir(image: "bundled")
        // Drop a sentinel; if the provisioner re-unpacked, it would be gone.
        let sentinel = (first as NSString).appendingPathComponent(".reuse-probe")
        FileManager.default.createFile(atPath: sentinel, contents: Data())

        let second = try provisioner.rootfsDir(image: "bundled")
        XCTAssertEqual(first, second)
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel), "the unpack was redone")
    }

    /// A crash mid-unpack leaves the tree with no version marker; the next run
    /// must discard it and start clean rather than boot a half-populated rootfs.
    func testPartialUnpackIsNotTrusted() throws {
        let (resources, container) = try makeBundle()
        let destination = container.appendingPathComponent("images/bundled", isDirectory: true)
        try FileManager.default.createDirectory(at: destination.appendingPathComponent("bin"),
                                                withIntermediateDirectories: true)
        // Looks started (bin/sh present) but no marker → not unpacked.
        FileManager.default.createFile(atPath: destination.appendingPathComponent("bin/sh").path,
                                       contents: Data("stale".utf8),
                                       attributes: [.posixPermissions: 0o755])
        XCTAssertFalse(BundledProvisioner.isUnpacked(at: destination))

        let provisioner = BundledProvisioner(resourcesURL: resources, containerRoot: container)
        _ = try provisioner.rootfsDir(image: "bundled")
        XCTAssertTrue(BundledProvisioner.isUnpacked(at: destination))
        // The stale content was replaced by the real fixture (/bin/true shebang).
        let sh = try String(contentsOfFile: destination.appendingPathComponent("bin/sh").path, encoding: .utf8)
        XCTAssertTrue(sh.contains("/bin/true"), sh)
    }

    func testMissingKernelThrowsAClearError() {
        let resources = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("empty-\(UUID().uuidString)")
        let provisioner = BundledProvisioner(resourcesURL: resources,
                                             containerRoot: resources.appendingPathComponent("c"))
        XCTAssertThrowsError(try provisioner.kernelPath()) { error in
            XCTAssertTrue("\(error)".contains("guest/kernel"), "\(error)")
        }
    }

    func testMissingRootfsArchiveThrowsAClearError() {
        let resources = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("empty-\(UUID().uuidString)")
        let provisioner = BundledProvisioner(resourcesURL: resources,
                                             containerRoot: resources.appendingPathComponent("c"))
        XCTAssertThrowsError(try provisioner.rootfsDir(image: "bundled")) { error in
            XCTAssertTrue("\(error)".contains("rootfs.tar.gz"), "\(error)")
        }
    }
}
