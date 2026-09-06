import XCTest
@testable import MLXCore

/// The rootfs lives on a mounted case-sensitive sparse bundle at the SAME
/// `images/` path the provisioners use. Fake hdiutil + mountpoint probe, so
/// the create → move-aside → attach ordering and every fallback are pinned
/// without touching a disk.
final class SandboxVolumeTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sandbox-volume-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// A fake hdiutil: `create` makes the bundle path exist, `attach` flips
    /// the mountpoint state. Records every invocation.
    private final class Fake {
        var calls: [[String]] = []
        var mounted = Set<String>()
        var failCreate = false
        var failAttach = false
        func tools(allowed: Bool = true) -> SandboxVolume.Tools {
            SandboxVolume.Tools(run: { args in
                self.calls.append(args)
                switch args.first {
                case "create":
                    if self.failCreate { return (1, "hdiutil: create failed - No space left on device") }
                    try FileManager.default.createDirectory(atPath: args.last!, withIntermediateDirectories: true)
                    return (0, "")
                case "attach":
                    if self.failAttach { return (1, "hdiutil: attach failed - image not recognized") }
                    let mp = args[args.firstIndex(of: "-mountpoint")! + 1]
                    self.mounted.insert(mp)
                    return (0, "/dev/disk9\tGUID_partition_scheme\t\n/dev/disk9s1\tApple_APFS\t\(mp)\n")
                case "detach":
                    self.mounted.remove(args[1])
                    return (0, "")
                default: return (2, "unknown")
                }
            }, isMountpoint: { self.mounted.contains($0) }, allowed: allowed)
        }
    }

    func testFirstUseCreatesTheBundleAndAttachesItAtImages() {
        let fake = Fake()
        let outcome = SandboxVolume.ensureAttached(root: root, tools: fake.tools())
        XCTAssertEqual(outcome, .attached(created: true, movedAside: false))
        XCTAssertEqual(fake.calls.map(\.first), ["create", "attach"])
        let create = fake.calls[0]
        XCTAssertTrue(create.contains("SPARSEBUNDLE") && create.contains(SandboxVolume.filesystem), "\(create)")
        XCTAssertEqual(create.last, root.appendingPathComponent(SandboxVolume.bundleName).path)
        let attach = fake.calls[1]
        XCTAssertTrue(attach.contains("-nobrowse"), "not in Finder's sidebar")
        XCTAssertEqual(attach[attach.firstIndex(of: "-owners")! + 1], "on",
                       "ownership enforced: the guest's _apt user must own its cache dir, or apt cannot download")
        XCTAssertEqual(attach[attach.firstIndex(of: "-mountpoint")! + 1],
                       root.appendingPathComponent("images", isDirectory: true).path,
                       "the SAME path every provisioner already uses")
        XCTAssertEqual(SandboxVolume.transcriptLine(outcome)?.contains("case-sensitive"), true)
    }

    func testAnAttachedVolumeIsLeftAlone() {
        let fake = Fake()
        fake.mounted.insert(root.appendingPathComponent("images", isDirectory: true).path)
        XCTAssertEqual(SandboxVolume.ensureAttached(root: root, tools: fake.tools()), .alreadyAttached)
        XCTAssertTrue(fake.calls.isEmpty, "no hdiutil call on a mounted volume")
        XCTAssertNil(SandboxVolume.transcriptLine(.alreadyAttached))
    }

    /// A Mac with a loose pre-volume `images/` (the old layout): moved to
    /// `images-old/` so the marker is gone and the base image re-pulls; the
    /// bundle mounts on a fresh empty `images/`.
    func testALoosePreVolumeImagesDirIsMovedAside() throws {
        let images = root.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images.appendingPathComponent("agent-shell"), withIntermediateDirectories: true)
        let fake = Fake()
        let outcome = SandboxVolume.ensureAttached(root: root, tools: fake.tools())
        XCTAssertEqual(outcome, .attached(created: true, movedAside: true))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("images-old/agent-shell").path))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: images.path), [], "fresh mountpoint")
        XCTAssertEqual(SandboxVolume.transcriptLine(outcome)?.contains("images-old"), true)
    }

    /// Every failure keeps the loose directory (the sandbox still works) and
    /// names why — never a throw out of the boot.
    func testFailuresFallBackToTheLooseDirectoryByName() {
        let images = root.appendingPathComponent("images", isDirectory: true)
        let mas = Fake()
        guard case .looseDirectory(let why) = SandboxVolume.ensureAttached(root: root, tools: mas.tools(allowed: false)) else {
            return XCTFail("MAS falls back")
        }
        XCTAssertTrue(why.contains("cannot mount"), why)
        XCTAssertTrue(FileManager.default.fileExists(atPath: images.path), "the loose dir exists for the provisioner")
        XCTAssertTrue(mas.calls.isEmpty)

        let noSpace = Fake(); noSpace.failCreate = true
        guard case .looseDirectory(let why2) = SandboxVolume.ensureAttached(root: root, tools: noSpace.tools()) else {
            return XCTFail("create failure falls back")
        }
        XCTAssertTrue(why2.contains("No space left"), why2)

        let badAttach = Fake(); badAttach.failAttach = true
        guard case .looseDirectory(let why3) = SandboxVolume.ensureAttached(root: root, tools: badAttach.tools()) else {
            return XCTFail("attach failure falls back")
        }
        XCTAssertTrue(why3.contains("attach failed"), why3)
        XCTAssertEqual(SandboxVolume.transcriptLine(.looseDirectory(reason: "x"))?.contains("HEAD vs head"), true,
                       "the caveat names the failure mode")
    }

    func testParseAttachTakesTheMountedPathAndIgnoresDeviceRows() {
        let out = "/dev/disk9\tGUID_partition_scheme\t\n/dev/disk9s1\tApple_APFS\t/Users/x/.mlx-serve/sandbox/images\n"
        XCTAssertEqual(SandboxVolume.parseAttach(out), "/Users/x/.mlx-serve/sandbox/images")
        XCTAssertNil(SandboxVolume.parseAttach("/dev/disk9\tGUID_partition_scheme\t\n"))
        XCTAssertNil(SandboxVolume.parseAttach(""))
    }

    /// The real probe: `/` is a mountpoint, a fresh temp dir is not.
    func testIsMountpointReadsStatfs() {
        XCTAssertTrue(SandboxVolume.isMountpoint("/"))
        XCTAssertFalse(SandboxVolume.isMountpoint(root.path))
        XCTAssertFalse(SandboxVolume.isMountpoint(root.appendingPathComponent("missing").path))
    }

    /// The case probe on the temp volume reports what the volume is (the
    /// dev Mac's is case-insensitive; a CI runner may differ) — and cleans up.
    func testCaseProbeLeavesNoFileBehind() {
        _ = SandboxVolume.isCaseSensitive(root)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    func testDetachOnlyRunsForAMountedVolume() {
        let fake = Fake()
        XCTAssertTrue(SandboxVolume.detach(root: root, tools: fake.tools()))
        XCTAssertTrue(fake.calls.isEmpty, "nothing mounted → nothing to do")
        let mp = root.appendingPathComponent("images", isDirectory: true).path
        fake.mounted.insert(mp)
        XCTAssertTrue(SandboxVolume.detach(root: root, tools: fake.tools()))
        XCTAssertEqual(fake.calls.last?.prefix(2).map { $0 }, ["detach", mp])
        XCTAssertFalse(fake.mounted.contains(mp))
    }

    /// The boot attaches BEFORE the rootfs lookup, skipping a dev override;
    /// no other site mounts.
    func testTheBootAttachesBeforeTheRootfsLookup() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Services/AgentSandbox.swift")
        let text = try String(contentsOf: url, encoding: .utf8)
        let attach = try XCTUnwrap(text.range(of: "SandboxVolume.ensureAttached(root: cacheDir)"))
        let lookup = try XCTUnwrap(text.range(of: "let rootfs = try provisioner.rootfsDir(image: image)"))
        XCTAssertLessThan(attach.lowerBound, lookup.lowerBound, "attach first, then look the rootfs up on it")
        XCTAssertTrue(text.contains("if Self.rootfsOverride() == nil {"), "a dev rootfs never mounts the volume")
    }
}
