import XCTest
@testable import MLXCore

/// CLIInstaller puts an `mlx-serve` symlink on the user's PATH so the CLI
/// works from Terminal. Preference: an existing home bin dir that's already
/// on the shell PATH (no admin), else /usr/local/bin behind a one-time admin
/// prompt. It never edits shell dotfiles.
final class CLIInstallerTests: XCTestCase {

    private let home = "/Users/tester"

    // MARK: - Target selection (pure)

    func testSelectsLocalBinWhenPresentAndOnPath() {
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: ["/Users/tester/.local/bin"],
            pathEntries: ["/usr/bin", "/bin", "/Users/tester/.local/bin"]
        )
        XCTAssertEqual(target.directory, "/Users/tester/.local/bin")
        XCTAssertFalse(target.requiresAdmin)
    }

    func testPrefersLocalBinOverHomeBin() {
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: ["/Users/tester/.local/bin", "/Users/tester/bin"],
            pathEntries: ["/Users/tester/bin", "/Users/tester/.local/bin"]
        )
        XCTAssertEqual(target.directory, "/Users/tester/.local/bin")
    }

    func testFallsBackToHomeBinWhenLocalBinAbsent() {
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: ["/Users/tester/bin"],
            pathEntries: ["/usr/bin", "/Users/tester/bin"]
        )
        XCTAssertEqual(target.directory, "/Users/tester/bin")
        XCTAssertFalse(target.requiresAdmin)
    }

    func testHomeBinDirNotOnPathIsSkipped() {
        // ~/.local/bin exists but the user's shell PATH never picks it up —
        // a symlink there would be dead weight (and we refuse to edit
        // dotfiles), so fall through to the admin target.
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: ["/Users/tester/.local/bin"],
            pathEntries: ["/usr/bin", "/bin"]
        )
        XCTAssertEqual(target.directory, "/usr/local/bin")
        XCTAssertTrue(target.requiresAdmin)
    }

    func testNoHomeCandidatesFallsBackToAdmin() {
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: [],
            pathEntries: ["/usr/bin", "/Users/tester/.local/bin"]
        )
        XCTAssertEqual(target.directory, "/usr/local/bin")
        XCTAssertTrue(target.requiresAdmin)
    }

    func testTrailingSlashOnPathEntryStillMatches() {
        let target = CLIInstaller.selectTarget(
            home: home,
            existingDirs: ["/Users/tester/.local/bin"],
            pathEntries: ["/Users/tester/.local/bin/"]
        )
        XCTAssertEqual(target.directory, "/Users/tester/.local/bin")
        XCTAssertFalse(target.requiresAdmin)
    }

    // MARK: - Shell PATH parsing (pure)

    func testParsePathEntriesSplitsOnColonAndDropsEmpties() {
        XCTAssertEqual(
            CLIInstaller.parsePathEntries("/usr/bin:/bin::/Users/tester/.local/bin"),
            ["/usr/bin", "/bin", "/Users/tester/.local/bin"]
        )
    }

    func testExtractPathIgnoresRcFileNoiseAroundMarkers() {
        // Interactive shells can echo arbitrary junk from rc files; only the
        // marker-delimited segment is the PATH.
        let output = """
        Welcome banner from .zshrc!
        __MLX_PATH_BEGIN__/usr/bin:/Users/tester/.local/bin__MLX_PATH_END__
        trailing noise
        """
        XCTAssertEqual(
            CLIInstaller.extractPath(fromShellOutput: output),
            "/usr/bin:/Users/tester/.local/bin"
        )
    }

    func testExtractPathReturnsNilWithoutMarkers() {
        XCTAssertNil(CLIInstaller.extractPath(fromShellOutput: "no markers here"))
    }

    // MARK: - Admin command quoting (pure)

    func testAdminCommandSingleQuotesPathWithSpaces() {
        let cmd = CLIInstaller.adminInstallShellCommand(
            binarySource: "/Applications/MLX Core.app/Contents/MacOS/mlx-serve")
        XCTAssertTrue(cmd.contains("mkdir -p /usr/local/bin"))
        XCTAssertTrue(cmd.contains("ln -sf '/Applications/MLX Core.app/Contents/MacOS/mlx-serve' /usr/local/bin/mlx-serve"))
    }

    func testShellQuoteEscapesEmbeddedSingleQuote() {
        XCTAssertEqual(CLIInstaller.shellQuote("a'b"), "'a'\\''b'")
    }

    // MARK: - Symlink install + status (temp-dir integration)

    private func makeTempDir() throws -> String {
        let dir = NSTemporaryDirectory() + "cli-installer-test-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        return dir
    }

    func testInstallIsUnavailableInTheAppStoreBuild() throws {
        try XCTSkipIf(BuildFeatures.current.cliInstaller, "only meaningful in the App Store build")
        XCTAssertThrowsError(try CLIInstaller.installIntoHomeBin(directory: "/tmp/x", binarySource: "/tmp/y")) { error in
            XCTAssertTrue("\(error)".contains("unavailableInThisBuild"), "\(error)")
        }
    }

    func testInstallIntoHomeBinCreatesSymlink() throws {
        try XCTSkipUnless(BuildFeatures.current.cliInstaller, "CLI install is compiled out of the App Store build")
        let dir = try makeTempDir()
        let source = dir + "/fake-mlx-serve"
        FileManager.default.createFile(atPath: source, contents: Data("x".utf8))

        let link = try CLIInstaller.installIntoHomeBin(directory: dir + "/bin-noexist-yet",
                                                       binarySource: source)
        XCTAssertEqual(link, dir + "/bin-noexist-yet/mlx-serve")
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: link), source)
    }

    func testInstallReplacesStaleOrDanglingLink() throws {
        try XCTSkipUnless(BuildFeatures.current.cliInstaller, "CLI install is compiled out of the App Store build")
        let dir = try makeTempDir()
        let source = dir + "/fake-mlx-serve"
        FileManager.default.createFile(atPath: source, contents: Data("x".utf8))
        let binDir = dir + "/bin"
        try FileManager.default.createDirectory(atPath: binDir, withIntermediateDirectories: true)
        // Dangling link at the target name (points at a path that never existed).
        try FileManager.default.createSymbolicLink(atPath: binDir + "/mlx-serve",
                                                   withDestinationPath: dir + "/gone")

        let link = try CLIInstaller.installIntoHomeBin(directory: binDir, binarySource: source)
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: link), source)
    }

    func testStatusFindsLinkPointingAtOurBinary() throws {
        let dir = try makeTempDir()
        let fakeHome = dir + "/home"
        let source = dir + "/fake-mlx-serve"
        FileManager.default.createFile(atPath: source, contents: Data("x".utf8))
        let binDir = fakeHome + "/.local/bin"
        try FileManager.default.createDirectory(atPath: binDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(atPath: binDir + "/mlx-serve",
                                                   withDestinationPath: source)

        let status = CLIInstaller.status(binarySource: source, home: fakeHome)
        XCTAssertEqual(status, .installed(linkPath: binDir + "/mlx-serve"))
    }

    func testStatusIgnoresForeignLinkAndMissingLink() throws {
        let dir = try makeTempDir()
        let fakeHome = dir + "/home"
        let source = dir + "/fake-mlx-serve"
        let other = dir + "/other-binary"
        FileManager.default.createFile(atPath: source, contents: Data("x".utf8))
        FileManager.default.createFile(atPath: other, contents: Data("y".utf8))

        // No link anywhere → notInstalled.
        XCTAssertEqual(CLIInstaller.status(binarySource: source, home: fakeHome), .notInstalled)

        // A link named mlx-serve pointing somewhere else (e.g. Homebrew's) is
        // not ours → still notInstalled.
        let binDir = fakeHome + "/bin"
        try FileManager.default.createDirectory(atPath: binDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(atPath: binDir + "/mlx-serve",
                                                   withDestinationPath: other)
        XCTAssertEqual(CLIInstaller.status(binarySource: source, home: fakeHome), .notInstalled)
    }
}
