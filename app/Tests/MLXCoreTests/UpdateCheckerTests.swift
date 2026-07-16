import XCTest
@testable import MLXCore

/// Pins the pure logic behind the in-app updater (Services/UpdateChecker.swift):
/// CalVer tag parsing/comparison, GitHub `releases/latest` JSON → `AppUpdate`
/// extraction (DMG asset pick, draft/prerelease skip), the daily auto-check
/// throttle, and the mounted-DMG app-bundle discovery used by the installer.
/// The network fetch, hdiutil mount, and bundle swap are exercised live; the
/// decisions they act on are all pinned here.
final class UpdateCheckerTests: XCTestCase {

    // MARK: - CalVer parsing

    func testParseCalVerAcceptsTaggedAndBareForms() {
        XCTAssertEqual(UpdateChecker.parseCalVer("v26.7.1"), [26, 7, 1])
        XCTAssertEqual(UpdateChecker.parseCalVer("26.7.1"), [26, 7, 1])
        XCTAssertEqual(UpdateChecker.parseCalVer("v26.12.30"), [26, 12, 30])
    }

    func testParseCalVerRejectsMalformed() {
        XCTAssertNil(UpdateChecker.parseCalVer(""))
        XCTAssertNil(UpdateChecker.parseCalVer("v"))
        XCTAssertNil(UpdateChecker.parseCalVer("latest"))
        XCTAssertNil(UpdateChecker.parseCalVer("v26.7.1-beta"))
        XCTAssertNil(UpdateChecker.parseCalVer("26..1"))
    }

    // MARK: - Version comparison

    func testIsNewerBasicOrdering() {
        XCTAssertTrue(UpdateChecker.isNewer("v26.7.2", than: "26.7.1"))
        XCTAssertTrue(UpdateChecker.isNewer("v27.1.1", than: "26.12.9"))
        XCTAssertFalse(UpdateChecker.isNewer("v26.7.1", than: "26.7.1"))
        XCTAssertFalse(UpdateChecker.isNewer("v26.7.1", than: "26.7.2"))
    }

    /// CalVer components must compare numerically — a lexicographic compare
    /// would sort v26.10.1 BELOW v26.9.9 and the updater would go silent for
    /// the last three months of every year.
    func testIsNewerComparesNumericallyNotLexically() {
        XCTAssertTrue(UpdateChecker.isNewer("v26.10.1", than: "26.9.9"))
        XCTAssertTrue(UpdateChecker.isNewer("v26.7.10", than: "26.7.9"))
    }

    /// Component-count mismatches zero-pad instead of failing: v26.8 == 26.8.0.
    func testIsNewerZeroPadsShorterVersions() {
        XCTAssertFalse(UpdateChecker.isNewer("v26.8", than: "26.8.0"))
        XCTAssertTrue(UpdateChecker.isNewer("v26.8.0.1", than: "26.8.0"))
    }

    /// A malformed tag (or local version) must never announce an update.
    func testIsNewerMalformedIsNeverNewer() {
        XCTAssertFalse(UpdateChecker.isNewer("latest", than: "26.7.1"))
        XCTAssertFalse(UpdateChecker.isNewer("v26.8.0", than: "dev"))
    }

    // MARK: - Bundled llama.cpp tag (surfaced in Settings ▸ Updates)

    /// The pinned llama.cpp release must mirror `LLAMA_TAG` in
    /// `scripts/fetch-llama.sh` (the source of truth). This pins the shape
    /// (`b<digits>`) so an accidental blanking/typo is caught; keep the value
    /// itself in sync with the fetch script when bumping the engine.
    func testBundledLlamaTagIsAWellFormedRelease() {
        XCTAssertEqual(UpdateChecker.bundledLlamaTag, "b10034",
                       "keep in sync with LLAMA_TAG in scripts/fetch-llama.sh")
        XCTAssertTrue(UpdateChecker.bundledLlamaTag.hasPrefix("b"))
        XCTAssertTrue(UpdateChecker.bundledLlamaTag.dropFirst().allSatisfy(\.isNumber))
    }

    // MARK: - GitHub releases/latest JSON → AppUpdate

    private func releaseJSON(
        tag: String = "v26.9.0",
        draft: Bool = false,
        prerelease: Bool = false,
        assets: [(name: String, url: String)] = [
            ("mlx-serve-bin-macos-arm64.tar.gz",
             "https://github.com/ddalcu/mlx-serve/releases/download/v26.9.0/mlx-serve-bin-macos-arm64.tar.gz"),
            ("MLXCore.dmg",
             "https://github.com/ddalcu/mlx-serve/releases/download/v26.9.0/MLXCore.dmg"),
        ]
    ) -> Data {
        let assetJSON = assets
            .map { #"{"name":"\#($0.name)","browser_download_url":"\#($0.url)","size":1234}"# }
            .joined(separator: ",")
        return Data("""
        {
          "url": "https://api.github.com/repos/ddalcu/mlx-serve/releases/246",
          "html_url": "https://github.com/ddalcu/mlx-serve/releases/tag/\(tag)",
          "tag_name": "\(tag)",
          "name": "mlx-serve \(tag)",
          "draft": \(draft),
          "prerelease": \(prerelease),
          "body": "## \(tag) — headline\\n- bullet one",
          "assets": [\(assetJSON)]
        }
        """.utf8)
    }

    func testFindUpdateParsesLatestReleaseShape() throws {
        let update = try XCTUnwrap(
            UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(), currentVersion: "26.7.1"))
        XCTAssertEqual(update.version, "26.9.0")
        XCTAssertEqual(update.tagName, "v26.9.0")
        XCTAssertEqual(update.dmgURL.absoluteString,
                       "https://github.com/ddalcu/mlx-serve/releases/download/v26.9.0/MLXCore.dmg")
        XCTAssertEqual(update.releasePageURL?.absoluteString,
                       "https://github.com/ddalcu/mlx-serve/releases/tag/v26.9.0")
        XCTAssertTrue(update.releaseNotes.contains("headline"))
    }

    /// The release carries two assets (CLI tarball + app DMG) — the updater
    /// must install from the DMG, never the tarball.
    func testFindUpdatePicksTheDMGAssetNotTheTarball() throws {
        let update = try XCTUnwrap(
            UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(), currentVersion: "26.7.1"))
        XCTAssertTrue(update.dmgURL.lastPathComponent.hasSuffix(".dmg"))
    }

    /// If the exact asset name ever changes, any single .dmg asset still works.
    func testFindUpdateFallsBackToAnyDMGAsset() throws {
        let json = releaseJSON(assets: [
            ("mlx-serve-bin-macos-arm64.tar.gz", "https://example.com/x.tar.gz"),
            ("MLX-Core-renamed.dmg", "https://example.com/MLX-Core-renamed.dmg"),
        ])
        let update = try XCTUnwrap(
            UpdateChecker.findUpdate(inReleaseJSON: json, currentVersion: "26.7.1"))
        XCTAssertEqual(update.dmgURL.absoluteString, "https://example.com/MLX-Core-renamed.dmg")
    }

    func testFindUpdateNilWhenNotNewer() {
        // Same version, and a local DEV build ahead of the latest release
        // (build.sh pre-bumps Info.plist) — both must stay silent.
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(tag: "v26.9.0"),
                                              currentVersion: "26.9.0"))
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(tag: "v26.9.0"),
                                              currentVersion: "26.10.0"))
    }

    func testFindUpdateNilWithoutDMGAsset() {
        let json = releaseJSON(assets: [
            ("mlx-serve-bin-macos-arm64.tar.gz", "https://example.com/x.tar.gz"),
        ])
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: json, currentVersion: "26.7.1"))
    }

    func testFindUpdateNilForDraftOrPrerelease() {
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(draft: true),
                                              currentVersion: "26.7.1"))
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: releaseJSON(prerelease: true),
                                              currentVersion: "26.7.1"))
    }

    func testFindUpdateNilOnMalformedJSON() {
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: Data("not json".utf8),
                                              currentVersion: "26.7.1"))
        XCTAssertNil(UpdateChecker.findUpdate(inReleaseJSON: Data("{}".utf8),
                                              currentVersion: "26.7.1"))
    }

    // MARK: - Auto-check throttle

    func testShouldAutoCheckFirstRunAndAfterInterval() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        XCTAssertTrue(UpdateChecker.shouldAutoCheck(now: now, lastCheck: nil))
        XCTAssertFalse(UpdateChecker.shouldAutoCheck(
            now: now, lastCheck: now.addingTimeInterval(-3600)))
        XCTAssertTrue(UpdateChecker.shouldAutoCheck(
            now: now, lastCheck: now.addingTimeInterval(-25 * 3600)))
    }

    /// A clock that moved BACKWARD past the last check (NTP fix, timezone
    /// mishap) must not wedge the throttle shut for days.
    func testShouldAutoCheckRecoversFromFutureLastCheck() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        XCTAssertTrue(UpdateChecker.shouldAutoCheck(
            now: now, lastCheck: now.addingTimeInterval(48 * 3600)))
    }

    // MARK: - Installer helpers

    private func makeTempDir() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("updater-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func testFindAppBundleLocatesTheAppInsideAMountedDMG() throws {
        let mount = try makeTempDir()
        // DMGs also contain the /Applications symlink + hidden metadata; the
        // finder must return the one real .app bundle.
        try FileManager.default.createDirectory(
            at: mount.appendingPathComponent("MLX Core.app"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: mount.appendingPathComponent("Applications"),
            withDestinationURL: URL(fileURLWithPath: "/Applications"))
        let found = UpdateChecker.findAppBundle(inMountedDMG: mount)
        XCTAssertEqual(found?.lastPathComponent, "MLX Core.app")
    }

    func testFindAppBundleNilWhenNoAppPresent() throws {
        let mount = try makeTempDir()
        XCTAssertNil(UpdateChecker.findAppBundle(inMountedDMG: mount))
    }

    /// Replaceable = a real .app bundle in a writable parent (the installed
    /// /Applications case). A `swift test`/xctest host or a dev binary under
    /// .build must return nil so the installer falls back to opening the DMG.
    func testInstallTargetRequiresARealAppBundle() throws {
        let dir = try makeTempDir()
        let app = dir.appendingPathComponent("MLX Core.app")
        try FileManager.default.createDirectory(at: app, withIntermediateDirectories: true)
        XCTAssertEqual(UpdateChecker.installTarget(for: app), app)
        XCTAssertNil(UpdateChecker.installTarget(
            for: dir.appendingPathComponent("not-a-bundle")))
    }
}
