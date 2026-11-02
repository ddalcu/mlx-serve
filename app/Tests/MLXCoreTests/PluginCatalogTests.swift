import XCTest
@testable import MLXCore

/// The plugin catalog is untrusted input and drives a build script: names are plain identifiers, versions
/// compare numerically, and install state comes from the receipts.
final class PluginCatalogTests: XCTestCase {
    func testDecodeDropsEntriesWithUnsafeNames() throws {
        let json = #"""
        {"kit":"1","plugins":[
          {"name":"good_one","kind":"model","version":"0.1.0","summary":"s","models":["m"],"requires_mlx_serve":">=26.11.1"},
          {"name":"../evil","kind":"model","version":"1"},
          {"name":"a b; rm -rf ~","kind":"model","version":"1"},
          {"name":"","kind":"model","version":"1"}]}
        """#
        let got = try PluginCatalog.decode(Data(json.utf8))
        XCTAssertEqual(got.map(\.name), ["good_one"])
        XCTAssertEqual(got[0].requiresMlxServe, ">=26.11.1")
    }

    func testVersionsCompareNumerically() {
        XCTAssertEqual(PluginCatalog.compare("26.9", "26.11.1"), .orderedAscending)
        XCTAssertEqual(PluginCatalog.compare("1.0", "1.0.0"), .orderedSame)
        XCTAssertEqual(PluginCatalog.compare("0.2.0", "0.1.9"), .orderedDescending)
    }

    func testStatusFollowsReceipts() {
        let e = PluginCatalog.Entry(name: "toy", kind: "model", version: "0.2.0")
        func r(_ v: String) -> OutsidePlugins.Receipt { .init(name: "toy", kind: "model", version: v, sha256: "x") }
        XCTAssertEqual(PluginCatalog.status(of: e, receipts: []), .notInstalled)
        XCTAssertEqual(PluginCatalog.status(of: e, receipts: [r("0.2.0")]), .installed)
        XCTAssertEqual(PluginCatalog.status(of: e, receipts: [r("0.1.0")]), .updateAvailable(installed: "0.1.0"))
    }

    func testRequirementGate() {
        XCTAssertTrue(PluginCatalog.runs(on: "26.11.1", requirement: ">=26.11.1"))
        XCTAssertFalse(PluginCatalog.runs(on: "26.10.4", requirement: ">=26.11.1"))
        XCTAssertTrue(PluginCatalog.runs(on: nil, requirement: ">=26.11.1"), "unknown version: the server itself refuses a mismatch")
    }

    /// The install script would download the kit named by the plugins repo's KIT_VERSION from a release, which a
    /// build ahead of its release does not have: the kit in ~/.mlx-serve/kit is handed over when it is this server's.
    func testKitIsHandedOverOnlyWhenItMatchesTheRunningServer() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("kit-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try ".{\n    .name = .mlx_serve_plugin_kit,\n    .version = \"26.11.1\",\n}\n".write(to: dir.appendingPathComponent("build.zig.zon"), atomically: true, encoding: .utf8)
        try "".write(to: dir.appendingPathComponent("build.zig"), atomically: true, encoding: .utf8)
        XCTAssertEqual(PluginCatalog.kit(at: dir.path, matching: "26.11.1"), dir.path)
        XCTAssertNil(PluginCatalog.kit(at: dir.path, matching: "26.10.3"), "a kit for another server builds a plugin it refuses")
        XCTAssertNil(PluginCatalog.kit(at: dir.path, matching: nil))
        XCTAssertNil(PluginCatalog.kit(at: dir.appendingPathComponent("absent").path, matching: "26.11.1"))
    }

    /// A dev build prints `mlx-serve 26.11.1 (fast)`: the plugin rules (kit match, requirement, agent text) read the release.
    func testRunningVersionIsTheReleaseWithoutTheBuildFlavor() {
        let out = "[mem] MLX buffer-pool cap 8192 MB (was 249036 MB)\nmlx-serve 26.11.1 (fast)\nmlx 0.32.3\n"
        XCTAssertEqual(PluginCatalog.runningVersion(fromVersionOutput: out), "26.11.1")
        XCTAssertEqual(PluginCatalog.runningVersion(fromVersionOutput: "mlx-serve 26.10.3\n"), "26.10.3")
        XCTAssertNil(PluginCatalog.runningVersion(fromVersionOutput: "mlx 0.32.3\n"))
    }

    func testAMissingKitIsNamedInTheFailure() {
        let log = "[fetch-kit] cannot download https://github.com/ddalcu/mlx-serve/releases/download/v26.11.1/mlx-serve-plugin-kit-26.11.1.tar.gz\n"
        let why = PluginCatalog.installFailure(status: 1, log: log, running: "26.11.1")
        XCTAssertTrue(why.contains("~/.mlx-serve/kit") && why.contains("26.11.1"), why)
        XCTAssertEqual(PluginCatalog.installFailure(status: 2, log: "error: build failed", running: "26.11.1"),
                       "build or install failed (exit 2); see the log")
    }

    /// The repo's install.sh changed its arguments under the app once (`--yes` dropped, so the install asked on a stdin
    /// the app never connects): the app takes only build.sh from the repo and installs with its own CLI.
    func testInstallBuildsWithTheRepoScriptAndInstallsWithOurCLI() {
        let steps = PluginManager.installSteps(name: "sushi_ds41", sourceDir: "/src", binary: "/app/mlx-serve")
        XCTAssertEqual(steps.map(\.exe), ["/bin/bash", "/app/mlx-serve"])
        XCTAssertEqual(steps[0].args, ["/src/scripts/build.sh", "sushi_ds41"])
        XCTAssertEqual(steps[1].args, ["plugin", "install", "/src/plugins/sushi_ds41/zig-out/lib/libsushi_ds41.dylib", "--yes"])
    }

    func testUninstallRemovesDylibAndReceiptOnly() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("pc-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        for f in ["toy.dylib", "toy.json", "other.dylib"] { FileManager.default.createFile(atPath: dir + "/" + f, contents: Data()) }
        try PluginCatalog.uninstall("toy", inDir: dir)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dir), ["other.dylib"])
        XCTAssertThrowsError(try PluginCatalog.uninstall("../x", inDir: dir))
    }
}

final class PluginAgentInstructionsTests: XCTestCase {
    func testInstructionsNameThePluginAndKeepTheSafetyRules() {
        let e = PluginCatalog.Entry(name: "toy_plugin", kind: "model", version: "0.1.0", summary: "A toy.", models: ["toy_arch"], requiresMlxServe: ">=26.11.1")
        let t = PluginAgentInstructions.text(for: e, binaryPath: "/Applications/MLX Core.app/Contents/MacOS/mlx-serve", runningVersion: "26.11.2")
        XCTAssertTrue(t.contains("scripts/build.sh toy_plugin"))
        XCTAssertTrue(t.contains("mlx-serve plugin install plugins/toy_plugin/zig-out/lib/libtoy_plugin.dylib --yes"))
        XCTAssertTrue(t.contains("MLX_SERVE_KIT=~/.mlx-serve/kit"), "the fallback for a version with no released kit")
        XCTAssertFalse(t.contains("install.sh toy_plugin --yes"), "install.sh no longer takes --yes")
        XCTAssertTrue(t.contains("plugins/toy_plugin/plugin.json"))
        XCTAssertTrue(t.contains("never download or run a prebuilt dylib"))
        XCTAssertTrue(t.contains(PluginCatalog.repoURL))
        XCTAssertTrue(t.contains(">=26.11.1") && t.contains("26.11.2"))
        XCTAssertTrue(t.contains("/Applications/MLX Core.app/Contents/MacOS"))
        XCTAssertFalse(t.contains("<name>"))
    }

    func testGenericInstructionsLeaveTheNameForTheAgent() {
        let t = PluginAgentInstructions.generic(binaryPath: "mlx-serve", runningVersion: nil)
        XCTAssertTrue(t.contains("scripts/build.sh <name>"))
        XCTAssertTrue(t.contains("unknown"))
    }
}
