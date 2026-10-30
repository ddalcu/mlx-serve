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
        XCTAssertTrue(t.contains("scripts/install.sh toy_plugin --yes"))
        XCTAssertTrue(t.contains("plugins/toy_plugin/plugin.json"))
        XCTAssertTrue(t.contains("never download or run a prebuilt dylib"))
        XCTAssertTrue(t.contains(PluginCatalog.repoURL))
        XCTAssertTrue(t.contains(">=26.11.1") && t.contains("26.11.2"))
        XCTAssertTrue(t.contains("/Applications/MLX Core.app/Contents/MacOS"))
        XCTAssertFalse(t.contains("<name>"))
    }

    func testGenericInstructionsLeaveTheNameForTheAgent() {
        let t = PluginAgentInstructions.generic(binaryPath: "mlx-serve", runningVersion: nil)
        XCTAssertTrue(t.contains("scripts/install.sh <name> --yes"))
        XCTAssertTrue(t.contains("unknown"))
    }
}
