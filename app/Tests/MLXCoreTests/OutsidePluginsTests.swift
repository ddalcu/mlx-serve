import XCTest
@testable import MLXCore

/// The Settings view of outside plugins reads the receipts `mlx-serve plugin install` writes and the plugins.json
/// the server reads at startup (src/host/plugins/loader.zig): same file, same defaults.
final class OutsidePluginsTests: XCTestCase {
    private var dir: String!

    override func setUpWithError() throws {
        dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("outside-plugins-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: dir)
    }

    private func write(_ name: String, _ text: String) {
        FileManager.default.createFile(atPath: (dir as NSString).appendingPathComponent(name), contents: Data(text.utf8))
    }

    func testReceiptsListInstalledPluginsByNameAndSkipStrays() {
        write("toy.dylib", "x")
        write("toy.json", #"{"name":"toy","kind":"model","version":"0.1.0","sha256":"ab12","signature":"ad-hoc","source":"/x/libtoy.dylib"}"#)
        write("snd.dylib", "x")
        write("snd.json", #"{"name":"snd","kind":"media","sha256":"cd34"}"#)
        // A receipt whose dylib is gone, and a dylib with no receipt, are not installed plugins.
        write("gone.json", #"{"name":"gone","kind":"model","sha256":"00"}"#)
        write("bare.dylib", "x")
        write("broken.dylib", "x")
        write("broken.json", "{")
        let got = OutsidePlugins.receipts(inDir: dir)
        XCTAssertEqual(got.map(\.name), ["snd", "toy"])
        XCTAssertEqual(got[1], .init(name: "toy", kind: "model", version: "0.1.0", sha256: "ab12", signature: "ad-hoc", source: "/x/libtoy.dylib"))
        XCTAssertEqual(got[0].version, "", "an older receipt without a version still lists")
    }

    func testSwitchesDefaultOffAndRoundTripKeepingOtherKeys() throws {
        let path = (dir as NSString).appendingPathComponent("plugins.json")
        XCTAssertEqual(OutsidePlugins.loadSwitches(path: path), .init())
        write("plugins.json", "{")
        XCTAssertFalse(OutsidePlugins.loadSwitches(path: path).enabled, "a broken file reads as off, as the server reads it")
        write("plugins.json", #"{"enabled":false,"later":1}"#)
        var s = OutsidePlugins.loadSwitches(path: path)
        s.enabled = true
        s.set("toy", on: false)
        try OutsidePlugins.save(s, path: path)
        XCTAssertEqual(OutsidePlugins.loadSwitches(path: path), .init(enabled: true, disabled: ["toy"]))
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any]
        XCTAssertEqual(raw?["later"] as? Int, 1)
        s.set("toy", on: true)
        XCTAssertTrue(s.isOn("toy"))
        XCTAssertEqual(s.disabled, [])
    }

    func testServedTypesFollowTheSwitchesAndTheKind() {
        let receipts = [
            OutsidePlugins.Receipt(name: "toy", kind: "model", sha256: "a", modelTypes: ["toy_lm"]),
            OutsidePlugins.Receipt(name: "snd", kind: "media", sha256: "b", modelTypes: ["toy_sound"]),
            OutsidePlugins.Receipt(name: "fmt", kind: "quant", sha256: "c"),
        ]
        XCTAssertEqual(OutsidePlugins.Served(receipts: receipts, switches: .init()), OutsidePlugins.Served(receipts: [], switches: .init()),
                       "plugins switched off serve nothing")
        let on = OutsidePlugins.Served(receipts: receipts, switches: .init(enabled: true, disabled: ["snd"]))
        XCTAssertEqual(on.chat, ["toy_lm"])
        XCTAssertEqual(on.media, [])
        let all = OutsidePlugins.Served(receipts: receipts, switches: .init(enabled: true))
        XCTAssertEqual(all.media, ["toy_sound"])
    }
}
