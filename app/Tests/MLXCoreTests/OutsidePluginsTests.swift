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

    func testReceiptOptionsAreReadAndAnUnknownKindDropsOnlyTheOptions() {
        write("toy.dylib", "x")
        write("toy.json", #"{"name":"toy","kind":"engine","sha256":"ab","options":[{"name":"TOY_REPLY","kind":"text","default":"hi","help":"What it says"},{"name":"TOY_FAST","kind":"bool"}]}"#)
        write("odd.dylib", "x")
        write("odd.json", #"{"name":"odd","kind":"model","sha256":"cd","options":[{"name":"ODD_X","kind":"color"}]}"#)
        let got = OutsidePlugins.receipts(inDir: dir)
        XCTAssertEqual(got.map(\.name), ["odd", "toy"])
        XCTAssertEqual(got[1].options, [.init(name: "TOY_REPLY", kind: .text, defaultValue: "hi", help: "What it says"),
                                        .init(name: "TOY_FAST", kind: .bool)])
        XCTAssertEqual(got[0].options, [], "a kind this app does not know drops the options, not the plugin")
    }

    func testOptionValuesRoundTripBesideTheSwitches() throws {
        let path = (dir as NSString).appendingPathComponent("plugins.json")
        write("plugins.json", #"{"enabled":true,"later":1}"#)
        var s = OutsidePlugins.loadSwitches(path: path)
        s.setOption("toy", "TOY_REPLY", "hello")
        s.setOption("toy", "TOY_FAST", "1")
        try OutsidePlugins.save(s, path: path)
        XCTAssertEqual(OutsidePlugins.loadSwitches(path: path).options, ["toy": ["TOY_REPLY": "hello", "TOY_FAST": "1"]])
        s.setOption("toy", "TOY_REPLY", nil)
        s.setOption("toy", "TOY_FAST", nil)
        XCTAssertEqual(s.options, [:], "clearing the last value forgets the plugin")
        try OutsidePlugins.save(s, path: path)
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any]
        XCTAssertEqual(raw?["later"] as? Int, 1)
        XCTAssertNil(raw?["options"])
        write("plugins.json", #"{"options":{"toy":{"TOY_FRAC":0.4,"TOY_REPLY":"hi"},"odd":3}}"#)
        XCTAssertEqual(OutsidePlugins.loadSwitches(path: path).options, ["toy": ["TOY_REPLY": "hi"]],
                       "a value that is not a string is skipped, not every plugin's values")
    }

    func testOnlyADeclaredValueItsKindAcceptsIsApplied() {
        let r = OutsidePlugins.Receipt(name: "toy", kind: "engine", sha256: "ab",
                                       options: [.init(name: "TOY_FAST", kind: .bool), .init(name: "TOY_FRAC", kind: .number),
                                                 .init(name: "TOY_REPLY", kind: .text)])
        var s = OutsidePlugins.Switches(enabled: true)
        s.setOption("toy", "TOY_FAST", "yes")
        s.setOption("toy", "TOY_FRAC", "0.4")
        s.setOption("toy", "TOY_REPLY", "hi there")
        s.setOption("toy", "OTHER", "1")
        XCTAssertEqual(OutsidePlugins.applied(r, s), ["TOY_FRAC": "0.4", "TOY_REPLY": "hi there"])
        let number = OutsidePlugins.Option(name: "N", kind: .number)
        XCTAssertEqual(["1", "-2.5", "1e3", "0.25E-2"].filter(number.accepts).count, 4)
        XCTAssertEqual(["", "x", "1.", ".5", "nan", "inf", "1_000", "0x10", "0.5\n"].filter(number.accepts), [])
        let flag = OutsidePlugins.Option(name: "B", kind: .bool)
        XCTAssertEqual(["0", "1", "true", "", "2"].filter(flag.accepts), ["0", "1"])
        XCTAssertFalse(OutsidePlugins.Option(name: "T", kind: .text).accepts(String(repeating: "x", count: 1025)))
    }

    func testTurningAPluginOnAlsoTurnsOnLoading() {
        let s = OutsidePlugins.Switches(enabled: false, disabled: ["toy", "snd"]).turningOn(["toy"], installed: ["toy", "snd"])
        XCTAssertTrue(s.enabled, "a plugin switched on with loading off would still not run")
        XCTAssertEqual(s.disabled, ["snd"], "only the named plugins change")
    }

    func testTurningLoadingOnLeavesEveryOtherInstalledPluginOff() {
        let s = OutsidePlugins.Switches(enabled: false, disabled: []).turningOn(["b"], installed: ["a", "b"])
        XCTAssertTrue(s.isOn("b"))
        XCTAssertFalse(s.isOn("a"), "a plugin the user never chose must not load because loading came on for another")
    }

    func testTurningOnWithLoadingAlreadyOnChangesOnlyTheNamedPlugins() {
        let s = OutsidePlugins.Switches(enabled: true, disabled: ["c"]).turningOn(["b"], installed: ["a", "b", "c"])
        XCTAssertEqual(s.disabled, ["c"])
        XCTAssertTrue(s.isOn("a"), "with loading on, a's own switch was already the user's choice")
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
