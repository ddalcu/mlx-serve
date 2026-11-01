import XCTest
@testable import MLXCore

/// Models ▸ Plugins as a list and a detail: each plugin once, its state, its packs, a selection that stays valid.
final class PluginListingTests: XCTestCase {
    private let sushi = PluginCatalog.Entry(name: "sushi_ds41", kind: "model", version: "0.2.0", summary: "DeepSeek STQ packs.",
                                            models: ["deepseek_v41 with expert_quant.format stq (beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw)"],
                                            requiresMlxServe: ">=26.11.1")
    private let turbo = PluginCatalog.Entry(name: "turboexl3_qwen38", kind: "model", version: "0.1.0",
                                            models: ["qwen4_exp with a trunk_quant block", "bare text (not a repo)"],
                                            requiresMlxServe: ">=26.12.1")

    private func receipt(_ name: String, _ version: String) -> OutsidePlugins.Receipt {
        OutsidePlugins.Receipt(name: name, kind: "model", version: version, sha256: "abc")
    }

    func testAnInstalledPluginIsListedOnceWithItsCatalogEntry() {
        let lists = PluginListing.items(receipts: [receipt("sushi_ds41", "0.1.0"), receipt("local_only", "1.0")], catalog: [sushi, turbo])
        XCTAssertEqual(lists.installed.map(\.name), ["sushi_ds41", "local_only"])
        XCTAssertEqual(lists.installed[0].entry, sushi)
        XCTAssertNil(lists.installed[1].entry, "a plugin the catalog does not list is still shown")
        XCTAssertEqual(lists.available.map(\.name), ["turboexl3_qwen38"])
    }

    func testStateReadsTheSwitchesTheCatalogAndTheRunningServer() {
        let lists = PluginListing.items(receipts: [receipt("sushi_ds41", "0.1.0")], catalog: [sushi, turbo])
        var on = OutsidePlugins.Switches(enabled: true)
        XCTAssertEqual(PluginListing.state(lists.installed[0], switches: on, running: "26.11.1"), .updateAvailable("0.2.0"))
        let current = PluginListing.items(receipts: [receipt("sushi_ds41", "0.2.0")], catalog: [sushi]).installed[0]
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.1"), .on)
        on.set("sushi_ds41", on: false)
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.1"), .off)
        XCTAssertEqual(PluginListing.state(current, switches: OutsidePlugins.Switches(enabled: false), running: "26.11.1"), .off,
                       "loading turned off for all plugins: none of them runs")
        XCTAssertEqual(PluginListing.state(lists.available[0], switches: on, running: "26.11.1"), .needsNewer(">=26.12.1"))
        XCTAssertEqual(PluginListing.state(lists.available[0], switches: on, running: "26.12.1"), .available)
    }

    func testPacksSplitTheDescriptionFromTheRepo() {
        XCTAssertEqual(PluginListing.packs(sushi), [.init(text: "deepseek_v41 with expert_quant.format stq",
                                                         repo: "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw")])
        XCTAssertEqual(PluginListing.packs(turbo).map(\.repo), [nil, nil], "only an org/name in the trailing parentheses is a repo")
        XCTAssertEqual(PluginListing.packs(turbo).last?.text, "bare text (not a repo)")
    }

    func testSelectionStaysOnAListedPluginAndFallsBackToTheFirst() {
        let lists = PluginListing.items(receipts: [receipt("sushi_ds41", "0.2.0")], catalog: [sushi, turbo])
        XCTAssertEqual(PluginListing.resolve(.plugin("turboexl3_qwen38"), lists), .plugin("turboexl3_qwen38"))
        XCTAssertEqual(PluginListing.resolve(.plugin("gone"), lists), .plugin("sushi_ds41"))
        XCTAssertEqual(PluginListing.resolve(nil, lists), .plugin("sushi_ds41"))
        XCTAssertEqual(PluginListing.resolve(.agent, lists), .agent)
        XCTAssertEqual(PluginListing.resolve(nil, (installed: [], available: [])), .agent, "with nothing listed the agent page is what is left")
    }
}
