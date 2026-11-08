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

    func testAPluginTheServerRefusedForAnotherBuildNeedsARebuild() {
        let current = PluginListing.items(receipts: [receipt("sushi_ds41", "0.2.0")], catalog: [sushi]).installed[0]
        let on = OutsidePlugins.Switches(enabled: true)
        let stale = ["sushi_ds41": PluginCatalog.ServerStatus(loaded: false, rebuild: true, why: "rebuild it")]
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: stale), .needsRebuild)
        let changed = ["sushi_ds41": PluginCatalog.ServerStatus(loaded: false, rebuild: false, why: "hash differs")]
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: changed), .on,
                       "a refusal a rebuild does not fix is not offered one")
        let older = PluginListing.items(receipts: [receipt("sushi_ds41", "0.1.0")], catalog: [sushi]).installed[0]
        XCTAssertEqual(PluginListing.state(older, switches: on, running: "26.11.2", server: stale), .updateAvailable("0.2.0"),
                       "an update rebuilds it too, and says more")
        XCTAssertEqual(PluginListing.rebuildable(PluginListing.items(receipts: [receipt("sushi_ds41", "0.2.0"), receipt("local_only", "1.0")],
                                                                     catalog: [sushi]).installed,
                                                 server: stale.merging(["local_only": stale["sushi_ds41"]!]) { a, _ in a }).map(\.name),
                       ["sushi_ds41"], "Rebuild all takes only what the catalog can build")
    }

    func testARunningServerThatHasNotAppliedTheSwitchNeedsARestart() {
        let current = PluginListing.items(receipts: [receipt("sushi_ds41", "0.2.0")], catalog: [sushi]).installed[0]
        let on = OutsidePlugins.Switches(enabled: true)
        let loaded = ["sushi_ds41": PluginCatalog.ServerStatus(loaded: true, rebuild: false, why: "")]
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: [:], serverUp: true), .restartNeeded,
                       "just installed: the running server loads plugins only at its start")
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: loaded, serverUp: true), .on)
        XCTAssertEqual(PluginListing.state(current, switches: .init(enabled: true, disabled: ["sushi_ds41"]), running: "26.11.2",
                                           server: loaded, serverUp: true), .restartNeeded, "switched off but still loaded")
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: [:], serverUp: false), .on,
                       "a stopped server loads it at its next start: nothing to restart")
    }

    func testAnOptionTheRunningServerHasNotAppliedNeedsARestart() {
        var r = receipt("sushi_ds41", "0.2.0")
        r.options = [.init(name: "SUSHI_DS41_FRAC", kind: .number)]
        let current = PluginListing.items(receipts: [r], catalog: [sushi]).installed[0]
        var on = OutsidePlugins.Switches(enabled: true)
        on.setOption("sushi_ds41", "SUSHI_DS41_FRAC", "0.4")
        let stale = ["sushi_ds41": PluginCatalog.ServerStatus(loaded: true, rebuild: false, why: "")]
        let applied = ["sushi_ds41": PluginCatalog.ServerStatus(loaded: true, rebuild: false, why: "", options: ["SUSHI_DS41_FRAC": "0.4"])]
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: stale, serverUp: true), .restartNeeded)
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: applied, serverUp: true), .on)
        XCTAssertEqual(PluginListing.state(current, switches: on, running: "26.11.2", server: stale, serverUp: false), .on,
                       "a stopped server sets it at its next start")
    }

    func testInstallProgressReadsAsThreeSteps() {
        typealias S = PluginListing.StepStatus
        func statuses(_ phase: PluginManager.Phase) -> [S] { PluginListing.installSteps(phase).map(\.status) }
        XCTAssertEqual(PluginListing.installSteps(.fetching(name: "x")).count, 3)
        XCTAssertEqual(statuses(.fetching(name: "x")), [.active, .waiting, .waiting])
        XCTAssertEqual(statuses(.awaitingConfirmation(name: "x", commit: String(repeating: "a", count: 40), version: "0.2.0")),
                       [.done, .waiting, .waiting], "nothing compiles until the user approves the fetched commit")
        XCTAssertEqual(statuses(.building(name: "x")), [.done, .active, .waiting])
        XCTAssertEqual(statuses(.installing(name: "x")), [.done, .done, .active])
        XCTAssertEqual(statuses(.done(name: "x")), [.done, .done, .done])
    }

    func testARowReadsByItsCatalogTitleAndFallsBackToItsName() throws {
        let json = #"{"plugins":[{"name":"sushi_ds41","title":"Sushi - DeepSeek 4.1 Flash","kind":"model"},{"name":"bare","kind":"model","title":"  "}]}"#
        let catalog = try PluginCatalog.decode(Data(json.utf8))
        let lists = PluginListing.items(receipts: [receipt("local_only", "1.0")], catalog: catalog)
        XCTAssertEqual(lists.available.map(\.title), ["Sushi - DeepSeek 4.1 Flash", "bare"], "a blank title is no title")
        XCTAssertEqual(lists.installed[0].title, "local_only", "a plugin the catalog does not list reads by its name")
    }

    func testPacksSplitTheDescriptionFromTheRepo() {
        XCTAssertEqual(PluginListing.packs(sushi), [.init(text: "deepseek_v41 with expert_quant.format stq",
                                                         repo: "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw")])
        XCTAssertEqual(PluginListing.packs(turbo).map(\.repo), [nil, nil], "only an org/name in the trailing parentheses is a repo")
        XCTAssertEqual(PluginListing.packs(turbo).last?.text, "bare text (not a repo)")
    }

    func testPacksSharingADescriptionStillGetOneIdEach() {
        let exl3 = PluginCatalog.Entry(name: "turboexl3_qwen38", kind: "model", version: "0.1.0", models: [
            "qwen4_exp with a trunk_quant block (OpensourceWTF/Qwen3.8-Flash-Next-EXL3-2.05-mlx-serve)",
            "qwen4_exp with a trunk_quant block (OpensourceWTF/Qwen3.8-Flash-Next-EXL3-3.05-mlx-serve)"])
        XCTAssertEqual(Set(PluginListing.packs(exl3).map(\.id)).count, 2, "a list keyed on the description draws the first pack for every row")
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
