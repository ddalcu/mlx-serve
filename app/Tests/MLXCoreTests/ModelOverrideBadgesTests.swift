import XCTest
@testable import MLXCore

/// My Models' Settings column: what a model carries that differs from the default.
final class ModelOverrideBadgesTests: XCTestCase {

    private let none = SocketBadge(stone: nil, skull: false)

    private func badges(_ o: ModelOverride, type: String = "qwen3", engines: [String]? = nil,
                        socket: SocketBadge? = nil) -> [OverrideBadge] {
        ModelOverrideBadges.badges(o, modelType: type, engines: engines, socket: socket ?? none)
    }

    func testAnUntouchedModelHasNoBadges() {
        XCTAssertEqual(badges(ModelOverride()), [])
    }

    func testTheEngineIsShownOnlyWhereTheServerSaysTheModelCanRunThere() {
        var o = ModelOverride()
        o.ane = true
        XCTAssertEqual(badges(o, engines: ["mlx", "ane"]).map(\.text), ["ANE"])
        o.ane = false
        XCTAssertEqual(badges(o, engines: ["mlx", "ane"]).map(\.text), ["MLX"], "an explicit GPU choice is worth showing")
        o.engineChoice = "ane"
        XCTAssertEqual(badges(o, engines: ["mlx"]), [], "a stored value on a model the engine cannot take is not in effect")
        XCTAssertEqual(badges(o).map(\.text), ["ANE"], "with no server row the stored choice shows")
        o.engineChoice = "llama"
        XCTAssertEqual(badges(o, engines: ["llama", "ds4"]).map(\.text), ["llama.cpp"])
        o.engineChoice = nil
        XCTAssertEqual(badges(o, engines: ["mlx", "ane"]), [])
    }

    func testInt8PrefillIsLossyAndOnlyOnPrismPacks() {
        var o = ModelOverride()
        o.int8Prefill = true
        let b = badges(o, type: "prism_hadamard_qwen35")
        XCTAssertEqual(b.map(\.text), ["Int8"])
        XCTAssertTrue(b[0].lossy)
        XCTAssertEqual(badges(o, type: "qwen3"), [])
    }

    func testKvCacheAndContextShareOneBadge() {
        var o = ModelOverride()
        o.kvQuant = .bits4
        o.ctxSize = 32768
        XCTAssertEqual(badges(o).map(\.text), ["KV 4-bit · 32k"])
        o.ctxSize = nil
        XCTAssertEqual(badges(o).map(\.text), ["KV 4-bit"])
        o.kvQuant = .off
        XCTAssertEqual(badges(o).map(\.text), ["KV off"])
        o.kvQuant = nil
        o.ctxSize = 100_000
        XCTAssertEqual(badges(o).map(\.text), ["100k ctx"])
        o.ctxSize = 100_123
        XCTAssertEqual(badges(o).map(\.text), ["100123 ctx"], "a size that is no round k is not rounded to one")
    }

    func testSamplingAndKwargsNameTheirContentsInTheTooltip() {
        var o = ModelOverride()
        o.generationDefaults = GenerationDefaults(rules: ["temperature": .init(value: .number(0.3))])
        o.templateKwargs = ["enable_thinking": false]
        let b = badges(o)
        XCTAssertEqual(b.map(\.kind), [.sampling, .kwargs])
        XCTAssertEqual(b.map(\.text), ["Sampling", "Kwargs"])
        XCTAssertTrue(b[0].help.contains("temperature"))
        XCTAssertTrue(b[1].help.contains("enable_thinking"))
    }

    func testSpeculationShowsTheStoneTheModelWillRun() {
        let lossy = badges(ModelOverride(), socket: SocketBadge(stone: .emerald, skull: true))
        XCTAssertEqual(lossy.map(\.text), ["+MTP"])
        XCTAssertEqual(lossy[0].gem, .emerald)
        XCTAssertTrue(lossy[0].lossy)

        let plain = badges(ModelOverride(), socket: SocketBadge(stone: .sapphire, skull: false))
        XCTAssertFalse(plain[0].lossy)
    }

    func testSpeculationWithoutAStoneNamesWhatWasSet() {
        var o = ModelOverride()
        o.drafter = "off"
        o.mtp = false
        XCTAssertEqual(badges(o).map(\.text), ["Spec off"])
        var accept = ModelOverride()
        accept.mtpAcceptance = .typical
        let b = badges(accept)
        XCTAssertEqual(b.map(\.text), ["Spec"])
        XCTAssertTrue(b[0].lossy)
        XCTAssertTrue(b[0].help.contains("typical"))
    }

    func testBadgesRunFromWhereTheModelRunsDownToThePromptKnobs() {
        var o = ModelOverride()
        o.ane = true
        o.kvQuant = .bits8
        o.mtp = true
        o.generationDefaults = GenerationDefaults(rules: ["top_p": .init(value: .number(0.9))])
        o.templateKwargs = ["x": 1]
        XCTAssertEqual(badges(o, type: "qwen3_5").map(\.kind), [.engine, .speculation, .kvCache, .sampling, .kwargs])
    }

    func testEveryTooltipOpensWithItsCategory() {
        var o = ModelOverride()
        o.ane = true
        o.kvQuant = .bits8
        let help = badges(o, type: "qwen3_5").map(\.help)
        XCTAssertTrue(help[0].hasPrefix("Engine"))
        XCTAssertTrue(help[1].hasPrefix("KV cache"))
    }

    func testOverflowKeepsTheFirstTwoAndCountsTheRest() {
        var o = ModelOverride()
        o.ane = true
        o.kvQuant = .bits8
        o.generationDefaults = GenerationDefaults(rules: ["top_p": .init(value: .number(0.9))])
        o.templateKwargs = ["x": 1]
        let all = badges(o, type: "qwen3_5")
        let split = ModelOverrideBadges.overflow(all)
        XCTAssertEqual(split.shown.map(\.kind), [.engine, .kvCache])
        XCTAssertEqual(split.hidden.map(\.kind), [.sampling, .kwargs])
        XCTAssertTrue(ModelOverrideBadges.overflow(Array(all.prefix(2))).hidden.isEmpty)
    }

    // MARK: - Column

    func testTheSettingsColumnYieldsToTheModelNameInANarrowPane() {
        XCTAssertTrue(ModelBrowserMetrics.showsLocalSettings(forDetailWidth: 0), "unmeasured reads as roomy")
        XCTAssertTrue(ModelBrowserMetrics.showsLocalSettings(forDetailWidth: 1000))
        XCTAssertFalse(ModelBrowserMetrics.showsLocalSettings(forDetailWidth: ModelBrowserMetrics.minDetailWidth))
        let edge = 24 + ModelBrowserMetrics.localSizeWidth + ModelBrowserMetrics.localActionsWidth
            + ModelBrowserMetrics.localSettingsWidth + 24 + ModelBrowserMetrics.modelColumnMinWidth
        XCTAssertTrue(ModelBrowserMetrics.showsLocalSettings(forDetailWidth: edge))
        XCTAssertFalse(ModelBrowserMetrics.showsLocalSettings(forDetailWidth: edge - 1))
    }

    // MARK: - Shelf Use

    func testAShelfRepoUsesItsOnlyBuildAndLeavesTheMenuToSeveral() {
        XCTAssertEqual(ModelBrowserUse.shelfPrimaryUse(onDisk: ["/m/q4"], selected: nil), "/m/q4")
        XCTAssertEqual(ModelBrowserUse.shelfPrimaryUse(onDisk: ["/m/q4"], selected: "/m/other"), "/m/q4")
        XCTAssertNil(ModelBrowserUse.shelfPrimaryUse(onDisk: ["/m/q4"], selected: "/m/q4/"), "already the model in use")
        XCTAssertNil(ModelBrowserUse.shelfPrimaryUse(onDisk: ["/m/q4", "/m/q8"], selected: nil), "two builds: which one?")
        XCTAssertNil(ModelBrowserUse.shelfPrimaryUse(onDisk: [], selected: nil))
    }
}
