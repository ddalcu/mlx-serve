import XCTest
@testable import MLXCore

/// Pure logic behind the Benchmarks window.
///
/// The numbers this feature publishes are only worth anything if the
/// methodology holds, and every rule below is one that silently produces
/// plausible-but-wrong data when it's missing:
///
///  * A repeated prompt hits the server's KV prefix cache, so a coding run
///    that reused its prefix measured a cache lookup instead of prefill.
///  * The counting run is MEANT to hit the cache: it measures decode over the
///    same prefix, so the discard rule must not apply to it.
///  * A run measures the server as configured. The settings that shaped a
///    number ride the row, or two rows with different kv-quant share a median.
final class BenchmarkLogicTests: XCTestCase {

    // MARK: - Stats

    func testMedianUsesTheMiddleValueNotTheMean() {
        XCTAssertEqual(BenchmarkStats.median([50, 51, 20]), 50, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.median([10, 20]), 15, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.median([42]), 42, accuracy: 0.001)
    }

    func testMedianOfNothingIsZeroRatherThanACrash() {
        XCTAssertEqual(BenchmarkStats.median([]), 0, accuracy: 0.001)
    }

    func testSpreadIsRelativeToTheMedianSoItComparesAcrossModels() {
        XCTAssertEqual(BenchmarkStats.spreadPercent([48, 50, 52]), 8, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.spreadPercent([50, 50, 50]), 0, accuracy: 0.001)
        XCTAssertEqual(BenchmarkStats.spreadPercent([]), 0, accuracy: 0.001)
    }

    // MARK: - Cache contamination

    func testTheTemplateHeaderAlwaysMatchesSoCachedTokensIsNeverZero() {
        // The chat template's own header and the "[probe " lead-in are
        // identical across runs by construction, so a `cached > 0` rule
        // discards EVERY run. (Shipped exactly that way once.)
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 1522, cachedTokens: 7))
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 2048, cachedTokens: 40))
    }

    func testAWarmHitOnTheWholePromptIsStillDiscarded() {
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 1522, cachedTokens: 1500))
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 2048, cachedTokens: 2048))
    }

    func testTheThresholdIsAFractionOfThePromptNotAFixedCount() {
        XCTAssertFalse(BenchmarkPrompt.prefillWasReused(promptTokens: 1000, cachedTokens: 100))
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 1000, cachedTokens: 101))
    }

    func testADegenerateRunCountsAsUnusable() {
        XCTAssertTrue(BenchmarkPrompt.prefillWasReused(promptTokens: 0, cachedTokens: 0))
    }

    func testTheCacheDiscardAppliesToTheCodingRunAndNeverToTheCountingRun() {
        // The counting run rides the coding run's archive on purpose: its
        // prefix hit is what makes it a decode-only measurement. Discarding it
        // for the hit would leave every ceiling at 0.
        XCTAssertFalse(LadderSample.keep(kind: .coding, promptTokens: 4000, cachedTokens: 3900, completionTokens: 192))
        XCTAssertTrue(LadderSample.keep(kind: .coding, promptTokens: 4000, cachedTokens: 12, completionTokens: 192))
        XCTAssertTrue(LadderSample.keep(kind: .counting, promptTokens: 4000, cachedTokens: 3900, completionTokens: 192))
        XCTAssertTrue(LadderSample.keep(kind: .counting, promptTokens: 4000, cachedTokens: 4000, completionTokens: 192))
    }

    func testAOneTokenCountingAnswerIsNotACeiling() {
        // gemma-4-e2b answers the counting task over a code context with "1"
        // and stops (live, 2026-09-17): a one-token decode rate is noise, and
        // publishing it as the speculation ceiling would be a lie. Below the
        // floor the rung's ceiling stays 0, which the table draws as a dash.
        XCTAssertFalse(LadderSample.keep(kind: .counting, promptTokens: 4000, cachedTokens: 4000, completionTokens: 1))
        XCTAssertFalse(LadderSample.keep(kind: .counting, promptTokens: 4000, cachedTokens: 4000,
                                         completionTokens: LadderSample.minCeilingTokens - 1))
        XCTAssertTrue(LadderSample.keep(kind: .counting, promptTokens: 4000, cachedTokens: 4000,
                                        completionTokens: LadderSample.minCeilingTokens))
        // A short CODING answer is still a prefill measurement.
        XCTAssertTrue(LadderSample.keep(kind: .coding, promptTokens: 4000, cachedTokens: 12, completionTokens: 1))
    }

    // MARK: - Suite

    func testTheLadderIsSixRungsInAscendingOrder() {
        XCTAssertEqual(BenchmarkSuite.ladder.map(\.targetTokens), [512, 1024, 2048, 4096, 8192, 16384])
        XCTAssertEqual(BenchmarkSuite.ladder.map(\.id), ["ctx-v1-512", "ctx-v1-1k", "ctx-v1-2k", "ctx-v1-4k", "ctx-v1-8k", "ctx-v1-16k"])
        for rung in BenchmarkSuite.ladder {
            XCTAssertEqual(rung.genTokens, 192)
            XCTAssertEqual(rung.runs, 2)
            XCTAssertEqual(rung.warmups, 1)
        }
    }

    func testSuiteLookupKnowsOnlyTheLadder() {
        XCTAssertEqual(BenchmarkSuite.byId("ctx-v1-4k")?.title, "4k")
        XCTAssertNil(BenchmarkSuite.byId("standard-v1"), "the pre-release suite is not carried")
    }

    // MARK: - Preflight

    func testPreflightBlocksALadderTheContextCannotHold() {
        // 16384 prompt + 192 answer + 256 template slack.
        XCTAssertEqual(LadderPreflight.need(BenchmarkSuite.ladder), 16832)
        switch LadderPreflight.decide(contextLength: 8192, ladder: BenchmarkSuite.ladder) {
        case .contextTooSmall(let have, let need):
            XCTAssertEqual(have, 8192)
            XCTAssertEqual(need, 16832)
        default: XCTFail("8K must block the 16k rung")
        }
    }

    func testPreflightIsReadyAtOrAboveTheNeed() {
        XCTAssertEqual(LadderPreflight.decide(contextLength: 16832, ladder: BenchmarkSuite.ladder), .ready)
        XCTAssertEqual(LadderPreflight.decide(contextLength: 49152, ladder: BenchmarkSuite.ladder), .ready)
    }

    func testPreflightWithNoModelIsItsOwnOutcome() {
        XCTAssertEqual(LadderPreflight.decide(contextLength: nil, ladder: BenchmarkSuite.ladder), .noModel)
    }

    // MARK: - Settings capture

    private var propsFixture: [String: Any] {
        // Captured from a live `/props` on 26.9.4-dev.
        let json = """
        {"default_generation_settings":{"model":"gemma4","n_ctx":49152},"total_slots":1,
         "settings":{"version":"26.9.4-dev","engine":"mlx","kv_quant":"8","kv_attn_mode":"auto",
           "decode_attn_quant":false,"prefill_chunk":8192,
           "mtp":{"loaded":false,"default_on":false,"acceptance":"exact","acceptance_param":null,"depth":4,"adaptive":true,"max_ctx":0},
           "drafter":"none","pld":{"default_on":true,"draft_len":5,"key_len":3},
           "max_concurrent":1,"prefix_cache":{"mem_bytes":2147483648,"disk_bytes":0}}}
        """
        return try! JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    }

    func testFlattenReadsEveryKeyTheBoardShows() {
        let flat = BenchmarkSettings.flatten(props: propsFixture)
        XCTAssertEqual(flat["engine"], "mlx")
        XCTAssertEqual(flat["version"], "26.9.4-dev")
        XCTAssertEqual(flat["n_ctx"], "49152")
        XCTAssertEqual(flat["kv_quant"], "8")
        XCTAssertEqual(flat["kv_attn_mode"], "auto")
        XCTAssertEqual(flat["decode_attn_quant"], "false")
        XCTAssertEqual(flat["prefill_chunk"], "8192")
        XCTAssertEqual(flat["mtp_loaded"], "false")
        XCTAssertEqual(flat["mtp_default_on"], "false")
        XCTAssertEqual(flat["mtp_depth"], "4")
        XCTAssertEqual(flat["mtp_adaptive"], "true")
        XCTAssertEqual(flat["mtp_acceptance"], "exact")
        XCTAssertEqual(flat["drafter"], "none")
        XCTAssertEqual(flat["pld_default_on"], "true")
        XCTAssertEqual(flat["pld_draft_len"], "5")
        XCTAssertEqual(flat["pld_key_len"], "3")
        XCTAssertEqual(flat["max_concurrent"], "1")
        XCTAssertEqual(flat["prefix_cache_mem"], "2147483648")
    }

    func testFlattenOfAServerWithoutSettingsIsEmptyNotACrash() {
        XCTAssertTrue(BenchmarkSettings.flatten(props: [:]).isEmpty)
        XCTAssertTrue(BenchmarkSettings.flatten(props: ["settings": "junk"]).isEmpty)
    }

    func testTheSignatureIsWhatChangesSpeed() {
        // Two rows may share a median only when these agree. Prefix-cache
        // size, max_concurrent and the server version do not move a single
        // request's tok/s, so they stay out.
        let base = BenchmarkSettings.flatten(props: propsFixture)
        XCTAssertEqual(BenchmarkSettings.signature(base), "kv8|daq0|mtp0|pld1|none")
        var kvOff = base; kvOff["kv_quant"] = "off"
        XCTAssertNotEqual(BenchmarkSettings.signature(kvOff), BenchmarkSettings.signature(base))
        var otherVersion = base; otherVersion["version"] = "27.0.0"; otherVersion["prefix_cache_mem"] = "1"
        XCTAssertEqual(BenchmarkSettings.signature(otherVersion), BenchmarkSettings.signature(base))
        XCTAssertEqual(BenchmarkSettings.signature([:]), "kv?|daq?|mtp?|pld?|?")
    }

    func testLossyIsDerivedFromTheRecordedSettings() {
        XCTAssertTrue(BenchmarkSettings.isLossy(["kv_quant": "8"]))
        XCTAssertTrue(BenchmarkSettings.isLossy(["kv_quant": "off", "decode_attn_quant": "true"]))
        XCTAssertFalse(BenchmarkSettings.isLossy(["kv_quant": "off", "decode_attn_quant": "false"]))
        XCTAssertFalse(BenchmarkSettings.isLossy([:]), "unknown settings are not accused of anything")
    }

    func testSummaryChipsNameWhatMattersInHumanWords() {
        let chips = BenchmarkSettings.summaryChips(BenchmarkSettings.flatten(props: propsFixture))
        XCTAssertEqual(chips, ["KV 8-bit", "PLD", "MTP off", "ctx 48K"])
        let lossless = BenchmarkSettings.summaryChips(["kv_quant": "off", "pld_default_on": "false",
                                                       "mtp_loaded": "true", "mtp_default_on": "true",
                                                       "drafter": "dflash", "n_ctx": "8192"])
        XCTAssertEqual(lossless, ["KV off", "MTP", "DFlash", "ctx 8K"])
        XCTAssertEqual(BenchmarkSettings.summaryChips([:]), [])
    }

    func testSummaryChipsNameTheEmbeddedEngineAndDropItsNonLevers() {
        XCTAssertEqual(BenchmarkSettings.summaryChips(["engine": "ds4", "kv_quant": "off", "decode_attn_quant": "false",
                                                       "mtp_default_on": "true", "drafter": "none", "n_ctx": "131072"]),
                       ["ds4", "MTP", "ctx 128K"])
        XCTAssertEqual(BenchmarkSettings.summaryChips(["engine": "llama", "kv_quant": "q8", "mtp_default_on": "false", "n_ctx": "8192"]),
                       ["llama.cpp", "KV q8", "ctx 8K"])
    }

    // MARK: - Sessions

    func testSessionsGroupBySessionIdAndSortRungsAscending() {
        let rows = [
            makeResult(session: "a", suite: "ctx-v1-8k", target: 8192, decode: 30),
            makeResult(session: "a", suite: "ctx-v1-512", target: 512, decode: 50),
            makeResult(session: "b", suite: "ctx-v1-512", target: 512, decode: 55, date: Date(timeIntervalSince1970: 5)),
            makeResult(session: "a", suite: "ctx-v1-4k", target: 4096, decode: 40),
        ]
        let sessions = BenchmarkSession.group(rows)
        XCTAssertEqual(sessions.map(\.id), ["a", "b"], "newest session first")
        XCTAssertEqual(sessions[0].rungs.map(\.targetTokens), [512, 4096, 8192])
        XCTAssertEqual(sessions[0].decode(at: 4096) ?? 0, 40, accuracy: 0.001)
        XCTAssertNil(sessions[0].decode(at: 16384))
        XCTAssertEqual(sessions[0].settings["kv_quant"], "8")
    }

    func testCommunitySortOrdersRungsNumericallyWithDashesLast() {
        let fast = family("a", decodes: [512: 60, 4096: 50])
        let slow = family("b", decodes: [512: 30, 4096: 25])
        let gap = family("c", decodes: [512: 90])   // no 4k rung
        let by4k = [gap, slow, fast].sorted(using: [BenchmarkFamilySort(.rung(4096), order: .reverse)])
        XCTAssertEqual(by4k.map(\.id), ["a", "b", "c"], "descending, missing rung last")
        let by4kAsc = [gap, slow, fast].sorted(using: [BenchmarkFamilySort(.rung(4096))])
        XCTAssertEqual(by4kAsc.map(\.id), ["b", "a", "c"], "ascending, missing rung still last")
        let byModel = [fast, slow].sorted(using: [BenchmarkFamilySort(.model)])
        XCTAssertEqual(byModel.map(\.id), ["a", "b"])
        let byDate = [slow, fast].sorted(using: [BenchmarkFamilySort(.date, order: .reverse)])
        XCTAssertEqual(byDate.map(\.id), ["a", "b"], "newest session first")
    }

    private func family(_ id: String, decodes: [Int: Double]) -> BenchmarkStore.CellFamily {
        BenchmarkStore.CellFamily(
            id: id, modelId: "model-\(id)", settings: [:], isLossy: false,
            hardware: BenchmarkHardware(chip: "Apple M4", gpuCores: 10, ramGB: 16, osVersion: "27.0", onBattery: false),
            rungs: decodes.keys.sorted().map {
                BenchmarkStore.CellFamily.Rung(targetTokens: $0, decodeTps: decodes[$0]!, prefillTps: 0,
                                               ceilingDecodeTps: 0, ttftMs: 0, sampleCount: 1)
            },
            sessionCount: 1,
            latestDate: Date(timeIntervalSince1970: id == "a" ? 2000 : 1000))
    }

    // MARK: - Drift

    func testDriftIsTheEndRemeasureAgainstTheStartAtATenPercentBar() {
        // llmprobe's classifyLoadDrift: (last − first) / first, ±10% is the
        // line between "figures" and "a range".
        XCTAssertEqual(BenchmarkDrift.percent(first: 60, last: 57) ?? 0, -5, accuracy: 0.001)
        XCTAssertEqual(BenchmarkDrift.verdict(percent: -5), .steady)
        XCTAssertEqual(BenchmarkDrift.verdict(percent: -10), .degraded)
        XCTAssertEqual(BenchmarkDrift.verdict(percent: 12.3), .improved)
        XCTAssertEqual(BenchmarkDrift.verdict(percent: nil), .unknown)
        XCTAssertNil(BenchmarkDrift.percent(first: 0, last: 50), "a zero start is no measurement")
        XCTAssertNil(BenchmarkDrift.percent(first: 50, last: nil))
        XCTAssertEqual(BenchmarkDrift.percent(first: 63.39, last: 60.1) ?? 0, -5.2, accuracy: 0.001, "rounded to one decimal")
        XCTAssertEqual(BenchmarkDrift.summary(first: 61.2, last: 58.4, percent: -4.6), "61.2 → 58.4 tok/s (-4.6%, steady)")
        XCTAssertEqual(BenchmarkDrift.summary(first: nil, last: nil, percent: nil), "not measured")
    }

    func testDriftRidesEveryRowAndTheSessionReadsItOnce() throws {
        var a = makeResult(session: "s", suite: "ctx-v1-512", target: 512, decode: 60)
        var b = makeResult(session: "s", suite: "ctx-v1-4k", target: 4096, decode: 50)
        a.driftDecodeTps = 57; a.driftPercent = -5
        b.driftDecodeTps = 57; b.driftPercent = -5
        let session = BenchmarkSession.group([b, a]).first!
        XCTAssertEqual(session.driftPercent ?? 0, -5, accuracy: 0.001)
        XCTAssertEqual(session.driftBaselineTps ?? 0, 60, accuracy: 0.001, "the baseline is the smallest rung")
        let back = try JSONDecoder().decode(BenchmarkResult.self, from: try JSONEncoder().encode(a))
        XCTAssertEqual(back.driftPercent ?? 0, -5, accuracy: 0.001)
        XCTAssertNil(BenchmarkSession.group([makeResult(session: "t", suite: "ctx-v1-512", target: 512, decode: 1)]).first?.driftBaselineTps)
    }

    // MARK: - Publishability

    func testARowWithNoCompletedRunsMeasuredNothing() {
        var empty = makeResult(session: "s", suite: "ctx-v1-512", target: 512, decode: 0)
        empty.runs = 0
        XCTAssertFalse(empty.isPublishable)
        var oneRun = makeResult(session: "s", suite: "ctx-v1-512", target: 512, decode: 42)
        oneRun.runs = 1
        XCTAssertTrue(oneRun.isPublishable)
        // No settings = no comparison axis, and the board drops the row
        // anyway (Firebase stores `{}` as absent). Never offered for sharing.
        var noSettings = oneRun
        noSettings.settings = [:]
        XCTAssertFalse(noSettings.isPublishable)
    }

    // MARK: - Wire format

    func testResultSurvivesACodableRoundTrip() throws {
        let original = makeResult(session: "s", suite: "ctx-v1-4k", target: 4096, decode: 61.5)
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(BenchmarkResult.self, from: data)
        XCTAssertEqual(decoded.sessionId, original.sessionId)
        XCTAssertEqual(decoded.targetTokens, 4096)
        XCTAssertEqual(decoded.ceilingDecodeTps ?? 0, 90, accuracy: 0.001)
        XCTAssertEqual(decoded.contextUsed, true)
        XCTAssertEqual(decoded.settings?["kv_quant"], "8")
        XCTAssertEqual(decoded.armId, "configured")
    }

    func testTheNoteIsTrimmedCappedAndAbsentWhenEmpty() {
        XCTAssertNil(BenchmarkResult.cleanNote("   \n"))
        XCTAssertEqual(BenchmarkResult.cleanNote("  david, fans on max "), "david, fans on max")
        XCTAssertEqual(BenchmarkResult.cleanNote(String(repeating: "x", count: 500))?.count, BenchmarkResult.maxNoteLength)
        var row = makeResult(session: "s", suite: "ctx-v1-512", target: 512, decode: 1)
        row.note = "david"
        let back = try! JSONDecoder().decode(BenchmarkResult.self, from: try! JSONEncoder().encode(row))
        XCTAssertEqual(back.note, "david")
        XCTAssertEqual(BenchmarkSession.group([row]).first?.note, "david")
        XCTAssertNil(BenchmarkSession.group([makeResult(session: "t", suite: "ctx-v1-512", target: 512, decode: 1)]).first?.note)
    }

    func testResultCarriesTheSchemaVersion() {
        XCTAssertEqual(BenchmarkResult.currentSchemaVersion, 2)
        XCTAssertEqual(makeResult(session: "s", suite: "ctx-v1-512", target: 512, decode: 1).schemaVersion, 2)
    }

    // MARK: - Helpers

    private func makeResult(session: String, suite: String, target: Int?, decode: Double,
                            date: Date = Date(timeIntervalSince1970: 10)) -> BenchmarkResult {
        BenchmarkResult(
            sessionId: session,
            suiteId: suite,
            modelId: "mlx-community/Qwen3.6-27B-4bit",
            engineVersion: "26.9.4",
            prefillTps: 900,
            decodeTps: decode,
            ttftMs: 240,
            promptTokens: target ?? 2048,
            completionTokens: 192,
            runs: 2,
            spreadPercent: 4,
            hardware: BenchmarkHardware(chip: "Apple M4 Max", gpuCores: 40, ramGB: 128,
                                        osVersion: "27.0", onBattery: false),
            date: date,
            targetTokens: target,
            ceilingDecodeTps: target == nil ? nil : 90,
            contextUsed: target == nil ? nil : true,
            settings: target == nil ? nil : ["kv_quant": "8", "pld_default_on": "true"]
        )
    }
}
