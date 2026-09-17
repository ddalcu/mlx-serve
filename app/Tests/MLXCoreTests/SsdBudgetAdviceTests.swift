import XCTest
@testable import MLXCore

final class SsdBudgetAdviceTests: XCTestCase {
    func testRecommendedIsFortySevenPercentOfRam() {
        XCTAssertEqual(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 128 << 30), 60)
        XCTAssertEqual(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 64 << 30), 30)
        XCTAssertEqual(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 512 << 30), 240)
    }

    func testRecommendedNeverBelowTheFloor() {
        XCTAssertEqual(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 16 << 30), SsdBudgetAdvice.minimumGiB)
        XCTAssertEqual(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 0), SsdBudgetAdvice.minimumGiB)
    }

    func testPresetLadderStopsAtTheMachineAndCarriesTheRecommendation() {
        let presets = SsdBudgetAdvice.presets(physicalMemoryBytes: 128 << 30)
        XCTAssertTrue(presets.contains(SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 128 << 30)))
        XCTAssertEqual(presets, presets.sorted())
        XCTAssertEqual(presets, Array(Set(presets)).sorted())
        XCTAssertNil(presets.first { $0 > 128 })
        XCTAssertNil(presets.first { $0 < SsdBudgetAdvice.minimumGiB })
    }

    func testLabelNamesTheRecommendation() {
        let rec = SsdBudgetAdvice.recommendedGiB(physicalMemoryBytes: 128 << 30)
        XCTAssertEqual(SsdBudgetAdvice.label(rec, recommended: rec), "\(rec) GiB (recommended)")
        XCTAssertEqual(SsdBudgetAdvice.label(24, recommended: rec), "24 GiB")
    }
}

final class SsdBudgetSettingsEncodingTests: XCTestCase {
    func testRoundTripsTheServerKey() {
        let o = ModelOverride(json: ["ssd_budget_gb": 60])
        XCTAssertEqual(o.ssdBudgetGB, 60)
        XCTAssertFalse(o.isEmpty)
        XCTAssertEqual(o.json["ssd_budget_gb"] as? Int, 60)
    }

    func testRejectsNonPositiveAndNonInteger() {
        XCTAssertNil(ModelOverride(json: ["ssd_budget_gb": 0]).ssdBudgetGB)
        XCTAssertNil(ModelOverride(json: ["ssd_budget_gb": "60"]).ssdBudgetGB)
        XCTAssertTrue(ModelOverride(json: ["ssd_budget_gb": 0]).isEmpty)
    }

    func testAbsentKeyStaysAbsentInTheWrittenFile() {
        var o = ModelOverride()
        o.ctxSize = 4096
        XCTAssertNil(o.json["ssd_budget_gb"])
    }

    func testFileRoundTrip() throws {
        let path = NSTemporaryDirectory() + "ssd-budget-\(UUID().uuidString).json"
        var file = ModelSettingsFile()
        file.set(ModelOverride(ssdBudgetGB: 60), for: "/models/Qwen3.8-Flash-Next/")
        try file.save(path: path)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let back = ModelSettingsFile.load(path: path)
        XCTAssertEqual(back.override(for: "/models/Qwen3.8-Flash-Next")?.ssdBudgetGB, 60)
    }
}

final class StreamedModelRowTests: XCTestCase {
    func testParsesTopLevelStreamingMarkerAndBudget() {
        let row = APIClient.parseModelInfo([
            "id": "Qwen3.8-Flash-Next", "streaming": true, "ssd_budget_gb": 60,
            "meta": ["architecture": "qwen4_exp"],
        ])
        XCTAssertTrue(row.streaming)
        XCTAssertEqual(row.ssdBudgetGB, 60)
        XCTAssertEqual(row.storageBadge, "SSD")
    }

    func testStreamingRowWithNoBudgetYet() {
        let row = APIClient.parseModelInfo(["id": "x", "streaming": true, "meta": [:]])
        XCTAssertTrue(row.streaming)
        XCTAssertNil(row.ssdBudgetGB)
        XCTAssertEqual(row.storageBadge, "SSD")
    }

    func testOrdinaryRowHasNoBadge() {
        let row = APIClient.parseModelInfo(["id": "y", "meta": ["architecture": "llama"]])
        XCTAssertFalse(row.streaming)
        XCTAssertNil(row.storageBadge)
    }

    func testSheetShowsTheBudgetRowOnlyForStreamingModels() {
        XCTAssertTrue(ModelSettingsApply.ssdBudgetRow(streaming: true))
        XCTAssertFalse(ModelSettingsApply.ssdBudgetRow(streaming: false))
    }
}

private func body(type: String, message: String = "nope") -> Data {
    try! JSONSerialization.data(withJSONObject: [
        "error": ["message": message, "type": type, "param": NSNull(), "code": 503],
    ])
}

final class LoadFailureVerdictTests: XCTestCase {
    func testNamedRefusalsDoNotRestartTheServer() {
        for t in ["expert_streaming_required", "ssd_budget_below_resident",
                  "ssd_budget_exceeds_wired_limit", "expert_streaming_mtp_unsupported",
                  "expert_streaming_unsupported_layout", "expert_slab_import_copied"] {
            XCTAssertFalse(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: body(type: t)), t)
            XCTAssertEqual(LoadFailureVerdict.errorType(fromBody: body(type: t)), t)
        }
    }

    func testEveryOtherFailureStillRestarts() {
        XCTAssertTrue(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: body(type: "out_of_memory")))
        XCTAssertTrue(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: body(type: "model_load_failed")))
        XCTAssertTrue(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: Data()))
        XCTAssertTrue(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: Data("<html>502</html>".utf8)))
        XCTAssertNil(LoadFailureVerdict.errorType(fromBody: Data("{}".utf8)))
    }

    func testTheTypeDecides_NeverTheProse() {
        let prose = body(type: "out_of_memory",
                         message: "set this model's ssd_budget_gb in model-settings.json")
        XCTAssertTrue(LoadFailureVerdict.shouldRestartAfterLoadFailure(body: prose))
    }
}

final class LoadFailureErrorShapeTests: XCTestCase {
    func testARefusalCarriesItsTypeAndTheBareSentence() {
        let sentence = "This dense qwen4_exp checkpoint streams its experts from SSD and needs a resident budget."
        let err = APIError.fromLoadFailure(code: 503, body: body(type: "expert_streaming_required", message: sentence))
        guard case let APIError.loadRefused(type, detail) = err else {
            return XCTFail("expected .loadRefused, got \(err)")
        }
        XCTAssertEqual(type, "expert_streaming_required")
        XCTAssertEqual(detail, sentence)
        XCTAssertEqual(err.errorDescription, sentence)
        XCTAssertFalse(err.errorDescription!.contains("{"))
    }

    func testAnOrdinaryFailureStaysBadStatusButLosesTheJSONWrapper() {
        let err = APIError.fromLoadFailure(code: 503, body: body(type: "out_of_memory", message: "not enough memory"))
        guard case let APIError.badStatus(code, detail) = err else {
            return XCTFail("expected .badStatus, got \(err)")
        }
        XCTAssertEqual(code, 503)
        XCTAssertEqual(detail, "not enough memory")
    }

    func testAForeignBodyDegradesToItsSnippet() {
        let err = APIError.fromLoadFailure(code: 502, body: Data("Bad Gateway".utf8))
        guard case let APIError.badStatus(_, detail) = err else { return XCTFail("expected .badStatus") }
        XCTAssertEqual(detail, "Bad Gateway")
    }
}
