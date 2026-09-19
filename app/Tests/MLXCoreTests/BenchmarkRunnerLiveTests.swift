import XCTest
@testable import MLXCore

/// The ladder against a LIVE server. Env-gated: `BENCH_LIVE_PORT=<port>`
/// (and optionally `BENCH_LIVE_MODEL=<id>`), skipped otherwise.
///
/// Pins the invariants the unit tests cannot: every rung lands near its
/// target token count (the byte→token fit works on this tokenizer), the
/// counting run measures a ceiling, and the settings rode the rows.
final class BenchmarkRunnerLiveTests: XCTestCase {

    @MainActor
    func testTheLadderClimbsAgainstALiveServer() async throws {
        guard let portText = ProcessInfo.processInfo.environment["BENCH_LIVE_PORT"],
              let port = UInt16(portText) else {
            throw XCTSkip("set BENCH_LIVE_PORT to run against a live server")
        }
        let api = APIClient()
        let props = try await api.fetchPropsRaw(port: port)
        let model = ProcessInfo.processInfo.environment["BENCH_LIVE_MODEL"]
            ?? ((props["default_generation_settings"] as? [String: Any])?["model"] as? String ?? "")
        XCTAssertFalse(model.isEmpty, "no model to benchmark")

        let ladder = Array(BenchmarkSuite.ladder.prefix(
            Int(ProcessInfo.processInfo.environment["BENCH_LIVE_RUNGS"] ?? "4") ?? 4))
        let runner = BenchmarkRunner(api: api)
        let rows = await runner.run(ladder: ladder, modelId: model, port: port)

        XCTAssertEqual(runner.phase, .done, "\(runner.phase)")
        XCTAssertEqual(rows.count, ladder.count)
        for row in rows {
            let target = row.targetTokens ?? 0
            print("rung \(target): prompt=\(row.promptTokens) prefill=\(row.prefillTps) decode=\(row.decodeTps) ceiling=\(row.ceilingDecodeTps ?? 0) ttft=\(row.ttftMs) ctxUsed=\(row.contextUsed ?? false) runs=\(row.runs)")
            // Within 15% of the rung — the point of the fit.
            XCTAssertLessThan(abs(Double(row.promptTokens - target)) / Double(target), 0.15, "rung \(target) landed at \(row.promptTokens)")
            XCTAssertGreaterThan(row.decodeTps, 0)
            XCTAssertGreaterThan(row.prefillTps, 0)
            // The ceiling is the checkpoint's choice: gemma-4-e2b answers "1"
            // and stops, which the length floor turns into an honest 0.
            XCTAssertEqual(row.runs, 2)
            XCTAssertEqual(row.settings?["kv_quant"], BenchmarkSettings.flatten(props: props)["kv_quant"])
            XCTAssertEqual(row.armId, "configured")
        }
        XCTAssertEqual(runner.discardedRuns, 0)
        print("drift: \(BenchmarkDrift.summary(first: rows.first?.decodeTps, last: rows.first?.driftDecodeTps, percent: rows.first?.driftPercent))")
        XCTAssertNotNil(rows.first?.driftPercent, "the drift re-measure never landed")
    }
}

/// Decodes the LIVE community database with the app's own coders and names
/// any row that fails. Env-gated: `BENCH_LIVE_COMMUNITY=1`.
final class BenchmarkCommunityLiveTests: XCTestCase {
    func testEveryLiveRowDecodes() async throws {
        guard ProcessInfo.processInfo.environment["BENCH_LIVE_COMMUNITY"] == "1" else {
            throw XCTSkip("set BENCH_LIVE_COMMUNITY=1")
        }
        let url = BenchmarkStore.communityFetchURL(limit: 500)!
        let (data, _) = try await URLSession.shared.data(from: url)
        let dict = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var ok = 0
        for (key, value) in dict {
            let rowData = try JSONSerialization.data(withJSONObject: value)
            do { _ = try BenchmarkStore.decoder.decode(BenchmarkResult.self, from: rowData); ok += 1 }
            catch { print("ROW \(key) FAILED: \(error)") }
        }
        print("decoded \(ok) of \(dict.count)")
        XCTAssertEqual(ok, dict.count)
    }
}
