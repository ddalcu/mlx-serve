import XCTest
@testable import MLXCore

final class MonitoringTests: XCTestCase {
    private func feed(_ counters: [String: Double] = [:], gauges: [String: Double] = [:],
                      sessions: [[String: Any]]? = []) throws -> MonitoringSnapshot {
        let actualGauges = counters.isEmpty && gauges.isEmpty ? ["requests_running": 0.0] : gauges
        var raw: [String: Any] = ["counters": counters, "gauges": actualGauges,
            "histograms": ["time_to_first_token_seconds": ["sum": 4.0, "count": 2.0]]]
        if let sessions { raw["sessions"] = sessions }
        return try MonitoringSnapshot.parse(JSONSerialization.data(withJSONObject: raw))
    }

    func testMissingAndInvalidNumbersStayUnknown() throws {
        let raw: [String: Any] = ["counters": ["generation_tokens_total": "wrong", "requests_success_total": true],
                                 "gauges": ["memory_mb": -1, "requests_running": 0]]
        let snapshot = try MonitoringSnapshot.parse(JSONSerialization.data(withJSONObject: raw))
        XCTAssertNil(snapshot.counters["generation_tokens_total"])
        XCTAssertNil(snapshot.counters["requests_success_total"])
        XCTAssertNil(snapshot.gauges["memory_mb"])
        XCTAssertEqual(snapshot.gauges["requests_running"], 0)
        XCTAssertThrowsError(try MonitoringSnapshot.parse(Data("<html>not metrics</html>".utf8)))
        XCTAssertThrowsError(try MonitoringSnapshot.parse(Data("{}".utf8)))
    }

    func testEmptyMetricsAreUnsupported() {
        XCTAssertThrowsError(try MonitoringSnapshot.parse(Data("{\"counters\":{},\"gauges\":{}}".utf8)))
    }

    func testAChartDoesNotFillBucketsBetweenObservedSamples() throws {
        var history = MonitoringHistory()
        history.accept(try feed(["generation_tokens_total": 0]), at: 0)
        history.accept(try feed(["generation_tokens_total": 40]), at: 20)
        let points = history.series(keys: ["generation_tokens_total"], from: 0, to: 20, count: 2)
        XCTAssertNil(points[0].value)
        XCTAssertEqual(points[1].value, 2)
    }

    func testUnknownCountersDoNotEnterPersistedHistory() throws {
        var history = MonitoringHistory()
        history.accept(try feed(["generation_tokens_total": 1, "future_unrelated_counter": 2]), at: 1)
        XCTAssertNil(history.samples.first?.counters["future_unrelated_counter"])
    }

    func testEndpointArchiveKeepsOnlyTheEightMostRecentServers() throws {
        var archive = MonitoringArchive()
        for i in 0..<12 {
            var history = MonitoringHistory()
            history.accept(try feed(["generation_tokens_total": 1]), at: Double(i + 1))
            archive.servers[String(i)] = history
        }
        archive.compact(at: 20)
        XCTAssertEqual(archive.servers.count, 8)
        XCTAssertNil(archive.servers["0"])
        XCTAssertNotNil(archive.servers["11"])
    }

    func testModelSizeUsesDiskBytesWhenTheModelIsUnloaded() {
        let model = APIClient.parseModelInfo(["id": "Example/Image", "loaded": false,
            "bytes_resident": UInt64(0), "bytes_on_disk": UInt64(2_500_000_000)])
        XCTAssertEqual(MonitorFormat.modelSize(model), ByteCountFormatter.string(fromByteCount: 2_500_000_000, countStyle: .memory))
    }

    func testHistogramAverageUsesCountAndMilliseconds() throws {
        let snapshot = try feed()
        XCTAssertEqual(snapshot.averageMilliseconds("time_to_first_token_seconds"), 2000)
        XCTAssertNil(snapshot.averageMilliseconds("absent"))
    }

    func testLiveRatesIncludeInflightWorkAndStopWithThePhase() throws {
        var live = MonitoringLiveHistory()
        _ = live.accept(try feed(gauges: ["generation_tokens_live": 100, "prefill_tokens_live": 200, "requests_running": 1]), at: 10)
        let rates = live.accept(try feed(gauges: ["generation_tokens_live": 140, "prefill_tokens_live": 260, "requests_running": 1]), at: 12)
        XCTAssertEqual(rates.decode, 20)
        XCTAssertEqual(rates.prefill, 30)
        let idle = live.accept(try feed(gauges: ["generation_tokens_live": 140, "prefill_tokens_live": 0, "requests_running": 0]), at: 13)
        XCTAssertEqual(idle.decode, 0)
        XCTAssertEqual(idle.prefill, 0)
        let reset = live.accept(try feed(gauges: ["generation_tokens_live": 1, "prefill_tokens_live": 1, "requests_running": 1]), at: 14)
        XCTAssertNil(reset.decode)
        XCTAssertNil(reset.prefill)
    }

    func testTotalsRejectResetsAndGapsAndDoNotInventModelAttribution() throws {
        var history = MonitoringHistory()
        let session: [[String: Any]] = [["request_id": 1, "model": "one", "phase": "decode"]]
        history.accept(try feed(["generation_tokens_total": 10], gauges: ["process_start_time_seconds": 1], sessions: session), at: 10)
        history.accept(try feed(["generation_tokens_total": 30], gauges: ["process_start_time_seconds": 1], sessions: session), at: 20)
        XCTAssertEqual(history.totals(from: 0, to: 25)["generation_tokens_total"], 20)
        XCTAssertEqual(history.modelTotals(from: 0, to: 25)["one"]?["generation_tokens_total"], 20)
        history.accept(try feed(["generation_tokens_total": 100], gauges: ["process_start_time_seconds": 2]), at: 30)
        history.accept(try feed(["generation_tokens_total": 200], gauges: ["process_start_time_seconds": 2]), at: 400)
        XCTAssertEqual(history.totals(from: 0, to: 500)["generation_tokens_total"], 20)
    }

    func testHistoricalChartHasGapsRatherThanConnectingMissingData() throws {
        var history = MonitoringHistory()
        history.accept(try feed(["generation_tokens_total": 0]), at: 0)
        history.accept(try feed(["generation_tokens_total": 20]), at: 10)
        history.accept(try feed(["generation_tokens_total": 40]), at: 400)
        let points = history.series(keys: ["generation_tokens_total"], from: 0, to: 400, count: 40)
        XCTAssertEqual(points.first?.value, 2)
        XCTAssertTrue(points.dropFirst().contains { $0.value == nil })
        XCTAssertEqual(points.count, 40)
    }

    func testCompactionKeepsOnly24HoursAndTheRecentSeconds() throws {
        var history = MonitoringHistory()
        for second in stride(from: 0, through: 90_000, by: 10) {
            history.accept(try feed(["generation_tokens_total": Double(second)]), at: Double(second))
        }
        history.compact(at: 90_000)
        XCTAssertTrue(history.samples.allSatisfy { $0.time >= 3_600 })
        XCTAssertLessThan(history.samples.count, 2000)
        XCTAssertTrue(history.samples.contains { $0.time == 89_990 })
    }

    func testRequestObservationsAreNotSuccessfulCompletionsAndIdsCanRestart() throws {
        var history = MonitoringHistory()
        let session: [[String: Any]] = [["request_id": 7, "model": "one", "phase": "prefill", "client": "opencode"]]
        history.accept(try feed(gauges: ["process_start_time_seconds": 1], sessions: session), at: 10)
        history.accept(try feed(gauges: ["process_start_time_seconds": 1], sessions: nil), at: 11)
        XCTAssertNil(history.requests.first?.endTime)
        history.accept(try feed(gauges: ["process_start_time_seconds": 1]), at: 12)
        XCTAssertNotNil(history.requests.first?.endTime)
        XCTAssertEqual(history.requests.first?.phase, "prefill")
        history.accept(try feed(gauges: ["process_start_time_seconds": 2], sessions: session), at: 13)
        XCTAssertEqual(history.requests.count, 2)
        XCTAssertEqual(history.visibleRequests(from: 0, to: 20).count, 2)
    }

    func testSingleModelCompletionIsAttributedButOverlappingActivityIsNot() throws {
        var history = MonitoringHistory()
        let one: [[String: Any]] = [["request_id": 1, "model": "one", "phase": "decode"]]
        history.accept(try feed(["requests_success_total": 0], sessions: one), at: 1)
        history.accept(try feed(["requests_success_total": 1]), at: 2)
        XCTAssertEqual(history.modelTotals(from: 0, to: 3)["one"]?["requests_success_total"], 1)
        let two = one + [["request_id": 2, "model": "two", "phase": "decode"]]
        history.accept(try feed(["requests_success_total": 2], sessions: two), at: 3)
        history.accept(try feed(["requests_success_total": 3], sessions: one), at: 4)
        XCTAssertEqual(history.modelTotals(from: 0, to: 5)["unattributed"]?["requests_success_total"], 2)
    }

    func testMissingCounterDoesNotProduceAPartialTotalThatLooksComplete() throws {
        var history = MonitoringHistory()
        history.accept(try feed(["generation_tokens_total": 0, "requests_success_total": 0]), at: 1)
        history.accept(try feed(["generation_tokens_total": 10, "requests_success_total": 1]), at: 2)
        history.accept(try feed(["generation_tokens_total": 20]), at: 3)
        XCTAssertEqual(history.totals(from: 0, to: 4)["generation_tokens_total"], 20)
        XCTAssertNil(history.totals(from: 0, to: 4)["requests_success_total"])
    }

    func testCachedSessionsDoNotBecomeRequestHistory() throws {
        var history = MonitoringHistory()
        history.accept(try feed(sessions: [["request_id": 0, "model": "one", "phase": "cached"]]), at: 1)
        XCTAssertTrue(history.requests.isEmpty)
    }

    func testMonitoringUsesTheStandardWorkspaceAndDoesNotRequireAChatModel() {
        XCTAssertTrue(ChatWorkspace.monitoring.isMonitoring)
        XCTAssertFalse(ChatWorkspace.monitoring.isThreeColumn)
        XCTAssertFalse(ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
            workspace: .monitoring, welcomePresented: false))
        XCTAssertEqual(MonitoringRange.allCases.map(\.seconds), [60, 600, 1800, 3600, 86400])
    }

    func testHistoryRoundTripAndCorruptFileReporting() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("monitor-history.json")
        let storage = MonitoringPersistence(url: url)
        var archive = MonitoringArchive()
        var history = MonitoringHistory()
        history.accept(try feed(["generation_tokens_total": 1]), at: 10)
        archive.servers["http://127.0.0.1:8002"] = history
        try await storage.save(archive)
        let restored = try await storage.load()
        XCTAssertEqual(restored.servers.values.first?.samples.count, 1)
        try Data("broken".utf8).write(to: url)
        do { _ = try await storage.load(); XCTFail("corrupt data must be reported") } catch {}
    }
}
