import XCTest
@testable import MLXCore

private actor MonitoringProbe {
    var inFlight: [String: Int] = [:]
    var maximum: [String: Int] = [:]
    var calls = 0
    func load(_ path: String, notify: @Sendable () -> Void) async throws -> Data {
        inFlight[path, default: 0] += 1
        maximum[path] = max(maximum[path] ?? 0, inFlight[path]!)
        defer { inFlight[path, default: 1] -= 1 }
        try await Task.sleep(nanoseconds: 2_000_000)
        if path == "/v1/models" { return Data("{\"data\":[]}".utf8) }
        calls += 1
        if calls <= 3 { notify() }
        return Data("{\"counters\":{\"generation_tokens_total\":1},\"gauges\":{\"requests_running\":0}}".utf8)
    }
}

private actor MonitoringGate {
    private var continuation: CheckedContinuation<Data, Never>?
    func wait() async -> Data { await withCheckedContinuation { continuation = $0 } }
    func release() { continuation?.resume(returning: Data("{\"counters\":{\"generation_tokens_total\":99},\"gauges\":{}}".utf8)); continuation = nil }
}

@MainActor
final class MonitoringStoreTests: XCTestCase {
    private func storage() -> MonitoringPersistence {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return MonitoringPersistence(url: root.appendingPathComponent("history.json"))
    }

    func testCollectionDoesNotNeedAViewAndRequestsNeverOverlap() async throws {
        let ticks = expectation(description: "background samples")
        ticks.expectedFulfillmentCount = 3
        let probe = MonitoringProbe()
        let store = MonitoringStore(persistence: storage(), transport: { _, path in
            try await probe.load(path) { ticks.fulfill() }
        }, pollInterval: 0.01, modelInterval: 0.01)
        store.setEndpoint(MonitoringEndpoint(url: URL(string: "http://127.0.0.1:8002")!))
        await fulfillment(of: [ticks], timeout: 3)
        store.setEndpoint(nil)
        let maximum = await probe.maximum
        XCTAssertEqual(maximum["/metrics.json"], 1)
        XCTAssertEqual(maximum["/v1/models"], 1)
        XCTAssertGreaterThan(store.history.samples.count, 0)
        XCTAssertNil(store.snapshot)
        XCTAssertEqual(store.status, .stopped)
    }

    func testAResponseFromTheOldEndpointCannotOverwriteTheNewOne() async throws {
        let oldStarted = expectation(description: "old request")
        let newReceived = expectation(description: "new request")
        let gate = MonitoringGate()
        let store = MonitoringStore(persistence: storage(), transport: { endpoint, path in
            if path == "/v1/models" { return Data("{\"data\":[]}".utf8) }
            if endpoint.url.port == 8002 { oldStarted.fulfill(); return await gate.wait() }
            newReceived.fulfill()
            return Data("{\"counters\":{\"generation_tokens_total\":2},\"gauges\":{}}".utf8)
        }, pollInterval: 10)
        store.setEndpoint(MonitoringEndpoint(url: URL(string: "http://127.0.0.1:8002")!))
        await fulfillment(of: [oldStarted], timeout: 3)
        store.setEndpoint(MonitoringEndpoint(url: URL(string: "http://127.0.0.1:8003")!))
        await fulfillment(of: [newReceived], timeout: 3)
        await gate.release()
        for _ in 0..<10 { await Task.yield() }
        XCTAssertEqual(store.snapshot?.counters["generation_tokens_total"], 2)
        store.setEndpoint(nil)
    }

    func testExpiredHistoryIsRemovedEvenWithoutSuccessfulMetrics() async throws {
        let disk = storage()
        var archive = MonitoringArchive()
        archive.servers["http://127.0.0.1:8002"] = MonitoringHistory(
            samples: [MonitoringSample(time: 1, counters: ["generation_tokens_total": 1])])
        try await disk.save(archive)
        let store = MonitoringStore(persistence: disk, clock: { 100_000 })
        await store.restoreHistory()
        let restored = try await disk.load()
        XCTAssertTrue(restored.servers.isEmpty)
        XCTAssertTrue(store.history.samples.isEmpty)
    }

    func testHTTPFailuresAreDistinctAndNeverBecomeZeroMetrics() {
        for (code, expected) in [(401, MonitoringError.authentication), (403, .authentication), (503, .disabled), (404, .unsupported)] {
            XCTAssertThrowsError(try MonitoringHTTP.check(status: code, data: Data())) { error in
                XCTAssertEqual(error as? MonitoringError, expected)
            }
        }
        XCTAssertNoThrow(try MonitoringHTTP.check(status: 200, data: Data("{}".utf8)))
    }

    func testEndpointStorageKeyDoesNotContainCredentials() {
        let endpoint = MonitoringEndpoint(url: URL(string: "http://user:password@LOCALHOST:8002/?key=secret")!, apiKey: "another-secret")
        XCTAssertEqual(endpoint.storageKey, "http://localhost:8002")
    }
}
