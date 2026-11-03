import XCTest
@testable import MLXCore

/// The dashboard's history: per day and per model, in a file of its own, so it outlives a restart.
final class UsageStoreTests: XCTestCase {
    private var path: String!

    override func setUp() {
        path = NSTemporaryDirectory() + "usage-\(UUID().uuidString).db"
    }

    override func tearDown() {
        for suffix in ["", "-wal", "-shm"] { try? FileManager.default.removeItem(atPath: path + suffix) }
    }

    private func delta(_ day: String, _ model: String, prompt: Int64 = 100, generated: Int64 = 10) -> UsageDelta {
        UsageDelta(day: day, model: model,
                   counters: UsageCounters(requests: 1, promptTokens: prompt, cachedTokens: 20, generatedTokens: generated,
                                           prefillNs: 1_000, decodeNs: 2_000))
    }

    func testAdditionsToOneDayAndModelAreSummed() throws {
        let store = try UsageStore(path: path)
        try store.add([delta("2026-10-30", "org/a"), delta("2026-10-30", "org/a", prompt: 50, generated: 5)])
        try store.add([delta("2026-10-30", "org/a", prompt: 1, generated: 1)])
        let rows = try store.rows()
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].counters, UsageCounters(requests: 3, promptTokens: 151, cachedTokens: 60, generatedTokens: 16,
                                                       prefillNs: 3_000, decodeNs: 6_000))
    }

    func testRowsSurviveReopeningTheFileOldestDayFirst() throws {
        do {
            let store = try UsageStore(path: path)
            try store.add([delta("2026-10-31", "org/b"), delta("2026-10-29", "org/a"), delta("2026-10-30", "org/a")])
        }
        let reopened = try UsageStore(path: path)
        XCTAssertEqual(try reopened.rows().map(\.day), ["2026-10-29", "2026-10-30", "2026-10-31"])
    }

    func testADayKeepsItsFastestPeaks() throws {
        let store = try UsageStore(path: path)
        func peaks(_ decode: Double, _ prefill: Double) -> UsageDelta {
            UsageDelta(day: "2026-10-31", model: "org/a",
                       counters: UsageCounters(requests: 1, peakDecodeTPS: decode, peakPrefillTPS: prefill))
        }
        try store.add([peaks(30.5, 2000)])
        try store.add([peaks(45.25, 1500), peaks(20, 0)])
        let row = try XCTUnwrap(store.rows().first?.counters)
        XCTAssertEqual(row.peakDecodeTPS, 45.25)
        XCTAssertEqual(row.peakPrefillTPS, 2000)
    }

    func testAFileWrittenBeforePeaksKeepsItsRowsAndGainsThem() throws {
        do {
            let old = try SQLiteDatabase(path: path)
            try old.execute("""
                CREATE TABLE usage_daily (day TEXT NOT NULL, model TEXT NOT NULL, requests INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL, generation_tokens INTEGER NOT NULL,
                    prefill_ns INTEGER NOT NULL, decode_ns INTEGER NOT NULL, PRIMARY KEY (day, model));
                INSERT INTO usage_daily VALUES ('2026-10-30', 'org/a', 2, 100, 20, 10, 1000, 2000);
                """)
        }
        let store = try UsageStore(path: path)
        try store.add([UsageDelta(day: "2026-10-31", model: "org/a", counters: UsageCounters(requests: 1, peakDecodeTPS: 25))])
        let rows = try store.rows()
        XCTAssertEqual(rows.map(\.counters.requests), [2, 1])
        XCTAssertEqual(rows.map(\.counters.peakDecodeTPS), [0, 25])
    }

    func testResetForgetsOneModelOrEverything() throws {
        let store = try UsageStore(path: path)
        try store.add([delta("2026-10-30", "org/a"), delta("2026-10-30", "org/b"), delta("2026-10-31", "org/a")])
        try store.reset(model: "org/a")
        XCTAssertEqual(try store.rows().map(\.model), ["org/b"])
        try store.reset(model: nil)
        XCTAssertTrue(try store.rows().isEmpty)
    }
}
