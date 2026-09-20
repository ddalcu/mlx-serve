import XCTest
@testable import MLXCore

/// Local history + the community wire format.
///
/// The community database is append-only and openly writable in phase 1, so
/// the READER is what has to be robust: one malformed row must never blank the
/// whole board, and a row written by a future version must not break a client
/// that predates it.
final class BenchmarkStoreTests: XCTestCase {

    // MARK: - Wire format

    func testDatesRideAsISO8601SoTheWebsiteCanReadThemDirectly() throws {
        let row = makeRow(decode: 50)
        let data = try BenchmarkStore.encoder.encode(row)
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertTrue(text.contains("\"date\":\""), "date is not a string — the website can't parse it")
        XCTAssertTrue(text.contains("T"), "date is not ISO8601")
    }

    func testWireRoundTripsThroughTheSharedCodersWithSettings() throws {
        let row = makeRow(decode: 61.5)
        let data = try BenchmarkStore.encoder.encode(row)
        let back = try BenchmarkStore.decoder.decode(BenchmarkResult.self, from: data)
        XCTAssertEqual(back.id, row.id)
        XCTAssertEqual(back.decodeTps, row.decodeTps, accuracy: 0.0001)
        XCTAssertEqual(back.hardware.gpuCores, row.hardware.gpuCores)
        XCTAssertEqual(back.date.timeIntervalSince1970, row.date.timeIntervalSince1970, accuracy: 1.0)
        XCTAssertEqual(back.settings, row.settings)
        XCTAssertEqual(back.targetTokens, 4096)
        XCTAssertEqual(back.ceilingDecodeTps ?? 0, 88, accuracy: 0.0001)
    }

    func testAPreReleaseRowIsDroppedOnRead() throws {
        // The single-prompt suite that shipped before the ladder carried no
        // rung; it is not carried forward, on disk or from the database.
        let v1 = """
        {"id":"r1","schemaVersion":1,"sessionId":"s","suiteId":"standard-v1","armId":"defaults",
         "armLabel":"Defaults","isLossy":false,"modelId":"m","engineVersion":"26.8.1",
         "prefillTps":900,"decodeTps":50,"ttftMs":200,"promptTokens":2048,"completionTokens":128,
         "runs":3,"spreadPercent":4,"date":"2026-09-01T10:00:00Z",
         "hardware":{"chip":"Apple M4","gpuCores":10,"ramGB":16,"osVersion":"27.0","onBattery":false}}
        """
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("{\"-Nold\":\(v1)}".utf8)).isEmpty)
        let v2 = try String(data: BenchmarkStore.encoder.encode(makeRow(decode: 50)), encoding: .utf8)!
        XCTAssertEqual(BenchmarkStore.decodeCommunity(Data("{\"-Nold\":\(v1),\"-Nnew\":\(v2)}".utf8)).count, 1)
    }

    func testARowFirebaseHandedBackWithoutItsEmptyFlagsStillDecodes() throws {
        // RTDB stores no empty object: `flags: {}` is dropped on write, so
        // every shared v2 row came back without it and the board was blank
        // while the rows sat in the database (live, 2026-09-17).
        var object = try JSONSerialization.jsonObject(
            with: BenchmarkStore.encoder.encode(makeRow(decode: 50))) as! [String: Any]
        object.removeValue(forKey: "flags")
        let row = try BenchmarkStore.decoder.decode(BenchmarkResult.self,
                                                    from: try JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(row.flags, [:])
        XCTAssertEqual(row.decodeTps, 50, accuracy: 0.001)
    }

    // MARK: - Reading the community database

    func testRTDBResponseIsADictionaryOfRowsNotAnArray() throws {
        let row = try String(data: BenchmarkStore.encoder.encode(makeRow(decode: 50)), encoding: .utf8)!
        let payload = "{\"-NpushKeyA\":\(row),\"-NpushKeyB\":\(row)}"
        XCTAssertEqual(BenchmarkStore.decodeCommunity(Data(payload.utf8)).count, 2)
    }

    func testAnEmptyDatabaseReadsAsNoRowsRatherThanAnError() {
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("null".utf8)).isEmpty)
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("".utf8)).isEmpty)
    }

    func testOneMalformedRowDoesNotBlankTheWholeBoard() throws {
        let good = try String(data: BenchmarkStore.encoder.encode(makeRow(decode: 50)), encoding: .utf8)!
        let payload = "{\"-Ngood\":\(good),\"-Njunk\":{\"nonsense\":true}}"
        let rows = BenchmarkStore.decodeCommunity(Data(payload.utf8))
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.armId, "configured")
    }

    func testUnknownFieldsFromAFutureVersionAreIgnored() throws {
        var object = try JSONSerialization.jsonObject(
            with: BenchmarkStore.encoder.encode(makeRow(decode: 50))) as! [String: Any]
        object["trust"] = "attested"
        object["attestationKeyId"] = "abc123"
        let wrapped = try JSONSerialization.data(withJSONObject: ["-Nrow": object])
        XCTAssertEqual(BenchmarkStore.decodeCommunity(wrapped).count, 1)
    }

    // MARK: - Local history

    func testMergeKeepsNewestFirstAndDedupesById() {
        let old = makeRow(decode: 50, date: Date(timeIntervalSince1970: 1_000))
        let new = makeRow(decode: 60, date: Date(timeIntervalSince1970: 2_000))
        let merged = BenchmarkStore.merged([old], adding: [new, old])
        XCTAssertEqual(merged.count, 2)
        XCTAssertEqual(merged.first?.decodeTps ?? 0, 60, accuracy: 0.001, "newest row is not first")
    }

    // MARK: - Aggregation

    func testFamilyKeyGroupsOnlyRowsThatAreActuallyComparable() {
        let a = makeRow(decode: 50, gpuCores: 40)
        let b = makeRow(decode: 30, gpuCores: 32)
        XCTAssertNotEqual(BenchmarkStore.familyKey(a), BenchmarkStore.familyKey(b))
        let c = makeRow(decode: 52, gpuCores: 40)
        XCTAssertEqual(BenchmarkStore.familyKey(a), BenchmarkStore.familyKey(c))
    }

    func testFamilyKeySeparatesSettingsSignatures() {
        // A kv-quant 8 row and a lossless row measured the same rung on the
        // same Mac and must never share a median.
        let quant = makeRow(decode: 50, settings: ["kv_quant": "8", "pld_default_on": "true"])
        let dense = makeRow(decode: 45, settings: ["kv_quant": "off", "pld_default_on": "true"])
        XCTAssertNotEqual(BenchmarkStore.familyKey(quant), BenchmarkStore.familyKey(dense))
        // ...but a different server version or cache size is the same family.
        let later = makeRow(decode: 51, settings: ["kv_quant": "8", "pld_default_on": "true",
                                                    "version": "27.0.0", "prefix_cache_mem": "1"])
        XCTAssertEqual(BenchmarkStore.familyKey(quant), BenchmarkStore.familyKey(later))
    }

    func testAggregateSessionsBuildsOneFamilyPerMachineModelAndSettings() {
        // Two sessions on one machine at the same settings → one family with
        // per-rung medians and n = sessions behind each rung.
        let rows = [
            makeRow(decode: 50, date: Date(timeIntervalSince1970: 1_000), session: "a", suite: "ctx-v1-512", target: 512),
            makeRow(decode: 40, date: Date(timeIntervalSince1970: 1_100), session: "a", suite: "ctx-v1-4k", target: 4096),
            makeRow(decode: 60, date: Date(timeIntervalSince1970: 3_000), session: "b", suite: "ctx-v1-512", target: 512),
            makeRow(decode: 30, date: Date(timeIntervalSince1970: 9_000), session: "c", suite: "ctx-v1-512", target: 512,
                    settings: ["kv_quant": "off"]),
        ]
        let families = BenchmarkStore.aggregateSessions(rows)
        XCTAssertEqual(families.count, 2)
        guard let quant = families.first(where: { $0.settings["kv_quant"] == "8" }) else {
            return XCTFail("kv-8 family missing")
        }
        XCTAssertEqual(quant.decode(at: 512) ?? 0, 55, accuracy: 0.001)
        XCTAssertEqual(quant.samples(at: 512), 2)
        XCTAssertEqual(quant.decode(at: 4096) ?? 0, 40, accuracy: 0.001)
        XCTAssertEqual(quant.samples(at: 4096), 1)
        XCTAssertNil(quant.decode(at: 16384))
        XCTAssertEqual(quant.sessionCount, 2)
        XCTAssertEqual(quant.latestDate, Date(timeIntervalSince1970: 3_000), "newest of ITS OWN sessions, not the board's")
    }

    // MARK: - Shared marker

    func testSharedRowsAreRememberedLocallyAndNeverOnTheRow() throws {
        // The marker must not ride the POST: the rules reject unknown fields,
        // so a `shared` field on the row would 401 every submission.
        let defaults = UserDefaults(suiteName: "BenchmarkStoreTests.\(UUID().uuidString)")!
        let r1 = makeRow(decode: 50, suite: "ctx-v1-512", target: 512)
        let r2 = makeRow(decode: 40, suite: "ctx-v1-1k", target: 1024)
        let session = BenchmarkSession(id: "s1", rungs: [r1, r2])
        XCTAssertFalse(BenchmarkStore.isShared(session, defaults: defaults))
        BenchmarkStore.markShared([r1.id], defaults: defaults)
        BenchmarkStore.markShared([r1.id], defaults: defaults)
        XCTAssertEqual(BenchmarkStore.sharedRowIds(defaults).count, 1)
        XCTAssertFalse(BenchmarkStore.isShared(session, defaults: defaults), "half a session is not shared")
        BenchmarkStore.markShared([r2.id], defaults: defaults)
        XCTAssertTrue(BenchmarkStore.isShared(session, defaults: defaults))

        let wire = try JSONSerialization.jsonObject(with: BenchmarkStore.encoder.encode(r1)) as! [String: Any]
        XCTAssertNil(wire["shared"])
        XCTAssertNil(wire["sharedAt"])
    }

    func testARetryAfterAPartialShareSendsOnlyTheRowsThatDidNotLand() {
        // A session POSTs one row at a time. When row 2 fails, row 1 is
        // already in the database; sharing again must not send it twice.
        let defaults = UserDefaults(suiteName: "BenchmarkStoreTests.\(UUID().uuidString)")!
        let r1 = makeRow(decode: 50, suite: "ctx-v1-512", target: 512)
        let r2 = makeRow(decode: 40, suite: "ctx-v1-1k", target: 1024)
        BenchmarkStore.markShared([r1.id], defaults: defaults)
        XCTAssertEqual(BenchmarkStore.unsent([r1, r2], defaults: defaults).map(\.id), [r2.id])
    }

    func testARowWithoutSettingsIsDroppedOnReadLikeTheWebsiteDoes() throws {
        // Firebase stores no empty object, so a row shared with `settings: {}`
        // comes back without the key. The website rejects such a row; the app
        // must too, or the two quote different tables for one database.
        var row = makeRow(decode: 50)
        row.settings = nil
        let wire = try String(data: BenchmarkStore.encoder.encode(row), encoding: .utf8)!
        XCTAssertTrue(BenchmarkStore.decodeCommunity(Data("{\"-Na\":\(wire)}".utf8)).isEmpty)
    }

    // MARK: - Helpers

    private func makeRow(decode: Double, gpuCores: Int = 40, date: Date = Date(),
                         session: String = "s1", suite: String = "ctx-v1-4k", target: Int = 4096,
                         settings: [String: String] = ["kv_quant": "8", "pld_default_on": "true"]) -> BenchmarkResult {
        BenchmarkResult(
            sessionId: session,
            suiteId: suite,
            modelId: "mlx-community/Qwen3.6-27B-4bit",
            engineVersion: "26.9.4",
            prefillTps: 900,
            decodeTps: decode,
            ttftMs: 240,
            promptTokens: target,
            completionTokens: 192,
            runs: 2,
            spreadPercent: 4,
            hardware: BenchmarkHardware(chip: "Apple M4 Max", gpuCores: gpuCores,
                                        ramGB: 128, osVersion: "27.0", onBattery: false),
            date: date,
            targetTokens: target,
            ceilingDecodeTps: 88,
            contextUsed: true,
            settings: settings
        )
    }
}
