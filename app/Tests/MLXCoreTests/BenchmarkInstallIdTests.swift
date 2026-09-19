import XCTest
@testable import MLXCore

/// A shared row carries a random per-install id, so rows from one Mac can be
/// grouped (or dropped) without the row saying whose Mac it is.
final class BenchmarkInstallIdTests: XCTestCase {

    func testTheInstallIdIsARandomUuidMadeOnceAndKept() {
        let defaults = UserDefaults(suiteName: "BenchmarkInstallIdTests.\(UUID().uuidString)")!
        let first = InstallId.current(defaults)
        XCTAssertNotNil(UUID(uuidString: first))
        XCTAssertEqual(InstallId.current(defaults), first)
        let other = UserDefaults(suiteName: "BenchmarkInstallIdTests.\(UUID().uuidString)")!
        XCTAssertNotEqual(InstallId.current(other), first)
    }

    func testSubmitStampsTheInstallIdOnEveryRow() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [SubmitRecorder.self]
        SubmitRecorder.bodies = []
        let client = BenchmarkCommunityClient(session: URLSession(configuration: config), installId: "install-1")
        let outcome = await client.submit([Self.row(), Self.row()])
        XCTAssertNil(outcome.error)
        let ids = try SubmitRecorder.bodies.map {
            try JSONSerialization.jsonObject(with: $0) as! [String: Any]
        }.map { $0["installId"] as? String }
        XCTAssertEqual(ids, ["install-1", "install-1"])
    }

    func testARowSharedByAnOlderAppStillDecodes() throws {
        var row = Self.row()
        row.installId = nil
        let wire = try String(data: BenchmarkStore.encoder.encode(row), encoding: .utf8)!
        XCTAssertFalse(wire.contains("installId"))
        let back = BenchmarkStore.decodeCommunity(Data("{\"-Na\":\(wire)}".utf8))
        XCTAssertEqual(back.count, 1)
        XCTAssertNil(back.first?.installId)
    }

    /// The rules reject any field they do not name, so a field the app sends
    /// and the rules lack fails every Share.
    func testEveryFieldAFullRowSendsIsOneTheDatabaseRulesAccept() throws {
        let rulesURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("database.rules.json")
        let rules = try JSONSerialization.jsonObject(with: Data(contentsOf: rulesURL)) as! [String: Any]
        let rowRules = ((rules["rules"] as! [String: Any])["results"] as! [String: Any])["$row"] as! [String: Any]
        let hardwareRules = rowRules["hardware"] as! [String: Any]

        var row = Self.row()
        row.installId = "install-1"
        row.quant = "4bit"; row.note = "fan on max"; row.flags = ["a": "b"]
        row.ceilingDecodeTps = 1; row.contextUsed = true
        row.driftDecodeTps = 1; row.driftPercent = 1
        let wire = try JSONSerialization.jsonObject(with: BenchmarkStore.encoder.encode(row)) as! [String: Any]
        for key in wire.keys {
            XCTAssertNotNil(rowRules[key], "the rules reject a row carrying \"\(key)\"")
        }
        for key in (wire["hardware"] as! [String: Any]).keys {
            XCTAssertNotNil(hardwareRules[key], "the rules reject hardware.\(key)")
        }
        let required = (rowRules[".validate"] as! String)
        XCTAssertFalse(required.contains("installId"), "requiring it would refuse every older app's Share")
    }

    private static func row() -> BenchmarkResult {
        BenchmarkResult(
            sessionId: "s1", suiteId: "ctx-v1-512", modelId: "org/model", engineVersion: "26.9.5",
            prefillTps: 100, decodeTps: 50, ttftMs: 10, promptTokens: 512, completionTokens: 64,
            runs: 1, spreadPercent: 0,
            hardware: BenchmarkHardware(chip: "Apple M4 Max", gpuCores: 40, ramGB: 128, osVersion: "27.0", onBattery: false),
            targetTokens: 512, settings: ["kv_quant": "8"])
    }
}

private final class SubmitRecorder: URLProtocol {
    nonisolated(unsafe) static var bodies: [Data] = []
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}
    override func startLoading() {
        if let stream = request.httpBodyStream {
            stream.open(); defer { stream.close() }
            var data = Data(); var buf = [UInt8](repeating: 0, count: 4096)
            while stream.hasBytesAvailable {
                let n = stream.read(&buf, maxLength: buf.count)
                if n <= 0 { break }
                data.append(buf, count: n)
            }
            Self.bodies.append(data)
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("{}".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}
