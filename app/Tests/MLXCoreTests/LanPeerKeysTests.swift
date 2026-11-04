import XCTest
@testable import MLXCore

/// Keys for LAN peers that run with --api-key: the file the server reads, and
/// the server's list of peers still asking for one.
final class LanPeerKeysTests: XCTestCase {

    func testFileRoundTripsAndDropsClearedKeys() throws {
        let tmp = NSTemporaryDirectory().appending("lan-keys-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        try LanPeerKeys.save(["Studio": "s3cret", "Mini": "  "], path: tmp)
        XCTAssertEqual(LanPeerKeys.load(path: tmp), ["Studio": "s3cret"])
        XCTAssertEqual(LanPeerKeys.load(path: tmp + ".missing"), [:])
    }

    func testRowsAreEveryPeerAskingPlusEveryPeerWithAKey() {
        XCTAssertEqual(LanPeerKeys.rows(needingKey: ["Studio", "Air"], saved: ["Mini": "k", "Studio": "old"]),
                       ["Air", "Mini", "Studio"])
    }

    func testModelsBodyCarriesThePeersNeedingAKey() throws {
        let body = #"{"object":"list","data":[{"id":"gemma"}],"lan_needs_key":["Studio"]}"#
        let parsed = APIClient.parseModelsBody(Data(body.utf8))
        XCTAssertEqual(parsed.models.map(\.name), ["gemma"])
        XCTAssertEqual(parsed.lanNeedsKey, ["Studio"])
        XCTAssertEqual(APIClient.parseModelsBody(Data(#"{"object":"list","data":[]}"#.utf8)).lanNeedsKey, [])
    }

    func testThePeersAddressesComeWithTheModelList() {
        let body = #"{"object":"list","data":[],"lan_peers":[{"name":"M4Max","url":"http://192.168.137.3:11234"}]}"#
        XCTAssertEqual(APIClient.parseModelsBody(Data(body.utf8)).lanPeers["M4Max"]?.kind, .unknown)
    }
}
