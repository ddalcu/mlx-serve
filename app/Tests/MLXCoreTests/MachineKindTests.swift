import XCTest
@testable import MLXCore

/// The Cluster map names each machine from what its server says about it.
final class MachineKindTests: XCTestCase {
    func testEachMachineIsNamedByItsKind() {
        func label(_ peer: String) -> String? {
            APIClient.parseModelsBody(Data(#"{"data":[],"lan_peers":[{"name":"p"\#(peer)}]}"#.utf8)).lanPeers["p"]?.kind.label
        }
        XCTAssertEqual(label(#","os":"macos","model":"Mac16,5""#), "Mac")
        XCTAssertEqual(label(#","os":"ios","model":"iPad16,3""#), "iPad")
        XCTAssertEqual(label(#","os":"ios","model":"iPhone17,1""#), "iPhone")
        XCTAssertEqual(label(#","os":"linux","model":"""#), "Linux PC")
        XCTAssertEqual(label(""), "Computer")
    }

    func testThisMacKnowsItsModel() {
        XCTAssertFalse(MachineKind.thisMac.model.isEmpty)
    }
}
