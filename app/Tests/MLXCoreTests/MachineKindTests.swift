import XCTest
@testable import MLXCore

/// The Cluster map names each machine from its `/health`.
final class MachineKindTests: XCTestCase {
    func testEachMachineIsNamedByItsKind() {
        func label(_ body: String) -> String { MachineKind(health: Data(body.utf8)).label }
        XCTAssertEqual(label(#"{"status":"ok","os":"macos","model":"Mac16,5"}"#), "Mac")
        XCTAssertEqual(label(#"{"status":"ok","os":"ios","model":"iPad16,3"}"#), "iPad")
        XCTAssertEqual(label(#"{"status":"ok","os":"ios","model":"iPhone17,1"}"#), "iPhone")
        XCTAssertEqual(label(#"{"status":"ok","os":"linux","model":""}"#), "Linux PC")
        XCTAssertEqual(label(#"{"status":"ok"}"#), "Computer")
    }

    func testThisMacKnowsItsModel() {
        XCTAssertFalse(MachineKind.thisMac.model.isEmpty)
    }
}
