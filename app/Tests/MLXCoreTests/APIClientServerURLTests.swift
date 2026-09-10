import XCTest
@testable import MLXCore

/// `APIClient.serverURL` must honour a specific interface host while keeping
/// wide binds on loopback — otherwise the app silently loses a server it
/// bound to a LAN IP.
final class APIClientServerURLTests: XCTestCase {

    func testDefaultHostProducesLoopbackURL() {
        let api = APIClient()
        XCTAssertEqual(api.serverURL(port: 11234, path: "/health").absoluteString,
                       "http://127.0.0.1:11234/health")
    }

    func testWideBind0000FallsBackToLoopback() {
        let api = APIClient()
        api.host = "0.0.0.0"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/health").absoluteString,
                       "http://127.0.0.1:11234/health")
    }

    func testWideBindIPv6FallsBackToLoopback() {
        let api = APIClient()
        api.host = "::"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/v1/models").absoluteString,
                       "http://127.0.0.1:11234/v1/models")
    }

    func testEmptyHostFallsBackToLoopback() {
        let api = APIClient()
        api.host = ""
        XCTAssertEqual(api.serverURL(port: 8080, path: "/props").absoluteString,
                       "http://127.0.0.1:8080/props")
    }

    func testExplicitLoopbackIsKept() {
        let api = APIClient()
        api.host = "127.0.0.1"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/metrics.json").absoluteString,
                       "http://127.0.0.1:11234/metrics.json")
    }

    func testSpecificLanIPIsUsedDirectly() {
        let api = APIClient()
        api.host = "192.168.1.10"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/v1/audio/speech").absoluteString,
                       "http://192.168.1.10:11234/v1/audio/speech")
    }

    func testIPv6HostIsBracketed() {
        let api = APIClient()
        api.host = "::1"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/health").absoluteString,
                       "http://[::1]:11234/health")
    }

    func testAlreadyBracketedIPv6IsNotDoubleBracketed() {
        let api = APIClient()
        api.host = "[::1]"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/health").absoluteString,
                       "http://[::1]:11234/health")
    }

    func testUnparseableHostFallsBackToLoopback() {
        let api = APIClient()
        api.host = "http://192.168.1.10"
        XCTAssertEqual(api.serverURL(port: 11234, path: "/health").absoluteString,
                       "http://127.0.0.1:11234/health")
    }
}
