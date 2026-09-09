import XCTest
@testable import MLXCore

/// Tests for `APIClient.serverURL(port:path:)` — the centralised URL builder
/// that replaced the hardcoded `127.0.0.1` literals. A wrong host here means
/// the app silently fails to reach a server bound to a specific LAN IP.
final class APIClientServerURLTests: XCTestCase {

    // MARK: - Default (loopback)

    func testDefaultHostIsLoopback() {
        let api = APIClient()
        XCTAssertEqual(api.host, "127.0.0.1")
    }

    func testDefaultHostProducesLoopbackURL() {
        let api = APIClient()
        let url = api.serverURL(port: 11234, path: "/health")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:11234/health")
    }

    // MARK: - Wide bind (0.0.0.0) falls back to loopback

    func testWideBind0000FallsBackToLoopback() {
        let api = APIClient()
        api.host = "0.0.0.0"
        let url = api.serverURL(port: 11234, path: "/health")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:11234/health",
                       "0.0.0.0 should fall back to 127.0.0.1 to stay in the loopback trust boundary")
    }

    func testWideBindIPv6FallsBackToLoopback() {
        let api = APIClient()
        api.host = "::"
        let url = api.serverURL(port: 11234, path: "/v1/models")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:11234/v1/models",
                       ":: should fall back to 127.0.0.1")
    }

    func testEmptyHostFallsBackToLoopback() {
        let api = APIClient()
        api.host = ""
        let url = api.serverURL(port: 8080, path: "/props")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:8080/props",
                       "Empty host should fall back to 127.0.0.1")
    }

    // MARK: - Specific LAN IP

    func testSpecificLanIPIsUsedDirectly() {
        let api = APIClient()
        api.host = "192.168.1.10"
        let url = api.serverURL(port: 11234, path: "/health")
        XCTAssertEqual(url.absoluteString, "http://192.168.1.10:11234/health")
    }

    func testSpecificLanIPWithDifferentPort() {
        let api = APIClient()
        api.host = "10.0.0.5"
        let url = api.serverURL(port: 9999, path: "/v1/chat/completions")
        XCTAssertEqual(url.absoluteString, "http://10.0.0.5:9999/v1/chat/completions")
    }

    // MARK: - Explicit loopback

    func testExplicitLoopbackIsKept() {
        let api = APIClient()
        api.host = "127.0.0.1"
        let url = api.serverURL(port: 11234, path: "/metrics.json")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:11234/metrics.json")
    }

    // MARK: - Path variations

    func testRootPath() {
        let api = APIClient()
        let url = api.serverURL(port: 11234, path: "/")
        XCTAssertEqual(url.absoluteString, "http://127.0.0.1:11234/")
    }

    func testDeepPath() {
        let api = APIClient()
        api.host = "192.168.0.100"
        let url = api.serverURL(port: 11234, path: "/v1/audio/speech")
        XCTAssertEqual(url.absoluteString, "http://192.168.0.100:11234/v1/audio/speech")
    }
}
