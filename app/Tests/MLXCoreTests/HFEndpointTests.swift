import XCTest
@testable import MLXCore

/// `HF_ENDPOINT` points every hub fetch at a mirror, the same rule as the Zig CLI's `hubBase`.
final class HFEndpointTests: XCTestCase {
    func testUnsetOrBlankIsHuggingFace() {
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: [:]), "https://huggingface.co")
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: ["HF_ENDPOINT": "  \n"]), "https://huggingface.co")
    }

    func testAMirrorIsUsedWithoutItsTrailingSlash() {
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: ["HF_ENDPOINT": "https://hf-mirror.com"]),
                       "https://hf-mirror.com")
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: ["HF_ENDPOINT": " https://hf-mirror.com// "]),
                       "https://hf-mirror.com")
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: ["HF_ENDPOINT": "http://127.0.0.1:8080/hub/"]),
                       "http://127.0.0.1:8080/hub")
    }

    func testAValueThatIsNotAnHttpURLFallsBackToHuggingFace() {
        for bad in ["hf-mirror.com", "https://", "ftp://hf-mirror.com", "https:///hub", "http://:8080"] {
            XCTAssertEqual(DownloadManager.hfEndpoint(environment: ["HF_ENDPOINT": bad]), "https://huggingface.co", bad)
        }
    }

    func testTheLoginShellsValueIsSeen() {
        XCTAssertTrue(LoginShellEnv.huggingFaceNames.contains("HF_ENDPOINT"),
                      "a Finder-launched app only sees HF_ENDPOINT through the login-shell probe")
        let merged = LoginShellEnv.merge(shell: ["HF_ENDPOINT": "https://hf-mirror.com"], into: [:])
        XCTAssertEqual(DownloadManager.hfEndpoint(environment: merged), "https://hf-mirror.com")
    }

    func testSearchAndTheReadmeAskTheMirrorTheEnvironmentNames() throws {
        let saved = getenv("HF_ENDPOINT").map { String(cString: $0) }
        setenv("HF_ENDPOINT", "https://hf-mirror.com", 1)
        defer { if let saved { setenv("HF_ENDPOINT", saved, 1) } else { unsetenv("HF_ENDPOINT") } }
        XCTAssertEqual(try XCTUnwrap(HFSearchService.searchURL(query: "qwen", filter: "mlx", skip: 0, limit: 5)).host,
                       "hf-mirror.com")
        XCTAssertEqual(ModelCard.readmeURL(repoId: "org/repo")?.host, "hf-mirror.com")
    }

    func testTheReadmeFetchFollowsTheEndpoint() {
        XCTAssertEqual(ModelCard.readmeURL(repoId: "org/repo", endpoint: "https://hf-mirror.com")?.absoluteString,
                       "https://hf-mirror.com/org/repo/resolve/main/README.md")
    }

    /// The server the app starts runs `/api/pull` itself: it gets the hub settings the login shell names when the
    /// app's own environment (a Finder launch) has none, and the app's own value wins.
    func testTheServerTheAppStartsSeesTheShellsHubSettings() {
        let shell = ["HF_ENDPOINT": "https://hf-mirror.com", "HF_TOKEN": "t", "HF_HOME": "/elsewhere"]
        let env = ServerManager.launchEnvironment(process: ["PATH": "/usr/bin"], hub: shell)
        XCTAssertEqual(env["HF_ENDPOINT"], "https://hf-mirror.com")
        XCTAssertEqual(env["HF_TOKEN"], "t")
        XCTAssertNil(env["HF_HOME"], "only the download settings are handed on")
        let own = ServerManager.launchEnvironment(process: ["HF_ENDPOINT": "https://other.test"], hub: shell)
        XCTAssertEqual(own["HF_ENDPOINT"], "https://other.test")
    }

    /// A bad value is named once, not once per URL a download builds.
    func testABadEndpointIsReportedOncePerValue() {
        let value = "bad-endpoint-\(UUID().uuidString)"
        XCTAssertTrue(DownloadManager.firstWarning(forEndpoint: value))
        XCTAssertFalse(DownloadManager.firstWarning(forEndpoint: value))
        XCTAssertTrue(DownloadManager.firstWarning(forEndpoint: value + "-other"))
    }
}
