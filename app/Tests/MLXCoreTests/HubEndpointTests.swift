import XCTest
@testable import MLXCore

/// `HF_ENDPOINT` names the hub every download uses (a mirror), and the token goes to that hub only (#952).
final class HubEndpointTests: XCTestCase {
    func testTheEndpointNamesTheHubAndAnythingThatIsNotAURLIsIgnored() {
        XCTAssertEqual(DownloadManager.hubBase(environment: [:]), "https://huggingface.co")
        XCTAssertEqual(DownloadManager.hubBase(environment: ["HF_ENDPOINT": "https://hf-mirror.com"]), "https://hf-mirror.com")
        XCTAssertEqual(DownloadManager.hubBase(environment: ["HF_ENDPOINT": " https://hf-mirror.com/ \n"]), "https://hf-mirror.com")
        XCTAssertEqual(DownloadManager.hubBase(environment: ["HF_ENDPOINT": "http://127.0.0.1:8123/hub//"]), "http://127.0.0.1:8123/hub")
        for bad in ["", "  ", "hf-mirror.com", "ftp://hf-mirror.com", "https://", "file:///etc"] {
            XCTAssertEqual(DownloadManager.hubBase(environment: ["HF_ENDPOINT": bad]), "https://huggingface.co", bad)
        }
    }

    func testEveryHubPathIsBuiltOnTheEndpoint() {
        let mirror = ["HF_ENDPOINT": "https://hf-mirror.com"]
        XCTAssertEqual(DownloadManager.hubURL("/api/models/org/repo/tree/main?recursive=true", environment: mirror)?.absoluteString,
                       "https://hf-mirror.com/api/models/org/repo/tree/main?recursive=true")
        XCTAssertEqual(DownloadManager.hubURL("/org/repo/resolve/main/config.json", environment: [:])?.absoluteString,
                       "https://huggingface.co/org/repo/resolve/main/config.json")
    }

    func testTheTokenGoesToTheHubAndNowhereElse() {
        let env = ["HF_TOKEN": "secret", "HF_ENDPOINT": "https://hf-mirror.com"]
        let auth = ["Authorization": "Bearer secret"]
        XCTAssertEqual(DownloadManager.hfHeaders(for: URL(string: "https://hf-mirror.com/org/repo/resolve/main/a")!, environment: env), auth)
        XCTAssertEqual(DownloadManager.hfHeaders(for: URL(string: "https://huggingface.co/org/repo/resolve/main/a")!, environment: env), [:])
        XCTAssertEqual(DownloadManager.hfHeaders(for: URL(string: "https://cdn.example.com/blob")!, environment: env), [:])
        XCTAssertEqual(DownloadManager.hfHeaders(for: URL(string: "http://hf-mirror.com:8080/a")!, environment: env), [:])
        XCTAssertEqual(DownloadManager.hfHeaders(for: URL(string: "https://huggingface.co/api/models/x")!, environment: ["HF_TOKEN": "secret"]), auth)
    }

    func testTheLoginShellIsAskedForTheEndpoint() {
        XCTAssertTrue(LoginShellEnv.huggingFaceNames.contains("HF_ENDPOINT"))
    }

    /// Settings ▸ Server ▸ Download models from (#1306) names the hub for the app and the server it launches,
    /// over the shell's `HF_ENDPOINT`; left empty, the shell's value (or huggingface.co) stays.
    func testTheSettingNamesTheHubForTheAppAndTheServer() {
        let shell = ["HF_ENDPOINT": "https://shell-mirror.example", "HF_TOKEN": "secret"]
        let env = LoginShellEnv.huggingFaceEnvironment(process: shell, hubSetting: " https://mirror.corp.example/ ")
        XCTAssertEqual(DownloadManager.hubBase(environment: env), "https://mirror.corp.example")
        XCTAssertEqual(env["HF_TOKEN"], "secret")
        let launched = ServerManager.launchEnvironment(process: shell, hubSetting: "https://mirror.corp.example")
        XCTAssertEqual(launched["HF_ENDPOINT"], "https://mirror.corp.example")
        for unset in ["", "   "] {
            XCTAssertEqual(LoginShellEnv.huggingFaceEnvironment(process: shell, hubSetting: unset)["HF_ENDPOINT"], "https://shell-mirror.example")
            XCTAssertEqual(ServerManager.launchEnvironment(process: shell, hubSetting: unset)["HF_ENDPOINT"], "https://shell-mirror.example")
        }
        XCTAssertNil(ServerManager.launchEnvironment(process: [:], hubSetting: "")["HF_ENDPOINT"])
    }

    /// The field says when what was typed is not a hub, rather than quietly downloading from huggingface.co.
    func testTheFieldRefusesWhatIsNotAHub() {
        for good in ["", "https://hf-mirror.com/", "http://127.0.0.1:8123"] {
            XCTAssertTrue(HubEndpointSetting.isUsable(good), good)
        }
        for bad in ["hf-mirror.com", "ftp://hf-mirror.com", "https://"] {
            XCTAssertFalse(HubEndpointSetting.isUsable(bad), bad)
        }
    }
}
