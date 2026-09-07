import XCTest
@testable import MLXCore

/// Providers, app side: the file codec the server reads, the row validation
/// that mirrors the server's, and provider rows landing in their own picker
/// group rather than under "On Your Network".
final class ProvidersConfigTests: XCTestCase {

    func testFileRoundTripsInTheServersKeys() throws {
        let entries = [
            ProviderEntry(name: "openai", url: "https://api.openai.com/v1", apiKeyEnv: "OPENAI_API_KEY"),
            ProviderEntry(name: "box", url: "http://192.168.1.20:8080/v1", apiKey: "k", enabled: false, models: ["llama", "phi"]),
        ]
        let data = try ProvidersFile.encode(entries)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"api_key_env\" : \"OPENAI_API_KEY\""))
        XCTAssertTrue(text.contains("\"enabled\" : false"))
        XCTAssertFalse(text.contains("\"api_key\" : \"\""), "empty keys are omitted, not written as empty strings")
        XCTAssertTrue(text.contains("https://api.openai.com/v1"), "slashes are not escaped")
        let back = ProvidersFile.decode(data)
        XCTAssertEqual(back.map(\.name), ["openai", "box"])
        XCTAssertEqual(back[1].models, ["llama", "phi"])
        XCTAssertFalse(back[1].enabled)
        XCTAssertTrue(back[0].enabled, "absent enabled reads as true")
    }

    func testDecodeAcceptsTheWrapperFormAndToleratesMissingKeys() {
        let wrapped = Data("{\"providers\":[{\"name\":\"a\",\"url\":\"http://a/v1\"}]}".utf8)
        XCTAssertEqual(ProvidersFile.decode(wrapped).map(\.name), ["a"])
        XCTAssertEqual(ProvidersFile.decode(Data("garbage".utf8)), [])
    }

    func testProblemMirrorsTheServersSkipRules() {
        XCTAssertNil(ProviderEntry(name: "open-ai_2.x", url: "https://x/v1").problem())
        XCTAssertNotNil(ProviderEntry(name: "", url: "https://x/v1").problem())
        XCTAssertNotNil(ProviderEntry(name: "bad name", url: "https://x/v1").problem())
        XCTAssertNotNil(ProviderEntry(name: "a@b", url: "https://x/v1").problem(), "@ is the routing delimiter")
        XCTAssertNotNil(ProviderEntry(name: "ok", url: "ftp://x/v1").problem())
        XCTAssertNotNil(ProviderEntry(name: "ok", url: "").problem())
        // Our own server is refused; another mlx-serve on a different port is fine.
        XCTAssertNotNil(ProviderEntry(name: "me", url: "http://localhost:11234").problem(serverPort: 11234))
        XCTAssertNotNil(ProviderEntry(name: "me", url: "http://127.0.0.1:11234/v1").problem(serverPort: 11234))
        XCTAssertNil(ProviderEntry(name: "other", url: "http://localhost:11235/v1").problem(serverPort: 11234))
        XCTAssertNil(ProviderEntry(name: "lan", url: "http://192.168.1.20:11234/v1").problem(serverPort: 11234))
        XCTAssertEqual(ProvidersFile.duplicateNames([
            ProviderEntry(name: "a", url: "http://a/v1"), ProviderEntry(name: "a", url: "http://b/v1"), ProviderEntry(name: "c", url: "http://c/v1"),
        ]), ["a"])
        XCTAssertEqual(ProviderEntry.parseModelList("gpt-5, gpt-5-mini\n\n o3 "), ["gpt-5", "gpt-5-mini", "o3"])
    }

    func testStatusDecodes() {
        let data = Data("{\"providers\":[{\"name\":\"openai\",\"url\":\"https://x/v1\",\"up\":true,\"probed\":true,\"models\":12}]}".utf8)
        XCTAssertEqual(ProviderStatus.decodeList(data), [ProviderStatus(name: "openai", url: "https://x/v1", up: true, probed: true, models: 12)])
    }

    /// A provider row is remote like a LAN row (same `<id>@<host>` routing, no
    /// local residency) but lists under its own heading.
    @MainActor
    func testProviderRowsAreRemoteAndGroupUnderProviders() {
        let info = APIClient.parseModelInfo([
            "id": "gpt-5@openai", "provider": "openai", "capabilities": ["chat"], "loaded": true,
        ])
        // The tray's In Memory list is what THIS Mac holds: 431 provider rows
        // once listed there, each with an eject button that could do nothing.
        let sm = ServerManager()
        sm.allModels = [info, APIClient.parseModelInfo(["id": "gemma@Studio", "lan_peer": "Studio", "loaded": true]),
                        APIClient.parseModelInfo(["id": "local", "loaded": true]), APIClient.parseModelInfo(["id": "stub", "loaded": false])]
        XCTAssertEqual(sm.residentModels.map(\.name), ["local"])
        XCTAssertEqual(info.provider, "openai")
        XCTAssertEqual(info.lanPeer, "openai", "routing surfaces treat it as a remote host")
        XCTAssertTrue(info.lanAdvertises("chat"))
        XCTAssertEqual(info.lanDisplayName, "gpt-5 · openai")

        let lan = APIClient.parseModelInfo(["id": "gemma@Studio", "lan_peer": "Studio", "capabilities": ["chat"]])
        XCTAssertNil(lan.provider)
        XCTAssertEqual(ModelPalette.remoteSection(for: lan), ModelPalette.networkSection)
        XCTAssertEqual(ModelPalette.remoteSection(for: info), ModelPalette.providersSection)
        let rows = ModelPalette.rows(local: [], lan: [lan, info])
        XCTAssertEqual(rows.first { $0.tag.hasSuffix("gpt-5@openai") }?.section, ModelPalette.providersSection)
        XCTAssertEqual(rows.first { $0.tag.hasSuffix("gemma@Studio") }?.section, ModelPalette.networkSection)
    }

    func testModelPickerReadsTheProvidersOwnList() {
        let body = Data(#"{"data":[{"id":"gpt-5"},{"id":""},{"object":"model"},{"id":"o3"}]}"#.utf8)
        XCTAssertEqual(ProviderEntry.modelIds(fromModelsBody: body), ["gpt-5", "o3"])
        XCTAssertNil(ProviderEntry.modelIds(fromModelsBody: Data("<html>".utf8)))
        XCTAssertEqual(ProviderEntry.modelsURLs(for: "https://openrouter.ai/api/v1/").map(\.absoluteString),
                       ["https://openrouter.ai/api/v1/models"])
        XCTAssertEqual(ProviderEntry.modelsURLs(for: "http://box:1234").map(\.absoluteString),
                       ["http://box:1234/models", "http://box:1234/v1/models"])
    }
}
