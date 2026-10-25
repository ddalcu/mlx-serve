import XCTest
@testable import MLXCore

final class ModelDiscoveryPolicyTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("model-policy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    private func model(_ relative: String) throws -> URL {
        let dir = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private func inputs(owned: [String] = [], custom: String? = nil, lmStudio: String? = nil) -> DownloadManager.LocalScanInputs {
        .init(ownedRoots: owned, lmStudioRoot: lmStudio, huggingFaceRoot: nil,
              customRoot: custom, toolRoots: [], inFlightDirs: [])
    }

    func testAliasesCollapseToOwnedThenCustomWithoutCollapsingSeparateCopies() throws {
        let owned = root.appendingPathComponent("owned")
        let custom = root.appendingPathComponent("custom")
        let alias = root.appendingPathComponent("alias")
        _ = try model("owned/acme/demo")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: owned)
        var found = DownloadManager.discoverLocalModels(inputs(owned: [owned.path], custom: alias.path, lmStudio: owned.path))
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.source, .mlxServe)
        found = DownloadManager.discoverLocalModels(inputs(custom: alias.path, lmStudio: owned.path))
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.source, .custom)
        _ = try model("custom/acme/demo")
        found = DownloadManager.discoverLocalModels(inputs(owned: [owned.path], custom: custom.path))
        XCTAssertEqual(found.count, 2)
        XCTAssertEqual(Set(found.map(\.path)).count, 2)
    }

    func testGgufQuantsStayDistinctWhenTheirRootIsRepeated() throws {
        let dir = root.appendingPathComponent("models/acme/demo")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in ["demo-Q4_K_M.gguf", "demo-Q8_0.gguf"] {
            try Data(repeating: 0, count: 2 * 1024 * 1024).write(to: dir.appendingPathComponent(file))
        }
        let folder = root.appendingPathComponent("models").path
        let found = DownloadManager.discoverLocalModels(inputs(owned: [folder], custom: folder, lmStudio: folder))
        XCTAssertEqual(found.count, 2)
        XCTAssertEqual(Set(found.map(\.path)).count, 2)
    }

    @MainActor
    func testDisabledDiscoveryCapturesOnlyConfiguredSourcesAndKeepsCustom() throws {
        let suite = "ModelDiscoveryPolicyTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let owned = root.appendingPathComponent("owned")
        let custom = root.appendingPathComponent("custom")
        _ = try model("owned/acme/downloaded")
        _ = try model("custom/acme/extra")
        defaults.set(false, forKey: ModelRoots.automaticDiscoveryKey)
        let manager = DownloadManager(modelsRoot: owned.path, defaults: defaults)
        manager.customRoot = custom.path
        let captured = manager.scanInputs()
        XCTAssertNil(captured.lmStudioRoot)
        XCTAssertNil(captured.huggingFaceRoot)
        XCTAssertTrue(captured.toolRoots.isEmpty)
        XCTAssertEqual(captured.customRoot, custom.path)
        let found = DownloadManager.discoverLocalModels(captured)
        XCTAssertEqual(Set(found.map(\.name)), ["acme/downloaded", "acme/extra"])
    }

    func testLmStudioFolderResolverReReadsMovedSettings() throws {
        let home = root.appendingPathComponent("home")
        let settings = home.appendingPathComponent(".lmstudio/settings.json")
        try FileManager.default.createDirectory(at: settings.deletingLastPathComponent(), withIntermediateDirectories: true)
        for folder in ["first", "second"] {
            let dir = root.appendingPathComponent(folder)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: ["downloadsFolder": dir.path]).write(to: settings)
            XCTAssertEqual(DownloadManager.lmStudioRootPath(home: home.path), dir.path)
        }
    }

    @MainActor
    func testRunningServerRequiresRestartWhenDiscoveryPolicyChanges() throws {
        let suite = "ModelDiscoveryPolicyTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let manager = DownloadManager(modelsRoot: root.path, defaults: defaults,
            toolRootResolver: { ToolModelRoots() }, huggingFaceRootResolver: { nil })
        let server = ServerManager()
        server.status = .running
        server.lastLaunchedOptions = ServerOptions()
        server.discoveryConfiguration = manager.discoveryConfiguration
        server.lastLaunchedDiscoveryConfiguration = manager.discoveryConfiguration
        XCTAssertFalse(server.needsRestartFor(ServerOptions()))
        defaults.set(false, forKey: ModelRoots.automaticDiscoveryKey)
        manager.reloadDiscoveryRoots()
        server.discoveryConfiguration = manager.discoveryConfiguration
        XCTAssertTrue(server.needsRestartFor(ServerOptions()))
        server.status = .stopped
        XCTAssertFalse(server.needsRestartFor(ServerOptions()))
    }
}
