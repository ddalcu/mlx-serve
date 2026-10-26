import Combine
import XCTest
@testable import MLXCore

@MainActor
final class ModelDiscoverySnapshotTests: XCTestCase {
    private func fixture() throws -> (URL, UserDefaults) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "ModelDiscoverySnapshotTests.\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        addTeardownBlock {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
        return (root, defaults)
    }

    func testPollingReusesCompleteSnapshotIncludingAbsentLmStudio() throws {
        let (root, defaults) = try fixture()
        var probes = 0
        let manager = DownloadManager(modelsRoot: root.path, defaults: defaults,
            toolRootResolver: { probes += 1; return ToolModelRoots() },
            huggingFaceRootResolver: { nil })
        for _ in 0..<5 {
            XCTAssertNil(manager.scanInputs().lmStudioRoot)
            XCTAssertTrue(manager.scanInputs().toolRoots.isEmpty)
            _ = manager.scanRoots()
            _ = manager.readRoots
        }
        XCTAssertEqual(probes, 1)
        manager.reloadDiscoveryRoots()
        XCTAssertEqual(probes, 2)
    }

    func testExplicitReloadFindsMovedLmStudioFolder() throws {
        let (root, defaults) = try fixture()
        let settings = root.appendingPathComponent(".lmstudio/settings.json")
        try FileManager.default.createDirectory(at: settings.deletingLastPathComponent(), withIntermediateDirectories: true)
        func move(to name: String) throws -> String {
            let dir = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: ["downloadsFolder": dir.path]).write(to: settings)
            return dir.path
        }
        let first = try move(to: "first")
        let manager = DownloadManager(modelsRoot: root.path, defaults: defaults,
            toolRootResolver: { ToolModelRoots.detected(home: root.path) }, huggingFaceRootResolver: { nil })
        let second = try move(to: "second")
        XCTAssertEqual(manager.scanInputs().lmStudioRoot, first)
        manager.reloadDiscoveryRoots()
        XCTAssertEqual(manager.scanInputs().lmStudioRoot, second)
        XCTAssertTrue(manager.discoveryConfiguration.roots.contains(second))
    }

    func testDisabledDiscoveryNeverResolvesExternalRoots() throws {
        let (root, defaults) = try fixture()
        defaults.set(false, forKey: ModelRoots.automaticDiscoveryKey)
        let manager = DownloadManager(modelsRoot: root.path, defaults: defaults,
            toolRootResolver: { XCTFail("disabled discovery probed tools"); return ToolModelRoots() },
            huggingFaceRootResolver: { XCTFail("disabled discovery probed HF"); return nil })
        manager.reloadDiscoveryRoots()
        XCTAssertNil(manager.scanInputs().lmStudioRoot)
        XCTAssertNil(manager.scanInputs().huggingFaceRoot)
        XCTAssertTrue(manager.scanInputs().toolRoots.isEmpty)
    }

    func testRestartComparesStoredConfigurationOnly() {
        let server = ServerManager()
        server.status = .running
        server.lastLaunchedOptions = ServerOptions()
        let original = ModelDiscoveryConfiguration(automaticallyDiscoverModels: true, roots: ["/first"])
        server.discoveryConfiguration = original
        server.lastLaunchedDiscoveryConfiguration = original
        for _ in 0..<5 { XCTAssertFalse(server.needsRestartFor(ServerOptions())) }
        server.discoveryConfiguration = ModelDiscoveryConfiguration(automaticallyDiscoverModels: false, roots: ["/first"])
        XCTAssertTrue(server.needsRestartFor(ServerOptions()))
        server.discoveryConfiguration = ModelDiscoveryConfiguration(automaticallyDiscoverModels: true, roots: ["/second"])
        XCTAssertTrue(server.needsRestartFor(ServerOptions()))
        server.status = .stopped
        XCTAssertFalse(server.needsRestartFor(ServerOptions()))
        server.status = .error("stopped")
        XCTAssertFalse(server.needsRestartFor(ServerOptions()))
    }

    func testRestartTargetPreservesHeadlessAndSelectedModelPaths() {
        XCTAssertEqual(ServerRestartTarget.resolve(modelPath: "/model", modelsDir: "/library"), .model("/model"))
        XCTAssertEqual(ServerRestartTarget.resolve(modelPath: "", modelsDir: "/library"), .headless("/library"))
        XCTAssertEqual(ServerRestartTarget.resolve(modelPath: nil, modelsDir: "/library"), .headless("/library"))
    }

    func testRefreshProgressPublishesOnTheRefresher() async {
        let refresher = ModelLibraryRefresher()
        var states: [Bool] = []
        let observation = refresher.$isRefreshing.sink { states.append($0) }
        defer { observation.cancel() }
        let applied = expectation(description: "scan applied")
        let inputs = DownloadManager.LocalScanInputs(ownedRoots: [], lmStudioRoot: nil,
            huggingFaceRoot: nil, customRoot: nil, toolRoots: [], inFlightDirs: [])
        refresher.refresh(inputs: inputs, scan: { _ in [] }) { _ in applied.fulfill() }
        await fulfillment(of: [applied], timeout: 5)
        XCTAssertEqual(states, [false, true, false])
    }

    func testOlderLoginShellResolutionCannotUndoDisabledDiscovery() async throws {
        let (root, defaults) = try fixture()
        let started = expectation(description: "resolver started")
        let release = DispatchSemaphore(value: 0)
        let manager = DownloadManager(modelsRoot: root.path, defaults: defaults,
            toolRootResolver: { ToolModelRoots() }, huggingFaceRootResolver: { nil },
            loginShellRootResolver: { started.fulfill(); release.wait(); return "/old-cache" })
        let pending = Task { await manager.refreshHuggingFaceRootFromLoginShell() }
        await fulfillment(of: [started], timeout: 5)
        defaults.set(false, forKey: ModelRoots.automaticDiscoveryKey)
        manager.reloadDiscoveryRoots()
        release.signal()
        let changed = await pending.value
        XCTAssertFalse(changed)
        XCTAssertNil(manager.scanInputs().huggingFaceRoot)
    }
}
