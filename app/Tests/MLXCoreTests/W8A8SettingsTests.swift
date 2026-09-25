import XCTest
@testable import MLXCore

@MainActor
final class W8A8SettingsTests: XCTestCase {
    func testDefaultAndDisabledLaunchFlagsAreAuthoritative() {
        let options = ServerOptions()
        XCTAssertEqual(options.qwenImageW8A8, ServerOptions.qwenImageW8A8Default)
        XCTAssertFalse(options.toCLIArgs().contains("--w8a8"))
        XCTAssertFalse(options.toCLIArgs().contains("--no-w8a8"))
        var edited = options
        edited.qwenImageW8A8.toggle()
        if ServerOptions.qwenImageW8A8Default {
            XCTAssertTrue(edited.toCLIArgs().contains("--no-w8a8"))
            XCTAssertFalse(edited.toCLIArgs().contains("--w8a8"))
        } else {
            XCTAssertTrue(edited.toCLIArgs().contains("--w8a8"))
            XCTAssertFalse(edited.toCLIArgs().contains("--no-w8a8"))
        }
    }

    func testEffectiveSettingRequiresNaxAndHasNoQualityWarning() {
        var options = ServerOptions()
        XCTAssertEqual(options.effectiveW8A8(naxAvailable: true), ServerOptions.qwenImageW8A8Default)
        options.qwenImageW8A8 = true
        XCTAssertTrue(options.effectiveW8A8(naxAvailable: true))
        XCTAssertFalse(options.effectiveW8A8(naxAvailable: false))
        options.qwenImageW8A8 = false
        XCTAssertFalse(options.effectiveW8A8(naxAvailable: true))
        XCTAssertNil(ServerOptions.serverFlagFields["qwenImageW8A8"]?.cost)
    }

    func testPersistenceMigrationAndReset() throws {
        let legacy = try JSONDecoder().decode(ServerOptions.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.qwenImageW8A8, ServerOptions.qwenImageW8A8Default)
        var options = legacy
        options.qwenImageW8A8 = false
        let restored = try JSONDecoder().decode(ServerOptions.self, from: JSONEncoder().encode(options))
        XCTAssertFalse(restored.qwenImageW8A8)
        XCTAssertEqual(SettingsReset.apply(.neuralEngine, to: restored).qwenImageW8A8, ServerOptions.qwenImageW8A8Default)
        XCTAssertFalse(SettingsReset.apply(.voice, to: restored).qwenImageW8A8)
    }

    func testChangeRequiresRestart() {
        let launched = ServerOptions()
        var edited = launched
        edited.qwenImageW8A8.toggle()
        XCTAssertFalse(edited.serverLaunchEquals(launched))
        XCTAssertTrue(ServerOptions.serverFlagFields["qwenImageW8A8"]?.needsRestart ?? false)
        let server = ServerManager()
        server.status = .running
        server.lastLaunchedOptions = launched
        XCTAssertTrue(server.needsRestartFor(edited))
        server.status = .stopped
        XCTAssertFalse(server.needsRestartFor(edited))
    }

    func testNaxAvailabilityUsesTheServerCapabilityReport() {
        XCTAssertTrue(EngineVersions.naxAvailable(in: EngineVersions.parse("nax on (M5 neural accelerators)")))
        XCTAssertFalse(EngineVersions.naxAvailable(in: EngineVersions.parse("nax off (macOS < 26.2)")))
        XCTAssertFalse(EngineVersions.naxAvailable(in: EngineVersions.parse("nax off (GPU has no NAX)")))
        XCTAssertFalse(EngineVersions.naxAvailable(in: EngineVersions.parse("nax online")))
        XCTAssertFalse(EngineVersions.naxAvailable(in: []))
    }

    func testRestartSupportsMediaOnlyAndChatInstalls() {
        let server = LaunchSpy()
        var options = ServerOptions()
        options.qwenImageW8A8 = true
        server.restart(modelPath: "", options: options)
        XCTAssertEqual(server.stopCount, 1)
        XCTAssertEqual(server.headlessOptions, options)
        XCTAssertNil(server.chatPath)
        server.restart(modelPath: "/models/chat", options: options)
        XCTAssertEqual(server.stopCount, 2)
        XCTAssertEqual(server.chatPath, "/models/chat")
        XCTAssertEqual(server.chatOptions, options)
    }

    func testMediaFirstLaunchUsesCurrentRatherThanStaleOptions() async throws {
        let server = LaunchSpy()
        server.lastLaunchedOptions = ServerOptions()
        var edited = ServerOptions()
        edited.qwenImageW8A8 = false
        let port = try await server.ensureRunning(forGenModelDir: "/models/image", options: edited)
        XCTAssertEqual(port, edited.port)
        XCTAssertEqual(server.headlessOptions, edited)
        server.headlessOptions = nil
        _ = try await server.ensureRunning(forGenModelDir: "/models/image", options: ServerOptions())
        XCTAssertNil(server.headlessOptions, "A running server changes only through explicit restart")
    }
}

@MainActor
private final class LaunchSpy: ServerManager {
    var stopCount = 0
    var headlessOptions: ServerOptions?
    var chatOptions: ServerOptions?
    var chatPath: String?

    override func stop() {
        stopCount += 1
        status = .stopped
    }

    override func startHeadless(modelsDir: String, options: ServerOptions) {
        headlessOptions = options
        port = options.port
        status = .running
    }

    override func start(modelPath: String, options: ServerOptions) {
        chatPath = modelPath
        chatOptions = options
        status = .running
    }
}
