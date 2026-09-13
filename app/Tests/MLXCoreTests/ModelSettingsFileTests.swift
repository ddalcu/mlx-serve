import XCTest
@testable import MLXCore

/// `~/.mlx-serve/model-settings.json` is the SERVER's file (`src/model_settings.zig`);
/// the app only edits it, so it must write exactly the server's keys and never
/// drop a key it does not know.
final class ModelSettingsFileTests: XCTestCase {

    private func tempPath() -> String {
        NSTemporaryDirectory() + "model-settings-\(UUID().uuidString)/model-settings.json"
    }

    func testRoundTripKeepsOnlySetFieldsAndTrimsTheSlash() throws {
        let path = tempPath()
        var file = ModelSettingsFile.load(path: path)
        XCTAssertTrue(file.isEmpty)
        file.set(ModelOverride(ctxSize: 65_536, kvQuant: .bits8, mtp: false), for: "/m/a/")
        try file.save(path: path)

        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"ctx_size\" : 65536"), text)
        XCTAssertTrue(text.contains("\"kv_quant\" : \"8\""), text)
        XCTAssertTrue(text.contains("\"mtp\" : false"), text)
        XCTAssertTrue(text.contains("\"/m/a\""), "trailing slash must be trimmed: \(text)")

        let back = ModelSettingsFile.load(path: path)
        XCTAssertEqual(back.override(for: "/m/a"), ModelOverride(ctxSize: 65_536, kvQuant: .bits8, mtp: false))
        XCTAssertEqual(back.override(for: "/m/a/"), back.override(for: "/m/a"))
        XCTAssertNil(back.override(for: "/m/b"))
    }

    func testAnEmptyOverrideRemovesTheEntry() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        file.set(ModelOverride(ctxSize: 4096), for: "/m/a")
        file.set(ModelOverride(), for: "/m/a")
        try file.save(path: path)
        XCTAssertTrue(ModelSettingsFile.load(path: path).isEmpty)
    }

    /// A future server field must survive an app-side edit of another model.
    func testUnknownKeysSurviveASave() throws {
        let path = tempPath()
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try """
        {"/m/a": {"ctx_size": 8192, "prefix_cache_mem": 123}, "/m/b": {"mtp": true}}
        """.write(toFile: path, atomically: true, encoding: .utf8)
        var file = ModelSettingsFile.load(path: path)
        XCTAssertEqual(file.override(for: "/m/a")?.ctxSize, 8192)
        XCTAssertEqual(file.override(for: "/m/b")?.mtp, true)
        file.set(ModelOverride(ctxSize: 8192, kvQuant: .off), for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"prefix_cache_mem\" : 123"), text)
        XCTAssertTrue(text.contains("\"kv_quant\" : \"off\""), text)
        XCTAssertEqual(ModelSettingsFile.load(path: path).override(for: "/m/b")?.mtp, true)
    }

    func testAMalformedFileLoadsEmptyAndABadValueIsIgnored() throws {
        let path = tempPath()
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try "{nope".write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertTrue(ModelSettingsFile.load(path: path).isEmpty)
        try #"{"/m/a": {"kv_quant": "16", "ctx_size": "big"}}"#.write(toFile: path, atomically: true, encoding: .utf8)
        XCTAssertNil(ModelSettingsFile.load(path: path).override(for: "/m/a"))
    }

    /// The overflow card's button goes to the per-model sheet only when that
    /// model already has overrides; otherwise Settings, as before.
    func testContextIncreaseTarget() {
        XCTAssertEqual(ContextIncreaseTarget.resolve(hasOverride: true), .modelSettings)
        XCTAssertEqual(ContextIncreaseTarget.resolve(hasOverride: false), .appSettings)
    }
}

/// The gauge and the turn budget read the context the SERVER advertises for
/// the live model; it already reflects `--ctx-size` and, now, the per-model
/// override. The slider is only the answer while no server has reported one.
final class EffectiveContextLengthTests: XCTestCase {
    @MainActor func testTheLiveModelOutranksTheGlobalSlider() {
        XCTAssertEqual(AgentEngine.effectiveContextLength(appContextSize: 16_384, modelContextLength: 65_536), 65_536)
        XCTAssertEqual(AgentEngine.effectiveContextLength(appContextSize: 16_384, modelContextLength: nil), 16_384)
        XCTAssertEqual(AgentEngine.effectiveContextLength(appContextSize: 0, modelContextLength: 0), 32_768)
    }
}
