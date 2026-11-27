import XCTest
@testable import MLXCore

/// `~/.mlx-serve/model-settings.json` is the SERVER's file (`src/host/registry/model_settings.zig`);
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

    func testMtpAcceptanceRoundTripsByName() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        file.set(ModelOverride(mtp: true, mtpAcceptance: .typical), for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"mtp_acceptance\" : \"typical\""), text)
        XCTAssertEqual(ModelSettingsFile.load(path: path).override(for: "/m/a")?.mtpAcceptance, .typical)
        XCTAssertEqual(ModelOverride(json: ["mtp_acceptance": "fast"]).mtpAcceptance, nil)
    }

    func testMtpGreedyTailRoundTripsAsABoolean() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        file.set(ModelOverride(mtpGreedyTail: true), for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"mtp_greedy_tail\" : true"), text)
        XCTAssertEqual(ModelSettingsFile.load(path: path).override(for: "/m/a")?.mtpGreedyTail, true)
        XCTAssertNil(ModelOverride(json: ["mtp_greedy_tail": "yes"]).mtpGreedyTail)
    }

    func testInt8PrefillRoundTripsAsABoolean() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        file.set(ModelOverride(int8Prefill: true), for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"int8_prefill\" : true"), text)
        XCTAssertEqual(ModelSettingsFile.load(path: path).override(for: "/m/a")?.int8Prefill, true)
        XCTAssertNil(ModelOverride(json: ["int8_prefill": "yes"]).int8Prefill)
    }

    func testAneRoundTripsAsABoolean() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        var o = ModelOverride()
        o.ane = true
        file.set(o, for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"ane\" : true"), text)
        XCTAssertEqual(ModelSettingsFile.load(path: path).override(for: "/m/a")?.ane, true)
        XCTAssertNil(ModelOverride(json: ["ane": "yes"]).ane)
        XCTAssertTrue(o.changesLoad(from: ModelOverride()), "the engine is chosen at load")
    }

    func testTheEnginePickerWritesEngineAndRetiresTheAneSwitch() throws {
        var o = ModelOverride(json: ["ane": true])
        XCTAssertEqual(o.engineChoice, "ane", "an older file's switch reads as the engine")
        XCTAssertEqual(ModelOverride(json: ["ane": false]).engineChoice, "mlx")
        o.engineChoice = "mlx"
        XCTAssertEqual(o.json["engine"] as? String, "mlx")
        XCTAssertNil(o.json["ane"], "one key decides")
        o.engineChoice = nil
        XCTAssertTrue(o.isEmpty)
        XCTAssertEqual(ModelOverride(json: ["engine": "llama"]).engineChoice, "llama")
        XCTAssertTrue(ModelOverride(json: ["engine": "ds4"]).changesLoad(from: ModelOverride()))
    }

    /// `chat_template_kwargs` round-trips as one object: the rows the sheet
    /// edits are exactly what the server reads, typed values kept.
    func testChatTemplateKwargsRoundTripAsOneObject() throws {
        let path = tempPath()
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                                withIntermediateDirectories: true)
        try """
        {"/m/a": {"chat_template_kwargs": {"preserve_thinking": true, "other": 1}}}
        """.write(toFile: path, atomically: true, encoding: .utf8)
        var file = ModelSettingsFile.load(path: path)
        var o = file.override(for: "/m/a")!
        XCTAssertEqual(o.templateKwargs["preserve_thinking"] as? Bool, true)
        XCTAssertTrue(o.hasSettings)

        o.templateKwargs["preserve_thinking"] = false
        o.templateKwargs["other"] = nil
        file.set(o, for: "/m/a")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"preserve_thinking\" : false"), text)
        XCTAssertFalse(text.contains("other"), text)

        o.templateKwargs = [:]
        file.set(o, for: "/m/a")
        try file.save(path: path)
        XCTAssertTrue(ModelSettingsFile.load(path: path).isEmpty)
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

    /// `alias` is the server's request name for the model: it round-trips,
    /// an alias alone keeps the entry, and a taken or malformed name is refused.
    func testAliasRoundTripsAndIsValidated() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        var o = ModelOverride()
        o.alias = "qwen"
        XCTAssertTrue(o.hasSettings)
        file.set(o, for: "/m/a/")
        try file.save(path: path)
        let text = try String(contentsOfFile: path, encoding: .utf8)
        XCTAssertTrue(text.contains("\"alias\" : \"qwen\""), text)
        let back = ModelSettingsFile.load(path: path)
        XCTAssertEqual(back.override(for: "/m/a")?.alias, "qwen")
        XCTAssertEqual(back.pathUsingAlias("qwen", except: "/m/b"), "/m/a")
        XCTAssertNil(back.pathUsingAlias("qwen", except: "/m/a/"))

        for bad in ["", "mlx-serve", "x@peer", "org/x", "two words", "a\"b"] {
            XCTAssertFalse(ModelOverride.isValidAlias(bad), bad)
        }
        XCTAssertTrue(ModelOverride.isValidAlias("qwen3.6-27b:4bit"))
    }

    /// The server reads the alias per request, so an alias-only edit never
    /// reloads the model.
    func testAnAliasOnlyEditNeedsNoReload() {
        var before = ModelOverride(ctxSize: 8192)
        var after = before
        after.alias = "q"
        XCTAssertFalse(after.changesLoad(from: before))
        after.ctxSize = 4096
        XCTAssertTrue(after.changesLoad(from: before))
        before.extra["future"] = 1
        XCTAssertTrue(ModelOverride(ctxSize: 8192).changesLoad(from: before))
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

/// The per-model `config_overrides` box: the server's own deep-merge grammar (`--config-overrides`),
/// written as a JSON OBJECT and held as the text the user typed.
final class ConfigOverridesFieldTests: XCTestCase {
    private func tempPath() -> String {
        NSTemporaryDirectory() + "model-settings-\(UUID().uuidString)/model-settings.json"
    }

    private let yarn = "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4},\"max_position_embeddings\":1048576}}"

    func testTheFieldAcceptsOnlyANonEmptyJSONObject() {
        XCTAssertNotNil(ConfigOverrides.object(yarn))
        XCTAssertNil(ConfigOverrides.object(""))
        XCTAssertNil(ConfigOverrides.object("   "))
        XCTAssertNil(ConfigOverrides.object("{}"), "an object that names nothing overrides nothing")
        XCTAssertNil(ConfigOverrides.object("[1,2]"))
        XCTAssertNil(ConfigOverrides.object("\"yarn\""))
        XCTAssertNil(ConfigOverrides.object("{oops"))
        XCTAssertEqual(ConfigOverrides.text([:]), "")
    }

    func testTheFieldKeepsTheStoredShape() {
        guard let o = ConfigOverrides.object(yarn),
              let tc = o["text_config"] as? [String: Any],
              let rp = tc["rope_parameters"] as? [String: Any] else { return XCTFail("the nested block did not survive") }
        XCTAssertEqual(rp["rope_type"] as? String, "yarn")
        XCTAssertEqual(tc["max_position_embeddings"] as? Int, 1_048_576)
        // The text is canonical, so two entries that mean the same compare equal.
        XCTAssertEqual(ConfigOverrides.text(o), ConfigOverrides.text(ConfigOverrides.object(ConfigOverrides.text(o))!))
    }

    func testTheKeyRoundTripsAsAnObjectAndMarksTheLoadChanged() throws {
        let path = tempPath()
        var file = ModelSettingsFile()
        file.set(ModelOverride(ctxSize: 1_048_576, configOverrides: yarn), for: "/m/a")
        try file.save(path: path)
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let entry = root["/m/a"] as? [String: Any] else { return XCTFail("the file is not an object of objects") }
        XCTAssertTrue(entry["config_overrides"] is [String: Any], "the server reads an object, not a string: \(entry)")

        let back = try XCTUnwrap(ModelSettingsFile.load(path: path).override(for: "/m/a"))
        XCTAssertEqual(ConfigOverrides.object(back.configOverrides ?? "")?.count, 1)
        XCTAssertTrue(back.changesLoad(from: ModelOverride(ctxSize: 1_048_576)))
        XCTAssertFalse(back.changesLoad(from: ModelOverride(ctxSize: 1_048_576, configOverrides: ConfigOverrides.text(ConfigOverrides.object(yarn)!))))
    }

    func testAValueThatIsNotAnObjectIsLeftAlone() {
        let stored = ModelOverride(json: ["config_overrides": "yarn"])
        XCTAssertNil(stored.configOverrides)
        XCTAssertEqual(stored.extra["config_overrides"] as? String, "yarn", "a shape the app does not know still survives the save")
    }

    /// An empty object names nothing, so it clears the key: the box must not refuse it and freeze the save.
    func testTheBoxAcceptsAnyObjectAndOnlyStoresANonEmptyOne() {
        XCTAssertTrue(ConfigOverrides.isObject("{}"))
        XCTAssertNil(ConfigOverrides.object("{}"))
        XCTAssertTrue(ConfigOverrides.isObject(yarn))
        XCTAssertFalse(ConfigOverrides.isObject("\"yarn\""))
        XCTAssertFalse(ConfigOverrides.isObject("[1,2]"))
        XCTAssertFalse(ConfigOverrides.isObject("{oops"))
        XCTAssertFalse(ConfigOverrides.isObject("   "))
    }
}

/// The box's edit decision: what the typed text does to the stored key, and when it blocks the save.
final class ConfigOverridesEditTests: XCTestCase {
    /// Already in the canonical order `ConfigOverrides.text` writes, so a `.set` compares by text.
    private let yarn = "{\"text_config\":{\"rope_parameters\":{\"factor\":4,\"rope_type\":\"yarn\"}}}"

    func testEmptyTextAndAnEmptyObjectBothClearTheKey() {
        XCTAssertEqual(ConfigOverrides.edit(""), .clear)
        XCTAssertEqual(ConfigOverrides.edit("  \n "), .clear)
        XCTAssertEqual(ConfigOverrides.edit("{}"), .clear)
        XCTAssertNil(ConfigOverrides.rejection(""))
        XCTAssertNil(ConfigOverrides.rejection("{}"), "a no-op object is not a mistake")
    }

    func testValidTextStoresAndHalfTypedTextOnlyBlocks() {
        XCTAssertEqual(ConfigOverrides.edit(yarn), .set(yarn))
        XCTAssertNil(ConfigOverrides.rejection(yarn))
        for bad in ["{oops", "\"yarn\"", "[1,2]", "4"] {
            XCTAssertEqual(ConfigOverrides.edit(bad), .keep, bad)
            XCTAssertNotNil(ConfigOverrides.rejection(bad), bad)
        }
    }
}

/// The one-click YaRN recipe: only the arch that scales this way, and only numbers the pack itself states.
final class YarnTemplateTests: XCTestCase {
    func testOfferedOnlyForTheArchAndOnlyWhenThereIsRoom() throws {
        XCTAssertNil(YarnTemplate.plan(modelType: nil, native: 262_144))
        XCTAssertNil(YarnTemplate.plan(modelType: "qwen3_5", native: 262_144))
        XCTAssertNil(YarnTemplate.plan(modelType: "qwen4_exp", native: 0), "no window read is no recipe")
        XCTAssertNil(YarnTemplate.plan(modelType: "qwen4_exp", native: 1_048_576), "already at the target")
        let four = try XCTUnwrap(YarnTemplate.plan(modelType: "qwen4_exp", native: 262_144))
        XCTAssertEqual(four.factor, 4)
        XCTAssertEqual(four.ctxSize, 1_048_576)
        let two = try XCTUnwrap(YarnTemplate.plan(modelType: "qwen4_exp_text", native: 524_288))
        XCTAssertEqual(two.factor, 2)
        XCTAssertEqual(two.ctxSize, 1_048_576)
    }

    func testTheDocumentNamesThePacksOwnWindowAndNothingElse() throws {
        let plan = try XCTUnwrap(YarnTemplate.plan(modelType: "qwen4_exp", native: 262_144))
        let data = try XCTUnwrap(plan.json.data(using: .utf8))
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let tc = try XCTUnwrap(root["text_config"] as? [String: Any])
        let rp = try XCTUnwrap(tc["rope_parameters"] as? [String: Any])
        XCTAssertEqual(rp["rope_type"] as? String, "yarn")
        XCTAssertEqual(rp["factor"] as? Int, 4)
        XCTAssertEqual(rp["original_max_position_embeddings"] as? Int, 262_144)
        XCTAssertEqual(tc["max_position_embeddings"] as? Int, 1_048_576)
        // What the sheet writes is what the server accepts.
        XCTAssertNotNil(ConfigOverrides.object(plan.json))
    }

    func testNativeWindowReadsTextConfigFirstThenTheRoot() {
        XCTAssertEqual(YarnTemplate.nativeWindow(config: ["text_config": ["max_position_embeddings": 262_144],
                                                           "max_position_embeddings": 32_768]), 262_144)
        XCTAssertEqual(YarnTemplate.nativeWindow(config: ["max_position_embeddings": 131_072]), 131_072)
        XCTAssertEqual(YarnTemplate.nativeWindow(config: [:]), 0)
        XCTAssertEqual(YarnTemplate.nativeWindow(config: ["text_config": ["max_position_embeddings": 0],
                                                          "max_position_embeddings": 4_096]), 4_096)
    }
}
