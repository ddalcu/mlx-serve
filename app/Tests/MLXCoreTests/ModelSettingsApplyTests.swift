import XCTest
@testable import MLXCore

/// Bar: the startup model restarts the server (a hot reload re-bills it under
/// `--max-resident-mem` and 503s); other resident models hot-reload; a
/// steering-only edit applies live; nothing resident = save only.
final class ModelSettingsApplyTests: XCTestCase {
    func testStartupModelRestarts() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: true, changed: [.ctxSize]), .restart)
    }
    func testOtherResidentModelReloads() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: false, changed: [.kvQuant]), .reload)
    }
    func testNotResidentSavesOnly() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: false, isStartupModel: true, changed: [.ctxSize]), .saveOnly)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: false, loaded: true, isStartupModel: true, changed: [.steering]), .saveOnly)
    }
    /// A steering-only edit applies live; any reload field keeps its reload/restart; nothing changed = save only.
    func testSteeringOnlyAppliesLiveButNeverBesideAReloadField() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: true, changed: [.steering]), .live)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: false, changed: [.steering]), .live)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: true, changed: [.steering, .ctxSize]), .restart)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: false, changed: [.steering, .mtp]), .reload)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: false, changed: []), .saveOnly)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: false, isStartupModel: false, changed: [.steering]), .saveOnly)
    }
    func testChangedFieldsNamesExactlyWhatMoved() {
        let a = ModelOverride(ctxSize: 4096, steering: .configured(name: "terse", ffn: -1, attn: 0))
        XCTAssertEqual(ModelSettingsApply.changedFields(from: a, to: a), [])
        XCTAssertEqual(ModelSettingsApply.changedFields(from: a, to: ModelOverride(ctxSize: 4096, steering: .off)), [.steering])
        XCTAssertEqual(ModelSettingsApply.changedFields(from: a, to: ModelOverride(ctxSize: 8192, steering: .configured(name: "terse", ffn: -1, attn: 0))), [.ctxSize])
        XCTAssertEqual(ModelSettingsApply.changedFields(from: a, to: ModelOverride()), [.ctxSize, .steering])
    }
}

/// The picker round-trips the three states and keeps entered scales across a re-pick.
final class ModelSettingsSteeringPickerTests: XCTestCase {
    func testTagRoundTrip() {
        XCTAssertEqual(ModelSettingsApply.steeringTag(nil), "")
        XCTAssertEqual(ModelSettingsApply.steeringTag(.off), "off")
        XCTAssertEqual(ModelSettingsApply.steeringTag(.configured(name: "terse", ffn: 1, attn: 0)), "name:terse")
        XCTAssertNil(ModelSettingsApply.steering(fromTag: "", current: .off))
        XCTAssertEqual(ModelSettingsApply.steering(fromTag: "off", current: nil), .off)
        XCTAssertEqual(ModelSettingsApply.steering(fromTag: "name:terse", current: nil), .configured(name: "terse", ffn: 1, attn: 0))
        XCTAssertEqual(ModelSettingsApply.steering(fromTag: "name:formal", current: .configured(name: "terse", ffn: -1, attn: 0.5)),
                       .configured(name: "formal", ffn: -1, attn: 0.5))
    }
    func testChoicesKeepAConfiguredNameTheRegistryLacks() {
        XCTAssertEqual(ModelSettingsApply.steeringChoices(registry: ["a", "b"], current: nil), ["a", "b"])
        XCTAssertEqual(ModelSettingsApply.steeringChoices(registry: ["a", "b"], current: .configured(name: "b", ffn: 1, attn: 0)), ["a", "b"])
        XCTAssertEqual(ModelSettingsApply.steeringChoices(registry: ["a"], current: .configured(name: "/x/y.f32", ffn: 1, attn: 0)), ["a", "/x/y.f32"])
    }
    func testRegistryListsOnlyBanks() throws {
        let dir = NSTemporaryDirectory() + "steering-reg-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        // "my bank" is a file no request can name: listing it saves a choice that never loads.
        for f in ["terse.f32", "formal.f32", "terse.json", "notes.txt", "my bank.f32"] {
            FileManager.default.createFile(atPath: dir + "/" + f, contents: Data([0]))
        }
        XCTAssertEqual(SteeringRegistry.names(in: dir), ["formal", "terse"])
        XCTAssertEqual(SteeringRegistry.names(in: dir + "/missing"), [])
    }
    func testSteeringBodyShapes() {
        let off = APIClient.steeringBody(model: "m", override: .off)
        XCTAssertTrue(off["name"] is NSNull)
        XCTAssertEqual(off["persist"] as? Bool, false)
        let reset = APIClient.steeringBody(model: "m", override: nil)
        XCTAssertEqual(reset["reset"] as? Bool, true)
        XCTAssertNil(reset["name"])
        let cfg = APIClient.steeringBody(model: "m", override: .configured(name: "terse", ffn: -1, attn: 0.5))
        XCTAssertEqual(cfg["name"] as? String, "terse")
        XCTAssertEqual(cfg["ffn"] as? Double, -1)
        XCTAssertEqual(cfg["attn"] as? Double, 0.5)
        XCTAssertEqual(cfg["persist"] as? Bool, false)
        XCTAssertEqual(cfg["model"] as? String, "m")
    }
    func testPropsSteeringParsesTheActiveBank() {
        let json: [String: Any] = ["settings": ["steering": ["file": "/Users/x/.mlx-serve/steering/terse.f32", "ffn": -1, "attn": 0]]]
        XCTAssertEqual(APIClient.SteeringInfo.parse(json), APIClient.SteeringInfo(name: "terse", file: "/Users/x/.mlx-serve/steering/terse.f32", ffn: -1, attn: 0))
        XCTAssertNil(APIClient.SteeringInfo.parse(["settings": ["steering": ["file": "", "ffn": 0, "attn": 0]]]))
        XCTAssertNil(APIClient.SteeringInfo.parse([:]))
    }
}

final class ModelSettingsMtpRowsTests: XCTestCase {
    func testNoHeadHidesBoth() {
        let r = ModelSettingsApply.mtpRows(available: false, mtp: true)
        XCTAssertFalse(r.mtp); XCTAssertFalse(r.acceptance)
    }
    func testHeadShowsAcceptanceUnlessOff() {
        XCTAssertTrue(ModelSettingsApply.mtpRows(available: true, mtp: nil).acceptance)
        XCTAssertFalse(ModelSettingsApply.mtpRows(available: true, mtp: false).acceptance)
    }
    func testOlderServerShowsBoth() {
        let r = ModelSettingsApply.mtpRows(available: nil, mtp: nil)
        XCTAssertTrue(r.mtp); XCTAssertTrue(r.acceptance)
    }
}

/// Mirrors the server's `mtp.dirAdvertisesMtp`: sidecar file, index marker, or qwen4's own head.
final class LocalMtpHeadProbeTests: XCTestCase {
    private func dir(_ name: String) throws -> String {
        let d = NSTemporaryDirectory() + "mtp-probe-\(UUID().uuidString)/\(name)"
        try FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }
    func testNoHead() throws {
        let d = try dir("plain")
        try "{}".write(toFile: d + "/config.json", atomically: true, encoding: .utf8)
        try #"{"weight_map":{"model.layers.0.mlp.up_proj.weight":"a.safetensors"}}"#
            .write(toFile: d + "/model.safetensors.index.json", atomically: true, encoding: .utf8)
        XCTAssertFalse(DownloadManager.dirHasMtpHead(atDir: d))
    }
    func testSidecarFile() throws {
        let d = try dir("sidecar")
        try FileManager.default.createDirectory(atPath: d + "/mtp", withIntermediateDirectories: true)
        try Data([1]).write(to: URL(fileURLWithPath: d + "/mtp/weights.safetensors"))
        XCTAssertTrue(DownloadManager.dirHasMtpHead(atDir: d))
    }
    func testIndexMarkers() throws {
        for key in ["language_model.mtp.fc.weight", "mtp.eh_proj.weight", "language_model.mtp.fc_hidden.weight"] {
            let d = try dir("idx")
            try #"{"weight_map":{"\#(key)":"a.safetensors"}}"#
                .write(toFile: d + "/model.safetensors.index.json", atomically: true, encoding: .utf8)
            XCTAssertTrue(DownloadManager.dirHasMtpHead(atDir: d), key)
        }
    }
}
