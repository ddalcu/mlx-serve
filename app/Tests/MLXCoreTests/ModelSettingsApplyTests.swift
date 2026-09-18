import XCTest
@testable import MLXCore

/// Bar: the startup model restarts the server (a hot reload re-bills it under
/// `--max-resident-mem` and 503s); other resident models hot-reload; nothing
/// resident = save only.
final class ModelSettingsApplyTests: XCTestCase {
    func testStartupModelRestarts() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: true), .restart)
    }
    func testOtherResidentModelReloads() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: true, isStartupModel: false), .reload)
    }
    func testNotResidentSavesOnly() {
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: true, loaded: false, isStartupModel: true), .saveOnly)
        XCTAssertEqual(ModelSettingsApply.plan(serverRunning: false, loaded: true, isStartupModel: true), .saveOnly)
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
