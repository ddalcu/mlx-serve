import XCTest
@testable import MLXCore

/// The Settings screen's MTP controls (Speculative Decoding ▸ Multi-Token
/// Prediction). MTP is the model's OWN trained head — a Qwen 3.5/3.6 checkpoint
/// either ships an `mtp/` sidecar or it doesn't — so the server auto-loads it
/// and the app must stay silent at the defaults, exactly like every other
/// match-default flag (see the "ServerOptions defaults must mirror the Zig
/// server defaults" gotcha in CLAUDE.md).
final class MTPSettingsTests: XCTestCase {

    private func args(_ mutate: (inout ServerOptions) -> Void) -> [String] {
        var o = ServerOptions()
        mutate(&o)
        return o.toCLIArgs(physicalMemoryBytes: 64 * 1024 * 1024 * 1024)
    }

    /// Defaults are ON / auto — the same as the server's — so a default launch
    /// must emit neither flag.
    func testDefaultsEmitNoMtpFlags() {
        let a = args { _ in }
        XCTAssertFalse(a.contains("--no-mtp"))
        XCTAssertFalse(a.contains("--mtp-depth"))
    }

    func testTurningMtpOffEmitsNoMtp() {
        let a = args { $0.enableMTP = false }
        XCTAssertTrue(a.contains("--no-mtp"))
    }

    /// Depth 0 is the server's "auto" sentinel (its adaptive controller tunes
    /// depth live) — passing `--mtp-depth 0` would be redundant, and pinning a
    /// depth the user didn't choose is worse.
    func testFixedDepthIsEmittedButAutoIsNot() {
        let auto = args { $0.mtpDepth = 0 }
        XCTAssertFalse(auto.contains("--mtp-depth"))

        let fixed = args { $0.mtpDepth = 3 }
        guard let i = fixed.firstIndex(of: "--mtp-depth") else {
            return XCTFail("expected --mtp-depth in \(fixed)")
        }
        XCTAssertEqual(fixed[i + 1], "3")
    }

    /// Both are server-launch flags: changing either must raise the "restart to
    /// apply" banner, or the UI would claim a setting is live when it isn't.
    func testMtpFieldsTripTheRestartDetector() {
        let base = ServerOptions()

        var offMTP = base
        offMTP.enableMTP = false
        XCTAssertFalse(base.serverLaunchEquals(offMTP))

        var deeper = base
        deeper.mtpDepth = 4
        XCTAssertFalse(base.serverLaunchEquals(deeper))
    }

    /// A config written before these fields existed must decode with MTP ON and
    /// depth auto — a tolerant decode, not a throw that would reset the user's
    /// whole config (token included).
    func testConfigStoredBeforeMtpExistedDecodesToTheDefaults() throws {
        let legacy = #"{"host":"0.0.0.0","port":11234,"enablePLD":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertTrue(decoded.enableMTP)
        XCTAssertEqual(decoded.mtpDepth, 0)
    }

    /// The rows are rendered from this metadata — a missing entry means a
    /// silently absent control.
    func testTheUIHasCopyForBothControls() {
        XCTAssertNotNil(ServerOptions.serverFlagFields["enableMTP"])
        XCTAssertNotNil(ServerOptions.serverFlagFields["mtpDepth"])
        XCTAssertTrue(ServerOptions.serverFlagFields["enableMTP"]?.needsRestart ?? false)
    }
}
