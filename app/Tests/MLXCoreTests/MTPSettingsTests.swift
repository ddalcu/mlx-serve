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
    /// `--mtp` rides every default launch (MoE heads ON — the app's one
    /// deliberate divergence from the server default); nothing else does.
    func testDefaultsEmitMtpAndNothingElse() {
        let a = args { _ in }
        XCTAssertFalse(a.contains("--no-mtp"))
        XCTAssertFalse(a.contains("--mtp-depth"))
        XCTAssertTrue(a.contains("--mtp"))
    }

    func testTurningMoeMtpOffEmitsNoFlag() {
        let a = args { $0.mtpOnMoE = false }
        XCTAssertFalse(a.contains("--mtp"))
        XCTAssertFalse(a.contains("--no-mtp"))
    }

    func testTurningMtpOffEmitsNoMtp() {
        let a = args { $0.enableMTP = false }
        XCTAssertTrue(a.contains("--no-mtp"))
    }

    /// A MoE checkpoint that ships an MTP head keeps it OFF for every request
    /// that omits `enable_mtp` (the server's `defaultEnableMtp` — the same
    /// verify-forward routing caution the drafter has). `--mtp` overrides that,
    /// and it is the ONLY way a client which sends no sampling/spec fields at
    /// all (Claude Code, llmprobe, curl) can reach a MoE head.
    func testForcingMtpOnMoeEmitsTheMtpFlag() {
        let a = args { $0.mtpOnMoE = true }
        XCTAssertTrue(a.contains("--mtp"))
    }

    /// Turning MTP off wins: `--mtp --no-mtp` would be an incoherent pair, and
    /// the head isn't even loaded, so forcing it on for MoE is meaningless.
    func testMtpOffSuppressesTheForceFlag() {
        let a = args {
            $0.enableMTP = false
            $0.mtpOnMoE = true
        }
        XCTAssertTrue(a.contains("--no-mtp"))
        XCTAssertFalse(a.contains("--mtp"))
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

        var unforced = base
        unforced.mtpOnMoE = false
        XCTAssertFalse(base.serverLaunchEquals(unforced))
    }

    /// A config written before these fields existed must decode with MTP ON and
    /// depth auto — a tolerant decode, not a throw that would reset the user's
    /// whole config (token included).
    func testConfigStoredBeforeMtpExistedDecodesToTheDefaults() throws {
        let legacy = #"{"host":"0.0.0.0","port":11234,"enablePLD":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertTrue(decoded.enableMTP)
        XCTAssertEqual(decoded.mtpDepth, 0)
        XCTAssertTrue(decoded.mtpOnMoE)
    }

    /// The default flipped ON, and every config written before that stored
    /// the OLD default as `forceMTPOnMoE:false` — a tolerant decode of that
    /// key would have kept MoE MTP off for everyone forever. The key was
    /// retired with the flip, so the stored value is ignored.
    func testConfigWithTheRetiredForceKeyDecodesToMoeMtpOn() throws {
        let old = #"{"host":"0.0.0.0","port":11234,"forceMTPOnMoE":false}"#.data(using: .utf8)!
        XCTAssertTrue(try JSONDecoder().decode(ServerOptions.self, from: old).mtpOnMoE)
        let off = #"{"host":"0.0.0.0","port":11234,"mtpOnMoE":false}"#.data(using: .utf8)!
        XCTAssertFalse(try JSONDecoder().decode(ServerOptions.self, from: off).mtpOnMoE)
    }

    /// The rows are rendered from this metadata — a missing entry means a
    /// silently absent control.
    func testTheUIHasCopyForBothControls() {
        XCTAssertNotNil(ServerOptions.serverFlagFields["enableMTP"])
        XCTAssertNotNil(ServerOptions.serverFlagFields["mtpDepth"])
        XCTAssertNotNil(ServerOptions.serverFlagFields["mtpOnMoE"])
        XCTAssertTrue(ServerOptions.serverFlagFields["enableMTP"]?.needsRestart ?? false)
        XCTAssertTrue(ServerOptions.serverFlagFields["mtpOnMoE"]?.needsRestart ?? false)
    }

    // MARK: DSpark (DeepSeek-V4 draft stages)

    /// DSpark is OPT-IN server-side (the stages cost ~11 GB resident), so the
    /// app default is off and a default launch emits nothing.
    func testDefaultsEmitNoDsparkFlag() {
        XCTAssertFalse(args { _ in }.contains("--dspark"))
    }

    func testEnablingDsparkEmitsTheFlag() {
        XCTAssertTrue(args { $0.enableDSpark = true }.contains("--dspark"))
    }

    /// Configs stored before the field existed decode to the default (off).
    func testConfigStoredBeforeDsparkExistedDecodesToOff() throws {
        let legacy = #"{"host":"0.0.0.0","port":11234,"enablePLD":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertFalse(decoded.enableDSpark)
    }

    func testTheUIHasCopyForDspark() {
        XCTAssertNotNil(ServerOptions.serverFlagFields["enableDSpark"])
        XCTAssertTrue(ServerOptions.serverFlagFields["enableDSpark"]?.needsRestart ?? false)
    }
}
