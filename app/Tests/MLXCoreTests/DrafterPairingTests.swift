import XCTest
@testable import MLXCore

/// Which drafter the app launches with, given the model it's about to serve.
///
/// The drafter used to be opt-in from Settings, which was fine while you also
/// had to go find the checkpoint yourself. Now it arrives with its target
/// (`DownloadManager.companionDrafterRepo`), so leaving the switch off by
/// default would mean downloading a speedup and never using it.
///
/// The whole reason this is a separate decision rather than "fill in
/// `drafterPath` when it's empty": empty means BOTH "never decided" and "the
/// user switched it off", and a default that can't tell those apart turns
/// itself back on behind the user's back at the next model switch.
final class DrafterPairingTests: XCTestCase {

    private let denseModel = "/models/mlx-community/gemma-4-e4b-it-4bit"
    private let drafterOnDisk = "/models/mlx-community/gemma-4-E4B-it-assistant-bf16"

    func testADenseGemmaWithItsDrafterOnDiskPairsWithoutBeingAsked() {
        XCTAssertEqual(
            DrafterPairing.decide(modelPath: denseModel, optedOut: false, onDiskPath: drafterOnDisk),
            drafterOnDisk)
    }

    func testTurningItOffStaysOff() {
        XCTAssertEqual(
            DrafterPairing.decide(modelPath: denseModel, optedOut: true, onDiskPath: drafterOnDisk),
            "",
            "an explicit off must survive every later model switch")
    }

    func testTheMoeTargetIsNeverPairedAutomatically() {
        // Its drafter repo exists and may well be on disk from an older build —
        // it just makes decode slower, so we never reach for it on our own.
        XCTAssertEqual(
            DrafterPairing.decide(modelPath: "/models/mlx-community/gemma-4-26b-a4b-it-4bit",
                                  optedOut: false,
                                  onDiskPath: "/models/mlx-community/gemma-4-26B-A4B-it-assistant-bf16"),
            "")
    }

    func testSwitchingToAModelWithNoDrafterClearsIt() {
        // A drafter is pinned to ONE Gemma 4 size; carrying it onto another
        // model is `DrafterTargetMismatch` at server start.
        XCTAssertEqual(
            DrafterPairing.decide(modelPath: "/models/Qwen/Qwen3.6-27B", optedOut: false, onDiskPath: nil),
            "")
        XCTAssertEqual(
            DrafterPairing.decide(modelPath: denseModel, optedOut: false, onDiskPath: nil),
            "",
            "paired size, but nothing downloaded yet")
    }

    // MARK: - Persistence

    func testAutoPairingIsTheDefaultAndOlderSettingsBlobsGetIt() throws {
        XCTAssertFalse(ServerOptions().drafterOptOut, "the drafter is on by default where we recommend it")

        // A settings file written before this field existed.
        let older = Data(#"{"port":11234,"drafterPath":""}"#.utf8)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: older)
        XCTAssertFalse(decoded.drafterOptOut)
    }

    func testAnOptOutSurvivesARoundTrip() throws {
        var o = ServerOptions()
        o.drafterOptOut = true
        let back = try JSONDecoder().decode(ServerOptions.self, from: try JSONEncoder().encode(o))
        XCTAssertTrue(back.drafterOptOut)
    }

    /// It's a UI preference, not a server flag — it must never reach the CLI.
    func testTheOptOutIsNotALaunchFlag() {
        var o = ServerOptions()
        o.drafterOptOut = true
        XCTAssertFalse(o.toCLIArgs().joined(separator: " ").lowercased().contains("optout"))
    }
}
