import XCTest
@testable import MLXCore

/// `--ane-prefill` from Settings (opt-in Neural Engine prefill offload).
/// The flag mirrors the server default (OFF), so a default launch emits
/// nothing and only an explicit ON reaches the command line; the advice
/// text is pure so the M5 caution and the RAM gate are testable facts, not
/// screenshot checks.
final class AnePrefillSettingsTests: XCTestCase {
    private func args(_ mutate: (inout ServerOptions) -> Void = { _ in }) -> [String] {
        var o = ServerOptions()
        mutate(&o)
        return o.toCLIArgs(physicalMemoryBytes: 128 * 1024 * 1024 * 1024)
    }

    func testDefaultLaunchOmitsAnePrefill() {
        XCTAssertFalse(args().contains("--ane-prefill"))
    }

    func testEnabledEmitsAnePrefill() {
        XCTAssertTrue(args { $0.anePrefill = true }.contains("--ane-prefill"))
    }

    /// Configs stored before the field existed decode to the default (off).
    func testConfigStoredBeforeAnePrefillExistedDecodesToOff() throws {
        let legacy = #"{"host":"0.0.0.0","port":11234,"enablePLD":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertFalse(decoded.anePrefill)
    }

    func testTheUIHasCopyForAnePrefill() {
        XCTAssertNotNil(ServerOptions.serverFlagFields["anePrefill"])
        XCTAssertTrue(ServerOptions.serverFlagFields["anePrefill"]?.needsRestart ?? false)
        // The explainer must say what the user needs to know before flipping
        // it: the extra memory copy, that the fit is checked per model at
        // load (not a flat RAM floor — the server's gate bills the actual
        // config), and where the win applies.
        let field = ServerOptions.serverFlagFields["anePrefill"]
        let text = (field?.explainer ?? "") + " " + (field?.cost ?? "")
        XCTAssertTrue(text.lowercased().contains("declin"))
        XCTAssertTrue(text.contains("27B"))
        XCTAssertFalse(text.contains("96 GB"), "the flat 96 GB gate is retired — the copy must not resurrect it")
        XCTAssertTrue(text.lowercased().contains("prefill") || text.lowercased().contains("prompt"))
    }

    // ── The three media offloads: same shape (server default OFF, ON emits) ──

    func testMediaOffloadsDefaultOffAndEmitOnlyWhenOn() throws {
        for flag in ["--ane-image", "--ane-video", "--ane-audio"] {
            XCTAssertFalse(args().contains(flag), flag)
        }
        XCTAssertTrue(args { $0.aneImage = true }.contains("--ane-image"))
        XCTAssertTrue(args { $0.aneVideo = true }.contains("--ane-video"))
        XCTAssertTrue(args { $0.aneAudio = true }.contains("--ane-audio"))
        let legacy = #"{"host":"0.0.0.0","port":11234,"anePrefill":true}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: legacy)
        XCTAssertFalse(decoded.aneImage || decoded.aneVideo || decoded.aneAudio)
        for key in ["aneImage", "aneVideo", "aneAudio"] {
            let text = ServerOptions.serverFlagFields[key]?.explainer ?? ""
            XCTAssertTrue(text.lowercased().contains("declin"), key)
            XCTAssertTrue(text.lowercased().contains("off by default"), key)
        }
    }

    /// The int8 route is on by default only on an M5-family Mac with 64 GiB of RAM, like the server; the app always says which.
    func testW8a8DefaultsFromChipAndRamAndEmitsItsChoice() throws {
        XCTAssertTrue(ServerOptions.defaultW8a8(physicalMemoryBytes: 64 << 30, isM5Family: true))
        XCTAssertTrue(ServerOptions.defaultW8a8(physicalMemoryBytes: 512 << 30, isM5Family: true))
        XCTAssertFalse(ServerOptions.defaultW8a8(physicalMemoryBytes: (64 << 30) - 1, isM5Family: true))
        XCTAssertFalse(ServerOptions.defaultW8a8(physicalMemoryBytes: 64 << 30, isM5Family: false)) // an M4 Max with 64 GB
        XCTAssertFalse(ServerOptions.defaultW8a8(physicalMemoryBytes: 512 << 30, isM5Family: false))
        let here = ServerOptions.defaultW8a8(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory, isM5Family: AnePrefillAdvice.liveIsM5Family)
        XCTAssertEqual(ServerOptions().w8a8, here)
        XCTAssertTrue(args { $0.w8a8 = true }.contains("--w8a8"))
        XCTAssertFalse(args { $0.w8a8 = true }.contains("--no-w8a8"))
        XCTAssertTrue(args { $0.w8a8 = false }.contains("--no-w8a8"))
        XCTAssertFalse(args { $0.w8a8 = false }.contains("--w8a8"))
        let legacy = #"{"host":"0.0.0.0","port":11234}"#.data(using: .utf8)!
        XCTAssertEqual(try JSONDecoder().decode(ServerOptions.self, from: legacy).w8a8, here)
        let off = #"{"host":"0.0.0.0","port":11234,"w8a8":false}"#.data(using: .utf8)!
        XCTAssertFalse(try JSONDecoder().decode(ServerOptions.self, from: off).w8a8)
        let on = #"{"host":"0.0.0.0","port":11234,"w8a8":true}"#.data(using: .utf8)!
        XCTAssertTrue(try JSONDecoder().decode(ServerOptions.self, from: on).w8a8)
        let field = ServerOptions.serverFlagFields["w8a8"]
        XCTAssertTrue(field?.needsRestart ?? false)
        XCTAssertTrue((field?.explainer ?? "").contains("4-bit"))
    }

    // ── Per-Mac advice (pure: chip brand string + RAM in) ──

    func testM4MaxWithEnoughRamGetsNoCaution() {
        XCTAssertNil(AnePrefillAdvice.caution(
            chipBrand: "Apple M4 Max",
            physicalMemoryBytes: 128 << 30))
    }

    /// The server gate is per model now (a 27B bills ~11 GB extra under the
    /// channel split, a small model ~1 GB), so 36 GB is a fine Mac for
    /// mid-size models and carries no caution.
    func test36GBGetsNoRamCaution() {
        XCTAssertNil(AnePrefillAdvice.caution(
            chipBrand: "Apple M4 Max",
            physicalMemoryBytes: 36 << 30))
    }

    func testSmallMacSaysOnlySmallModelsFit() {
        let text = AnePrefillAdvice.caution(
            chipBrand: "Apple M4 Max",
            physicalMemoryBytes: 24 << 30)
        XCTAssertNotNil(text)
        XCTAssertTrue(text?.lowercased().contains("small models") ?? false)
    }

    /// M5-family GPUs carry NAX (neural accelerator) cores that raise the
    /// GPU's own prefill baseline, so the ANE's relative win shrinks toward
    /// nothing — the advice says "not recommended" rather than hiding the
    /// switch, because it is a measured-per-machine lever, not a hard gate.
    func testM5FamilyGetsTheNotRecommendedCaution() {
        for brand in ["Apple M5 Max", "Apple M5 Pro", "Apple M5 Ultra", "Apple M5"] {
            let text = AnePrefillAdvice.caution(
                chipBrand: brand,
                physicalMemoryBytes: 128 << 30)
            XCTAssertNotNil(text, brand)
            XCTAssertTrue(text?.lowercased().contains("not recommended") ?? false, brand)
        }
    }

    /// M4 vs M5 is a PREFIX decision on the family token, not a substring
    /// scan — "M5" appears inside no M4-era brand, but the check must not
    /// fire on some future "M45" either.
    func testFamilyDetectionIsTokenExact() {
        XCTAssertNil(AnePrefillAdvice.caution(
            chipBrand: "Apple M45 Hypothetical",
            physicalMemoryBytes: 128 << 30))
    }

    /// The RAM note outranks the M5 note: a Mac where most models won't fit
    /// should hear that, not a performance nuance.
    func testRamShortfallWinsOverM5Note() {
        let text = AnePrefillAdvice.caution(
            chipBrand: "Apple M5 Max",
            physicalMemoryBytes: 24 << 30)
        XCTAssertTrue(text?.lowercased().contains("small models") ?? false)
    }

    /// An unreadable brand string (sysctl failure) is no information — with
    /// enough RAM the toggle carries no caution rather than a false warning.
    func testUnknownChipWithEnoughRamGetsNoCaution() {
        XCTAssertNil(AnePrefillAdvice.caution(
            chipBrand: nil,
            physicalMemoryBytes: 128 << 30))
    }
}
