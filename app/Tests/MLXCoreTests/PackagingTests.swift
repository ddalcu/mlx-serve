import XCTest

/// Packaging artifacts are plists a reviewer's tooling reads,
/// not code — so nothing else would catch a wrong key. These pin the contracts
/// that would otherwise fail silently at submission or, worse, at launch.
final class PackagingTests: XCTestCase {

    private var appDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
    }

    private func plist(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: appDir.appendingPathComponent(name))
        return try XCTUnwrap(PropertyListSerialization.propertyList(from: data, options: [], format: nil) as? [String: Any])
    }

    // MARK: entitlements

    func testAppEntitlementsAreSandboxedWithVirtualization() throws {
        let ent = try plist("MLXCore-MAS.entitlements")
        XCTAssertEqual(ent["com.apple.security.app-sandbox"] as? Bool, true,
                       "the App Store build MUST be sandboxed")
        XCTAssertEqual(ent["com.apple.security.virtualization"] as? Bool, true,
                       "the guest can't boot without the virtualization entitlement")
        XCTAssertEqual(ent["com.apple.security.network.client"] as? Bool, true)
        XCTAssertEqual(ent["com.apple.security.network.server"] as? Bool, true)
        XCTAssertEqual(ent["com.apple.security.files.user-selected.read-write"] as? Bool, true)
        XCTAssertEqual(ent["com.apple.security.device.audio-input"] as? Bool, true)
    }

    /// The app must NOT carry the Hypervisor.framework entitlement — that's the
    /// wrong API (and not what VZ needs), and an over-broad entitlement is a
    /// review flag.
    func testAppDoesNotCarryTheHypervisorEntitlement() throws {
        let ent = try plist("MLXCore-MAS.entitlements")
        XCTAssertNil(ent["com.apple.security.hypervisor"])
    }

    /// THE load-bearing invariant: the helper's entitlements must be EXACTLY
    /// {app-sandbox, inherit}. `inherit` only works when the child's own
    /// entitlement set is limited to these two; a third key breaks sandbox
    /// inheritance and the helper fails to start under the sandbox.
    func testHelperEntitlementsAreExactlySandboxPlusInherit() throws {
        let ent = try plist("mlx-serve-MAS.entitlements")
        XCTAssertEqual(Set(ent.keys),
                       ["com.apple.security.app-sandbox", "com.apple.security.inherit"],
                       "the helper must carry ONLY app-sandbox + inherit, or inheritance breaks")
        XCTAssertEqual(ent["com.apple.security.app-sandbox"] as? Bool, true)
        XCTAssertEqual(ent["com.apple.security.inherit"] as? Bool, true)
    }

    // MARK: rendered signing entitlements

    /// App Store Connect rejects a manually-codesigned upload whose app binary
    /// lacks `com.apple.application-identifier` + `com.apple.developer.team-identifier`
    /// — Xcode injects them; our codesign must too. build.sh renders them into a
    /// temp copy of the static entitlements when signing with a real identity;
    /// this pins the renderer's output.
    func testRenderedMASEntitlementsCarryBothIdentifierKeys() throws {
        let script = appDir.deletingLastPathComponent()
            .appendingPathComponent("scripts/render-mas-entitlements.sh")
        let out = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("rendered-\(UUID().uuidString).entitlements")
        defer { try? FileManager.default.removeItem(at: out) }

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/bash")
        p.arguments = [script.path,
                       appDir.appendingPathComponent("MLXCore-MAS.entitlements").path,
                       "TEAM123456", "com.dalcu.mlx-core.mas", out.path]
        try p.run()
        p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0, "renderer failed")

        let data = try Data(contentsOf: out)
        let ent = try XCTUnwrap(PropertyListSerialization.propertyList(
            from: data, options: [], format: nil) as? [String: Any])
        XCTAssertEqual(ent["com.apple.application-identifier"] as? String,
                       "TEAM123456.com.dalcu.mlx-core.mas")
        XCTAssertEqual(ent["com.apple.developer.team-identifier"] as? String, "TEAM123456")
        // The base entitlements must survive the render untouched.
        XCTAssertEqual(ent["com.apple.security.app-sandbox"] as? Bool, true)
        XCTAssertEqual(ent["com.apple.security.virtualization"] as? Bool, true)
        XCTAssertNil(ent["com.apple.security.hypervisor"])
    }

    /// The renderer must never touch the HELPER's entitlements: inherit only
    /// works when the child's set is EXACTLY {app-sandbox, inherit}, so the
    /// identifier keys would break it. The static file stays the signed truth.
    func testHelperEntitlementsFileIsNotATemplate() throws {
        let ent = try plist("mlx-serve-MAS.entitlements")
        XCTAssertEqual(ent.count, 2, "helper entitlements must stay exactly two keys")
    }

    // MARK: Info-MAS.plist

    func testMASInfoPlistHasStoreKeysAndSeparateBundleID() throws {
        let info = try plist("Info-MAS.plist")
        XCTAssertEqual(info["CFBundleIdentifier"] as? String, "com.dalcu.mlx-core.mas",
                       "the store build needs a distinct id to coexist with the DMG build")
        XCTAssertNotNil(info["LSApplicationCategoryType"], "App Store submission requires a category")
        XCTAssertEqual(info["ITSAppUsesNonExemptEncryption"] as? Bool, false)
    }

    /// AppleEvents automation is compiled out of the store build, so the usage
    /// string must be absent — claiming it would advertise a capability the app
    /// doesn't ship, which review flags.
    func testMASInfoPlistDropsAppleEvents() throws {
        let info = try plist("Info-MAS.plist")
        XCTAssertNil(info["NSAppleEventsUsageDescription"],
                     "the store build has no AppleScript automation")
    }

    /// The mic + speech usage strings must survive — Voice mode still ships.
    func testMASInfoPlistKeepsMicAndSpeechUsageStrings() throws {
        let info = try plist("Info-MAS.plist")
        XCTAssertNotNil(info["NSMicrophoneUsageDescription"])
        XCTAssertNotNil(info["NSSpeechRecognitionUsageDescription"])
    }

    // MARK: privacy manifest

    func testPrivacyManifestDeclaresNoTrackingAndTheRequiredReasonAPIs() throws {
        let privacy = try plist("PrivacyInfo.xcprivacy")
        XCTAssertEqual(privacy["NSPrivacyTracking"] as? Bool, false)

        let apis = try XCTUnwrap(privacy["NSPrivacyAccessedAPITypes"] as? [[String: Any]])
        let categories = Set(apis.compactMap { $0["NSPrivacyAccessedAPIType"] as? String })
        // UserDefaults is the one App Store Connect reliably rejects when omitted.
        XCTAssertTrue(categories.contains("NSPrivacyAccessedAPICategoryUserDefaults"),
                      "declared API categories: \(categories)")
        // Every declaration must carry at least one reason code, or the upload is rejected.
        for api in apis {
            let reasons = api["NSPrivacyAccessedAPITypeReasons"] as? [String] ?? []
            XCTAssertFalse(reasons.isEmpty, "\(api["NSPrivacyAccessedAPIType"] ?? "?") has no reason code")
        }
    }
}
