import XCTest
@testable import MLXCore

/// Settings ▸ Interface ▸ Language: which catalog a choice loads, what
/// launch writes, and what the picker offers.
final class AppLanguageTests: XCTestCase {

    private var appRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
    }

    /// The catalogs the app ships — the same list `install()` reads from
    /// `Bundle.main` at launch, pinned from the plist so this sees the real
    /// bundle rather than the test host's.
    private var shippedLocalizations: [String] {
        let plist = try! PropertyListSerialization.propertyList(
            from: Data(contentsOf: appRoot.appendingPathComponent("Info.plist")),
            format: nil) as? [String: Any]
        return plist?["CFBundleLocalizations"] as? [String] ?? []
    }

    // MARK: - Which catalog a choice loads

    /// The picker is the bundle's own list: a catalog added later shows up
    /// without a code change, and one a build drops is not offered. The
    /// two-language enum this replaced shipped Korean and hid it.
    func testThePickerOffersEveryCatalogTheBundleShips() {
        let codes = shippedLocalizations
        XCTAssertFalse(codes.contains("Base"), "Base is the storyboard shell, not a language")
        XCTAssertTrue(codes.contains("ko"), "Korean ships ko.lproj beside zh-Hans.lproj")
        for code in codes {
            XCTAssertEqual(AppLanguageChoice(raw: code, localizations: codes).code, code,
                           "\(code) ships but the picker drops it")
        }
    }

    /// Every offered code is loadable: English is the development region, so
    /// its keys are their own text, and everything else ships an `.lproj`.
    func testEveryOfferedCatalogResolves() throws {
        let plist = try PropertyListSerialization.propertyList(
            from: Data(contentsOf: appRoot.appendingPathComponent("Info.plist")),
            format: nil) as? [String: Any]
        let developmentRegion = try XCTUnwrap(plist?["CFBundleDevelopmentRegion"] as? String)
        for code in shippedLocalizations {
            let lproj = appRoot
                .appendingPathComponent("Sources/MLXServe/Resources/\(code).lproj")
            XCTAssertTrue(FileManager.default.fileExists(atPath: lproj.path) || code == developmentRegion,
                          "\(code) is offered with no catalog behind it")
        }
    }

    /// Following the Mac resolves to no catalog of ours: macOS picks the
    /// bundle the way it always has, so a Korean Mac keeps Korean.
    func testSystemDefaultNamesNoCatalog() {
        let codes = shippedLocalizations
        XCTAssertNil(AppLanguageOverride.catalog(preference: nil, localizations: codes))
        XCTAssertNil(AppLanguageOverride.catalog(preference: "", localizations: codes))
        // Stored by the build this replaced, whose "system" was a value.
        XCTAssertNil(AppLanguageOverride.catalog(preference: "system", localizations: codes))
        // Left by a build with more languages than this one ships.
        XCTAssertNil(AppLanguageOverride.catalog(preference: "ja", localizations: codes))
    }

    /// A named catalog is written verbatim — one code, no region, no
    /// fallback: English means English on a Chinese Mac, and 简体中文 means
    /// 简体中文 on an English one.
    func testANamedCatalogWritesExactlyItsCode() {
        let codes = shippedLocalizations
        for code in codes {
            XCTAssertEqual(AppLanguageOverride.catalog(preference: code, localizations: codes), code, code)
        }
    }

    // MARK: - What launch writes

    /// A named catalog is written as that one code, and the companion records
    /// it — the mark that lets "follow the Mac" take it back later.
    func testANamedCatalogWritesItsCodeAndStampsTheCompanion() {
        let (defaults, suite) = sandboxedDefaults()
        defer { removeSuite(suite) }

        AppLanguageOverride.install(preference: "zh-Hans",
                                    localizations: shippedLocalizations, defaults: defaults)
        XCTAssertEqual(defaults.array(forKey: "AppleLanguages") as? [String], ["zh-Hans"])
        XCTAssertEqual(defaults.string(forKey: InterfacePrefKey.languageApplied), "zh-Hans")
    }

    /// Following the Mac takes back the override this app wrote, so macOS
    /// resolves the bundle natively from the next launch on.
    func testSystemDefaultTakesBackTheOverrideThisAppWrote() {
        let (defaults, suite) = sandboxedDefaults()
        defer { removeSuite(suite) }
        defaults.set(["en"], forKey: "AppleLanguages")
        defaults.set("en", forKey: InterfacePrefKey.languageApplied)

        AppLanguageOverride.install(preference: nil,
                                    localizations: shippedLocalizations, defaults: defaults)
        defaults.synchronize()
        let persisted = UserDefaults.standard.persistentDomain(forName: suite) ?? [:]
        XCTAssertNil(persisted["AppleLanguages"], "the override this app wrote is still in place")
        XCTAssertNil(persisted[InterfacePrefKey.languageApplied])
    }

    /// System Settings' per-app language writes the SAME key in the SAME
    /// domain, so a value this app never wrote is not deleted — macOS
    /// resolves it natively, and that is the Mac the row follows.
    func testSystemDefaultLeavesAValueThisAppNeverWrote() {
        let (defaults, suite) = sandboxedDefaults()
        defer { removeSuite(suite) }
        defaults.set(["ko"], forKey: "AppleLanguages")

        AppLanguageOverride.install(preference: "",
                                    localizations: shippedLocalizations, defaults: defaults)
        XCTAssertEqual(defaults.array(forKey: "AppleLanguages") as? [String], ["ko"])
    }

    /// The user picked a language here, then changed the app's language in
    /// System Settings: the override is no longer the one this app wrote, so
    /// following the Mac leaves it alone — and the mark goes with it, so a
    /// later System Settings pick of the code this app once wrote is not
    /// mistaken for ours and taken back.
    func testSystemDefaultLeavesAnOverrideChangedByTheSystem() {
        let (defaults, suite) = sandboxedDefaults()
        defer { removeSuite(suite) }
        defaults.set(["ko"], forKey: "AppleLanguages")
        defaults.set("en", forKey: InterfacePrefKey.languageApplied)

        AppLanguageOverride.install(preference: nil,
                                    localizations: shippedLocalizations, defaults: defaults)
        defaults.synchronize()
        let persisted = UserDefaults.standard.persistentDomain(forName: suite) ?? [:]
        XCTAssertEqual(persisted["AppleLanguages"] as? [String], ["ko"])
        XCTAssertNil(persisted[InterfacePrefKey.languageApplied])
    }

    /// The mark dropped above does its job: had it stayed, a launch after the
    /// user later picked that same language in System Settings would read the
    /// value as this app's own and delete it.
    func testADroppedMarkLeavesALaterSystemPickOfThatCodeAlone() {
        let (defaults, suite) = sandboxedDefaults()
        defer { removeSuite(suite) }
        // A launch that found the override changed under us: value kept, mark
        // dropped.
        defaults.set(["ko"], forKey: "AppleLanguages")
        defaults.set("en", forKey: InterfacePrefKey.languageApplied)
        AppLanguageOverride.install(preference: nil,
                                    localizations: shippedLocalizations, defaults: defaults)
        // The user now sets that same language (the one this app once wrote)
        // in System Settings — the same key, the same domain.
        defaults.set(["en"], forKey: "AppleLanguages")

        AppLanguageOverride.install(preference: nil,
                                    localizations: shippedLocalizations, defaults: defaults)
        defaults.synchronize()
        let persisted = UserDefaults.standard.persistentDomain(forName: suite) ?? [:]
        XCTAssertEqual(persisted["AppleLanguages"] as? [String], ["en"])
        XCTAssertNil(persisted[InterfacePrefKey.languageApplied])
    }

    // MARK: - What the picker shows

    /// Each language is shown in its own spelling, the script spelled out the
    /// way System Settings lists it; only "System Default" is translated.
    func testThePickerShowsAutonymsVerbatim() {
        XCTAssertEqual(AppLanguageChoice(raw: "en", localizations: ["en"]).label, "English")
        XCTAssertEqual(AppLanguageChoice(raw: "zh-Hans", localizations: ["zh-Hans"]).label, "简体中文")
        XCTAssertEqual(AppLanguageChoice(raw: "ko", localizations: ["ko"]).label, "한국어")
        XCTAssertEqual(AppLanguageChoice(raw: "ja", localizations: ["ja"]).label, "日本語")
        // Not a row here: reads as following the Mac.
        XCTAssertNil(AppLanguageChoice(raw: "ja", localizations: ["en"]).code)
        XCTAssertEqual(AppLanguageChoice(raw: "", localizations: ["en"]).label, L10n.text("System Default"))
    }

    /// One language, one row: the bundle answers with `CFBundleLocalizations`
    /// followed by the `.lproj` directories it ships, so a code that is both
    /// — every catalogue but English here — comes back twice, and the picker
    /// drew 简体中文 and 한국어 twice each. English is the development region,
    /// so it is declared with no `.lproj` of its own to repeat it.
    func testThePickerDrawsOneRowPerLanguage() throws {
        let lproj = try FileManager.default
            .contentsOfDirectory(atPath: appRoot.appendingPathComponent("Sources/MLXServe/Resources").path)
            .filter { $0.hasSuffix(".lproj") }
            .map { ($0 as NSString).deletingPathExtension }
            .sorted()
        let asTheBundleListsThem = shippedLocalizations + lproj
        XCTAssertLessThan(Set(asTheBundleListsThem).count, asTheBundleListsThem.count,
                          "the bundle's own list repeats a code")

        XCTAssertEqual(AppLanguageChoice.rows(for: asTheBundleListsThem).map(\.label),
                       [L10n.text("System Default"), "English", "简体中文", "한국어"])
        // Base is the storyboard shell, not a language.
        XCTAssertEqual(AppLanguageChoice.rows(for: ["Base", "en", "Base"]).map(\.label),
                       [L10n.text("System Default"), "English"])
    }

    /// A suite per test: the launch path writes real keys, and none of them
    /// may land in the suite `swift test` itself answers from.
    private func sandboxedDefaults() -> (UserDefaults, String) {
        let suite = "AppLanguageOverrideTests-\(UUID().uuidString)"
        return (UserDefaults(suiteName: suite)!, suite)
    }

    private func removeSuite(_ suite: String) {
        UserDefaults.standard.removePersistentDomain(forName: suite)
    }
}
