import XCTest
@testable import MLXCore

/// The language override: what a preference value means, and what the bundle
/// lookup does with it.
final class LanguageSettingsTests: XCTestCase {
    private static let resourcesRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // MLXCoreTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // app
        .appendingPathComponent("Sources/MLXServe/Resources")

    /// The packaged resources, as the app bundle carries them (`build.sh`
    /// copies this directory into `Contents/Resources`).
    private static let resourcesBundle = Bundle(path: resourcesRoot.path)!

    // MARK: - The preference value

    func testUnknownOrEmptyPreferenceMeansSystem() {
        for raw in ["", "de", "zh-Hant", "SYSTEM"] {
            XCTAssertEqual(AppLanguage(rawValue: raw) ?? .system, .system, "raw \(raw) should fall back to the system")
        }
    }

    func testSystemHasNoLocaleAndTheOthersNameThemselves() {
        XCTAssertNil(AppLanguage.system.code)
        XCTAssertNil(AppLanguage.system.locale, "system means 'no override', not 'the en locale'")
        XCTAssertEqual(AppLanguage.english.code, "en")
        XCTAssertEqual(AppLanguage.simplifiedChinese.code, "zh-Hans")
        XCTAssertEqual(AppLanguage.simplifiedChinese.locale, Locale(identifier: "zh-Hans"))
    }

    /// A language names itself, so those labels are never routed through the
    /// catalog; only "System" is a translatable choice.
    func testOnlyTheSystemLabelIsTranslatable() {
        XCTAssertEqual(AppLanguage.english.label, "English")
        XCTAssertEqual(AppLanguage.simplifiedChinese.label, "简体中文")
        XCTAssertEqual(AppLanguage.system.label, "System")
        for language in AppLanguage.allCases {
            XCTAssertFalse(language.id.isEmpty)
        }
    }

    // MARK: - Resolving a bundle

    func testResolvesTheShippedLocalizationAndRejectsUnknownCodes() {
        XCTAssertNotNil(BundleLanguageOverride.bundle(for: "zh-Hans", in: Self.resourcesBundle))
        XCTAssertNil(BundleLanguageOverride.bundle(for: "de", in: Self.resourcesBundle),
                     "a language this build ships no .lproj for must not resolve")
    }

    func testTheResolvedBundleAnswersWithTheTranslatedString() throws {
        let bundle = try XCTUnwrap(BundleLanguageOverride.bundle(for: "zh-Hans", in: Self.resourcesBundle))
        XCTAssertEqual(bundle.localizedString(forKey: "Settings", value: nil, table: nil), "设置")
        // An unknown key still falls back to the key itself, so a string that
        // has not been translated yet reads as its English source.
        XCTAssertEqual(bundle.localizedString(forKey: "Not a key in the catalog", value: nil, table: nil),
                       "Not a key in the catalog")
    }

    // MARK: - The lookup override

    /// Applying a language this build does not ship must leave lookups exactly
    /// as they were — the override is a redirection, never a breakage.
    func testAnUnavailableLanguageLeavesLookupsAlone() {
        // The test host carries no .lproj of its own, so this is the shipped
        // "language the bundle cannot provide" path: the redirection stays
        // off and every lookup answers exactly as it did before.
        BundleLanguageOverride.apply(.simplifiedChinese)
        XCTAssertNil(BundleLanguageOverride.languageBundle)
        XCTAssertEqual(Bundle.main.localizedString(forKey: "Absent key", value: nil, table: nil), "Absent key")

        // Resolved against the packaged resources, the same call does redirect.
        BundleLanguageOverride.apply(.simplifiedChinese, in: Self.resourcesBundle)
        XCTAssertNotNil(BundleLanguageOverride.languageBundle)

        // And `.system` restores the untouched path.
        BundleLanguageOverride.apply(.system, in: Self.resourcesBundle)
        XCTAssertNil(BundleLanguageOverride.languageBundle)
        XCTAssertEqual(Bundle.main.localizedString(forKey: "Absent key", value: nil, table: nil), "Absent key")
    }

    /// The exchange is scoped to `Bundle.main`: another bundle's own strings
    /// must keep resolving through the framework that owns them.
    func testOtherBundlesKeepTheirOwnLookup() {
        BundleLanguageOverride.apply(.simplifiedChinese, in: Self.resourcesBundle)
        defer { BundleLanguageOverride.apply(.system) }

        let other = Bundle(for: Self.self)
        XCTAssertNotEqual(other, Bundle.main)
        XCTAssertEqual(other.localizedString(forKey: "Absent key", value: "fallback", table: nil), "fallback")
    }

    /// Applying twice is what every scene does; it must stay a plain redirection.
    func testReapplyingIsIdempotent() {
        BundleLanguageOverride.apply(.simplifiedChinese, in: Self.resourcesBundle)
        let first = BundleLanguageOverride.languageBundle
        BundleLanguageOverride.apply(.simplifiedChinese, in: Self.resourcesBundle)
        XCTAssertNotNil(BundleLanguageOverride.languageBundle)
        XCTAssertEqual(BundleLanguageOverride.languageBundle?.bundlePath, first?.bundlePath)
        BundleLanguageOverride.apply(.system)
    }

    /// The row writes through `select`, so the choice is both resolved and
    /// recorded; the order matters because the write is what re-renders.
    func testSelectResolvesTheBundleAndRecordsTheChoice() throws {
        let suite = "LanguageSettingsTests.select"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)

        AppLanguage.select(.simplifiedChinese, into: defaults, in: Self.resourcesBundle)
        XCTAssertNotNil(BundleLanguageOverride.languageBundle)
        XCTAssertEqual(defaults.string(forKey: InterfacePrefKey.language), "zh-Hans")

        AppLanguage.select(.system, into: defaults, in: Self.resourcesBundle)
        XCTAssertNil(BundleLanguageOverride.languageBundle)
        XCTAssertEqual(defaults.string(forKey: InterfacePrefKey.language), "system")
        defaults.removePersistentDomain(forName: suite)
    }
}
