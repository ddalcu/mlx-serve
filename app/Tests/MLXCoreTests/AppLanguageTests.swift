import XCTest
@testable import MLXCore

/// Settings ▸ Interface ▸ Language: which catalog a choice loads.
final class AppLanguageTests: XCTestCase {

    /// The override names two catalogs: Chinese ships a `.lproj`, and English
    /// is the development region — a name with nothing behind it resolves to
    /// whatever the Mac says, which is the language the row exists to change.
    func testEveryCatalogTheOverrideCanNameIsShipped() {
        let app = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // app
        let resources = app.appendingPathComponent("Sources/MLXServe/Resources")
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: resources.appendingPathComponent("zh-Hans.lproj").path),
            "no zh-Hans.lproj behind the Chinese choice")

        let plist = try! PropertyListSerialization.propertyList(
            from: Data(contentsOf: app.appendingPathComponent("Info.plist")), format: nil) as? [String: Any]
        XCTAssertEqual(plist?["CFBundleDevelopmentRegion"] as? String, "en",
                       "English text comes from the development region")
    }

    func testSystemDefaultReadsBothChineseSpellingsAsSimplified() {
        for preference in ["zh-Hans", "zh-Hant", "zh-TW", "zh-CN", "zh"] {
            XCTAssertEqual(AppLanguage.system.catalog(preferredLanguages: [preference]),
                           "zh-Hans", preference)
        }
        // A second preference never outranks the first one.
        XCTAssertEqual(AppLanguage.system.catalog(preferredLanguages: ["zh-HK", "en-US"]), "zh-Hans")
    }

    func testSystemDefaultReadsEveryOtherLanguageAsEnglish() {
        for preference in ["en", "en-GB", "ja-JP", "ko-KR", "fr-FR", "de-DE"] {
            XCTAssertEqual(AppLanguage.system.catalog(preferredLanguages: [preference]),
                           "en", preference)
        }
        XCTAssertEqual(AppLanguage.system.catalog(preferredLanguages: []), "en")
    }

    /// The named choices are not hints: English has to be loadable on a Mac
    /// whose own language is Chinese, and Chinese on a Mac set to English.
    func testANamedLanguageWinsOverTheMac() {
        XCTAssertEqual(AppLanguage.english.catalog(preferredLanguages: ["zh-Hans"]), "en")
        XCTAssertEqual(AppLanguage.simplifiedChinese.catalog(preferredLanguages: ["en-US"]), "zh-Hans")
    }

    /// Every choice names a catalog explicitly. English is one of them: on a
    /// Mac set to Chinese the bundle would pick zh-Hans on its own, so
    /// "English" has to load its (empty) catalog to mean anything.
    func testEveryChoiceResolvesToAShippedCatalog() {
        XCTAssertEqual(AppLanguageOverride.targetLanguage(preference: "zh-Hans", preferredLanguages: ["en-US"]),
                       "zh-Hans")
        XCTAssertEqual(AppLanguageOverride.targetLanguage(preference: "en", preferredLanguages: ["zh-Hant"]),
                       "en")
        XCTAssertEqual(AppLanguageOverride.targetLanguage(preference: nil, preferredLanguages: ["zh-TW"]),
                       "zh-Hans")
        XCTAssertEqual(AppLanguageOverride.targetLanguage(preference: nil, preferredLanguages: ["ja-JP"]),
                       "en")
        // A preference from a build that had more languages than this one.
        XCTAssertEqual(AppLanguageOverride.targetLanguage(preference: "ko", preferredLanguages: ["ko-KR"]),
                       "en")
    }

    /// Each language is shown in its own spelling; only the "follow the Mac"
    /// option is a word to translate.
    func testThePickerShowsAutonymsVerbatim() {
        XCTAssertEqual(AppLanguage.english.pickerLabel, "English")
        XCTAssertEqual(AppLanguage.simplifiedChinese.pickerLabel, "简体中文")
        XCTAssertEqual(AppLanguage.system.pickerLabel, L10n.text("System Default"))
    }
}
