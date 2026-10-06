import XCTest
@testable import MLXCore

final class SpeechLocaleTests: XCTestCase {
    func testEachEngineUsesItsOwnCapabilities() {
        let legacy = SpeechLocale.resolve(
            preferredLanguages: ["en-GB", "ja-JP"],
            supportedLocales: [Locale(identifier: "en_GB"), Locale(identifier: "ja_JP")],
            isAvailable: { $0.identifier == "ja_JP" },
            fallback: Locale(identifier: "en_JP"))
        let modern = SpeechLocale.resolve(
            preferredLanguages: ["en-GB", "ja-JP"],
            supportedLocales: [Locale(identifier: "en_GB")],
            isAvailable: { _ in true },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(legacy, .available(Locale(identifier: "ja_JP")))
        XCTAssertTrue(legacy.isAvailable)
        XCTAssertEqual(modern, .available(Locale(identifier: "en_GB")))
    }

    func testSkipsUnmappableFirstPreference() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["ko-KR", "ja-JP"],
            supportedLocales: [Locale(identifier: "ja_JP")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertEqual(result, .available(Locale(identifier: "ja_JP")))
    }

    func testMatchesScriptOnlySimplifiedChinese() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["zh-Hans"],
            supportedLocales: [Locale(identifier: "zh_TW"), Locale(identifier: "zh_CN")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertEqual(result, .available(Locale(identifier: "zh_CN")))
    }

    func testMatchesScriptOnlyTraditionalChinese() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["zh-Hant"],
            supportedLocales: [Locale(identifier: "zh_CN"), Locale(identifier: "zh_TW")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertEqual(result, .available(Locale(identifier: "zh_TW")))
    }

    func testMatchesRegionlessLanguageDeterministically() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["en"],
            supportedLocales: [Locale(identifier: "en_US"), Locale(identifier: "en_GB")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertTrue(["en_GB", "en_US"].contains(result.locale.identifier))
    }

    func testFallsBackToSupportedRegionOfSameLanguage() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["en-AU"],
            supportedLocales: [Locale(identifier: "en_US")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertEqual(result, .available(Locale(identifier: "en_US")))
    }

    func testUnavailableResultContainsActualSupportedLocale() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["ja-JP"],
            supportedLocales: [Locale(identifier: "ja_JP")],
            isAvailable: { _ in false },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(result, .unavailable(Locale(identifier: "ja_JP")))
    }

    func testUnsupportedResultReportsFirstPreferredLanguage() {
        let result = SpeechLocale.resolve(
            preferredLanguages: ["xx-YY"],
            supportedLocales: [Locale(identifier: "ja_JP")],
            isAvailable: { _ in true },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(result, .unsupported(Locale(identifier: "xx-YY")))
        XCTAssertFalse(result.isAvailable)
    }

    func testEmptyPreferencesAreUnsupported() {
        let fallback = Locale(identifier: "en_JP")
        let result = SpeechLocale.resolve(
            preferredLanguages: [],
            supportedLocales: [Locale(identifier: "ja_JP")],
            isAvailable: { _ in true },
            fallback: fallback)
        XCTAssertEqual(result, .unsupported(fallback))
    }

    func testSupportedLocaleInputOrderDoesNotAffectSelection() {
        let first = SpeechLocale.resolve(
            preferredLanguages: ["en"],
            supportedLocales: [Locale(identifier: "en_US"), Locale(identifier: "en_GB")],
            isAvailable: { _ in true },
            fallback: .current)
        let second = SpeechLocale.resolve(
            preferredLanguages: ["en"],
            supportedLocales: [Locale(identifier: "en_GB"), Locale(identifier: "en_US")],
            isAvailable: { _ in true },
            fallback: .current)
        XCTAssertEqual(first, second)
    }
}
