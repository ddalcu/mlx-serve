import XCTest
@testable import MLXCore

/// Unit tests for `SpeechLocale` — the pure locale-resolution layer behind Voice
/// mode. Pins the bug fix: recognition must key off the user's *preferred*
/// languages (canonicalized to Apple's recognizer keys), never the bundle-derived
/// `Locale.current`. The live `SFSpeechRecognizer` probe is the untestable shell.
final class SpeechLocaleTests: XCTestCase {

    // MARK: canonicalKey — two-step normalization (the crux)

    func testCanonicalKeyJapaneseNeedsExplicitUnderscore() {
        // Locale(identifier: "ja-JP") is NOT canonicalized → needs the -→_ step.
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "ja-JP"), "ja_JP")
    }

    func testCanonicalKeyCollapsesScriptRegionTags() {
        // A naive -→_ alone leaves zh_Hans_CN (not a real key); Locale(identifier:)
        // collapses it to Apple's recognizer key.
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "zh-Hans-CN"), "zh_CN")
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "zh-Hant-TW"), "zh_TW")
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "en-US"), "en_US")
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "yue-Hant-HK"), "yue_HK")
    }

    func testCanonicalKeyLeavesScriptOnlyTagUnkeyed() {
        // No region to key on → stays zh-Hans, filtered out downstream.
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "zh-Hans"), "zh-Hans")
    }

    func testCanonicalKeyRegionless() {
        XCTAssertEqual(SpeechLocale.canonicalKey(from: "en"), "en")
    }

    // MARK: candidateKeys — ordering + dedup

    func testCandidateKeysPreservesOrderAndDedups() {
        let keys = SpeechLocale.candidateKeys(preferredLanguages: ["ja-JP", "en-US", "ja_JP"])
        XCTAssertEqual(keys, ["ja_JP", "en_US"])  // ordered, dup collapsed
    }

    func testCandidateKeysEmpty() {
        XCTAssertEqual(SpeechLocale.candidateKeys(preferredLanguages: []), [])
    }

    // MARK: resolve — contract 1 (installed wins, in user's priority order)

    func testResolvePicksFirstInstalledCandidate() {
        // ja_JP installed → chosen even though en_US is also installed.
        let locale = SpeechLocale.resolve(
            preferredLanguages: ["ja-JP", "en-US"],
            supportsOnDevice: { $0.identifier == "ja_JP" },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(locale.identifier, "ja_JP")
    }

    func testResolveSkipsUninstalledFirstAndTakesSecond() {
        // First preferred (en_GB) not installed → falls to second (en_US). This is
        // the whole point: the user's *installed* language wins, not the bundle.
        let locale = SpeechLocale.resolve(
            preferredLanguages: ["en-GB", "en-US"],
            supportsOnDevice: { $0.identifier == "en_US" },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(locale.identifier, "en_US")
    }

    // MARK: resolve — contract 2 (nothing installed → name the user's own language)

    func testResolveFallsBackToFirstPreferredWhenNoneInstalled() {
        // Nothing installed → return first candidate (ja_JP) so the card names the
        // user's language, NOT the bundle-derived fallback (en_JP).
        let locale = SpeechLocale.resolve(
            preferredLanguages: ["ja-JP"],
            supportsOnDevice: { _ in false },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(locale.identifier, "ja_JP")
    }

    // MARK: resolve — degenerate inputs

    func testResolveEmptyPreferredUsesFallback() {
        let locale = SpeechLocale.resolve(
            preferredLanguages: [],
            supportsOnDevice: { _ in true },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(locale.identifier, "en_JP")
    }

    // MARK: regression — the en_JP bug

    func testBugFixJapaneseUserNeverResolvesToEnJP() {
        // The reported bug: Japanese system, bundle ran as en_JP. preferredLanguages
        // is ["ja-JP"] and ja_JP is installed → must resolve to ja_JP, never en_JP.
        let locale = SpeechLocale.resolve(
            preferredLanguages: ["ja-JP"],
            supportsOnDevice: { $0.identifier == "ja_JP" },
            fallback: Locale(identifier: "en_JP"))
        XCTAssertEqual(locale.identifier, "ja_JP")
        XCTAssertNotEqual(locale.identifier, "en_JP")
    }
}
