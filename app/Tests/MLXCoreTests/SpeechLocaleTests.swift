import XCTest
@testable import MLXCore

final class SpeechLocaleTests: XCTestCase {
    func testTheFirstPreferredLanguageIsTheOneSpoken() {
        XCTAssertEqual(SpeechLocale.reportingLocale(preferredLanguages: ["ja-JP", "en-US"], fallback: Locale(identifier: "en_US")).identifier, "ja-JP")
        XCTAssertEqual(SpeechLocale.reportingLocale(preferredLanguages: [], fallback: Locale(identifier: "fr_FR")).identifier, "fr_FR")
    }
}
