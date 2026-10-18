import AppKit
import XCTest
@testable import MLXCore

final class GenerationPercentageTests: XCTestCase {
    func testEveryLocalizationKeepsThreeDigitPercentageWidth() throws {
        let resources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Resources")
        let font = NSFont.monospacedSystemFont(ofSize: 10, weight: .medium)
        for language in ["en", "ko", "zh-Hans"] {
            let bundle = language == "en" ? nil : try XCTUnwrap(Bundle(url: resources.appendingPathComponent("\(language).lproj")))
            for key in ["GPU %3lld%%", "Mem %3lld%%"] {
                let format = bundle?.localizedString(forKey: key, value: key, table: nil) ?? key
                let widths = [0, 9, 10, 99, 100].map { value in
                    let text = String(format: format, Int64(value))
                    return (text as NSString).size(withAttributes: [.font: font]).width
                }
                XCTAssertEqual(try XCTUnwrap(widths.min()), try XCTUnwrap(widths.max()), accuracy: 0.01, "\(language): \(key)")
                XCTAssertTrue(String(format: format, Int64(7)).contains("  7%"))
            }
        }
    }
}
