import XCTest
import Foundation

final class KoreanLocalizationTests: XCTestCase {
    private var resources: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Resources")
    }

    func testKoreanCatalogResolvesAndFormats() throws {
        let url = resources.appendingPathComponent("ko.lproj")
        let data = try Data(contentsOf: url.appendingPathComponent("Localizable.strings"))
        let entries = try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: String])
        let bundle = try XCTUnwrap(Bundle(url: url))
        XCTAssertGreaterThan(entries.count, 2000)
        for (key, value) in entries {
            XCTAssertFalse(value.isEmpty, key)
            XCTAssertEqual(bundle.localizedString(forKey: key, value: nil, table: nil), value, key)
        }
        XCTAssertEqual(bundle.localizedString(forKey: "Settings", value: nil, table: nil), "설정")
        XCTAssertEqual(bundle.localizedString(forKey: "unknown-key", value: nil, table: nil), "unknown-key")
        XCTAssertEqual(String(format: bundle.localizedString(forKey: "Download %@ (%lld MB)", value: nil, table: nil), "model", 512), "model 다운로드 (512 MB)")
    }

    func testFormatArgumentsKeepTheirPositionsAndTypes() throws {
        let data = try Data(contentsOf: resources.appendingPathComponent("ko.lproj/Localizable.strings"))
        let entries = try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: String])
        let pattern = try NSRegularExpression(pattern: #"(?<![0-9])%(?:(\d+)\$)?[-+ #0]*\d*(?:\.\d+)?(ll|l|h|hh|z|t|q)?([dioufFeEgGxXcs@p%])"#)
        func arguments(_ text: String, source: Bool) -> [Int: String] {
            var next = 1, result: [Int: String] = [:]
            for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
                let conversion = (text as NSString).substring(with: match.range(at: 3))
                if conversion == "%" { continue }
                let position = match.range(at: 1).location == NSNotFound
                    ? next : Int((text as NSString).substring(with: match.range(at: 1)))!
                next += 1
                let prefix = (text as NSString).substring(to: match.range.location)
                // Korean drops the English plural suffix, not a content argument.
                if source && conversion == "@" && ["tool", "model", "session", "token", "source", "SOURCE", "rung"].contains(where: { prefix.hasSuffix($0) }) { continue }
                let length = match.range(at: 2).location == NSNotFound ? "" : (text as NSString).substring(with: match.range(at: 2))
                result[position] = length + conversion
            }
            return result
        }
        for (key, value) in entries {
            XCTAssertEqual(arguments(key, source: true), arguments(value, source: false), key)
        }
    }

    func testKoreanPermissionDescriptionsResolve() throws {
        let bundle = try XCTUnwrap(Bundle(url: resources.appendingPathComponent("ko.lproj")))
        for key in ["NSAppleEventsUsageDescription", "NSLocalNetworkUsageDescription", "NSMicrophoneUsageDescription", "NSSpeechRecognitionUsageDescription"] {
            let value = bundle.localizedString(forKey: key, value: nil, table: "InfoPlist")
            XCTAssertNotEqual(value, key)
            XCTAssertTrue(value.unicodeScalars.contains { (0xAC00...0xD7A3).contains($0.value) })
        }
    }
}
