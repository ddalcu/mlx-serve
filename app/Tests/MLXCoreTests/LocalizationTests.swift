import XCTest
@testable import MLXCore

/// The zh-Hans catalog and the lookups it feeds.
///
/// The catalog is read from the source tree (`#filePath`) rather than the test
/// bundle: SwiftPM excludes `Sources/MLXServe/Resources`, which `app/build.sh`
/// copies into the app bundle's `Contents/Resources` — the same place
/// `Bundle.main` (and therefore SwiftUI's own literal lookup) reads it from.
final class LocalizationTests: XCTestCase {

    private static let catalogURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // MLXCoreTests
        .deletingLastPathComponent()   // Tests
        .deletingLastPathComponent()   // app
        .appendingPathComponent("Sources/MLXServe/Resources/zh-Hans.lproj/Localizable.strings")

    private static let specifier = try! NSRegularExpression(
        pattern: #"%(?:\d+\$)?[-+ #0]*\d*(?:\.\d+)?(?:ll|l|h|hh|z|t|q)?[dioufFeEgGxXcs@p%]"#)

    private func catalog() throws -> [(key: String, value: String)] {
        let text = try String(contentsOf: Self.catalogURL, encoding: .utf8)
        return text.split(separator: "\n").compactMap { line -> (String, String)? in
            let line = line.trimmingCharacters(in: .whitespaces)
            guard line.hasPrefix("\""), let (key, rest) = Self.takeQuoted(line.dropFirst()) else { return nil }
            let tail = rest.trimmingCharacters(in: .whitespaces)
            guard tail.hasPrefix("=") else { return nil }
            let afterEquals = tail.dropFirst().trimmingCharacters(in: .whitespaces)
            guard afterEquals.hasPrefix("\""), let (value, _) = Self.takeQuoted(afterEquals.dropFirst()) else { return nil }
            return (key, value)
        }
    }

    /// Reads one `"…"` literal, honouring backslash escapes, and returns it with the remainder.
    private static func takeQuoted(_ text: Substring) -> (String, Substring)? {
        var out = "", escaped = false
        var index = text.startIndex
        while index < text.endIndex {
            let ch = text[index]
            if escaped {
                switch ch {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "\\", "\"": out.append(ch)
                default: out.append("\\"); out.append(ch)
                }
                escaped = false
            } else if ch == "\\" {
                escaped = true
            } else if ch == "\"" {
                return (out, text[text.index(after: index)...])
            } else {
                out.append(ch)
            }
            index = text.index(after: index)
        }
        return nil
    }

    func testCatalogParsesAndEveryEntryHasText() throws {
        let entries = try catalog()
        XCTAssertGreaterThan(entries.count, 500, "the zh-Hans catalog looks truncated")
        for (key, value) in entries {
            XCTAssertFalse(key.isEmpty, "empty key")
            XCTAssertFalse(value.trimmingCharacters(in: .whitespaces).isEmpty, "empty translation for \(key)")
            XCTAssertNotEqual(key, value, "\(key) is listed but not translated")
        }
    }

    func testCatalogIsChinese() throws {
        let entries = try catalog()
        let withoutCJK = entries.filter { entry in
            !entry.value.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
        }
        // Model-variant labels ("Gemma 4 E2B (4-bit)"), "Agent" and "API key" stay
        // Latin by design; anything else without Chinese is a leftover.
        XCTAssertLessThanOrEqual(withoutCJK.count, 25,
                                 "untranslated: \(withoutCJK.map(\.key).prefix(12))")
    }

    func testPlaceholdersSurviveTranslation() throws {
        for (key, value) in try catalog() {
            let keySpecs = Self.specs(key), valueSpecs = Self.specs(value)
            for spec in valueSpecs {
                XCTAssertTrue(keySpecs.contains(spec),
                              "\(key) → \(value): translation adds \(spec)")
            }
            XCTAssertLessThanOrEqual(valueSpecs.count, keySpecs.count,
                                     "\(key) → \(value): more placeholders than the source string")
        }
    }

    /// The file is a working table: resolving it through a bundle returns the
    /// same strings the catalog declares, escapes and all.
    func testCatalogResolvesThroughBundle() throws {
        let bundle = try XCTUnwrap(Bundle(path: Self.catalogURL.deletingLastPathComponent().path),
                                   "zh-Hans.lproj does not load as a bundle")
        for (key, value) in try catalog() {
            XCTAssertEqual(bundle.localizedString(forKey: key, value: nil, table: nil), value,
                           "bundle lookup disagrees with the catalog for \(key)")
        }
    }

    func testUnknownKeysAndFormattingFallBackToEnglish() {
        XCTAssertEqual(L10n.text("Not a key in any catalog"), "Not a key in any catalog")
        XCTAssertEqual(L10n.format("Download %@ (%lld MB)", "flux", 512), "Download flux (512 MB)")
    }

    private static func specs(_ text: String) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        return specifier.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }
}
