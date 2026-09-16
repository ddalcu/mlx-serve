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

    /// The sweep that wrapped runtime strings in `L10n` once called it with a
    /// key transformed BEFORE lookup (`title.uppercased()`), which can never
    /// hit: the table is keyed on the source literals, so render-time casing
    /// must happen after resolution. Guards the whole class, not the instance —
    /// a "simplification" back to `L10n.text(x.uppercased())` fails here.
    func testNoL10nKeysAreTransformedBeforeLookup() throws {
        let sourcesRoot = Self.catalogURL
            .deletingLastPathComponent()  // zh-Hans.lproj
            .deletingLastPathComponent()  // Resources
            .deletingLastPathComponent()  // MLXServe
            .deletingLastPathComponent()  // Sources
        let pattern = try NSRegularExpression(
            pattern: #"L10n\.text\([A-Za-z0-9_.]+\.uppercased\(\)\)"#)
        var offenders: [String] = []
        let files = try FileManager.default
            .enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)!
            .compactMap { ($0 as? URL)?.pathExtension == "swift" ? $0 as? URL : nil }
        for url in files {
            let text = try String(contentsOf: url, encoding: .utf8)
            let matches = pattern.matches(in: text, range: NSRange(text.startIndex..., in: text))
            if !matches.isEmpty { offenders.append(url.lastPathComponent) }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "L10n.text called with a key that is uppercased before resolution: \(offenders)")
    }

    /// Tray section headers render upper-cased, but every `TraySectionHeader`
    /// title literal must be a translated catalog entry — "translated in the
    /// file, unreachable at runtime" is the failure this pins down.
    func testTraySectionHeaderTitlesAreTranslated() throws {
        let view = Self.catalogURL
            .deletingLastPathComponent()  // zh-Hans.lproj
            .deletingLastPathComponent()  // Resources
            .deletingLastPathComponent()  // MLXServe
            .appendingPathComponent("Views/StatusMenuView.swift")
        let source = try String(contentsOf: view, encoding: .utf8)
        let pattern = try NSRegularExpression(
            pattern: #"TraySectionHeader\(\s*title:\s*"([^"]+)""#)
        let bundle = try XCTUnwrap(
            Bundle(path: Self.catalogURL.deletingLastPathComponent().path))
        var titles: [String] = []
        for match in pattern.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
            let range = try XCTUnwrap(Range(match.range(at: 1), in: source))
            titles.append(String(source[range]))
        }
        XCTAssertEqual(Set(titles), ["Server", "In Memory", "Media Generation"],
                       "header call sites changed shape; update the expected set")
        for title in titles {
            XCTAssertNotEqual(bundle.localizedString(forKey: title, value: title, table: nil),
                              title, "tray header \"\(title)\" is listed but not translated")
        }
    }

    private static func specs(_ text: String) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        return specifier.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }
}
