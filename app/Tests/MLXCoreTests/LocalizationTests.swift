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
        // Model-variant labels ("Gemma 4 E2B (4-bit)") and "API key" stay
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

    // MARK: - Lookup plumbing

    /// Every Swift file under `Sources`, by name and text.
    private static func sourceFiles() throws -> [(name: String, text: String)] {
        let root = catalogURL
            .deletingLastPathComponent()  // zh-Hans.lproj
            .deletingLastPathComponent()  // Resources
            .deletingLastPathComponent()  // MLXServe
            .deletingLastPathComponent()  // Sources
        let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)!
        return try walker.compactMap { element in
            guard let url = element as? URL, url.pathExtension == "swift" else { return nil }
            return (url.lastPathComponent, try String(contentsOf: url, encoding: .utf8))
        }
    }

    /// Every call in one file, keyed by callee: `row(` → "row", `L10n.text(` →
    /// "L10n.text". Argument text keeps balanced parens, so a nested call cannot
    /// truncate the scan, and identifier boundaries are honoured — `row(` never
    /// matches `someRow(`. Built once per file: scanning per helper is what made
    /// this test take 18 seconds.
    private static func callIndex(in text: String) -> [String: [String]] {
        let characters = Array(text)
        var index: [String: [String]] = [:]
        var cursor = 0
        while cursor < characters.count {
            guard characters[cursor].isLetter || characters[cursor] == "_" else {
                cursor += 1
                continue
            }
            var name = ""
            var end = cursor
            while end < characters.count {
                let character = characters[end]
                if character.isLetter || character.isNumber || character == "_" {
                    name.append(character)
                    end += 1
                } else if character == ".", end + 1 < characters.count,
                          characters[end + 1].isLetter || characters[end + 1] == "_" {
                    name.append(character)
                    end += 1
                } else {
                    break
                }
            }
            guard end < characters.count, characters[end] == "(" else {
                cursor = max(end, cursor + 1)
                continue
            }
            var depth = 0
            var close = end
            while close < characters.count {
                if characters[close] == "(" { depth += 1 }
                if characters[close] == ")" {
                    depth -= 1
                    if depth == 0 { break }
                }
                close += 1
            }
            if close < characters.count {
                index[name, default: []].append(String(characters[(end + 1)..<close]))
            }
            // Keep walking INSIDE the argument list: most L10n calls are nested
            // in Text(...)/Label(...) and would otherwise be skipped.
            cursor = end + 1
        }
        return index
    }

    /// `_ title: String` → `("_", "title")`; `label: String = ""` → `("label", "label")`.
    private static func parameters(_ signature: String) -> [(label: String, parameter: String)] {
        var parts: [String] = []
        var depth = 0
        var current = ""
        for character in signature {
            switch character {
            case "(", "<", "[":
                depth += 1
                current.append(character)
            case ")", ">", "]":
                depth -= 1
                current.append(character)
            case "," where depth == 0:
                parts.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        parts.append(current)
        return parts.compactMap { part in
            guard let colon = part.firstIndex(of: ":") else { return nil }
            let tokens = part[..<colon].split(separator: " ").map(String.init)
            guard let first = tokens.first else { return nil }
            return (first, tokens.count > 1 ? tokens[tokens.count - 1] : first)
        }
    }

    /// Helpers whose body looks one of their parameters up in the catalog:
    /// the function name, that parameter, and the external label call sites use.
    private static func lookupHelpers(
        in files: [(name: String, text: String)]
    ) -> [(name: String, label: String)] {
        let lookup = try! NSRegularExpression(
            pattern: #"L10n\.(?:text|format)\(\s*([A-Za-z_]\w*)\s*[,)]"#)
        let definition = try! NSRegularExpression(pattern: #"func\s+(\w+)\s*\(([^)]*)\)"#)
        var seen = Set<String>()
        var out: [(String, String)] = []
        for file in files {
            let ns = file.text as NSString
            for match in lookup.matches(in: file.text, range: NSRange(location: 0, length: ns.length)) {
                let parameter = ns.substring(with: match.range(at: 1))
                let before = NSRange(location: 0, length: match.range.location)
                guard let definition = definition.matches(in: file.text, range: before).last else { continue }
                let name = ns.substring(with: definition.range(at: 1))
                let signature = ns.substring(with: definition.range(at: 2))
                for (label, candidate) in parameters(signature) where candidate == parameter {
                    if seen.insert("\(name)(\(parameter))").inserted {
                        out.append((name, label))
                    }
                }
            }
        }
        return out
    }

    /// The literal passed as `label` — or positionally when the parameter is
    /// unlabelled — inside one call's argument text.
    private static func literalArgument(_ arguments: String, label: String) -> String? {
        if label == "_" {
            let trimmed = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("\"") else { return nil }
            return takeQuoted(trimmed.dropFirst()).map(\.0)
        }
        let pattern = try! NSRegularExpression(
            pattern: #"\b\#(NSRegularExpression.escapedPattern(for: label)):\s*\""#)
        let ns = arguments as NSString
        guard let match = pattern.firstMatch(in: arguments, range: NSRange(location: 0, length: ns.length))
        else { return nil }
        let rest = arguments[Range(match.range, in: arguments)!.upperBound...]
        return takeQuoted(rest).map(\.0)
    }

    // MARK: - Lookup guards

    /// A key has to be *written* at the call site: a literal, an expression that
    /// carries one, a plain member path (a runtime label the catalog may name),
    /// or an allow-listed producer. Anything else — a formatted number, a
    /// duration, a size, a path assembled from parts — is a value the table can
    /// never hold, so the lookup is dead work. Checking transform spellings
    /// alone is what let `state?.percentFormatted ?? ""` through: nothing there
    /// is uppercased, wrapped or formatted, it simply cannot be a key.
    func testEveryLookupArgumentCanBeAKey() throws {
        var calls = 0
        var offenders: [String] = []
        for file in try Self.sourceFiles() {
            let index = Self.callIndex(in: file.text)
            for arguments in (index["L10n.text"] ?? []) + (index["L10n.format"] ?? []) {
                calls += 1
                let key = Self.firstArgument(arguments)
                if Self.looksLikeValue(key) {
                    offenders.append("\(file.name): L10n(\(key.trimmingCharacters(in: .whitespacesAndNewlines)))")
                }
            }
        }
        XCTAssertGreaterThan(calls, 200, "the scan stopped matching L10n call sites")
        XCTAssertTrue(offenders.isEmpty,
                      "lookup arguments that can never be a key:\n\(offenders.joined(separator: "\n"))")
    }

    /// The first argument of a call, split at a top-level comma and honouring
    /// string literals, so a comma inside copy does not truncate it.
    private static func firstArgument(_ arguments: String) -> String {
        let characters = Array(arguments)
        var depth = 0
        var index = 0
        while index < characters.count {
            let character = characters[index]
            if character == "\"" {
                var cursor = index + 1
                while cursor < characters.count, characters[cursor] != "\"" {
                    cursor += characters[cursor] == "\\" ? 2 : 1
                }
                index = cursor
            } else if "([{".contains(character) {
                depth += 1
            } else if ")]}".contains(character) {
                depth -= 1
            } else if character == ",", depth == 0 {
                return String(characters[0..<index])
            }
            index += 1
        }
        return arguments
    }

    private static func looksLikeValue(_ key: String) -> Bool {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        // A transformed value can never be a key, whatever else it contains.
        for transform in ["String(", ".uppercased(", ".lowercased(", ".capitalized", ".formatted("]
        where trimmed.contains(transform) { return true }
        if isLiteral(trimmed) || isPlainPath(trimmed) || containsLiteral(trimmed) { return false }
        return !labelProducers.contains { trimmed.contains($0) }
    }

    /// `"…"` with something in it and no interpolation.
    private static func isLiteral(_ text: String) -> Bool {
        guard text.hasPrefix("\"") else { return false }
        guard let (body, _) = takeQuoted(text.dropFirst()) else { return false }
        return !body.isEmpty && !body.contains("\\(")
    }

    /// `title`, `m.lanDisplayName`, `state?.percentFormatted`.
    private static func isPlainPath(_ text: String) -> Bool {
        let allowed = Set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.? ")
        return !text.isEmpty && text.allSatisfy { allowed.contains($0) } && !text.contains("??")
    }

    /// `copied ? "Copied" : "Copy"` — a literal spelling the value out.
    private static func containsLiteral(_ text: String) -> Bool {
        var rest = Substring(text)
        while let quote = rest.firstIndex(of: "\"") {
            guard let (body, remainder) = takeQuoted(rest[rest.index(after: quote)...]) else { break }
            if !body.isEmpty && !body.contains("\\(") { return true }
            rest = remainder
        }
        return false
    }

    /// Producers whose branches do yield catalog keys — model and variant
    /// labels, tool names, preflight copy, status lines. Adding an entry needs
    /// the same evidence as adding a key: a branch that resolves in the table.
    private static let labelProducers = [
        "actionLabel", "actionTitle", "buttonLabel", "caption",
        "controller.partialTranscript", "defect?.explanation", "detail", "displayName",
        "engineExplainer", "label", "lead", "modeLabel", "modelPickerLabel", "placeholder", "pretty",
        "run.summary ?? run.status.label", "shortMessage", "startupModelLabel", "statusLine", "title",
    ]

    /// Every string literal that reaches an `L10n` lookup through a helper
    /// parameter has to exist in the catalog. Scoped to one helper in one file
    /// it missed `sectionLabel` in another — this walks the helpers out of the
    /// sources instead of a hand-written list.
    ///
    /// An *interpolated* literal is checked the same way, and can never pass:
    /// the helper looks its parameter up in the table, so a caller that builds
    /// the sentence first (`fileChip(detail: "PDF · \(count) chars")`) hands the
    /// lookup a finished string no key can match. The format has to run at the
    /// producer (`L10n.format`) and the helper render the resolved text — that
    /// is the defect this arm of the test exists to catch.
    ///
    /// `latinLabels` are values that are deliberately rendered in Latin — HTTP
    /// verbs and sampling symbols — where the lookup falls through to the
    /// literal by design.
    func testEveryLiteralReachingALookupHelperIsTranslated() throws {
        let files = try Self.sourceFiles()
        let calls = files.map { (name: $0.name, index: Self.callIndex(in: $0.text)) }
        let keys = Set(try catalog().map(\.key))
        let latinLabels: Set<String> = ["BASE", "Top-p"]
        var sites = 0
        var missing: [String] = []
        for helper in Self.lookupHelpers(in: files) {
            for file in calls {
                for arguments in file.index[helper.name] ?? [] {
                    guard let literal = Self.literalArgument(arguments, label: helper.label) else { continue }
                    sites += 1
                    if latinLabels.contains(literal) || keys.contains(literal) { continue }
                    missing.append("\(file.name): \"\(literal)\" reaching \(helper.name)")
                }
            }
        }
        XCTAssertGreaterThan(sites, 30, "the scan stopped finding literal call sites")
        XCTAssertTrue(missing.isEmpty,
                      "literals reaching a lookup with no catalog entry:\n\(missing.joined(separator: "\n"))")
    }

    private static func specs(_ text: String) -> [String] {
        let range = NSRange(text.startIndex..., in: text)
        return specifier.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map { String(text[$0]) }
        }
    }
}
