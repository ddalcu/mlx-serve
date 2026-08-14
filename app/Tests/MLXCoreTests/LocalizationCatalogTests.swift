import XCTest
@testable import MLXCore

/// Every user-facing string the app draws must exist in the String Catalog with
/// a Simplified Chinese translation.
///
/// The app has no localization *call sites*: SwiftUI's `Text("…")`,
/// `Button("…")`, `.help("…")` and friends take a `LocalizedStringKey` for a
/// string LITERAL, so they already perform a `Bundle.main` lookup at render
/// time. Adding `zh-Hans.lproj/Localizable.strings` to the bundle localizes
/// them with no code change, and a missing entry falls through to the key —
/// which IS the English text. That fallback is what makes this class of bug
/// silent: an untranslated string doesn't crash or blank out, it just renders
/// in English inside a Chinese screen, and only a Chinese speaker looking at
/// that exact pane would ever notice.
///
/// So the guard is coverage, not behaviour: extract the keys from the sources
/// the way SwiftUI does, and demand the catalog answer every one of them. The
/// extractor lives HERE rather than in a script, because a scan that only runs
/// when someone remembers to run it is not a guard.
///
/// Deliberately NOT covered: strings that are model-facing rather than
/// user-facing (system prompts, tool names and descriptions) — see
/// `testModelFacingCopyIsNeverLocalized`.
final class LocalizationCatalogTests: XCTestCase {

    // MARK: - Locations

    private var appRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // app
    }

    private var sourcesRoot: URL { appRoot.appendingPathComponent("Sources/MLXServe") }
    private var catalogURL: URL { appRoot.appendingPathComponent("Localization/Localizable.xcstrings") }

    // MARK: - The call sites whose first string literal is a LocalizedStringKey

    /// Spelled as the text that immediately precedes the `(`. A view or
    /// modifier absent from this list is invisible to the audit, so new copy
    /// surfaces get added here, not worked around.
    private static let localizedInitializers = [
        "Text", "Button", "Label", "Toggle", "Picker", "Section", "TextField",
        "SecureField", "Menu", "Stepper", "Link", "DisclosureGroup", "GroupBox",
        "CommandMenu", "Alert",
    ]

    /// Modifiers taking a `LocalizedStringKey` first argument.
    private static let localizedModifiers = [
        ".help", ".navigationTitle", ".navigationSubtitle", ".alert",
        ".confirmationDialog",
    ]

    // MARK: - Extraction

    /// A Swift string literal starting at `open` (the opening quote), returned
    /// with its interpolations intact and its escapes unresolved — the shape a
    /// `LocalizedStringKey` is spelled with in source. Nil for a multiline or
    /// unterminated literal.
    private func literal(in text: String, openingAt open: String.Index) -> String? {
        var i = text.index(after: open)
        var out = ""
        var depth = 0            // nesting inside \( … )
        while i < text.endIndex {
            let c = text[i]
            if c == "\\" {
                let next = text.index(after: i)
                guard next < text.endIndex else { return nil }
                if text[next] == "(" { depth += 1 }
                out.append(c)
                out.append(text[next])
                i = text.index(after: next)
                continue
            }
            if depth > 0 {
                if c == "(" { depth += 1 }
                if c == ")" { depth -= 1 }
                out.append(c)
                i = text.index(after: i)
                continue
            }
            if c == "\"" { return out }
            if c == "\n" { return nil }
            out.append(c)
            i = text.index(after: i)
        }
        return nil
    }

    /// Source with `//` line comments removed, so commented-out code and the
    /// prose in doc comments never reach the extractor.
    private func stripLineComments(_ source: String) -> String {
        var out = ""
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            var inString = false
            var i = line.startIndex
            var kept = ""
            while i < line.endIndex {
                let c = line[i]
                if c == "\\", inString {
                    kept.append(c)
                    let next = line.index(after: i)
                    if next < line.endIndex { kept.append(line[next]); i = line.index(after: next) }
                    else { i = next }
                    continue
                }
                if c == "\"" { inString.toggle() }
                if !inString, c == "/" {
                    let next = line.index(after: i)
                    if next < line.endIndex, line[next] == "/" { break }
                }
                kept.append(c)
                i = line.index(after: i)
            }
            out += kept + "\n"
        }
        return out
    }

    /// Every localizable key in one file, with the call site that produced it.
    private func keys(in source: String) -> [String] {
        let text = stripLineComments(source)
        var found: [String] = []
        let prefixes = Self.localizedInitializers.map { $0 } + Self.localizedModifiers.map { $0 }

        for prefix in prefixes {
            var search = text.startIndex
            while let hit = text.range(of: prefix, range: search ..< text.endIndex) {
                search = hit.upperBound
                // An initializer must not be the tail of a longer identifier
                // (`AttributedText(` must not read as `Text(`).
                if !prefix.hasPrefix("."), hit.lowerBound > text.startIndex {
                    let before = text[text.index(before: hit.lowerBound)]
                    if before.isLetter || before.isNumber || before == "_" || before == "." { continue }
                }
                // Skip whitespace to the `(`, then to the first argument.
                var i = hit.upperBound
                while i < text.endIndex, text[i] == " " { i = text.index(after: i) }
                guard i < text.endIndex, text[i] == "(" else { continue }
                i = text.index(after: i)
                while i < text.endIndex, text[i] == " " || text[i] == "\n" { i = text.index(after: i) }
                guard i < text.endIndex, text[i] == "\"" else { continue }
                guard let key = literal(in: text, openingAt: i) else { continue }
                if key.trimmingCharacters(in: .whitespaces).isEmpty { continue }
                found.append(key)
            }
        }
        return found
    }

    private func swiftFiles(under root: URL) throws -> [URL] {
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return [] }
        return e.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }.sorted { $0.path < $1.path }
    }

    /// Keys that carry no translatable prose — a bare interpolation, a number,
    /// punctuation. `Text("\(index + 1)")` is a value, not a sentence, and
    /// demanding a Chinese translation for it would be noise in the catalog.
    private func carriesProse(_ key: String) -> Bool {
        // Remove interpolated segments, then ask whether any letters survive.
        var out = ""
        var i = key.startIndex
        var depth = 0
        while i < key.endIndex {
            let c = key[i]
            if c == "\\", key.index(after: i) < key.endIndex, key[key.index(after: i)] == "(" {
                depth += 1
                i = key.index(i, offsetBy: 2)
                continue
            }
            if depth > 0 {
                if c == "(" { depth += 1 }
                if c == ")" { depth -= 1 }
                i = key.index(after: i)
                continue
            }
            out.append(c)
            i = key.index(after: i)
        }
        return out.contains { $0.isLetter }
    }

    // MARK: - Source spelling → runtime lookup key

    /// The catalog key is the string the RUNTIME holds, not the way it is
    /// spelled in source: `Text("say \\"hi\\"")` looks up `say "hi"`.
    private func unescape(_ s: String) -> String {
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            let c = s[i]
            guard c == "\\", s.index(after: i) < s.endIndex else {
                out.append(c); i = s.index(after: i); continue
            }
            let n = s[s.index(after: i)]
            let simple: [Character: Character] = ["n": "\n", "t": "\t", "r": "\r",
                                                  "\"": "\"", "'": "'", "\\": "\\", "0": "\0"]
            if let r = simple[n] {
                out.append(r); i = s.index(i, offsetBy: 2); continue
            }
            if n == "u", s.index(i, offsetBy: 2) < s.endIndex, s[s.index(i, offsetBy: 2)] == "{",
               let close = s[s.index(i, offsetBy: 3)...].firstIndex(of: "}") {
                let hex = String(s[s.index(i, offsetBy: 3) ..< close])
                if let v = UInt32(hex, radix: 16), let scalar = Unicode.Scalar(v) {
                    out.append(Character(scalar))
                    i = s.index(after: close)
                    continue
                }
            }
            out.append(c); i = s.index(after: i)
        }
        return out
    }

    /// SwiftUI turns an interpolated literal into a FORMAT string —
    /// `"Plan (\\(n) steps)"` is looked up as `"Plan (%lld steps)"`, with the
    /// specifier chosen by the interpolated expression's TYPE. A source scan
    /// cannot know that type, so instead of guessing it, this builds the
    /// pattern every possible spelling would match and asks the catalog which
    /// key answers it. A literal `%` doubles once a string has any formatting,
    /// so it matches either way.
    private func lookupPattern(forSourceSpelling key: String) -> NSRegularExpression? {
        var pattern = "^"
        var literal = ""
        var i = key.startIndex
        var depth = 0
        func flush() {
            guard !literal.isEmpty else { return }
            for chunk in unescape(literal).split(separator: "%", omittingEmptySubsequences: false).map(String.init).enumerated() {
                if chunk.offset > 0 { pattern += "%%?" }
                pattern += NSRegularExpression.escapedPattern(for: chunk.element)
            }
            literal = ""
        }
        while i < key.endIndex {
            let c = key[i]
            if c == "\\", key.index(after: i) < key.endIndex, key[key.index(after: i)] == "(" {
                flush()
                pattern += "%(?:lld|llu|lf|[dfu@])"
                depth = 1
                i = key.index(i, offsetBy: 2)
                continue
            }
            if depth > 0 {
                if c == "(" { depth += 1 }
                if c == ")" { depth -= 1 }
                i = key.index(after: i)
                continue
            }
            literal.append(c)
            i = key.index(after: i)
        }
        flush()
        pattern += "$"
        return try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators])
    }

    /// The catalog keys a source spelling can resolve to at runtime.
    private func catalogKeys(matching key: String, in catalog: Catalog) -> [String] {
        guard key.contains("\\(") else {
            let exact = unescape(key)
            return catalog.strings[exact] != nil ? [exact] : []
        }
        guard let rx = lookupPattern(forSourceSpelling: key) else { return [] }
        return catalog.strings.keys.filter { candidate in
            let range = NSRange(candidate.startIndex ..< candidate.endIndex, in: candidate)
            return rx.firstMatch(in: candidate, range: range) != nil
        }
    }

    // MARK: - Catalog

    private struct Catalog {
        var sourceLanguage: String
        /// key → language → (state, value)
        var strings: [String: [String: (state: String, value: String)]]
    }

    private func loadCatalog() throws -> Catalog {
        let data = try Data(contentsOf: catalogURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("catalog is not a JSON object")
        }
        let source = root["sourceLanguage"] as? String ?? ""
        var out: [String: [String: (state: String, value: String)]] = [:]
        let strings = root["strings"] as? [String: Any] ?? [:]
        for (key, raw) in strings {
            var perLanguage: [String: (state: String, value: String)] = [:]
            if let entry = raw as? [String: Any],
               let locs = entry["localizations"] as? [String: Any] {
                for (lang, l) in locs {
                    if let l = l as? [String: Any], let unit = l["stringUnit"] as? [String: Any] {
                        perLanguage[lang] = (state: unit["state"] as? String ?? "",
                                             value: unit["value"] as? String ?? "")
                    }
                }
            }
            out[key] = perLanguage
        }
        return Catalog(sourceLanguage: source, strings: out)
    }

    // MARK: - Tests

    func testTheCatalogDeclaresEnglishAsItsSourceLanguage() throws {
        let catalog = try loadCatalog()
        XCTAssertEqual(catalog.sourceLanguage, "en",
                       "Keys are the English source strings, so a missing entry falls back to English.")
    }

    /// The audit itself has to work, and it is easy for a rewrite of the
    /// literal scanner to quietly start returning nothing — which would make
    /// every coverage test below pass forever.
    func testTheExtractorReadsTheShapesTheAppActuallyUses() {
        let sample = """
        struct S: View {
            var body: some View {
                Text("Plain")
                Button("Tap me") { }
                Label("Titled", systemImage: "gear")
                Text("Count \\(items.count) here")
                Text(someVariable)
                Text("Escaped \\" quote")
                AttributedText("not a Text call")
                thing.help("A tip")
                    .navigationTitle("Pane")
                // Text("commented out")
                Text("")
            }
        }
        """
        let found = keys(in: sample)
        XCTAssertTrue(found.contains("Plain"))
        XCTAssertTrue(found.contains("Tap me"))
        XCTAssertTrue(found.contains("Titled"))
        XCTAssertTrue(found.contains("Count \\(items.count) here"))
        XCTAssertTrue(found.contains("A tip"))
        XCTAssertTrue(found.contains("Pane"))
        XCTAssertTrue(found.contains("Escaped \\\" quote"))
        XCTAssertFalse(found.contains("not a Text call"), "A longer identifier ending in Text is not a Text call.")
        XCTAssertFalse(found.contains("commented out"), "Commented-out code is not a call site.")
        XCTAssertFalse(found.contains(""), "An empty literal is not copy.")
        XCTAssertFalse(found.contains("someVariable"), "A variable is not a LocalizedStringKey.")
    }

    func testThePunctuationOnlyFilterKeepsSentencesAndDropsBareValues() {
        XCTAssertTrue(carriesProse("Download"))
        XCTAssertTrue(carriesProse("Plan (\\(plan.steps.count) steps)"))
        XCTAssertFalse(carriesProse("\\(index + 1)"))
        XCTAssertFalse(carriesProse("\\(a) / \\(b)"))
        XCTAssertFalse(carriesProse("·"))
    }

    /// The coverage guard. A new English string with no Chinese translation
    /// fails here, naming itself.
    func testEveryUserFacingStringHasASimplifiedChineseTranslation() throws {
        let catalog = try loadCatalog()
        var missing: [String] = []
        var untranslated: [String] = []

        for file in try swiftFiles(under: sourcesRoot) {
            let source = try String(contentsOf: file, encoding: .utf8)
            for key in keys(in: source) where carriesProse(key) {
                let resolved = catalogKeys(matching: key, in: catalog)
                guard !resolved.isEmpty else {
                    missing.append("\(file.lastPathComponent): \"\(key)\"")
                    continue
                }
                for candidate in resolved {
                    guard let zh = catalog.strings[candidate]?["zh-Hans"],
                          zh.state == "translated", !zh.value.isEmpty else {
                        untranslated.append("\(file.lastPathComponent): \"\(candidate)\"")
                        continue
                    }
                }
            }
        }

        XCTAssertTrue(missing.isEmpty,
                      "\(missing.count) string(s) are drawn by the app but absent from Localization/Localizable.xcstrings:\n"
                      + missing.prefix(40).joined(separator: "\n"))
        XCTAssertTrue(untranslated.isEmpty,
                      "\(untranslated.count) string(s) have no Simplified Chinese translation:\n"
                      + untranslated.prefix(40).joined(separator: "\n"))
    }

    /// A key nobody draws any more is copy a translator is still paying for,
    /// and it hides real coverage gaps behind a green test.
    func testTheCatalogHasNoEntriesTheAppNeverDraws() throws {
        let catalog = try loadCatalog()
        var drawn = Set<String>()
        for file in try swiftFiles(under: sourcesRoot) {
            let source = try String(contentsOf: file, encoding: .utf8)
            for key in keys(in: source) {
                for resolved in catalogKeys(matching: key, in: catalog) { drawn.insert(resolved) }
            }
        }
        // Copy held as data and localized through `String(localized:)` is
        // reached by a runtime call the scanner cannot see, so it is declared.
        let runtimeKeys = Set(try runtimeLocalizedKeys().map { unescape($0) })
        let orphans = catalog.strings.keys.filter { !drawn.contains($0) && !runtimeKeys.contains($0) }.sorted()
        XCTAssertTrue(orphans.isEmpty,
                      "\(orphans.count) catalog entrie(s) are no longer drawn by any call site:\n"
                      + orphans.prefix(40).joined(separator: "\n"))
    }

    /// Keys reached through `String(localized: "…")` rather than a SwiftUI
    /// literal — the copy this app deliberately keeps as testable data in
    /// `Services/` and its model types.
    private func runtimeLocalizedKeys() throws -> [String] {
        var found: [String] = []
        for file in try swiftFiles(under: sourcesRoot) {
            let text = stripLineComments(try String(contentsOf: file, encoding: .utf8))
            var search = text.startIndex
            while let hit = text.range(of: "String(localized:", range: search ..< text.endIndex) {
                search = hit.upperBound
                var i = hit.upperBound
                while i < text.endIndex, text[i] == " " || text[i] == "\n" { i = text.index(after: i) }
                guard i < text.endIndex, text[i] == "\"" else { continue }
                guard let key = literal(in: text, openingAt: i) else { continue }
                found.append(key)
            }
        }
        return found
    }

    func testRuntimeLocalizedCopyIsAlsoTranslated() throws {
        let catalog = try loadCatalog()
        var untranslated: [String] = []
        for key in try runtimeLocalizedKeys().map({ unescape($0) }) where carriesProse(key) {
            guard let zh = catalog.strings[key]?["zh-Hans"], zh.state == "translated", !zh.value.isEmpty else {
                untranslated.append(key)
                continue
            }
        }
        XCTAssertTrue(untranslated.isEmpty,
                      "\(untranslated.count) `String(localized:)` key(s) have no Simplified Chinese translation:\n"
                      + untranslated.prefix(40).joined(separator: "\n"))
    }

    /// A catalog value is the FINISHED text — the escapes were already resolved
    /// when the key was authored. A translation copied from the Swift source
    /// keeps that source's spelling, so `\n` arrives as a backslash and an `n`
    /// and renders that way on screen, and `\"` puts a stray backslash in front
    /// of every quote. Both shipped in the first draft of this catalog, and
    /// neither is visible in a diff of a language you don't read.
    func testTranslationsCarryNoSourceSpellingEscapes() throws {
        let catalog = try loadCatalog()
        let artifact = try NSRegularExpression(pattern: "\\\\[ntr\"]|\\\\u\\{")
        for (key, localizations) in catalog.strings {
            for (language, unit) in localizations {
                let range = NSRange(unit.value.startIndex ..< unit.value.endIndex, in: unit.value)
                XCTAssertNil(artifact.firstMatch(in: unit.value, range: range),
                             "\(language) translation of \"\(key)\" contains a literal escape "
                             + "sequence — write the character itself: \(unit.value)")
            }
        }
    }

    /// A format specifier is a promise about the ARGUMENT at that position, and
    /// the arguments are supplied by the call site, which the translation
    /// cannot change. `%lld` read as `%@` is a pointer dereference of an
    /// integer — a crash, not a typo.
    ///
    /// Chinese moves word order and has no plural suffix, so a translation must
    /// be able to reorder arguments and drop the ones it doesn't need. Both are
    /// only safe POSITIONALLY (`%1$lld`), which is why every specifier in a
    /// translation must carry an index: a bare `%@` in a two-argument string is
    /// silently order-dependent, and the day someone reorders the sentence it
    /// starts reading the wrong argument.
    func testEveryTranslatedFormatSpecifierMatchesItsKey() throws {
        let catalog = try loadCatalog()
        let specifier = try NSRegularExpression(pattern: "%(\\d+\\$)?(lld|llu|lf|[dfu@])")

        func types(in s: String) -> [(index: Int?, type: String)] {
            let range = NSRange(s.startIndex ..< s.endIndex, in: s)
            // `%%` is a literal percent, never an argument.
            let withoutLiterals = s.replacingOccurrences(of: "%%", with: "\u{1}")
            let r2 = NSRange(withoutLiterals.startIndex ..< withoutLiterals.endIndex, in: withoutLiterals)
            _ = range
            return specifier.matches(in: withoutLiterals, range: r2).map { m in
                let idx: Int?
                if m.range(at: 1).location != NSNotFound,
                   let r = Range(m.range(at: 1), in: withoutLiterals) {
                    idx = Int(withoutLiterals[r].dropLast())
                } else {
                    idx = nil
                }
                let type = Range(m.range(at: 2), in: withoutLiterals).map { String(withoutLiterals[$0]) } ?? ""
                return (index: idx, type: type)
            }
        }

        for (key, localizations) in catalog.strings {
            let keyTypes = types(in: key).map(\.type)
            guard let zh = localizations["zh-Hans"] else { continue }
            let valueSpecs = types(in: zh.value)

            if keyTypes.isEmpty {
                XCTAssertTrue(valueSpecs.isEmpty,
                              "\"\(key)\" takes no arguments, but its translation asks for \(valueSpecs.count).")
                continue
            }
            for spec in valueSpecs {
                guard let index = spec.index else {
                    XCTFail("\"\(key)\": the translation uses a non-positional \"%\(spec.type)\". "
                            + "Write \"%1$\(spec.type)\" — Chinese reorders and drops arguments.")
                    continue
                }
                guard index >= 1, index <= keyTypes.count else {
                    XCTFail("\"\(key)\": the translation reads argument \(index), but the string has \(keyTypes.count).")
                    continue
                }
                XCTAssertEqual(spec.type, keyTypes[index - 1],
                               "\"\(key)\": argument \(index) is %\(keyTypes[index - 1]), "
                               + "but the translation reads it as %\(spec.type).")
            }
        }
    }

    /// System prompts, tool names and tool descriptions are read by the MODEL,
    /// not by the user. Translating them would break the three ported tests
    /// that assert exact prompt equality with the iPhone app, and a localized
    /// tool description measurably degrades tool calling on small models — the
    /// grammar a 2B model matches against is the English one it was trained on.
    ///
    /// The user-visible half of that decision is the reply language, which is
    /// a line resolved INTO the prompt, never a translation of it.
    func testModelFacingCopyIsNeverLocalized() throws {
        let modelFacing = [
            "Services/AgentPrompt.swift",
            "Services/AgentWriter.swift",
            "Models/Agent.swift",
        ]
        for path in modelFacing {
            let url = sourcesRoot.appendingPathComponent(path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                XCTFail("\(path) is named as model-facing but no longer exists — update this list.")
                continue
            }
            let source = try String(contentsOf: url, encoding: .utf8)
            XCTAssertFalse(source.contains("String(localized:"),
                           "\(path) is model-facing: its text is read by the model, not the user.")
            XCTAssertFalse(source.contains("NSLocalizedString("),
                           "\(path) is model-facing: its text is read by the model, not the user.")
        }
    }
}
