import XCTest
@testable import MLXCore

/// Every user-facing string the app draws must exist in the String Catalog,
/// translated into every language the catalog targets. Nothing here names a
/// language: the target list is DERIVED from the catalog (`targetLanguages`),
/// so adding one is a data change — translate, declare it in the two plists —
/// and this audit begins enforcing it in the same commit. Today that list is
/// Simplified Chinese.
///
/// The app has no localization *call sites*: SwiftUI's `Text("…")`,
/// `Button("…")`, `.help("…")` and friends take a `LocalizedStringKey` for a
/// string LITERAL, so they already perform a `Bundle.main` lookup at render
/// time. Adding `<lang>.lproj/Localizable.strings` to the bundle localizes
/// them with no code change, and a missing entry falls through to the key —
/// which IS the English text. That fallback is what makes this class of bug
/// silent: an untranslated string doesn't crash or blank out, it just renders
/// in English inside an otherwise translated screen, and only someone reading
/// that exact pane in that language would ever notice.
///
/// So the guard is coverage, not behaviour: extract the keys from the sources
/// the way SwiftUI does, and demand the catalog answer every one of them. The
/// extractor lives HERE rather than in a script, because a scan that only runs
/// when someone remembers to run it is not a guard.
///
/// Catalog DRIFT — a missing key, a language with no translation for one, or
/// an ORPHANED key nothing draws any more — is enforced STRICTLY only under
/// `MLX_SERVE_STRICT_LOCALIZATION=1`: the `/release` checklist's job, not
/// `swift test`'s. All three fall back gracefully rather than misbehaving
/// (missing/untranslated render the English source at runtime; an orphan is
/// inert JSON), so none is a correctness bug for a PR that only edits copy —
/// REWORDING a string orphans its old key and leaves the new one unanswered
/// in the SAME edit, so treating "missing" as blocking while "orphan" was
/// lenient (or vice versa) would still trip every copy change. A default run
/// REPORTS what's outstanding (`reportOrFail`/`reportOrFailCoverage`)
/// instead of failing, so a contributor changing a model blurb never needs
/// to know Chinese, or block on someone who does, to get green. Structural
/// checks on translations that DO exist — no leftover source escapes,
/// format specifiers matching their key — stay unconditional, because
/// neither requires writing a new translation to satisfy.
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
        "CommandMenu", "Alert", "ContentUnavailableView", "Window",
    ]

    /// Modifiers taking a `LocalizedStringKey` first argument.
    private static let localizedModifiers = [
        ".help", ".navigationTitle", ".navigationSubtitle", ".alert",
        ".confirmationDialog", ".accessibilityLabel",
    ]

    // MARK: - The call sites whose string literal is COPY HELD AS DATA

    /// The other half of the app's copy: a literal that is stored in a `String`
    /// and drawn later. `Text(item.title)` and `.help(row.help)` pick the
    /// NON-localizing initializer — a `String` is drawn verbatim, so the bundle
    /// is never asked — and the literal is nowhere near the view that draws it.
    /// Those go through `String(localized:)` at the literal, which is a real
    /// lookup, and `testRuntimeLocalizedCopyIsAlsoTranslated` then demands the
    /// translation. Both lists are the audit's whole vocabulary for this class:
    /// a helper or a field absent from them is invisible, so a new copy-bearing
    /// row type gets added here rather than worked around.

    /// App-local view builders whose FIRST positional argument is copy typed
    /// `String` (`destinationRow("New Chat", icon:…)` → `Text(title)`).
    private static let copyCarryingCalls = [
        "destinationRow", "destinationLabel", "sectionHeader",
        // The Agents editor's chrome and the panes that reuse it. Each takes
        // its copy as `_ title: String` and draws it with `Text(title)`, which
        // looks exactly like a `Text("…")` at the call site and localizes
        // nothing.
        "AgentSection", "AgentLabeledField", "AgentEditorRow",
        "SortableHeader", "SettingsSubheader",
    ]

    /// `file → argument labels` whose value is copy typed `String`. Scoped per
    /// file because the same word means different things elsewhere: a
    /// `description:` in `BrowserManager` names a JS evaluation for a log line,
    /// and `MCPCatalog`'s is a third party's own blurb.
    private static let copyCarryingLabels: [String: [String]] = [
        "Views/ChatEmptyState.swift": ["title", "help"],
        "Views/ChatView.swift": ["help"],
        "Views/StatusMenuView.swift": ["title", "subtitle", "help"],
        "Views/TrayChrome.swift": ["title", "subtitle", "help"],
        "Views/VoiceTrayPanel.swift": ["title", "subtitle", "help", "label"],
        "Views/QuickLauncherView.swift": ["title", "subtitle", "help"],
        "Views/AgentsWindow.swift": ["title", "subtitle", "help", "label", "caption"],
        "Views/AgentViews.swift": ["title", "subtitle", "help"],
        "Views/AgentEditorChrome.swift": ["title", "subtitle", "help"],
        "Views/SettingsView.swift": ["title", "subtitle", "help", "label", "explainer"],
        "Views/ModelBrowserView.swift": ["title", "subtitle", "help", "label"],
        "Views/TasksView.swift": ["title", "subtitle", "label", "caption"],
        "Views/ImageGenView.swift": ["title", "help", "label", "caption"],
        "Views/VideoGenView.swift": ["title", "help", "label", "caption"],
        "Views/AudioGenView.swift": ["title", "help", "label", "caption"],
        "Views/MusicGenView.swift": ["title", "help", "label", "caption"],
        "Views/Model3DGenView.swift": ["title", "help", "label", "caption"],
        "Views/ContextPill.swift": ["label"],
        "Views/CodeBlockView.swift": ["label"],
        "Views/SandboxTerminalView.swift": ["message"],
        // Pure data by design (`ComposerTipTests`), so every sentence the
        // composer's four glyphs have is in this one type.
        "Views/ComposerTip.swift": ["title", "body", "detail"],
        // The example TITLES are menu labels; the BODIES are the model's own
        // published phrasings and a reword is a quality regression, so `prompt`
        // is deliberately not registered here.
        "Views/H3PromptExamples.swift": ["title"],
        "Models/SettingsCategory.swift": ["title"],
        // Every server-launch knob's label and explainer, drawn by the
        // Settings rows and read by its search index.
        "Models/ServerOptions.swift": ["title", "explainer"],
        "Models/RecommendedModels.swift": ["tagline", "blurb"],
        "Models/WelcomeModelPicks.swift": ["label"],
        "Models/CommunityLinks.swift": ["title", "explainer", "actionLabel"],
        "Services/MediaGenProgress.swift": ["stage"],
        "Services/MusicGenService.swift": ["message"],
        // The sandbox's failures are shown to the user in an alert, not logged.
        "Services/AgentSandbox.swift": ["message"],
        // Our blurbs ABOUT third-party servers, drawn in the Marketplace list,
        // and the field labels for the credentials each one needs. NOT
        // `placeholder`: those are sample values (`ghp_...`, `<ORG>`,
        // `postgres://user:pass@localhost:5432/mydb`) — a token format is not
        // copy, and translating one would teach the user a credential that
        // does not exist.
        "Services/MCPCatalog.swift": ["description", "label"],
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

    /// Every language the catalog translates INTO — the source language is the
    /// keys themselves, so it is never a target.
    ///
    /// Derived rather than listed, because "which languages must be complete"
    /// and "which languages ship" have to be the same answer: `xcstringstool`
    /// compiles one `.lproj` per language it finds, both bundle paths call it
    /// without naming one, and `tests/test_release_workflow_gates.sh` demands
    /// both plists declare every language here. Adding a language is therefore
    /// a DATA change — translate the catalog, declare it in the two plists —
    /// and this audit starts enforcing it in the same commit. Half a language
    /// is the silent-English bug per screen, which is the thing this file
    /// exists to prevent, so there is deliberately no "draft" state: a language
    /// the catalog carries is a language every key must answer.
    private func targetLanguages(_ catalog: Catalog) -> [String] {
        var found = Set<String>()
        for localizations in catalog.strings.values { found.formUnion(localizations.keys) }
        found.remove(catalog.sourceLanguage)
        return found.sorted()
    }

    /// The target languages that have no usable translation for one key.
    private func untranslatedLanguages(forKey key: String, in catalog: Catalog) -> [String] {
        targetLanguages(catalog).filter { language in
            guard let unit = catalog.strings[key]?[language] else { return true }
            return unit.state != "translated" || unit.value.isEmpty
        }
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

    /// The data-copy scanner needs its own self-test for the same reason the
    /// literal one does: it reports by finding NOTHING, so a rewrite that
    /// stops matching turns the coverage assertion into a green no-op. It reads
    /// three spellings of stored copy — the argument, the assignment and the
    /// computed property — and must not read the localized form of any of them.
    func testTheDataCopyScannerReadsAllThreeSpellingsAndSkipsLocalizedOnes() {
        let sample = """
        struct S {
            let rows = [
                Item(title: "Passed as an argument", help: "And its tooltip"),
                Item(title: String(localized: "Already localized"), help: ""),
            ]
            init() {
                title = "Assigned into a stored property"
                other = "Not a registered label"
            }
            var title: String {
                switch self {
                case .a: "Returned from a computed property"
                case .b: String(localized: "Localized in the same block")
                }
            }
            // title: "commented out"
        }
        """
        let found = bareCopyLiterals(in: sample, labels: ["title", "help"])
        XCTAssertTrue(found.contains("Passed as an argument"))
        XCTAssertTrue(found.contains("And its tooltip"))
        XCTAssertTrue(found.contains("Assigned into a stored property"))
        XCTAssertTrue(found.contains("Returned from a computed property"))
        XCTAssertFalse(found.contains("Already localized"), "String(localized:) IS the fix.")
        XCTAssertFalse(found.contains("Localized in the same block"),
                       "Inside a block the call is the only thing marking copy as localized.")
        XCTAssertFalse(found.contains("Not a registered label"))
        XCTAssertFalse(found.contains("commented out"))
    }

    func testThePunctuationOnlyFilterKeepsSentencesAndDropsBareValues() {
        XCTAssertTrue(carriesProse("Download"))
        XCTAssertTrue(carriesProse("Plan (\\(plan.steps.count) steps)"))
        XCTAssertFalse(carriesProse("\\(index + 1)"))
        XCTAssertFalse(carriesProse("\\(a) / \\(b)"))
        XCTAssertFalse(carriesProse("·"))
    }

    /// The coverage guard. A new English string with no Chinese translation is
    /// REPORTED here, naming itself — not failed. Coverage is a release gate
    /// (`MLX_SERVE_STRICT_LOCALIZATION=1`, see `/release`), not a per-PR one:
    /// a contributor adding a button label or a model blurb should never need
    /// to know Chinese, or wait on someone who does, to get `swift test`
    /// green. An entry this misses still renders — untranslated and missing
    /// both fall back to the English source at runtime (see the class comment
    /// above) — this test's whole job is making that fallback VISIBLE instead
    /// of silent, at whichever cadence the strict flag is run.
    func testEveryUserFacingStringIsTranslatedIntoEveryTargetLanguage() throws {
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
                    untranslated += untranslatedLanguages(forKey: candidate, in: catalog)
                        .map { "[\($0)] \(file.lastPathComponent): \"\(candidate)\"" }
                }
            }
        }

        reportOrFailCoverage(
            missing: missing,
            missingLabel: "string(s) are drawn by the app but absent from Localization/Localizable.xcstrings",
            untranslated: untranslated,
            untranslatedLabel: "string/language pair(s) have no translation")
    }

    /// A key nobody draws any more is copy a translator is still paying for.
    /// Reported rather than failed by default — see `strictLocalizationCoverage`'s
    /// doc comment: REWORDING a string orphans its old key in the very same
    /// edit that leaves the new one unanswered, so this and the two coverage
    /// tests are one release-gated policy, not three independent ones.
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
        // reached by a runtime call the scanner cannot see, so it is declared —
        // and it interpolates the same way, so it resolves the same way too.
        for key in try runtimeLocalizedKeys() {
            for resolved in catalogKeys(matching: key, in: catalog) { drawn.insert(resolved) }
        }
        let orphans = catalog.strings.keys.filter { !drawn.contains($0) }.sorted()
        reportOrFail(orphans, label: "catalog entrie(s) are no longer drawn by any call site")
    }

    /// A literal at a registered copy position, with the line it sits on.
    /// Deliberately blind to `String(localized: "…")`: the scan looks for a
    /// quote where the localizing call would be, so a localized literal simply
    /// isn't a match.
    private func bareCopyLiterals(in source: String, labels: [String]) -> [String] {
        let text = stripLineComments(source)
        var found: [String] = []

        func collect(_ marker: String, requiringWordBoundary: Bool) {
            var search = text.startIndex
            while let hit = text.range(of: marker, range: search ..< text.endIndex) {
                search = hit.upperBound
                if requiringWordBoundary, hit.lowerBound > text.startIndex {
                    let before = text[text.index(before: hit.lowerBound)]
                    if before.isLetter || before.isNumber || before == "_" || before == "." { continue }
                }
                var i = hit.upperBound
                while i < text.endIndex, text[i] == " " || text[i] == "\n" { i = text.index(after: i) }
                guard i < text.endIndex, text[i] == "\"" else { continue }
                guard let key = literal(in: text, openingAt: i), carriesProse(key) else { continue }
                found.append(key)
            }
        }

        /// The literals inside `var <name>: String { … }` / `func <name>(…) -> String { … }`
        /// — the third spelling of stored copy, and the only one where the
        /// literal is nowhere near the field name.
        func collectComputed(_ name: String) {
            var search = text.startIndex
            while let hit = text.range(of: "var \(name): String {", range: search ..< text.endIndex) {
                search = hit.upperBound
                var depth = 1
                var i = hit.upperBound
                while i < text.endIndex, depth > 0 {
                    let c = text[i]
                    if c == "{" { depth += 1 }
                    if c == "}" { depth -= 1 }
                    if c == "\"" {
                        if let key = literal(in: text, openingAt: i) {
                            // Inside a block the literal carries no label, so
                            // the only thing separating localized copy from
                            // bare copy is the call it sits in.
                            let localizing = "String(localized: "
                            let start = text.index(i, offsetBy: -localizing.count, limitedBy: text.startIndex)
                            let wrapped = start.map { text[$0 ..< i] == localizing } ?? false
                            if carriesProse(key), !wrapped { found.append(key) }
                            i = text.index(i, offsetBy: key.count + 2)
                            continue
                        }
                    }
                    i = text.index(after: i)
                }
                search = i
            }
        }

        for label in labels {
            collect("\(label):", requiringWordBoundary: true)
            // `control.title = "Start Server"` — an assignment into stored copy.
            collect("\(label) =", requiringWordBoundary: true)
            collectComputed(label)
        }
        return found
    }

    /// The coverage guard for copy held as DATA.
    ///
    /// The audit above reads SwiftUI literals, which are `LocalizedStringKey`s
    /// and therefore already bundle lookups. This one reads the copy that is
    /// stored in a `String` first — a row catalogue's `title`, a tooltip handed
    /// to a helper — because `Text(someString)` picks the verbatim initializer
    /// and never asks the bundle. Nothing about that is visible at the draw
    /// site: it renders in English on a Chinese screen exactly like a missing
    /// catalog entry, with the additional property that adding the entry does
    /// not fix it. The fix is `String(localized:)` at the literal.
    func testCopyDrawnFromAStringPropertyIsLocalizedWhereItIsWritten() throws {
        var bare: [String] = []

        for file in try swiftFiles(under: sourcesRoot) {
            let source = try String(contentsOf: file, encoding: .utf8)
            let relative = file.path.replacingOccurrences(of: sourcesRoot.path + "/", with: "")

            for key in bareCopyLiterals(in: source, labels: Self.copyCarryingLabels[relative] ?? []) {
                bare.append("\(relative): \"\(key)\"")
            }
            // The positional helpers are app-local view builders, so they are
            // looked for everywhere rather than per file.
            let text = stripLineComments(source)
            for call in Self.copyCarryingCalls {
                var search = text.startIndex
                while let hit = text.range(of: call + "(", range: search ..< text.endIndex) {
                    search = hit.upperBound
                    if hit.lowerBound > text.startIndex {
                        let before = text[text.index(before: hit.lowerBound)]
                        if before.isLetter || before.isNumber || before == "_" { continue }
                    }
                    var i = hit.upperBound
                    while i < text.endIndex, text[i] == " " || text[i] == "\n" { i = text.index(after: i) }
                    guard i < text.endIndex, text[i] == "\"" else { continue }
                    guard let key = literal(in: text, openingAt: i), carriesProse(key) else { continue }
                    bare.append("\(relative): \(call)(\"\(key)\")")
                }
            }
        }

        XCTAssertTrue(bare.isEmpty,
                      "\(bare.count) literal(s) sit at a position that is drawn from a String, "
                      + "where SwiftUI performs NO lookup. Wrap each in String(localized:):\n"
                      + bare.prefix(40).joined(separator: "\n"))
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

    /// Same coverage guard as `testEveryUserFacingStringIsTranslatedIntoEveryTargetLanguage`,
    /// for the `String(localized:)` call sites instead of SwiftUI literals —
    /// see that test's doc comment for why this reports rather than fails by
    /// default.
    func testRuntimeLocalizedCopyIsAlsoTranslated() throws {
        let catalog = try loadCatalog()
        var missing: [String] = []
        var untranslated: [String] = []
        // Through `catalogKeys(matching:)`, not a direct subscript: a
        // `String(localized:)` interpolates exactly like a `Text` literal, so
        // `String(localized: "Delete “\(name)”?")` asks the bundle for
        // `Delete “%@”?`. Looking up the SOURCE spelling passes only while the
        // catalog carries a key no runtime lookup can ever ask for.
        for key in try runtimeLocalizedKeys() where carriesProse(key) {
            let resolved = catalogKeys(matching: key, in: catalog)
            guard !resolved.isEmpty else {
                missing.append(unescape(key))
                continue
            }
            for candidate in resolved {
                untranslated += untranslatedLanguages(forKey: candidate, in: catalog)
                    .map { "[\($0)] \(candidate)" }
            }
        }
        reportOrFailCoverage(
            missing: missing,
            missingLabel: "`String(localized:)` key(s) are absent from the catalog",
            untranslated: untranslated,
            untranslatedLabel: "`String(localized:)` key/language pair(s) have no translation")
    }

    /// Catalog DRIFT — a key that's missing, untranslated, or orphaned — is
    /// enforced STRICTLY only when this is set: the `/release` checklist's
    /// job, not every `swift test`. All three fall back gracefully rather
    /// than crashing or misbehaving (missing/untranslated render the English
    /// source at runtime; an orphan is inert JSON nobody reads), so none of
    /// them is a correctness bug for a PR that only edits copy — changing a
    /// blurb's WORDING orphans the old catalog entry and leaves the new one
    /// unanswered in the SAME edit, and neither half should cost the person
    /// editing English prose a trip through Chinese, or a blocked merge
    /// waiting on someone who can make one. It is translation debt that has
    /// to be paid before a release ships it silently untranslated (or, for
    /// an orphan, cleaned up so a translator stops paying for dead copy).
    /// Default runs still print exactly what's outstanding, so the debt
    /// stays visible without blocking anyone who can't pay it down
    /// themselves.
    private var strictLocalizationCoverage: Bool {
        ProcessInfo.processInfo.environment["MLX_SERVE_STRICT_LOCALIZATION"] == "1"
    }

    private func reportOrFail(_ entries: [String], label: String, file: StaticString = #filePath, line: UInt = #line) {
        guard strictLocalizationCoverage else {
            guard !entries.isEmpty else { return }
            print("⚠️ [i18n] \(entries.count) \(label) "
                + "— not blocking (set MLX_SERVE_STRICT_LOCALIZATION=1 to enforce, e.g. before a release):")
            for entry in entries.prefix(40) { print("  \(entry)") }
            return
        }
        XCTAssertTrue(entries.isEmpty, "\(entries.count) \(label):\n" + entries.prefix(40).joined(separator: "\n"),
                      file: file, line: line)
    }

    private func reportOrFailCoverage(
        missing: [String], missingLabel: String,
        untranslated: [String], untranslatedLabel: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        reportOrFail(missing, label: missingLabel, file: file, line: line)
        reportOrFail(untranslated, label: untranslatedLabel, file: file, line: line)
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
            for (language, unit) in localizations where language != catalog.sourceLanguage {
                let valueSpecs = types(in: unit.value)

                if keyTypes.isEmpty {
                    XCTAssertTrue(valueSpecs.isEmpty,
                                  "[\(language)] \"\(key)\" takes no arguments, "
                                  + "but its translation asks for \(valueSpecs.count).")
                    continue
                }
                for spec in valueSpecs {
                    guard let index = spec.index else {
                        XCTFail("[\(language)] \"\(key)\": the translation uses a non-positional "
                                + "\"%\(spec.type)\". Write \"%1$\(spec.type)\" — a translation must "
                                + "be free to reorder the arguments, and to drop the ones its "
                                + "grammar has no use for.")
                        continue
                    }
                    guard index >= 1, index <= keyTypes.count else {
                        XCTFail("[\(language)] \"\(key)\": the translation reads argument \(index), "
                                + "but the string has \(keyTypes.count).")
                        continue
                    }
                    XCTAssertEqual(spec.type, keyTypes[index - 1],
                                   "[\(language)] \"\(key)\": argument \(index) is "
                                   + "%\(keyTypes[index - 1]), but the translation reads it as "
                                   + "%\(spec.type).")
                }
            }
        }
    }

    /// The audit's own blind spot: every coverage loop above iterates the
    /// languages the catalog carries, so a catalog with NO target language
    /// passes all of them by iterating nothing. That is the state this file was
    /// written to end, and it is one careless delete away.
    func testTheCatalogCarriesAtLeastOneTargetLanguage() throws {
        let catalog = try loadCatalog()
        XCTAssertFalse(targetLanguages(catalog).isEmpty,
                       "The catalog translates into nothing, so every coverage assertion in this "
                       + "file is vacuous. Restore the translations, or delete this file too.")
    }

    /// System prompts, tool names and tool descriptions are read by the MODEL,
    /// not by the user. Translating them would break the three ported tests
    /// that assert exact prompt equality with the iPhone app, and a localized
    /// tool description measurably degrades tool calling on small models — the
    /// grammar a 2B model matches against is the English one it was trained on.
    ///
    /// The user-visible half of that decision is the reply language, which is
    /// a line resolved INTO the prompt, never a translation of it.
    /// Text that describes bytes ALREADY WRITTEN — a banner an older build
    /// appended into saved chat content — is not copy, and translating it
    /// breaks the only thing it does.
    ///
    /// `TruncationNotice.stripped(from:)` scrubs that banner out of sessions
    /// on disk, and everything on disk is English (it was written before any
    /// of this existed). Derive the scrubber from the sentence the app now
    /// DISPLAYS and a Chinese app searches saved content for a Chinese banner,
    /// finds nothing, and hands the English one back to the model as assistant
    /// prose. An xctest binary has no `.lproj`, so `String(localized:)`
    /// resolves to its key there and no behavioural test can tell the two
    /// apart — which is why this is a scan.
    func testTheFrozenLegacyBannerTextIsNeverLocalized() throws {
        let url = sourcesRoot.appendingPathComponent("Services/TruncationNotice.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        guard let start = source.range(of: "private static func legacyFootnote(") else {
            return XCTFail("TruncationNotice.legacyFootnote is gone — the scrubber's markers "
                           + "must still come from frozen English, not from the displayed sentence.")
        }
        // The function body, to its closing brace at the same nesting level.
        var depth = 0
        var i = start.upperBound
        var body = ""
        var seenBrace = false
        while i < source.endIndex {
            let c = source[i]
            if c == "{" { depth += 1; seenBrace = true }
            if c == "}" { depth -= 1; if seenBrace, depth == 0 { break } }
            body.append(c)
            i = source.index(after: i)
        }
        XCTAssertFalse(body.contains("String(localized:"),
                       "legacyFootnote describes bytes already saved to disk in English.")
        XCTAssertTrue(source.contains("legacyFootnote(cause: cause, maxTokens: maxTokens)"),
                      "the legacy banner text(…) must be built from legacyFootnote, never from "
                      + "the localized footnote(…) the app displays.")
    }

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
