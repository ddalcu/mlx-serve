import AppKit
import SwiftUI
import XCTest
@testable import MLXCore

/// One ladder for the app's text: the system's own styles, snapped to whole
/// even points, never under 10, reached only through `Font.app(_:)`.
///
/// Why a ladder and not the system styles themselves: macOS has no dynamic
/// type. `NSFont.preferredFont(forTextStyle:)` hands every user the same
/// numbers, and `.dynamicTypeSize(_:)` — an iOS environment — does not move a
/// semantic font here at all: a `Text` at `.body` renders the same height from
/// `.xSmall` through `.accessibility4`. So a semantic style buys a NAME, not a
/// size that follows anyone, and the numbers the app renders are the ones in
/// `AppType`.
///
/// The chat transcript is the one deliberate exception and keeps its own
/// four-step picker (`ChatTextSize`): it is the only text a user chooses a size
/// for, and snapping its ladder would collapse the gap between prose and code
/// that its own comment documents.
final class SystemTypeTests: XCTestCase {

    // MARK: - The ladder

    /// The two rules the ladder exists to keep: whole even points, never under
    /// the floor. Whatever a view renders has to pass both.
    func testEveryStepIsEvenAndNeverBelowTheFloor() {
        XCTAssertEqual(AppType.table.count, 11, "a step was added without saying so here")
        for step in AppType.table {
            XCTAssertTrue(
                AppType.isLegal(step.pointSize),
                "\(step.style) is \(step.pointSize): sizes are whole even points, at least \(AppType.floor)"
            )
        }
        // The rule has teeth: an odd size and one under the floor both fail.
        XCTAssertFalse(AppType.isLegal(11), "odd")
        XCTAssertFalse(AppType.isLegal(8), "under the floor")
        XCTAssertTrue(AppType.isLegal(AppType.floor), "the floor itself is legal")
    }

    /// The ladder is anchored to the platform, not to taste: each step is the
    /// system's size for the style it names, rounded up to an even point and
    /// floored. When macOS moves one this fails — which is the signal to
    /// re-derive `AppType.table`, not to let the app drift on its own.
    func testEveryStepIsTheSystemStyleItNames() {
        for step in AppType.table {
            let live = NSFont.preferredFont(forTextStyle: appKitStyle(for: step.style)).pointSize
            XCTAssertEqual(live, step.system, accuracy: 0.01, """
                macOS now renders this style at \(live)pt; AppType.table says \(step.system). \
                Re-derive the ladder: odd steps up one, floor \(AppType.floor).
                """)

            var expected = live
            if expected.truncatingRemainder(dividingBy: 2) != 0 { expected += 1 }
            expected = max(expected, AppType.floor)
            XCTAssertEqual(step.pointSize, expected, accuracy: 0.01,
                           "\(step.style): \(step.pointSize) is not the system's \(live)pt snapped")
        }
    }

    /// Every step the app can ask for is on the table, so `pointSize(for:)`
    /// never falls back to the floor for a name the app actually uses.
    func testEveryStepTheAppAsksForIsOnTheTable() throws {
        let used = try usedStyles()
        XCTAssertFalse(used.isEmpty, "the scan found no steps — it is not walking the tree")
        let known = Set(AppType.table.map { "\($0.style)" })
        for name in used.sorted() {
            XCTAssertTrue(known.contains(name), "\(name) is not in AppType.table")
        }
    }

    // MARK: - The call sites

    /// A stated point size is the one thing that puts a number on screen the
    /// ladder does not know about. The exceptions are geometry and symbols, and
    /// they are listed rather than inferred.
    func testNoPointSizeIsStatedForText() throws {
        // Geometry in the transcript renderer: a 1pt glyph gives a table rule
        // its height, a 6pt one the blank line between blocks. Neither is text,
        // so neither is on the ladder.
        let geometry: [(file: String, needle: String)] = [
            ("Views/ChatView.swift", "NSFont.systemFont(ofSize: 1)"),
            ("Views/ChatView.swift", "NSFont.systemFont(ofSize: 6)"),
        ]
        let pattern = try NSRegularExpression(pattern: #"(?:\.system\(size:|ofSize:)\s*[0-9]"#)
        var offenders: [String] = []
        for (file, code) in try swiftSources() {
            let lines = SourceScan.strippingComments(code).components(separatedBy: "\n")
            for (index, line) in lines.enumerated() {
                let range = NSRange(line.startIndex..., in: line)
                guard pattern.firstMatch(in: line, range: range) != nil else { continue }
                if geometry.contains(where: { $0.file == file && line.contains($0.needle) }) { continue }
                // A size on an SF Symbol is a mark, not text: a glyph scales
                // with the control around it and has no semantic step to name.
                // The `Image(` is on one of the three lines above the `.font`.
                let window = lines[max(0, index - 3)...index].joined(separator: "\n")
                if window.contains("Image(") { continue }
                offenders.append("\(file):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            A point size is stated for text:
            \(offenders.joined(separator: "\n"))

            Text takes a step from the ladder — `.app(.body)`,
            `.app(.caption2, weight: .medium)` — or, on the AppKit side, a
            number from `AppType`. What stays a literal is geometry (the 1pt
            and 6pt spacers) and symbols on an `Image`.
            """)
    }

    /// Every text size goes through `Font.app`, so the ladder can be
    /// re-derived in one place. A bare `.font(.body)` is the hole this closes:
    /// it reads as a semantic style and is the one that silently drifts.
    func testEveryTextFontCallGoesThroughTheLadder() throws {
        let pattern = try NSRegularExpression(
            pattern: #"\.font\(\.(largeTitle|title2|title3|title|headline|body|callout|subheadline|footnote|caption1|caption2|caption)\b"#)
        var offenders: [String] = []
        for (file, code) in try swiftSources() {
            let lines = SourceScan.strippingComments(code).components(separatedBy: "\n")
            for (index, line) in lines.enumerated() {
                let range = NSRange(line.startIndex..., in: line)
                guard pattern.firstMatch(in: line, range: range) != nil else { continue }
                offenders.append("\(file):\(index + 1): \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            A view states its text size as a bare system style:
            \(offenders.joined(separator: "\n"))

            `.font(.body)` renders whatever macOS says that is today. Name the
            step through the ladder instead: `.font(.app(.body))`.
            """)
    }

    // MARK: - The transcript

    /// Text that states NO size at all renders at the platform default — 13pt on
    /// macOS, an odd number the ladder does not contain — so it is off-scale by
    /// construction. `.font()` on an ANCESTOR fixes that, and SwiftUI does
    /// propagate it (measured: a `Text` inside `VStack { … }.font(…)` renders
    /// at the ancestor's size), which is why the check walks outward.
    ///
    /// Two places it deliberately does not look, because a lexical scan cannot
    /// see them and a false positive on correct code is worse than a miss:
    ///
    /// * **A project helper.** `footerBar` sets `.app(.caption)` on its own
    ///   HStack 160 lines away from every call site; a scan sees the call and
    ///   not the container. So text lexically inside a call to a capitalized
    ///   name that is not a known SwiftUI/AppKit view is left alone — the
    ///   helper may be the thing that sets the font.
    /// * **AppKit draws it.** A menu item, an `Alert`'s title or its buttons
    ///   are `NSMenuItem`/`NSAlert` text: `.font()` does not reach them on this
    ///   platform, so demanding one would be demanding a lie. They are listed
    ///   below with the surface that owns them.
    func testEveryStatedTextNamesASize() throws {
        let appKitDrawn: [(file: String, needle: String, surface: String)] = [
            ("MLXServeApp.swift", "Text(\"\\(width.label) chat column\")", "View ▸ Interface menu (NSMenu)"),
            ("MLXServeApp.swift", "Label(\"Interface\", systemImage:", "View ▸ Interface menu section header"),
            ("Views/AgentsWindow.swift", "Alert(title: Text(\"Agents\")", "Alert title (NSAlert)"),
            ("Views/AgentsWindow.swift", "dismissButton: .default(Text(\"OK\"))", "Alert button (NSAlert)"),
            ("Views/AgentsWindow.swift", "message: Text(\"This can't be undone.\")", "Alert message (NSAlert)"),
            ("Views/AgentsWindow.swift", "primaryButton: .destructive(Text(\"Delete\"))", "Alert button (NSAlert)"),
            ("Views/AgentsWindow.swift", "Text(\"No clips yet\")", "voice-picker menu (NSMenu)"),
            ("Views/AgentsWindow.swift", "Text(\"No voices installed\")", "voice-picker menu (NSMenu)"),
            ("Services/CLILauncher.swift", "Label(\"\\(spec.displayName) in Sandbox\"", "new-session menu (NSMenu)"),
            ("Services/CLILauncher.swift", "Label(\"Shell in Sandbox\"", "new-session menu (NSMenu)"),
        ]
        let knownContainers: Set<String> = [
            "VStack", "HStack", "ZStack", "Group", "Section", "List", "Form", "LazyVStack",
            "LazyVGrid", "Grid", "Table", "TableColumn", "ScrollView", "ScrollViewReader",
            "Menu", "Button", "Toggle", "Label", "Text", "TextField", "SecureField", "TextEditor",
            "Picker", "Link", "NavigationSplitView", "NavigationStack", "NavigationLink",
            "TabView", "Divider", "Spacer", "Color", "Image", "ProgressView", "Gauge", "Canvas",
            "ControlGroup", "DisclosureGroup", "SettingsRow", "EmptyView", "AnyView", "TupleView",
            "_ConditionalContent", "ConditionalContent", "Optional", "ForEach", "Alert",
            "confirmationDialog", "ContentUnavailableView", "ViewThatFits", "LabeledContent",
            "MenuBarExtra", "Form", "List", "Table", "GridRow", "FlowLayout", "AttachmentFlowLayout",
        ]
        let offenders = try bareStatedText(knownContainers: knownContainers)
            .filter { line in !appKitDrawn.contains { line.hasPrefix($0.file + ":") && line.contains($0.needle) } }
        XCTAssertTrue(offenders.isEmpty, """
            Text with no size of its own, and no ancestor that sets one:
            \(offenders.joined(separator: "\n"))

            It renders at the platform default (13pt on macOS), which is not on
            the ladder. Either give it a step — `.app(.callout)`, or the step its
            neighbours use — or set one on the container it lives in. Text inside
            a project helper or on an AppKit surface is not reported: the helper
            may be the thing that sets the size, and a menu item or an Alert is
            drawn by AppKit.
            """)
    }

    // MARK: - The transcript

    /// The chat-only size picker is untouched by the ladder and still moves the
    /// transcript: the setting is the user's, and one who set Extra Large must
    /// still get Extra Large after this change.
    func testTheTranscriptStillFollowsItsOwnSizeSetting() {
        let original = UserDefaults.standard.object(forKey: InterfacePrefKey.textSize)
        defer {
            if let original { UserDefaults.standard.set(original, forKey: InterfacePrefKey.textSize) }
            else { UserDefaults.standard.removeObject(forKey: InterfacePrefKey.textSize) }
        }
        UserDefaults.standard.set(ChatTextSize.small.rawValue, forKey: InterfacePrefKey.textSize)
        let small = ChatMetrics.transcriptFontSize
        UserDefaults.standard.set(ChatTextSize.xlarge.rawValue, forKey: InterfacePrefKey.textSize)
        let large = ChatMetrics.transcriptFontSize
        XCTAssertEqual(small, ChatTextSize.small.proseSize)
        XCTAssertEqual(large, ChatTextSize.xlarge.proseSize)
        XCTAssertLessThan(small, large, "the setting still has to move the transcript")
    }

    // MARK: - Helpers

    /// Every `Text`/`Label` that names no size and has no ancestor block that
    /// does — reported as `file:line: text`, with the project-helper case
    /// skipped (see `testEveryStatedTextNamesASize` for why it must be).
    private func bareStatedText(knownContainers: Set<String>) throws -> [String] {
        // A statement whose text is visible on screen. `.tag(…)`, a TextField's
        // `prompt:` and `.help()` are drawn by the control or never drawn.
        let skip = try NSRegularExpression(
            pattern: "\\.tag\\(|prompt:\\s*Text\\(|accessibility(Text|Label|Value|Hint)|\\.help\\(\"")
        let stated = try NSRegularExpression(pattern: "\\b(?:Text|Label)\\(\\s*\"")
        var found: [String] = []
        for (file, code) in try swiftSources() {
            let lines = SourceScan.strippingComments(code).components(separatedBy: "\n")
            let depths = Self.braceDepths(lines)
            for i in 0..<lines.count {
                let line = lines[i]
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("//") { continue }
                let range = NSRange(line.startIndex..., in: line)
                guard stated.firstMatch(in: line, range: range) != nil else { continue }
                if skip.firstMatch(in: line, range: range) != nil { continue }
                // Its own chain: the next two lines, which is where a trailing
                // `.font(…)` / `.app(…)` on the same expression lands.
                if (i..<min(i + 3, lines.count)).contains(where: {
                    lines[$0].contains(".font(") || lines[$0].contains(".app(")
                }) { continue }
                if Self.ancestorSetsSize(lines: lines, depths: depths, at: i) { continue }
                if Self.insideProjectHelper(lines: lines, depths: depths, at: i, known: knownContainers) { continue }
                found.append("\(file):\(i + 1): \(trimmed)")
            }
        }
        return found
    }

    /// Depth of each line, counted before the line's own braces.
    private static func braceDepths(_ lines: [String]) -> [Int] {
        var depth = 0
        return lines.map { line in
            let here = depth
            depth += line.reduce(0) { $1 == "{" ? $0 + 1 : ($1 == "}" ? $0 - 1 : $0) }
            return here
        }
    }

    /// Is the line lexically inside a call to a capitalized name that is NOT a
    /// known SwiftUI/AppKit view? That call is a project helper, and a helper is
    /// allowed to set the font on itself — `footerBar` is, in fact, exactly
    /// that. Returns true (skip) for those, false for a plain container chain.
    /// Is the line lexically inside a call to a capitalized name that is NOT a
    /// known SwiftUI/AppKit view? That call is a project helper, and a helper is
    /// allowed to set the font on itself — `footerBar` is, in fact, exactly
    /// that. Returns true (skip) for those, false for a plain container chain.
    private static func insideProjectHelper(lines: [String], depths: [Int],
                                            at index: Int, known: Set<String>) -> Bool {
        let call = try? NSRegularExpression(pattern: #"\b([A-Z][A-Za-z0-9_]*)\s*[({]"#)
        var level = depths[index]
        var j = index - 1
        while j >= 0 {
            while j >= 0 && depths[j] >= level { j -= 1 }
            guard j >= 0, let call else { break }
            let openers = (0...j).reversed().filter { depths[$0] == level - 1 }
            guard let open = openers.first(where: { lines[$0].contains("{") }) else { level -= 1; j -= 1; continue }
            // A declaration is not a call. `struct PaneTitle: View {` matches the
            // call regex on `View`, and reading that as a project helper would
            // swallow EVERY bare Text in the file — the scan would pass vacuously,
            // which is the one failure mode it cannot afford.
            if Self.isDeclaration(lines[open]) { return false }
            let range = NSRange(lines[open].startIndex..., in: lines[open])
            if let m = call.firstMatch(in: lines[open], range: range),
               let nameRange = Range(m.range(at: 1), in: lines[open]) {
                let name = String(lines[open][nameRange])
                if !known.contains(name) { return true }
            }
            level -= 1
            j -= 1
            if level <= 0 { break }
        }
        return false
    }

    /// Does an enclosing block set a size? The block's own lines count, AND the
    /// two lines after its closing brace — SwiftUI's own idiom is a trailing
    /// modifier on the container:
    ///
    ///     Button { } label: { Text("Check Now") }
    ///         .font(.app(.callout))
    ///
    /// so a scan that only reads a block's interior reports every one of them.
    private static func ancestorSetsSize(lines: [String], depths: [Int], at index: Int) -> Bool {
        var level = depths[index]
        while level > 0 {
            guard let open = (0..<index).reversed().first(where: { depths[$0] == level - 1 && lines[$0].contains("{") })
            else { return false }
            let close = (index..<lines.count).first(where: { depths[$0] < level }) ?? lines.count - 1
            let window = lines[open...min(close + 2, lines.count - 1)].joined(separator: "\n")
            if window.contains(".font(") || window.contains(".app(") { return true }
            level -= 1
        }
        return false
    }

    /// Does this line open a type, function or stored property rather than call
    /// anything?
    private static func isDeclaration(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("//") || trimmed.hasPrefix("*") { return true }
        return trimmed.range(of: #"^\s*(?:@\w+\s+)*(?:(?:public|private|fileprivate|internal|final|indirect|@\w+)\s+)*(struct|class|enum|extension|actor|protocol|func|init|subscript|var|let|case)\b"#,
                            options: .regularExpression) != nil
    }

    /// Every step named at a call site, spelled as `Font.TextStyle` prints.
    private func usedStyles() throws -> Set<String> {
        let pattern = try NSRegularExpression(pattern: #"\.app\(\.([a-zA-Z0-9]+)"#)
        var used: Set<String> = []
        for (_, code) in try swiftSources() {
            let code = SourceScan.strippingComments(code)
            for match in pattern.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                guard let range = Range(match.range(at: 1), in: code) else { continue }
                used.insert(String(code[range]))
            }
        }
        return used
    }

    /// Every Swift file under the app's source root, by relative path.
    private func swiftSources() throws -> [(String, String)] {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe")
        let walker = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil),
                                   "no source tree at \(root.path)")
        var files: [(String, String)] = []
        for case let url as URL in walker where url.pathExtension == "swift" {
            let path = url.path.replacingOccurrences(of: root.path + "/", with: "")
            files.append((path, try String(contentsOf: url, encoding: .utf8)))
        }
        // A scan that reads nothing passes every zero-count assertion, so the
        // walk itself is part of what is under test.
        XCTAssertGreaterThan(files.count, 100, "the walk read \(files.count) files — it is not walking the tree")
        return files
    }

    /// The AppKit style behind each SwiftUI one. macOS calls the first title
    /// `title1` and its only caption `caption1`; `title` and `caption` are the
    /// same two rows under the names SwiftUI uses.
    private static let appKit: [(Font.TextStyle, NSFont.TextStyle)] = [
        (.largeTitle, .largeTitle), (.title, .title1), (.title2, .title2), (.title3, .title3),
        (.headline, .headline), (.body, .body), (.callout, .callout), (.subheadline, .subheadline),
        (.footnote, .footnote), (.caption, .caption1), (.caption2, .caption2),
    ]

    private func appKitStyle(for style: Font.TextStyle) -> NSFont.TextStyle {
        Self.appKit.first { $0.0 == style }?.1 ?? .body
    }
}
