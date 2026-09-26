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
