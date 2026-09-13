import XCTest
@testable import MLXCore

/// A confirmation dialog must answer the Return key.
///
/// Every dialog in this app was reachable only by mouse: Escape cancelled
/// (AppKit gives that away free) but the affirmative button had no key, so
/// "delete these 12 chats" — a decision the user had already made twice by the
/// time the sheet appeared — needed a trip to the trackpad to say yes.
///
/// This is a CLASS guard, not a fix for one dialog. Adding a dialog is exactly
/// when nobody thinks about its keyboard, so the scan makes every current and
/// future one declare an answer: either a single `.defaultAction` button, or a
/// line in `exempt` saying why Return must stay dead. A dialog that does
/// neither fails with its own title in the message.
///
/// Single-button dialogs are excluded by construction — SwiftUI already makes
/// a lone button the default, and asking for an explicit shortcut there would
/// be a rule about nothing.
final class DialogDefaultActionTests: XCTestCase {

    /// Dialogs that deliberately have NO default action, and why.
    ///
    /// The RAM warnings are the whole list, and they are not an oversight: the
    /// dialog exists to INTERRUPT a generate the user already asked for, and
    /// its affirmative ("Generate Anyway") commits the machine to swapping.
    /// A caution whose dangerous branch sits under the Return key is a caution
    /// you dismiss by reflex, so here the keyboard's only answer stays Escape.
    private let exempt: [String: String] = [
        "Model exceeds your Mac's RAM":
            "the affirmative overcommits RAM — Escape is the only key answer by design",
    ]

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe")
    }

    func testEveryMultiButtonDialogDeclaresItsDefaultAction() throws {
        let files = try swiftFiles()
        XCTAssertGreaterThan(files.count, 20, "source scan found almost nothing — has the layout moved?")

        var checked = 0
        var offenders: [String] = []

        for file in files {
            let code = Self.strippingComments(try String(contentsOf: file, encoding: .utf8))
            for dialog in Self.dialogs(in: code) {
                // The legacy `Alert(title:primaryButton:secondaryButton:)` shape
                // takes no `.keyboardShortcut` — it becomes an NSAlert, whose
                // FIRST button is the default, and that is `primaryButton`. So
                // these answer Return by construction; what the scan checks is
                // that they actually name a primary (or are a lone OK), rather
                // than skipping them for having no `Button(` in them. An audit
                // that quietly ignores what it doesn't recognise passes forever.
                if dialog.actions.contains("Alert(") {
                    XCTAssertTrue(dialog.actions.contains("primaryButton:")
                                  || dialog.actions.contains("dismissButton:"),
                                  "\(file.lastPathComponent): legacy Alert with no primaryButton — "
                                  + "nothing for Return to hit")
                    checked += 1
                    continue
                }
                let buttons = dialog.actions.components(separatedBy: "Button(").count - 1
                guard buttons >= 2 else { continue }          // lone button is already the default
                if exempt.keys.contains(where: { dialog.header.contains($0) }) { continue }

                checked += 1
                let defaults = dialog.actions.components(separatedBy: ".keyboardShortcut(.defaultAction)").count - 1
                if defaults != 1 {
                    offenders.append("\(file.lastPathComponent): \(dialog.label) — "
                                     + "\(buttons) buttons, \(defaults) default actions (want exactly 1)")
                }
            }
        }

        XCTAssertGreaterThan(checked, 5, "the scan matched too few dialogs to be doing its job")
        XCTAssertTrue(offenders.isEmpty,
                      "these dialogs give Return nothing to do — add "
                      + ".keyboardShortcut(.defaultAction) to the affirmative button, or list them in `exempt`:\n"
                      + offenders.joined(separator: "\n"))
    }

    func testTheConversationDeleteDialogConfirmsOnReturn() throws {
        // The one the user actually asked for, named so a failure says so
        // rather than pointing at a scan.
        let url = sourcesRoot.appendingPathComponent("Views/ChatView.swift")
        let code = Self.strippingComments(try String(contentsOf: url, encoding: .utf8))
        let deleteDialog = Self.dialogs(in: code).first { $0.header.contains("SidebarDeleteConfirm.title") }

        let dialog = try XCTUnwrap(deleteDialog, "the sidebar's delete confirmation is gone")
        XCTAssertTrue(dialog.actions.contains(".keyboardShortcut(.defaultAction)"),
                      "Return must confirm a conversation delete")

        // And it must be the DELETE button that Return hits, not Cancel —
        // a default action on the wrong branch reads as working and does the
        // opposite of what the key promises.
        let delete = try XCTUnwrap(dialog.actions.range(of: "Button(\"Delete\""))
        let cancel = try XCTUnwrap(dialog.actions.range(of: "Button(\"Cancel\""))
        let shortcut = try XCTUnwrap(dialog.actions.range(of: ".keyboardShortcut(.defaultAction)"))
        XCTAssertTrue(shortcut.lowerBound > delete.lowerBound && shortcut.lowerBound < cancel.lowerBound,
                      "the default action belongs to Delete, not Cancel")
    }

    func testTheExemptionListOnlyNamesDialogsThatExist() throws {
        // An exemption for a dialog that has been deleted or retitled is a
        // hole in the scan that reads like a decision.
        var headers: [String] = []
        for file in try swiftFiles() {
            let code = Self.strippingComments(try String(contentsOf: file, encoding: .utf8))
            headers.append(contentsOf: Self.dialogs(in: code).map(\.header))
        }
        for key in exempt.keys {
            XCTAssertTrue(headers.contains { $0.contains(key) },
                          "`exempt` names \(key.debugDescription), which no dialog matches any more")
        }
    }

    // MARK: - Scanning

    private func swiftFiles() throws -> [URL] {
        let e = FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        return (e?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }.sorted { $0.path < $1.path }
    }

    struct Dialog {
        /// The modifier's arguments — the title lives here.
        let header: String
        /// The trailing closure holding the buttons.
        let actions: String
        var label: String { String(header.prefix(60)).replacingOccurrences(of: "\n", with: " ") }
    }

    /// Comments stripped — without this the scan trips over prose about
    /// `.alert(` (the old sandbox window had a comment saying exactly that).
    /// Shared with the other source scans (`SourceScan`).
    static func strippingComments(_ source: String) -> String {
        SourceScan.strippingComments(source)
    }

    /// Every `.alert(…) { … }` / `.confirmationDialog(…) { … }` in a file, with
    /// its arguments and its actions closure. Paren- and brace-matched rather
    /// than line-based: these span dozens of lines and nest freely.
    static func dialogs(in code: String) -> [Dialog] {
        let chars = Array(code)
        var found: [Dialog] = []
        for opener in [".alert(", ".confirmationDialog("] {
            var search = code.startIndex
            while let r = code.range(of: opener, range: search..<code.endIndex) {
                search = r.upperBound
                let openParen = code.distance(from: code.startIndex, to: r.upperBound) - 1
                guard let closeParen = match(chars, from: openParen, open: "(", close: ")"),
                      let braceStart = nextBrace(chars, after: closeParen),
                      let braceEnd = match(chars, from: braceStart, open: "{", close: "}")
                else { continue }
                found.append(Dialog(header: String(chars[(openParen + 1)..<closeParen]),
                                    actions: String(chars[(braceStart + 1)..<braceEnd])))
            }
        }
        return found
    }

    /// Index of the delimiter closing the one at `from`, skipping string literals.
    private static func match(_ chars: [Character], from: Int, open: Character, close: Character) -> Int? {
        var depth = 0
        var i = from
        var inString = false
        while i < chars.count {
            let c = chars[i]
            if inString {
                if c == "\\" { i += 2; continue }
                if c == "\"" { inString = false }
            } else if c == "\"" {
                inString = true
            } else if c == open {
                depth += 1
            } else if c == close {
                depth -= 1
                if depth == 0 { return i }
            }
            i += 1
        }
        return nil
    }

    /// The `{` that starts the trailing closure — only whitespace may separate
    /// it from the call, or we have run into the next statement entirely.
    private static func nextBrace(_ chars: [Character], after: Int) -> Int? {
        var i = after + 1
        while i < chars.count, chars[i].isWhitespace { i += 1 }
        return i < chars.count && chars[i] == "{" ? i : nil
    }
}
