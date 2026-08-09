import XCTest
@testable import MLXCore

/// A view presented in a SHEET must be handed every `@EnvironmentObject` it
/// reads, AT THE SHEET.
///
/// Live crash 2026-08-09: the welcome screen moved from its own window into a
/// `.sheet` on the chat window, attached after that scene's whole
/// `.environmentObject` chain, on the assumption that a sheet inherits the
/// environment of the view it hangs on. It does not — SwiftUI presents sheet
/// content in its own hosting context — so the app trapped in
/// `WelcomeView.server.getter` on every launch. Every other sheet in the app
/// already injected explicitly; the convention was there and the new one
/// ignored it.
///
/// `testTheChatWindowInjectsEveryObjectItsHostedPanesRead` cannot see this
/// class by construction: it asks what the WINDOW injects, and the window did
/// inject `ServerManager`. The bug is WHERE the reading happens. Same blind
/// spot, same shape, as the `TasksView().taskList` trap.
final class SheetEnvironmentAuditTests: XCTestCase {

    /// How each environment type is spelled at an injection site. A type with
    /// no entry fails loudly rather than being skipped — an audit that quietly
    /// ignores what it doesn't recognise is an audit that passes forever.
    private let stems: [String: String] = [
        "AppState": "appstate",
        "ServerManager": "server",
        "DownloadManager": "downloads",
        "HFSearchService": "hfsearch",
        "MCPManager": "mcpmanager",
        "TaskScheduler": "taskscheduler",
        "AgentStore": "agents",
        "ChatTurnEngine": "chatengine",
        "AgentMemory": "agentmemory",
        "ToolExecutor": "toolexecutor",
        "VoiceModeController": "voice",
        "ProcessRegistry": "processregistry",
        "ImageGenService": "imagegen",
        "VideoGenService": "videogen",
        "AudioGenService": "audiogen",
        "MusicGenService": "musicgen",
        "Model3DGenService": "model3dgen",
    ]

    private var root: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func source(_ relativePath: String) throws -> String {
        try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// The body of the trailing closure that starts at or after `from`.
    private func closureBody(in text: String, from: String.Index) -> String? {
        guard let open = text[from...].firstIndex(of: "{") else { return nil }
        var depth = 0
        var i = open
        while i < text.endIndex {
            if text[i] == "{" { depth += 1 }
            if text[i] == "}" {
                depth -= 1
                if depth == 0 { return String(text[text.index(after: open)..<i]) }
            }
            i = text.index(after: i)
        }
        return nil
    }

    /// Every `@EnvironmentObject` a sheet rooted at `type` can end up reading.
    ///
    /// The whole FILE, plus one hop into the files of the view types it
    /// constructs: the environment is inherited DOWNWARD from the sheet's root,
    /// so a child three levels in trapping on a missing object is the same
    /// crash — `WelcomeView` hosts `RecommendedStarterCard` from another file,
    /// which reads two of its own. Over-demanding is the safe direction here:
    /// the cost of an extra `.environmentObject` is nothing, and the cost of a
    /// missing one is a launch crash.
    private func environmentTypes(reachableFrom type: String) throws -> Set<String> {
        let envPattern = try NSRegularExpression(pattern: #"@EnvironmentObject\s+(?:private\s+)?var\s+\w+\s*:\s*(\w+)"#)
        let usePattern = try NSRegularExpression(pattern: #"\b([A-Z]\w*(?:View|Card|Sheet|Row|Panel))\s*\("#)

        var seen = Set<String>()
        var queue = [type]
        var found = Set<String>()

        while let next = queue.popLast() {
            guard !seen.contains(next), seen.count < 24 else { continue }
            seen.insert(next)
            guard let body = try structBody(of: next) else { continue }

            let range = NSRange(body.startIndex..., in: body)
            for m in envPattern.matches(in: body, range: range) {
                if let r = Range(m.range(at: 1), in: body) { found.insert(String(body[r])) }
            }
            for m in usePattern.matches(in: body, range: range) {
                if let r = Range(m.range(at: 1), in: body) { queue.append(String(body[r])) }
            }
        }
        return found
    }

    /// `view`'s OWN struct body, not the file holding it.
    ///
    /// File granularity is too coarse: `ToolApprovalSheet` is declared in
    /// ChatView.swift, so reading the file made a sheet that owns nothing
    /// appear to read every object ChatView does — an audit demanding
    /// injections nobody needs gets edited until it stops complaining, which
    /// is how a guard dies.
    private func structBody(of view: String) throws -> String? {
        let dir = root.appendingPathComponent("Sources/MLXServe/Views")
        let files = try FileManager.default.contentsOfDirectory(at: dir,
                                                                includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "swift" {
            let text = try String(contentsOf: file, encoding: .utf8)
            guard let decl = text.range(of: "struct \(view): View") else { continue }
            let after = text[decl.upperBound...]
            // The next TOP-LEVEL declaration: nested types are indented, so a
            // newline-anchored match can't cut the body short at one.
            let ends = ["\nstruct ", "\nprivate struct ", "\nfinal class ", "\nextension ", "\nenum "]
                .compactMap { after.range(of: $0)?.lowerBound }
            return String(after[..<(ends.min() ?? after.endIndex)])
        }
        return nil
    }

    func testEverySheetHandsItsContentTheObjectsThatContentReads() throws {
        let files = ["Sources/MLXServe/MLXServeApp.swift",
                     "Sources/MLXServe/Views/ChatView.swift"]
        let viewPattern = try NSRegularExpression(pattern: #"\b([A-Z]\w*(?:View|Sheet))\s*\("#)
        var checked = 0

        for path in files {
            let text = try source(path)
            var search = text.startIndex
            while let hit = text.range(of: ".sheet(", range: search..<text.endIndex) {
                search = hit.upperBound
                guard let body = closureBody(in: text, from: hit.upperBound) else { continue }

                let range = NSRange(body.startIndex..., in: body)
                var presented = Set<String>()
                for m in viewPattern.matches(in: body, range: range) {
                    if let r = Range(m.range(at: 1), in: body) { presented.insert(String(body[r])) }
                }

                for type in presented.sorted() {
                    let needs = try environmentTypes(reachableFrom: type)
                    if needs.isEmpty { continue }
                    checked += 1
                    let injected = body.lowercased()
                    for envType in needs.sorted() {
                        guard let stem = stems[envType] else {
                            XCTFail("""
                                \(type) is presented in a sheet in \(path) and reads \
                                @EnvironmentObject \(envType), which this audit doesn't know \
                                how to spell. Add it to `stems`.
                                """)
                            continue
                        }
                        XCTAssertTrue(injected.contains(".environmentobject(") && injected.contains(stem), """
                            \(path): the sheet presenting \(type) must inject \(envType) — \
                            a sheet does NOT inherit the environment of the view it hangs on, \
                            and SwiftUI traps at first render when it is missing.
                            """)
                    }
                }
            }
        }

        XCTAssertGreaterThan(checked, 0,
                             "the audit stopped finding sheets — fix the audit, not the app")
    }
}
