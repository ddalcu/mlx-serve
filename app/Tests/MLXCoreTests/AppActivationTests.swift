import XCTest
import AppKit
@testable import MLXCore

/// CLASS GUARD — "an LSUIElement app presented UI without becoming .regular first".
///
/// MLX Core launches as `.accessory` (LSUIElement). An accessory app that isn't
/// active has NO key window — so a window opened while `.accessory` comes up
/// unemphasized: no blinking caret, no first-responder focus, the title bar
/// looks half-lit. `ActivationPolicyManager` only flips to `.regular` when it
/// sees `didBecomeKeyNotification`… which cannot fire until the app is active.
/// Chicken-and-egg: the window sits semi-focused until the user clicks or types,
/// which finally activates the app, which finally fires the notification, which
/// finally flips the policy. Modal `NSOpenPanel`s (the code-editor folder
/// picker) fail the same way and can't be clicked until you "play around with it".
///
/// The rule these pin: **flip to `.regular` BEFORE the window/panel is
/// presented, then activate** — never the other way round.
@MainActor
final class AppActivationTests: XCTestCase {

    /// Records the ORDER of the calls — order is the whole bug.
    private final class FakeApp: AppActivating {
        var currentPolicy: NSApplication.ActivationPolicy
        var steps: [AppActivation.Step] = []

        init(policy: NSApplication.ActivationPolicy) { currentPolicy = policy }

        func setPolicy(_ policy: NSApplication.ActivationPolicy) {
            currentPolicy = policy
            steps.append(.setPolicy(policy))
        }

        func activate() { steps.append(.activate) }
    }

    func testAccessoryAppFlipsToRegularBeforeActivating() {
        let app = FakeApp(policy: .accessory)
        AppActivation.focus(app)

        // The old code activated while still .accessory (and only flipped later,
        // reactively) — which is precisely why the window came up half-focused.
        XCTAssertEqual(app.steps, [.setPolicy(.regular), .activate])
        XCTAssertEqual(app.currentPolicy, .regular)
    }

    func testAlreadyRegularAppJustActivates() {
        let app = FakeApp(policy: .regular)
        AppActivation.focus(app)
        XCTAssertEqual(app.steps, [.activate], "no redundant policy flip when a window is already open")
    }

    func testFocusIsIdempotent() {
        let app = FakeApp(policy: .accessory)
        AppActivation.focus(app)
        AppActivation.focus(app)
        XCTAssertEqual(app.steps, [.setPolicy(.regular), .activate, .activate])
    }

    // MARK: - Window titles

    /// `openAndFocus` raises the newly-opened window by TITLE. The map lived
    /// inline in a switch inside MLXServeApp, where the three call sites that
    /// bypass `openAndFocus` couldn't reach it.
    func testWindowTitleMapCoversEveryOpenableWindow() {
        XCTAssertEqual(AppActivation.windowTitle(for: "chat"), "MLX Core")
        XCTAssertEqual(AppActivation.windowTitle(for: "modelBrowser"), "Model Browser")
        XCTAssertEqual(AppActivation.windowTitle(for: "settings"), "Settings")
        XCTAssertEqual(AppActivation.windowTitle(for: "imageGen"), "Image Generation")
        XCTAssertEqual(AppActivation.windowTitle(for: "videoGen"), "Video Generation")
        XCTAssertEqual(AppActivation.windowTitle(for: "audioGen"), "Audio Generation")
        XCTAssertEqual(AppActivation.windowTitle(for: "model3dGen"), "3D Generation")
        XCTAssertEqual(AppActivation.windowTitle(for: "serverLog"), "Server Log")
        XCTAssertEqual(AppActivation.windowTitle(for: "tasks"), "Tasks")
    }

    // MARK: - Raising the right window

    /// The raise-by-title lookup was silently dead for the window users open
    /// most: `ChatView` sets `.navigationTitle("")`, so the chat window's
    /// `NSWindow.title` is EMPTY and never equals "MLX Core". Matching must fall
    /// back to the scene identifier.
    func testChatWindowIsMatchedByIdentifierWhenItsTitleIsBlank() {
        XCTAssertTrue(AppActivation.windowMatches(id: "chat", title: "", identifier: "chat"))
        XCTAssertTrue(AppActivation.windowMatches(id: "chat", title: "", identifier: "SwiftUI-Window-chat"))
    }

    func testWindowMatchesByTitleWhenTitled() {
        XCTAssertTrue(AppActivation.windowMatches(id: "settings", title: "Settings", identifier: nil))
        XCTAssertTrue(AppActivation.windowMatches(id: "tasks", title: "Tasks", identifier: nil))
    }

    /// Never grab an unrelated window — raising the wrong one is its own focus bug.
    func testWindowDoesNotMatchAnotherScene() {
        XCTAssertFalse(AppActivation.windowMatches(id: "chat", title: "Settings", identifier: "settings"))
        XCTAssertFalse(AppActivation.windowMatches(id: "chat", title: "", identifier: nil),
                       "a blank, unidentified window is not evidence it's ours")
    }

    // MARK: - Source audit (the universal part)

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe")
    }

    private func swiftSources() throws -> [(name: String, body: String)] {
        let fm = FileManager.default
        guard let en = fm.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil) else { return [] }
        var out: [(String, String)] = []
        for case let url as URL in en where url.pathExtension == "swift" {
            out.append((url.lastPathComponent, try String(contentsOf: url, encoding: .utf8)))
        }
        return out
    }

    /// Every file picker in the app must be presented through `AppActivation`,
    /// which brings the app forward first. A raw `panel.runModal()` from an
    /// inactive accessory app is the "can't click the folder picker" bug — and
    /// there were 16 of them.
    func testNoRawPanelPresentation() throws {
        var offenders: [String] = []
        for (name, body) in try swiftSources() where name != "AppActivation.swift" {
            for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
                let l = line.trimmingCharacters(in: .whitespaces)
                if l.hasPrefix("//") { continue }
                // Alerts are fine (NSAlert.runModal is app-modal and orders itself
                // front); this is about NSOpenPanel/NSSavePanel.
                if l.contains("panel.runModal()") || l.contains("panel.begin(") {
                    offenders.append("\(name): \(l)")
                }
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            File pickers must go through AppActivation.runModal/beginPanel so the \
            accessory app is brought forward first — otherwise the panel opens \
            unfocused and won't take clicks. Offenders:
            \(offenders.joined(separator: "\n"))
            """)
    }

    /// Same rule for windows: opening one while `.accessory` yields a
    /// semi-focused window. Every site goes through `AppActivation.openWindow`.
    func testNoRawOpenWindowCalls() throws {
        var offenders: [String] = []
        for (name, body) in try swiftSources() where name != "AppActivation.swift" {
            for line in body.split(separator: "\n", omittingEmptySubsequences: false) {
                let l = line.trimmingCharacters(in: .whitespaces)
                if l.hasPrefix("//") { continue }
                // The routed form (`AppActivation.openWindow(id:using:)`) contains
                // the raw substring, so it has to be exempted explicitly.
                if l.contains("openWindow(id:"), !l.contains("AppActivation.openWindow(id:") {
                    offenders.append("\(name): \(l)")
                }
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            Windows must be opened via AppActivation.openWindow(id:using:) so the \
            app flips to .regular BEFORE the window appears. Offenders:
            \(offenders.joined(separator: "\n"))
            """)
    }
}
