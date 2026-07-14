import AppKit
import SwiftUI

/// The seam that makes the ordering rule testable without a live `NSApp`.
@MainActor
protocol AppActivating: AnyObject {
    var currentPolicy: NSApplication.ActivationPolicy { get }
    func setPolicy(_ policy: NSApplication.ActivationPolicy)
    func activate()
}

/// Brings this `LSUIElement` app properly forward before it presents UI.
///
/// THE RULE: flip to `.regular` **before** the window or panel is presented,
/// then activate. Never the other way round.
///
/// Why it matters: an accessory app that isn't active has NO key window. So a
/// window opened while `.accessory` is ordered on screen but never becomes key —
/// no blinking caret, no first responder, the title bar half-lit. And
/// `ActivationPolicyManager` can't rescue it, because the only thing it listens
/// for (`didBecomeKeyNotification`) cannot fire until the app is active. The
/// window therefore sits semi-focused until the user clicks or types, which
/// finally activates the app, which finally fires the notification, which
/// finally flips the policy — the "if I start typing it gets activated" symptom.
/// A modal `NSOpenPanel` presented from an inactive accessory app fails the same
/// way: it opens behind/unfocused and won't take clicks until you fight with it.
///
/// `openAndFocus` used to do `openWindow(id:)` → `activate()` with the policy
/// still `.accessory`, which is the inverted order.
@MainActor
enum AppActivation {

    /// One step of the sequence. Recorded (and asserted on) in tests, because
    /// the ORDER is the bug.
    enum Step: Equatable {
        case setPolicy(NSApplication.ActivationPolicy)
        case activate
    }

    /// The live app.
    private final class RealApp: AppActivating {
        var currentPolicy: NSApplication.ActivationPolicy { NSApp.activationPolicy() }
        func setPolicy(_ policy: NSApplication.ActivationPolicy) { NSApp.setActivationPolicy(policy) }
        func activate() { NSApp.activate(ignoringOtherApps: true) }
    }

    private static let real = RealApp()

    /// Make the app a foregroundable, activatable app RIGHT NOW — synchronously,
    /// so it has taken effect before the caller presents anything (a modal panel
    /// spins its own run loop, so deferring the activation to a later main-queue
    /// turn would land too late).
    static func focus(_ app: AppActivating? = nil) {
        let target = app ?? real
        if target.currentPolicy != .regular {
            target.setPolicy(.regular)
        }
        target.activate()
    }

    // MARK: - Windows

    /// Title of the window a scene id opens — used to raise the right one once
    /// SwiftUI has created it. Lived inline in a `switch` in MLXServeApp, where
    /// the call sites that bypassed `openAndFocus` couldn't reach it.
    static func windowTitle(for id: String) -> String {
        switch id {
        case "chat":         return "MLX Core"
        case "modelBrowser": return "Model Browser"
        case "imageGen":     return "Image Generation"
        case "videoGen":     return "Video Generation"
        case "audioGen":     return "Audio Generation"
        case "model3dGen":   return "3D Generation"
        case "settings":     return "Settings"
        case "serverLog":    return "Server Log"
        case "tasks":        return "Tasks"
        default:             return "Browser"
        }
    }

    /// Is this NSWindow the one scene `id` opens?
    ///
    /// Title alone is NOT enough: `ChatView` sets `.navigationTitle("")`, so the
    /// chat window — the one users open most — has an EMPTY `NSWindow.title` and
    /// never matched the old `title == "MLX Core"` lookup, which meant the raise
    /// silently did nothing for it. Fall back to the scene identifier, and match
    /// nothing at all rather than grab an unrelated window.
    static func windowMatches(id: String, title: String, identifier: String?) -> Bool {
        if !title.isEmpty, title == windowTitle(for: id) { return true }
        if let identifier, identifier.localizedCaseInsensitiveContains(id) { return true }
        return false
    }

    /// Open a SwiftUI window scene with real focus. The ONLY way the app opens a
    /// window (pinned by `AppActivationTests.testNoRawOpenWindowCalls`).
    static func openWindow(id: String, using open: OpenWindowAction) {
        // .regular FIRST: the window must be created by an app that is already
        // allowed to be frontmost, or it comes up unemphasized.
        focus()
        open(id: id)
        // The window doesn't exist until SwiftUI has built it, so raise it (and
        // re-assert activation) on the next turn of the run loop. If we can't
        // identify it, leave it be — `openWindow` already ordered it front, and
        // an active app makes its front window key.
        DispatchQueue.main.async {
            NSApp.activate(ignoringOtherApps: true)
            NSApp.windows
                .first { windowMatches(id: id, title: $0.title, identifier: $0.identifier?.rawValue) }?
                .makeKeyAndOrderFront(nil)
        }
    }

    // MARK: - Panels

    /// Run a file picker modally, focused. The ONLY way the app runs a panel
    /// modally (pinned by `AppActivationTests.testNoRawPanelPresentation`).
    ///
    /// `ActivationPolicyManager` deliberately doesn't count panels as user
    /// windows (the quick launcher is a panel), so it won't flip the policy for
    /// us — this must.
    @discardableResult
    static func runModal(_ panel: NSSavePanel) -> NSApplication.ModalResponse {
        focus()
        panel.level = .modalPanel
        let response = panel.runModal()
        // The picker may have been the only thing keeping us .regular.
        ActivationPolicyManager.shared.reapply()
        return response
    }

    /// Non-modal (`begin`) variant, for pickers presented from a callback.
    static func beginPanel(_ panel: NSSavePanel, completion: @escaping (NSApplication.ModalResponse) -> Void) {
        focus()
        panel.level = .modalPanel
        panel.begin { response in
            completion(response)
            ActivationPolicyManager.shared.reapply()
        }
    }
}
