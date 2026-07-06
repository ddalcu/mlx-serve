import AppKit

/// Makes a Dock-icon click useful when the app has no window to bring back:
/// it opens the menu-bar tray popover (the app's home surface) instead of
/// doing nothing.
///
/// A Dock click delivers the "reopen" event
/// (`applicationShouldHandleReopen`). We only take over when there is truly
/// nothing for AppKit to restore — no visible windows, no minimized windows
/// (Dock click deminiaturizes those), and the app isn't ⌘H-hidden (Dock click
/// unhides). SwiftUI's `MenuBarExtra` has no public "open the popover" API,
/// so opening the tray = finding the status item's `NSStatusBarButton` among
/// `NSApp.windows` and performing a click on it — the same introspection the
/// MenuBarExtraAccess package uses. If the button isn't found (SwiftUI
/// internals changed), we fall back to default reopen behavior — never a
/// dead click with no fallback.
enum DockReopen {

    /// Pure decision: take over the reopen only when nothing is restorable.
    static func shouldOpenTray(hasVisibleWindows: Bool,
                               appIsHidden: Bool,
                               hasMiniaturizedWindows: Bool) -> Bool {
        !hasVisibleWindows && !appIsHidden && !hasMiniaturizedWindows
    }

    /// Depth-first search for the status item's button in a view hierarchy.
    static func firstStatusBarButton(in view: NSView?) -> NSStatusBarButton? {
        guard let view else { return nil }
        if let button = view as? NSStatusBarButton { return button }
        for sub in view.subviews {
            if let found = firstStatusBarButton(in: sub) { return found }
        }
        return nil
    }

    /// Toggle the MenuBarExtra popover by clicking our status item. Returns
    /// whether a status item was found and clicked.
    @discardableResult
    static func openTrayPopover() -> Bool {
        for window in NSApp.windows
        where NSStringFromClass(type(of: window)).contains("NSStatusBarWindow") {
            if let button = firstStatusBarButton(in: window.contentView) {
                button.performClick(nil)
                return true
            }
        }
        return false
    }
}

/// App delegate installed via `@NSApplicationDelegateAdaptor` — exists solely
/// for the Dock-reopen hook (SwiftUI has no scene-level equivalent).
final class MLXCoreAppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldHandleReopen(_ sender: NSApplication,
                                       hasVisibleWindows flag: Bool) -> Bool {
        let minimized = sender.windows.contains { $0.isMiniaturized }
        if DockReopen.shouldOpenTray(hasVisibleWindows: flag,
                                     appIsHidden: sender.isHidden,
                                     hasMiniaturizedWindows: minimized),
           DockReopen.openTrayPopover() {
            return false  // handled — don't let SwiftUI spawn a window
        }
        return true
    }
}
