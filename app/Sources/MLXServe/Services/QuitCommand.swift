import AppKit

/// The app menu's Quit item (⌘Q). With `ServerOptions.quitKeepsRunning` on it
/// does what closing the window does: the windows go, the app and the server
/// stay in the menu bar. Only this menu item is redirected, never
/// `applicationShouldTerminate`: the tray's Quit, `AppRelaunch` and a logout
/// all terminate through that, and must still quit.
enum QuitCommand {
    enum Action: Equatable {
        case terminate
        /// Indices into the window list.
        case closeWindows([Int])
    }

    static func action(keepRunning: Bool,
                       windows: [ActivationPolicyManager.WindowInfo]) -> Action {
        guard keepRunning else { return .terminate }
        return .closeWindows(windows.indices.filter {
            ActivationPolicyManager.countsAsUserWindow(windows[$0])
        })
    }

    /// Reads the setting at the keypress: the scene that hosts the menu observes nothing.
    @MainActor static func perform() {
        let windows = NSApp.windows
        switch action(keepRunning: ServerOptions.load().quitKeepsRunning,
                      windows: windows.map(ActivationPolicyManager.WindowInfo.init)) {
        case .terminate:
            NSApp.terminate(nil)
        case .closeWindows(let indices):
            for i in indices { windows[i].close() }
        }
    }
}
