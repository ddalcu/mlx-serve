import AppKit

/// Keeps `NSApp.activationPolicy` in sync with the app's windows.
///
/// MLX Core is an `LSUIElement` menu-bar app, so it launches as `.accessory`
/// (no Dock icon) — but accessory apps are also invisible to ⌘Tab, which
/// strands any open window (Chat, Audio Generation, the intro window, …).
/// This manager flips the app to `.regular` while at least one real user
/// window is open (or minimized) and back to `.accessory` when the last one
/// goes away, so ⌘Tab and the Dock icon exist exactly when there's a window
/// to switch to. Replaces the old one-off `.regular` flip in ChatView, which
/// only fired for Chat and never reverted.
///
/// Menu-bar chrome never counts: the status item (`NSStatusBarWindow`), the
/// MenuBarExtra tray popover (panel- or window-hosted depending on macOS
/// release — excluded by class name either way), and the ⌃Space quick
/// launcher (non-activating borderless `NSPanel`).
final class ActivationPolicyManager {
    static let shared = ActivationPolicyManager()

    /// Lightweight window descriptor so the policy decision is pure/testable.
    struct WindowInfo {
        let isVisible: Bool
        let isMiniaturized: Bool
        let isPanel: Bool
        let canBecomeMain: Bool
        let className: String

        init(isVisible: Bool, isMiniaturized: Bool, isPanel: Bool,
             canBecomeMain: Bool, className: String) {
            self.isVisible = isVisible
            self.isMiniaturized = isMiniaturized
            self.isPanel = isPanel
            self.canBecomeMain = canBecomeMain
            self.className = className
        }

        init(window: NSWindow) {
            isVisible = window.isVisible
            isMiniaturized = window.isMiniaturized
            isPanel = window is NSPanel
            canBecomeMain = window.canBecomeMain
            className = NSStringFromClass(type(of: window))
        }
    }

    // MARK: - Pure decision

    static func countsAsUserWindow(_ w: WindowInfo) -> Bool {
        // Minimized windows report isVisible == false but must keep the app
        // .regular — dropping to .accessory would make them unreachable.
        guard w.isVisible || w.isMiniaturized else { return false }
        guard !w.isPanel, w.canBecomeMain else { return false }
        if w.className == "NSStatusBarWindow" { return false }
        if w.className.contains("MenuBarExtra") { return false }
        return true
    }

    static func policy(for windows: [WindowInfo]) -> NSApplication.ActivationPolicy {
        windows.contains(where: countsAsUserWindow) ? .regular : .accessory
    }

    // MARK: - Live wiring

    private var observers: [NSObjectProtocol] = []

    /// Call once at launch. Recomputes on key-window changes and window
    /// closes; both are delivered on the main queue.
    func start() {
        guard observers.isEmpty else { return }
        let nc = NotificationCenter.default
        observers.append(nc.addObserver(forName: NSWindow.didBecomeKeyNotification,
                                        object: nil, queue: .main) { [weak self] _ in
            self?.reapply()
        })
        observers.append(nc.addObserver(forName: NSWindow.willCloseNotification,
                                        object: nil, queue: .main) { [weak self] _ in
            // The closing window is still visible during willClose — recompute
            // on the next runloop turn, after it's actually gone.
            DispatchQueue.main.async { self?.reapply() }
        })
        observers.append(nc.addObserver(forName: NSWindow.didMiniaturizeNotification,
                                        object: nil, queue: .main) { [weak self] _ in
            self?.reapply()
        })
        reapply()
    }

    /// Recompute from the full window list (never incremental — a recompute
    /// can't drift, and it naturally handles windows we never saw open).
    func reapply() {
        let desired = Self.policy(for: NSApp.windows.map(WindowInfo.init))
        if NSApp.activationPolicy() != desired {
            NSApp.setActivationPolicy(desired)
        }
    }
}
