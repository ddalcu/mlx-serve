import AppKit
import Combine

/// Is ⌘ being HELD right now?
///
/// The sidebar numbers its conversation rows while the key is down, which needs
/// the modifier as a piece of continuous STATE — and SwiftUI has no such thing.
/// `.onModifierKeysChanged` reports only over a hovered view, and a `keyDown`
/// monitor never fires for a modifier pressed on its own; the event that
/// carries it is `.flagsChanged`.
///
/// Deliberately not a general "which modifiers are down" service: one published
/// boolean, so nothing re-renders on Shift or Option.
final class ModifierKeyMonitor: ObservableObject {

    @Published private(set) var commandHeld = false

    private var flagsMonitor: Any?
    private var resignObserver: NSObjectProtocol?

    init() {
        // Local: this app's events only. A global monitor would need
        // Accessibility permission to watch every keystroke on the machine,
        // which is an outrageous price for a badge.
        flagsMonitor = NSEvent.addLocalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.apply(event.modifierFlags)
            return event          // never swallowed — ⌘ still reaches everything else
        }
        // ⌘-Tab is a key-DOWN we see and a key-UP we do not: the release lands
        // in whatever app the user switched to, so without this the badges
        // stay on until the next time ⌘ is pressed and let go here.
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            self?.setHeld(false)
        }
    }

    deinit {
        // Copied out first: `deinit` must not hop actors to read them.
        if let flagsMonitor { NSEvent.removeMonitor(flagsMonitor) }
        if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
    }

    private func apply(_ flags: NSEvent.ModifierFlags) {
        setHeld(flags.intersection(.deviceIndependentFlagsMask).contains(.command))
    }

    private func setHeld(_ value: Bool) {
        guard commandHeld != value else { return }   // publish only on a real edge
        if Thread.isMainThread { commandHeld = value }
        else { DispatchQueue.main.async { [weak self] in self?.commandHeld = value } }
    }
}
