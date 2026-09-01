import SwiftUI
import AppKit
import Carbon.HIToolbox

// MARK: - Pure capture rules

/// What one key event means to the recorder — pure so the rules are testable
/// without synthesizing NSEvents.
enum HotKeyCapture {
    enum Verdict: Equatable {
        case captured(keyCode: UInt32, modifiers: UInt32)
        /// A bare letter/digit would fight every other app's typing the
        /// instant this combo won a global registration.
        case needsModifier
        /// A key `HotKeyDisplay` cannot name would render as "Key 47" in
        /// Settings and the tray — refused here rather than bound unnameable.
        case unmapped
    }

    static func verdict(keyCode: UInt32, modifiers: UInt32) -> Verdict {
        guard modifiers != 0 else { return .needsModifier }
        guard HotKeyDisplay.hasName(keyCode) else { return .unmapped }
        return .captured(keyCode: keyCode, modifiers: modifiers)
    }

    static func modifiers(from flags: NSEvent.ModifierFlags) -> UInt32 {
        var mods: UInt32 = 0
        if flags.contains(.control) { mods |= UInt32(controlKey) }
        if flags.contains(.option) { mods |= UInt32(optionKey) }
        if flags.contains(.shift) { mods |= UInt32(shiftKey) }
        if flags.contains(.command) { mods |= UInt32(cmdKey) }
        return mods
    }
}

// MARK: - AppKit capture view

/// Captures the next key chord typed while it holds first responder — the
/// AppKit half of `HotKeyRecorderControl`. A plain SwiftUI `.onKeyPress`
/// can't claim key equivalents or ⎋-to-cancel the way a raw NSView can.
private final class HotKeyCaptureView: NSView {
    var onCapture: ((UInt32, UInt32) -> Void)?
    var onCancel: (() -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        handle(event)
    }

    /// ⌘-combos never reach `keyDown`: the window offers every key equivalent
    /// to the view hierarchy and then the menu bar FIRST (the same fact behind
    /// `ChatDeleteShortcut`), so recording ⌘W would close the Settings window
    /// instead of binding it. While this view holds first responder the chord
    /// IS the recording — claim it.
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard window?.firstResponder === self else { return super.performKeyEquivalent(with: event) }
        handle(event)
        return true
    }

    /// Clicking anywhere else disarms the recorder — without this the button
    /// keeps saying "Press keys…" while the keystrokes go elsewhere.
    override func resignFirstResponder() -> Bool {
        onCancel?()
        return super.resignFirstResponder()
    }

    private func handle(_ event: NSEvent) {
        if event.keyCode == UInt16(kVK_Escape) {
            onCancel?()
            return
        }
        switch HotKeyCapture.verdict(keyCode: UInt32(event.keyCode),
                                     modifiers: HotKeyCapture.modifiers(from: event.modifierFlags)) {
        case .captured(let keyCode, let mods):
            onCapture?(keyCode, mods)
        case .needsModifier, .unmapped:
            NSSound.beep()
        }
    }
}

private struct HotKeyCaptureRepresentable: NSViewRepresentable {
    var isRecording: Bool
    var onCapture: (UInt32, UInt32) -> Void
    var onCancel: () -> Void

    func makeNSView(context: Context) -> HotKeyCaptureView {
        let view = HotKeyCaptureView()
        view.onCapture = onCapture
        view.onCancel = onCancel
        return view
    }

    func updateNSView(_ view: HotKeyCaptureView, context: Context) {
        view.onCapture = onCapture
        view.onCancel = onCancel
        // First responder follows `isRecording` in BOTH directions: leaving it
        // held after a capture would keep swallowing (and beeping at) every
        // later keystroke in the window. The resign fires `onCancel`, which is
        // idempotent by then.
        DispatchQueue.main.async {
            if isRecording {
                if view.window?.firstResponder !== view { view.window?.makeFirstResponder(view) }
            } else {
                if view.window?.firstResponder === view { view.window?.makeFirstResponder(nil) }
            }
        }
    }
}

// MARK: - Settings control

/// The control half of a settings row: a "Record Shortcut" button + Reset.
/// Click to arm, press a combo (must include a modifier) to bind it, or ⎋ to
/// back out unchanged. Pair with `SettingsRow` for the title/explainer.
struct HotKeyRecorderControl: View {
    /// Applies the stored combo to the live registration; false means the
    /// system refused it (another app already owns the combo).
    let onChange: () -> Bool

    @State private var recording = false
    @State private var display = QuickLauncherHotKey.display
    @State private var refusedCombo: String? = nil

    var body: some View {
        VStack(alignment: .trailing, spacing: 4) {
            HStack {
                ZStack {
                    HotKeyCaptureRepresentable(
                        isRecording: recording,
                        onCapture: { keyCode, mods in apply(keyCode: keyCode, carbonModifiers: mods) },
                        onCancel: { recording = false }
                    )
                    .frame(width: 1, height: 1)

                    Button(recording ? "Press keys… (⎋ to cancel)" : display) {
                        recording = true
                    }
                    .frame(minWidth: 140)
                }
                Button("Reset") {
                    QuickLauncherHotKeyStore.reset()
                    display = QuickLauncherHotKey.display
                    refusedCombo = nil
                    _ = onChange()
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .disabled(QuickLauncherHotKeyStore.isDefault)
            }
            if let refused = refusedCombo {
                Text("\(refused) is taken by another app — kept \(display).")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    /// `RegisterEventHotKey` refuses a combo another app already owns, and
    /// silently keeping a refused combo is a launcher that just stops
    /// summoning — revert to the previous binding and say so in the row.
    private func apply(keyCode: UInt32, carbonModifiers: UInt32) {
        let hadOverride = !QuickLauncherHotKeyStore.isDefault
        let previous = (QuickLauncherHotKeyStore.keyCode, QuickLauncherHotKeyStore.carbonModifiers)
        QuickLauncherHotKeyStore.set(keyCode: keyCode, carbonModifiers: carbonModifiers)
        recording = false
        if onChange() {
            refusedCombo = nil
        } else {
            refusedCombo = HotKeyDisplay.string(keyCode: keyCode, carbonModifiers: carbonModifiers)
            if hadOverride {
                QuickLauncherHotKeyStore.set(keyCode: previous.0, carbonModifiers: previous.1)
            } else {
                QuickLauncherHotKeyStore.reset()
            }
            _ = onChange()
        }
        display = QuickLauncherHotKey.display
    }
}
