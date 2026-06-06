import Foundation

/// Pure presentation for the menu-bar voice **status dot**.
///
/// The tray dot is deliberately *static*: a solid color that encodes the turn
/// state, with **no** `TimelineView`, no time input, and no per-frame redraw.
///
/// Why it must be static — and why this is NOT the same as the in-window orb:
/// MLX Core is an `LSUIElement` accessory app, so the `MenuBarExtra(.window)`
/// popover is hosted in a non-key panel. A continuously redrawing view there —
/// a `repeatForever` animation *or* a running `TimelineView(.animation)` —
/// saturates the popover's run loop in the default mode and starves SwiftUI
/// `Button` hit-testing: every tray button goes dead while the model `Picker`
/// and voice `Menu` keep working, because clicking an NSMenu-backed control
/// spins its own `NSEventTrackingRunLoopMode`. That's the "tray locks up but the
/// dropdown still opens" freeze. Swapping `repeatForever` for
/// `TimelineView(.animation)` (the earlier attempt) did not help: both are
/// never-settling redraw loops in the popover.
///
/// The breathe lives only in the in-window orb (`VoiceModeView`), which is in a
/// normal `Window` and is unaffected. Keeping this presentation time-free *by
/// construction* (no time parameter, nothing to animate) is what guarantees the
/// popover can't be wedged again — see `VoiceTrayDotTests` and `VoicePulse`.
enum VoiceTrayDot {
    /// The dot's color family for a turn state. A plain enum (not a SwiftUI
    /// `Color`) so it's `Equatable` and unit-testable without importing SwiftUI;
    /// the view maps each case to a concrete color.
    enum Tint: Equatable {
        case active    // mic open / user talking
        case thinking  // model generating, nothing audible yet
        case speaking  // reading the answer aloud
        case error     // recoverable failure
        case idle      // starting / closed
    }

    /// Map a turn state to its dot tint. Total and deterministic — the same
    /// state always yields the same tint, and the `.error` payload is ignored,
    /// so transient churn can never make the dot flicker.
    static func tint(for state: VoiceTurnState) -> Tint {
        switch state {
        case .listening, .recognizing: return .active
        case .thinking:                return .thinking
        case .speaking:                return .speaking
        case .error:                   return .error
        case .idle:                    return .idle
        }
    }
}
