import Foundation

/// Time-driven "breathe" math for the in-window voice orb (`VoiceModeView`).
///
/// IMPORTANT — orb only, never the tray. A continuously-firing breathe wedges
/// SwiftUI `Button` hit-testing when it's hosted inside the `MenuBarExtra(.window)`
/// popover of this `LSUIElement` app: the tray's buttons stop responding to
/// clicks while AppKit pop-up controls (the model `Picker`, the voice `Menu`)
/// keep working from their own event-tracking loop — the "tray locks up but the
/// dropdown still opens" report. This is true for BOTH a `repeatForever`
/// animation and a running `TimelineView(.animation)`: both are never-settling
/// redraw loops. The tray status dot is therefore *static* (see `VoiceTrayDot`).
///
/// The orb lives in a normal `Window`, where a `TimelineView` driving these pure
/// functions is fine. Keeping the curve here (instead of inline in the view)
/// also makes it unit-testable — see `VoicePulseTests`.
enum VoicePulse {
    /// Normalized breathe phase in `0...1` at time `t` (a smooth sine, one full
    /// cycle per `period` seconds). `t` is any monotonic seconds value, e.g.
    /// `timelineDate.timeIntervalSinceReferenceDate`.
    static func phase(at t: TimeInterval, period: TimeInterval = 2.2) -> Double {
        guard period > 0 else { return 1 }
        return (sin(2 * .pi * t / period) + 1) / 2
    }

    /// Extra scale added to the orb's mic-level scale: `0` when not animating,
    /// otherwise a breathe between `0` and `amplitude`.
    static func orbBreathe(animating: Bool, at t: TimeInterval,
                           period: TimeInterval = 3.0, amplitude: Double = 0.05) -> Double {
        guard animating else { return 0 }
        return amplitude * phase(at: t, period: period)
    }
}
