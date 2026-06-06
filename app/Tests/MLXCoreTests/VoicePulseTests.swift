import XCTest
@testable import MLXCore

/// Pins the time-driven pulse math for the in-window voice **orb**
/// (`VoiceModeView`).
///
/// Regression context: a repeating breathe hosted inside the
/// `MenuBarExtra(.window)` popover (whether `repeatForever` or a running
/// `TimelineView(.animation)`) wedges SwiftUI's event handling — the tray's
/// `Button`s stop responding to clicks while AppKit pop-up controls (the model
/// `Picker`, the voice `Menu`) keep working from their own event-tracking loop
/// ("the dropdown is still clickable but nothing else is"). So the *tray* dot is
/// static (`VoiceTrayDot`); the breathe lives only in the orb, which is in a
/// normal `Window`. This test locks the orb's curve so nobody reintroduces a
/// constant (dead) or out-of-range pulse.
final class VoicePulseTests: XCTestCase {

    func testPhaseStaysInUnitRange() {
        for i in 0..<400 {
            let p = VoicePulse.phase(at: Double(i) * 0.05)
            XCTAssertGreaterThanOrEqual(p, 0)
            XCTAssertLessThanOrEqual(p, 1)
        }
    }

    func testPhaseIsPeriodic() {
        let period = 2.2
        XCTAssertEqual(VoicePulse.phase(at: 0.7, period: period),
                       VoicePulse.phase(at: 0.7 + period, period: period),
                       accuracy: 1e-9)
    }

    func testPhaseActuallyVaries() {
        // The whole point of the fix: a *live* breathe, not a constant. A quarter
        // period in, the phase must have moved a lot.
        let period = 2.2
        let a = VoicePulse.phase(at: 0, period: period)
        let b = VoicePulse.phase(at: period / 4, period: period)
        XCTAssertGreaterThan(abs(a - b), 0.3)
    }

    func testOrbBreatheZeroWhenNotAnimating() {
        XCTAssertEqual(VoicePulse.orbBreathe(animating: false, at: 1.23), 0)
    }

    func testOrbBreatheWithinAmplitude() {
        let amp = 0.05
        for i in 0..<400 {
            let s = VoicePulse.orbBreathe(animating: true, at: Double(i) * 0.04, amplitude: amp)
            XCTAssertGreaterThanOrEqual(s, -1e-9)
            XCTAssertLessThanOrEqual(s, amp + 1e-9)
        }
    }
}
