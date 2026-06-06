import XCTest
@testable import MLXCore

/// Pins the menu-bar voice status dot's presentation as a pure, **time-free**
/// mapping from turn state → color family.
///
/// Regression context: the tray dot used to be driven by a continuously-firing
/// `TimelineView(.animation)` (and before that a `repeatForever` animation). In
/// this `LSUIElement` accessory app, a never-settling redraw hosted in the
/// `MenuBarExtra(.window)` popover saturates its run loop and starves SwiftUI
/// `Button` hit-testing — every tray button goes dead while the model `Picker`
/// and voice `Menu` keep working from their own NSMenu tracking loop ("the tray
/// locks up but the dropdown still opens"). The fix makes the tray dot *static*:
/// state is encoded as color only, with no time input and no per-frame redraw.
///
/// `VoiceTrayDot.tint(for:)` taking **no time argument** is what guarantees the
/// popover can't be wedged again — there is nothing to animate. These tests lock
/// the color mapping and that time-independence (calling it twice for the same
/// state is identical; the error payload never changes the tint).
final class VoiceTrayDotTests: XCTestCase {

    func testActiveStatesShareTheListeningTint() {
        XCTAssertEqual(VoiceTrayDot.tint(for: .listening), .active)
        XCTAssertEqual(VoiceTrayDot.tint(for: .recognizing), .active)
    }

    func testThinkingSpeakingErrorIdleTints() {
        XCTAssertEqual(VoiceTrayDot.tint(for: .thinking), .thinking)
        XCTAssertEqual(VoiceTrayDot.tint(for: .speaking), .speaking)
        XCTAssertEqual(VoiceTrayDot.tint(for: .error("boom")), .error)
        XCTAssertEqual(VoiceTrayDot.tint(for: .idle), .idle)
    }

    /// The dot must not depend on transient data: any error string maps to the
    /// same error tint, so partial-transcript churn / changing messages can't
    /// make the dot flicker between colors.
    func testErrorTintIgnoresPayload() {
        XCTAssertEqual(VoiceTrayDot.tint(for: .error("a")),
                       VoiceTrayDot.tint(for: .error("b")))
    }

    /// Deterministic by construction — same state in, same tint out. Documents
    /// that there is no time/animation input feeding the tray dot.
    func testTintIsDeterministic() {
        for state: VoiceTurnState in [.idle, .listening, .recognizing, .thinking, .speaking, .error("x")] {
            XCTAssertEqual(VoiceTrayDot.tint(for: state), VoiceTrayDot.tint(for: state))
        }
    }
}
