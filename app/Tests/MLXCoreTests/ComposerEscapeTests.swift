import XCTest
@testable import MLXCore

/// Escape stops the reply that is being written.
///
/// The only way to stop a generation was the Send button turning into a red
/// stop disc — a mouse target, at the far end of the row from where you are
/// typing. Escape is what every other streaming UI uses and it was doing
/// nothing.
///
/// WHERE this is handled is the load-bearing decision. Escape is claimed by two
/// existing controls: the edit bubble's Cancel and the tool-approval sheet's
/// Deny, both `.keyboardShortcut(.cancelAction)` — AppKit KEY EQUIVALENTS,
/// which are offered the keystroke BEFORE the responder chain sees a keyDown.
/// So stop-generation deliberately does NOT add a third key equivalent, which
/// would be a three-way race with no documented winner and would break Escape
/// while editing a message. It rides `cancelOperation(_:)` in the composer's
/// own text view — the responder chain — which puts the precedence in the
/// right order by construction:
///
///   editing a message   → the edit's Cancel button (key equivalent) wins
///   approving a tool    → the sheet's Deny (key equivalent) wins
///   otherwise, typing   → the composer field stops the turn
final class ComposerEscapeTests: XCTestCase {

    /// Comments stripped: this file's own doc comment explains the two
    /// `.cancelAction` claims below, and a raw grep counts that as a third.
    private var chatViewSource: String {
        SourceScan.source("Views/ChatView.swift", from: #filePath)
    }

    // MARK: - The decision

    func testEscapeStopsAGenerationInFlight() {
        XCTAssertEqual(ComposerKey.onEscape(isGenerating: true), .stop)
    }

    func testEscapeIsPassedThroughWhenNothingIsGenerating() {
        // NOT swallowed. With no turn to stop, Escape has to keep whatever
        // meaning AppKit gives it here — dismissing a popover, clearing a
        // field's selection. A key that silently does nothing is worse than
        // one that does the platform's thing.
        XCTAssertEqual(ComposerKey.onEscape(isGenerating: false), .pass)
    }

    // MARK: - The wiring

    func testTheComposerFieldHandlesCancelOperation() {
        let source = chatViewSource
        XCTAssertFalse(source.isEmpty, "could not read ChatView.swift")
        XCTAssertTrue(source.contains("NSResponder.cancelOperation(_:)"),
                      "Escape reaches the composer through cancelOperation — the responder chain, "
                      + "so the two .cancelAction buttons still win when they are on screen")
    }

    func testStopGenerationAddsNoCompetingCancelActionShortcut() {
        // The guard for the race this design avoids: exactly the two controls
        // that legitimately own Escape may claim it as a key equivalent. A
        // third would take Escape away from the edit bubble.
        let source = chatViewSource
        let claims = SourceScan.count(".keyboardShortcut(.cancelAction)", in: source)
        XCTAssertEqual(claims, 2,
                       "expected exactly two .cancelAction claims in ChatView (the edit bubble's Cancel "
                       + "and the tool-approval Deny); a new one races them for Escape")
    }

    func testTheComposerStopsThisSessionsTurn() {
        // Stopping is per-session: other tabs may be generating and Escape in
        // this one must not reach across.
        let source = chatViewSource
        guard let range = source.range(of: "private var composerField: some View {") else {
            return XCTFail("composerField is gone — has the composer been renamed?")
        }
        let body = String(source[range.lowerBound...].prefix(900))
        XCTAssertTrue(body.contains("onCancel:"), "the composer field must pass an onCancel")
        XCTAssertTrue(body.contains("stopGeneration()"),
                      "onCancel routes through stopGeneration(), the same call the stop disc makes")
    }

    func testTheStopDiscAndEscapeShareOneCall() {
        // Two ways to stop must not become two implementations of stopping.
        let source = chatViewSource
        let calls = SourceScan.count("chatEngine.stop(sessionId: sessionId)", in: source)
        XCTAssertEqual(calls, 1,
                       "stopping the turn lives in exactly one place (stopGeneration); "
                       + "the disc and Escape both call it")
    }
}
