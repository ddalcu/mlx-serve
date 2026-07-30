import XCTest
@testable import MLXCore

/// Which agent the Agents window opens on.
///
/// It used to be "the first one, if nothing is selected yet" — fine when the
/// only way in was the menu bar, wrong the moment a locked composer disc says
/// "Set by Chef · Edit Agent…" and lands you on whoever sorts first. The window
/// is also a `Window` (one instance, reused), so a deep link has to retarget a
/// window that is ALREADY open and already showing someone else.
final class AgentsWindowFocusTests: XCTestCase {

    private let chef = UUID()
    private let coder = UUID()
    private let first = UUID()

    /// The whole point: the request wins over whatever the window was showing.
    func testADeepLinkRetargetsAnAlreadyOpenWindow() {
        XCTAssertEqual(AgentsWindowFocus.selection(pending: chef, current: coder, first: first), chef)
    }

    func testADeepLinkWinsOnAColdOpenToo() {
        XCTAssertEqual(AgentsWindowFocus.selection(pending: chef, current: nil, first: first), chef)
    }

    /// Cold open with no request — the old behaviour, unchanged.
    func testAColdOpenWithNoRequestLandsOnTheFirstAgent() {
        XCTAssertEqual(AgentsWindowFocus.selection(pending: nil, current: nil, first: first), first)
    }

    /// nil means "leave the selection alone". Re-running this on every publish
    /// must not yank the user back to the top of the list mid-edit.
    func testAnExistingSelectionIsLeftAloneWithoutARequest() {
        XCTAssertNil(AgentsWindowFocus.selection(pending: nil, current: coder, first: first))
    }

    func testNothingToSelectIsNotASelection() {
        XCTAssertNil(AgentsWindowFocus.selection(pending: nil, current: nil, first: nil))
    }

    /// A deep link to the agent already on screen still has to be honoured —
    /// the window's `onChange(of: selectedId)` can't fire for an unchanged id,
    /// so the caller reloads the draft itself rather than treating it as a
    /// no-op (the click would otherwise do nothing at all).
    func testADeepLinkToTheAgentAlreadyShowingIsStillASelection() {
        XCTAssertEqual(AgentsWindowFocus.selection(pending: chef, current: chef, first: first), chef)
    }
}
