import XCTest
import AppKit
@testable import MLXCore

/// #133: quitting via ⌘Q or the Quit menu left `mlx-serve` running with a
/// model resident, eating gigabytes until it was killed by hand. The tray's
/// power button was the only quit path that worked, because it is the only one
/// that called `server.stop()` before `NSApplication.terminate` — every other
/// route (⌘Q, the app menu, Dock ▸ Quit, `terminate` from anywhere else) goes
/// straight to termination and nothing signalled the child.
///
/// The teardown belongs to the object that SPAWNED the process, not to one
/// button, so `ServerManager` observes the termination notification itself.
@MainActor
final class ServerQuitTeardownTests: XCTestCase {

    func testTerminationNotificationStopsTheServer() {
        let server = ServerManager()
        server.status = .running

        NotificationCenter.default.post(name: NSApplication.willTerminateNotification,
                                       object: NSApplication.shared)

        XCTAssertEqual(server.status, .stopped,
                       "a quit that does not stop the server orphans mlx-serve with the model resident")
    }

    /// Delivery must be SYNCHRONOUS on the posting thread (`queue: nil`), not
    /// hopped onto the main queue. `applicationWillTerminate` is the app's last
    /// runloop turn — work enqueued for the next one may never run, which is
    /// the same leak with extra steps. Posting and asserting with no runloop
    /// turn in between is what pins it.
    func testTeardownRunsBeforeTheAppCanExit() {
        let server = ServerManager()
        server.status = .starting
        NotificationCenter.default.post(name: NSApplication.willTerminateNotification, object: nil)
        // No `await`, no runloop spin: if this passes, the handler already ran.
        XCTAssertEqual(server.status, .stopped)
    }
}
