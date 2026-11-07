import XCTest
@testable import MLXCore

/// A `kill` or a closed terminal must quit the app the way ⌘Q does, so `willTerminate` stops the server it started.
final class QuitSignalsTests: XCTestCase {
    func testASignalRunsTheQuitInsteadOfKillingTheProcess() {
        let quit = expectation(description: "quit ran")
        let sources = QuitSignals.install([SIGUSR2]) { quit.fulfill() }
        defer { sources.forEach { $0.cancel() }; signal(SIGUSR2, SIG_DFL) }
        kill(getpid(), SIGUSR2)
        wait(for: [quit], timeout: 2)
    }
}
