import XCTest
@testable import MLXCore

final class UsageDashboardTests: XCTestCase {
    /// Two sheets on one window is one sheet plus a thing nobody can see: the blocking gate waits for the dashboard.
    func testTheGateStandsDownWhileTheDashboardIsUp() {
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false, workspace: .conversation,
                                            welcomePresented: false, usagePresented: true))
        XCTAssertTrue(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false, workspace: .conversation,
                                            welcomePresented: false, usagePresented: false))
    }
}
