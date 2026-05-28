import XCTest
@testable import MLXCore

final class ServerControlButtonPresentationTests: XCTestCase {
    func testStartingShowsLoadingProgressWhileRemainingStoppable() {
        let presentation = ServerControlButtonPresentation(status: .starting)

        XCTAssertEqual(presentation.title, "Loading Model...")
        XCTAssertTrue(presentation.showsProgress)
        XCTAssertNil(presentation.systemImageName)
        XCTAssertEqual(presentation.tint, .loading)
        XCTAssertEqual(presentation.help, "Loading model. Click to stop.")
    }

    func testRunningShowsStopPresentation() {
        let presentation = ServerControlButtonPresentation(status: .running)

        XCTAssertEqual(presentation.title, "Stop Server")
        XCTAssertFalse(presentation.showsProgress)
        XCTAssertEqual(presentation.systemImageName, "stop.fill")
        XCTAssertEqual(presentation.tint, .red)
    }

    func testStoppedShowsStartPresentation() {
        let presentation = ServerControlButtonPresentation(status: .stopped)

        XCTAssertEqual(presentation.title, "Start Server")
        XCTAssertFalse(presentation.showsProgress)
        XCTAssertEqual(presentation.systemImageName, "play.fill")
        XCTAssertEqual(presentation.tint, .accent)
    }
}
