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

    /// A headless start is up in a second and puts nothing resident, so the
    /// button must not describe work that is not happening.
    func testAHeadlessStartDoesNotClaimToBeLoadingAModel() {
        let starting = ServerControlButtonPresentation(status: .starting, loadsModel: false)
        XCTAssertEqual(starting.title, "Starting Server...")
        let stopped = ServerControlButtonPresentation(status: .stopped, loadsModel: false)
        XCTAssertTrue(stopped.help.contains("no model resident"))
        XCTAssertTrue(ServerControlButtonPresentation(status: .stopped).help.contains("load the selected model"))
    }
}
