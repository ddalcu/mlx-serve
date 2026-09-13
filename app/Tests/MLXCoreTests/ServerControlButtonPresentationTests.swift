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

    /// A headless start must not claim to be loading a model.
    func testAHeadlessStartDoesNotClaimToBeLoadingAModel() {
        let starting = ServerControlButtonPresentation(status: .starting, loadsModel: false)
        XCTAssertEqual(starting.title, "Starting Server...")
        let stopped = ServerControlButtonPresentation(status: .stopped, loadsModel: false)
        XCTAssertTrue(stopped.help.contains("no model resident"))
        XCTAssertTrue(ServerControlButtonPresentation(status: .stopped).help.contains("load the selected model"))
    }

    /// A hot-load in flight reads as loading even though the server is running.
    func testALoadingModelOutranksRunning() {
        let loading = ServerControlButtonPresentation(status: .running, isLoadingModel: true)
        XCTAssertEqual(loading.title, "Loading Model...")
        XCTAssertTrue(loading.showsProgress)
        XCTAssertEqual(ServerControlButtonPresentation(status: .running).title, "Stop Server")
    }
}
