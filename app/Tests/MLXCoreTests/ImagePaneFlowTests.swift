import XCTest
@testable import MLXCore

/// The Image pane's pure flow logic: how a finished picture becomes the next
/// run's source, and what the one preview shows when two services feed it.
final class ImagePaneFlowTests: XCTestCase {

    // MARK: - Handing a finished result to the enlarger

    func testAFinishedGenerationBecomesTheSourceWithoutAFilePanel() {
        // THE FRICTION THIS REMOVES: the likeliest photo in the app to want
        // enlarged is the one just generated and on screen, and reaching it
        // used to mean Reveal in Finder or an NSOpenPanel aimed at the app's
        // own output folder.
        let out = ImageSourceHandoff.resolve(path: "/tmp/gen/apple.png",
                                             isRunning: false,
                                             exists: { _ in true })
        XCTAssertEqual(out, .accepted(URL(fileURLWithPath: "/tmp/gen/apple.png")))
    }

    func testAResultTheUserAlreadyDeletedIsRefusedRatherThanHandedOver() {
        // `recent` is rebuilt from the output folders and the preview holds a
        // path, so both can outlive the file — a handoff that doesn't check
        // arms the pane with a source whose only symptom is a failed run
        // minutes later, after the checkpoint has loaded.
        let out = ImageSourceHandoff.resolve(path: "/tmp/gen/gone.png",
                                            isRunning: false,
                                            exists: { _ in false })
        XCTAssertEqual(out, .missing("gone.png"))
    }

    func testAnEnlargeInFlightKeepsItsOwnSource() {
        // The button sits on the preview, which keeps drawing while a run is
        // in flight. Swapping the source under a running job would leave the
        // controls describing an input the result did not come from.
        let out = ImageSourceHandoff.resolve(path: "/tmp/gen/apple.png",
                                            isRunning: true,
                                            exists: { _ in true })
        XCTAssertEqual(out, .busy)
    }
}
