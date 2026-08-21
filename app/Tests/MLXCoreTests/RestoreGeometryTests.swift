import XCTest
@testable import MLXCore

/// The upscale pane's client-side geometry rules — the /16 snapping the server
/// refuses without, and the memory heads-up that exists because the failure it
/// replaces looked like a success.
final class RestoreGeometryTests: XCTestCase {

    func testTheTargetCanvasIsWhatCostsMemory_notTheSource() {
        // THE BUG THIS PINS, from the user's own output folder (2026-08-21):
        // a 904x960 photo at 2x is a 1808x1920 canvas, and the restore came
        // back as a 1808x1920 PNG whose every pixel was 255 — pure white, a
        // 200, no error anywhere. The run before it came back a flat grey
        // (min 123, max 128). Both are MLX returning degenerate values at the
        // Metal working-set edge, which it does instead of failing.
        //
        // The source is small and the TARGET is what does not fit, so a gate
        // that looks at the source cannot see this at all.
        let target = RestoreGeometry.upscaledTarget(width: 904, height: 960, factor: 2)
        XCTAssertEqual(target.width, 1808)
        XCTAssertEqual(target.height, 1920)

        // 3.47 Mpx x ~4.1 KB/px is ~14 GB of transient on top of the model.
        // THIS PANE WARNS ABOUT THE IMPOSSIBLE, NOT THE UNAVAILABLE: 14 + 7 is
        // under a 24 GB machine's total, so the pane stays quiet there and the
        // server's free-memory gate is what refuses it in practice. On a 16 GB
        // Mac it can never work whatever else is closed, and that is what this
        // sentence is for.
        XCTAssertNil(RestoreGeometry.memoryWarning(
            targetWidth: target.width, targetHeight: target.height,
            modelGB: 7, totalRAMGB: 24))
        let warn = RestoreGeometry.memoryWarning(
            targetWidth: target.width, targetHeight: target.height,
            modelGB: 7, totalRAMGB: 16)
        XCTAssertNotNil(warn)
        // It quotes the numbers it compared, and the canvas it is talking
        // about — a warning that just says "too big" leaves nothing to act on.
        XCTAssertTrue(warn!.contains("1808"), warn!)
        XCTAssertTrue(warn!.contains("16"), warn!)
    }

    func testASizeThatFitsIsNotWarnedAbout() {
        // 1024x1024 is ~4.1 GB of transient (measured peak 11.27 GB against
        // 7.25 GB resident); with a 7 GB model that is ~11 GB on a 24 GB Mac.
        XCTAssertNil(RestoreGeometry.memoryWarning(
            targetWidth: 1024, targetHeight: 1024, modelGB: 7, totalRAMGB: 24))
        // The same canvas that is hopeless on 16 GB is fine on 64 — the gate
        // tracks the machine, it does not ban a size.
        XCTAssertNil(RestoreGeometry.memoryWarning(
            targetWidth: 1808, targetHeight: 1920, modelGB: 7, totalRAMGB: 64))
    }

    func testAnUnknownMachineIsNeverWarnedAbout() {
        // `RAMChecker.totalGB` returning 0 means "could not measure". A gate
        // that cannot see must not refuse everything.
        XCTAssertNil(RestoreGeometry.memoryWarning(
            targetWidth: 4096, targetHeight: 4096, modelGB: 7, totalRAMGB: 0))
    }

    func testSnappingOnlyEverRemovesPixels() {
        // The restore-only path crops to the /16 grid the server needs and
        // must never invent border pixels the photo does not have.
        XCTAssertEqual(RestoreGeometry.snap(905), 896)
        XCTAssertEqual(RestoreGeometry.snap(960), 960)
        XCTAssertEqual(RestoreGeometry.snap(3), 16)
        let crop = RestoreGeometry.centeredCrop(width: 905, height: 960)
        XCTAssertEqual(crop, RestoreGeometry.CropRect(x: 4, y: 0, width: 896, height: 960))
        XCTAssertNil(RestoreGeometry.centeredCrop(width: 896, height: 960))
    }
}
