import XCTest
@testable import MLXCore

/// The app must not be a tighter gate than the server it drives (#254).
///
/// The H3 pane hard-locked Steps at 16 and Frames at 124, while
/// `gen.handleVideoH3` accepts any step count (its only floor is turbo's own
/// 4) and snaps `num_frames` onto the 17k+5 ladder from 5 up. That made a
/// community few-step adapter — a REF2VA Turbo distillation, which the pane's
/// own Style LoRAs slot exists to load — unusable without hand-rolling the
/// HTTP request or rebuilding the app, and it made a cheap short test clip
/// unreachable at any settings.
///
/// The floors survive as ADVICE: a range the model works badly in is worth a
/// sentence, not a wall.
final class H3ReachableRangeTests: XCTestCase {

    private let h3 = VideoModelPreset.minimaxH3
    private let ref2va = VideoModelPreset.minimaxH3Ref2VA

    // MARK: - Steps

    func testEveryH3PresetOffersTheServersOwnStepFloor() {
        // 4 is the lowest the server will run (turbo's distillation floor);
        // nothing about the REF2VA partition raises it, and REF2VA is exactly
        // the pack with no Turbo toggle to unlock the low end.
        for model in [h3, VideoModelPreset.minimaxH3Q4, ref2va] {
            XCTAssertEqual(model.stepsRange.lowerBound, 4,
                           "\(model.id) locks the Steps slider above what the server accepts")
            XCTAssertEqual(model.stepsRange.upperBound, 50)
        }
    }

    func testLowStepsAdviseADistilledAdapterInsteadOfBeingBlocked() {
        XCTAssertEqual(h3.testedStepsFloor, 16, "16 is the lowest tier we have a verdict on")
        // Undistilled below the floor: say so.
        XCTAssertNotNil(ref2va.stepsAdvisory(steps: 4, distilled: false))
        XCTAssertNotNil(ref2va.stepsAdvisory(steps: 15, distilled: false))
        // A few-step adapter is the whole reason the low end is reachable —
        // warning there would be advice against what the user just set up.
        XCTAssertNil(ref2va.stepsAdvisory(steps: 4, distilled: true))
        // At or above the floor there is nothing to say either way.
        XCTAssertNil(ref2va.stepsAdvisory(steps: 16, distilled: false))
        XCTAssertNil(ref2va.stepsAdvisory(steps: 30, distilled: false))
    }

    func testBackendsWithoutAStepVerdictNeverAdvise() {
        // LTX's range is tested end to end, so it declares no floor and the
        // advisory must stay silent rather than invent one.
        let ltx = VideoModelPreset.ltx25Q8
        XCTAssertEqual(ltx.testedStepsFloor, 0)
        XCTAssertNil(ltx.stepsAdvisory(steps: ltx.stepsRange.lowerBound, distilled: false))
    }

    // MARK: - Frames

    func testFrameLadderReachesTheEnginesOwnFloor() {
        // `minimax_h3.temporalShape` aligns `max(5, length)` onto 17k+5, so 5
        // is the engine floor and 56 is the server's own default length — the
        // pane could not ask for either.
        XCTAssertEqual(h3.frameOptions.first, 5)
        XCTAssertTrue(h3.frameOptions.contains(56), "the server's default clip length must be reachable")
        XCTAssertTrue(h3.frameOptions.contains(124))
        XCTAssertEqual(h3.frameOptions.last, 362)
        for n in h3.frameOptions {
            XCTAssertEqual(n % 17, 5, "\(n) is off the 17k+5 ladder")
        }
    }

    func testQualityTiersStillDefaultToTheValidatedLengths() {
        // Lowering the ladder floor must not lower any DEFAULT: a preset is a
        // recommendation, and 5 frames is a test clip, not a render.
        for q in QualityPreset.allCases {
            XCTAssertGreaterThanOrEqual(h3.settings(q).numFrames, 124,
                                        "\(q) defaults below the model's stated range")
            XCTAssertTrue(h3.frameOptions.contains(h3.settings(q).numFrames))
        }
    }

    func testShortClipsAdviseRatherThanDisappear() {
        XCTAssertEqual(h3.testedFrameFloor, 107,
                       "107 is the lowest rung covering MiniMax's own stated 4-second minimum")
        XCTAssertNotNil(h3.framesAdvisory(5))
        XCTAssertNotNil(h3.framesAdvisory(90))
        XCTAssertNil(h3.framesAdvisory(107))
        XCTAssertNil(h3.framesAdvisory(124))
        XCTAssertEqual(VideoModelPreset.ltx25Q8.testedFrameFloor, 0)
        XCTAssertNil(VideoModelPreset.ltx25Q8.framesAdvisory(9))
    }

    // MARK: - Unattended paths keep the old floor

    func testChatPreviewsStayInsideTheTestedRange() {
        // The slider reaches below the trained range because it warns there.
        // The chat tool has no slider and no reader — a `generate_video` call
        // must not quietly produce a 1-second off-distribution clip just
        // because the ladder now goes that low.
        for raw in [nil, "0", "0.5", "1", "2", "4", "60", "nonsense"] {
            let n = MediaToolArgs.videoFrames(raw, model: h3)
            XCTAssertGreaterThanOrEqual(n, h3.testedFrameFloor, "\(raw ?? "nil") → \(n) is below the trained range")
            XCTAssertTrue(h3.frameOptions.contains(n), "\(raw ?? "nil") → \(n) is off the 17k+5 ladder")
        }
        // LTX declares no floor, so its answers are unchanged.
        let ltx = VideoModelPreset.ltx23Q4
        for raw in [nil, "1", "4", "60"] {
            XCTAssertTrue(ltx.frameOptions.contains(MediaToolArgs.videoFrames(raw, model: ltx)))
        }
    }

    func testChatPreviewStepsStayOnTheFastTier() {
        // `videoSteps` reads the fast TIER, which still says 16 — widening the
        // slider's range must not drag the unattended default down with it.
        XCTAssertEqual(MediaChatDefaults.videoSteps(for: h3), 16)
        XCTAssertEqual(MediaChatDefaults.videoSteps(for: ref2va), 16)
    }

    // MARK: - The pane actually shows them

    func testThePaneRendersBothAdvisories() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/VideoGenView.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(source.contains("stepsAdvisory"),
                      "a reachable low step count with no warning is the other half of this bug")
        XCTAssertTrue(source.contains("framesAdvisory"),
                      "a reachable short clip needs the same sentence")
    }
}
