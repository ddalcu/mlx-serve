import XCTest
@testable import MLXCore

/// The switcher reports what the values ARE, so "do these values still mean
/// Good?" has to have an answer — and one that survives `applyQualityDefaults`
/// ending in a frame clamp.
final class VideoQualityMatchTests: XCTestCase {

    /// Canvases across the payload budget: the largest ones cap the frame
    /// ladder below what some tiers ask for, which is the case the clamp
    /// mirroring exists for.
    private let canvases = [(1344, 768), (960, 544), (768, 512), (1920, 1088)]

    /// The invariant the whole feature rests on: applying a tier must leave
    /// the switcher reading that tier, on every model, at every canvas, at
    /// every window count. Written as a sweep rather than against known
    /// numbers, because the numbers are the presets' to change.
    func testApplyingATierLeavesTheSwitcherOnIt() {
        for model in VideoModelPreset.all {
            for tier in QualityPreset.allCases {
                for (w, h) in canvases {
                    for windows in [1, 2] {
                        let applied = VideoQualityMatch.resolve(tier: tier, model: model,
                                                                width: w, height: h,
                                                                chainWindows: windows)
                        let read = VideoQualityMatch.match(applied, model: model,
                                                           width: w, height: h,
                                                           chainWindows: windows,
                                                           preferring: tier)
                        XCTAssertEqual(read, tier,
                                       "\(model.id) \(tier.label) at \(w)x\(h) x\(windows) windows")
                    }
                }
            }
        }
    }

    /// Guards the sweep above against being vacuous: if no tier's length were
    /// ever clamped, the clamp mirroring would be untested and could rot.
    func testSomeTierIsActuallyClampedSomewhere() {
        var sawClamp = false
        for model in VideoModelPreset.all {
            for tier in QualityPreset.allCases {
                for (w, h) in canvases where !sawClamp {
                    let applied = VideoQualityMatch.resolve(tier: tier, model: model,
                                                            width: w, height: h, chainWindows: 1)
                    if applied.numFrames != model.settings(tier).numFrames { sawClamp = true }
                }
            }
        }
        XCTAssertTrue(sawClamp, "no tier's length is clamped at any canvas — the sweep proves nothing")
    }

    /// One tracked value moved by one step is Custom. Steps stands in for the
    /// whole set: they are compared as one `Equatable`.
    func testMovingATrackedValueReadsAsCustom() {
        let model = VideoModelPreset.ltx23Q4
        var v = VideoQualityMatch.resolve(tier: .good, model: model,
                                          width: 768, height: 512, chainWindows: 1)
        v.steps += 1
        XCTAssertNil(VideoQualityMatch.match(v, model: model, width: 768, height: 512,
                                             chainWindows: 1, preferring: .good))
    }

    /// A tier describes a full render and always turns Turbo off, so Turbo on
    /// is a state no tier claims.
    func testTurboReadsAsCustom() {
        let model = VideoModelPreset.minimaxH3
        var v = VideoQualityMatch.resolve(tier: .good, model: model,
                                          width: 960, height: 544, chainWindows: 1)
        XCTAssertFalse(v.turbo, "a tier never asks for Turbo")
        v.turbo = true
        XCTAssertNil(VideoQualityMatch.match(v, model: model, width: 960, height: 544,
                                             chainWindows: 1, preferring: .good))
    }

    /// `preferring` settles a genuine ambiguity — it must never pull the
    /// highlight off a tier the values really do match.
    func testPreferringNeverOverridesARealMatch() {
        let model = VideoModelPreset.ltx23Q4
        let v = VideoQualityMatch.resolve(tier: .quality, model: model,
                                          width: 768, height: 512, chainWindows: 1)
        XCTAssertEqual(VideoQualityMatch.match(v, model: model, width: 768, height: 512,
                                               chainWindows: 1, preferring: .fast),
                       .quality)
    }

    /// Why `preferring` exists: LTX's Fast and Good differ in nothing but
    /// length, so a canvas whose ladder caps below both makes them the same
    /// state and the answer is the user's last click.
    func testLtxFastAndGoodDifferOnlyInLength() {
        let model = VideoModelPreset.ltx23Q4
        let fast = VideoQualityMatch.resolve(tier: .fast, model: model,
                                            width: 768, height: 512, chainWindows: 1)
        let good = VideoQualityMatch.resolve(tier: .good, model: model,
                                            width: 768, height: 512, chainWindows: 1)
        XCTAssertNotEqual(fast, good, "at a canvas that holds both, the two tiers are distinct")
        var fastAtGoodLength = fast
        fastAtGoodLength.numFrames = good.numFrames
        XCTAssertEqual(fastAtGoodLength, good, "length is the ONLY difference between Fast and Good")
    }

    /// Both tiers collapsing onto one state is resolved, not left to order.
    func testAnAmbiguousStateHonoursTheLastPick() {
        let model = VideoModelPreset.ltx23Q4
        var v = VideoQualityMatch.resolve(tier: .good, model: model,
                                          width: 768, height: 512, chainWindows: 1)
        v.numFrames = VideoQualityMatch.resolve(tier: .fast, model: model,
                                                width: 768, height: 512,
                                                chainWindows: 1).numFrames
        // Now the state IS Fast. Preferring Good must not claim it.
        XCTAssertEqual(VideoQualityMatch.match(v, model: model, width: 768, height: 512,
                                               chainWindows: 1, preferring: .good),
                       .fast)
    }
}
