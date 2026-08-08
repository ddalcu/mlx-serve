import XCTest
@testable import MLXCore

/// MiniMax-H3's memory and time models.
///
/// Both are derived from the engine's own row arithmetic and CALIBRATED against
/// measurements taken on real generations — the calibration cases below are the
/// point of the file. A model that cannot reproduce the numbers we measured is
/// not a conservative estimate, it is a different model.
final class H3PlanningTests: XCTestCase {

    private let h3 = VideoModelPreset.minimaxH3
    private let h3q4 = VideoModelPreset.minimaxH3Q4

    // MARK: - Row arithmetic (the engine's own, mirrored)

    /// `videoLatentT` in `minimax_h3.zig`: 17 source frames fold to 5 latent
    /// ones. This is the whole reason the frame ladder is 17k+5 and not LTX's
    /// 8N+1, so getting it wrong mis-sizes every downstream number.
    func testLatentFrameCountMatchesTheEnginesLadder() {
        XCTAssertEqual(H3Plan.latentT(frames: 5), 2)
        XCTAssertEqual(H3Plan.latentT(frames: 22), 7)
        XCTAssertEqual(H3Plan.latentT(frames: 124), 37)
        XCTAssertEqual(H3Plan.latentT(frames: 209), 62)
        XCTAssertEqual(H3Plan.latentT(frames: 362), 107)
        // Below the ladder floor the engine still allocates 2.
        XCTAssertEqual(H3Plan.latentT(frames: 1), 2)
    }

    /// The packed sequence is `[text | cond | audio | video]`. Audio is STEREO —
    /// two rows per latent audio step — which is easy to drop and worth ~0.4 GB
    /// of cache at 768p.
    func testSequenceRowsCountEveryStreamInThePackedSequence() {
        let rows = H3Plan.rows(width: 1344, height: 768, frames: 124, promptTokens: 250)
        // video: 37 latent frames x (1344/32 x 768/32 = 1008) = 37,296
        // audio: round(124 x 5/3) = 207 latents x 2 channels = 414
        XCTAssertEqual(rows, 37_296 + 414 + 250)

        // A keyframe adds a whole conditioning frame's worth of rows.
        let withKey = H3Plan.rows(width: 1344, height: 768, frames: 124, promptTokens: 250, keyframes: 1)
        XCTAssertEqual(withKey - rows, 1008)

        // Aspect matters, not just pixel count: 960x544 is 30x17 patches.
        XCTAssertEqual(H3Plan.rows(width: 960, height: 544, frames: 362, promptTokens: 250),
                       107 * 510 + 603 * 2 + 250)
    }

    // MARK: - Memory (CALIBRATION — these are measured numbers)

    /// The fast recipe caches one `[S, hidden]` bf16 per block for the whole
    /// run, and at long frame counts that cache — not the weights — is what
    /// runs a Mac out of memory. Measured on the 8-bit pack at 1344x768:
    /// ~20 GB at 124 frames, ~34 GB at 209.
    func testPabCacheReproducesTheMeasuredFootprints() {
        // Decimal GB, because that is the unit those two figures were written
        // down in. Everything that compares against a MACHINE's memory uses
        // GiB (`RAMChecker.totalGB` is `physicalMemory / 1024^3`), and mixing
        // the two silently is a 7% error at this scale.
        let gb = 1_000_000_000.0
        let at124 = Double(H3Plan.pabCacheBytes(rows: H3Plan.rows(width: 1344, height: 768, frames: 124, promptTokens: 250))) / gb
        let at209 = Double(H3Plan.pabCacheBytes(rows: H3Plan.rows(width: 1344, height: 768, frames: 209, promptTokens: 250))) / gb
        XCTAssertEqual(at124, 20.4, accuracy: 0.5, "measured ~20 GB at 768p/124f")
        XCTAssertEqual(at209, 34.1, accuracy: 0.5, "measured ~34 GB at 768p/209f")
        // It is LINEAR in rows — the square term is time, not memory (SDPA is
        // fused, so the [S, S] score matrix is never materialized; at 362
        // frames it would be 1.3 PB).
        let a = H3Plan.pabCacheBytes(rows: 10_000)
        XCTAssertEqual(H3Plan.pabCacheBytes(rows: 20_000), a * 2)
    }

    /// Turning the fast recipe OFF removes the cache entirely (`bcast_k = 1`
    /// never allocates it). That makes "Max quality" also the LOW-MEMORY mode,
    /// which is backwards from every other quality toggle in the app and is
    /// exactly the trade a user out of memory needs offered to them.
    func testMaxQualityIsAlsoTheLowMemoryMode() {
        let long = (w: 1344, h: 768, f: 362)
        let fast = H3Plan.peakBytes(model: h3, width: long.w, height: long.h, frames: long.f, fast: true)
        let slow = H3Plan.peakBytes(model: h3, width: long.w, height: long.h, frames: long.f, fast: false)
        XCTAssertLessThan(slow, fast)
        let gib = 1024.0 * 1024.0 * 1024.0
        XCTAssertEqual(Double(fast) / gib, 85.0, accuracy: 4.0)
        // Without the cache the same run bottoms out at the staged LOAD peak —
        // the weights still have to get in — so the trade is ~85 GiB for ~38,
        // paid for in roughly 4x the runtime.
        XCTAssertEqual(Double(slow) / gib, h3.stagedPeakGB, accuracy: 1.0)
    }

    /// The three answers that decide what the pane may offer on a given Mac.
    func testLongestClipFitsOnlyTheCanvasWeMeasuredItOn() {
        let gib = 1024.0 * 1024.0 * 1024.0
        let wide = Double(H3Plan.peakBytes(model: h3, width: 1344, height: 768, frames: 362, fast: true)) / gib
        let longForm = Double(H3Plan.peakBytes(model: h3, width: 960, height: 544, frames: 362, fast: true)) / gib
        // 362 frames at 960x544 is the run we actually did, on an M5 Max.
        XCTAssertLessThan(longForm, 64.0, "the measured 362-frame run must read as fitting")
        XCTAssertGreaterThan(wide, 64.0, "1344x768 at 362 frames is a 128 GB job")
        XCTAssertLessThan(wide, 102.0, "...but it IS a 128 GB job, not an impossible one")

        // The 4-bit pack is the low-RAM option and must bill lower everywhere.
        for f in [124, 209, 362] {
            XCTAssertLessThan(H3Plan.peakBytes(model: h3q4, width: 1344, height: 768, frames: f, fast: true),
                              H3Plan.peakBytes(model: h3, width: 1344, height: 768, frames: f, fast: true))
        }
    }

    /// A load that never samples still needs its staged weight peak, so the
    /// floor of the estimate is the pack's measured `max(TE, DiT) + VAEs` —
    /// not `approxRAMGB`, which is that number rounded up for a coarser gate.
    func testPeakNeverFallsBelowTheStagedLoadPeak() {
        let tiny = H3Plan.peakBytes(model: h3, width: 768, height: 768, frames: 124, fast: true)
        XCTAssertGreaterThanOrEqual(Double(tiny) / (1024 * 1024 * 1024), h3.stagedPeakGB - 0.01)
        // The declared staged peak has to be the pack's real arithmetic: the
        // 8-bit files are TE 26.28 + DiT 32.83 + VAEs 4.85 + 0.56 GiB, and the
        // encoder is freed before the DiT loads.
        XCTAssertEqual(h3.stagedPeakGB, max(26.28, 32.83) + 4.85 + 0.56, accuracy: 0.05)
        XCTAssertEqual(h3q4.stagedPeakGB, max(14.72, 17.41) + 4.85 + 0.56, accuracy: 0.05)
    }

    // MARK: - The frame cap the pane offers

    /// REGRESSION: `safeFrameCap` was LTX's VAE-staging formula (12 GB per
    /// megapixel per 100 frames) applied to every backend. On H3 it computed
    /// 677 frames on a 128 GB Mac — so the warning never fired — and 32 on a
    /// 48 GB one, below H3's own 124-frame floor, so it fired always. Neither
    /// reading is information.
    func testFrameCapIsComputedFromH3sOwnMemoryModel() {
        // A 128 GB Mac gets the whole ladder at both canvases.
        XCTAssertEqual(RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 128), 362)
        XCTAssertEqual(RAMChecker.safeFrameCap(model: h3, width: 1344, height: 768, available: 128), 362)
        // A 64 GB Mac does not — and the widescreen canvas runs out first,
        // which is the whole reason 960x544 is offered as the long-form one.
        let at64wide = RAMChecker.safeFrameCap(model: h3, width: 1344, height: 768, available: 64)
        let at64long = RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 64)
        XCTAssertLessThan(at64wide, 362)
        XCTAssertGreaterThan(at64long, at64wide)
        // A 48 GB Mac can still do real work on the 4-bit pack — the answer
        // must be a usable clip length, not a number below the model's floor.
        let cap48 = RAMChecker.safeFrameCap(model: h3q4, width: 960, height: 544, available: 48)
        XCTAssertGreaterThanOrEqual(cap48, 124)
        // Every answer lands ON the ladder, or the slider snaps away from it.
        for (w, h, ram) in [(960, 544, 128), (1344, 768, 96), (960, 544, 48), (768, 768, 64)] {
            let cap = RAMChecker.safeFrameCap(model: h3, width: w, height: h, available: ram)
            XCTAssertTrue(h3.frameOptions.contains(cap), "\(cap) frames is off the 17k+5 ladder")
        }
        // More memory is never fewer frames; a bigger canvas is never more.
        XCTAssertGreaterThanOrEqual(RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 96),
                                    RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 64))
        XCTAssertGreaterThanOrEqual(RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 64),
                                    RAMChecker.safeFrameCap(model: h3, width: 1344, height: 768, available: 64))
    }

    /// `frameCap` has a FLOOR, so its answer is ambiguous at the bottom: 124
    /// means both "124 frames fits" and "nothing fits". A 24 GB Mac cannot load
    /// the 8-bit pack at all, and the pane read that as "124 frames is fine".
    func testFitsAnswersTheQuestionTheCapCannotAtTheFloor() {
        let floor = h3.frameOptions.first!
        XCTAssertEqual(RAMChecker.safeFrameCap(model: h3, width: 960, height: 544, available: 24), floor)
        XCTAssertFalse(H3Plan.fits(model: h3, width: 960, height: 544, frames: floor, fast: true, availableGB: 24))
        // ...while the same floor on a machine that can hold it does fit.
        XCTAssertTrue(H3Plan.fits(model: h3, width: 960, height: 544, frames: floor, fast: true, availableGB: 64))
    }

    /// The 4-bit pack is sold as the 32 GB option, so the model must not round
    /// it out of reach: `approxRAMGB` is a rounded-up 26 where the pack's real
    /// staged peak is 22.8, and the difference is exactly one 32 GB Mac.
    func testFourBitPackFitsTheMacIttIsSoldFor() {
        XCTAssertGreaterThan(h3q4.stagedPeakGB, 0, "the measured staged peak must be declared")
        XCTAssertLessThan(h3q4.stagedPeakGB, Double(h3q4.approxRAMGB), "approxRAMGB is the rounded-up one")
        XCTAssertTrue(H3Plan.fits(model: h3q4, width: 960, height: 544, frames: 124, fast: true, availableGB: 32))
        // The 8-bit pack genuinely does not fit there, and must still say so.
        XCTAssertFalse(H3Plan.fits(model: h3, width: 960, height: 544, frames: 124, fast: true, availableGB: 32))
    }

    /// LTX keeps its own formula, byte for byte — the H3 model is not a general
    /// video memory model and must not be applied to a backend it was not
    /// derived from (which is the bug being fixed, in the other direction).
    func testLtxKeepsItsOwnFrameCapFormula() {
        let ltx = VideoModelPreset.ltx23Q4
        XCTAssertEqual(RAMChecker.safeFrameCap(model: ltx, width: 704, height: 480, available: 128), ltx.maxFrames)
        // Reproduce the original arithmetic here rather than trusting the
        // result: a refactor that "improved" it would silently change the
        // warning threshold on every LTX run.
        for (w, h, ram) in [(704, 480, 32), (768, 512, 24), (704, 480, 16)] {
            let pixelMP = Double(w * h) / 1_000_000.0
            let headroom = max(0, ram - ltx.approxRAMGB)
            let expected = min(ltx.maxFrames, max(9, Int((Double(headroom) / max(2.0, pixelMP * 12.0)) * 100)))
            XCTAssertEqual(RAMChecker.safeFrameCap(model: ltx, width: w, height: h, available: ram), expected)
        }
    }

    // MARK: - Reference limits

    /// MiniMax caps ref2va at 12 FILES across all three lists. The per-type
    /// caps sum to 15, so the pane happily built a 15-file set that the server
    /// now refuses — a picker that lets you assemble a request only to fail at
    /// generate time is a worse version of the same 400.
    func testReferenceListsStopAtTheCombinedTotalNotJustTheirOwnCap() {
        XCTAssertEqual(H3RefLimits.total, 12)
        XCTAssertLessThan(H3RefLimits.total, H3RefLimits.images + H3RefLimits.videos + H3RefLimits.audios)

        // Nothing attached: each list offers its own cap.
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.images, current: 0, totalAttached: 0), 9)
        // 9 images + 3 clips = 12 files: audio cannot be added even though its
        // own list is empty. This is the case the per-type caps cannot see.
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.audios, current: 0, totalAttached: 12), 0)
        // 8 images + 3 clips = 11: exactly one more file of any type.
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.audios, current: 0, totalAttached: 11), 1)
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.images, current: 8, totalAttached: 11), 1)
        // A type's own cap still binds first when it is the tighter one.
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.videos, current: 3, totalAttached: 5), 0)
        // Never negative — a set attached before the cap existed must not make
        // the picker do arithmetic on a negative count.
        XCTAssertEqual(H3RefLimits.remaining(perType: H3RefLimits.images, current: 9, totalAttached: 15), 0)
    }

    /// The user needs to be told WHY the add button vanished from an empty list.
    func testTheCombinedLimitExplainsItselfOnlyWhenItBinds() {
        XCTAssertNil(H3RefLimits.totalNote(attached: 4))
        let atCap = H3RefLimits.totalNote(attached: 12)
        XCTAssertNotNil(atCap)
        XCTAssertTrue(atCap!.contains("12"), atCap!)
    }

    // MARK: - Time

    /// Anchored on the M4 Max acceptance runs. The estimate is a range, and the
    /// point estimate has to land near what we measured or it is worse than no
    /// number at all.
    func testTimeEstimateReproducesTheMeasuredRuns() {
        let m4max = H3Hardware(gpuCores: 40, label: "M4 Max")
        // 1344x768, 124 frames, 30 steps, fast recipe on: measured 49 minutes.
        let a = H3TimeEstimate.seconds(model: h3, width: 1344, height: 768, frames: 124, steps: 30, fast: true, hardware: m4max)
        XCTAssertEqual(a / 60.0, 49.0, accuracy: 10.0)
        // Same canvas, 209 frames: measured 1 h 57 m.
        let b = H3TimeEstimate.seconds(model: h3, width: 1344, height: 768, frames: 209, steps: 30, fast: true, hardware: m4max)
        XCTAssertEqual(b / 60.0, 117.0, accuracy: 25.0)
        // Fast recipe off at the capstone geometry: measured 2 h 19 m.
        let c = H3TimeEstimate.seconds(model: h3, width: 1344, height: 768, frames: 124, steps: 30, fast: false, hardware: m4max)
        XCTAssertEqual(c / 60.0, 139.0, accuracy: 30.0)
        XCTAssertGreaterThan(c, a, "the fast recipe must never be estimated slower")
    }

    /// Cost is super-linear in rows: at a matched 124 frames, 1344x768 measured
    /// 2.9x the time of 960x544 while costing 1.98x the pixels. A model that is
    /// merely linear in pixels under-promises the wide canvas by ~50%.
    func testWideCanvasCostsItsMeasuredRatioNotItsPixelRatio() {
        let m5max = H3Hardware(gpuCores: 40, label: "M5 Max")
        let wide = H3TimeEstimate.seconds(model: h3, width: 1344, height: 768, frames: 124, steps: 30, fast: true, hardware: m5max)
        let small = H3TimeEstimate.seconds(model: h3, width: 960, height: 544, frames: 124, steps: 30, fast: true, hardware: m5max)
        XCTAssertEqual(wide / small, 2.9, accuracy: 0.45)
    }

    /// Steps scale the SAMPLING, not the fixed stages (weight load, text
    /// encode, VAE decode), so halving them must not halve the estimate.
    func testFixedStagesSurviveAStepChange() {
        let hw = H3Hardware(gpuCores: 40, label: "M4 Max")
        let at30 = H3TimeEstimate.seconds(model: h3, width: 768, height: 768, frames: 124, steps: 30, fast: true, hardware: hw)
        let at15 = H3TimeEstimate.seconds(model: h3, width: 768, height: 768, frames: 124, steps: 15, fast: true, hardware: hw)
        XCTAssertGreaterThan(at15, at30 * 0.5)
        XCTAssertLessThan(at15, at30)
    }

    /// A faster GPU is a faster estimate, and the scan must produce something
    /// usable on hardware we have never seen.
    func testHardwareScanScalesTheAnchorAndNeverReturnsZero() {
        let base = H3Hardware(gpuCores: 40, label: "M4 Max")
        let half = H3Hardware(gpuCores: 20, label: "M4 Pro")
        let big = H3Hardware(gpuCores: 80, label: "M3 Ultra")
        let args = (w: 1344, h: 768, f: 124, s: 30)
        let t = { (hw: H3Hardware) in
            H3TimeEstimate.seconds(model: self.h3, width: args.w, height: args.h, frames: args.f, steps: args.s, fast: true, hardware: hw)
        }
        XCTAssertGreaterThan(t(half), t(base))
        XCTAssertLessThan(t(big), t(base))
        // A machine that reports nothing usable still gets the anchor, never 0
        // and never an infinity that renders as "∞ min".
        let unknown = H3Hardware(gpuCores: 0, label: "")
        XCTAssertGreaterThan(t(unknown), 0)
        XCTAssertTrue(t(unknown).isFinite)
        // The live scan on this machine must produce a plausible Mac.
        XCTAssertGreaterThan(H3Hardware.current.gpuCores, 0)
    }

    /// The step clock turns SSE progress events into laps. Events are not
    /// guaranteed to be one-per-step-in-order — a stage change repeats a step
    /// number, and the load/encode stages report before sampling begins.
    func testStepClockOnlyLapsOnRealForwardProgress() {
        var clock = H3StepClock()
        clock.observe(step: 0, at: 0)
        clock.observe(step: 0, at: 5)      // same step, a stage label changed
        XCTAssertTrue(clock.durations.isEmpty)
        clock.observe(step: 1, at: 100)    // step 0 took 100 s (graph build)
        clock.observe(step: 2, at: 160)
        clock.observe(step: 3, at: 220)
        XCTAssertEqual(clock.durations, [100, 60, 60])
        // A step number going BACKWARDS is a new generation's events arriving
        // on an old clock; it must not produce a negative lap.
        clock.observe(step: 1, at: 300)
        XCTAssertEqual(clock.durations, [100, 60, 60])
        // Two laps past the discarded first one is enough to speak.
        XCTAssertEqual(clock.eta(totalSteps: 30)!, 60 * 26, accuracy: 60)
    }

    /// Under the fast recipe most steps cost NOTHING — the engine reuses a
    /// cached velocity (measured 128 of 200 in a live run) — so the lap
    /// distribution is bimodal and a MEDIAN lands on one spike or the other
    /// with nothing in between. Both readings are wrong: while 2 of every 3
    /// steps are cached the bar sat at 75% saying "about 0 sec left" with
    /// minutes to go, and where the cadence evens out to 1:1 the same estimate
    /// flips to a full-cost lap and doubles what is left. What remains is
    /// (real + skipped) steps at the cadence observed so far, which is the
    /// AMORTIZED lap.
    func testFastRecipeCachedStepsNeitherVanishNorDoubleTheEstimate() {
        // The engine's own `sc_consec < 2` cap: 2 cached steps per real one.
        var twoOfThree: [Double] = [90]                      // step 0: graph build, discarded
        for i in 0..<15 { twoOfThree.append(i % 3 == 0 ? 12 : 0.02) }
        // 14 steps left, 5 of them real work at 12 s.
        XCTAssertEqual(H3TimeEstimate.liveEta(stepDurations: twoOfThree, totalSteps: 30)!,
                       56, accuracy: 12)

        // Late in the schedule the cadence evens out and the median sits on the
        // knife edge — the same run must not suddenly claim twice the work.
        var alternating: [Double] = [90]
        for i in 0..<17 { alternating.append(i % 2 == 0 ? 12 : 0.02) }
        // 12 steps left, half of them free — 76 s, not the 144 s a full-cost
        // lap would claim.
        XCTAssertEqual(H3TimeEstimate.liveEta(stepDurations: alternating, totalSteps: 30)!,
                       76, accuracy: 16)
    }

    /// A sub-second estimate rounds to zero, and "about 0 sec left" on a job
    /// with work still to do reads as a stuck progress bar.
    func testASubSecondEstimateNeverRendersAsZero() {
        XCTAssertEqual(H3TimeEstimate.duration(0.3), "about 1 sec")
        XCTAssertEqual(H3TimeEstimate.duration(0), "unknown")
    }

    /// After a real run the estimate should stop being someone else's
    /// measurement. One scalar is fitted — the machine factor — because the
    /// SHAPE of the curve is already known and a per-geometry table would need
    /// a run at every geometry before it could say anything.
    func testOwnHistoryCalibratesTheAnchorWithOneRun() {
        var history = H3RunHistory(records: [])
        XCTAssertNil(history.speedFactor)

        // This Mac took twice as long as the anchor predicts.
        let predicted = H3TimeEstimate.seconds(model: h3, width: 960, height: 544, frames: 124,
                                               steps: 30, fast: true, hardware: .anchor)
        history.record(predictedOnAnchor: predicted, measured: predicted * 2)
        XCTAssertEqual(history.speedFactor!, 2.0, accuracy: 0.01)
        XCTAssertEqual(history.runs, 1)

        // A different geometry inherits the factor — that is the point of
        // fitting the machine rather than the run.
        let other = H3TimeEstimate.seconds(model: h3, width: 1344, height: 768, frames: 209,
                                           steps: 30, fast: true, hardware: .anchor)
        XCTAssertEqual(history.apply(toAnchorSeconds: other), other * 2, accuracy: 1)

        // One wild outlier (a thermally-throttled run, or a run that shared the
        // GPU with a chat model) must not own the estimate: the fit is a MEDIAN.
        history.record(predictedOnAnchor: predicted, measured: predicted * 2)
        history.record(predictedOnAnchor: predicted, measured: predicted * 40)
        XCTAssertEqual(history.speedFactor!, 2.0, accuracy: 0.01)

        // Old records fall off, or a machine never re-learns after an upgrade.
        for _ in 0..<40 { history.record(predictedOnAnchor: predicted, measured: predicted) }
        XCTAssertLessThanOrEqual(history.runs, H3RunHistory.keptRuns)
        XCTAssertEqual(history.speedFactor!, 1.0, accuracy: 0.01)
    }

    /// The estimate a user reads is a RANGE with named provenance — a bare
    /// point estimate on a 3-hour job reads as a promise.
    func testEstimateTextSaysHowLongAndWhereTheNumberCameFrom() {
        let hw = H3Hardware(gpuCores: 40, label: "M4 Max")
        let short = H3TimeEstimate.describe(seconds: 90, source: .hardwareModel(hw))
        XCTAssertTrue(short.contains("min") || short.contains("sec"), short)
        let long = H3TimeEstimate.describe(seconds: 3 * 3600, source: .hardwareModel(hw))
        XCTAssertTrue(long.contains("h"), long)
        XCTAssertTrue(long.lowercased().contains("estimated"), long)
        // Measured-on-this-Mac must say so — it is a different kind of claim.
        let measured = H3TimeEstimate.describe(seconds: 600, source: .ownHistory(runs: 3))
        XCTAssertTrue(measured.lowercased().contains("your"), measured)
    }

    /// Live ETA from step cadence. Step 0 carries graph build + Metal JIT and
    /// would inflate the whole run; and under the fast recipe a step is only
    /// comparable to the steps that share its refresh state, so the estimate
    /// takes the trailing MEDIAN rather than the last step.
    func testLiveEtaIgnoresTheFirstStepAndResistsAnOutlier() {
        // 30-step run: step 0 is 300 s of JIT, the rest are ~60 s.
        var laps: [Double] = [300]
        laps.append(contentsOf: Array(repeating: 60.0, count: 9))
        let eta = H3TimeEstimate.liveEta(stepDurations: laps, totalSteps: 30)
        XCTAssertEqual(eta!, 20 * 60, accuracy: 90)
        // A single cached step (~0.02 s under velocity caching) must not make
        // the whole remaining run look instant.
        var mixed = laps
        mixed.append(0.02)
        let eta2 = H3TimeEstimate.liveEta(stepDurations: mixed, totalSteps: 30)
        XCTAssertGreaterThan(eta2!, 10 * 60)
        // Too little data to say anything: say nothing.
        XCTAssertNil(H3TimeEstimate.liveEta(stepDurations: [300], totalSteps: 30))
        XCTAssertNil(H3TimeEstimate.liveEta(stepDurations: [], totalSteps: 30))
        // Finished run → no time remaining, not a negative number.
        XCTAssertEqual(H3TimeEstimate.liveEta(stepDurations: Array(repeating: 60.0, count: 31), totalSteps: 30) ?? -1, 0)
    }
}

// MARK: - Turbo LoRA acquisition

extension H3PlanningTests {

    /// The adapter ships INSIDE our packs now, so a fresh download brings it.
    /// Everyone who already has the pack does not — and they are exactly the
    /// people who will flip Turbo on. The decision of what to do about that is
    /// pure so it can be pinned without a network or a 20 GB checkout.
    func testTurboLoraFetchDecision() {
        // The ordinary case that motivated the feature: pack on disk from
        // before the adapter shipped, user flips Turbo → fetch it.
        XCTAssertEqual(
            TurboLoraFetch.decide(turboRequested: true, backendSupportsTurbo: true,
                                  isRemote: false, fileOnDisk: false),
            .fetch)
        // Already there: never re-fetch 744 MB.
        XCTAssertEqual(
            TurboLoraFetch.decide(turboRequested: true, backendSupportsTurbo: true,
                                  isRemote: false, fileOnDisk: true),
            .ready)
        // Turbo off: the adapter is not needed, so nothing is downloaded on a
        // toggle the user did not flip.
        XCTAssertEqual(
            TurboLoraFetch.decide(turboRequested: false, backendSupportsTurbo: true,
                                  isRemote: false, fileOnDisk: false),
            .notNeeded)
        // A preset that cannot use it (REF2VA today, and every LTX preset)
        // must never trigger a download — turbo state survives preset switches.
        XCTAssertEqual(
            TurboLoraFetch.decide(turboRequested: true, backendSupportsTurbo: false,
                                  isRemote: false, fileOnDisk: false),
            .notNeeded)
        // A LAN model lives on someone else's Mac: we cannot put a file in
        // their pack, and downloading it into OURS would fix nothing. The
        // server there answers its own named 400.
        XCTAssertEqual(
            TurboLoraFetch.decide(turboRequested: true, backendSupportsTurbo: true,
                                  isRemote: true, fileOnDisk: false),
            .unavailableRemotely)
    }

    func testTurboLoraFileNameMatchesWhatTheServerLooksFor() {
        // The server resolves `<pack dir>/turbo_lora.safetensors` and the
        // bundle allowlists the same name; a rename on either side is a
        // download that lands where nothing reads it.
        XCTAssertEqual(TurboLoraFetch.fileName, "turbo_lora.safetensors")
        XCTAssertTrue(MediaBundle.minimaxH3(repo: "r", displayName: "d")
            .components[0].selection.keepSafetensors?.contains(TurboLoraFetch.fileName) ?? false)
        // NOT a ready marker: a pack downloaded before the adapter shipped
        // must keep reading as complete, or the pane offers a re-download of
        // the whole 69 GB.
        XCTAssertFalse(MediaBundle.minimaxH3(repo: "r", displayName: "d")
            .components[0].readyMarkers.contains(TurboLoraFetch.fileName))
    }
}

extension H3PlanningTests {

    /// The on-demand fetch asks for ONE file out of a 69 GB repo. If the
    /// allowlist let the transformer through, "download the missing adapter"
    /// would re-pull the whole pack — the exact outcome this feature exists to
    /// avoid for people who already have it.
    func testTurboLoraSelectionPicksTheAdapterAndNoOtherWeights() {
        let entries: [[String: Any]] = [
            ["path": "config.json", "type": "file", "size": 900],
            ["path": "tokenizer.json", "type": "file", "size": 19_000_000],
            ["path": "transformer.safetensors", "type": "file", "size": 20_000_000_000],
            ["path": "text_encoder.safetensors", "type": "file", "size": 15_000_000_000],
            ["path": "video_vae.safetensors", "type": "file", "size": 5_000_000_000],
            ["path": "audio_vae.safetensors", "type": "file", "size": 600_000_000],
            ["path": TurboLoraFetch.fileName, "type": "file", "size": 744_000_000],
        ]
        let picked = DownloadManager.selectNeededFiles(
            from: entries,
            selection: FileSelection(keepSafetensors: [TurboLoraFetch.fileName]))
        let paths = Set(picked.map { $0.0 })

        XCTAssertTrue(paths.contains(TurboLoraFetch.fileName))
        for heavy in ["transformer.safetensors", "text_encoder.safetensors",
                      "video_vae.safetensors", "audio_vae.safetensors"] {
            XCTAssertFalse(paths.contains(heavy), "\(heavy) must not ride along with the adapter")
        }
        // Whatever small json it also takes is already on disk and gets
        // size-skipped, so the transfer is the adapter alone.
        let bytes = picked.reduce(Int64(0)) { $0 + $1.1 }
        XCTAssertLessThan(bytes, 1_000_000_000)
    }
}

extension H3PlanningTests {

    /// The adapter's NAME is a contract between three places that cannot see
    /// each other: the server resolves `<pack>/turbo_lora.safetensors`, the
    /// bundle allowlists it for new installs, and the on-demand fetch asks HF
    /// for it. A rename in one of them downloads a file nothing loads, or
    /// looks for one nothing downloads — with no error, just Turbo quietly
    /// 400ing forever. Scanning the Zig source is the only way to hold the
    /// two languages together.
    func testTurboLoraFileNameIsPinnedAgainstTheServer() throws {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .deletingLastPathComponent()  // repo root
        let genZig = repo.appendingPathComponent("src/gen.zig")
        // NOT a skip on failure: a skip reads as a pass, and the whole point
        // is to notice a rename nobody would otherwise see.
        let source = try String(contentsOf: genZig, encoding: .utf8)
        XCTAssertTrue(source.contains(TurboLoraFetch.fileName),
                      "src/gen.zig no longer spells \(TurboLoraFetch.fileName) — the app would fetch a file the server never looks for")
    }
}
