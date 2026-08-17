import XCTest
@testable import MLXCore

/// The custom-resolution grid is a MIRROR of the server's `clampFluxDim` /
/// `clampKreaDim` in `src/gen.zig`. Neither side can call the other, so this is
/// documented duplication in the `isMediaModelType` / `modalityFromType` mould,
/// and these tests are what keep the two from drifting: a hint that disagrees
/// with what the server actually samples is worse than no hint, because the
/// user reads a number the image never had.
final class CustomResolutionTests: XCTestCase {

    // MARK: - The grid each backend actually samples on

    /// `clampFluxDim`: multiple of 32 in [256, 1536].
    /// `clampKreaDim`: multiple of 16 in [256, 2048], shared by krea AND mage_flow.
    func testGridsMatchTheServersOwnClamps() {
        XCTAssertEqual(ImageModelPreset.flux2Klein4B_Q4.resolutionGrid,
                       ResolutionGrid(alignment: 32, minDim: 256, maxDim: 1536))
        XCTAssertEqual(ImageModelPreset.flux2Klein9B_Q4.resolutionGrid,
                       ResolutionGrid(alignment: 32, minDim: 256, maxDim: 1536))
        XCTAssertEqual(ImageModelPreset.krea2Turbo.resolutionGrid,
                       ResolutionGrid(alignment: 16, minDim: 256, maxDim: 2048))
        XCTAssertEqual(ImageModelPreset.mageFlowTurbo.resolutionGrid,
                       ResolutionGrid(alignment: 16, minDim: 256, maxDim: 2048))
        XCTAssertEqual(ImageModelPreset.mageFlowEditTurbo.resolutionGrid,
                       ResolutionGrid(alignment: 16, minDim: 256, maxDim: 2048))
    }

    /// Load-bearing invariant, not a coincidence: the range check runs on the
    /// RAW input and the snap runs after it, so an in-range value must never
    /// snap ABOVE the ceiling. That holds only while every `maxDim` is itself a
    /// multiple of its `alignment`. A future backend whose max isn't would
    /// silently hand the server an over-max number.
    func testEveryGridsMaxIsOnItsOwnAlignment() {
        for p in ImageModelPreset.all {
            let g = p.resolutionGrid
            XCTAssertEqual(g.maxDim % g.alignment, 0, "\(p.id) max \(g.maxDim) off its \(g.alignment) grid")
            XCTAssertEqual(g.minDim % g.alignment, 0, "\(p.id) min \(g.minDim) off its \(g.alignment) grid")
            XCTAssertEqual(g.snap(g.maxDim), g.maxDim, "\(p.id) snapping its own max moved it")
        }
    }

    // MARK: - Snapping direction

    /// The server rounds UP (`((v + 31) / 32) * 32`), it does not round to
    /// nearest. 481 is the case that tells them apart: nearest gives 480, the
    /// server gives 512. Rounding the friendly way here would print a hint for
    /// a resolution the server never generates.
    func testSnapRoundsUpTheWayTheServerDoes() {
        let flux = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        XCTAssertEqual(flux.snap(481), 512)
        XCTAssertEqual(flux.snap(500), 512)
        XCTAssertEqual(flux.snap(512), 512)   // already on the grid — untouched
        XCTAssertEqual(flux.snap(513), 544)
        XCTAssertEqual(flux.snap(1520), 1536)

        let krea = ImageModelPreset.krea2Turbo.resolutionGrid
        XCTAssertEqual(krea.snap(257), 272)
        XCTAssertEqual(krea.snap(512), 512)
    }

    // MARK: - resolve(): ok / corrected / invalid

    func testOnGridResolutionsPassThroughUntouched() {
        let g = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        XCTAssertEqual(g.resolve(width: 512, height: 512), .ok(width: 512, height: 512))
        XCTAssertEqual(g.resolve(width: 1024, height: 1024), .ok(width: 1024, height: 1024))
        XCTAssertEqual(g.resolve(width: 1536, height: 640), .ok(width: 1536, height: 640))
    }

    /// An off-grid value is a CORRECTION, not a refusal — the user asked for
    /// something the model can nearly do, so do it and say so.
    func testOffGridResolutionIsCorrectedAndSaysSo() {
        let g = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        guard case let .corrected(w, h, note) = g.resolve(width: 500, height: 500) else {
            return XCTFail("expected a correction")
        }
        XCTAssertEqual(w, 512)
        XCTAssertEqual(h, 512)
        XCTAssertTrue(note.contains("512"), "the note must name the size actually used: \(note)")
        XCTAssertTrue(note.contains("32"), "the note must name the step so the next guess lands: \(note)")
    }

    /// Only one axis off the grid still corrects, and leaves the other alone.
    func testCorrectionTouchesOnlyTheAxisThatNeedsIt() {
        let g = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        guard case let .corrected(w, h, _) = g.resolve(width: 1024, height: 700) else {
            return XCTFail("expected a correction")
        }
        XCTAssertEqual(w, 1024)
        XCTAssertEqual(h, 704)
    }

    /// Out of range is an ERROR, not a clamp. Silently turning 4000 into 1536
    /// is not a "slight correction" — the user would be looking at a picture a
    /// third of the size they asked for with no idea why.
    func testOutOfRangeIsRefusedRatherThanClamped() {
        let g = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        for bad in [4000, 1537, 255, 1] {
            guard case let .invalid(msg) = g.resolve(width: bad, height: 512) else {
                return XCTFail("\(bad) should be refused, not clamped")
            }
            XCTAssertTrue(msg.contains("256") && msg.contains("1536"),
                          "the refusal must state the range it enforced: \(msg)")
        }
    }

    func testZeroAndNegativeAreRefused() {
        let g = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
        for bad in [0, -1, -512] {
            guard case .invalid = g.resolve(width: bad, height: 512) else {
                return XCTFail("\(bad) should be refused")
            }
            guard case .invalid = g.resolve(width: 512, height: bad) else {
                return XCTFail("\(bad) should be refused on the height axis too")
            }
        }
    }

    /// Krea's range is wider than FLUX's, and the refusal must quote the
    /// grid it was actually given — not a hardcoded FLUX sentence.
    func testRefusalQuotesTheSelectedModelsOwnRange() {
        let krea = ImageModelPreset.krea2Turbo.resolutionGrid
        guard case let .invalid(msg) = krea.resolve(width: 2049, height: 1024) else {
            return XCTFail("2049 is over Krea's ceiling")
        }
        XCTAssertTrue(msg.contains("2048"), "expected Krea's own ceiling: \(msg)")
        // …and the same number is FINE on Krea while it would fail on FLUX.
        XCTAssertEqual(krea.resolve(width: 2048, height: 1024), .ok(width: 2048, height: 1024))
        guard case .invalid = ImageModelPreset.flux2Klein4B_Q4.resolutionGrid
            .resolve(width: 2048, height: 1024) else {
            return XCTFail("2048 is over FLUX's ceiling")
        }
    }

    // MARK: - The 512×512 preset

    /// The server has always accepted 512 on FLUX (`clampFluxDim` takes any
    /// multiple of 32 from 256), so this was a missing menu row, not a missing
    /// capability.
    func testFluxOffersA512SquarePreset() {
        for p in [ImageModelPreset.flux2Klein4B_Q4, .flux2Klein9B_Q4] {
            XCTAssertTrue(p.resolutions.contains { $0.width == 512 && $0.height == 512 },
                          "\(p.id) has no 512×512 row")
        }
    }

    // MARK: - Video grids

    /// Video is where the grid EARNS its keep: the image backends silently
    /// rewrite an off-grid size, but `handleVideo` returns a hard 400
    /// ("width and height must be multiples of 32"), so an unvalidated size is
    /// a failed generation rather than a slightly different picture.
    func testVideoGridsMatchTheServersOwnRefusals() {
        // H3 has no two-stage pipeline; /32 always.
        XCTAssertEqual(VideoModelPreset.minimaxH3.resolutionGrid(twoStage: false).alignment, 32)
        XCTAssertEqual(VideoModelPreset.minimaxH3.resolutionGrid(twoStage: true).alignment, 32)
        // LTX one-stage is /32…
        XCTAssertEqual(VideoModelPreset.ltx25Q8.resolutionGrid(twoStage: false).alignment, 32)
        // …and two-stage denoises at HALF the canvas, which is why the server
        // demands /64 there ("two-stage pipelines need width/height divisible
        // by 64"). Same model, different grid — so the grid cannot be a
        // constant on the preset.
        XCTAssertEqual(VideoModelPreset.ltx25Q8.resolutionGrid(twoStage: true).alignment, 64)
    }

    /// Every shipped video row must clear its own model's ONE-STAGE grid, or
    /// the menu offers a size the server refuses outright.
    func testEveryVideoPresetRowIsOnItsOwnGrid() {
        for p in VideoModelPreset.all {
            let g = p.resolutionGrid(twoStage: false)
            for r in p.resolutions {
                XCTAssertEqual(g.resolve(width: r.width, height: r.height),
                               .ok(width: r.width, height: r.height),
                               "\(p.id) row \(r.id) is off its own grid")
            }
        }
    }

    /// LTX's shipped rows are all /64 on purpose — every one of them has to
    /// survive being picked with a two-stage tier selected. H3's do not, which
    /// is exactly why the two backends cannot share one alignment.
    func testEveryLtxRowAlsoSurvivesTwoStageWhileH3RowsNeedNot() {
        for p in VideoModelPreset.all where p.backend != .minimaxH3 {
            let g = p.resolutionGrid(twoStage: true)
            for r in p.resolutions {
                XCTAssertEqual(g.resolve(width: r.width, height: r.height),
                               .ok(width: r.width, height: r.height),
                               "\(p.id) row \(r.id) would be refused by a two-stage pipeline")
            }
        }
        // Documents the asymmetry rather than asserting H3 is broken: 544 and
        // 672 are /32 and not /64, so applying LTX's two-stage grid to H3 would
        // refuse the model's own fastest canvas.
        let h3 = VideoModelPreset.minimaxH3
        let offBy64 = h3.resolutions.filter { $0.width % 64 != 0 || $0.height % 64 != 0 }
        XCTAssertFalse(offBy64.isEmpty, "H3 rows are all /64 — the per-pipeline grid may be unnecessary")
    }

    /// The same off-grid size is corrected differently depending on the
    /// pipeline, and the note names the step that actually applied.
    func testTwoStageCorrectsToACoarserGridThanOneStage() {
        let p = VideoModelPreset.ltx25Q8
        guard case let .corrected(w1, _, _) = p.resolutionGrid(twoStage: false)
            .resolve(width: 700, height: 448) else { return XCTFail("expected a correction") }
        XCTAssertEqual(w1, 704) // /32
        guard case let .corrected(w2, _, note) = p.resolutionGrid(twoStage: true)
            .resolve(width: 700, height: 448) else { return XCTFail("expected a correction") }
        XCTAssertEqual(w2, 704) // /64 too, as it happens
        XCTAssertTrue(note.contains("64"), "the note must name the step in force: \(note)")
        // A size that is fine one-stage and NOT fine two-stage is the case the
        // whole per-pipeline split exists for.
        XCTAssertEqual(p.resolutionGrid(twoStage: false).resolve(width: 736, height: 448),
                       .ok(width: 736, height: 448))
        guard case .corrected = p.resolutionGrid(twoStage: true)
            .resolve(width: 736, height: 448) else {
            return XCTFail("736 is /32 but not /64 — two-stage must correct it")
        }
    }

    func testVideoCustomIsOfferedLastForEveryVideoModel() {
        for p in VideoModelPreset.all {
            XCTAssertTrue(p.resolutionOptions().last?.isCustom == true, "\(p.id) does not end with Custom")
        }
    }

    /// Same sentinel hazard as the image path: the chat's `generate_video`
    /// passes the saved bucket straight through when the model names no size.
    func testVideoCustomSentinelResolvesForNonUIConsumers() {
        var s = VideoGenSettings()
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = 704
        s.customHeight = 448
        let c = s.concreteResolution(for: .ltx25Q8)
        XCTAssertFalse(c.isCustom)
        XCTAssertEqual(c.width, 704)
        XCTAssertEqual(c.height, 448)
    }

    // MARK: - The sentinel must never reach a consumer that wants a SIZE

    /// `ResolutionOption.custom` carries -1 × -1 by design — it is a menu row,
    /// not a size. The pane holds the real numbers, so every non-UI consumer
    /// has to resolve it or the sentinel rides the wire: the chat's
    /// `generate_image` passes the saved bucket straight through when the model
    /// names no size, which would have sent width -1 to the server.
    func testTheCustomSentinelResolvesToARealSizeForNonUIConsumers() {
        var s = ImageGenSettings()
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = 768
        s.customHeight = 512

        let concrete = s.concreteResolution(for: .flux2Klein4B_Q4)
        XCTAssertFalse(concrete.isCustom)
        XCTAssertEqual(concrete.width, 768)
        XCTAssertEqual(concrete.height, 512)
    }

    /// A saved custom size is snapped onto the grid on the way out, so the
    /// agent path sends the same numbers the pane would have sent.
    func testASavedOffGridCustomSizeIsSnappedForConsumers() {
        var s = ImageGenSettings()
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = 500
        s.customHeight = 500
        let concrete = s.concreteResolution(for: .flux2Klein4B_Q4)
        XCTAssertEqual(concrete.width, 512)
        XCTAssertEqual(concrete.height, 512)
    }

    /// A saved custom size that the CURRENT model cannot honor (settings are
    /// shared across models — 2048 is fine on Krea, over FLUX's ceiling) falls
    /// back to that model's default rather than being clamped silently.
    func testAnOutOfRangeSavedCustomFallsBackToTheModelsDefault() {
        var s = ImageGenSettings()
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = 2048
        s.customHeight = 2048
        let flux = s.concreteResolution(for: .flux2Klein4B_Q4)
        XCTAssertEqual(flux.id, ImageModelPreset.flux2Klein4B_Q4.defaultResolution.id)
        // The very same settings are honored on the model that can do it.
        let krea = s.concreteResolution(for: .krea2Turbo)
        XCTAssertEqual(krea.width, 2048)
    }

    /// Non-custom picks are passed through untouched — this must not become a
    /// second place that reinterprets an ordinary bucket.
    func testANormalBucketIsUnchangedByTheConsumerResolver() {
        var s = ImageGenSettings()
        s.resolutionId = "1216x832"
        let c = s.concreteResolution(for: .flux2Klein4B_Q4)
        XCTAssertEqual(c.width, 1216)
        XCTAssertEqual(c.height, 832)
    }

    /// Class guard: `MediaToolArgs.resolution` hands `saved` back verbatim when
    /// the model asked for nothing, so a sentinel handed in is a sentinel sent.
    /// Its own filters only protect the pixel/aspect paths.
    func testMediaToolArgsPassesSavedThroughSoItMustNeverBeHandedTheSentinel() {
        let passed = MediaToolArgs.resolution(nil,
                                              options: ImageModelPreset.flux2Klein4B_Q4.resolutions,
                                              saved: .custom)
        XCTAssertTrue(passed.isCustom,
                      "passthrough changed — the guard at the call site may no longer be load-bearing")
        XCTAssertLessThan(passed.width, 0, "the sentinel really does carry a negative size")
    }

    // MARK: - Persistence

    /// The sentinel carries no size, so `resolvedResolution` has to recognise
    /// it by id — matching against `resolutions` (where it deliberately isn't)
    /// silently reopens a saved custom pick on the model's default.
    func testACustomPickSurvivesASaveLoadRoundTrip() {
        var s = ImageGenSettings()
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = 768
        s.customHeight = 512

        let data = try! JSONEncoder().encode(s)
        let back = try! JSONDecoder().decode(ImageGenSettings.self, from: data)

        XCTAssertTrue(back.resolvedResolution(for: .flux2Klein4B_Q4).isCustom)
        XCTAssertEqual(back.customWidth, 768)
        XCTAssertEqual(back.customHeight, 512)
    }

    /// The hand-listed tolerant decoder drops any field you forget to add —
    /// this is the tripwire for that class, not a restatement of the round trip.
    func testSettingsBlobFromAnOlderBuildStillDecodes() {
        let legacy = #"{"modelId":"mflux/flux2-klein-4b-q4","resolutionId":"1024x1024","steps":8}"#
        let s = try! JSONDecoder().decode(ImageGenSettings.self, from: Data(legacy.utf8))
        // Absent custom keys fall back to the defaults rather than zeroing,
        // which would make the fields open on an invalid 0 × 0.
        XCTAssertEqual(s.customWidth, ImageGenSettings().customWidth)
        XCTAssertEqual(s.customHeight, ImageGenSettings().customHeight)
        XCTAssertFalse(s.resolvedResolution(for: .flux2Klein4B_Q4).isCustom)
    }

    /// Custom is offered on every image model (each has a grid), and is LAST so
    /// the fixed buckets stay the obvious pick.
    func testCustomIsTheLastOptionForEveryImageModel() {
        for p in ImageModelPreset.all {
            XCTAssertTrue(p.resolutionOptions(editMode: false).last?.isCustom == true,
                          "\(p.id) does not end with Custom")
        }
    }

    /// A stale Custom selection must survive an edit-mode flip, or the picker
    /// silently re-points to a fixed bucket while the fields still show a size.
    func testCustomSurvivesAnEditModeChange() {
        let p = ImageModelPreset.flux2Klein4B_Q4
        XCTAssertTrue(p.validResolution(.custom, editMode: true).isCustom)
        XCTAssertTrue(p.validResolution(.custom, editMode: false).isCustom)
    }

    /// Every shipped preset row must itself be on its model's grid — a menu
    /// entry the server would silently rewrite is the bug this whole feature
    /// exists to surface.
    func testEveryPresetResolutionIsOnItsOwnGrid() {
        for p in ImageModelPreset.all {
            let g = p.resolutionGrid
            for r in p.resolutions where !r.isMatchSource && !r.isCustom {
                XCTAssertEqual(g.resolve(width: r.width, height: r.height),
                               .ok(width: r.width, height: r.height),
                               "\(p.id) row \(r.id) is not on its own grid")
            }
        }
    }
}
