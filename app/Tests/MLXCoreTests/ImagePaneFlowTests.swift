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

    // MARK: - What a source image is FOR (the verb picker)

    func testEnlargeIsOfferedOnEveryImageModel() {
        // Enlarge runs a DIFFERENT model family (SeedVR2), so unlike Edit and
        // Variation it is not a capability of the image preset at all. A
        // capability check that treated it as one would hide it exactly where
        // it is most useful — on the txt2img-only models, whose output is the
        // most likely thing to want bigger.
        for p in ImageModelPreset.all {
            XCTAssertTrue(ImageSourceVerb.available(for: p).contains(.enlarge), p.id)
        }
    }

    func testATxt2ImgOnlyModelOffersEnlargeAndNothingElse() {
        // Mage-Flow Turbo has neither instruction editing nor a VAE encoder
        // for renoise variations, so before Enlarge existed a source image
        // attached here was a DEAD state: the pane drew the thumbnail, offered
        // no mode, and Generate sent `image` without `mode:"edit"` — which the
        // server 400s by name. One verb is now the honest answer.
        let turbo = ImageModelPreset.mageFlowTurbo
        XCTAssertFalse(turbo.supportsReferenceEdit)
        XCTAssertFalse(turbo.supportsImg2Img)
        XCTAssertEqual(ImageSourceVerb.available(for: turbo), [.enlarge])
    }

    func testAnEditorWithNoVariationPathNeverOffersVariation() {
        // The old `effectiveEditMode` rule, now expressed as availability:
        // where editing is the only thing the BACKEND can do with a source, a
        // stale persisted "variation" must not send a request it rejects.
        let editor = ImageModelPreset.mageFlowEditTurbo
        XCTAssertEqual(ImageSourceVerb.available(for: editor), [.edit, .enlarge])
        XCTAssertEqual(ImageSourceVerb.resolve(.variation, for: editor), .edit)
    }

    func testAModelSwitchKeepsTheVerbWhenItStillApplies() {
        let flux = ImageModelPreset.flux2Klein4B_Q4
        XCTAssertEqual(ImageSourceVerb.available(for: flux), [.edit, .variation, .enlarge])
        // Enlarge is available everywhere, so it can never be taken away by a
        // model switch — picking a photo to enlarge and then changing the
        // image model must not silently turn it into an edit.
        for p in ImageModelPreset.all {
            XCTAssertEqual(ImageSourceVerb.resolve(.enlarge, for: p), .enlarge, p.id)
        }
        XCTAssertEqual(ImageSourceVerb.resolve(.variation, for: flux), .variation)
    }

    // MARK: - One preview, two services

    func testSwitchingVerbsNeverBlanksAResultThatIsOnScreen() {
        // THE AMNESIA THIS FIXES: the preview used to belong to whichever pane
        // was mounted, so looking at a generated image and then setting up an
        // enlarge threw the image away. The resolver takes NO verb for exactly
        // that reason — what is on screen is a property of what has been made
        // and what is selected, not of what the controls are set to.
        let s = ImagePanePreview.resolve(
            generate: .done("/out/apple.png"), enlarge: .idle,
            selected: MediaSessionItem(path: "/out/apple.png", origin: .generated))
        XCTAssertEqual(s, .result(.generated, "/out/apple.png"))
    }

    func testPickingAnOlderPictureOutOfTheStripBringsItBack() {
        // The strip is the pane's memory. Selecting a row two runs ago has to
        // beat "the newest thing that finished", or the strip is decoration.
        XCTAssertEqual(
            ImagePanePreview.resolve(
                generate: .done("/out/newest.png"), enlarge: .idle,
                selected: MediaSessionItem(path: "/out/older.png", origin: .generated)),
            .result(.generated, "/out/older.png"))
    }

    func testTheNewerResultTakesThePreview() {
        // Focus moves when a run finishes, so an enlarge of a generated image
        // replaces it rather than hiding behind it.
        XCTAssertEqual(
            ImagePanePreview.resolve(
                generate: .done("/out/apple.png"), enlarge: .done("/out/apple_upscaled.png"),
                selected: MediaSessionItem(path: "/out/apple_upscaled.png", origin: .enlarged)),
            .result(.enlarged, "/out/apple_upscaled.png"))
    }

    func testARunningJobOutranksAStaleResult() {
        XCTAssertEqual(
            ImagePanePreview.resolve(
                generate: .done("/out/apple.png"), enlarge: .running("Restoring…"),
                selected: MediaSessionItem(path: "/out/apple.png", origin: .generated)),
            .running(.enlarged, "Restoring…"))
    }

    func testARemountedPaneStillShowsWhatItsServicesFinishedEarlier() {
        // The pane is a PAGE of the chat window and is destroyed when you look
        // at a conversation; the services outlive it. So on the first render
        // after coming back, nothing is selected yet and the finished run is
        // still the right thing to draw.
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .done("/out/apple.png"),
                                     enlarge: .idle, selected: nil),
            .result(.generated, "/out/apple.png"))
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .idle, enlarge: .idle, selected: nil),
            .empty)
    }

    func testAFailureIsShownAgainstTheThingThatFailed() {
        // The two sides fail for different reasons and offer different
        // remedies (a prompt to change vs a scale to lower), so the origin has
        // to survive to the view.
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .idle, enlarge: .failed("Out of memory"),
                                     selected: nil),
            .failed(.enlarged, "Out of memory"))
    }

    // MARK: - The persisted pick survives the rename

    func testAnOlderBuildsEditModeFlagBecomesTheVerbItMeant() {
        // `editMode: Bool` is no longer a stored property, so it cannot ride
        // the synthesized CodingKeys — without an explicit migration every
        // existing user silently loses their pick, the same class the
        // multi-LoRA change already hit.
        let variation = try! JSONDecoder().decode(
            ImageGenSettings.self, from: Data(#"{"editMode":false}"#.utf8))
        XCTAssertEqual(variation.sourceVerb, .variation)

        let edit = try! JSONDecoder().decode(
            ImageGenSettings.self, from: Data(#"{"editMode":true}"#.utf8))
        XCTAssertEqual(edit.sourceVerb, .edit)

        // A blob written by THIS build wins over the legacy key if both are
        // somehow present, and a blob with neither takes the default.
        let both = try! JSONDecoder().decode(
            ImageGenSettings.self, from: Data(#"{"editMode":true,"sourceVerb":"enlarge"}"#.utf8))
        XCTAssertEqual(both.sourceVerb, .enlarge)
        XCTAssertEqual(try! JSONDecoder().decode(
            ImageGenSettings.self, from: Data("{}".utf8)).sourceVerb, .edit)
    }

    func testTheVerbRoundTripsThroughTheSettingsBlob() {
        var s = ImageGenSettings()
        s.sourceVerb = .enlarge
        let back = try! JSONDecoder().decode(ImageGenSettings.self,
                                             from: try! JSONEncoder().encode(s))
        XCTAssertEqual(back.sourceVerb, .enlarge)
    }

    // MARK: - The session strip

    private func gen(_ n: String) -> String { "/out/images/2026-08-21/\(n)" }
    private func up(_ n: String) -> String { "/out/upscales/2026-08-21/\(n)" }

    func testTheStripIsOneTimelineNotTwoListsGluedTogether() {
        // Both writers stamp `yyyy-MM-dd_HH-mm-ss` onto the filename, so the
        // merge orders by the name itself and needs no stat of 120 files on
        // every redraw. Generated and enlarged interleave: the strip is what
        // this pane MADE, in the order it made it, and grouping by which
        // service produced a picture is a distinction nobody is looking for.
        let items = MediaSessionStrip.items(
            generated: [gen("2026-08-21_10-56-58_apple.png"),
                        gen("2026-08-21_09-00-00_pear.png")],
            enlarged: [up("2026-08-21_11-10-00_apple_upscaled.png"),
                       up("2026-08-21_09-30-00_pear_upscaled.png")])
        XCTAssertEqual(items.map(\.filename), [
            "2026-08-21_11-10-00_apple_upscaled.png",
            "2026-08-21_10-56-58_apple.png",
            "2026-08-21_09-30-00_pear_upscaled.png",
            "2026-08-21_09-00-00_pear.png",
        ])
        XCTAssertEqual(items.map(\.origin), [.enlarged, .generated, .enlarged, .generated])
    }

    func testAFileWithNoTimestampStillShowsUp() {
        // `recent` is rebuilt by scanning the output folders, so anything the
        // user dropped in there is in the list. It has no age we can read, so
        // it sorts last — but dropping it would make the strip disagree with
        // the folder the "Open output folder" link opens.
        let items = MediaSessionStrip.items(
            generated: [gen("scan.png"), gen("2026-08-21_10-00-00_apple.png")],
            enlarged: [])
        XCTAssertEqual(items.map(\.filename), ["2026-08-21_10-00-00_apple.png", "scan.png"])
    }

    func testDeletingTheSelectedItemLandsOnItsNeighbourNotOnNothing() {
        // Clearing out a run of bad results is the reason delete exists, and a
        // strip that blanks the preview after every one makes the app feel
        // like it lost its place — the next picture is right there.
        let items = MediaSessionStrip.items(
            generated: [gen("2026-08-21_12-00-00_c.png"),
                        gen("2026-08-21_11-00-00_b.png"),
                        gen("2026-08-21_10-00-00_a.png")],
            enlarged: [])
        XCTAssertEqual(
            MediaSessionStrip.selectionAfterDelete(items, removing: items[1].path, selected: items[1].path),
            items[2].path)
        // The last one has no "next", so it falls back to the one before it.
        XCTAssertEqual(
            MediaSessionStrip.selectionAfterDelete(items, removing: items[2].path, selected: items[2].path),
            items[1].path)
    }

    func testDeletingSomethingElseLeavesTheSelectionAlone() {
        let items = MediaSessionStrip.items(
            generated: [gen("2026-08-21_12-00-00_c.png"), gen("2026-08-21_11-00-00_b.png")],
            enlarged: [])
        XCTAssertEqual(
            MediaSessionStrip.selectionAfterDelete(items, removing: items[0].path, selected: items[1].path),
            items[1].path)
        // Nothing left to select is the one case that legitimately clears it.
        XCTAssertNil(MediaSessionStrip.selectionAfterDelete(
            [items[0]], removing: items[0].path, selected: items[0].path))
    }

    // MARK: - Strip thumbnails

    func testAStripTileDecodesAThumbnailNotTheWholePicture() throws {
        // A strip row is 56pt. The pane's own output is up to 4096 square, and
        // the strip holds 60 of them — decoding those at full size on every
        // SwiftUI body pass is the difference between a film strip and a
        // pane that crawls. `CGImageSource` reads a reduced image directly
        // instead of decoding and then shrinking.
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let big = dir.appendingPathComponent("2026-08-21_10-00-00_big.png")
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 2048, pixelsHigh: 1024,
                                   bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                   isPlanar: false, colorSpaceName: .deviceRGB,
                                   bytesPerRow: 0, bitsPerPixel: 0)!
        try rep.representation(using: .png, properties: [:])!.write(to: big)

        let thumb = try XCTUnwrap(MediaThumbnails.load(path: big.path, maxPixel: 128))
        XCTAssertEqual(max(thumb.size.width, thumb.size.height), 128, accuracy: 1)

        // A path whose file is gone returns nil rather than throwing — the
        // strip keeps the row so it can still be selected and deleted.
        XCTAssertNil(MediaThumbnails.load(path: dir.appendingPathComponent("gone.png").path,
                                          maxPixel: 128))
    }
}
