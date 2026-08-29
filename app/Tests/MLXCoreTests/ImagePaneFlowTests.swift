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
        // that reason — what is on screen is a property of what has finished,
        // not of what the controls are currently set to.
        let s = ImagePanePreview.resolve(generate: .done("/out/apple.png"),
                                         enlarge: .idle,
                                         focus: .generated)
        XCTAssertEqual(s, .result(.generated, "/out/apple.png"))
    }

    func testTheNewerResultTakesThePreview() {
        // Focus moves when a run finishes, so an enlarge of a generated image
        // replaces it rather than hiding behind it.
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .done("/out/apple.png"),
                                     enlarge: .done("/out/apple_upscaled.png"),
                                     focus: .enlarged),
            .result(.enlarged, "/out/apple_upscaled.png"))
    }

    func testARunningJobOutranksAStaleResult() {
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .done("/out/apple.png"),
                                     enlarge: .running("Restoring…"),
                                     focus: .generated),
            .running(.enlarged, "Restoring…"))
    }

    func testAFocusWhoseSideHasNothingFallsBackRatherThanGoingBlank() {
        // Cancelling an enlarge returns its service to idle while focus still
        // names it. The generated image is still there and still the last
        // thing that finished, so it comes back.
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .done("/out/apple.png"),
                                     enlarge: .idle,
                                     focus: .enlarged),
            .result(.generated, "/out/apple.png"))
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .idle, enlarge: .idle, focus: .generated),
            .empty)
    }

    func testAFailureIsShownAgainstTheThingThatFailed() {
        // The two sides fail for different reasons and offer different
        // remedies (a prompt to change vs a scale to lower), so the origin has
        // to survive to the view.
        XCTAssertEqual(
            ImagePanePreview.resolve(generate: .idle,
                                     enlarge: .failed("Out of memory"),
                                     focus: .enlarged),
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
}
