import XCTest
@testable import MLXCore

/// The pure core of the four in-chat media tools: what a model is allowed to
/// ask for, and what it gets when it asks for something silly.
///
/// A small local model will happily request 60 seconds of video or a 8192²
/// image, so every model-supplied number is clamped here rather than at the
/// service — the clamp is the whole reason these tools can be handed to a 2B.
final class MediaToolArgsTests: XCTestCase {

    // MARK: - Image size

    func testImageSizeSnapsToAnAllowedBucket() {
        let m = ImageModelPreset.flux2Klein4B_Q4
        // Exact bucket comes back verbatim.
        XCTAssertEqual(MediaToolArgs.imageSize("1216x832", model: m, saved: m.defaultResolution).id, "1216x832")
        // Off-grid snaps to the nearest bucket the model was trained on.
        let snapped = MediaToolArgs.imageSize("1200x800", model: m, saved: m.defaultResolution)
        XCTAssertTrue(m.resolutions.contains(snapped), "must land on a trained bucket, got \(snapped.id)")
        XCTAssertEqual(snapped.id, "1216x832")
    }

    func testOversizedImageRequestIsClampedIntoTheCatalog() {
        let m = ImageModelPreset.flux2Klein4B_Q4
        let huge = MediaToolArgs.imageSize("8192x8192", model: m, saved: m.defaultResolution)
        XCTAssertTrue(m.resolutions.contains(huge))
        XCTAssertLessThanOrEqual(huge.width, 1536)
    }

    func testUnparseableOrAbsentImageSizeFallsBackToTheSavedBucket() {
        let m = ImageModelPreset.flux2Klein4B_Q4
        let saved = m.resolutions[3]
        XCTAssertEqual(MediaToolArgs.imageSize(nil, model: m, saved: saved), saved)
        XCTAssertEqual(MediaToolArgs.imageSize("big", model: m, saved: saved), saved)
        XCTAssertEqual(MediaToolArgs.imageSize("", model: m, saved: saved), saved)
    }

    func testImageSizeAcceptsTheSpellingsModelsActuallyEmit() {
        let m = ImageModelPreset.flux2Klein4B_Q4
        for spelling in ["1024x1024", "1024X1024", "1024 x 1024", "1024\u{00D7}1024"] {
            XCTAssertEqual(MediaToolArgs.imageSize(spelling, model: m, saved: m.defaultResolution).id,
                           "1024x1024", "failed on \(spelling)")
        }
    }

    func testABareAspectRatioIsAShapeNotAFallback() {
        // Measured live: asked for "widescreen", the model sent {"size":"16:9"}
        // and got a SQUARE, because a ratio isn't WIDTHxHEIGHT. Refusing the
        // spelling models actually use just means silently ignoring them.
        let m = ImageModelPreset.flux2Klein4B_Q4
        let wide = MediaToolArgs.imageSize("16:9", model: m, saved: m.defaultResolution)
        XCTAssertGreaterThan(wide.width, wide.height, "16:9 must come back landscape, got \(wide.id)")
        XCTAssertEqual(Double(wide.width) / Double(wide.height), 16.0 / 9.0, accuracy: 0.05)

        let tall = MediaToolArgs.imageSize("9:16", model: m, saved: m.defaultResolution)
        XCTAssertGreaterThan(tall.height, tall.width, "9:16 must come back portrait, got \(tall.id)")

        XCTAssertEqual(MediaToolArgs.imageSize("1:1", model: m, saved: m.resolutions[3]).id, "1024x1024")
    }

    func testAmongEqualShapesTheAspectPicksASizeNearTheUsersOwn() {
        // Ranking a ratio by AREA would pick the smallest matching bucket (a
        // ratio's own "area" is 144 pixels); the honest anchor is the size the
        // user's own settings are already on. Mage-Flow has four exact squares.
        let m = ImageModelPreset.mageFlowTurbo8bit
        let big = m.resolutions.first { $0.id == "2048x2048" }!
        let small = m.resolutions.first { $0.id == "512x512" }!
        XCTAssertEqual(MediaToolArgs.imageSize("1:1", model: m, saved: big).id, "2048x2048")
        XCTAssertEqual(MediaToolArgs.imageSize("1:1", model: m, saved: small).id, "512x512")
    }

    func testAnAspectNeverEscalatesAChatPreviewToTheBiggestBucket() {
        // Measured live: "widescreen" from a chat landed on 2048×1152 — the
        // model's largest 16:9, eight times the pixels of the size the user
        // works at — because an EXACT ratio match outranked everything. A chat
        // generation is a preview; the tray window is where size is chosen.
        let m = ImageModelPreset.mageFlowTurbo8bit
        let usual = m.resolutions.first { $0.id == "1024x1024" }!
        let picked = MediaToolArgs.imageSize("16:9", model: m, saved: usual)
        XCTAssertEqual(picked.id, "1344x768")
        XCTAssertLessThan(picked.width * picked.height, 2 * usual.width * usual.height,
                          "a ratio must not multiply the pixel count")
    }

    func testAShapeThatMatchesNothingStillGetsTheClosestShape() {
        // 4:1 has no near neighbour on FLUX's bucket list, so the tolerance
        // window is empty — falling back to "closest area" would return a
        // square. The closest SHAPE has to win outright.
        let m = ImageModelPreset.flux2Klein4B_Q4
        XCTAssertEqual(MediaToolArgs.imageSize("4:1", model: m, saved: m.defaultResolution).id,
                       "1536x640")
    }

    func testVideoAcceptsAnAspectToo() throws {
        let m = VideoModelPreset.ltx23Q4
        let req = try MediaToolArgs.video(["prompt": "clouds", "size": "3:4"], model: m,
                                          saved: m.defaultResolution, keepResident: false, lanId: nil)
        XCTAssertGreaterThan(req.height, req.width, "3:4 must come back portrait, got \(req.width)x\(req.height)")
    }

    func testDistilledModelsIgnoreStepTiersEntirely() {
        // Mage-Flow is distillation-fixed at 4: the tiers buy time, not quality.
        let req = try! MediaToolArgs.image(["prompt": "a fox"],
                                           model: .mageFlowTurbo8bit,
                                           saved: ImageModelPreset.mageFlowTurbo8bit.defaultResolution,
                                           seed: -1, keepResident: false, lanId: nil)
        XCTAssertEqual(req.steps, ImageModelPreset.mageFlowTurbo8bit.fixedSteps)
    }

    func testImageUsesFastStepsNotTheSavedQualityTier() {
        let m = ImageModelPreset.flux2Klein4B_Q4
        let req = try! MediaToolArgs.image(["prompt": "a fox"], model: m,
                                           saved: m.defaultResolution,
                                           seed: -1, keepResident: false, lanId: nil)
        XCTAssertEqual(req.steps, m.settings(MediaChatDefaults.imageQuality).steps)
        XCTAssertEqual(req.prompt, "a fox")
    }

    func testImageWithoutAPromptIsRejectedNamingTheKey() {
        XCTAssertThrowsError(try MediaToolArgs.image(["size": "1024x1024"],
                                                     model: .flux2Klein4B_Q4,
                                                     saved: ImageModelPreset.flux2Klein4B_Q4.defaultResolution,
                                                     seed: -1, keepResident: false, lanId: nil)) { err in
            XCTAssertTrue("\(err)".contains("prompt"), "\(err)")
        }
    }

    // MARK: - Speech

    func testSpeechSpeedIsClampedToTheUsableRange() {
        XCTAssertEqual(MediaToolArgs.speechSpeed("0.1"), 0.5, accuracy: 0.0001)
        XCTAssertEqual(MediaToolArgs.speechSpeed("9"), 2.0, accuracy: 0.0001)
        XCTAssertEqual(MediaToolArgs.speechSpeed("1.25"), 1.25, accuracy: 0.0001)
        XCTAssertEqual(MediaToolArgs.speechSpeed(nil), MediaChatDefaults.speechSpeed, accuracy: 0.0001)
        XCTAssertEqual(MediaToolArgs.speechSpeed("fast"), MediaChatDefaults.speechSpeed, accuracy: 0.0001)
    }

    func testSpeechNeedsTextAndSaysSo() {
        XCTAssertThrowsError(try MediaToolArgs.speech([:], model: .qwen3TTS06B8bit,
                                                      keepResident: false, lanId: nil)) { err in
            XCTAssertTrue("\(err)".contains("text"), "\(err)")
        }
    }

    func testSpeechCarriesTheClampedSpeed() throws {
        let req = try MediaToolArgs.speech(["text": "hello there", "speed": "5"],
                                           model: .qwen3TTS06B8bit, keepResident: false, lanId: nil)
        XCTAssertEqual(req.text, "hello there")
        XCTAssertEqual(req.speed, 2.0, accuracy: 0.0001)
    }

    // MARK: - Music

    func testMusicDurationIsClampedToTheServersRange() {
        XCTAssertEqual(MediaToolArgs.musicSeconds("1"), 10)
        XCTAssertEqual(MediaToolArgs.musicSeconds("9000"), 600)
        XCTAssertEqual(MediaToolArgs.musicSeconds("45"), 45)
        XCTAssertEqual(MediaToolArgs.musicSeconds(nil), MediaChatDefaults.musicSeconds)
        XCTAssertEqual(MediaToolArgs.musicSeconds("a while"), MediaChatDefaults.musicSeconds)
    }

    // An omitted duration used to be a flat 30 s — which the tool description
    // actively invited ("omit for 30") — so a full lyric sheet was cut off
    // mid-song. With lyrics in hand the fallback is derived from them; the
    // model's own number still wins whenever it sends one.
    func testAnOmittedDurationIsSizedToTheLyrics() {
        let sheet = (1...24).map { "line \($0)" }.joined(separator: "\n")
        let sized = MediaToolArgs.musicSeconds(nil, lyrics: "[verse]\n" + sheet)
        XCTAssertGreaterThan(sized, 100, "24 sung lines do not fit in 30 seconds")
        XCTAssertLessThanOrEqual(sized, MediaChatDefaults.musicSecondsRange.upperBound)

        // Section tags are not sung, so they don't buy time.
        XCTAssertEqual(MediaToolArgs.musicSeconds(nil, lyrics: "[verse]\n[chorus]\n[outro]"),
                       MediaChatDefaults.musicSeconds)
        // No lyrics → the short instrumental preview, unchanged.
        XCTAssertEqual(MediaToolArgs.musicSeconds(nil, lyrics: ""), MediaChatDefaults.musicSeconds)
        // An explicit request is never second-guessed.
        XCTAssertEqual(MediaToolArgs.musicSeconds("45", lyrics: sheet), 45)
        // A novel's worth of lyrics still lands in the server's range.
        let huge = (1...5000).map { "line \($0)" }.joined(separator: "\n")
        XCTAssertEqual(MediaToolArgs.musicSeconds(nil, lyrics: huge),
                       MediaChatDefaults.musicSecondsRange.upperBound)
    }

    func testMusicDefaultsToAShortPreviewNotTheWindowsSixtySeconds() {
        // The tray window keeps its own 60s default; a chat generation blocks
        // decode on one GPU, so it stays a preview.
        XCTAssertEqual(MediaChatDefaults.musicSeconds, 30)
        XCTAssertNotEqual(MediaChatDefaults.musicSeconds, MusicGenSettings().durationSeconds)
    }

    func testMusicPassesLyricsThroughAndOmitsThemWhenAbsent() throws {
        let with = try MediaToolArgs.music(["prompt": "lo-fi", "lyrics": "[Verse]\nla la"],
                                           model: .acestepXLTurbo8bit, language: "en",
                                           keepResident: false, lanId: nil)
        XCTAssertEqual(with.lyrics, "[Verse]\nla la")
        let without = try MediaToolArgs.music(["prompt": "lo-fi"],
                                              model: .acestepXLTurbo8bit, language: "en",
                                           keepResident: false, lanId: nil)
        XCTAssertTrue(without.lyrics.isEmpty)
        XCTAssertEqual(without.durationSeconds, MediaChatDefaults.musicSeconds)
    }

    // MARK: - Music: the rest of the knobs

    func testBpmIsClampedToWhatTheServerAccepts() {
        // The engine 400s outside [30,300] — a clamp is the difference between a
        // fast track and a failed turn.
        XCTAssertEqual(MediaToolArgs.musicBpm("128"), 128)
        XCTAssertEqual(MediaToolArgs.musicBpm("500"), 300)
        XCTAssertEqual(MediaToolArgs.musicBpm("5"), 30)
        XCTAssertEqual(MediaToolArgs.musicBpm("120 bpm"), 120)
        // Nothing readable → omit the field entirely so the model decides.
        XCTAssertNil(MediaToolArgs.musicBpm("fast"))
        XCTAssertNil(MediaToolArgs.musicBpm(nil))
    }

    func testKeyscaleIsCanonicalisedOrDropped() {
        // keyscale is NOT range-checked server-side: an unrecognised string is
        // passed straight into the conditioning as junk. Omitting is the honest
        // failure — the model picks a key instead of being told nonsense.
        XCTAssertEqual(MediaToolArgs.musicKeyscale("A minor"), "A minor")
        XCTAssertEqual(MediaToolArgs.musicKeyscale("a minor"), "A minor")
        XCTAssertEqual(MediaToolArgs.musicKeyscale("  bb MAJOR "), "Bb major")
        XCTAssertEqual(MediaToolArgs.musicKeyscale("H sharp lydian"), "")
        XCTAssertEqual(MediaToolArgs.musicKeyscale(nil), "")
        // Whatever comes back must be something the catalogue itself lists.
        for k in MusicOptions.keyscales {
            XCTAssertEqual(MediaToolArgs.musicKeyscale(k.lowercased()), k)
        }
    }

    func testTimeSignatureTakesEitherSpelling() {
        // The picker shows "4/4"; the wire wants the beats-per-bar number. A
        // model will write either.
        XCTAssertEqual(MediaToolArgs.musicTimeSignature("4/4"), "4")
        XCTAssertEqual(MediaToolArgs.musicTimeSignature("6/8"), "6")
        XCTAssertEqual(MediaToolArgs.musicTimeSignature("3"), "3")
        XCTAssertEqual(MediaToolArgs.musicTimeSignature("7/8"), "", "an unsupported meter is dropped")
        XCTAssertEqual(MediaToolArgs.musicTimeSignature(nil), "")
    }

    func testVocalLanguageTakesACodeOrANameAndOtherwiseKeepsTheSetting() {
        XCTAssertEqual(MediaToolArgs.musicLanguage("es", fallback: "en"), "es")
        XCTAssertEqual(MediaToolArgs.musicLanguage("Spanish", fallback: "en"), "es")
        XCTAssertEqual(MediaToolArgs.musicLanguage("japanese", fallback: "en"), "ja")
        XCTAssertEqual(MediaToolArgs.musicLanguage("auto", fallback: "en"), "unknown")
        // A language we don't serve falls back to the user's own setting rather
        // than conditioning the singer on a code the model invented.
        XCTAssertEqual(MediaToolArgs.musicLanguage("Klingon", fallback: "en"), "en")
        XCTAssertEqual(MediaToolArgs.musicLanguage(nil, fallback: "unknown"), "unknown")
    }

    func testEveryMusicKnobReachesTheRequest() throws {
        let req = try MediaToolArgs.music([
            "prompt": "driving synthwave",
            "lyrics": "[Chorus]\nneon nights",
            "duration_seconds": "45",
            "bpm": "128",
            "keyscale": "a minor",
            "time_signature": "4/4",
            "vocal_language": "English",
        ], model: .acestepXLTurbo8bit, language: "unknown", keepResident: false, lanId: nil)
        XCTAssertEqual(req.durationSeconds, 45)
        XCTAssertEqual(req.bpm, 128)
        XCTAssertEqual(req.keyscale, "A minor")
        XCTAssertEqual(req.timesignature, "4")
        XCTAssertEqual(req.vocalLanguage, "en")
        XCTAssertEqual(req.lyrics, "[Chorus]\nneon nights")

        // And every one of them survives into the wire body the server reads.
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["bpm"] as? Int, 128)
        XCTAssertEqual(body["keyscale"] as? String, "A minor")
        XCTAssertEqual(body["timesignature"] as? String, "4")
        XCTAssertEqual(body["vocal_language"] as? String, "en")
        XCTAssertEqual(body["duration_seconds"] as? Int, 45)
    }

    func testOmittedMusicKnobsStayOffTheWire() throws {
        // The engine's own "model decides" convention is an ABSENT field, so a
        // knob the user never mentioned must not appear at all.
        let req = try MediaToolArgs.music(["prompt": "ambient piano"],
                                          model: .acestepXLTurbo8bit, language: "unknown",
                                          keepResident: false, lanId: nil)
        let body = MusicGenService.requestBody(req, modelName: "m")
        for key in ["bpm", "keyscale", "timesignature", "lyrics"] {
            XCTAssertNil(body[key], "\(key) must be absent when nobody asked for it")
        }
    }

    func testMusicNeedsAPrompt() {
        XCTAssertThrowsError(try MediaToolArgs.music(["lyrics": "la"], model: .acestepXLTurbo8bit,
                                                     language: "en", keepResident: false, lanId: nil)) { err in
            XCTAssertTrue("\(err)".contains("prompt"), "\(err)")
        }
    }

    // MARK: - Video

    func testOverLongVideoIsClampedToTheChatCeiling() {
        let m = VideoModelPreset.ltx23Q4
        let ceiling = MediaToolArgs.videoFrames("60", model: m)
        XCTAssertLessThanOrEqual(ceiling, m.framesCovering(durationSeconds: MediaChatDefaults.videoMaxSeconds)!)
        XCTAssertTrue(m.frameOptions.contains(ceiling), "must land on the 8N+1 ladder, got \(ceiling)")
    }

    func testVideoFramesLandOnTheLadderForEveryRequest() {
        let m = VideoModelPreset.ltx23Q4
        for raw in [nil, "0", "0.5", "1", "2", "3", "4", "100", "nonsense"] {
            let n = MediaToolArgs.videoFrames(raw, model: m)
            XCTAssertTrue(m.frameOptions.contains(n), "\(raw ?? "nil") → \(n) is off the ladder")
            XCTAssertGreaterThanOrEqual(n, 9)
        }
    }

    func testVideoDefaultsToOneStageAndFewSteps() throws {
        let req = try MediaToolArgs.video(["prompt": "clouds"], model: .ltx23Q4,
                                          saved: VideoModelPreset.ltx23Q4.defaultResolution,
                                          keepResident: false, lanId: nil)
        XCTAssertEqual(req.mode, .oneStage)
        XCTAssertEqual(req.steps, MediaChatDefaults.videoSteps(for: .ltx23Q4))
        XCTAssertEqual(req.numFrames, MediaToolArgs.videoFrames(nil, model: .ltx23Q4))
    }

    /// Chat previews run each model's own FAST tier, never a shared constant:
    /// 8 steps is LTX's fast preset, but H3 is not step-distilled — its
    /// validated floor is 16 — so the LTX-shaped constant produced a bad clip
    /// after 15+ minutes of GPU. LTX stays byte-identical at 8.
    func testVideoStepsFollowTheModelsOwnFastTier() throws {
        XCTAssertEqual(MediaChatDefaults.videoSteps(for: .ltx23Q4), 8)
        XCTAssertEqual(MediaChatDefaults.videoSteps(for: .minimaxH3),
                       VideoModelPreset.minimaxH3.settings(.fast).steps)
        let h3 = try MediaToolArgs.video(["prompt": "clouds"], model: .minimaxH3,
                                         saved: VideoModelPreset.minimaxH3.defaultResolution,
                                         keepResident: false, lanId: nil)
        XCTAssertEqual(h3.steps, VideoModelPreset.minimaxH3.settings(.fast).steps)
        XCTAssertGreaterThanOrEqual(h3.steps, 16, "below H3's validated floor")
    }

    func testVideoSizeSnapsToATrainedBucket() throws {
        let m = VideoModelPreset.ltx23Q4
        let req = try MediaToolArgs.video(["prompt": "clouds", "size": "1920x1080"], model: m,
                                          saved: m.defaultResolution, keepResident: false, lanId: nil)
        XCTAssertTrue(m.resolutions.contains(where: { $0.width == req.width && $0.height == req.height }),
                      "\(req.width)x\(req.height) is not a trained bucket")
    }

    func testVideoNeedsAPrompt() {
        XCTAssertThrowsError(try MediaToolArgs.video(["seconds": "2"], model: .ltx23Q4,
                                                     saved: VideoModelPreset.ltx23Q4.defaultResolution,
                                                     keepResident: false, lanId: nil)) { err in
            XCTAssertTrue("\(err)".contains("prompt"), "\(err)")
        }
    }

    // MARK: - Per-turn budget

    func testOneMediaGenerationPerTurn() {
        var budget = MediaTurnBudget()
        let turn = UUID()
        XCTAssertNil(budget.claim(.image, turn: turn), "the first generation of a turn is always allowed")
        let refusal = budget.claim(.music, turn: turn)
        XCTAssertNotNil(refusal)
        // A refusal the model can act on: full sentences, not a code. The
        // console learned this live — a bare "denied" just gets retried.
        XCTAssertTrue(refusal!.hasSuffix("."), refusal!)
        XCTAssertGreaterThan(refusal!.split(separator: " ").count, 8, refusal!)
    }

    func testTheRefusalLeadsWithWhatWasNotMade() {
        // Live 2026-07-28: refused mid-turn, a 4B still told the user both
        // things had been generated. The rule first and the fact second reads as
        // policy the model can narrate around; the fact first does not.
        for kind in MediaKind.allCases {
            let text = MediaTurnBudget.refusal(for: kind)
            XCTAssertTrue(text.hasPrefix("NOT GENERATED"), text)
            XCTAssertTrue(text.contains("was NOT made"), "must give the model the words to relay: \(text)")
            XCTAssertTrue(text.contains(kind.article), "must name what wasn't made: \(text)")
        }
    }

    func testTheBudgetRefusesEveryModalityOnceSpent() {
        for kind in [MediaKind.image, .speech, .music, .video] {
            var budget = MediaTurnBudget()
            let turn = UUID()
            _ = budget.claim(.image, turn: turn)
            XCTAssertNotNil(budget.claim(kind, turn: turn),
                            "\(kind) must be refused after the turn's one generation")
        }
    }

    func testANewTurnTokenStartsCleanWithNoResetCall() {
        // The regression this shape exists for: the budget used to be cleared by
        // a `reset()` at the top of ONE loop, so a second driver on the same
        // engine (the headless harness) inherited it spent and silently refused
        // every generation forever. A token can't be forgotten.
        var budget = MediaTurnBudget()
        _ = budget.claim(.video, turn: UUID())
        XCTAssertNil(budget.claim(.video, turn: UUID()), "a different turn is a fresh budget")
        XCTAssertNil(budget.claim(.music, turn: UUID()))
    }

    func testTheSameTokenAcrossManyRoundsStaysOneGeneration() {
        // A turn is many tool ROUNDS; the budget spans all of them.
        var budget = MediaTurnBudget()
        let turn = UUID()
        XCTAssertNil(budget.claim(.image, turn: turn))
        for round in 0..<5 {
            XCTAssertNotNil(budget.claim(.image, turn: turn), "round \(round) got a second generation")
        }
    }

    // MARK: - Tool identities

    func testTheFourMediaToolsAreAdvertisedAndGateable() {
        let declared = Set(AgentPrompt.toolDefinitions.compactMap {
            ($0["function"] as? [String: Any])?["name"] as? String
        })
        for name in ["generate_image", "generate_speech", "generate_music", "generate_video"] {
            XCTAssertTrue(declared.contains(name), "\(name) must be advertised")
            XCTAssertNotNil(AgentToolKind(rawValue: name), "\(name) must be gateable")
        }
        XCTAssertFalse(declared.contains("generate_audio"),
                       "generate_audio is ambiguous next to generate_music — it was split")
    }

    func testMediaGroupHoldsAllFourTools() {
        XCTAssertEqual(Set(AgentToolGroup.media.tools),
                       [.generateImage, .generateSpeech, .generateMusic, .generateVideo])
    }

    @MainActor
    func testSwitchingMediaOffRemovesAllFourFromTheRequest() {
        // The composer's Tools ▸ Media rows are subtractive at the resolution
        // chokepoint; what the model is SENT has to follow, or it calls a tool
        // that dispatch will then refuse.
        var defaults = AppDefaultsSnapshot()
        defaults.toolsEnabled = true
        defaults.disabledTools = Set(AgentToolGroup.media.tools)
        let resolved = AgentResolution.resolve(agent: nil, defaults: defaults)
        let json = AgentPrompt.toolDefinitionsJSON(allowing: resolved.tools)
        for name in ["generate_image", "generate_speech", "generate_music", "generate_video"] {
            XCTAssertFalse(json.contains("\"\(name)\""), "\(name) survived Media being switched off")
            XCTAssertNotNil(AgentEngine.disallowedToolRefusal(name: name, allowed: resolved.tools),
                            "\(name) must also be refused at dispatch")
        }
        XCTAssertTrue(json.contains("\"shell\""), "only the media group comes out")
    }

    func testEachMediaToolCanBeAdvertisedOnItsOwn() {
        // One tool per LINE is what makes the line-based filter safe; two on one
        // line would take each other along.
        for tool in AgentToolGroup.media.tools {
            let json = AgentPrompt.toolDefinitionsJSON(allowing: [tool])
            let arr = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [[String: Any]] ?? []
            XCTAssertEqual(arr.count, 1, "\(tool.rawValue) doesn't isolate — check it's alone on its line")
            XCTAssertEqual((arr.first?["function"] as? [String: Any])?["name"] as? String, tool.rawValue)
        }
    }

    func testAnAgentThatHadGenerateAudioKeepsItsSpeechCapability() throws {
        // The decoder drops names it doesn't know, so a straight rename would
        // silently STRIP the capability from every agent that had it.
        let json = #"""
        {"tools":true,"mcp":false,"web":true,"advancedTools":["generate_audio","readFile"]}
        """#
        let caps = try JSONDecoder().decode(AgentCapabilities.self, from: Data(json.utf8))
        XCTAssertEqual(caps.advancedTools, [.generateSpeech, .readFile])
    }
}
