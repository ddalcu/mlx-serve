import XCTest
@testable import MLXCore

/// YuE2 in the Music pane: the third music family. It shares the endpoint and
/// almost nothing else — lyric-conditioned like Music 3, but with a score it
/// plans (`cot`) or is handed (`abc`), no wordless mode, and none of the
/// tempo/key/language fields. The family gates the FIELDS, because values
/// linger in @State across a model switch and the server names each a 400.
final class Yue2MusicTests: XCTestCase {

    private let yue2 = MusicModelPreset.yue2_3B_8bit

    // MARK: - Capabilities

    func testPresetDeclaresItsOwnKnobSet() {
        XCTAssertEqual(yue2.family, .yue2)
        XCTAssertEqual(yue2.repo, "ddalcu/YuE2-3B-MLX-Serve-8bit")
        XCTAssertTrue(yue2.requiresLyrics, "the model is lyric-conditioned; the server 400s empty lyrics")
        XCTAssertFalse(yue2.supportsInstrumental)
        XCTAssertFalse(yue2.supportsTempoAndKey)
        XCTAssertFalse(yue2.supportsMusicalMeta)
        XCTAssertFalse(yue2.supportsReferenceAudio)
        XCTAssertFalse(yue2.supportsSourceAudio)
        XCTAssertTrue(yue2.supportsSteps)
        XCTAssertEqual(yue2.stepsRange, 1...100)
        XCTAssertEqual(yue2.fixedSteps, 32)
        XCTAssertTrue(yue2.supportsScore)
        XCTAssertEqual(yue2.durationRange, 5...360)
        XCTAssertTrue(MusicModelPreset.all.contains(yue2), "catalog must offer it")
        XCTAssertEqual(Set(MusicModelPreset.all.map(\.id)).count, MusicModelPreset.all.count)
    }

    func testTheOtherFamiliesKeepTheirOwnKnobSets() {
        for p in [MusicModelPreset.acestepXLTurbo8bit, .miniMaxMusic3_8bit] {
            XCTAssertFalse(p.supportsScore, p.name)
            XCTAssertTrue(p.supportsInstrumental, p.name)
            XCTAssertTrue(p.supportsTempoAndKey, p.name)
        }
        XCTAssertEqual(MusicModelPreset.miniMaxMusic3_8bit.stepsRange, 4...100)
        XCTAssertFalse(MusicModelPreset.acestepXLTurbo8bit.supportsSteps)
    }

    // MARK: - Request fields

    func testRequestBodyCarriesNoFieldAnotherBackendOwns() {
        let req = MusicGenRequest(
            model: yue2, prompt: "English, pop", lyrics: "[Verse]\nla",
            instrumental: true, vocalLanguage: "ja", bpm: 96, keyscale: "C major",
            timesignature: "3", durationSeconds: 90, seed: 7, steps: 16,
            refAudioPath: "/tmp/ref.wav", task: .cover, srcAudioPath: "/tmp/src.wav")
        let body = MusicGenService.requestBody(req, modelName: "yue2", refAudioB64: "AAAA", srcAudioB64: "AAAA")
        for key in ["instrumental", "bpm", "keyscale", "vocal_language", "timesignature",
                    "ref_audio", "src_audio", "task", "cover_strength"] {
            XCTAssertNil(body[key], "\(key) is another backend's field; the server names it a 400")
        }
        XCTAssertEqual(body["lyrics"] as? String, "[Verse]\nla",
                       "a leftover instrumental flag yields to the lyrics instead of dropping them")
        XCTAssertEqual(body["prompt"] as? String, "English, pop")
        XCTAssertEqual(body["steps"] as? Int, 16)
        XCTAssertEqual(body["duration_seconds"] as? Int, 90)
        XCTAssertEqual(body["seed"] as? Int, 7)
        XCTAssertEqual(body["stream"] as? Bool, true)
    }

    func testStepsAndDurationAreClampedIntoTheServerRange() {
        let req = MusicGenRequest(model: yue2, prompt: "p", lyrics: "l", durationSeconds: 900, steps: 500)
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["steps"] as? Int, 100)
        XCTAssertEqual(body["duration_seconds"] as? Int, 360)
        let low = MusicGenService.requestBody(
            MusicGenRequest(model: yue2, prompt: "p", lyrics: "l", durationSeconds: 1, steps: 0), modelName: "m")
        XCTAssertEqual(low["steps"] as? Int, 1)
        XCTAssertEqual(low["duration_seconds"] as? Int, 5)
    }

    func testPlanRidesWithTheScoreOnlyWhereThePlanReadsOne() {
        func body(_ plan: MusicPlan, _ score: String, _ model: MusicModelPreset = .yue2_3B_8bit) -> [String: Any] {
            MusicGenService.requestBody(
                MusicGenRequest(model: model, prompt: "p", lyrics: "l", plan: plan, score: score), modelName: "m")
        }
        let full = body(.full, "  X:1\nK:C\n  ")
        XCTAssertEqual(full["cot"] as? String, "full")
        XCTAssertEqual(full["abc"] as? String, "X:1\nK:C", "the score is trimmed")
        XCTAssertEqual(body(.melody, "X:1")["cot"] as? String, "melody")
        // An empty box means "let the model write one".
        XCTAssertNil(body(.full, "   \n")["abc"])
        XCTAssertEqual(body(.full, "")["cot"] as? String, "full")
        // `off` renders without a score; the server 400s abc beside it, so it is dropped.
        let off = body(.off, "X:1")
        XCTAssertEqual(off["cot"] as? String, "off")
        XCTAssertNil(off["abc"])
        // Another family never sees the fields at all.
        for other in [MusicModelPreset.miniMaxMusic3_8bit, .acestepXLTurbo8bit] {
            let b = body(.full, "X:1", other)
            XCTAssertNil(b["cot"], other.name)
            XCTAssertNil(b["abc"], other.name)
        }
    }

    func testInstrumentalNeverSatisfiesTheLyricsGateOnYue2() {
        XCTAssertFalse(MusicGenRequest.lyricsSatisfied(model: yue2, lyrics: "  ", instrumental: true))
        XCTAssertFalse(MusicGenRequest.lyricsSatisfied(model: yue2, lyrics: "", instrumental: false))
        XCTAssertTrue(MusicGenRequest.lyricsSatisfied(model: yue2, lyrics: "[Verse]\nla", instrumental: true))
        // Music 3 keeps the lift: it asks for wordless in text.
        XCTAssertTrue(MusicGenRequest.lyricsSatisfied(model: .miniMaxMusic3_8bit, lyrics: "", instrumental: true))
    }

    func testSidecarRecordsThePlanAndWhetherTheScoreWasEdited() {
        let planned = MusicGenService.settingsText(
            MusicGenRequest(model: yue2, prompt: "pop", lyrics: "la", plan: .melody), resolvedSeed: 1, modelName: "m")
        XCTAssertTrue(planned.contains("score: melody"), planned)
        XCTAssertFalse(planned.contains("score_source"), planned)
        let edited = MusicGenService.settingsText(
            MusicGenRequest(model: yue2, prompt: "pop", lyrics: "la", plan: .full, score: "X:1"),
            resolvedSeed: 1, modelName: "m")
        XCTAssertTrue(edited.contains("score_source: edited"), edited)
        let none = MusicGenService.settingsText(
            MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "pop", lyrics: "la"), resolvedSeed: 1, modelName: "m")
        XCTAssertFalse(none.contains("score"), none)
        XCTAssertEqual(MusicGenService.scorePath(forWav: "/a/b/song.wav"), "/a/b/song.abc")
    }

    // MARK: - Download bundle

    func testBundleMarkersAreTheFilesTheServerNeeds() {
        let comp = yue2.bundle.components[0]
        XCTAssertEqual(comp.repo, yue2.repo)
        for m in ["config.json", "model.safetensors", "vae.safetensors", "vae_config.json", "qwen.tiktoken"] {
            XCTAssertTrue(comp.readyMarkers.contains(m), m)
        }
        // The server skips a dir without the VAE: the app must say the same.
        XCTAssertEqual(DownloadManager.requiredMediaMarker(modelType: "yue2"), "vae.safetensors")
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1600],
            ["path": "README.md", "type": "file", "size": 4000],
            ["path": "config.json", "type": "file", "size": 1042],
            ["path": "model.safetensors", "type": "file", "size": 4_264_488_042],
            ["path": "vae.safetensors", "type": "file", "size": 265_441_814],
            ["path": "vae_config.json", "type": "file", "size": 1378],
            ["path": "qwen.tiktoken", "type": "file", "size": 2_561_218],
            ["path": "yue2_generation_config.json", "type": "file", "size": 466],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: comp.selection).map(\.0)
        for marker in comp.readyMarkers {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
        XCTAssertTrue(picked.contains("yue2_generation_config.json"), "the sampling defaults ride along")
        XCTAssertFalse(picked.contains("README.md"))
    }

    // MARK: - Routing

    func testArchitectureRoutesToTheMusicPaneAndResolvesItsPreset() {
        XCTAssertTrue(isMediaModelType("yue2"))
        XCTAssertTrue(discoverableMediaModelType("yue2"))
        XCTAssertEqual(MediaModality(modelType: "yue2"), .music)
        let models = [ModelInfo(name: "someone/YuE2-3B-MLX-4bit",
                                quantBits: 4, layers: 0, hiddenSize: 0, vocabSize: 0,
                                contextLength: 0, modelMaxTokens: 0,
                                architecture: "yue2",
                                capabilities: ["audio", "music"])]
        let p = CustomMediaModels.musicPreset(for: "someone/YuE2-3B-MLX-4bit", from: models)
        XCTAssertEqual(p?.family, .yue2)
        XCTAssertEqual(p?.repo, "someone/YuE2-3B-MLX-4bit")
        XCTAssertFalse(p?.supportsTempoAndKey ?? true)
        XCTAssertEqual(yue2.capabilityLabel, "Best for editable songs (score + vocals)")
    }

    // MARK: - Prompts and templates

    func testTemplatesSpeakYue2sOwnFormat() {
        let styles = MusicPrompt.builtinStyles(for: .yue2)
        XCTAssertGreaterThanOrEqual(styles.count, 3)
        XCTAssertEqual(Set(styles.map(\.title)).count, styles.count)
        for s in styles {
            XCTAssertTrue(s.body.contains(","), "a tag line, not prose: \(s.title)")
            XCTAssertFalse(s.body.contains("\n"), s.title)
        }
        let lyrics = MusicPrompt.builtinLyrics(for: .yue2)
        XCTAssertFalse(lyrics.isEmpty)
        for l in lyrics {
            XCTAssertTrue(l.body.contains("[Verse]") && l.body.contains("[Chorus]"), l.title)
            XCTAssertFalse(l.body.contains("[verse]"), "the card's tags are capitalized: \(l.title)")
        }
        XCTAssertEqual(MusicOptions.sectionTagHint(for: .yue2),
                       "[Intro] [Verse] [Pre-Chorus] [Chorus] [Interlude] [Bridge] [Outro]")
        XCTAssertEqual(MusicOptions.sectionTagHint(for: .minimaxMusic3), MusicOptions.sectionTagHint)
        XCTAssertEqual(MusicPrompt.builtinLyrics(for: .acestep), MusicPrompt.builtinLyrics)
    }

    func testRewriterAsksForTagsNotProse() {
        let style = MusicPromptRewriter.request(.style, text: "sad piano", family: .yue2,
                                                other: "", instrumental: false, language: "en")
        XCTAssertTrue(style.system.contains("comma-separated tags"), style.system)
        let lyrics = MusicPromptRewriter.request(.lyrics, text: "la", family: .yue2,
                                                 other: "", instrumental: false, language: "en")
        XCTAssertTrue(lyrics.system.contains("[Pre-Chorus]"), lyrics.system)
    }

    func testAgentIsToldTheEngineTakesLyricsAndNoWordlessTrack() {
        let note = AgentPrompt.musicEngineNote(yue2)
        XCTAssertTrue(note.contains("requires `lyrics`"), note)
        XCTAssertTrue(note.contains("cannot make a wordless track"), note)
        XCTAssertTrue(note.contains("ignores tempo, key"), note)
        XCTAssertFalse(AgentPrompt.musicEngineNote(.miniMaxMusic3_8bit).contains("cannot make a wordless track"))
    }

    // MARK: - Progress and settings

    func testStageLabelsNameYue2sPhases() {
        XCTAssertEqual(MediaSSE.stageLabel("abc"), "Writing the score")
        XCTAssertEqual(MediaSSE.stageLabel("semantic"), "Composing")
        XCTAssertEqual(MediaSSE.stageLabel("nar"), "Rendering")
    }

    func testPlanAndScoreAreStickyAndAnOldBlobStillDecodes() throws {
        var s = MusicGenSettings()
        s.modelId = yue2.id
        s.plan = .melody
        s.score = "X:1\nK:D"
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(back.plan, .melody)
        XCTAssertEqual(back.score, "X:1\nK:D")
        XCTAssertEqual(back.resolvedModel, yue2)
        // A blob from before these keys existed keeps the defaults.
        let old = try JSONDecoder().decode(MusicGenSettings.self, from: Data(#"{"prompt":"pop"}"#.utf8))
        XCTAssertEqual(old.plan, .full)
        XCTAssertEqual(old.score, "")
    }
}
