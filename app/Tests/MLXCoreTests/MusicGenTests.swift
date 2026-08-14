import XCTest
@testable import MLXCore

/// Music tab (ACE-Step text2music): preset catalog, the
/// `/v1/audio/music-generations` wire contract, output-path slugging,
/// sticky-settings round-trip, and bundle readiness.
final class MusicGenTests: XCTestCase {

    // MARK: - Preset catalog

    func testMusicPresetCatalogIsWellFormed() {
        XCTAssertFalse(MusicModelPreset.all.isEmpty)
        // Default (first) is the XL Turbo 8-bit build.
        XCTAssertEqual(MusicModelPreset.all.first?.id, MusicModelPreset.acestepXLTurbo8bit.id)
        for p in MusicModelPreset.all {
            XCTAssertFalse(p.id.isEmpty)
            XCTAssertFalse(p.repo.isEmpty)
            XCTAssertGreaterThan(p.approxRAMGB, 0)
            // Steps are checkpoint facts: ACE Turbo is distillation-fixed at
            // 8; Music 3 runs the reference 30-step flow-match schedule.
            XCTAssertEqual(p.fixedSteps, p.family == .acestep ? 8 : 30)
        }
        // Published converted repo → the pane offers a one-click download
        // (a `local/` prefix would show the convert-locally hint instead).
        XCTAssertFalse(MusicModelPreset.acestepXLTurbo8bit.isLocalOnly)
        XCTAssertEqual(MusicModelPreset.acestepXLTurbo8bit.repo,
                       "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit")
    }

    // MARK: - Request wire contract

    func testRequestBodyOmitsEmptyOptionalFields() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "upbeat synthwave",
            lyrics: "  ",
            vocalLanguage: "en",
            durationSeconds: 45,
            seed: 7
        )
        let body = MusicGenService.requestBody(req, modelName: "acestep-v15-xl-turbo-8bit")
        XCTAssertEqual(body["model"] as? String, "acestep-v15-xl-turbo-8bit")
        XCTAssertEqual(body["prompt"] as? String, "upbeat synthwave")
        XCTAssertEqual(body["duration_seconds"] as? Int, 45)
        XCTAssertEqual(body["seed"] as? Int, 7)
        XCTAssertEqual(body["stream"] as? Bool, true)
        // Blank/absent optionals never ride the wire (server defaults apply).
        XCTAssertNil(body["lyrics"], "whitespace-only lyrics must be omitted")
        XCTAssertNil(body["bpm"])
        XCTAssertNil(body["keyscale"])
        XCTAssertNil(body["timesignature"])
    }

    func testRequestBodyCarriesLyricsAndMetadataWhenSet() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "power ballad",
            lyrics: "la la la",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 120,
            seed: 42
        )
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["lyrics"] as? String, "la la la")
        XCTAssertEqual(body["vocal_language"] as? String, "en")
        XCTAssertEqual(body["bpm"] as? Int, 128)
        XCTAssertEqual(body["keyscale"] as? String, "F# minor")
        XCTAssertEqual(body["timesignature"] as? String, "4")
    }

    func testRequestBodyResolvesRandomSeed() {
        let req = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "jazz", seed: -1)
        let body = MusicGenService.requestBody(req, modelName: "m")
        // -1 resolves to a concrete non-negative seed (so the run is loggable).
        let seed = body["seed"] as? Int
        XCTAssertNotNil(seed)
        XCTAssertGreaterThanOrEqual(seed ?? -1, 0)
    }

    // MARK: - Output path

    func testMakeOutputPathSlugsAndCaps() {
        let path = MusicGenService.makeOutputPath(prompt: "Upbeat SYNTH-wave!!  with pads & bass, a very long prompt that keeps going and going")
        XCTAssertTrue(path.hasSuffix(".wav"))
        XCTAssertTrue(path.contains(MediaStorage.musicRoot))
        let file = (path as NSString).lastPathComponent
        // slug: lowercase, non-alphanumerics collapsed to '-', capped at 40.
        XCTAssertTrue(file.contains("upbeat-synth-wave-with-pads-bass"), file)
        let slug = file.split(separator: "_").last.map(String.init) ?? ""
        XCTAssertLessThanOrEqual(slug.replacingOccurrences(of: ".wav", with: "").count, 40)
    }

    // MARK: - Sticky settings

    func testMusicSettingsRoundTripAndLegacyDecode() throws {
        var s = MusicGenSettings()
        s.modelId = "acestep-v15-xl-turbo-8bit"
        s.durationSeconds = 90
        s.vocalLanguage = "ja"
        s.keepResident = true
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back, s)
        XCTAssertEqual(back.resolvedModel.id, MusicModelPreset.acestepXLTurbo8bit.id)

        // Migration-safe: an old/partial payload decodes to defaults.
        let legacy = try JSONDecoder().decode(MusicGenSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy, MusicGenSettings())
        XCTAssertEqual(legacy.durationSeconds, 60)

        // Unknown model id falls back to the catalog default.
        var stale = MusicGenSettings()
        stale.modelId = "acestep-v99-does-not-exist"
        XCTAssertEqual(stale.resolvedModel.id, MusicModelPreset.acestepXLTurbo8bit.id)
    }

    // MARK: - Advanced-option dropdown catalogs

    /// Every dropdown value must be accepted by the server verbatim — the
    /// whole point of dropdowns is that users can't type an invalid value.
    func testMusicOptionCatalogsAreServerValid() {
        // Language codes ⊆ the reference pipeline's VALID_LANGUAGES.
        let validLanguages: Set<String> = [
            "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
            "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
            "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
            "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
            "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
            "unknown",
        ]
        XCTAssertFalse(MusicOptions.languages.isEmpty)
        XCTAssertEqual(MusicOptions.languages.first?.code, "unknown", "Auto first")
        for (label, code) in MusicOptions.languages {
            XCTAssertFalse(label.isEmpty)
            XCTAssertTrue(validLanguages.contains(code), "invalid language code \(code)")
        }
        // BPM within the server's [30,300] gate, ascending.
        for (label, bpm) in MusicOptions.bpms {
            XCTAssertTrue((30...300).contains(bpm), "bpm \(bpm) outside server range")
            XCTAssertTrue(label.hasPrefix("\(bpm)"), "label leads with the number: \(label)")
        }
        XCTAssertEqual(MusicOptions.bpms.map(\.bpm), MusicOptions.bpms.map(\.bpm).sorted())
        // Keyscales: 24 entries in "<note>[#|b] major|minor" form.
        XCTAssertEqual(MusicOptions.keyscales.count, 24)
        for key in MusicOptions.keyscales {
            XCTAssertNotNil(key.range(of: #"^[A-G][#b]? (major|minor)$"#, options: .regularExpression), key)
        }
        // Time signatures: the server's VALID_TIME_SIGNATURES.
        XCTAssertEqual(MusicOptions.timeSignatures.map(\.value).sorted(), ["2", "3", "4", "6"])
    }

    // MARK: - Built-in example / template catalogs

    func testBuiltinStylesAndLyricsAreWellFormed() {
        XCTAssertGreaterThanOrEqual(MusicPrompt.builtinLyrics.count, 4)
        // Both engine families have their own starters — Music 3 has no
        // bpm/key/meter controls, so its examples carry that in the caption.
        for family in [MusicEngineFamily.acestep, .minimaxMusic3] {
            let styles = MusicPrompt.builtinStyles(for: family)
            XCTAssertGreaterThanOrEqual(styles.count, 6)
            for p in styles {
                XCTAssertFalse(p.title.isEmpty)
                XCTAssertGreaterThan(p.body.count, 40, "style descriptions should be descriptive: \(p.title)")
            }
            XCTAssertEqual(Set(styles.map(\.title)).count, styles.count)
        }
        for p in MusicPrompt.builtinLyrics {
            XCTAssertFalse(p.title.isEmpty)
            // Original lyric templates model the structured [Verse]/[Chorus] convention.
            XCTAssertTrue(p.body.contains("[Verse]") && p.body.contains("[Chorus]"),
                          "lyric template needs section tags: \(p.title)")
        }
        // Titles are unique (they're the dedup + display key).
        XCTAssertEqual(Set(MusicPrompt.builtinLyrics.map(\.title)).count, MusicPrompt.builtinLyrics.count)
    }

    // MARK: - Saved-prompt store (pure)

    func testAutoTitleSkipsSectionTagsAndCaps() {
        XCTAssertEqual(MusicPromptStore.autoTitle(from: "Bright modern pop with punchy drums"),
                       "Bright modern pop with punchy drums")
        // First non-tag line of structured lyrics, not the [Verse] tag.
        XCTAssertEqual(MusicPromptStore.autoTitle(from: "[Verse]\nWoke up with the sunlight"),
                       "Woke up with the sunlight")
        XCTAssertEqual(MusicPromptStore.autoTitle(from: ""), "Untitled")
        // Long lines cap at 40 chars + ellipsis.
        let long = MusicPromptStore.autoTitle(from: String(repeating: "a", count: 80))
        XCTAssertTrue(long.hasSuffix("…"))
        XCTAssertLessThanOrEqual(long.count, 41)
    }

    func testStoreAddDedupesByTitleNewestFirst() {
        var list: [MusicPrompt] = []
        list = MusicPromptStore.adding(MusicPrompt(title: "A", body: "one"), to: list)
        list = MusicPromptStore.adding(MusicPrompt(title: "B", body: "two"), to: list)
        list = MusicPromptStore.adding(MusicPrompt(title: "A", body: "one-updated"), to: list)
        XCTAssertEqual(list.map(\.title), ["A", "B"])
        XCTAssertEqual(list.first?.body, "one-updated", "same title replaces + moves to front")
        list = MusicPromptStore.removing(title: "A", from: list)
        XCTAssertEqual(list.map(\.title), ["B"])
    }

    // MARK: - Saved-prompt library (persistence)

    @MainActor
    func testLibrarySavesLoadsAndDeletesAcrossInstances() {
        let suite = "music-lib-test-\(UUID().uuidString)"
        let ud = UserDefaults(suiteName: suite)!
        defer { ud.removePersistentDomain(forName: suite) }

        let lib = MusicPromptLibrary(defaults: ud, key: "k")
        lib.saveStyle(title: "My synth", body: "warm analog synths")
        lib.saveLyrics(title: "My hook", body: "[Verse]\nla la\n[Chorus]\noh oh")
        lib.saveStyle(title: "  ", body: "blank-title auto-names")  // auto-title fallback
        lib.saveStyle(title: "empty body", body: "   ")             // ignored

        XCTAssertEqual(lib.savedStyles.count, 2, "empty-body save ignored")
        XCTAssertEqual(lib.savedLyrics.map(\.title), ["My hook"])
        XCTAssertTrue(lib.savedStyles.contains { $0.title == "blank-title auto-names" })

        // A fresh instance on the same store reloads everything.
        let reloaded = MusicPromptLibrary(defaults: ud, key: "k")
        XCTAssertEqual(reloaded.savedStyles.map(\.title), lib.savedStyles.map(\.title))
        XCTAssertEqual(reloaded.savedLyrics.first?.body, "[Verse]\nla la\n[Chorus]\noh oh")

        reloaded.deleteStyle(title: "My synth")
        let after = MusicPromptLibrary(defaults: ud, key: "k")
        XCTAssertFalse(after.savedStyles.contains { $0.title == "My synth" })
    }

    // MARK: - Settings sidecar

    func testSettingsSidecarDocumentsPromptLyricsAndParams() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "  upbeat synthwave  ",
            lyrics: "[Verse]\nla la la",
            vocalLanguage: "en",
            bpm: 120,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 30,
            seed: -1
        )
        let txt = MusicGenService.settingsText(req, resolvedSeed: 777, modelName: "acestep-8bit")
        XCTAssertTrue(txt.contains("model: acestep-8bit"))
        XCTAssertTrue(txt.contains("seed: 777"), "the RESOLVED seed, never -1")
        XCTAssertTrue(txt.contains("duration_seconds: 30"))
        XCTAssertTrue(txt.contains("bpm: 120"))
        XCTAssertTrue(txt.contains("keyscale: F# minor"))
        XCTAssertTrue(txt.contains("timesignature: 4"))
        XCTAssertTrue(txt.contains("# Style prompt\nupbeat synthwave"), "prompt trimmed")
        XCTAssertTrue(txt.contains("# Lyrics\n[Verse]\nla la la"))
        XCTAssertEqual(MusicGenService.sidecarPath(forWav: "/a/b/track.wav"), "/a/b/track.txt")
    }

    func testSettingsSidecarOmitsAutoFieldsAndMarksInstrumental() {
        let req = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "jazz trio",
                                  lyrics: "", vocalLanguage: "unknown", durationSeconds: 60, seed: 5)
        let txt = MusicGenService.settingsText(req, resolvedSeed: 5, modelName: "m")
        XCTAssertFalse(txt.contains("bpm:"))
        XCTAssertFalse(txt.contains("keyscale:"))
        XCTAssertFalse(txt.contains("vocal_language:"), "'unknown' language omitted")
        XCTAssertTrue(txt.contains("# Lyrics\n[Instrumental]"))
    }

        // MARK: - Bundle file selection (real repo listing)

    /// The PUBLISHED repo's actual file listing (huggingface.co API,
    /// 2026-07-05) through the music bundle's selection: exactly the engine
    /// files download; the model-card assets (README/LICENSE/screenshot) and
    /// .gitattributes never do.
    func testMusicSelectionAgainstPublishedRepoListing() {
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1600],
            ["path": "LICENSE", "type": "file", "size": 1546],
            ["path": "README.md", "type": "file", "size": 5900],
            ["path": "config.json", "type": "file", "size": 908],
            ["path": "model.safetensors", "type": "file", "size": 5_081_359_469],
            ["path": "music-tab.png", "type": "file", "size": 342_599],
            ["path": "text_encoder/added_tokens.json", "type": "file", "size": 700],
            ["path": "text_encoder/config.json", "type": "file", "size": 1500],
            ["path": "text_encoder/model.safetensors", "type": "file", "size": 1_191_600_000],
            ["path": "text_encoder/special_tokens_map.json", "type": "file", "size": 700],
            ["path": "text_encoder/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "text_encoder/tokenizer_config.json", "type": "file", "size": 5000],
            ["path": "vae.safetensors", "type": "file", "size": 337_483_555],
        ]
        let sel = MusicModelPreset.acestepXLTurbo8bit.bundle.components[0].selection
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0)
        XCTAssertEqual(Set(picked), [
            "config.json", "model.safetensors", "vae.safetensors",
            "text_encoder/added_tokens.json", "text_encoder/config.json",
            "text_encoder/model.safetensors", "text_encoder/special_tokens_map.json",
            "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json",
        ])
        // Every readiness marker must be in the downloaded set — otherwise a
        // fresh pull could never turn "ready".
        for marker in MusicModelPreset.acestepXLTurbo8bit.bundle.components[0].readyMarkers
        where marker.hasSuffix(".json") || marker.hasSuffix(".safetensors") {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
    }

    // MARK: - Bundle readiness

    func testMusicBundleRequiresAllComponents() throws {
        let b = MusicModelPreset.acestepXLTurbo8bit.bundle
        XCTAssertEqual(b.components.count, 1, "self-contained repo — no dependency components")
        let comp = b.components[0]

        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "music-bundle-test-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }
        let modelDir = (root as NSString).appendingPathComponent(comp.repo)
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)

        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Top-level weights alone → still not ready (text_encoder markers missing).
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("model.safetensors"), contents: Data([0, 1]))
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("vae.safetensors"), contents: Data([0, 1]))
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Full text_encoder subdir → ready.
        let te = (modelDir as NSString).appendingPathComponent("text_encoder")
        try fm.createDirectory(atPath: te, withIntermediateDirectories: true)
        for f in ["config.json", "model.safetensors", "tokenizer.json"] {
            fm.createFile(atPath: (te as NSString).appendingPathComponent(f), contents: Data([0]))
        }
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    // MARK: - MiniMax Music 3 (second music family — the field-gating tests)

    func testMusic3PresetDeclaresItsOwnKnobSet() {
        let p = MusicModelPreset.miniMaxMusic3_8bit
        XCTAssertEqual(p.family, .minimaxMusic3)
        XCTAssertFalse(p.supportsMusicalMeta, "bpm/keyscale/timesignature/vocal_language are ACE-Step-only")
        XCTAssertTrue(p.requiresLyrics, "the model is lyric-conditioned; the server 400s empty lyrics")
        XCTAssertEqual(p.durationRange, 5...360)
        XCTAssertFalse(p.isLocalOnly)
        XCTAssertEqual(p.repo, "ddalcu/MiniMax-Music3-MLX-Serve-8bit")
        XCTAssertTrue(MusicModelPreset.all.contains(p), "catalog must offer it")
        // ACE keeps its full knob set.
        let ace = MusicModelPreset.acestepXLTurbo8bit
        XCTAssertTrue(ace.supportsMusicalMeta)
        XCTAssertFalse(ace.requiresLyrics)
        XCTAssertEqual(ace.durationRange, 10...600)
    }

    func testRequestBodyDropsAcestepOnlyFieldsForMusic3() {
        // Fields left over from an ACE-Step session must NOT ride the wire —
        // the server names each one a 400 (the app gates the FIELDS, not just
        // the controls).
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "upbeat synthwave",
            lyrics: "[verse]\nneon lights",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 60,
            seed: 7
        )
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertNil(body["bpm"])
        XCTAssertNil(body["keyscale"])
        XCTAssertNil(body["timesignature"])
        XCTAssertNil(body["vocal_language"])
        XCTAssertEqual(body["lyrics"] as? String, "[verse]\nneon lights")
        XCTAssertEqual(body["prompt"] as? String, "upbeat synthwave")
    }

    func testRequestBodyClampsDurationToTheModelsRange() {
        // Sticky settings persist across model switches: ACE's 600 s against
        // music3's 360 cap (and ACE's own 10 s floor) must clamp, not 400.
        let long = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "p", lyrics: "l", durationSeconds: 600)
        XCTAssertEqual(MusicGenService.requestBody(long, modelName: "m")["duration_seconds"] as? Int, 360)
        let short = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "p", durationSeconds: 5)
        XCTAssertEqual(MusicGenService.requestBody(short, modelName: "m")["duration_seconds"] as? Int, 10)
    }

    func testSettingsSidecarOmitsAcestepFieldsForMusic3() {
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "synthwave",
            lyrics: "[verse]\nla",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 30,
            seed: 3
        )
        let text = MusicGenService.settingsText(req, resolvedSeed: 3, modelName: "m")
        XCTAssertFalse(text.contains("bpm:"))
        XCTAssertFalse(text.contains("keyscale:"))
        XCTAssertFalse(text.contains("timesignature:"))
        XCTAssertFalse(text.contains("vocal_language:"))
        XCTAssertTrue(text.contains("# Lyrics\n[verse]"))
    }

    func testMusic3FamilyResolvesFromArchitecture() {
        // A downloaded pack (any dir name) surfaces in the pane with music3's
        // gated knob set — resolution keys on the server-reported arch.
        let models = [ModelInfo(name: "ddalcu/MiniMax-Music3-MLX-Serve-8bit",
                                quantBits: 8, layers: 0, hiddenSize: 0, vocabSize: 0,
                                contextLength: 0, modelMaxTokens: 0,
                                architecture: "minimax_music3",
                                capabilities: ["audio", "music"])]
        let p = CustomMediaModels.musicPreset(for: "ddalcu/MiniMax-Music3-MLX-Serve-8bit", from: models)
        XCTAssertEqual(p?.family, .minimaxMusic3)
        XCTAssertEqual(p?.repo, "ddalcu/MiniMax-Music3-MLX-Serve-8bit")
        XCTAssertFalse(p?.supportsMusicalMeta ?? true)
    }

    func testMusic3SelectionAgainstPackListing() {
        // Simulated HF tree of the published pack: the selection must pull the
        // five safetensors + config + both tokenizer dirs and skip the junk;
        // every readiness marker must be in the downloaded set.
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1600],
            ["path": "LICENSE", "type": "file", "size": 9000],
            ["path": "README.md", "type": "file", "size": 4000],
            ["path": "config.json", "type": "file", "size": 1400],
            ["path": "language_model.safetensors", "type": "file", "size": 9_890_000_000],
            ["path": "rvq_depth_decoder.safetensors", "type": "file", "size": 710_000_000],
            ["path": "transformer.safetensors", "type": "file", "size": 2_600_000_000],
            ["path": "condition_encoder.safetensors", "type": "file", "size": 100_000_000],
            ["path": "vocoder.safetensors", "type": "file", "size": 220_000_000],
            ["path": "tokenizer/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "tokenizer/tokenizer_config.json", "type": "file", "size": 377],
            ["path": "tokenizer/chat_template.jinja", "type": "file", "size": 4168],
            ["path": "music_tokenizer/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "music_tokenizer/vocab.json", "type": "file", "size": 2_700_000],
            ["path": "music_tokenizer/merges.txt", "type": "file", "size": 1_670_000],
        ]
        let comp = MusicModelPreset.miniMaxMusic3_8bit.bundle.components[0]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: comp.selection).map(\.0)
        for f in ["config.json", "language_model.safetensors", "rvq_depth_decoder.safetensors",
                  "transformer.safetensors", "condition_encoder.safetensors", "vocoder.safetensors",
                  "tokenizer/tokenizer.json"] {
            XCTAssertTrue(picked.contains(f), "\(f) must be downloaded")
        }
        XCTAssertFalse(picked.contains("README.md"))
        for marker in comp.readyMarkers {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
    }
}
