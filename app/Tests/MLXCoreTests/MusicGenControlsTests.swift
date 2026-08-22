import XCTest
@testable import MLXCore

/// The music pane's newer controls: the instrumental flag (a request-level
/// rule the server names a 400 on), MiniMax Music 3's `steps`, free-entry BPM,
/// and the sticky settings that used to reset on every navigation.
final class MusicGenControlsTests: XCTestCase {

    // MARK: - Instrumental

    func testInstrumentalReplacesLyricsRatherThanRidingBesideThem() {
        // The server names `instrumental` + non-empty `lyrics` a 400 on BOTH
        // backends, so the client must never emit the pair. The flag wins:
        // it is the thing the user just ticked, and on Music 3 an omitted
        // lyrics field is the ONLY spelling of "no words" that is accepted.
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "lo-fi piano",
            lyrics: "[verse]\nleftovers from the last track",
            instrumental: true,
            durationSeconds: 30
        )
        let body = MusicGenService.requestBody(req, modelName: "music3")
        XCTAssertEqual(body["instrumental"] as? Bool, true)
        XCTAssertNil(body["lyrics"], "lyrics must not ride along with the flag")
    }

    func testInstrumentalOffSendsLyricsAndNoFlag() {
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "lo-fi piano",
            lyrics: "[verse]\nrain on the window",
            instrumental: false,
            durationSeconds: 30
        )
        let body = MusicGenService.requestBody(req, modelName: "music3")
        XCTAssertNil(body["instrumental"], "the flag is omitted, never sent false")
        XCTAssertEqual(body["lyrics"] as? String, "[verse]\nrain on the window")
    }

    func testInstrumentalSatisfiesTheLyricsRequirementOnMusic3() {
        // Music 3 is lyric-conditioned, so the pane disables Generate on empty
        // lyrics. Ticking instrumental has to LIFT that gate — otherwise the
        // checkbox is unreachable on the only model that needs it most.
        XCTAssertTrue(MusicModelPreset.miniMaxMusic3_8bit.requiresLyrics)
        XCTAssertFalse(MusicGenRequest.lyricsSatisfied(model: .miniMaxMusic3_8bit,
                                                       lyrics: "   ", instrumental: false))
        XCTAssertTrue(MusicGenRequest.lyricsSatisfied(model: .miniMaxMusic3_8bit,
                                                      lyrics: "   ", instrumental: true))
        XCTAssertTrue(MusicGenRequest.lyricsSatisfied(model: .miniMaxMusic3_8bit,
                                                      lyrics: "[verse]\nhi", instrumental: false))
        // ACE-Step never required lyrics, with or without the flag.
        XCTAssertTrue(MusicGenRequest.lyricsSatisfied(model: .acestepXLTurbo8bit,
                                                      lyrics: "", instrumental: false))
    }

    func testInstrumentalIsRecordedInTheSidecar() {
        // The `.txt` beside the WAV is what makes a track reproducible; a flag
        // that changed the output but not the sidecar is a silent setting.
        let req = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "lo-fi",
                                  instrumental: true, durationSeconds: 30)
        let text = MusicGenService.settingsText(req, resolvedSeed: 7, modelName: "music3")
        XCTAssertTrue(text.contains("instrumental: true"), text)
    }

    // MARK: - Steps (MiniMax Music 3)

    func testStepsAreSentOnlyForTheBackendThatAcceptsThem() {
        // ACE-Step Turbo is distillation-fixed at 8 steps and the server
        // silently IGNORES the field there — sending it would be a control
        // that visibly does nothing. Music 3 accepts 4-100.
        XCTAssertTrue(MusicModelPreset.miniMaxMusic3_8bit.supportsSteps)
        XCTAssertFalse(MusicModelPreset.acestepXLTurbo8bit.supportsSteps)

        let m3 = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "lo-fi",
                                 lyrics: "la", durationSeconds: 30, steps: 12)
        XCTAssertEqual(MusicGenService.requestBody(m3, modelName: "music3")["steps"] as? Int, 12)

        let ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "lo-fi",
                                  durationSeconds: 30, steps: 12)
        XCTAssertNil(MusicGenService.requestBody(ace, modelName: "ace")["steps"])
    }

    func testStepsAreClampedIntoTheServerRangeRatherThanEarningA400() {
        // Sticky settings outlive a model switch, exactly as the duration
        // clamp already handles — a stored 200 must not become a 400.
        let hi = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "x", lyrics: "la", steps: 999)
        XCTAssertEqual(MusicGenService.requestBody(hi, modelName: "m")["steps"] as? Int,
                       MusicModelPreset.miniMaxMusic3_8bit.stepsRange.upperBound)
        let lo = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "x", lyrics: "la", steps: 1)
        XCTAssertEqual(MusicGenService.requestBody(lo, modelName: "m")["steps"] as? Int,
                       MusicModelPreset.miniMaxMusic3_8bit.stepsRange.lowerBound)
        // nil = leave it to the server's own default.
        let none = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "x", lyrics: "la", steps: nil)
        XCTAssertNil(MusicGenService.requestBody(none, modelName: "m")["steps"])
    }

    // MARK: - Free-entry BPM

    func testBpmRangeMatchesWhatTheServerAccepts() {
        // The pane offered 10 fixed BPMs while the server takes 30-300, so a
        // user who wanted 92 could not ask for it. The range is the contract.
        XCTAssertEqual(MusicOptions.bpmRange, 30...300)
        for opt in MusicOptions.bpms {
            XCTAssertTrue(MusicOptions.bpmRange.contains(opt.bpm), "preset \(opt.bpm) outside the server range")
        }
    }

    func testBpmFieldReadsPastedTextAndClampsToTheRange() {
        // Same forgiving reader the seed box uses — a BPM copied out of a
        // caption ("~128 bpm") must land, and an out-of-range number clamps
        // rather than silently earning a 400 at generate time.
        XCTAssertEqual(SeedText.parse("128", in: MusicOptions.bpmRange), 128)
        XCTAssertEqual(SeedText.parse("~128 bpm", in: MusicOptions.bpmRange), 128)
        XCTAssertEqual(SeedText.parse("9999", in: MusicOptions.bpmRange), 300)
        XCTAssertEqual(SeedText.parse("1", in: MusicOptions.bpmRange), 30)
        XCTAssertNil(SeedText.parse("  ", in: MusicOptions.bpmRange))
    }

    // MARK: - Sticky settings

    func testEverySettingThePaneShowsSurvivesARoundTrip() {
        // Navigating away UNMOUNTS the pane, so anything `hydrate`/`persist`
        // does not carry is lost on the next visit — not just across launches.
        var s = MusicGenSettings()
        s.modelId = MusicModelPreset.miniMaxMusic3_8bit.id
        s.durationSeconds = 95
        s.vocalLanguage = "ja"
        s.bpm = 92
        s.keyscale = "C major"
        s.timesignature = "3"
        s.seed = 4242
        s.steps = 18
        s.instrumental = true
        s.showAdvanced = true
        s.keepResident = true

        let data = try! JSONEncoder().encode(s)
        let back = try! JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back, s)
    }

    func testSettingsDecodeFromAnOlderBlobThatLacksTheNewKeys() {
        // A shipped install already holds a 4-key blob; a decode that throws
        // would silently reset the pane to defaults for every existing user.
        let old = #"{"modelId":"music3-8bit","durationSeconds":120,"vocalLanguage":"en","keepResident":true}"#
        let s = try! JSONDecoder().decode(MusicGenSettings.self, from: Data(old.utf8))
        XCTAssertEqual(s.durationSeconds, 120)
        XCTAssertTrue(s.keepResident)
        XCTAssertNil(s.bpm)
        XCTAssertEqual(s.seed, -1)
        XCTAssertFalse(s.instrumental)
    }

    // MARK: - The audio pane's sub-mode

    func testAudioTabIsRawRepresentableForAppStorageAndDefaultsToVoice() {
        // The left menu's "Audio & Music" row must reopen on the tab you left
        // it on; @AppStorage needs a stable raw value to do that, and the
        // stored strings are a persistence contract — renaming the display
        // text must not silently reset everyone to Voice.
        XCTAssertEqual(AudioGenView.Tab.voice.rawValue, "Voice")
        XCTAssertEqual(AudioGenView.Tab.music.rawValue, "Music")
        XCTAssertEqual(AudioGenView.Tab(rawValue: "Music"), .music)
        XCTAssertNil(AudioGenView.Tab(rawValue: "music"))
        XCTAssertEqual(AudioGenView.Tab.allCases.first, .voice)
    }
}

/// Fallout from testing the pane by hand: a peer's music model offering itself
/// as a TTS voice, two different downloads wearing one size label, and a
/// section-tag vocabulary the UI never named.
final class AudioPaneGapTests: XCTestCase {

    func testAPeersMusicModelIsNotOfferedAsASpeechVoice() {
        // The server advertises a music backend ADDITIVELY as ["audio","music"]
        // (src/server.zig, ready and stub paths alike), so "audio" is the
        // MODALITY and not a speech tag. Asking for "audio" in the Voice pane
        // put a peer's ACE-Step / MiniMax Music 3 in the voice picker.
        let music = APIClient.parseModelInfo([
            "id": "MiniMax-Music3-MLX-Serve-8bit@studio", "lan_peer": "studio",
            "capabilities": ["audio", "music"],
        ])
        let tts = APIClient.parseModelInfo([
            "id": "Qwen3-TTS-12Hz-0.6B@studio", "lan_peer": "studio",
            "capabilities": ["audio"],
        ])

        XCTAssertFalse(music.lanAdvertises("speech"), "a music peer must not be a voice")
        XCTAssertTrue(tts.lanAdvertises("speech"))
        // The Music pane's own ask was always exact — no TTS backend says "music".
        XCTAssertTrue(music.lanAdvertises("music"))
        XCTAssertFalse(tts.lanAdvertises("music"))
        // The raw modality still matches both; only "speech" discriminates.
        XCTAssertTrue(music.lanAdvertises("audio"))
        XCTAssertTrue(tts.lanAdvertises("audio"))
    }

    func testASpeechCapabilityIsStillLocalOnlyForNonPeers() {
        // Unchanged contract: this predicate is about LAN entries only.
        let local = APIClient.parseModelInfo([
            "id": "Qwen3-TTS-12Hz-0.6B", "capabilities": ["audio"],
        ])
        XCTAssertFalse(local.lanAdvertises("speech"))
        XCTAssertFalse(local.lanAdvertises("audio"))
    }

    func testTwoDifferentDownloadsDoNotWearTheSameSizeLabel() {
        // `%.0f` is round-half-to-EVEN, so 2.0 GB and 2.5 GB both printed
        // "~2 GB" — and the larger one was understated, which is the
        // misleading direction for a download prompt.
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 2.0), "~2 GB")
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 2.5), "~2.5 GB")
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 4.5), "~4.5 GB")
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 3.1), "~3 GB")
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 13.6), "~13.5 GB")
        // Sub-1 GB keeps its decimal — "~0 GB" for Kokoro would be a lie.
        XCTAssertEqual(MediaBundle.sizeLabel(forGB: 0.35), "~0.4 GB")
    }

    func testSectionTagsAreTheOnesTheModelCardLists() {
        // The helper text said "like [verse] or [chorus]" and left the other
        // seven undiscoverable. These are verbatim from MiniMax-Music3's card.
        XCTAssertEqual(MusicOptions.sectionTags,
                       ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[post-chorus]",
                        "[bridge]", "[instrumental]", "[solo]", "[outro]"])
        XCTAssertTrue(MusicOptions.sectionTagHint.contains("[bridge]"))
        // Lowercase, because the engines lowercase tags before the model sees
        // them — showing "[Verse]" would imply a distinction that is not there.
        for tag in MusicOptions.sectionTags {
            XCTAssertEqual(tag, tag.lowercased())
        }
    }
}

/// Tempo and key are supported by BOTH engines — as conditioning fields on
/// ACE-Step and as caption text on Music 3 (Global Metadata on MiniMax's model
/// card; its own example caption reads "BPM: 96. Key: C major."). The pane hid
/// them on Music 3 along with the two genuinely unsupported knobs.
final class MusicTempoKeyTests: XCTestCase {

    func testTempoAndKeyGoToBothEngines() {
        for preset in [MusicModelPreset.acestepXLTurbo8bit, .miniMaxMusic3_8bit] {
            XCTAssertTrue(preset.supportsTempoAndKey, "\(preset.name) should take tempo/key")
            let req = MusicGenRequest(model: preset, prompt: "acoustic pop", lyrics: "la",
                                      bpm: 96, keyscale: "C major", durationSeconds: 30)
            let body = MusicGenService.requestBody(req, modelName: "m")
            XCTAssertEqual(body["bpm"] as? Int, 96, "\(preset.name)")
            XCTAssertEqual(body["keyscale"] as? String, "C major", "\(preset.name)")
        }
    }

    func testTheTwoUndocumentedKnobsStayAceStepOnly() {
        // MiniMax's card documents no equivalent for meter or vocal language,
        // and the server still names each a 400 — so the FIELDS are gated, not
        // just the controls: values linger in @State across a model switch.
        let m3 = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "pop", lyrics: "la",
                                 vocalLanguage: "ja", timesignature: "3", durationSeconds: 30)
        let body = MusicGenService.requestBody(m3, modelName: "m")
        XCTAssertNil(body["vocal_language"])
        XCTAssertNil(body["timesignature"])

        let ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "pop",
                                  vocalLanguage: "ja", timesignature: "3", durationSeconds: 30)
        let aceBody = MusicGenService.requestBody(ace, modelName: "m")
        XCTAssertEqual(aceBody["vocal_language"] as? String, "ja")
        XCTAssertEqual(aceBody["timesignature"] as? String, "3")
    }

    func testTheSidecarRecordsTempoAndKeyOnBothEngines() {
        for preset in [MusicModelPreset.acestepXLTurbo8bit, .miniMaxMusic3_8bit] {
            let req = MusicGenRequest(model: preset, prompt: "pop", lyrics: "la",
                                      bpm: 96, keyscale: "C major", durationSeconds: 30)
            let text = MusicGenService.settingsText(req, resolvedSeed: 1, modelName: "m")
            XCTAssertTrue(text.contains("bpm: 96"), "\(preset.name): \(text)")
            XCTAssertTrue(text.contains("keyscale: C major"), "\(preset.name): \(text)")
        }
    }
}

extension MusicTempoKeyTests {
    func testKeyLabelsCarryTheirConventionalCharacter() {
        // The BPM menu's labels carry a genre anchor; the key menu was 24 bare
        // note names, which asks a non-musician to pick blind.
        XCTAssertEqual(MusicOptions.keyLabel("C major"), "C major — open")
        XCTAssertEqual(MusicOptions.keyLabel("A minor"), "A minor — plain sad")
        // A menu clips rather than wraps, so the labels have a length budget:
        // moods are one word (two only where one will not do).
        for (key, mood) in MusicOptions.keyMoods {
            XCTAssertLessThanOrEqual(mood.split(separator: " ").count, 2, "\(key) mood is too long")
        }
        // Keys without a conventional association show bare rather than getting
        // one invented for them.
        XCTAssertEqual(MusicOptions.keyLabel("C# major"), "C# major")
        // Every mood key must be a real catalogue entry, or the label silently
        // never fires.
        for key in MusicOptions.keyMoods.keys {
            XCTAssertTrue(MusicOptions.keyscales.contains(key), "\(key) is not in the catalogue")
        }
        // The wire value is the bare key — a label must never reach the server.
        for key in MusicOptions.keyscales {
            XCTAssertTrue(MusicOptions.keyLabel(key).hasPrefix(key))
        }
    }
}
