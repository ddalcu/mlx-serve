import XCTest
@testable import MLXCore

/// AI Radio keeps generating tracks from ONE theme. The risk is sameness: every
/// request must push the model somewhere new, and the queue must only ever ask
/// for a track when nothing is buffered.
final class AIRadioTests: XCTestCase {

    // MARK: direction (the variation nudge)

    func testConsecutiveTracksGetDifferentDirections() {
        let directions = (0..<12).map { RadioDirection.nudge(forTrack: $0, salt: 3) }
        for (a, b) in zip(directions, directions.dropFirst()) { XCTAssertNotEqual(a, b) }
        XCTAssertGreaterThan(Set(directions).count, 8, "a dozen tracks must not recycle a handful of nudges")
    }

    func testDirectionIsDeterministicAndSaltedPerStation() {
        XCTAssertEqual(RadioDirection.nudge(forTrack: 4, salt: 9), RadioDirection.nudge(forTrack: 4, salt: 9))
        XCTAssertNotEqual((0..<6).map { RadioDirection.nudge(forTrack: $0, salt: 1) },
                          (0..<6).map { RadioDirection.nudge(forTrack: $0, salt: 2) })
    }

    func testNudgeNamesTwoDifferentAspectsOfTheTrack() {
        let nudge = RadioDirection.nudge(forTrack: 0, salt: 0)
        XCTAssertEqual(nudge.components(separatedBy: "; ").count, 2)
    }

    // MARK: the LLM request

    func testRadioRequestCarriesThemeHistoryAndNudge() {
        let r = MusicPromptRewriter.radioRequest(theme: "mellow lo-fi for coding",
                                                 recent: ["slow piano lo-fi", "dusty vinyl beat"],
                                                 nudge: "faster tempo; brighter mood",
                                                 family: .acestep, instrumental: true)
        XCTAssertTrue(r.user.contains("mellow lo-fi for coding"))
        XCTAssertTrue(r.user.contains("slow piano lo-fi"))
        XCTAssertTrue(r.user.contains("dusty vinyl beat"))
        XCTAssertTrue(r.user.contains("faster tempo; brighter mood"))
        XCTAssertTrue(r.user.lowercased().contains("instrumental"))
    }

    func testRadioSystemPromptDemandsVariationAndKeepsTheFamilyFormat() {
        let ace = MusicPromptRewriter.radioRequest(theme: "x", recent: [], nudge: "n", family: .acestep, instrumental: true)
        XCTAssertTrue(ace.system.contains("different"))
        XCTAssertTrue(ace.system.contains(MusicPrompt.builtinStyles[0].body))
        XCTAssertFalse(ace.system.contains("Global Metadata"))
        let m3 = MusicPromptRewriter.radioRequest(theme: "x", recent: [], nudge: "n", family: .minimaxMusic3, instrumental: true)
        XCTAssertTrue(m3.system.contains("Global Metadata"))
    }

    func testRecentHistoryInTheRequestIsCapped() {
        let many = (1...20).map { "track number \($0)" }
        let r = MusicPromptRewriter.radioRequest(theme: "x", recent: many, nudge: "n", family: .acestep, instrumental: true)
        XCTAssertTrue(r.user.contains("track number 20"))
        XCTAssertFalse(r.user.contains("track number 1\n"), "only the last few prompts are shown")
    }

    func testFallbackPromptKeepsTheThemeAndTheNudgeWithoutAChatModel() {
        let p = RadioDirection.fallbackPrompt(theme: "synthwave night drive", nudge: "slower tempo; piano lead")
        XCTAssertTrue(p.contains("synthwave night drive"))
        XCTAssertTrue(p.contains("slower tempo"))
    }

    // MARK: the buffer

    func testQueueAsksForATrackOnlyWhenNothingIsBuffered() {
        var q = RadioQueue()
        XCTAssertTrue(q.needsTrack)
        q.push(prompt: "a", path: "/a.wav")
        XCTAssertFalse(q.needsTrack)
        XCTAssertEqual(q.pop(), "/a.wav")
        XCTAssertTrue(q.needsTrack)
        XCTAssertNil(q.pop())
    }

    func testQueueKeepsTheLastPromptsOnly() {
        var q = RadioQueue()
        for i in 1...10 { q.push(prompt: "p\(i)", path: "/\(i).wav") }
        XCTAssertEqual(q.recentPrompts.count, RadioQueue.historyLimit)
        XCTAssertEqual(q.recentPrompts.last, "p10")
    }

    func testQueueIsFirstInFirstOut() {
        var q = RadioQueue()
        q.push(prompt: "a", path: "/a.wav"); q.push(prompt: "b", path: "/b.wav")
        XCTAssertEqual(q.pop(), "/a.wav")
        XCTAssertEqual(q.pop(), "/b.wav")
    }

    // MARK: the station's generation settings

    func testStationCapsTrackLengthAtTwoMinutes() {
        var long = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "")
        long.durationSeconds = 300
        XCTAssertEqual(AIRadio.stationRequest(from: long).durationSeconds, AIRadio.maxTrackSeconds)
        XCTAssertEqual(AIRadio.maxTrackSeconds, 120)
        var short = long
        short.durationSeconds = 45
        XCTAssertEqual(AIRadio.stationRequest(from: short).durationSeconds, 45)
    }

    func testStationLowersStepsOnlyWhereTheModelTakesThem() {
        var music3 = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "")
        music3.steps = 60
        XCTAssertEqual(AIRadio.stationRequest(from: music3).steps, AIRadio.maxSteps)
        music3.steps = nil   // the model's own default (30) is also too slow for a station
        XCTAssertEqual(AIRadio.stationRequest(from: music3).steps, AIRadio.maxSteps)
        music3.steps = 8
        XCTAssertEqual(AIRadio.stationRequest(from: music3).steps, 8)
        // ACE-Step Turbo is fixed at 8 and ignores the field: nothing to lower.
        XCTAssertNil(AIRadio.stationRequest(from: MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "")).steps)
    }

    func testStationKeepsTheModelLoadedAndLetsEachPromptDecideTempoAndKey() {
        var t = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "x")
        t.bpm = 90; t.keyscale = "C major"; t.timesignature = "4"; t.task = .cover; t.srcAudioPath = "/a.wav"
        let r = AIRadio.stationRequest(from: t)
        XCTAssertTrue(r.keepResident)
        XCTAssertNil(r.bpm)
        XCTAssertEqual(r.keyscale, "")
        XCTAssertEqual(r.timesignature, "")
        XCTAssertEqual(r.task, .text2music)
        XCTAssertNil(r.srcAudioPath)
    }

    // MARK: language

    func testRadioStylePromptKeepsTheLanguageAndCultureTheThemeNames() {
        let r = MusicPromptRewriter.radioRequest(theme: "romanian manele", recent: [], nudge: "n",
                                                 family: .yue2, instrumental: false)
        XCTAssertTrue(r.system.contains("language"))
    }

    func testRadioLyricsAreWrittenInTheLanguageTheThemeNames() {
        let r = MusicPromptRewriter.radioLyricsRequest(theme: "romanian manele about summer", style: "manele, accordion",
                                                       nudge: "a faster tempo", recent: ["old style"],
                                                       family: .yue2, fallbackLanguage: "en")
        XCTAssertTrue(r.user.contains("romanian manele about summer"))
        XCTAssertTrue(r.user.contains("language the station theme names"))
        XCTAssertTrue(r.user.contains("English"), "the pane's language is only the fallback")
        XCTAssertTrue(r.user.contains("manele, accordion"))
        XCTAssertTrue(r.system.contains("[verse]") || r.system.contains("[Verse"))
    }

    // MARK: the pixel font never shows a stray "?"

    func testPixelTextFoldsWhatTheFontLacksInsteadOfDrawingQuestionMarks() {
        XCTAssertEqual(PixelFont.printable("Composing…"), "COMPOSING...")
        XCTAssertEqual(PixelFont.printable("Mănăstire șlagăr"), "MANASTIRE SLAGAR")
        XCTAssertEqual(PixelFont.printable("a – b — c"), "A - B - C")
        XCTAssertEqual(PixelFont.printable("it’s “live”"), "IT'S \"LIVE\"")
        XCTAssertFalse(PixelFont.printable("Writing the next track…").contains("?"))
        XCTAssertEqual(Px.textWidth("Composing…"), Px.textWidth("COMPOSING..."))
    }

    // MARK: the radio strip as a progress bar

    func testTheStripFillsInWholePixelsAndClamps() {
        XCTAssertEqual(RadioStrip.fillWidth(progress: 0.5, width: 100), 50)
        XCTAssertEqual(RadioStrip.fillWidth(progress: 0.333, width: 100), 33)
        XCTAssertEqual(RadioStrip.fillWidth(progress: 1.7, width: 100), 100)
        XCTAssertEqual(RadioStrip.fillWidth(progress: -1, width: 100), 0)
        XCTAssertEqual(RadioStrip.fillWidth(progress: nil, width: 100), 0, "no number yet: no fill, the block sweeps instead")
    }

    func testTheWaitingBlockBouncesInsideTheStrip() {
        let width: CGFloat = 100, block = RadioStrip.blockWidth
        let travel = width - block
        let times = stride(from: 0.0, through: 20.0, by: 0.37).map { $0 }
        for t in times {
            let x = RadioStrip.blockOffset(time: t, width: width)
            XCTAssertGreaterThanOrEqual(x, 0)
            XCTAssertLessThanOrEqual(x, travel)
            XCTAssertEqual(x, x.rounded())
        }
        XCTAssertEqual(RadioStrip.blockOffset(time: 0, width: width), 0)
        let atEnd = RadioStrip.blockOffset(time: Double(travel) / RadioStrip.blockSpeed, width: width)
        XCTAssertEqual(atEnd, travel, accuracy: 1)
        let back = RadioStrip.blockOffset(time: Double(travel) / RadioStrip.blockSpeed + 0.5, width: width)
        XCTAssertLessThan(back, atEnd, "it turns round at the far edge")
    }
}
