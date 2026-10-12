import XCTest
@testable import MLXCore

/// What the Modern history says about a clip: its title, model, date and day group.
final class TrackMetaTests: XCTestCase {

    func testTheTitleIsThePromptsFirstSentence() {
        XCTAssertEqual(TrackMeta.title(prompt: "funk meets orchestra in a groovy pleasing drive. Brass stabs."),
                       "Funk meets orchestra in a groovy pleasing drive")
        XCTAssertEqual(TrackMeta.title(prompt: "upbeat synthwave with driving bass\nand dreamy pads"),
                       "Upbeat synthwave with driving bass")
    }

    /// A template prompt opens with a header line and labelled lines.
    func testHeadersAndLabelsAreSkipped() {
        let prompt = """
        Global Metadata
        Basic Attributes: bpm is 118. key is F, and scale is minor. Retro Synthwave / Outrun.
        Global Emotional Progression: Opens cool and mechanical.
        """
        XCTAssertEqual(TrackMeta.title(prompt: prompt), "Retro Synthwave / Outrun", "Music 3's genre closes Basic Attributes")
        XCTAssertEqual(TrackMeta.title(prompt: "Mood: dark ambient drone. Slow."), "Dark ambient drone")
        XCTAssertEqual(TrackMeta.title(prompt: "Global Metadata"), "Global Metadata", "a lone short line is the title")
    }

    func testALongTitleIsCutAtAWord() throws {
        let title = try XCTUnwrap(TrackMeta.title(prompt: String(repeating: "lush orchestral strings ", count: 6)))
        XCTAssertLessThanOrEqual(title.count, TrackMeta.titleLimit + 1)
        XCTAssertTrue(title.hasSuffix("strings…") || title.hasSuffix("lush…") || title.hasSuffix("orchestral…"), title)
        XCTAssertNil(TrackMeta.title(prompt: "  \n "))
    }

    /// No sidecar: the slug in the file name, with the word the 40-character cut split dropped.
    func testTheFileNameIsTheFallbackTitle() {
        XCTAssertEqual(TrackMeta.title(stem: "2026-08-23_18-05-00_rock-fusion-deep-bass-line-rock-percussi"),
                       "Rock fusion deep bass line rock…")
        XCTAssertEqual(TrackMeta.title(stem: "2026-08-23_18-05-00_short-jingle"), "Short jingle")
        XCTAssertEqual(TrackMeta.title(stem: "my take"), "My take")
    }

    func testTheSidecarGivesTitleAndModel() {
        let music = """
        model: ddalcu/YuE2-3B-MLX-Serve-8bit
        duration_seconds: 65
        seed: 7

        # Style prompt
        dreamy indie pop, female vocals

        # Lyrics
        [verse]
        """
        let meta = TrackMeta.parse(sidecar: music, path: "/m/2026-10-11_15-42-07_dreamy-indie-pop-female-vocals.wav")
        XCTAssertEqual(meta.title, "Dreamy indie pop, female vocals")
        XCTAssertEqual(meta.model, "YuE2-3B-MLX-Serve-8bit")

        let voice = "model: Qwen3-TTS\nspeed: 1.00\n\n# Text\nHello there. How are you?\n"
        XCTAssertEqual(TrackMeta.parse(sidecar: voice, path: "/v/x.wav").title, "Hello there")
        XCTAssertEqual(TrackMeta.parse(sidecar: nil, path: "/v/2026-10-11_15-42-07_hello-there.wav").model, "")
    }

    /// The stamp is written with the user's calendar and read back with it (a Buddhist year is 2569).
    func testTheDateIsReadWithTheWritersCalendar() throws {
        var buddhist = Calendar(identifier: .buddhist)
        buddhist.timeZone = TimeZone(identifier: "UTC")!
        var gregorian = Calendar(identifier: .gregorian)
        gregorian.timeZone = TimeZone(identifier: "UTC")!
        let a = try XCTUnwrap(TrackMeta.date(stem: "2569-10-11_15-42-07_x", calendar: buddhist))
        let b = try XCTUnwrap(TrackMeta.date(stem: "2026-10-11_15-42-07_x", calendar: gregorian))
        XCTAssertEqual(a, b)
        XCTAssertNil(TrackMeta.date(stem: "take one", calendar: gregorian))
    }

    func testTheSameClipAlwaysGetsTheSameCover() {
        XCTAssertEqual(TrackMeta.coverHues("/a/one.wav").0, TrackMeta.coverHues("/b/one.wav").0)
        XCTAssertNotEqual(TrackMeta.coverHues("/a/one.wav").0, TrackMeta.coverHues("/a/two.wav").0)
    }

    func testTheHistoryGroupsByDay() {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let now = cal.date(from: DateComponents(year: 2026, month: 10, day: 11, hour: 18))!
        let dates: [String: Date] = [
            "a": cal.date(from: DateComponents(year: 2026, month: 10, day: 11, hour: 15))!,
            "b": cal.date(from: DateComponents(year: 2026, month: 10, day: 11, hour: 9))!,
            "c": cal.date(from: DateComponents(year: 2026, month: 10, day: 10, hour: 23))!,
            "d": cal.date(from: DateComponents(year: 2026, month: 9, day: 2, hour: 23))!,
        ]
        let days = TrackDay.group(["a", "b", "c", "d", "e"], date: { dates[$0] }, now: now, calendar: cal)
        XCTAssertEqual(days.map(\.paths), [["a", "b"], ["c"], ["d"], ["e"]])
        XCTAssertEqual(days[0].label, "Today")
        XCTAssertEqual(days[1].label, "Yesterday")
        XCTAssertEqual(days[3].label, "Earlier")
    }
}
