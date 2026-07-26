import XCTest
@testable import MLXCore

@MainActor
final class VoicePreviewerTests: XCTestCase {

    /// Synth seam that parks until the test releases it, so "a second preview
    /// starts while the first is still synthesizing" is testable deterministically.
    ///
    /// A LATCH, not a one-shot signal: once opened it stays open, so a waiter
    /// that has not parked YET still passes straight through. The naive version
    /// dropped a `releaseAll()` that landed before the task reached `wait()`,
    /// and the test then blocked forever on `await inFlight.value` — which hung
    /// the entire 1727-test suite rather than failing, so it read as "the build
    /// is slow" instead of "a test is broken".
    private final class Gate {
        private var conts: [CheckedContinuation<Void, Never>] = []
        private var open = false
        func wait() async {
            if open { return }
            await withCheckedContinuation { conts.append($0) }
        }
        func releaseAll() {
            open = true
            let c = conts
            conts = []
            c.forEach { $0.resume() }
        }
    }

    func testPreviewSynthesizesTheVoiceAndPlaysIt() async {
        var synthCalls: [(String, String)] = []
        var played: [Data] = []
        let p = VoicePreviewer(
            synthesize: { text, voice in synthCalls.append((text, voice)); return Data("audio".utf8) },
            play: { played.append($0) })

        p.preview("af_bella")
        await settle(p)

        XCTAssertEqual(synthCalls.count, 1)
        XCTAssertEqual(synthCalls.first?.1, "af_bella")
        XCTAssertTrue(synthCalls.first?.0.contains("Bella") == true,
                      "the preview text should name the voice")
        XCTAssertEqual(played.count, 1)
        XCTAssertNil(p.active, "idle once playback finishes")
        XCTAssertNil(p.error)
    }

    func testActiveTracksTheClickedSpecSoTheRightRowShowsTheSpinner() async {
        let gate = Gate()
        let p = VoicePreviewer(
            synthesize: { _, _ in await gate.wait(); return Data() },
            play: { _ in })

        p.preview("am_puck")
        XCTAssertEqual(p.active, "am_puck")
        XCTAssertTrue(p.isPreviewing("am_puck"))
        XCTAssertFalse(p.isPreviewing("af_bella"))

        gate.releaseAll()
        await settle(p)
        XCTAssertNil(p.active)
    }

    func testASecondPreviewSUPERSEDESTheFirstRatherThanQueueingIt() async {
        // Clicking through voices must play the LAST one, not all of them in a
        // row — a queue here means the picker talks over itself for ten seconds.
        let gate = Gate()
        var played: [String] = []
        let p = VoicePreviewer(
            synthesize: { _, voice in await gate.wait(); return Data(voice.utf8) },
            play: { played.append(String(decoding: $0, as: UTF8.self)) })

        p.preview("af_bella")
        p.preview("am_michael")
        XCTAssertEqual(p.active, "am_michael", "the newest click owns the UI")

        gate.releaseAll()
        await settle(p)

        XCTAssertEqual(played, ["am_michael"],
                       "the superseded preview must be dropped, not played after")
    }

    func testPreviewCutsWhateverIsCurrentlySounding() async {
        var stops = 0
        let p = VoicePreviewer(
            synthesize: { _, _ in Data() },
            play: { _ in },
            stopPlayback: { stops += 1 })

        p.preview("af_bella")
        await settle(p)
        p.preview("af_sky")
        await settle(p)
        XCTAssertEqual(stops, 2, "each preview stops the previous playback first")
    }

    func testFailedSynthesisSurfacesAnErrorAndClearsActive() async {
        let p = VoicePreviewer(synthesize: { _, _ in nil }, play: { _ in })
        p.preview("af_bella")
        await settle(p)
        XCTAssertNotNil(p.error, "a silent no-op would read as a broken button")
        XCTAssertNil(p.active)
    }

    func testASuccessfulPreviewClearsAPreviousError() async {
        var fail = true
        let p = VoicePreviewer(synthesize: { _, _ in fail ? nil : Data() }, play: { _ in })
        p.preview("af_bella")
        await settle(p)
        XCTAssertNotNil(p.error)

        fail = false
        p.preview("af_bella")
        await settle(p)
        XCTAssertNil(p.error)
    }

    func testBlankSpecIsIgnored() async {
        var calls = 0
        let p = VoicePreviewer(synthesize: { _, _ in calls += 1; return Data() }, play: { _ in })
        p.preview("")
        p.preview("   ")
        await settle(p)
        XCTAssertEqual(calls, 0)
        XCTAssertNil(p.active)
    }

    func testStopAbandonsInFlightWorkWithoutPlayingIt() async {
        let gate = Gate()
        var played = 0
        let p = VoicePreviewer(
            synthesize: { _, _ in await gate.wait(); return Data() },
            play: { _ in played += 1 })

        p.preview("af_bella")
        p.stop()
        XCTAssertNil(p.active)

        gate.releaseAll()
        await settle(p)
        XCTAssertEqual(played, 0, "work abandoned by stop() must not surface later")
    }

    func testUnattachedPreviewerReportsUnavailableRatherThanFailingSilently() async {
        // A view holds this as @StateObject and attaches in .onAppear; if that
        // wiring is ever dropped the button must SAY so, not do nothing.
        let p = VoicePreviewer()
        p.preview("af_bella")
        await settle(p)
        XCTAssertNotNil(p.error)
        XCTAssertNil(p.active)
    }

    func testBlendSpecIsPassedThroughVerbatim() async {
        var seen: String?
        let p = VoicePreviewer(
            synthesize: { _, voice in seen = voice; return Data() },
            play: { _ in })
        p.preview("af_bella,af_sky")
        await settle(p)
        XCTAssertEqual(seen, "af_bella,af_sky", "the server does the blending")
    }

    /// Await the in-flight preview, then let any SUPERSEDED task unwind. Both
    /// halves matter: the second is what proves a dropped preview never
    /// surfaces late.
    private func settle(_ p: VoicePreviewer? = nil) async {
        await p?.inFlight?.value
        for _ in 0..<8 { await Task.yield() }
    }
}
