import XCTest
@testable import MLXCore

final class MusicGenPlaybackTests: XCTestCase {

    /// Starting a generation leaves the song that is playing alone; the new song takes over when it is done.
    func testPlaybackSwitchesOnlyWhenTheNewSongIsDone() {
        XCTAssertNil(MusicGenPlayback.trackToPlay(on: .running(step: 0, total: 8, message: "Loading model…"), radioOn: false))
        XCTAssertNil(MusicGenPlayback.trackToPlay(on: .failed("boom"), radioOn: false))
        XCTAssertEqual(MusicGenPlayback.trackToPlay(on: .completed(path: "/m/new.wav"), radioOn: false), "/m/new.wav")
    }

    /// The radio plays its own tracks.
    func testTheRadioKeepsItsOwnPlayback() {
        XCTAssertNil(MusicGenPlayback.trackToPlay(on: .completed(path: "/m/new.wav"), radioOn: true))
    }

    /// While a song plays the deck keeps showing it; the generation's progress is on the button.
    func testTheDeckKeepsThePlayingSongWhileGenerating() {
        let running = MusicGenService.Phase.running(step: 3, total: 8, message: "Generating…")
        XCTAssertNil(MusicGenPlayback.deckProgress(phase: running, radioOn: false, playing: true))
        XCTAssertEqual(MusicGenPlayback.deckProgress(phase: running, radioOn: false, playing: false)?.message, "Generating…")
        XCTAssertNil(MusicGenPlayback.deckProgress(phase: running, radioOn: true, playing: false))
    }

    func testButtonProgressNamesThePercentWhenTheTotalIsKnown() {
        XCTAssertEqual(MusicGenPlayback.buttonProgress(step: 3, total: 12, message: "Generating…").text, "Generating… 25%")
        XCTAssertEqual(MusicGenPlayback.buttonProgress(step: 3, total: 12, message: "Generating…").fraction, 0.25)
        let open = MusicGenPlayback.buttonProgress(step: 40, total: 0, message: "Writing lyrics…")
        XCTAssertEqual(open.text, "Writing lyrics…")
        XCTAssertNil(open.fraction)
    }
}
