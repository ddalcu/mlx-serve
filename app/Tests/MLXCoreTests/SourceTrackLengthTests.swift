import XCTest
@testable import MLXCore

/// A Cover or Vocal-to-BGM track is exactly as long as the clip it is made
/// from, so the pane can say how long it will be before anything is generated.
final class SourceTrackLengthTests: XCTestCase {

    func testTheLengthComesOutOfTheWavsOwnHeader() {
        // The writer the source clips are actually written with.
        let wav = AudioReference.wavData(fromMonoFloat: [Float](repeating: 0, count: 48_000 * 5),
                                         sampleRate: 48_000)
        let seconds = try? XCTUnwrap(SourceTrackLength.seconds(header: wav))
        XCTAssertEqual(seconds ?? 0, 5.0, accuracy: 0.001)
    }

    /// The samples are never read: a ten-minute clip is ~100 MB, and the
    /// header alone answers.
    func testTheHeaderAloneIsEnough() throws {
        let wav = AudioReference.wavData(fromMonoFloat: [Float](repeating: 0, count: 48_000 * 90),
                                         sampleRate: 48_000)
        let seconds = try XCTUnwrap(SourceTrackLength.seconds(header: wav.prefix(44)))
        XCTAssertEqual(seconds, 90.0, accuracy: 0.001)
    }

    /// Anything that is not a PCM WAV answers nothing rather than a guess.
    func testSomethingElseAnswersNothing() {
        XCTAssertNil(SourceTrackLength.seconds(header: Data("not a wave".utf8)))
        XCTAssertNil(SourceTrackLength.seconds(header: Data()))
    }

    /// Words, not a clock: this is prose in a hint line.
    func testTheLengthIsSpokenTheWayTheHintReadsIt() {
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 45), "45 seconds")
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 1), "1 second")
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 120), "2 minutes")
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 60), "1 minute")
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 185), "3 minutes 5 seconds")
        XCTAssertEqual(SourceTrackLength.spoken(seconds: 61.4), "1 minute 1 second")
    }
}
