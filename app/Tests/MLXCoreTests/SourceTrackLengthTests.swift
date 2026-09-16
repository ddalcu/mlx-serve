import XCTest
@testable import MLXCore

/// A Cover or Vocal-to-BGM track is exactly as long as the clip it is made
/// from, so the pane can say how long it will be before anything is generated.
final class SourceTrackLengthTests: XCTestCase {

    /// The shape a source clip is actually written in: `referenceWav` writes
    /// 48 kHz STEREO (a mono file is duplicated into two channels), so a
    /// duration that forgets the channel count is out by a factor of two.
    private func sourceClipHeader(seconds: Int) -> Data {
        AudioReference.wavData(
            fromInterleaved: [Float](repeating: 0, count: 48_000 * seconds * 2),
            channels: 2, sampleRate: 48_000)
    }

    func testTheLengthComesOutOfTheWavsOwnHeader() throws {
        let seconds = try XCTUnwrap(SourceTrackLength.seconds(header: sourceClipHeader(seconds: 5)))
        XCTAssertEqual(seconds, 5.0, accuracy: 0.001)
    }

    /// The samples are never read: a ten-minute clip is ~100 MB, and the
    /// header alone answers.
    func testTheHeaderAloneIsEnough() throws {
        let header = sourceClipHeader(seconds: 90).prefix(44)
        let seconds = try XCTUnwrap(SourceTrackLength.seconds(header: header))
        XCTAssertEqual(seconds, 90.0, accuracy: 0.001)
    }

    /// Mono answers too, at its own rate — the reader takes every term from
    /// the header rather than assuming the writer's shape.
    func testAMonoClipAtAnotherRateReadsCorrectly() throws {
        let wav = AudioReference.wavData(fromMonoFloat: [Float](repeating: 0, count: 16_000 * 3),
                                         sampleRate: 16_000)
        XCTAssertEqual(try XCTUnwrap(SourceTrackLength.seconds(header: wav)), 3.0, accuracy: 0.001)
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
