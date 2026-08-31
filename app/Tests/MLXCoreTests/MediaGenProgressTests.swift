import XCTest
@testable import MLXCore

/// The progress meter behind an in-chat media generation. It exists because
/// these block chat decode on a single GPU: a window that sits there with no
/// feedback for two minutes reads as a hang, not as work.
final class MediaGenProgressTests: XCTestCase {

    private func progress(kind: MediaKind = .image, step: Int, total: Int,
                          message: String = "Generating") -> MediaGenProgress {
        MediaGenProgress(kind: kind, step: step, total: total, message: message,
                         startedAt: Date(timeIntervalSince1970: 0))
    }

    // MARK: - fraction

    func testFractionIsNilWhenTheServerReportsNoTotal() {
        // TTS length is model-determined: the server sends a growing frame count
        // and total 0. A bar drawn at 0/0 would sit empty and look stuck, so the
        // caller gets nil and renders an indeterminate one.
        XCTAssertNil(progress(step: 41, total: 0).fraction)
        XCTAssertNil(progress(step: 0, total: 0).fraction)
        XCTAssertNil(progress(step: 3, total: -1).fraction)
    }

    func testFractionTracksStepsAndNeverLeavesTheUnitRange() {
        XCTAssertEqual(progress(step: 2, total: 8).fraction!, 0.25, accuracy: 0.0001)
        XCTAssertEqual(progress(step: 0, total: 8).fraction!, 0.0, accuracy: 0.0001)
        // A server that overshoots its own total must not push the bar past full.
        XCTAssertEqual(progress(step: 99, total: 8).fraction!, 1.0, accuracy: 0.0001)
        XCTAssertEqual(progress(step: -3, total: 8).fraction!, 0.0, accuracy: 0.0001)
    }

    // MARK: - text

    func testDetailTextCountsStepsOnlyWhenThereAreStepsToCount() {
        XCTAssertEqual(progress(step: 3, total: 8, message: "Composing").detailText, "Composing — step 3 of 8")
        XCTAssertEqual(progress(step: 0, total: 0, message: "Loading model").detailText, "Loading model")
    }

    func testEveryKindHasItsOwnTitleAndIcon() {
        var titles = Set<String>()
        for kind in [MediaKind.image, .speech, .music, .video] {
            XCTAssertFalse(kind.progressTitle.isEmpty)
            XCTAssertFalse(kind.icon.isEmpty)
            titles.insert(kind.progressTitle)
        }
        XCTAssertEqual(titles.count, 4, "a shared title makes two different jobs look like one")
    }

    func testElapsedTextIsMinutesAndSecondsAndNeverGoesNegative() {
        let p = progress(step: 1, total: 8)
        XCTAssertEqual(p.elapsedText(now: Date(timeIntervalSince1970: 7)), "0:07")
        XCTAssertEqual(p.elapsedText(now: Date(timeIntervalSince1970: 75)), "1:15")
        XCTAssertEqual(p.elapsedText(now: Date(timeIntervalSince1970: 605)), "10:05")
        // Clock skew / a startedAt in the future must read 0:00, not "-0:03".
        XCTAssertEqual(p.elapsedText(now: Date(timeIntervalSince1970: -3)), "0:00")
    }

    // MARK: - SSE mapping

    func testProgressEventsCarryStepTotalAndStage() {
        let ev = MediaSSE.classify(["type": "progress", "step": 3, "total": 8, "stage": "diffuse"])
        XCTAssertEqual(ev, .progress(step: 3, total: 8, stage: "diffuse"))
    }

    func testProgressEventsSurviveMissingFields() {
        // The four backends don't all send the same keys — a missing total is
        // the indeterminate case, a missing stage is just "Generating".
        XCTAssertEqual(MediaSSE.classify(["type": "progress", "step": 12]),
                       .progress(step: 12, total: 0, stage: "Generating"))
    }

    func testCompleteAndErrorAreDistinctAndEverythingElseIsIgnored() {
        XCTAssertEqual(MediaSSE.classify(["type": "complete", "data": "AAAA"]), .complete)
        XCTAssertEqual(MediaSSE.classify(["type": "error", "message": "out of memory"]),
                       .failed("out of memory"))
        // An error with no message still has to be an error, with words in it.
        guard case .failed(let m) = MediaSSE.classify(["type": "error"]) else {
            return XCTFail("a typed error event must not be ignored")
        }
        XCTAssertFalse(m.isEmpty)
        XCTAssertEqual(MediaSSE.classify(["type": "keepalive"]), .ignored)
        XCTAssertEqual(MediaSSE.classify([:]), .ignored)
    }

    func testPreviewJPEGReadsOptInProgressPayload() {
        // A 1x1 JPEG SOI stub, base64. classify() must stay Equatable without
        // the bytes — the image is a sidecar.
        let b64 = Data([0xFF, 0xD8, 0xFF, 0x00]).base64EncodedString()
        let ev: [String: Any] = ["type": "progress", "step": 8, "total": 30,
                                 "preview": b64, "mime": "image/jpeg", "w": 256, "h": 144]
        XCTAssertEqual(MediaSSE.classify(ev), .progress(step: 8, total: 30, stage: "Generating"))
        XCTAssertEqual(MediaSSE.previewJPEG(ev), Data([0xFF, 0xD8, 0xFF, 0x00]))
        XCTAssertNil(MediaSSE.previewJPEG(["type": "progress", "step": 1]))
        // `preview` is the ONE key the server emits. Reading invented aliases
        // would make a typo look like a working client.
        XCTAssertNil(MediaSSE.previewJPEG(["image": b64]))
        XCTAssertNil(MediaSSE.previewJPEG(["b64_json": b64]))
    }

    func testStageNamesTheEnginesEmitBecomeReadableLabels() {
        XCTAssertEqual(MediaSSE.stageLabel("encode"), "Encoding prompt")
        XCTAssertEqual(MediaSSE.stageLabel("diffuse"), "Composing")
        XCTAssertEqual(MediaSSE.stageLabel("decode"), "Rendering audio")
        // Anything else is passed through — inventing a translation for an
        // unknown stage would just mislabel it.
        XCTAssertEqual(MediaSSE.stageLabel("upsampling"), "upsampling")
    }
}
