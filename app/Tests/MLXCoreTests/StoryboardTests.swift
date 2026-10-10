import XCTest
import AppKit
import AVFoundation
@testable import MLXCore

final class StoryboardTests: XCTestCase {

    private let h3Ladder = Array(stride(from: 5, through: 362, by: 17))
    private let ltxLadder = Array(stride(from: 9, through: 193, by: 8))

    // MARK: - Lengths

    /// A shot's seconds pick the shortest rung that covers them, the longest when none does.
    func testShotFramesCoverTheirSecondsOnTheLadder() {
        XCTAssertEqual(Storyboard.frames(seconds: 8, fps: 24, ladder: h3Ladder), 192)
        XCTAssertEqual(Storyboard.frames(seconds: 15, fps: 24, ladder: h3Ladder), 362)
        XCTAssertEqual(Storyboard.frames(seconds: 20, fps: 24, ladder: h3Ladder), 362)
        XCTAssertEqual(Storyboard.frames(seconds: 8, fps: 24, ladder: ltxLadder), 193)
    }

    /// Every join shares one frame: a shot opens on the frame the one before it ended on.
    func testDeliveredFramesShareOneSeamFramePerJoin() {
        XCTAssertEqual(Storyboard.deliveredFrames([192, 192, 192]), 574)
        XCTAssertEqual(Storyboard.deliveredFrames([124]), 124)
        XCTAssertEqual(Storyboard.deliveredFrames([]), 0)
    }

    /// The tested floor starts the range (an unattended batch must not land below it),
    /// the canvas's longest rung ends it, and a floor past the ceiling collapses onto it.
    func testShotSecondsRunFromTheTestedFloorToTheLongestRung() {
        XCTAssertEqual(Storyboard.secondsRange(ladder: h3Ladder, fps: 24, floorFrames: 124), 5...15)
        XCTAssertEqual(Storyboard.secondsRange(ladder: ltxLadder, fps: 24, floorFrames: 0), 1...8)
        XCTAssertEqual(Storyboard.secondsRange(ladder: [5, 22, 39], fps: 24, floorFrames: 124), 1...1)
    }

    /// Planned shots stop at 10 s (cheaper per second than the 15 s ceiling), never below the floor.
    func testPlannedShotsStopAtTenSecondsInsideTheShotRange() {
        XCTAssertEqual(Storyboard.plannedRange(5...15), 5...10)
        XCTAssertEqual(Storyboard.plannedRange(1...8), 1...8)
        XCTAssertEqual(Storyboard.plannedRange(12...15), 12...12)
    }

    func testLengthLabelSwitchesToMinutesPastOneMinute() {
        XCTAssertEqual(Storyboard.lengthLabel(45), "45 s")
        XCTAssertEqual(Storyboard.lengthLabel(300), "5:00")
        XCTAssertEqual(Storyboard.lengthLabel(75), "1:15")
    }

    // MARK: - Planner reply

    func testPlannerReplyParsesIntoShots() {
        let reply = """
        Here is your storyboard:
        === SHOT 1 | 8s ===
        integrated_multimodal_description:
        A red fox trots through snow.

        === SHOT 2 | 20 s ===
        The fox stops and looks back.
        === SHOT 3 | 2s ===
        """
        let shots = Storyboard.parse(reply, seconds: 5...15)
        XCTAssertEqual(shots.map(\.seconds), [8, 15])
        XCTAssertEqual(shots[0].prompt, "integrated_multimodal_description:\nA red fox trots through snow.")
        XCTAssertEqual(shots[1].prompt, "The fox stops and looks back.")
    }

    func testAReplyWithNoShotHeadersYieldsNoShots() {
        XCTAssertTrue(Storyboard.parse("A fox in the snow, eight seconds.", seconds: 5...15).isEmpty)
    }

    /// The planner is told the bounds it must plan inside.
    func testPlannerRequestCarriesTheLengthsAndTheFormat() {
        let r = PromptRewriter.storyboard(idea: "a fox's day", format: .h3Base, totalSeconds: 300, shotSeconds: 5...15)
        XCTAssertTrue(r.system.contains("between 5 and 15 seconds"))
        XCTAssertTrue(r.system.contains("integrated_multimodal_description:"))
        XCTAssertTrue(r.user.contains("300 seconds"))
        XCTAssertTrue(r.user.contains("a fox's day"))
    }

    // MARK: - Shot requests

    private var base: VideoGenRequest {
        var r = VideoGenRequest(model: .minimaxH3, prompt: "story", width: 960, height: 544,
                                numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        r.seed = 10
        r.firstFrameImagePath = "/u/first.png"
        r.lastFrameImagePath = "/u/last.png"
        r.audioPath = "/u/clip.wav"
        r.chainWindows = 3
        return r
    }

    /// Shot 0 opens on the user's first frame, later shots on the previous shot's last frame;
    /// only the final shot lands on the user's last frame.
    func testShotsHandOffTheirLastFrameAndOnlyTheEndLandsOnTheUsersLastFrame() {
        let first = Storyboard.shotRequest(base: base, prompt: "a", frames: 192, index: 0, count: 3, previousLastFrame: nil)
        XCTAssertEqual(first.firstFrameImagePath, "/u/first.png")
        XCTAssertNil(first.lastFrameImagePath)
        let middle = Storyboard.shotRequest(base: base, prompt: "b", frames: 124, index: 1, count: 3, previousLastFrame: "/s/1.png")
        XCTAssertEqual(middle.firstFrameImagePath, "/s/1.png")
        XCTAssertNil(middle.lastFrameImagePath)
        let last = Storyboard.shotRequest(base: base, prompt: "c", frames: 124, index: 2, count: 3, previousLastFrame: "/s/2.png")
        XCTAssertEqual(last.firstFrameImagePath, "/s/2.png")
        XCTAssertEqual(last.lastFrameImagePath, "/u/last.png")
    }

    /// Each shot is one window with its own prompt, length and seed, and no shared soundtrack.
    func testAShotIsOneWindowWithItsOwnPromptLengthAndSeed() {
        let r = Storyboard.shotRequest(base: base, prompt: "b", frames: 141, index: 2, count: 3, previousLastFrame: "/s/2.png")
        XCTAssertEqual(r.prompt, "b")
        XCTAssertEqual(r.numFrames, 141)
        XCTAssertEqual(r.chainWindows, 1)
        XCTAssertEqual(r.seed, 14)
        XCTAssertNil(r.audioPath)
    }

    func testOnlyAPackWithAKeyframeRowCanStoryboard() {
        XCTAssertTrue(VideoModelPreset.minimaxH3.supportsStoryboard)
        XCTAssertTrue(VideoModelPreset.ltx23Q4.supportsStoryboard)
        XCTAssertFalse(VideoModelPreset.minimaxH3Ref2VA.supportsStoryboard)
    }

    // MARK: - Frames on disk

    private func rgb(frames: [(UInt8, UInt8, UInt8)], width: Int, height: Int) -> Data {
        var d = Data(capacity: frames.count * width * height * 3)
        for c in frames { for _ in 0..<(width * height) { d.append(contentsOf: [c.0, c.1, c.2]) } }
        return d
    }

    func testLastFramePNGIsTheFinalFrame() throws {
        let f = VideoGenService.DecodedFrames(
            rgb: rgb(frames: [(255, 0, 0), (0, 255, 0), (0, 0, 255)], width: 8, height: 4),
            frames: 3, height: 4, width: 8, fps: 24)
        let png = try XCTUnwrap(VideoGenService.lastFramePNG(f))
        let rep = try XCTUnwrap(NSBitmapImageRep(data: png))
        XCTAssertEqual(rep.pixelsWide, 8)
        XCTAssertEqual(rep.pixelsHigh, 4)
        var px = [Int](repeating: 0, count: 4)
        rep.getPixel(&px, atX: 3, y: 2)
        XCTAssertEqual(Array(px.prefix(3)), [0, 0, 255])
    }

    /// Three 10-frame shots join into 28 frames: each later shot drops the frame it opened on.
    func testStitchedClipDropsOneSeamFramePerJoin() async throws {
        let (w, h, fps, n) = (64, 64, 24, 10)
        let sr = 16000, ch = 2
        let pcm = Data(count: (sr * n / fps) * ch * 2)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-story-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        var shots: [URL] = []
        for (i, c) in [(UInt8(200), UInt8(0), UInt8(0)), (0, 200, 0), (0, 0, 200)].enumerated() {
            let url = dir.appendingPathComponent("shot-\(i).mp4")
            try VideoGenService.writeMP4(rgb: rgb(frames: Array(repeating: c, count: n), width: w, height: h),
                                         frames: n, width: w, height: h, fps: fps, to: url,
                                         audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)
            shots.append(url)
        }
        let out = dir.appendingPathComponent("joined.mp4")
        try await VideoGenService.stitchShots(shots, fps: fps, to: out)

        let asset = AVURLAsset(url: out)
        let videos = try await asset.loadTracks(withMediaType: .video)
        let audios = try await asset.loadTracks(withMediaType: .audio)
        let video = try XCTUnwrap(videos.first)
        XCTAssertFalse(audios.isEmpty, "the soundtrack was dropped")
        let reader = try AVAssetReader(asset: asset)
        // Decoded frames, which honour the edit list; compressed buffers include markers.
        let output = AVAssetReaderTrackOutput(track: video, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        var count = 0
        while let s = output.copyNextSampleBuffer() { count += CMSampleBufferGetNumSamples(s) }
        XCTAssertEqual(count, 3 * n - 2)
    }
}
