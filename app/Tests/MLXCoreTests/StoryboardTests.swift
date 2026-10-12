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

    /// The last two frames before each join fade toward the middle of the next shot's first frame in the
    /// clip, by a third and two thirds, read off the frames' own times whatever their spacing.
    func testTheLastTwoFramesBeforeAJoinBlendTowardTheNextShot() {
        // Typed up front: literals inferred inside XCTAssertEqual time out the type checker on CI.
        let kept: [[Double]] = [[0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.75]]
        let ends: [Double] = [0.4, 0.7, 0.8]
        let frames: [Range<Double>] = [0.2..<0.3, 0.3..<0.4, 0.5..<0.6, 0.6..<0.7]
        let into: [Double] = [0.45, 0.45, 0.725, 0.725]
        let weights: [Double] = [1.0 / 3, 2.0 / 3, 1.0 / 3, 2.0 / 3]
        let blends = Storyboard.seamBlends(keptFrames: kept, ends: ends)
        XCTAssertEqual(blends.map(\.join), [0, 0, 1, 1])
        XCTAssertEqual(blends.map(\.frame), frames)
        XCTAssertEqual(blends.map(\.into), into)
        XCTAssertEqual(blends.map(\.weight), weights)
        let single: [[Double]] = [[0, 0.1]]
        XCTAssertTrue(Storyboard.seamBlends(keptFrames: single, ends: [0.2]).isEmpty)
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
        // Enhance's shots replace the old ones whole: the planner never saw a join picture.
        XCTAssertTrue(shots.allSatisfy { $0.endFramePath == nil })
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

    /// A shot's own last frame lands that shot; the final shot keeps the pane's last frame.
    func testAShotsOwnLastFrameLandsItAndTheFinalShotKeepsThePanes() {
        let middle = Storyboard.shotRequest(base: base, prompt: "b", frames: 124, index: 1, count: 3,
                                            previousLastFrame: "/k/0.png", endFrame: "/k/1.png")
        XCTAssertEqual(middle.firstFrameImagePath, "/k/0.png")
        XCTAssertEqual(middle.lastFrameImagePath, "/k/1.png")
        let last = Storyboard.shotRequest(base: base, prompt: "c", frames: 124, index: 2, count: 3,
                                          previousLastFrame: "/k/1.png", endFrame: "/k/2.png")
        XCTAssertEqual(last.lastFrameImagePath, "/u/last.png")
    }

    /// The next shot opens on the user's picture when there is one on disk, else on the rendered last frame.
    func testTheNextShotOpensOnTheUsersPictureWhenItIsThere() {
        let there: (String) -> Bool = { $0 == "/k/1.png" }
        XCTAssertEqual(Storyboard.nextOpening(keyframe: "/k/1.png", renderedLastFrame: "/s/1.png", exists: there), "/k/1.png")
        XCTAssertEqual(Storyboard.nextOpening(keyframe: nil, renderedLastFrame: "/s/1.png", exists: there), "/s/1.png")
        XCTAssertEqual(Storyboard.nextOpening(keyframe: "/gone.png", renderedLastFrame: "/s/1.png", exists: there), "/s/1.png")
    }

    /// A picture left on the final shot becomes the film's Last frame when that is empty, and is dropped otherwise.
    func testAPictureLeftOnTheFinalShotMovesToTheLastFrame() {
        let shots = [StoryboardSegment(prompt: "a", endFramePath: "/k/0.png"),
                     StoryboardSegment(prompt: "b", endFramePath: "/k/1.png")]
        let moved = Storyboard.settlingFinalShot(shots, lastFrame: nil)
        XCTAssertEqual(moved.shots.map(\.endFramePath), ["/k/0.png", nil])
        XCTAssertEqual(moved.lastFrame, "/k/1.png")
        let kept = Storyboard.settlingFinalShot(shots, lastFrame: "/u/last.png")
        XCTAssertEqual(kept.shots.map(\.endFramePath), ["/k/0.png", nil])
        XCTAssertEqual(kept.lastFrame, "/u/last.png")
    }

    /// A picture is cropped to the canvas, centred, without stretching.
    func testAPictureCoversTheCanvasCentred() {
        XCTAssertEqual(Storyboard.coverRect(source: CGSize(width: 400, height: 400), canvas: CGSize(width: 1536, height: 672)),
                       CGRect(x: 0, y: -432, width: 1536, height: 1536))
        XCTAssertEqual(Storyboard.coverRect(source: CGSize(width: 1920, height: 1080), canvas: CGSize(width: 1280, height: 704)),
                       CGRect(x: 0, y: -8, width: 1280, height: 720))
    }

    /// Any picture the Mac reads becomes a PNG of the canvas's size; one it cannot read is nil.
    func testAJoinPictureBecomesACanvasSizedPNG() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-join-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let tiff = dir.appendingPathComponent("join.tiff")
        let source = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 120, pixelsHigh: 40, bitsPerSample: 8,
                                      samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
                                      colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        try XCTUnwrap(source.representation(using: .tiff, properties: [:])).write(to: tiff)
        let png = try XCTUnwrap(VideoGenService.canvasPNG(from: tiff, width: 64, height: 48))
        let rep = try XCTUnwrap(NSBitmapImageRep(data: png))
        XCTAssertEqual([rep.pixelsWide, rep.pixelsHigh], [64, 48])
        XCTAssertEqual(png.prefix(4), Data([0x89, 0x50, 0x4E, 0x47]))
        let junk = dir.appendingPathComponent("junk.png")
        try Data("not a picture".utf8).write(to: junk)
        XCTAssertNil(VideoGenService.canvasPNG(from: junk, width: 64, height: 48))
    }

    /// A shot's picture that is no longer on disk is dropped on restore.
    func testAMissingShotPictureIsDroppedOnRestore() {
        let shots = [StoryboardSegment(prompt: "a", endFramePath: "/k/0.png"),
                     StoryboardSegment(prompt: "b", endFramePath: "/gone.png")]
        let restored = Storyboard.withoutMissingEndFrames(shots) { $0 == "/k/0.png" }
        XCTAssertEqual(restored.map(\.endFramePath), ["/k/0.png", nil])
    }

    /// A shot saved before shots had pictures still reads.
    func testAShotSavedBeforePicturesStillDecodes() throws {
        let json = #"{"id":"6C1F3E2A-8B1D-4E6F-9A0B-2C3D4E5F6A7B","prompt":"a fox","seconds":9}"#
        let shot = try JSONDecoder().decode(StoryboardSegment.self, from: Data(json.utf8))
        XCTAssertEqual(shot.prompt, "a fox")
        XCTAssertNil(shot.endFramePath)
    }

    /// A shot's own last frame is sent as the request's last frame, so every pack that storyboards must take one.
    func testEveryPackThatStoryboardsTakesALastFrame() {
        for preset in VideoModelPreset.all where preset.supportsStoryboard {
            XCTAssertTrue(preset.supportsLastFrame, preset.id)
        }
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

    /// Joined frames, decoded, as the colour at each frame's centre.
    private func centreColours(of url: URL) async throws -> [(r: Double, g: Double, b: Double)] {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let video = try XCTUnwrap(tracks.first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: video, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        var colours: [(r: Double, g: Double, b: Double)] = []
        while let s = output.copyNextSampleBuffer() {
            guard let px = CMSampleBufferGetImageBuffer(s) else { continue }
            CVPixelBufferLockBaseAddress(px, .readOnly)
            let row = CVPixelBufferGetBytesPerRow(px)
            let p = CVPixelBufferGetBaseAddress(px)!.assumingMemoryBound(to: UInt8.self)
                + (CVPixelBufferGetHeight(px) / 2) * row + (CVPixelBufferGetWidth(px) / 2) * 4
            colours.append((Double(p[2]), Double(p[1]), Double(p[0])))
            CVPixelBufferUnlockBaseAddress(px, .readOnly)
        }
        return colours
    }

    /// Before each join the clip fades into the next shot over two frames, toward the frame the clip shows
    /// (the shot's own opening frame is cut); nothing is added or lost. The shots carry sound as generated
    /// ones do, at two frame and sample rates: sound moves the video onto the audio's timescale.
    func testTheJoinedClipFadesIntoEachShotOverTwoFrames() async throws {
        for (fps, sr) in [(24, 16000), (25, 32000)] {
            let (w, h, n, ch) = (64, 64, 2 * fps, 2)
            let pcm = Data(count: (sr * n / fps) * ch * 2)
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-seam-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            let red: (UInt8, UInt8, UInt8) = (200, 0, 0), green: (UInt8, UInt8, UInt8) = (0, 200, 0)
            var shots: [URL] = []
            let opening: [(UInt8, UInt8, UInt8)] = [(0, 0, 200)]
            let shotFrames: [[(UInt8, UInt8, UInt8)]] = [Array(repeating: red, count: n),
                                                         opening + Array(repeating: green, count: n - 1)]
            for (i, frames) in shotFrames.enumerated() {
                let url = dir.appendingPathComponent("shot-\(i).mp4")
                try VideoGenService.writeMP4(rgb: rgb(frames: frames, width: w, height: h),
                                             frames: n, width: w, height: h, fps: fps, to: url,
                                             audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)
                shots.append(url)
            }
            let out = dir.appendingPathComponent("joined.mp4")
            try await VideoGenService.stitchShots(shots, to: out)
            let c = try await centreColours(of: out)
            XCTAssertEqual(c.count, 2 * n - 1, "\(fps) fps")
            let seam = n - 1
            // The join is re-encoded, and a software encoder (CI) smears some colour across a hard
            // change into the frames either side; a blended frame carries over 100 of the other colour.
            XCTAssertGreaterThan(c[seam - 2].r, 150, "\(fps) fps"); XCTAssertLessThan(c[seam - 2].g, 80, "\(fps) fps")
            XCTAssertGreaterThan(c[seam - 1].g, 100, "\(fps) fps"); XCTAssertGreaterThan(c[seam - 1].r, c[seam - 1].g, "\(fps) fps")
            XCTAssertGreaterThan(c[seam].r, 100, "\(fps) fps"); XCTAssertGreaterThan(c[seam].g, c[seam].r, "\(fps) fps")
            XCTAssertLessThan(c[seam + 1].r, 80, "\(fps) fps"); XCTAssertGreaterThan(c[seam + 1].g, 150, "\(fps) fps")
            XCTAssertTrue(c[(seam - 1)...(seam + 1)].allSatisfy { $0.b < 30 }, "the cut opening frame leaked in at \(fps) fps")
        }
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
        try await VideoGenService.stitchShots(shots, to: out)

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
