import AVFoundation
import ImageIO
import XCTest
@testable import MLXCore

/// Word-timed captions: how words become caption states, and that a burn puts
/// them on the frames while keeping the soundtrack.
final class CaptionBurnerTests: XCTestCase {

    private func w(_ text: String, _ start: Double, _ end: Double) -> CaptionWord {
        CaptionWord(text: text, start: start, end: end)
    }

    func testWordsComeFromVerboseJSONAtTheTopLevelOrInsideSegments() throws {
        let top = #"{"text":"Hi Leo","words":[{"word":" Hi","start":0.1,"end":0.4},{"word":"Leo","start":0.4,"end":0.9}]}"#
        XCTAssertEqual(CaptionPlan.words(fromVerboseJSON: Data(top.utf8)), [w("Hi", 0.1, 0.4), w("Leo", 0.4, 0.9)])
        let nested = #"{"segments":[{"start":0,"end":1,"text":"x","words":[{"word":"x","start":0.2,"end":0.5}]}]}"#
        XCTAssertEqual(CaptionPlan.words(fromVerboseJSON: Data(nested.utf8)), [w("x", 0.2, 0.5)])
        XCTAssertEqual(CaptionPlan.words(fromVerboseJSON: Data("{}".utf8)), [])
    }

    func testGroupsBreakAtThreeWordsAPauseOrPunctuation() {
        let words = [w("Merry", 0, 0.4), w("Christmas,", 0.4, 1.0), w("Leo", 1.0, 1.5),
                     w("you", 2.5, 2.7), w("did", 2.7, 2.9), w("it", 2.9, 3.0), w("again", 3.0, 3.4)]
        XCTAssertEqual(CaptionPlan.groups(words).map { $0.map(\.text) },
                       [["Merry", "Christmas,"], ["Leo"], ["you", "did", "it"], ["again"]])
    }

    /// One state per word, back to back inside a group, never overlapping, the
    /// live word the one being said.
    func testStatesTileEachGroupWithTheSpokenWordLive() {
        let words = [w("a", 0, 0.3), w("b", 0.3, 0.6), w("c", 2.0, 2.4)]
        let states = CaptionPlan.states(words)
        XCTAssertEqual(states.count, 3)
        XCTAssertEqual(states.map(\.active), [0, 1, 0])
        XCTAssertEqual(states[0].end, states[1].start)
        for (x, y) in zip(states, states.dropFirst()) { XCTAssertLessThanOrEqual(x.end, y.start) }
        XCTAssertEqual(states[1].end, 0.85, accuracy: 1e-9, "a group stays up a beat after its last word")
        XCTAssertEqual(states[2].words, ["C"], "captions are set in capitals")
    }

    func testAShortPauseKeepsTheCaptionUpUntilTheNextOne() {
        let states = CaptionPlan.states([w("a", 0, 0.3), w("b.", 0.3, 0.6), w("c", 0.9, 1.2)])
        XCTAssertEqual(states[1].end, 0.9, accuracy: 1e-9)
    }

    /// The brightest pixel (red + green) in a band of rows, half a second in.
    private func brightest(_ url: URL, rows: ClosedRange<Double>) async throws -> Int {
        let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        let (image, _) = try await gen.image(at: CMTime(seconds: 0.5, preferredTimescale: 600))
        let rep = NSBitmapImageRep(cgImage: image)
        var best = 0
        for y in Int(rows.lowerBound * Double(rep.pixelsHigh))..<Int(rows.upperBound * Double(rep.pixelsHigh)) {
            for x in stride(from: 0, to: rep.pixelsWide, by: 2) {
                guard let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                best = max(best, Int((c.redComponent + c.greenComponent) * 127.5))
            }
        }
        return best
    }

    /// A two-second grey clip with a silent soundtrack, in a folder removed at teardown.
    private func greyClip() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("caption-burn-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let (width, height, frames) = (256, 448, 48)
        let clip = dir.appendingPathComponent("clip.mp4")
        try VideoGenService.writeMP4(rgb: Data(repeating: 40, count: width * height * 3 * frames), frames: frames,
                                     width: width, height: height, fps: 24, to: clip,
                                     audioPCM: Data(repeating: 0, count: 2 * 2 * 32000 * 2),
                                     audioSampleRate: 32000, audioChannels: 2)
        return clip
    }

    /// The burn draws onto the frames (the caption band changes, the top of the
    /// frame does not) and the soundtrack survives the export.
    func testBurnDrawsTheCaptionAndKeepsTheAudio() async throws {
        let clip = try greyClip()
        let out = clip.deletingLastPathComponent().appendingPathComponent("captioned.mp4")
        try await CaptionBurner.burn(video: clip, words: [w("HELLO", 0, 1.5)], to: out)

        let audio = try await AVURLAsset(url: out).loadTracks(withMediaType: .audio)
        XCTAssertFalse(audio.isEmpty, "the soundtrack was dropped")
        let before = try await brightest(clip, rows: 0.0...0.3)
        let band = try await brightest(out, rows: 0.6...0.8)
        let after = try await brightest(out, rows: 0.0...0.3)
        XCTAssertGreaterThan(band, 200, "no caption in the caption band")
        // The Core Animation tool turned this frame white, and a real clip a fifth brighter.
        XCTAssertEqual(after, before, accuracy: 3, "the picture outside the caption changed")
    }

    func testMemeTextRunsAcrossTheTopAndLeavesTheRestAlone() async throws {
        let clip = try greyClip()
        let out = clip.deletingLastPathComponent().appendingPathComponent("meme.mp4")
        try await CaptionBurner.burn(video: clip, words: [], top: "when the wifi drops mid game", to: out)
        let top = try await brightest(out, rows: 0.0...0.2)
        let middle = try await brightest(out, rows: 0.5...0.9)
        let before = try await brightest(clip, rows: 0.5...0.9)
        XCTAssertGreaterThan(top, 200, "no meme text across the top")
        XCTAssertEqual(middle, before, accuracy: 3)
    }

    func testAGIFPlaysForwardThenBackAndLoopsForever() async throws {
        let clip = try greyClip()
        let gif = clip.deletingLastPathComponent().appendingPathComponent("clip.gif")
        try await LoopingGIF.make(from: clip, to: gif, fps: 6)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(gif as CFURL, nil))
        XCTAssertEqual(CGImageSourceGetCount(source), 12 + 10, "12 frames forward, 10 back without repeating the ends")
        let props = CGImageSourceCopyProperties(source, nil) as? [CFString: Any]
        let loop = (props?[kCGImagePropertyGIFDictionary] as? [CFString: Any])?[kCGImagePropertyGIFLoopCount] as? Int
        XCTAssertEqual(loop, 0, "a reaction GIF loops forever")
    }

    func testMemeTextGoesTopAndBottomOfAPicture() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("meme-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("p.png"), out = dir.appendingPathComponent("m.png")
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 512, pixelsHigh: 768, bitsPerSample: 8, samplesPerPixel: 4,
                                   hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        for y in 0..<768 { for x in 0..<512 { rep.setColor(NSColor(white: 0.16, alpha: 1), atX: x, y: y) } }
        try rep.representation(using: .png, properties: [:])!.write(to: src)
        try CaptionBurner.burn(image: src, top: "his royal highness", bottom: "duke of the couch", to: out)
        let img = try XCTUnwrap(NSBitmapImageRep(data: try Data(contentsOf: out)))
        func brightest(_ rows: ClosedRange<Double>) -> Int {
            var best = 0
            for y in Int(rows.lowerBound * 768)..<Int(rows.upperBound * 768) {
                for x in stride(from: 0, to: 512, by: 2) {
                    let c = img.colorAt(x: x, y: y)!.usingColorSpace(.deviceRGB)!
                    best = max(best, Int(c.redComponent * 255))
                }
            }
            return best
        }
        XCTAssertGreaterThan(brightest(0.0...0.2), 200, "no top text")
        XCTAssertGreaterThan(brightest(0.8...0.99), 200, "no bottom text")
        XCTAssertLessThan(brightest(0.4...0.6), 60, "the middle of the picture changed")
    }

    /// A song shares as a video: the cover photo for as long as the song, with the song as its sound.
    func testASongBecomesAVideoOfItsCoverWithTheSongAsItsSound() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("song-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let photo = dir.appendingPathComponent("p.png"), wav = dir.appendingPathComponent("s.wav"), out = dir.appendingPathComponent("s.mp4")
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 300, pixelsHigh: 400, bitsPerSample: 8, samplesPerPixel: 4,
                                   hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        try rep.representation(using: .png, properties: [:])!.write(to: photo)
        let tone = (0..<(16_000 * 3)).map { Float(sin(Double($0) * 0.05) * 0.3) }
        try AudioReference.wavData(fromMonoFloat: tone, sampleRate: 16_000).write(to: wav)
        try await SongVideo.make(cover: photo, title: "Sarah's Day", song: wav, shape: .square, to: out)
        let asset = AVURLAsset(url: out)
        let audio = try await asset.loadTracks(withMediaType: .audio)
        let video = try await asset.loadTracks(withMediaType: .video)
        XCTAssertFalse(audio.isEmpty, "the song was dropped")
        XCTAssertFalse(video.isEmpty)
        let seconds = try await asset.load(.duration).seconds
        XCTAssertEqual(seconds, 3, accuracy: 0.6)
    }
}
