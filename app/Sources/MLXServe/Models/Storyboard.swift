import Foundation
import CoreGraphics

/// One shot of a storyboard: generated as its own window, opening on the last
/// frame of the shot before it.
struct StoryboardSegment: Codable, Equatable, Identifiable {
    var id = UUID()
    var prompt: String = ""
    var seconds: Int = 8
    /// A picture this shot lands on and the next one opens on, in place of the
    /// shot's own rendered last frame. Never the final shot's: that is the pane's.
    var endFramePath: String? = nil
}

/// A long clip as a list of shots the APP chains: each shot is an ordinary
/// single-window request, so neither the server's 6-window cap nor its
/// one-response transport cap bounds the total, and a failed shot costs one
/// shot.
enum Storyboard {

    /// The shortest rung of `ladder` that covers `seconds`, the longest when none does.
    static func frames(seconds: Int, fps: Int, ladder: [Int]) -> Int {
        let needed = seconds * max(1, fps)
        return ladder.first { $0 >= needed } ?? ladder.last ?? needed
    }

    /// Frames of the joined clip: every join shares one frame.
    static func deliveredFrames(_ shotFrames: [Int]) -> Int {
        guard !shotFrames.isEmpty else { return 0 }
        return shotFrames.reduce(0, +) - (shotFrames.count - 1)
    }

    /// Whole seconds a shot can ask for: from the model's tested floor (or the
    /// ladder's first rung) to its longest rung at this canvas.
    static func secondsRange(ladder: [Int], fps: Int, floorFrames: Int) -> ClosedRange<Int> {
        let f = Double(max(1, fps))
        let hi = max(1, Int(Double(ladder.last ?? fps) / f))
        let lo = max(1, Int(Double(max(floorFrames, ladder.first ?? 1)) / f))
        return min(lo, hi)...hi
    }

    /// The longest story Enhance plans; longer ones are built with Add shot.
    static let longestPlannedStory = 120

    /// Attention grows with the square of a shot's length, so three 10 s shots
    /// cost less than two 15 s ones. A shot can still be dragged to the ceiling.
    static func plannedRange(_ range: ClosedRange<Int>) -> ClosedRange<Int> {
        range.lowerBound...max(range.lowerBound, min(range.upperBound, 10))
    }

    /// "45 s" under a minute, "5:00" past it.
    static func lengthLabel(_ seconds: Int) -> String {
        seconds < 60 ? L10n.format("%lld s", Int64(seconds))
                     : String(format: "%d:%02d", seconds / 60, seconds % 60)
    }

    /// The header the planner writes above each shot: `=== SHOT 2 | 8s ===`.
    private static let header = try! NSRegularExpression(
        pattern: #"^[#*\s]*=+\s*SHOT\s+\d+\s*[|:\-–—]\s*(\d+)\s*s[a-z]*\s*=+[*\s]*$"#,
        options: [.caseInsensitive, .anchorsMatchLines])

    /// The planner's reply as shots, seconds clamped into `seconds`; a shot
    /// with no prompt is dropped, and a reply with no headers is no plan.
    static func parse(_ reply: String, seconds: ClosedRange<Int>) -> [StoryboardSegment] {
        let text = reply as NSString
        let matches = header.matches(in: reply, range: NSRange(location: 0, length: text.length))
        return matches.enumerated().compactMap { i, m in
            let start = m.range.location + m.range.length
            let end = i + 1 < matches.count ? matches[i + 1].range.location : text.length
            let prompt = PromptRewriter.clean(text.substring(with: NSRange(location: start, length: end - start)))
            guard !prompt.isEmpty, let s = Int(text.substring(with: m.range(at: 1))) else { return nil }
            return StoryboardSegment(prompt: prompt, seconds: min(seconds.upperBound, max(seconds.lowerBound, s)))
        }
    }

    /// Shot `index` of `count`: the pane's settings with this shot's prompt and
    /// length, opening on the previous shot's last frame (shot 0 keeps the
    /// user's first frame) and landing on `endFrame`, or on the user's last
    /// frame at the end.
    static func shotRequest(base: VideoGenRequest, prompt: String, frames: Int, index: Int, count: Int,
                            previousLastFrame: String?, endFrame: String? = nil) -> VideoGenRequest {
        var r = base
        r.prompt = prompt
        r.numFrames = frames
        r.chainWindows = 1
        // The server's own chain offset: each window draws seed and seed+1.
        r.seed = base.seed &+ 2 * index
        if index > 0 { r.firstFrameImagePath = previousLastFrame }
        if index < count - 1 { r.lastFrameImagePath = endFrame }
        // One clip cannot condition every shot; each shot makes its own sound.
        r.audioPath = nil
        return r
    }

    /// One frame of the joined clip that fades toward the next shot.
    struct SeamBlend: Equatable {
        let join: Int
        /// When the frame shows, in the joined clip's seconds.
        let frame: Range<Double>
        /// The middle of the next shot's first frame in the clip: the frame to fade into.
        let into: Double
        let weight: Double
    }

    /// The last two frames before every join fade toward the first frame the
    /// next shot shows, by a third and two thirds. Read off the frames' own
    /// times, so no frame or sample rate is assumed: `keptFrames[i]` holds when
    /// each frame shot i keeps starts in the clip, `ends[i]` when the last ends.
    static func seamBlends(keptFrames: [[Double]], ends: [Double]) -> [SeamBlend] {
        var blends: [SeamBlend] = []
        for i in keptFrames.indices.dropLast() where i < ends.count - 1 {
            let frames = keptFrames[i], next = keptFrames[i + 1]
            guard let opening = next.first else { continue }
            let into = (opening + (next.count > 1 ? next[1] : ends[i + 1])) / 2
            for (back, weight) in [(2, 1.0 / 3), (1, 2.0 / 3)] where frames.count >= back {
                let k = frames.count - back
                let end = k + 1 < frames.count ? frames[k + 1] : ends[i]
                guard end > frames[k] else { continue }
                blends.append(SeamBlend(join: i, frame: frames[k]..<end, into: into, weight: weight))
            }
        }
        return blends
    }

    /// Where a picture is drawn to cover a canvas: scaled to fill it, centred,
    /// the overflow cropped, never stretched.
    static func coverRect(source: CGSize, canvas: CGSize) -> CGRect {
        guard source.width > 0, source.height > 0 else { return CGRect(origin: .zero, size: canvas) }
        let scale = max(canvas.width / source.width, canvas.height / source.height)
        let size = CGSize(width: source.width * scale, height: source.height * scale)
        return CGRect(x: (canvas.width - size.width) / 2, y: (canvas.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    /// What the next shot opens on: the user's picture for this join while it
    /// is on disk, else this shot's rendered last frame.
    static func nextOpening(keyframe: String?, renderedLastFrame: String,
                            exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> String {
        keyframe.flatMap { exists($0) ? $0 : nil } ?? renderedLastFrame
    }

    /// The shots with no picture on the final one: a picture left there (the
    /// shot after it was removed) becomes the film's Last frame when that is empty,
    /// and is dropped otherwise.
    static func settlingFinalShot(_ shots: [StoryboardSegment], lastFrame: String?)
        -> (shots: [StoryboardSegment], lastFrame: String?) {
        guard let picture = shots.last?.endFramePath else { return (shots, lastFrame) }
        var settled = shots
        settled[settled.count - 1].endFramePath = nil
        return (settled, lastFrame ?? picture)
    }

    /// Restored shots without the pictures that are no longer on disk.
    static func withoutMissingEndFrames(_ shots: [StoryboardSegment],
                                        exists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) })
        -> [StoryboardSegment] {
        shots.map { shot in
            var shot = shot
            if let path = shot.endFramePath, !exists(path) { shot.endFramePath = nil }
            return shot
        }
    }
}

extension VideoModelPreset {
    /// A storyboard hands each shot the previous one's last frame, so it needs
    /// a first-frame anchor: the REF2VA pack has none.
    var supportsStoryboard: Bool { !supportsReferences }
}
