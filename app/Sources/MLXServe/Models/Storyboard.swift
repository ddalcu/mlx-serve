import Foundation

/// One shot of a storyboard: generated as its own window, opening on the last
/// frame of the shot before it.
struct StoryboardSegment: Codable, Equatable, Identifiable {
    var id = UUID()
    var prompt: String = ""
    var seconds: Int = 8
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
    /// user's first frame) and landing on the user's last frame only at the end.
    static func shotRequest(base: VideoGenRequest, prompt: String, frames: Int, index: Int, count: Int,
                            previousLastFrame: String?) -> VideoGenRequest {
        var r = base
        r.prompt = prompt
        r.numFrames = frames
        r.chainWindows = 1
        // The server's own chain offset: each window draws seed and seed+1.
        r.seed = base.seed &+ 2 * index
        if index > 0 { r.firstFrameImagePath = previousLastFrame }
        if index < count - 1 { r.lastFrameImagePath = nil }
        // One clip cannot condition every shot; each shot makes its own sound.
        r.audioPath = nil
        return r
    }
}

extension VideoModelPreset {
    /// A storyboard hands each shot the previous one's last frame, so it needs
    /// a first-frame anchor: the REF2VA pack has none.
    var supportsStoryboard: Bool { !supportsReferences }
}
