import Foundation

/// P3.3/3.4 skeletal-animation decisions, kept pure for tests. The SceneKit
/// layer (Model3DSceneView / AvatarView) applies these values to SCNSkinner
/// bones each frame; UniRig joints carry NO semantic names, so the speech
/// drive targets a heuristic bone (highest leaf ≈ the head) and amplitude →
/// jaw-nod is the plan's v1 (convincing at avatar scale, no visemes).
enum SkeletalAnimator {

    /// Defensive cap for malformed parent arrays (cycles/self-references).
    static let maxDepth = 64
    /// Max jaw/head nod, radians (~12°).
    static let maxJawRadians = 12.0 * .pi / 180.0
    /// Idle sway period, seconds.
    static let idlePeriod = 6.0

    /// Depth of each joint from its root (root = 0). Tolerates malformed
    /// parents by capping at `maxDepth` instead of hanging.
    static func depths(parents: [Int?]) -> [Int] {
        parents.indices.map { i in
            var depth = 0
            var cur = parents[i]
            while let p = cur, depth < maxDepth {
                depth += 1
                cur = (p >= 0 && p < parents.count) ? parents[p] : nil
            }
            return depth
        }
    }

    /// The bone the speech envelope drives: the LEAF joint with the highest
    /// bind-pose Y (UniRig emits head-end joints last in the growth direction;
    /// non-leaf spines are skeleton trunk, arms sit lower).
    static func pickSpeechBone(positions: [Double], parents: [Int?]) -> Int {
        let count = parents.count
        guard count > 0 else { return 0 }
        var isLeaf = [Bool](repeating: true, count: count)
        for p in parents.compactMap({ $0 }) where p >= 0 && p < count {
            isLeaf[p] = false
        }
        var best = 0
        var bestY = -Double.infinity
        for j in 0..<count where isLeaf[j] {
            let y = positions[j * 3 + 1]
            if y > bestY {
                bestY = y
                best = j
            }
        }
        return best
    }

    /// Amplitude [0,1] → jaw-nod angle, clamped.
    static func jawAngle(amplitude: Double) -> Double {
        let a = min(max(amplitude, 0), 1)
        return a * maxJawRadians
    }

    /// One smoothing step: fast attack (speech onset opens quickly), slow
    /// release (gentle close reads natural). Call per metering tick.
    static func smoothedAmplitude(previous: Double, target: Double) -> Double {
        let coeff = target > previous ? 0.65 : 0.25 // attack vs release blend
        return previous + (target - previous) * coeff
    }

    /// Subtle idle sway for bone `depth` at time `t` — deeper bones sway a bit
    /// more (whip effect), everything stays under ~3.4°, period `idlePeriod`.
    static func idleSwayRadians(depth: Int, time: Double) -> Double {
        let d = Double(min(depth, 8))
        let amplitude = 0.012 + 0.005 * d // 0.69° .. 3.0°
        let phase = 0.35 * d // slight per-depth lag
        return amplitude * sin(2.0 * .pi * time / idlePeriod + phase)
    }

    // MARK: - Emotes (P3.3 "emote triggers")

    enum Emote: String, CaseIterable {
        case nod // head dips twice (yes)
        case sway // whole-body side sway (hello)
    }

    /// Emote clip length, seconds.
    static let emoteDuration = 1.4

    /// Rotation offset (radians) an in-flight emote adds at `elapsed` seconds:
    /// `head` applies to the speech bone's x (nod axis), `root` to the root
    /// bone's z (sway axis). Zero outside [0, emoteDuration]; envelope is
    /// sin²-windowed so clips start/end at exactly the idle pose (no pop).
    static func emoteOffset(_ emote: Emote, elapsed: Double) -> (head: Double, root: Double) {
        guard elapsed >= 0, elapsed < emoteDuration else { return (0, 0) }
        let u = elapsed / emoteDuration // 0..1
        let window = sin(.pi * u) * sin(.pi * u) // 0 → 1 → 0
        switch emote {
        case .nod:
            return (head: window * sin(2.0 * 2.0 * .pi * u) * (14.0 * .pi / 180.0), root: 0)
        case .sway:
            return (head: 0, root: window * sin(1.5 * 2.0 * .pi * u) * (8.0 * .pi / 180.0))
        }
    }
}

/// A one-shot emote request handed to the scene view (Equatable so SwiftUI
/// state changes propagate; a new `startedAt` restarts the clip).
struct EmoteTrigger: Equatable {
    let kind: SkeletalAnimator.Emote
    let startedAt: TimeInterval
}
