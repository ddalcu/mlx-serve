import XCTest
@testable import MLXCore

/// P3.3/3.4 pure logic: bone-depth computation for the skeletal idle, the
/// jaw/head bone pick heuristic (UniRig joints are UNNAMED — the speech drive
/// targets the highest leaf joint), and the amplitude→jaw-angle mapping with
/// attack/release smoothing. The SceneKit application layer is thin and
/// untestable; everything decision-shaped lives here.
final class SkeletalAnimatorTests: XCTestCase {

    func testBoneDepthsFromParents() {
        // pelvis(root) → spine → head; plus a root-child arm.
        let depths = SkeletalAnimator.depths(parents: [nil, 0, 1, 0])
        XCTAssertEqual(depths, [0, 1, 2, 1])
    }

    func testDepthsTolerateForwardReferencesAndCycles() {
        // Malformed parents (cycle) must not hang; depth caps out.
        let depths = SkeletalAnimator.depths(parents: [1, 0])
        XCTAssertEqual(depths.count, 2)
        XCTAssertTrue(depths.allSatisfy { $0 <= SkeletalAnimator.maxDepth })
    }

    func testJawBonePickIsHighestLeaf() {
        // Chain: pelvis(y=0) → spine(y=0.5) → head(y=1.0), arm leaf at y=0.6.
        // Head (highest LEAF) wins — not the arm, not the non-leaf spine.
        let positions: [Double] = [0, 0, 0, 0, 0.5, 0, 0, 1.0, 0, 0.3, 0.6, 0]
        let parents: [Int?] = [nil, 0, 1, 1]
        XCTAssertEqual(SkeletalAnimator.pickSpeechBone(positions: positions, parents: parents), 2)
    }

    func testJawBonePickSingleJoint() {
        XCTAssertEqual(SkeletalAnimator.pickSpeechBone(positions: [0, 0, 0], parents: [nil]), 0)
    }

    func testJawAngleClampsAndScales() {
        XCTAssertEqual(SkeletalAnimator.jawAngle(amplitude: 0), 0, accuracy: 1e-9)
        let half = SkeletalAnimator.jawAngle(amplitude: 0.5)
        let full = SkeletalAnimator.jawAngle(amplitude: 1.0)
        XCTAssertGreaterThan(full, half)
        XCTAssertLessThanOrEqual(full, SkeletalAnimator.maxJawRadians + 1e-9)
        // Over-range input clamps.
        XCTAssertEqual(SkeletalAnimator.jawAngle(amplitude: 7), SkeletalAnimator.maxJawRadians, accuracy: 1e-9)
        XCTAssertEqual(SkeletalAnimator.jawAngle(amplitude: -1), 0, accuracy: 1e-9)
    }

    func testAmplitudeSmoothingAttacksFasterThanItReleases() {
        // Speech onset should open the jaw quickly; decay should be gentler
        // (abrupt closes look robotic).
        let up = SkeletalAnimator.smoothedAmplitude(previous: 0, target: 1)
        let down = SkeletalAnimator.smoothedAmplitude(previous: 1, target: 0)
        XCTAssertGreaterThan(up, 0.5, "attack should cover most of the gap")
        XCTAssertGreaterThan(down, 0.2, "release should keep residual openness")
        XCTAssertLessThan(1 - up, down, "attack must be faster than release")
    }

    func testEmoteOffsetsStartAndEndAtZeroAndStayBounded() {
        for emote in SkeletalAnimator.Emote.allCases {
            let atStart = SkeletalAnimator.emoteOffset(emote, elapsed: 0)
            XCTAssertEqual(atStart.head, 0, accuracy: 1e-9)
            XCTAssertEqual(atStart.root, 0, accuracy: 1e-9)
            let after = SkeletalAnimator.emoteOffset(emote, elapsed: SkeletalAnimator.emoteDuration + 0.01)
            XCTAssertEqual(after.head, 0)
            XCTAssertEqual(after.root, 0)
            // Peak exists but stays under 15°.
            var peak = 0.0
            for i in 0...100 {
                let o = SkeletalAnimator.emoteOffset(emote, elapsed: SkeletalAnimator.emoteDuration * Double(i) / 100.0)
                peak = max(peak, max(abs(o.head), abs(o.root)))
            }
            XCTAssertGreaterThan(peak, 0.05, "\(emote) should visibly move")
            XCTAssertLessThan(peak, 15.0 * .pi / 180.0, "\(emote) must stay subtle")
        }
    }

    func testNodMovesHeadOnlyAndSwayMovesRootOnly() {
        let nod = SkeletalAnimator.emoteOffset(.nod, elapsed: 0.4)
        XCTAssertNotEqual(nod.head, 0)
        XCTAssertEqual(nod.root, 0)
        let sway = SkeletalAnimator.emoteOffset(.sway, elapsed: 0.4)
        XCTAssertEqual(sway.head, 0)
        XCTAssertNotEqual(sway.root, 0)
    }

    func testIdleSwayIsSubtleAndDepthScaled() {
        // Deeper bones sway a touch more, but everything stays tiny (idle, not dance).
        let shallow = SkeletalAnimator.idleSwayRadians(depth: 0, time: 1.3)
        let deep = SkeletalAnimator.idleSwayRadians(depth: 4, time: 1.3)
        XCTAssertLessThanOrEqual(abs(shallow), abs(deep) + 1e-9)
        XCTAssertLessThan(abs(deep), 0.06, "idle sway must stay subtle (<~3.4°)")
        // Periodic: same phase a full period later.
        let later = SkeletalAnimator.idleSwayRadians(depth: 4, time: 1.3 + SkeletalAnimator.idlePeriod)
        XCTAssertEqual(deep, later, accuracy: 1e-6)
    }
}
