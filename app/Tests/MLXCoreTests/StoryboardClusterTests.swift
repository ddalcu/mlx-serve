import XCTest
@testable import MLXCore

final class StoryboardClusterTests: XCTestCase {

    private func row(_ name: String, caps: [String] = ["video"], peer: String?) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 0, hiddenSize: 0, vocabSize: 0, contextLength: 0,
                  modelMaxTokens: 0, supportsVision: false, supportsEmbeddings: false,
                  capabilities: caps, lanPeer: peer)
    }

    /// Only peers serving the same pack id, as a video model, render shots.
    func testPeersAreTheSamePackOnOtherMachines() {
        let pack = "MiniMax-H3-FL2VA-MLX-Serve-4bit"
        let peers = StoryboardCluster.peers(of: "ddalcu/\(pack)", lanModels: [
            row("\(pack)@m4max", peer: "m4max"),
            row("MiniMax-H3-FL2VA-MLX-Serve-8bit@studio", peer: "studio"),
            row("\(pack)@chatbox", caps: ["chat"], peer: "chatbox"),
            row("ddalcu/\(pack)@5060-linux", peer: "5060-linux"),
        ])
        XCTAssertEqual(peers.map(\.peer), ["m4max", "5060-linux"])
        XCTAssertEqual(peers.first?.modelId, "\(pack)@m4max")
    }

    /// The film's final shot needs no join picture; every other one does.
    func testJoinsAnchoredIgnoresTheFinalShot() {
        XCTAssertTrue(StoryboardCluster.joinsAnchored(["a", "b", nil]))
        XCTAssertFalse(StoryboardCluster.joinsAnchored(["a", nil, nil]))
        XCTAssertTrue(StoryboardCluster.joinsAnchored([nil]))
    }

    /// Two equal machines halve four equal shots.
    func testEqualMachinesSplitTheShots() {
        XCTAssertEqual(StoryboardCluster.wallSeconds(costs: [10, 10, 10, 10], factors: [1, 1]), 20, accuracy: 1e-9)
    }

    /// A machine 7x slower never takes the last shot when the fast one would finish it first.
    func testTheSlowMachineTakesOnlyShotsItFinishesSooner() {
        let alone = StoryboardCluster.wallSeconds(costs: Array(repeating: 10, count: 8), factors: [1])
        let both = StoryboardCluster.wallSeconds(costs: Array(repeating: 10, count: 8), factors: [1, 7])
        XCTAssertEqual(alone, 80, accuracy: 1e-9)
        XCTAssertEqual(both, 70, accuracy: 1e-9)
        XCTAssertEqual(StoryboardCluster.wallSeconds(costs: [10, 10], factors: [1, 7]), 20, accuracy: 1e-9)
    }

    /// A failed device's shot goes back in line and the device takes no more.
    func testAFailedShotReturnsToTheQueue() {
        var d = ShotDispatcher(costs: [30, 20], factors: [1, 1])
        XCTAssertEqual(d.next(for: 0, now: 0), 0)
        XCTAssertEqual(d.next(for: 1, now: 0), 1)
        d.failed(1, shot: 1)
        XCTAssertNil(d.next(for: 1, now: 5))
        d.finished(0)
        XCTAssertEqual(d.next(for: 0, now: 30), 1)
        d.finished(0)
        XCTAssertTrue(d.done)
    }
    /// A Mac guessed as fast as this one turns out 7x slower: once this Mac runs out of
    /// shots it renders that shot again, and the first copy to finish wins.
    func testAnIdleMacRerunsAShotASlowerMacIsStillOn() {
        var d = ShotDispatcher(costs: [10, 10], factors: [1, 1])
        XCTAssertEqual(d.next(for: 0, now: 0), 0)
        XCTAssertEqual(d.next(for: 1, now: 0), 1)
        d.measured(1, factor: 7)
        XCTAssertEqual(d.finished(0), [])
        XCTAssertEqual(d.next(for: 0, now: 10), 1)
        XCTAssertEqual(d.finished(0), [1], "the slow copy is the one to cancel")
        XCTAssertTrue(d.done)
    }

    /// No second copy of a shot that is about to finish where it is.
    func testNoRerunOfAShotThatEndsSoonAnyway() {
        var d = ShotDispatcher(costs: [10, 12], factors: [1, 1])
        XCTAssertEqual(d.next(for: 0, now: 0), 1)
        XCTAssertEqual(d.next(for: 1, now: 0), 0)
        _ = d.finished(1)
        XCTAssertNil(d.next(for: 1, now: 10))
    }

    /// With a slow Mac misjudged, the film still ends when this Mac alone would end it.
    func testAMisjudgedSlowMacNeverMakesTheFilmLater() {
        var d = ShotDispatcher(costs: [10, 10], factors: [1, 1])
        _ = d.next(for: 0, now: 0)
        _ = d.next(for: 1, now: 0)
        d.measured(1, factor: 7)
        _ = d.finished(0)
        _ = d.next(for: 0, now: 10)
        XCTAssertEqual(d.busyUntil[0], 20, accuracy: 1e-9)
    }
    /// A shot the user put on a machine renders there, however slow it is.
    func testAPinnedShotRendersOnItsMachine() {
        let plan = StoryboardCluster.simulate(costs: [10, 10], factors: [1, 7], pins: [nil, 1])
        XCTAssertEqual(plan.machine, [0, 1])
        XCTAssertEqual(plan.seconds, 70, accuracy: 1e-9)
        XCTAssertEqual(StoryboardCluster.simulate(costs: [10, 10], factors: [1, 7], pins: [nil, nil]).machine, [0, 0])
    }

    /// No second copy of a pinned shot elsewhere: the user chose where it runs.
    func testAPinnedShotIsNeverRerunElsewhere() {
        var d = ShotDispatcher(costs: [10, 10], factors: [1, 1], pins: [nil, 1])
        XCTAssertEqual(d.next(for: 0, now: 0), 0)
        XCTAssertEqual(d.next(for: 1, now: 0), 1)
        d.measured(1, factor: 7)
        d.finished(0)
        XCTAssertNil(d.next(for: 0, now: 10))
    }

    /// A pinned machine that fails gives its shot to the others.
    func testAFailedMachinesPinnedShotGoesToAnother() {
        var d = ShotDispatcher(costs: [10, 10], factors: [1, 1], pins: [1, 1])
        XCTAssertNil(d.next(for: 0, now: 0))
        XCTAssertNotNil(d.next(for: 1, now: 0))
        d.failed(1, shot: 0)
        XCTAssertNotNil(d.next(for: 0, now: 1))
    }
}
