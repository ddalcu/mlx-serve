import XCTest
@testable import MLXCore

/// What the player deck's keys do, in either look, with AI Radio off and on.
final class DeckActionsTests: XCTestCase {

    /// The radio owns playback while it runs: it locks the keys that pick the next track and takes Stop and Next.
    func testEveryKeyRoutesTheSameInBothLooks() {
        let off: [DeckKey: DeckRoute] = [
            .previous: .own, .playPause: .own, .stop: .own, .next: .own, .shuffle: .own, .repeatAll: .own,
            .export: .own, .reveal: .own, .radio: .radio, .historyRow: .own,
        ]
        let on: [DeckKey: DeckRoute] = [
            .previous: .locked, .playPause: .own, .stop: .radio, .next: .radio, .shuffle: .locked, .repeatAll: .locked,
            .export: .own, .reveal: .own, .radio: .radio, .historyRow: .locked,
        ]
        XCTAssertEqual(Set(off.keys), Set(DeckKey.allCases), "every key has a row in the table")
        for key in DeckKey.allCases {
            XCTAssertEqual(key.route(radioOn: false), off[key], "\(key), radio off")
            XCTAssertEqual(key.route(radioOn: true), on[key], "\(key), radio on")
        }
    }

    func testOnlyTheLockedRouteLocks() {
        XCTAssertTrue(DeckKey.shuffle.isLocked(radioOn: true))
        XCTAssertFalse(DeckKey.shuffle.isLocked(radioOn: false))
        XCTAssertFalse(DeckKey.stop.isLocked(radioOn: true), "Stop goes to the radio, it is not locked")
    }
}
