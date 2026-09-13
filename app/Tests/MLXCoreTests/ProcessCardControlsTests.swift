import XCTest
@testable import MLXCore

/// Pure presentation seam for the tool-call card's kill X.
final class ProcessCardControlsTests: XCTestCase {

    func testAliveHandleYieldsButton() {
        let live: Set<String> = ["bg1"]
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1"]) { live.contains($0) }, ["bg1"])
    }

    func testDeadOrUnknownYieldsNone() {
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1", "bg2"]) { _ in false }, [])
    }

    func testNilHandlesYieldsNone() {
        XCTAssertEqual(ProcessCardControls.killable(handles: nil) { _ in true }, [])
    }

    func testMixedKeepsOnlyLive() {
        let live: Set<String> = ["bg2"]
        XCTAssertEqual(ProcessCardControls.killable(handles: ["bg1", "bg2", "bg3"]) { live.contains($0) }, ["bg2"])
    }

    /// Handles persisted on a message survive a restart, but the registry isn't
    /// persisted — so on load nothing is alive → no kill buttons.
    @MainActor
    func testPersistedHandlesAfterRestartYieldNone() {
        let freshRegistry = ProcessRegistry()
        let out = ProcessCardControls.killable(handles: ["bg1", "bg2"], isAlive: freshRegistry.isAlive)
        XCTAssertEqual(out, [])
    }

    // MARK: - Which CHAT a handle belongs to

    /// Ownership within one transcript is only half the answer: the counter is
    /// app-wide, so `bg1` in an old chat and `bg1` started today in another
    /// chat are the same NAME. Asked globally, the old card lit up as running
    /// and its ✕ would have killed a stranger's process.
    ///
    /// `registerSandboxed` is the one entry point that registers without
    /// launching anything, which is what makes this testable.
    @MainActor
    func testAHandleIsOnlyAliveForTheChatThatStartedIt() {
        let registry = ProcessRegistry()
        let mine = UUID(), theirs = UUID()
        let p = registry.registerSandboxed(command: "sleep 9", guestPID: 42,
                                           logPath: "/tmp/x.log", sessionId: mine)

        XCTAssertTrue(registry.isAlive(handle: p.handle, sessionId: mine))
        XCTAssertFalse(registry.isAlive(handle: p.handle, sessionId: theirs),
                       "another chat's card must not claim this process")
    }

    /// A surface with no session of its own (a task run's transcript) keeps the
    /// old, global answer rather than losing its badge.
    @MainActor
    func testASessionlessCallerStillSeesTheProcess() {
        let registry = ProcessRegistry()
        let p = registry.registerSandboxed(command: "sleep 9", guestPID: 42,
                                           logPath: "/tmp/x.log", sessionId: UUID())
        XCTAssertTrue(registry.isAlive(handle: p.handle, sessionId: nil))
    }

    @MainActor
    func testAnUnknownHandleIsAliveForNobody() {
        let registry = ProcessRegistry()
        XCTAssertFalse(registry.isAlive(handle: "bg1", sessionId: UUID()))
        XCTAssertFalse(registry.isAlive(handle: "bg1", sessionId: nil))
    }

    // MARK: - Which card a reused handle belongs to

    /// Numbering restarts at `bg1` every launch, and the registry knows only the
    /// name. So a card from an old session asked "is bg1 alive?", got yes about
    /// a process started today, and offered to kill it. The LAST card to
    /// announce a handle owns it.
    func testAReusedHandleBelongsToTheLatestCardOnly() {
        let owned = ProcessCardControls.handleOwnership([(1, ["bg1"]), (2, ["bg1"])])
        XCTAssertEqual(owned[1] ?? [], [])
        XCTAssertEqual(owned[2] ?? [], ["bg1"])
    }

    /// Handles are independent: losing `bg1` to a later card must not take the
    /// card's own `bg2` with it.
    func testOwnershipIsPerHandleNotPerCard() {
        let owned = ProcessCardControls.handleOwnership([(1, ["bg1", "bg2"]), (2, ["bg1"])])
        XCTAssertEqual(owned[1] ?? [], ["bg2"])
        XCTAssertEqual(owned[2] ?? [], ["bg1"])
    }

    /// One round can start several processes, and one card can announce a
    /// handle twice — neither may duplicate a pill.
    func testACardKeepsEveryHandleNobodyElseClaimed() {
        let owned = ProcessCardControls.handleOwnership([(1, ["bg1", "bg2", "bg1"])])
        XCTAssertEqual(owned[1] ?? [], ["bg1", "bg2"])
    }

    // MARK: - Header pill vs pill beside the call

    /// A multi-tool card puts each pill beside the tool that started it, so the
    /// header shows only what no call claimed — plus, while collapsed, a
    /// button-less "running" for the claimed ones, which are hidden with the
    /// panel.
    func testLiveHandlesSplitIntoClaimedAndUnclaimed() {
        let split = ProcessCardControls.split(live: ["bg1", "bg2", "bg3"],
                                              claimedBy: ["bg1", nil, "bg3"])
        XCTAssertEqual(split.claimed, ["bg1", "bg3"])
        XCTAssertEqual(split.unclaimed, ["bg2"])
    }

    /// A call whose result named a handle that has since exited claims nothing:
    /// the header must not grow a pill for a dead process.
    func testAClaimOnADeadHandleAddsNothing() {
        let split = ProcessCardControls.split(live: [], claimedBy: ["bg1"])
        XCTAssertTrue(split.claimed.isEmpty)
        XCTAssertTrue(split.unclaimed.isEmpty)
    }

    func testNoAnnouncementsOwnNothing() {
        XCTAssertTrue(ProcessCardControls.handleOwnership([(Int, [String])]()).isEmpty)
    }
}
