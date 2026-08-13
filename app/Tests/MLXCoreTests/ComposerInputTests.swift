import XCTest
@testable import MLXCore

/// Pins the pure layout + key logic behind the auto-growing composer input
/// (GrowingTextEditor). The NSViewRepresentable itself is untestable UI, so the
/// height-clamping and Return-key decision are factored out and tested here.
final class ComposerInputTests: XCTestCase {

    // MARK: - ComposerLayout.resolve (auto-grow then scroll)

    func testSingleLineFloorsAtMinHeight() {
        // One short line of content must not collapse below the 1-line floor.
        let r = ComposerLayout.resolve(contentHeight: 5, lineHeight: 20,
                                       minLines: 1, maxLines: 15, verticalInset: 16)
        XCTAssertEqual(r.height, 20 * 1 + 16, accuracy: 0.01)
        XCTAssertFalse(r.scrolls)
    }

    func testGrowsWithContentBelowTheCap() {
        // 10 lines of content sits between the min and max — height tracks it,
        // no scrolling yet.
        let r = ComposerLayout.resolve(contentHeight: 200, lineHeight: 20,
                                       minLines: 1, maxLines: 15, verticalInset: 16)
        XCTAssertEqual(r.height, 200 + 16, accuracy: 0.01)
        XCTAssertFalse(r.scrolls)
    }

    func testClampsAtMaxAndScrolls() {
        // A big paste (30 lines) clamps at the 15-line cap and turns scrolling on
        // — the bug being fixed: TextField(axis:.vertical) never scrolled.
        let r = ComposerLayout.resolve(contentHeight: 600, lineHeight: 20,
                                       minLines: 1, maxLines: 15, verticalInset: 16)
        XCTAssertEqual(r.height, 20 * 15 + 16, accuracy: 0.01)
        XCTAssertTrue(r.scrolls)
    }

    func testExactlyAtCapDoesNotScroll() {
        let r = ComposerLayout.resolve(contentHeight: 20 * 15, lineHeight: 20,
                                       minLines: 1, maxLines: 15, verticalInset: 16)
        XCTAssertEqual(r.height, 20 * 15 + 16, accuracy: 0.01)
        XCTAssertFalse(r.scrolls)
    }

    // MARK: - ComposerKey.onReturn (send vs newline vs swallow)

    func testShiftReturnAlwaysInsertsNewline() {
        XCTAssertEqual(ComposerKey.onReturn(shift: true, isIdle: true), .newline)
        XCTAssertEqual(ComposerKey.onReturn(shift: true, isIdle: false), .newline)
    }

    func testBareReturnSendsWhenIdle() {
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: true), .send)
    }

    func testBareReturnSwallowedWhileGenerating() {
        // While a turn is generating a bare Return must NOT insert a stray
        // newline and must NOT start a second send — it is swallowed.
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: false), .ignore)
    }

    // MARK: - ComposerIntent.wantsAgent (system/file/web action detection)

    func testAgentIntentFiresOnToolTasks() {
        // These are the contract examples shown in the picker preview.
        XCTAssertTrue(ComposerIntent.wantsAgent("create an index.html and run it"))
        XCTAssertTrue(ComposerIntent.wantsAgent("search the web for the latest swift news"))
        // Plus other clear build/do tasks.
        XCTAssertTrue(ComposerIntent.wantsAgent("make me a website"))
        XCTAssertTrue(ComposerIntent.wantsAgent("build a game"))
        XCTAssertTrue(ComposerIntent.wantsAgent("fix the bug in app.js"))
        XCTAssertTrue(ComposerIntent.wantsAgent("npm install react"))
        XCTAssertTrue(ComposerIntent.wantsAgent("clone the repo and list files"))
    }

    func testAgentIntentStaysQuietOnOrdinaryChat() {
        // The contract examples that must NOT fire.
        XCTAssertFalse(ComposerIntent.wantsAgent("write a poem"))
        XCTAssertFalse(ComposerIntent.wantsAgent("explain how DNS works"))
        // Other plain questions / content requests.
        XCTAssertFalse(ComposerIntent.wantsAgent("what is the capital of France"))
        XCTAssertFalse(ComposerIntent.wantsAgent("summarize this article for me"))
        XCTAssertFalse(ComposerIntent.wantsAgent("tell me a story about a dragon"))
    }

    // MARK: - ComposerIntent.wantsMCP (enabled server names + "mcp")

    func testMCPIntentMatchesServerNameOrLiteralMCP() {
        let servers = ["github", "linear"]
        XCTAssertTrue(ComposerIntent.wantsMCP("check my github issues", serverNames: servers))
        XCTAssertTrue(ComposerIntent.wantsMCP("use the mcp server", serverNames: servers))
        XCTAssertTrue(ComposerIntent.wantsMCP("create a linear ticket for this", serverNames: servers))
    }

    func testMCPIntentQuietWithoutAMatch() {
        let servers = ["github", "linear"]
        XCTAssertFalse(ComposerIntent.wantsMCP("write a poem about cats", serverNames: servers))
        // The word "mcp" alone still matches (the view separately gates on there
        // being enabled servers), but an unrelated message with no servers does not.
        XCTAssertFalse(ComposerIntent.wantsMCP("what time is it", serverNames: []))
    }

    // MARK: - ComposerIntent.nudge (the whole pre-send decision, in one place)

    private func nudge(_ text: String,
                       onlyWhenAsked: Bool = false,
                       toolsOn: Bool = false, toolsLocked: Bool = false,
                       mcpOn: Bool = false, mcpLocked: Bool = false,
                       servers: [String] = [],
                       suppressed: SessionIntentSuppression = SessionIntentSuppression(),
                       session: UUID = UUID()) -> IntentPrompt? {
        ComposerIntent.nudge(for: text, onlyToolsWhenAsked: onlyWhenAsked,
                             toolsOn: toolsOn, toolsLocked: toolsLocked,
                             mcpOn: mcpOn, mcpLocked: mcpLocked,
                             enabledServers: servers, suppressed: suppressed,
                             sessionId: session)
    }

    func testNudgeOffersToolsForAnAgenticMessage() {
        XCTAssertEqual(nudge("make me a website"), .agent)
    }

    /// A named server is the more specific signal, so MCP wins the tie.
    func testNudgeOffersMCPAheadOfToolsWhenAServerIsNamed() {
        XCTAssertEqual(nudge("create a linear ticket for this", servers: ["linear"]), .mcp)
    }

    func testNudgeIsSilentWhenTheModeIsAlreadyOnOrLockedOrDeclined() {
        XCTAssertNil(nudge("make me a website", toolsOn: true))
        XCTAssertNil(nudge("make me a website", toolsLocked: true),
                     "offering a mode the tab's agent decides is a dead offer")
        XCTAssertNil(nudge("check my github issues", mcpOn: true, servers: ["github"]))
        XCTAssertNil(nudge("check my github issues", mcpLocked: true, servers: ["github"]))

        let chat = UUID()
        var declined = SessionIntentSuppression()
        declined.suppress(.agent, for: chat)
        XCTAssertNil(nudge("make me a website", suppressed: declined, session: chat))
        // …but only for the chat that declined it.
        XCTAssertEqual(nudge("make me a website", suppressed: declined, session: UUID()), .agent)
    }

    /// MCP needs a server to name — with none enabled there is nothing to turn on.
    func testNudgeNeverOffersMCPWithNoEnabledServers() {
        XCTAssertNil(nudge("use the mcp server", servers: []))
    }

    // MARK: - "Only use tools when I ask" (Settings ▸ Agent)

    /// The whole point of the setting: tools are opt-in, so the composer never
    /// interrupts a send to offer them. Both nudges go quiet, and the
    /// per-message detection is bypassed entirely.
    func testToolOptInSettingSilencesEveryNudge() {
        XCTAssertNil(nudge("make me a website", onlyWhenAsked: true))
        XCTAssertNil(nudge("npm install react", onlyWhenAsked: true))
        XCTAssertNil(nudge("check my github issues", onlyWhenAsked: true, servers: ["github"]),
                     "the MCP nudge is the same interruption — the setting covers it too")
        // Sanity: the same messages DO nudge with the setting off.
        XCTAssertEqual(nudge("make me a website"), .agent)
        XCTAssertEqual(nudge("check my github issues", servers: ["github"]), .mcp)
    }

    // MARK: - SessionIntentSuppression (stop nagging per chat, keyed by session)

    func testSuppressionIsPerPromptAndPerSession() {
        let s1 = UUID(), s2 = UUID()
        var sup = SessionIntentSuppression()
        XCTAssertFalse(sup.isSuppressed(.agent, for: s1))

        sup.suppress(.agent, for: s1)
        XCTAssertTrue(sup.isSuppressed(.agent, for: s1))
        // Other prompt in the same session is unaffected.
        XCTAssertFalse(sup.isSuppressed(.mcp, for: s1))
        // Same prompt in a different session is unaffected (no cross-tab leak).
        XCTAssertFalse(sup.isSuppressed(.agent, for: s2))
    }
}
