import XCTest
@testable import MLXCore

/// Pins which source the chat toolbar's Think/Agent/MCP toggles reflect: a
/// Telegram bridge session must mirror `serverOptions.telegram` (so the toolbar
/// stays in sync with Settings — one source of truth), while a normal session
/// uses the in-app per-session / app-level state.
final class ChatModeTogglesTests: XCTestCase {

    func testTelegramSessionReflectsTelegramConfigNotInApp() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: true,
            telegramThinking: true, telegramAgent: true, telegramMCP: false,
            inAppThinking: false, inAppAgent: false, inAppMCP: true)   // in-app differs → must be ignored
        XCTAssertEqual(t, ChatModeToggles(thinking: true, agent: true, mcp: false))
    }

    func testNormalSessionReflectsInAppStateNotTelegram() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: true, telegramAgent: true, telegramMCP: true,   // telegram differs → ignored
            inAppThinking: false, inAppAgent: true, inAppMCP: false)
        XCTAssertEqual(t, ChatModeToggles(thinking: false, agent: true, mcp: false))
    }

    // MARK: - Agent lock
    //
    // `AgentResolution` decides Tools and MCP from the agent's capabilities and
    // ignores the chat's own toggles entirely — so before this, a chat with an
    // agent showed one thing on the discs and ran another (every agent defaults
    // `web: true`, which forces the tool loop on while the wrench reads OFF).
    // The discs now show what the agent decided and say so.

    private let chef = AgentModeLock(name: "Chef", thinking: nil, tools: true, mcp: false)

    func testAgentDecidesToolsAndMcpAndTheDiscsSayWhoDid() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: false, telegramAgent: false, telegramMCP: false,
            inAppThinking: false, inAppAgent: false, inAppMCP: true,   // the chat's own → overridden
            agentLock: chef)
        XCTAssertTrue(t.agent, "the agent's capabilities decide the tool loop")
        XCTAssertFalse(t.mcp, "the agent's capabilities decide MCP")
        XCTAssertEqual(t.toolsLockedBy, "Chef")
        XCTAssertEqual(t.mcpLockedBy, "Chef")
    }

    /// Thinking is the one an agent may leave unset (`enableThinking: nil`), and
    /// `AgentResolution` falls back to the surface's own value there. Locking it
    /// anyway would take away a control nobody is deciding for you.
    func testThinkingStaysTheChatsOwnWhenTheAgentDidNotDecideIt() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: false, telegramAgent: false, telegramMCP: false,
            inAppThinking: true, inAppAgent: false, inAppMCP: false,
            agentLock: chef)
        XCTAssertTrue(t.thinking, "the chat's own thinking stands")
        XCTAssertNil(t.thinkingLockedBy, "nothing is deciding it, so it must not read as locked")
    }

    func testThinkingIsLockedWhenTheAgentDecidedIt() {
        let strict = AgentModeLock(name: "Coder", thinking: false, tools: true, mcp: false)
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: false, telegramAgent: false, telegramMCP: false,
            inAppThinking: true, inAppAgent: false, inAppMCP: false,   // chat says on → overridden
            agentLock: strict)
        XCTAssertFalse(t.thinking)
        XCTAssertEqual(t.thinkingLockedBy, "Coder")
    }

    /// A bridge session running as an agent is still that agent's conversation.
    func testTheLockAppliesOverTheTelegramSourceToo() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: true,
            telegramThinking: true, telegramAgent: false, telegramMCP: true,
            inAppThinking: false, inAppAgent: false, inAppMCP: false,
            agentLock: chef)
        XCTAssertTrue(t.agent)
        XCTAssertFalse(t.mcp)
        XCTAssertTrue(t.thinking, "unset by the agent → the bridge's own value, not the in-app one")
    }

    func testNoAgentLeavesEveryControlUnlocked() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: false, telegramAgent: false, telegramMCP: false,
            inAppThinking: true, inAppAgent: true, inAppMCP: true,
            agentLock: nil)
        XCTAssertEqual(t, ChatModeToggles(thinking: true, agent: true, mcp: true))
        XCTAssertFalse(t.isLocked)
    }

    func testIsLockedIsTrueWhenAnyControlIsLocked() {
        XCTAssertTrue(ChatModeToggles(thinking: false, agent: true, mcp: false,
                                      toolsLockedBy: "Chef").isLocked)
        XCTAssertFalse(ChatModeToggles(thinking: false, agent: true, mcp: false).isLocked)
    }
}
