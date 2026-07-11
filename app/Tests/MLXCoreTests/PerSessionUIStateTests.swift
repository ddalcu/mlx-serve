import XCTest
@testable import MLXCore

/// Per-tab UI-state regressions. The text-chat window reuses one `ChatDetailView`
/// across `sessionId` changes (SwiftUI swaps the `sessionId` `let` rather than
/// rebuilding the view), so any app-wide `@State`/engine flag that isn't keyed by
/// session id leaks — or gets wiped — across chat tabs. Two such flags bit users:
///
///  1. "Allow all tools this session" was a single `Bool` that an explicit
///     `.onChange(of: sessionId)` reset on every tab switch → granting it in tab A
///     was forgotten the moment you looked at tab B (and on return to A).
///  2. The Stop button read the engine's single `isGenerating` → every tab showed
///     Stop while ANY chat was mid-turn.
///
/// Both fixes scope the decision to the session id; these pin the pure helpers.
final class PerSessionUIStateTests: XCTestCase {

    // MARK: - "Allow all this session" is remembered per tab

    func testAllowAllIsScopedPerSession() {
        let a = UUID(); let b = UUID()
        var store = SessionToolAllowList()
        XCTAssertFalse(store.allowsAll(a))
        store.allowAll(a)
        XCTAssertTrue(store.allowsAll(a), "granting allow-all for A is remembered")
        XCTAssertFalse(store.allowsAll(b), "allow-all must NOT leak into another tab")
    }

    func testSwitchingTabsAndBackPreservesAllowAll() {
        // The live bug: allow-all in A, switch to B, switch back to A → A forgot.
        // A keyed store has nothing to wipe on a switch, so A survives any number
        // of switches; only an explicit re-arm clears a session's entry.
        let a = UUID(); let b = UUID()
        var store = SessionToolAllowList()
        store.allowAll(a)
        // (viewing B then returning to A performs no mutation on the store)
        XCTAssertTrue(store.allowsAll(a), "A still trusted after switching away and back")
        XCTAssertFalse(store.allowsAll(b), "B was never granted")
    }

    func testRearmReprompsOnlyThatSession() {
        let a = UUID(); let b = UUID()
        var store = SessionToolAllowList()
        store.allowAll(a); store.allowAll(b)
        store.rearm(a)   // e.g. user toggled Agent off in tab A
        XCTAssertFalse(store.allowsAll(a), "re-arming A makes A prompt again")
        XCTAssertTrue(store.allowsAll(b), "B is untouched by A's re-arm")
    }

    // MARK: - Stop / generating indicator scoped per chat

    func testComposerIdleWhenNothingGenerating() {
        let s = UUID()
        XCTAssertEqual(ChatTurnEngine.composerState(isGenerating: false, activeTurnSessionId: nil, for: s), .idle)
        XCTAssertEqual(ChatTurnEngine.composerState(isGenerating: false, activeTurnSessionId: s, for: s), .idle)
    }

    func testComposerGeneratingHereOnlyForActiveTurnSession() {
        let a = UUID(); let b = UUID()
        XCTAssertEqual(ChatTurnEngine.composerState(isGenerating: true, activeTurnSessionId: a, for: a), .generatingHere,
                       "the chat that owns the in-flight turn shows Stop")
        XCTAssertEqual(ChatTurnEngine.composerState(isGenerating: true, activeTurnSessionId: a, for: b), .busyElsewhere,
                       "a different chat never shows Stop — it shows Send (disabled while busy)")
    }

    func testComposerBusyElsewhereWhenActiveSessionUnknown() {
        let s = UUID()
        XCTAssertEqual(ChatTurnEngine.composerState(isGenerating: true, activeTurnSessionId: nil, for: s), .busyElsewhere,
                       "generating with no recorded owner is still 'not this chat'")
    }

    // MARK: - Think / MCP toggles persist per session (ChatSession migration)

    func testThinkAndMCPToggleSurviveCodableRoundTrip() throws {
        var s = ChatSession(title: "t")
        s.enableThinking = true
        s.useMCP = true
        s.mode = .agent
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(ChatSession.self, from: data)
        XCTAssertTrue(back.enableThinking, "Think toggle is remembered per chat across save/load")
        XCTAssertTrue(back.useMCP, "MCP toggle is remembered per chat across save/load")
        XCTAssertEqual(back.mode, .agent)
    }

    /// The attached document folder (mini RAG) persists per session so it can be
    /// re-indexed after a relaunch — under the App Sandbox the matching
    /// security-scoped bookmark is what makes the path reachable again.
    func testAttachedFolderPathSurvivesCodableRoundTrip() throws {
        var s = ChatSession(title: "t")
        s.attachedFolderPath = "/Users/someone/Documents/papers"
        let back = try JSONDecoder().decode(ChatSession.self, from: JSONEncoder().encode(s))
        XCTAssertEqual(back.attachedFolderPath, "/Users/someone/Documents/papers")
    }

    func testLegacySessionWithoutAttachedFolderDecodesToNil() throws {
        let legacy = """
        {"id":"\(UUID().uuidString)","title":"old","messages":[],
         "createdAt":0,"updatedAt":0,"mode":"chat","isExternalBridge":false}
        """.data(using: .utf8)!
        let back = try JSONDecoder().decode(ChatSession.self, from: legacy)
        XCTAssertNil(back.attachedFolderPath, "missing key → no folder, not a decode failure")
    }

    func testLegacySessionWithoutTogglesDecodesToOff() throws {
        // A chat saved before these fields existed must still decode (no throw),
        // defaulting both toggles off — same back-compat pattern as workingDirectory.
        let legacy = """
        {"id":"\(UUID().uuidString)","title":"old","messages":[],
         "createdAt":0,"updatedAt":0,"mode":"chat","isExternalBridge":false}
        """.data(using: .utf8)!
        let dec = JSONDecoder()
        let back = try dec.decode(ChatSession.self, from: legacy)
        XCTAssertFalse(back.enableThinking, "missing Think key → off, not a decode failure")
        XCTAssertFalse(back.useMCP, "missing MCP key → off, not a decode failure")
    }
}
