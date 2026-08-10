import XCTest
@testable import MLXCore

/// What an empty conversation says above its composer.
///
/// A thread belonging to an AGENT is not a blank app: you have already chosen
/// who you are talking to, so the greeting names them and says what they are
/// for, and the discovery chips — which advertise what the APP can do — stop
/// being an answer to any question the user is asking.
final class ChatGreetingTests: XCTestCase {

    // MARK: The greeting names the agent

    /// The name is the thing. "Agent" was the 30pt word with the name as its
    /// caption, which is the same inversion the sidebar rows had.
    func testAnAgentThreadIsHeadedByTheAgentsName() {
        XCTAssertEqual(ChatGreeting.heading(agentName: "Coder copy"), "Coder copy")
    }

    func testAPlainThreadKeepsTheAppsGreeting() {
        XCTAssertEqual(ChatGreeting.heading(agentName: nil), ChatGreeting.plainHeading)
    }

    /// A half-saved agent can't head a screen with an empty string.
    func testABlankAgentNameFallsBackToTheAppsGreeting() {
        XCTAssertEqual(ChatGreeting.heading(agentName: "   "), ChatGreeting.plainHeading)
    }

    /// The brief is what tells you what to ask this agent — the reason it, and
    /// not a row of app features, belongs under the name.
    func testTheSubtitleIsTheAgentsBrief() {
        XCTAssertEqual(ChatGreeting.subtitle(agentBrief: "a hands-on programmer",
                                             serverRunning: true),
                       "a hands-on programmer")
    }

    /// A stopped server outranks the brief: it is the one thing on this screen
    /// the user has to act on before anything else works.
    func testAStoppedServerOutranksTheBrief() {
        XCTAssertEqual(ChatGreeting.subtitle(agentBrief: "a hands-on programmer",
                                             serverRunning: false),
                       ChatGreeting.serverStopped)
        XCTAssertEqual(ChatGreeting.subtitle(agentBrief: nil, serverRunning: false),
                       ChatGreeting.serverStopped)
    }

    /// An agent with no description gets no caption rather than an empty line
    /// holding space under the name.
    func testNoBriefMeansNoSubtitle() {
        XCTAssertNil(ChatGreeting.subtitle(agentBrief: nil, serverRunning: true))
        XCTAssertNil(ChatGreeting.subtitle(agentBrief: "  ", serverRunning: true))
    }

    // MARK: The chips are for a blank app, not for an agent

    func testAPlainEmptyChatOffersTheDiscoveryChips() {
        XCTAssertTrue(ChatGreeting.showsDiscoveryChips(hasAgent: false, isExternalBridge: false))
    }

    /// Every chip navigates OUT of the conversation — offered directly above
    /// the field where you were about to type to someone specific. Worse, an
    /// agent whose capabilities exclude image generation cannot do what
    /// "Create Media" sits there offering.
    func testAnAgentThreadDoesNotOfferThem() {
        XCTAssertFalse(ChatGreeting.showsDiscoveryChips(hasAgent: true, isExternalBridge: false))
    }

    /// Unchanged: a Telegram thread is read-only, so it never offered them.
    func testAnExternalBridgeStillOffersNothing() {
        XCTAssertFalse(ChatGreeting.showsDiscoveryChips(hasAgent: false, isExternalBridge: true))
        XCTAssertFalse(ChatGreeting.showsDiscoveryChips(hasAgent: true, isExternalBridge: true))
    }

    /// Hiding the chips must not COST anything: every feature they advertise
    /// stays reachable from the Tools menu, which is their always-available
    /// twin and iterates the same catalog. Without this, dropping the row on
    /// agent threads would be the quiet-affordance-loss class.
    func testEveryChipFeatureIsStillReachableFromTheToolsMenu() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/MLXServeApp.swift")
        let s = try String(contentsOf: url, encoding: .utf8)
        XCTAssertTrue(s.contains("CommandMenu(\"Tools\")"),
                      "the Tools menu is what keeps these features reachable once the chips are gone")
        for needle in ["mediaItems", "showModels", "showTasks"] {
            XCTAssertTrue(s.contains(needle),
                          "the Tools menu must still reach \(needle)")
        }
    }
}
