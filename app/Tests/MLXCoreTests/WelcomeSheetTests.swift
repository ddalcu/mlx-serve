import XCTest
@testable import MLXCore

/// The welcome screen is a SHEET on the chat window, not a floating window of
/// its own.
///
/// As a window it was `.level = .floating` with its own `NSHostingView`, which
/// bought three problems: it inherited NO environment (every object the starter
/// card reads had to be hand-injected or SwiftUI trapped at first render), it
/// could be left on screen with nothing behind it, and every way out of it had
/// to remember to open Chat or the user landed on an empty desktop
/// (`WelcomeExit`, live dead end 2026-08-08). A sheet is attached to the chat
/// window, so the thing behind it is always a composer — the dead end stops
/// being a rule to keep and becomes impossible to build.
final class WelcomeSheetTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: The chat window opens either way

    /// A sheet with no host window is a screen nobody can see, so the launch
    /// opens Chat in BOTH branches now — `showWelcome` used to skip it.
    func testTheChatWindowOpensOnEveryLaunchBranch() {
        for decision in [LaunchDecision.showWelcome, .openChat] {
            XCTAssertTrue(decision.opensChatWindow,
                          "\(decision) must open the window the welcome sheet hangs on")
        }
    }

    func testOnlyTheWelcomeBranchPresentsTheSheet() {
        XCTAssertTrue(LaunchDecision.showWelcome.presentsWelcome)
        XCTAssertFalse(LaunchDecision.openChat.presentsWelcome)
    }

    /// Unchanged: ticking "Don't show this again" is honoured whether or not
    /// anything is downloaded — the chat gate offers the same starter card.
    func testSuppressionStillDecidesTheBranch() {
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: true, hasChatModels: false),
                       .openChat)
        XCTAssertEqual(LaunchDecision.resolve(welcomeSuppressed: false, hasChatModels: true),
                       .showWelcome)
    }

    // MARK: Only one sheet at a time

    /// The chat window already presents a blocking "you need a model first"
    /// gate. Two sheets on one window is one sheet plus a thing nobody can
    /// see, and the welcome answers the gate's question BETTER — it is the
    /// screen with the starter models on it. So the gate stands down while the
    /// welcome is up, the same way it stands down over the models pane:
    /// deferred, not dismissed.
    func testTheModelGateStandsDownWhileTheWelcomeIsUp() {
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation, welcomePresented: true))
    }

    /// …and returns the moment the welcome closes, if there is still no model.
    func testTheGateReturnsWhenTheWelcomeCloses() {
        XCTAssertTrue(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation, welcomePresented: false))
    }

    /// Every other reason the gate stands down is untouched.
    func testTheGatesExistingRulesAreUnchanged() {
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .models(.recommended),
                                            welcomePresented: false))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: true,
                                            workspace: .conversation, welcomePresented: false))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: false, cancelled: false,
                                            workspace: .conversation, welcomePresented: false))
    }

    // MARK: It really is a sheet on that window

    /// Presented from the chat scene, AFTER the environment objects — a sheet
    /// inherits the environment of the view it hangs on, which is what retires
    /// the hand-injection the `NSHostingView` needed.
    func testTheWelcomeIsPresentedAsASheetOnTheChatScene() throws {
        let s = try source("Sources/MLXServe/MLXServeApp.swift")
        guard let chat = s.range(of: "Window(\"MLX Core\", id: \"chat\")") else {
            return XCTFail("the chat scene moved — update this audit")
        }
        let after = String(s[chat.upperBound...])
        guard let sheet = after.range(of: "showWelcome") else {
            return XCTFail("the chat scene must present the welcome sheet")
        }
        let block = String(after[..<sheet.upperBound])
        XCTAssertTrue(block.contains(".sheet("),
                      "the welcome is a sheet on the chat window")
        XCTAssertTrue(block.contains("WelcomeView(") || after.contains("WelcomeView("),
                      "…presenting the welcome screen itself")
    }

    /// No second window for it anywhere. An `NSWindow` built by hand is how it
    /// got a floating level and an empty environment in the first place.
    func testNothingBuildsAWelcomeWindowAnyMore() throws {
        let s = try source("Sources/MLXServe/AppState.swift")
        for gone in ["showWelcomeWindow", "_welcomeWindow"] {
            XCTAssertFalse(s.contains(gone),
                           "\(gone) is retired — the welcome is a sheet on the chat window")
        }
    }
}
