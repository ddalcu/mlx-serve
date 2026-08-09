import XCTest
@testable import MLXCore

/// Source audit for the composer's Tools / MCP discs.
///
/// These two are flipped constantly, and while the glyph OPENED a menu the
/// frequent action cost a click plus a row in a long list. Click now TOGGLES
/// (`primaryAction:`) and the configuration list moved to secondary-click
/// (plus press-and-hold, which is what `primaryAction:` gives us natively).
///
/// A source audit rather than a behavior test because there is no seam here: a
/// `Menu` that quietly loses its `primaryAction:` still compiles, still renders
/// identically, and only differs in what one click does — exactly the kind of
/// regression nothing else in the suite can see.
final class ComposerModeControlTests: XCTestCase {

    private func chatViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The body of a `private var <name>: some View { … }`, up to the next
    /// member declaration.
    private func declaration(_ name: String, in source: String) throws -> String {
        let start = try XCTUnwrap(source.range(of: "private var \(name): some View {"),
                                  "ChatView must still declare \(name)")
        let rest = source[start.upperBound...]
        let end = rest.range(of: "\n    private ") ?? rest.range(of: "\n    @ViewBuilder")
        return String(rest[..<(end?.lowerBound ?? rest.endIndex)])
    }

    func testToolsAndMcpDiscsToggleOnClick() throws {
        let source = try chatViewSource()
        for control in ["agentToggle", "mcpToggle"] {
            let body = try declaration(control, in: source)
            XCTAssertTrue(body.contains("primaryAction:"), """
                \(control) must toggle on a plain click (`Menu { … } label: { … } \
                primaryAction: { … }`). Without it the glyph only opens the menu \
                and the most frequent action in the composer costs two steps.
                """)
            XCTAssertTrue(body.contains(".contextMenu {"), """
                \(control) must offer its configuration list on secondary-click — \
                that is the ONLY discoverable way in once click toggles \
                (press-and-hold works but nobody finds it).
                """)
        }
    }

    /// The menus carry the per-tool switches / workspace / marketplace ONLY.
    /// An on/off row there would be a second way to do what one click does,
    /// and two controls for one boolean is how they end up disagreeing.
    func testMenusDoNotDuplicateTheOnOffSwitch() throws {
        let source = try chatViewSource()
        for banned in ["Turn Tools Off", "Turn Tools On", "Turn MCP Off", "Turn MCP On"] {
            XCTAssertFalse(source.contains("\"\(banned)\""),
                           "\(banned) is the click action now — it must not also be a menu row")
        }
    }

    /// A right-click menu that nothing announces is a feature nobody uses. The
    /// wording moved to `ComposerTip` (pinned by `ComposerTipTests`); what this
    /// audit still owns is that the two controls actually SHOW it.
    func testTooltipsNameBothGestures() throws {
        let source = try chatViewSource()
        for control in ["agentToggle", "mcpToggle"] {
            let body = try declaration(control, in: source)
            XCTAssertTrue(body.contains(".composerTip("),
                          "\(control) must carry the hover card that names the secondary-click menu")
        }
    }

    /// Every composer control is a bare glyph, so each one owes the user a hover
    /// card — and exactly ONE explanation. A leftover `.help` renders a second,
    /// differently-worded tooltip on top of the card, and the two drift.
    func testEveryComposerControlHasOneHoverCardAndNoNativeTooltip() throws {
        let source = try chatViewSource()
        for control in ["attachmentMenu", "thinkToggle", "agentToggle", "mcpToggle"] {
            let body = try declaration(control, in: source)
            XCTAssertTrue(body.contains(".composerTip("), "\(control) has no hover card")
            XCTAssertFalse(body.contains(".help("),
                           "\(control) still has a native tooltip competing with its hover card")
        }
    }

    /// The hover card is drawn by the COMPOSER CONTAINER, not by the control: the
    /// container clips to its rounded rect, so a card overlaid on the disc itself
    /// is cut off at the container's edge (and lands on top of the text field).
    /// The anchor preference is what carries it out past the clip.
    func testHoverCardIsRenderedOutsideTheClippedComposerContainer() throws {
        let source = try chatViewSource()
        let clip = try XCTUnwrap(source.range(of: ".clipShape(RoundedRectangle(cornerRadius: 18))"),
                                 "the composer container still clips — the card must escape it")
        let overlay = try XCTUnwrap(source.range(of: ".composerTipOverlay()"),
                                    "the composer container must render the hover card")
        XCTAssertLessThan(clip.lowerBound, overlay.lowerBound,
                          "the card overlay must be applied AFTER the clip, or it gets cut off")

        // And the card must never take a click: it sits directly over the row's
        // most-flipped controls, and a popover/overlay that eats the mouse-down
        // is worse than no card at all.
        let tipSource = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ComposerTip.swift"), encoding: .utf8)
        XCTAssertTrue(tipSource.contains(".allowsHitTesting(false)"),
                      "the hover card must not be a click target")
        XCTAssertFalse(tipSource.contains(".popover("),
                       "an NSPopover swallows the click that would toggle the control it explains")
    }

    /// "Enable/Disable All Tools" were bulk rows above the per-tool switches they
    /// duplicated — the same set is one click per row away, and the pair read as
    /// a second on/off for the loop the wrench already toggles.
    func testToolMenuHasNoBulkAllRows() throws {
        let source = try chatViewSource()
        for banned in ["Enable All Tools", "Disable All Tools"] {
            XCTAssertFalse(source.contains(banned), "\(banned) was removed as redundant")
        }
        XCTAssertFalse(source.contains("setAllTools"), "the bulk helper goes with its rows")
    }

    /// The sandbox shield is gone from the composer row (2026-07-29) — it was a
    /// status glyph that only ever opened Settings, sitting in the row reserved
    /// for controls that configure the turn being written.
    func testSandboxShieldIsNotInTheComposerRow() throws {
        let source = try chatViewSource()
        XCTAssertFalse(source.contains("sandboxShield"),
                       "the sandbox shield was removed from the chat composer")
    }

    // MARK: - Agent lock

    /// The discs' locked values must come from the SAME resolution the turn runs
    /// under, or they go back to disagreeing with it — which is the bug this
    /// exists to fix (every agent defaults `web: true`, so `AgentResolution`
    /// forced the tool loop on while the wrench still rendered OFF).
    func testTheLockIsBuiltFromTheTurnsOwnResolutionNotFromTheAgentDirectly() throws {
        let source = try chatViewSource()
        let start = try XCTUnwrap(source.range(of: "private var agentModeLock: AgentModeLock? {"),
                                  "ChatView must derive the composer's lock from a resolution")
        let rest = source[start.upperBound...]
        let body = String(rest[..<(rest.range(of: "\n    private ")?.lowerBound ?? rest.endIndex)])
        XCTAssertTrue(body.contains("resolvedAgentSettings("), """
            the lock's Tools/MCP values must come from `resolvedAgentSettings` — \
            reading `capabilities` here would be a second copy of the rule and the \
            icons would drift from what the turn actually runs.
            """)
    }

    /// Belt-and-braces, same shape as the tool-dispatch refusal: the locked disc
    /// no longer offers a primary action, and the setters refuse anyway — the
    /// pre-send intent nudge calls them too.
    func testTheToggleSettersRefuseWhileLocked() throws {
        let source = try chatViewSource()
        for setter in ["private func setToolsEnabled(_ on: Bool) {",
                       "private func setMCPEnabled(_ on: Bool) {"] {
            let start = try XCTUnwrap(source.range(of: setter), "missing \(setter)")
            let rest = source[start.upperBound...]
            let body = String(rest[..<(rest.range(of: "\n    }")?.upperBound ?? rest.endIndex)])
            XCTAssertTrue(body.contains("LockedBy"),
                          "\(setter) must no-op while an agent owns the control")
        }
    }

    /// A nudge offering to turn on a mode the agent forbids is a dead offer —
    /// accepting it changes nothing and the message sends anyway.
    func testThePreSendNudgeIsSuppressedForLockedModes() throws {
        let source = try chatViewSource()
        let start = try XCTUnwrap(source.range(of: "private func detectIntentPrompt(for text: String) -> IntentPrompt? {"))
        let rest = source[start.upperBound...]
        let body = String(rest[..<(rest.range(of: "\n    }")?.upperBound ?? rest.endIndex)])
        XCTAssertTrue(body.contains("LockedBy"),
                      "detectIntentPrompt must not offer a mode the agent decides")
    }

    /// "Edit Agent…" on a locked disc must land on THAT agent, not on whoever
    /// sorts first — the whole reason the row exists is that the user just read
    /// the agent's name on the card.
    func testEditAgentDeepLinksToTheAgentThatLockedTheControl() throws {
        let source = try chatViewSource()
        let start = try XCTUnwrap(source.range(of: "private func lockedModeMenu("),
                                  "the locked discs must still offer a way into the agent")
        let rest = source[start.upperBound...]
        let body = String(rest[..<(rest.range(of: "\n    private ")?.lowerBound ?? rest.endIndex)])
        XCTAssertTrue(body.contains("openAgentSettings("), """
            Edit Agent… must route through `AppState.openAgentSettings` — a bare \
            openWindow(id: "agents") opens the window on the first agent in the list.
            """)
    }

    // MARK: - The agent is chosen when the chat is, and fixed after

    /// The picker moved OUT of the composer row: it configured the whole
    /// conversation, not the message being written, and it sat next to four
    /// controls the agent then overrode.
    func testTheComposerRowNoLongerHoldsTheAgentPicker() throws {
        let source = try chatViewSource()
        XCTAssertFalse(source.contains("agentChip"),
                       "the agent picker lives next to New Chat now, not in the composer")
    }

    /// Switching mid-thread left half a conversation running under someone
    /// else's prompt, tools and model, with nothing but the transcript to show
    /// where the seam was. A session's agent is now decided when the session is
    /// created and never after — structurally, by there being no setter.
    func testASessionsAgentCannotBeChangedAfterItIsCreated() throws {
        let agentsSource = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Services/AppStateAgents.swift"), encoding: .utf8)
        XCTAssertFalse(agentsSource.contains("func setAgent("), """
            `setAgent(_:forSession:)` is gone on purpose — a session's agent is \
            fixed at creation. Starting a chat AS an agent goes through \
            `startChat(withAgent:)`.
            """)
        XCTAssertTrue(agentsSource.contains("func startChat(withAgent"),
                      "the sidebar needs one call that creates the session AND applies the agent")
    }

    /// It has to actually apply the agent — its model, workspace and voice all
    /// live outside the turn, and a session that only carries the ID would run
    /// the persona against whatever model happened to be loaded.
    func testStartingAChatAsAnAgentAppliesThatAgentsSelection() throws {
        let agentsSource = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Services/AppStateAgents.swift"), encoding: .utf8)
        let start = try XCTUnwrap(agentsSource.range(of: "func startChat(withAgent"))
        let rest = agentsSource[start.upperBound...]
        let body = String(rest[..<(rest.range(of: "\n    }")?.upperBound ?? rest.endIndex)])
        XCTAssertTrue(body.contains("newChatSession("), "it must create the session")
        XCTAssertTrue(body.contains("applyAgentSelection("),
                      "model / workspace / voice all live outside the turn")
    }

    /// Starting a chat AS an agent must remain reachable, wherever the control
    /// lives.
    ///
    /// It has moved three times — composer chip → the sidebar's Agents row →
    /// beside New Chat → into the agent's own editor, once Agents became a pane
    /// where everything about an agent lives together. Every move has been one
    /// edit away from deleting the capability instead of relocating it: taking
    /// the menu off the Agents row once left `startChat(withAgent:)` with no
    /// caller anywhere in the UI, and the test passed, because it was pinned to
    /// a control's coordinates. A session's agent is fixed once the session
    /// exists (there is no `setAgent`), so this is the only moment it can be
    /// decided — pin that it is reachable AT ALL, from anywhere.
    func testStartingAChatAsAnAgentStaysReachable() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let views = try FileManager.default
            .contentsOfDirectory(at: root.appendingPathComponent("Sources/MLXServe/Views"),
                                 includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
        let callers = try views.filter {
            try String(contentsOf: $0, encoding: .utf8).contains("startChat(withAgent:")
        }
        XCTAssertFalse(callers.isEmpty, """
            Nothing in the UI calls startChat(withAgent:) any more. A session's \
            agent can only be chosen when the session is created, so with no \
            caller the capability is gone, not moved.
            """)
    }

    /// Agents is a DESTINATION, not a menu: every other row in that column
    /// means "go somewhere", and a row that instead starts a conversation was
    /// the odd one out — with the editor buried at the bottom of its menu.
    func testTheAgentsRowIsADestinationNotAMenu() throws {
        let chat = try chatViewSource()
        let start = try XCTUnwrap(chat.range(of: "private var agentsRow"))
        let rest = chat[start.upperBound...]
        let body = String(rest[..<(rest.range(of: "\n    }")?.upperBound ?? rest.endIndex)])
        XCTAssertTrue(body.contains("appState.showAgents()"),
                      "the Agents row opens the Agents pane")
        XCTAssertFalse(body.contains("Menu {"),
                       "the Agents row must not open a menu any more")
    }

    // MARK: - Content passing under floating chrome

    /// Transcript text ran straight into the floating model-picker cluster.
    /// The fix is the platform's own scroll-edge effect (`scrollEdgeEffectStyle`,
    /// macOS 26+) on the scrolling surfaces — NOT a hand-drawn band, which is how
    /// this went wrong the first time: a custom strip pulled in via
    /// `ignoresSafeArea` looked native and swallowed every click in the toolbar
    /// band's layer.
    func testScrollingSurfacesUseTheNativeScrollEdgeEffect() throws {
        let source = try chatViewSource()
        let sidebar = try XCTUnwrap(source.range(of: "struct ChatSidebar: View {"))
        let sidebarBody = String(source[sidebar.upperBound...]
            .prefix(while: { _ in true }))
            .components(separatedBy: "\n// MARK:").first ?? ""
        XCTAssertTrue(sidebarBody.contains(".scrollEdgeEffectStyle("),
                      "the session list needs the native edge effect under its chrome")

        let occurrences = source.components(separatedBy: ".scrollEdgeEffectStyle(").count - 1
        XCTAssertGreaterThanOrEqual(occurrences, 2,
                                    "both the sidebar and the transcript scroll under floating chrome")
    }

    /// The toolbar carries no material — and its BAR still exists.
    ///
    /// Everything that lived in it moved somewhere it belongs: the model
    /// picker, the mode discs and the server control to the COMPOSER row (they
    /// configure the message, or report what you discover by typing), Settings
    /// to a sidebar destination. What remained was an empty band, so the
    /// material goes.
    ///
    /// The bar does NOT go with it. `.toolbar(.hidden, for: .windowToolbar)`
    /// removes the bar itself, and the traffic lights and the sidebar-collapse
    /// button are its residents — hiding it produced a window that could not be
    /// closed, zoomed, or have its sidebar collapsed (live 2026-08-09). The
    /// older trap it was chosen to avoid — `scrollEdgeEffectStyle` attached to
    /// an invisible bar, transcript text clipping mid-line (2026-07-30) — was
    /// about the floating CLUSTER, which no longer exists; the one surface that
    /// still needs frosting owns a backdrop of its own instead.
    func testTheToolbarKeepsItsBarAndLosesItsMaterial() throws {
        let raw = try chatViewSource()
        let source = raw
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")

        XCTAssertFalse(source.contains(".toolbar(.hidden, for: .windowToolbar)"), """
            Hiding the window toolbar takes the traffic lights and the \
            sidebar-collapse button with it. Hide its MATERIAL instead.
            """)
        let hidden = source.components(separatedBy: ".toolbarBackground(.hidden, for: .windowToolbar)").count - 1
        XCTAssertEqual(hidden, 2, """
            Both split views (the chat's two-column and the three-column Tasks / \
            Agents) must drop the toolbar material — nothing lives in that band.
            """)
        XCTAssertFalse(source.contains(".toolbarBackground(.visible"),
                       "no column should re-assert the material the window no longer needs")
        XCTAssertFalse(source.contains("private var floatingToolbar"),
                       "the floating cluster is gone; its controls live in the composer row")
    }

    /// The hidden material sits on the SPLITS themselves, never on one pane
    /// inside them. Attached to ChatDetailView it covered only conversation
    /// mode — switching to Models / Settings / Create brought the default
    /// toolbar material back and the window chrome flipped per mode, which is
    /// exactly the "no material now, in every mode" contract broken.
    func testBothSplitViewsCarryTheHiddenMaterialThemselves() throws {
        let source = try chatViewSource()
        func body(of marker: String, until end: String) throws -> String {
            let afterStart = try XCTUnwrap(source.range(of: marker),
                                           "marker \(marker) not found").upperBound
            let tail = source[afterStart...]
            let stop = try XCTUnwrap(tail.range(of: end),
                                     "end marker \(end) not found").lowerBound
            return String(tail[..<stop])
        }
        let modifier = ".toolbarBackground(.hidden, for: .windowToolbar)"
        let three = try body(of: "private var threeColumnSplitView",
                             until: "private var standardSplitView")
        XCTAssertTrue(three.contains(modifier),
                      "the three-column split (Tasks/Agents) must drop the material itself")
        let standard = try body(of: "private var standardSplitView",
                                until: "private func createPane")
        XCTAssertTrue(standard.contains(modifier), """
            the standard split must drop the material itself — on ChatDetailView \
            it covers only conversation mode, and Models/Settings/Create regain \
            the default toolbar band.
            """)
    }

    /// The sidebar's pinned destinations rely on the PLATFORM to frost what
    /// scrolls beneath them — `scrollEdgeEffectStyle`, which needs the
    /// toolbar's bar to attach to.
    ///
    /// They carried a hand-drawn `.regularMaterial` backdrop for exactly as
    /// long as the bar was hidden outright. With the bar back the backdrop
    /// goes: a band drawn by hand over a sidebar is the same class as the strip
    /// that once looked native and swallowed every click in it.
    func testThePinnedDestinationsRelyOnTheScrollEdgeEffect() throws {
        let source = try chatViewSource()
        guard let start = source.range(of: "safeAreaInset(edge: .top)"),
              let end = source.range(of: "private func sectionHeader",
                                     range: start.upperBound..<source.endIndex) else {
            return XCTFail("the sidebar's destination inset moved — update this audit")
        }
        let inset = String(source[start.upperBound..<end.lowerBound])
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")
        XCTAssertFalse(inset.contains(".background(.regularMaterial)"), """
            The destinations block draws its own band again. The toolbar's bar \
            is back, so the platform's scroll-edge effect has something to \
            attach to — a hand-drawn strip over a sidebar is the class that \
            once looked native and ate every click in it.
            """)
        XCTAssertFalse(inset.contains(".background(.bar)"),
                       "`.bar` also draws a separator — the rule removed from the Tasks header")
        // The effect itself must still be asked for.
        XCTAssertTrue(source.contains(".scrollEdgeEffectStyle("),
                      "the sidebar list still needs the native edge effect")
    }

    // MARK: - Hover card dismissal

    /// The card is a non-hit-testing overlay, so the only thing that takes it
    /// down is a hover-exit — which opening a menu does NOT deliver. It then sits
    /// under the open menu. `NSMenu.didBeginTrackingNotification` fires for both
    /// buttons (SwiftUI's `Menu` and `.contextMenu` are both NSMenus) and for
    /// press-and-hold, so one observer covers every way in.
    func testTheHoverCardIsTakenDownWhenAMenuOpensOverIt() throws {
        let tipSource = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ComposerTip.swift"), encoding: .utf8)
        XCTAssertTrue(tipSource.contains("NSMenu.didBeginTrackingNotification"), """
            opening a menu over the composer must dismiss the hover card — left-click \
            on the agent/attach menus, right-click on the Tools/MCP discs.
            """)
    }
}
