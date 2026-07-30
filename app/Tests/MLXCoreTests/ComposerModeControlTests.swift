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
        for control in ["agentChip", "attachmentMenu", "thinkToggle", "agentToggle", "mcpToggle"] {
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
}
