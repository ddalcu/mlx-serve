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

    /// A right-click menu that nothing announces is a feature nobody uses.
    func testTooltipsNameBothGestures() throws {
        let source = try chatViewSource()
        for control in ["agentToggle", "mcpToggle"] {
            let body = try declaration(control, in: source)
            XCTAssertTrue(body.lowercased().contains("right-click"),
                          "\(control)'s tooltip must name the secondary-click menu")
        }
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
