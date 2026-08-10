import XCTest
@testable import MLXCore

/// The welcome screen's right panel used to reprint the selected card's title
/// and description above its actual content — the same two lines the user had
/// just clicked, costing ~90pt of the panel's height and making the two columns
/// read as two separate things saying the same thing. The panel is now content
/// only, tied to its card by a drawn connector.
final class WelcomeConnectorTests: XCTestCase {

    private func welcomeSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/WelcomeView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    /// The copy lives on the card. Rendering `feature.title` / `.description`
    /// in the panel too is the repetition this replaced — and it comes back the
    /// moment someone "adds a heading so the panel isn't bare".
    func testRightPanelDoesNotReprintTheSelectedCardsCopy() throws {
        let source = try welcomeSource()
        guard let panel = source.range(of: "private func rightPanel(for feature: WelcomeFeature)"),
              let next = source.range(of: "private func panelContent(for panel:") else {
            return XCTFail("rightPanel(for:)/panelContent(for:) not found — update this audit")
        }
        let body = String(source[panel.lowerBound..<next.lowerBound])
        XCTAssertFalse(body.contains("feature.title"), """
            The right panel must not reprint the selected card's title — the \
            card the user just clicked is still on screen, connected to it.
            """)
        XCTAssertFalse(body.contains("feature.description"), """
            Same for the description: two lines of duplicated copy at the top \
            of the panel is what the connector replaced.
            """)
    }

    /// `WelcomeExit` only closes the dead end if the view actually goes through
    /// it. Each of the three callbacks must be invoked from exactly ONE place —
    /// `leave(_:)` — or a future button wires up its own two-of-three
    /// combination again, which is precisely how "Browse all models" came to
    /// dismiss the window without opening Chat.
    func testEveryExitCallbackIsInvokedOnlyFromLeave() throws {
        let source = try welcomeSource()
        guard let start = source.range(of: "struct WelcomeView"),
              // WelcomeModelRow further down legitimately calls its OWN
              // onOpenChat — the closure the parent binds to leave(.useModel).
              let end = source.range(of: "struct WelcomeCardAnchorKey") else {
            return XCTFail("WelcomeView / WelcomeCardAnchorKey not found — update this audit")
        }
        let view = String(source[start.lowerBound..<end.lowerBound])
        for call in ["onOpenChat()", "onOpenModelBrowser()", "onDismiss()"] {
            let count = view.components(separatedBy: call).count - 1
            XCTAssertEqual(count, 1, """
                `\(call)` is invoked \(count) times in WelcomeView — it belongs \
                only inside leave(_:), which is what guarantees every route out \
                of this window opens a window the user can act in.
                """)
        }
        XCTAssertTrue(view.contains("private func leave(_ exit: WelcomeExit)"))
    }

    /// The connector has to track the SELECTED card, not a fixed row. Anchored
    /// to the first card it would look correct for the default selection and
    /// point at nothing for the other two — the kind of bug a static screenshot
    /// of a fresh launch never shows.
    func testConnectorIsAnchoredToTheSelectedCard() throws {
        let source = try welcomeSource()
        XCTAssertTrue(source.contains("WelcomeCardAnchorKey"),
                      "the connector reads the card's live frame through a preference key")
        guard let anchor = source.range(of: "key: WelcomeCardAnchorKey.self") else {
            return XCTFail("no card→connector anchor preference is published")
        }
        // The published value must be gated on selection — read the argument
        // list that follows the key.
        let tail = source[anchor.upperBound...].prefix(300)
        XCTAssertTrue(tail.contains("isSelected"), """
            The anchor must be published only by the SELECTED card, or the \
            connector points at whichever card happens to publish last.
            """)
    }
}
