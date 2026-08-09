import XCTest
@testable import MLXCore

/// The Agents editor draws its own surfaces (`AgentEditorChrome`) instead of
/// handing the job to a grouped `Form`.
///
/// That was not a style preference: a grouped Form owns the card radius, the row
/// insets, the label typography and the section spacing, so the mockup's
/// geometry was unreachable from inside one — and macOS right-aligns a
/// TextField's text in a Form row, which put the agent's own NAME hard against
/// the trailing edge of its field. Once the numbers are ours, they can drift,
/// so the relationships between them are pinned here rather than left as
/// literals scattered through a view body.
final class AgentEditorLayoutTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func agentsWindow() throws -> String {
        try source("Sources/MLXServe/Views/AgentsWindow.swift")
    }

    /// The editor's own struct, so an audit about the EDITOR can't be satisfied
    /// (or tripped) by the list row that shares the file.
    ///
    /// Comments are stripped: this file's own prose explains what the editor is
    /// NOT (a grouped Form), and a scan that reads the warning as the thing it
    /// warns about fails on the sentence describing the fix.
    private func editorBody() throws -> String {
        let s = try agentsWindow()
        guard let start = s.range(of: "private struct AgentEditor: View") else {
            throw XCTSkip("the editor moved — update this audit")
        }
        let after = s[start.upperBound...]
        let end = after.range(of: "\n// MARK: - Voice picker")
        let body = String(after[..<(end?.lowerBound ?? after.endIndex)])
        return body.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")
    }

    // MARK: Geometry

    /// A well sunk INTO a card cannot be rounder than the card holding it —
    /// the inner corner would bulge out of the outer one at every corner.
    func testANestedWellIsNeverRounderThanTheCardHoldingIt() {
        XCTAssertLessThanOrEqual(AgentEditorMetrics.wellRadius, AgentEditorMetrics.cardRadius)
    }

    /// A card's padding has to clear the well it holds, or the well's own inset
    /// text lands closer to the card edge than the card's other content does.
    func testTheCardsPaddingClearsTheWellItHolds() {
        XCTAssertGreaterThanOrEqual(AgentEditorMetrics.cardPadding, AgentEditorMetrics.wellPadding)
    }

    /// Rhythm: two sections must read as further apart than a label is from the
    /// field it names. Equal spacing is what made the old Form read as one
    /// undifferentiated column of settings.
    func testSectionsSeparateMoreThanALabelFromItsField() {
        XCTAssertGreaterThan(AgentEditorMetrics.sectionSpacing, AgentEditorMetrics.labelSpacing)
    }

    /// The column is padded AND capped, like the chat column: a form field run
    /// the full width of a 1400pt window is a text field you scan rather than
    /// read.
    func testTheEditorColumnIsCappedAndPadded() {
        XCTAssertGreaterThan(AgentEditorMetrics.contentMaxWidth, 400)
        XCTAssertGreaterThan(AgentEditorMetrics.contentPadding, 0)
    }

    // MARK: The chrome is shared

    /// Every surface in the editor comes from `AgentEditorChrome`. A card
    /// hand-rolled at a call site is how six cards end up with five radii —
    /// the exact drift the shared chrome exists to prevent.
    func testEveryEditorSurfaceComesFromTheSharedChrome() throws {
        let body = try editorBody()
        XCTAssertFalse(body.contains("RoundedRectangle(cornerRadius:"),
                       "the editor drew its own shape — route it through AgentEditorChrome")
        XCTAssertFalse(body.contains("Capsule()"),
                       "the editor drew its own pill — use agentPillButton()")
    }

    /// The editor owns its layout instead of handing it to macOS. A grouped
    /// Form decides the radii, the insets and the label typography, and
    /// right-aligns a TextField's text — so its return is not a style change,
    /// it is the mockup becoming unreachable again.
    func testTheEditorOwnsItsOwnSurfacesRatherThanAGroupedForm() throws {
        let body = try editorBody()
        XCTAssertFalse(body.contains(".formStyle(.grouped)"),
                       "a grouped Form owns the geometry this editor is specified in")
        XCTAssertTrue(body.contains("AgentCard"),
                      "the editor draws its groups on the shared card")
    }

    /// One builder for every section title. Two ways to draw a heading is two
    /// headings that stop matching.
    func testSectionTitlesComeFromOneBuilder() throws {
        let body = try editorBody()
        for title in ["Prompt", "Identity", "Capabilities", "Model", "Workspace", "Sampling"] {
            XCTAssertTrue(body.contains("AgentSection(\"\(title)\")"),
                          "“\(title)” must be titled by the shared section header")
        }
    }

    /// The prompt's helper text and counter sit BELOW the editor, with the
    /// write action, the way the mockup has them: above, they separated the
    /// section's title from the thing it titles.
    func testThePromptActionsFollowTheEditorTheyDescribe() throws {
        let body = try editorBody()
        guard let section = body.range(of: "private var promptSection") else {
            return XCTFail("the prompt section moved — update this audit")
        }
        let rest = body[section.upperBound...]
        guard let editor = rest.range(of: "TextEditor("),
              let actions = rest.range(of: "Write it for me") else {
            return XCTFail("expected the prompt editor and its write action")
        }
        XCTAssertLessThan(editor.lowerBound, actions.lowerBound,
                          "the editor comes first; the actions describe what's above them")
    }
}
