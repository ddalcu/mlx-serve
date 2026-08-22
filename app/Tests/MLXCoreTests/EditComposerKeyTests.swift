import XCTest
@testable import MLXCore

/// Editing a past message is a SEND, so its field is the composer's field.
///
/// The edit bubble shipped as a plain SwiftUI `TextEditor` with a Save button:
/// Return inserted a newline and the only way to resubmit was the mouse. Every
/// other place you type a message in this app sends on Return and breaks the
/// line on Shift+Return, so the edit field was the one input with its own
/// rules — and the rules it had were the ones nobody wants (a message you
/// finished typing sits there while Return grows it).
///
/// Reusing `GrowingTextEditor` is what makes the two agree by construction:
/// the Return decision is `ComposerKey.onReturn` for both, so the edit field
/// cannot drift from the composer without the composer drifting too.
final class EditComposerKeyTests: XCTestCase {

    private var chatViewSource: String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/ChatView.swift")
        return (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    }

    // MARK: - The submit gate

    func testAnEmptyDraftCannotBeSubmitted() {
        XCTAssertFalse(ComposerKey.editCanSubmit(""))
    }

    func testAWhitespaceOnlyDraftCannotBeSubmitted() {
        // The whole point of the gate: Return on a draft the user has blanked
        // must not resubmit the turn with nothing in it, which would drop the
        // reply and everything after it in exchange for an empty message.
        XCTAssertFalse(ComposerKey.editCanSubmit("   "))
        XCTAssertFalse(ComposerKey.editCanSubmit("\n\n"))
        XCTAssertFalse(ComposerKey.editCanSubmit(" \t\n "))
    }

    func testARealDraftCanBeSubmitted() {
        XCTAssertTrue(ComposerKey.editCanSubmit("hi"))
        XCTAssertTrue(ComposerKey.editCanSubmit("  hi  "))
        XCTAssertTrue(ComposerKey.editCanSubmit("line one\nline two"))
    }

    // MARK: - Return, composed through the shared decision

    func testBareReturnSubmitsARealDraft() {
        XCTAssertEqual(ComposerKey.onReturn(shift: false,
                                            isIdle: ComposerKey.editCanSubmit("hello")),
                       .send)
    }

    func testBareReturnIsSwallowedOnABlankDraft() {
        // Swallowed, NOT a newline: a bare Return is the user asking to send.
        // Answering it by growing the field is how the old edit box behaved.
        XCTAssertEqual(ComposerKey.onReturn(shift: false,
                                            isIdle: ComposerKey.editCanSubmit("  ")),
                       .ignore)
    }

    func testShiftReturnBreaksTheLineEvenOnABlankDraft() {
        for draft in ["", "   ", "hello"] {
            XCTAssertEqual(ComposerKey.onReturn(shift: true,
                                                isIdle: ComposerKey.editCanSubmit(draft)),
                           .newline,
                           "Shift+Return is always a newline (draft: \(draft.debugDescription))")
        }
    }

    // MARK: - The wiring these decisions are worth nothing without

    func testTheEditFieldIsTheComposersField() {
        // A pure Return decision proves nothing if the edit bubble still hosts
        // a stock `TextEditor`, which consumes Return itself and never asks.
        let source = chatViewSource
        XCTAssertFalse(source.isEmpty, "could not read ChatView.swift")

        guard let range = source.range(of: "private var editingContent: some View {") else {
            return XCTFail("editingContent is gone — has the edit bubble been renamed?")
        }
        let body = String(source[range.lowerBound...].prefix(2000))
        XCTAssertTrue(body.contains("GrowingTextEditor("),
                      "the edit field must be the composer's GrowingTextEditor, so Return runs ComposerKey.onReturn")
        // `GrowingTextEditor(text:` contains `TextEditor(text:`, so a plain
        // `contains` here passes on the fix and fails on the fix's own name.
        // Count both and subtract: what's left is a STOCK TextEditor.
        let allEditors = body.components(separatedBy: "TextEditor(text:").count - 1
        let growing = body.components(separatedBy: "GrowingTextEditor(text:").count - 1
        XCTAssertEqual(allEditors - growing, 0,
                       "a stock TextEditor eats Return — that is the bug this fixes")
    }

    func testTheSaveButtonAndTheReturnKeyReadTheSameGate() {
        // The failure this forbids is a silent one: Save disabled on a draft
        // Return still submits (or the reverse). One spelling of the gate, so
        // the two controls cannot disagree.
        let source = chatViewSource
        XCTAssertFalse(source.isEmpty, "could not read ChatView.swift")

        let adHoc = source.components(separatedBy: "editDraft.trimmingCharacters").count - 1
        XCTAssertEqual(adHoc, 0,
                       "the edit submit gate is ComposerKey.editCanSubmit — no call site re-derives it by trimming editDraft itself")
        XCTAssertGreaterThanOrEqual(
            source.components(separatedBy: "ComposerKey.editCanSubmit").count - 1, 2,
            "both the Save button and the Return key must read the gate")
    }
}
