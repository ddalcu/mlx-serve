import XCTest
import AppKit
import SwiftUI
@testable import MLXCore

// A live input-method composition reaches no SwiftUI state until it commits.
final class ComposerMarkedTextTests: XCTestCase {

    // MARK: - ComposerKey.onReturn with the composition state

    func testReturnMidCompositionConfirmsInsteadOfSending() {
        // Enter belongs to the input method while marked text is held, over any other arm.
        for shift in [false, true] {
            for isIdle in [false, true] {
                for canSteer in [false, true] {
                    XCTAssertEqual(ComposerKey.onReturn(shift: shift, isIdle: isIdle,
                                                         canSteer: canSteer,
                                                         hasMarkedText: true),
                                   .confirmMarkedText,
                                   "shift: \(shift), isIdle: \(isIdle), canSteer: \(canSteer)")
                }
            }
        }
    }

    func testReturnWithNothingComposedKeepsTheOldLadder() {
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: true, hasMarkedText: false), .send)
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: false, hasMarkedText: false), .ignore)
        XCTAssertEqual(ComposerKey.onReturn(shift: false, isIdle: false, canSteer: true, hasMarkedText: false), .steer)
        XCTAssertEqual(ComposerKey.onReturn(shift: true, isIdle: true, hasMarkedText: false), .newline)
    }

    // MARK: - ComposerKey.shouldAdoptExternalText

    func testADifferingExternalTextSyncsWhenNothingIsComposed() {
        XCTAssertTrue(ComposerKey.shouldAdoptExternalText(field: "", bound: "hi", hasMarkedText: false),
                      "a cleared or recalled draft must still reach the field")
    }

    func testIdenticalTextIsNotRewritten() {
        XCTAssertFalse(ComposerKey.shouldAdoptExternalText(field: "hi", bound: "hi", hasMarkedText: false),
                       "rewriting equal text would move the caret on every redraw")
    }

    func testMarkedTextSurvivesAnExternalRewrite() {
        // A redraw mid-composition must not delete the underlined string on screen.
        XCTAssertFalse(ComposerKey.shouldAdoptExternalText(field: "こんにちは", bound: "", hasMarkedText: true),
                       "the binding is blind to marked text — it is the STALE side of this comparison")
        XCTAssertFalse(ComposerKey.shouldAdoptExternalText(field: "hi", bound: "hi", hasMarkedText: true))
    }

    // MARK: - ComposerKey.showsPlaceholder

    func testTheExampleLineOnlySitsOnAQuietEmptyField() {
        XCTAssertTrue(ComposerKey.showsPlaceholder("", hasMarkedText: false))
        XCTAssertFalse(ComposerKey.showsPlaceholder("hi", hasMarkedText: false))
        XCTAssertFalse(ComposerKey.showsPlaceholder("hi", hasMarkedText: true))
        // The binding is blind to marked text, so emptiness alone draws THROUGH it.
        XCTAssertFalse(ComposerKey.showsPlaceholder("", hasMarkedText: true),
                       "an underlined composition is text the user can see")
    }

    @MainActor
    func testCompositionReportsTheLiveStateThroughCommitAndUnmark() {
        let tv = ComposerTextView()
        var states: [Bool] = []
        tv.onMarkedTextChanged = { [weak tv] marked in
            XCTAssertEqual(marked, tv?.hasMarkedText())
            states.append(marked)
        }
        tv.setMarkedText("にほん", selectedRange: NSRange(location: 3, length: 0),
                         replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertEqual(states.last, true)
        tv.insertText("日本", replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertEqual(states.last, false)
        XCTAssertEqual(tv.string, "日本")
        states.removeAll()
        tv.setMarkedText("ご", selectedRange: NSRange(location: 1, length: 0),
                         replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertEqual(states.last, true)
        tv.unmarkText()
        XCTAssertEqual(states.last, false)
    }

    @MainActor
    func testEditorPreservesMarkedTextAndReportsTheCommittedBinding() throws {
        _ = NSApplication.shared
        var draft = ""
        var marked = false
        let binding = Binding<String>(get: { draft }, set: { draft = $0 })
        var editor = GrowingTextEditor(text: binding, isFocused: .constant(false),
                                       measuredHeight: .constant(40), isIdle: true,
                                       onMarkedTextChanged: { marked = $0 }, onSend: {})
        let host = NSHostingView(rootView: editor)
        host.frame = NSRect(x: 0, y: 0, width: 320, height: 80)
        host.layout()
        func textView(in view: NSView) -> ComposerTextView? {
            if let tv = view as? ComposerTextView { return tv }
            for child in view.subviews {
                if let tv = textView(in: child) { return tv }
            }
            return nil
        }
        let tv = try XCTUnwrap(textView(in: host))
        tv.setMarkedText("にほん", selectedRange: NSRange(location: 3, length: 0),
                         replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertTrue(marked)
        XCTAssertEqual(draft, "")
        editor.isIdle = false
        host.rootView = editor
        host.layout()
        XCTAssertTrue(tv.hasMarkedText())
        XCTAssertEqual(tv.string, "にほん")
        tv.insertText("日本", replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertFalse(marked)
        XCTAssertEqual(draft, "日本")
        draft = "recalled draft"
        editor.isIdle = true
        host.rootView = editor
        host.layout()
        XCTAssertEqual(tv.string, draft)
        tv.setMarkedText("ご", selectedRange: NSRange(location: 1, length: 0),
                         replacementRange: NSRange(location: NSNotFound, length: 0))
        tv.unmarkText()
        XCTAssertFalse(marked)
        XCTAssertEqual(draft, tv.string)
    }

    @MainActor
    func testReturnInTheEditorPassesCompositionAndThenSends() {
        var sent = 0
        var accepted = 0
        let editor = GrowingTextEditor(text: .constant(""), isFocused: .constant(false),
                                       measuredHeight: .constant(40), isIdle: true,
                                       onSend: { sent += 1 },
                                       onKeyCommand: { _ in accepted += 1; return false })
        let coordinator = editor.makeCoordinator()
        let tv = ComposerTextView()
        tv.setMarkedText("にほん", selectedRange: NSRange(location: 3, length: 0),
                         replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertFalse(coordinator.textView(tv, doCommandBy: #selector(NSResponder.insertNewline(_:))))
        XCTAssertEqual(sent, 0)
        XCTAssertEqual(accepted, 0)
        tv.insertText("日本", replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertTrue(coordinator.textView(tv, doCommandBy: #selector(NSResponder.insertNewline(_:))))
        XCTAssertEqual(sent, 1)
    }
}
