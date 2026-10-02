import XCTest
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

    // MARK: - Wiring; headless tests cannot drive an NSTextView, so consumers are pinned.

    private var chatView: String {
        SourceScan.source("Views/ChatView.swift", from: #filePath)
    }

    func testTheComposerDecidesReturnWithTheComposingState() {
        guard let body = SourceScan.declarationBody(
            from: "func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {",
            in: chatView) else {
            return XCTFail("the composer's doCommandBy is gone — renamed?")
        }
        XCTAssertTrue(body.contains("hasMarkedText:"),
                      "onReturn must be called with the field's marked-text state; its default keeps the old ladder")
    }

    func testTheRedrawGuardIsTheSharedDecision() {
        guard let body = SourceScan.declarationBody(
            from: "func updateNSView(_ scroll: NSScrollView, context: Context) {",
            in: chatView) else {
            return XCTFail("GrowingTextEditor.updateNSView is gone — renamed?")
        }
        XCTAssertTrue(body.contains("shouldAdoptExternalText("),
                      "the string assignment must run through the marked-text guard, not a bare inequality")
    }

    func testThePlaceholderAsksAboutTheComposingState() {
        guard let body = SourceScan.declarationBody(
            from: "private var composerField: some View {",
            in: chatView) else {
            return XCTFail("composerField is gone — renamed?")
        }
        XCTAssertTrue(body.contains("showsPlaceholder("),
                      "the overlay must not gate on `inputText.isEmpty` alone")
    }

    func testTheFieldReportsCompositionToSwiftUI() {
        // ComposerTextView is the file's last declaration, so its anchor's suffix is its body.
        guard let start = chatView.range(of: "final class ComposerTextView: NSTextView {") else {
            return XCTFail("ComposerTextView is gone — renamed?")
        }
        let body = String(chatView[start.lowerBound...])
        XCTAssertTrue(body.contains("override func setMarkedText"),
                      "composition beginning is invisible to SwiftUI unless the field reports it")
        XCTAssertTrue(body.contains("override func unmarkText"),
                      "composition ending is invisible to SwiftUI unless the field reports it")
    }

    func testTheEditorWiresTheReportThrough() {
        guard let body = SourceScan.declarationBody(
            from: "func makeNSView(context: Context) -> NSScrollView {",
            in: chatView) else {
            return XCTFail("GrowingTextEditor.makeNSView is gone — renamed?")
        }
        XCTAssertTrue(body.contains("onMarkedTextChanged"),
                      "the report must reach the coordinator, or the SwiftUI side never learns")
    }
}
