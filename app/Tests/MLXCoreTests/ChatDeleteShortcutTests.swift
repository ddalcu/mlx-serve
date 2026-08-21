import XCTest
import AppKit
@testable import MLXCore

/// ⌘⌫ has two owners and the menu bar wins by default.
///
/// A menu item's key equivalent is offered the keystroke by
/// `performKeyEquivalent` BEFORE it ever reaches the first responder, so
/// putting "Delete Chat" on ⌘⌫ took that chord away from every text field in
/// the app — including the composer, where ⌘⌫ has meant "delete to the start of
/// the line" since long before this app existed. Typing it mid-message raised a
/// "Delete this chat?" dialog, and `.disabled(chatDeletionTarget == nil)` could
/// never help: a chat is open in exactly the state where you are typing into
/// one.
///
/// So the command ROUTES rather than claims. With a text editor holding the
/// keyboard the keystroke is handed back to it; only when nothing is being
/// typed into does it delete chats.
final class ChatDeleteShortcutTests: XCTestCase {

    // MARK: The rule

    func testTypingIntoAFieldKeepsTheTextEditingMeaning() {
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: true, selectedChats: 1),
                       .deleteToLineStart)
    }

    func testWithNothingFocusedItDeletesChats() {
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: false, selectedChats: 1),
                       .deleteChats)
    }

    /// Several rows picked is unambiguous — you were working in the list. It
    /// outranks focus because focus in this window is not a reliable signal on
    /// its own: nothing here ever takes the keyboard away from the composer
    /// except the resign below, and a stuck one would make ⌘⌫ a line delete
    /// forever with a dozen chats selected behind it.
    func testAMultiSelectionOutranksAFocusedField() {
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: true, selectedChats: 4),
                       .deleteChats)
    }

    func testNothingSelectedAndTypingIsStillATextEdit() {
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: true, selectedChats: 0),
                       .deleteToLineStart)
    }

    // MARK: What counts as "typing into"

    /// The composer, the in-place message editor and every `GrowingTextEditor`
    /// are `NSTextView`s.
    func testATextViewIsATextEditor() {
        XCTAssertTrue(KeyboardFocus.isTextEditor(NSTextView()))
    }

    /// An `NSTextField` never becomes first responder itself — the window's
    /// FIELD EDITOR does, and that is an `NSTextView`. Naming the field type
    /// alone would therefore have matched nothing that is ever focused, which
    /// is exactly how a guard like this passes its own test and fails live.
    func testAFocusedTextFieldIsItsFieldEditorWhichIsATextView() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 60),
                              styleMask: [.titled], backing: .buffered, defer: true)
        let field = NSTextField(string: "hello")
        window.contentView?.addSubview(field)
        window.makeFirstResponder(field)
        XCTAssertTrue(KeyboardFocus.isTextEditor(window.firstResponder),
                      "a focused NSTextField's first responder is its field editor")
    }

    func testAPlainViewIsNotATextEditor() {
        XCTAssertFalse(KeyboardFocus.isTextEditor(NSView()))
    }

    /// No key window, no focus, nothing being typed into.
    func testNoResponderIsNotATextEditor() {
        XCTAssertFalse(KeyboardFocus.isTextEditor(nil))
    }

    // MARK: The wiring

    /// The menu command must consult the router. Without this the enum is a
    /// rule nobody applies — which is the state the bug shipped in.
    func testTheMenuCommandRoutesTheKeystrokeInsteadOfClaimingIt() throws {
        let source = SourceScan.source("AppState.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func requestChatDeletionFromMenu", in: source),
            "requestChatDeletionFromMenu moved — repoint this scan")
        XCTAssertTrue(body.contains("ChatDeleteShortcut.route"), """
            requestChatDeletionFromMenu does not consult ChatDeleteShortcut, so \
            ⌘⌫ deletes a chat while the user is typing a message.
            """)
        XCTAssertTrue(body.contains("deleteToBeginningOfLine"), """
            the text-editing branch must PERFORM the deletion it took the \
            keystroke for — a menu key equivalent has already swallowed the \
            event, so simply returning early leaves ⌘⌫ doing nothing in the \
            composer.
            """)
    }
}

/// Who holds the keyboard — and the reason "is a text field focused?" was not
/// a usable question in this window until it was fixed.
///
/// Live report 2026-08-12: after one click in the composer, ⌘⌫ NEVER deleted a
/// chat again. The routing rule was right and the state it read was stuck true,
/// for two compounding reasons — and both had to go, since either one alone
/// keeps the composer holding the keyboard forever.
final class KeyboardFocusTests: XCTestCase {

    /// A conversation row is a `.buttonStyle(.plain)` Button, which takes no
    /// first responder under macOS's default keyboard navigation — the same
    /// fact that made `.onDeleteCommand` on that column never fire. So clicking
    /// a chat moved the selection while the composer kept the keyboard, and
    /// nothing in the window ever took it back.
    func testClickingAConversationRowMovesTheKeyboardOutOfTheComposer() throws {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "private func selectRow", in: source),
            "selectRow moved — repoint this scan")
        XCTAssertTrue(body.contains("KeyboardFocus.resignTextEditor"), """
            clicking a conversation row leaves the composer holding the \
            keyboard — the rows take no first responder themselves, so nothing \
            else will move it.
            """)
    }

    /// The second half. `updateNSView` re-took first responder on EVERY update
    /// while `isFocused` was true, and `onResignFocus` publishes the cleared
    /// flag asynchronously — so the re-grab landed first and put the keyboard
    /// straight back. A resign could not stick, and neither could a click into
    /// anything else in the window.
    func testTheComposerTakesFocusOnTheEDGEAndNotOnEveryUpdate() throws {
        let source = SourceScan.source("Views/ChatView.swift", from: #filePath)
        // Anchored on the COMPOSER's, not the first `updateNSView` in the file
        // — the transcript's own text view has one too.
        let body = try XCTUnwrap(
            SourceScan.declarationBody(from: "func updateNSView(_ scroll: NSScrollView", in: source),
            "the composer's updateNSView moved — repoint this scan")
        XCTAssertTrue(body.contains("appliedFocus"), """
            the focus mirror is level-triggered again: a sticky `isFocused` \
            re-takes the keyboard on every SwiftUI update, so the field can \
            never be left.
            """)
    }

    /// It only moves the keyboard when a text editor has it. Calling
    /// `makeFirstResponder(nil)` unconditionally would yank focus out of
    /// whatever else legitimately holds it.
    func testResignLeavesANonTextResponderAlone() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 60),
                              styleMask: [.titled], backing: .buffered, defer: true)
        let button = NSButton(title: "x", target: nil, action: nil)
        window.contentView?.addSubview(button)
        let before = window.firstResponder
        KeyboardFocus.resignTextEditor(in: window)
        XCTAssertTrue(window.firstResponder === before)
    }

    func testResignMovesTheKeyboardOffAFocusedField() {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 60),
                              styleMask: [.titled], backing: .buffered, defer: true)
        let field = NSTextField(string: "hello")
        window.contentView?.addSubview(field)
        window.makeFirstResponder(field)
        XCTAssertTrue(KeyboardFocus.isTextEditor(window.firstResponder))

        KeyboardFocus.resignTextEditor(in: window)
        XCTAssertFalse(KeyboardFocus.isTextEditor(window.firstResponder),
                       "the field editor still holds the keyboard after a resign")
    }

    func testResignWithNoWindowIsHarmless() {
        KeyboardFocus.resignTextEditor(in: nil)
    }
}
