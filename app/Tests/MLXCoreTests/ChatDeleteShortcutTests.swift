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
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: true), .deleteToLineStart)
    }

    func testWithNothingFocusedItDeletesChats() {
        XCTAssertEqual(ChatDeleteShortcut.route(editingText: false), .deleteChats)
    }

    // MARK: What counts as "typing into"

    /// The composer, the in-place message editor and every `GrowingTextEditor`
    /// are `NSTextView`s.
    func testATextViewIsATextEditor() {
        XCTAssertTrue(ChatDeleteShortcut.isTextEditor(NSTextView()))
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
        XCTAssertTrue(ChatDeleteShortcut.isTextEditor(window.firstResponder),
                      "a focused NSTextField's first responder is its field editor")
    }

    func testAPlainViewIsNotATextEditor() {
        XCTAssertFalse(ChatDeleteShortcut.isTextEditor(NSView()))
    }

    /// No key window, no focus, nothing being typed into.
    func testNoResponderIsNotATextEditor() {
        XCTAssertFalse(ChatDeleteShortcut.isTextEditor(nil))
    }

    // MARK: The wiring

    /// The menu command must consult the router. Without this the enum is a
    /// rule nobody applies — which is the state the bug shipped in.
    func testTheMenuCommandRoutesTheKeystrokeInsteadOfClaimingIt() throws {
        let source = SourceScan.source("AppState.swift", from: #filePath)
        let action = try XCTUnwrap(
            source.range(of: "func requestChatDeletionFromMenu"),
            "requestChatDeletionFromMenu moved — repoint this scan")
        let body = String(source[action.lowerBound...].prefix(1200))
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
