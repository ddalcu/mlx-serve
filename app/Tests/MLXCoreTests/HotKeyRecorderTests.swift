import XCTest
import Carbon.HIToolbox
@testable import MLXCore

/// The recorder's capture rules and the display map they lean on. The AppKit
/// event plumbing (performKeyEquivalent, first-responder handoff) is not
/// testable here — these pin the decisions it feeds.
final class HotKeyRecorderTests: XCTestCase {
    func testAComboNeedsAModifierAndADisplayableKey() {
        XCTAssertEqual(HotKeyCapture.verdict(keyCode: UInt32(kVK_ANSI_K), modifiers: 0),
                       .needsModifier)
        // F13 has no name in the display map, so binding it would render as
        // "Key 105" everywhere the combo is shown.
        XCTAssertEqual(HotKeyCapture.verdict(keyCode: UInt32(kVK_F13), modifiers: UInt32(cmdKey)),
                       .unmapped)
        XCTAssertEqual(HotKeyCapture.verdict(keyCode: UInt32(kVK_ANSI_K),
                                             modifiers: UInt32(cmdKey | shiftKey)),
                       .captured(keyCode: UInt32(kVK_ANSI_K), modifiers: UInt32(cmdKey | shiftKey)))
    }

    /// Punctuation is recordable, so it has to display as itself — the
    /// "Key 47" fallback is for combos stored by older builds only.
    func testPunctuationKeysDisplayAsThemselves() {
        XCTAssertEqual(HotKeyDisplay.string(keyCode: UInt32(kVK_ANSI_Period),
                                            carbonModifiers: UInt32(cmdKey)), "⌘.")
        XCTAssertEqual(HotKeyDisplay.keySymbol(UInt32(kVK_ANSI_Semicolon)), ";")
        XCTAssertTrue(HotKeyDisplay.hasName(UInt32(kVK_ANSI_Grave)))
    }
}
