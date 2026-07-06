import XCTest
@testable import MLXCore

/// ActivationPolicyManager keeps the app's activation policy in sync with its
/// visible windows: any real user window (Chat, Audio Generation, Settings,
/// Welcome, …) → `.regular` (app appears in ⌘Tab + Dock); only menu-bar
/// chrome (status item, tray popover, quick-launcher panel) → `.accessory`.
/// The decision is pure over lightweight window descriptors.
final class ActivationPolicyTests: XCTestCase {

    private func userWindow(visible: Bool = true, miniaturized: Bool = false) -> ActivationPolicyManager.WindowInfo {
        .init(isVisible: visible,
              isMiniaturized: miniaturized,
              isPanel: false,
              canBecomeMain: true,
              className: "NSWindow")
    }

    func testNoWindowsIsAccessory() {
        XCTAssertEqual(ActivationPolicyManager.policy(for: []), .accessory)
    }

    func testChromeOnlyStaysAccessory() {
        let chrome: [ActivationPolicyManager.WindowInfo] = [
            // The status item in the menu bar — always "visible".
            .init(isVisible: true, isMiniaturized: false, isPanel: false,
                  canBecomeMain: false, className: "NSStatusBarWindow"),
            // The MenuBarExtra(.window) tray popover host.
            .init(isVisible: true, isMiniaturized: false, isPanel: true,
                  canBecomeMain: false, className: "NSMenuBarExtraPanel"),
            // The ⌃Space quick launcher (non-activating borderless NSPanel).
            .init(isVisible: true, isMiniaturized: false, isPanel: true,
                  canBecomeMain: false, className: "QuickLauncherPanel"),
        ]
        XCTAssertEqual(ActivationPolicyManager.policy(for: chrome), .accessory)
    }

    func testMenuBarExtraNamedWindowExcludedEvenIfNotAPanel() {
        // Defense in depth: SwiftUI's tray-popover host class has varied
        // across macOS releases (panel or plain window) — the name check must
        // exclude it either way.
        let tray = ActivationPolicyManager.WindowInfo(
            isVisible: true, isMiniaturized: false, isPanel: false,
            canBecomeMain: true, className: "NSMenuBarExtraWindow")
        XCTAssertEqual(ActivationPolicyManager.policy(for: [tray]), .accessory)
    }

    func testOneVisibleUserWindowIsRegular() {
        XCTAssertEqual(ActivationPolicyManager.policy(for: [userWindow()]), .regular)
    }

    func testUserWindowAmongChromeIsRegular() {
        let windows: [ActivationPolicyManager.WindowInfo] = [
            .init(isVisible: true, isMiniaturized: false, isPanel: false,
                  canBecomeMain: false, className: "NSStatusBarWindow"),
            userWindow(),
        ]
        XCTAssertEqual(ActivationPolicyManager.policy(for: windows), .regular)
    }

    func testClosedUserWindowDoesNotCount() {
        XCTAssertEqual(ActivationPolicyManager.policy(for: [userWindow(visible: false)]),
                       .accessory)
    }

    func testMiniaturizedUserWindowStillCountsAsRegular() {
        // A minimized window reports isVisible == false; dropping to
        // .accessory would strand it (no Dock, no ⌘Tab — unreachable).
        XCTAssertEqual(
            ActivationPolicyManager.policy(for: [userWindow(visible: false, miniaturized: true)]),
            .regular)
    }

    func testIntroWelcomeWindowCountsAsRegular() {
        // The intro/welcome window (shown on every launch) is a titled,
        // floating-level NSWindow created by AppState.showWelcomeWindow — the
        // app must be ⌘Tab-selectable while it's up, floating level or not.
        let welcome = ActivationPolicyManager.WindowInfo(
            isVisible: true, isMiniaturized: false, isPanel: false,
            canBecomeMain: true, className: "NSWindow")
        let chrome = ActivationPolicyManager.WindowInfo(
            isVisible: true, isMiniaturized: false, isPanel: false,
            canBecomeMain: false, className: "NSStatusBarWindow")
        XCTAssertEqual(ActivationPolicyManager.policy(for: [chrome, welcome]), .regular)
    }

    func testNonMainableBorderlessWindowDoesNotCount() {
        // Borderless helper windows (tooltips, overlays) can't become main
        // and must not summon a Dock icon.
        let overlay = ActivationPolicyManager.WindowInfo(
            isVisible: true, isMiniaturized: false, isPanel: false,
            canBecomeMain: false, className: "NSWindow")
        XCTAssertEqual(ActivationPolicyManager.policy(for: [overlay]), .accessory)
    }
}
