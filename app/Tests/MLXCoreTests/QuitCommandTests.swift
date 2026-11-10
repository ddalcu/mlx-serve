import XCTest
@testable import MLXCore

/// ⌘Q with "Keep running when I press ⌘Q" on closes the user windows and
/// leaves the app and server up; off, it quits as before (issue #1024).
final class QuitCommandTests: XCTestCase {

    private typealias Window = ActivationPolicyManager.WindowInfo

    private let chat = Window(isVisible: true, isMiniaturized: false, isPanel: false,
                              canBecomeMain: true, className: "SwiftUI.AppKitWindow")
    private let minimized = Window(isVisible: false, isMiniaturized: true, isPanel: false,
                                   canBecomeMain: true, className: "SwiftUI.AppKitWindow")
    private let statusItem = Window(isVisible: true, isMiniaturized: false, isPanel: false,
                                    canBecomeMain: false, className: "NSStatusBarWindow")
    private let trayPopover = Window(isVisible: true, isMiniaturized: false, isPanel: true,
                                     canBecomeMain: false, className: "NSMenuBarExtraPanel")
    private let launcher = Window(isVisible: true, isMiniaturized: false, isPanel: true,
                                  canBecomeMain: false, className: "QuickLauncherPanel")

    func testOffQuitsTheApp() {
        XCTAssertEqual(QuitCommand.action(keepRunning: false, windows: [chat, statusItem]), .terminate)
    }

    func testOnClosesEveryUserWindowAndNeverTheMenuBarChrome() {
        let windows = [statusItem, chat, trayPopover, minimized, launcher]
        XCTAssertEqual(QuitCommand.action(keepRunning: true, windows: windows), .closeWindows([1, 3]))
    }

    func testOnNeverQuitsEvenWithNoWindowOpen() {
        XCTAssertEqual(QuitCommand.action(keepRunning: true, windows: [statusItem]), .closeWindows([]))
    }

    func testTheSettingIsAppLevelAndOffByDefault() throws {
        XCTAssertFalse(ServerOptions().quitKeepsRunning)
        let legacy = try JSONDecoder().decode(ServerOptions.self, from: Data("{}".utf8))
        XCTAssertFalse(legacy.quitKeepsRunning)

        var on = ServerOptions()
        on.quitKeepsRunning = true
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: try JSONEncoder().encode(on))
        XCTAssertTrue(decoded.quitKeepsRunning)
        XCTAssertEqual(on.toCLIArgs(), ServerOptions().toCLIArgs())
        XCTAssertTrue(on.serverLaunchEquals(ServerOptions()))
    }
}
