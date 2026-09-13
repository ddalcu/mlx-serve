import XCTest
@testable import MLXCore

/// The terminal theme catalogue + how a session's colors are decided. Pure —
/// SwiftTerm only ever sees the resolved 16 + 2 colors.
final class TerminalThemeTests: XCTestCase {

    func testEveryThemeIsAComplete16ColorPaletteWithAUniqueId() {
        XCTAssertGreaterThanOrEqual(TerminalTheme.all.count, 6)
        XCTAssertEqual(Set(TerminalTheme.all.map(\.id)).count, TerminalTheme.all.count)
        for theme in TerminalTheme.all {
            XCTAssertEqual(theme.ansi.count, 16, theme.id)
            XCTAssertNotEqual(theme.foreground, theme.background, "\(theme.id): text must be visible")
        }
        XCTAssertTrue(TerminalTheme.all.contains { $0.id == TerminalTheme.defaultId })
    }

    func testResolutionSessionPickBeatsDefaultAndOverrideOnlyPaintsTheDefault() {
        let defaults = UserDefaults(suiteName: "TerminalThemeTests")!
        defaults.removePersistentDomain(forName: "TerminalThemeTests")
        // Nothing set: the built-in default, its own background.
        var r = TerminalTheme.resolve(sessionThemeId: nil, defaults: defaults)
        XCTAssertEqual(r.theme.id, TerminalTheme.defaultId)
        XCTAssertEqual(r.background, r.theme.background)
        // Settings pick a default theme + a background override.
        defaults.set("solarized-dark", forKey: InterfacePrefKey.terminalTheme)
        defaults.set("#112233", forKey: InterfacePrefKey.terminalBackground)
        r = TerminalTheme.resolve(sessionThemeId: nil, defaults: defaults)
        XCTAssertEqual(r.theme.id, "solarized-dark")
        XCTAssertEqual(r.background, TerminalTheme.RGB(hex: "#112233"))
        // A per-session pick wins and keeps that theme's OWN ground — the
        // override belongs to the default, or every theme looks the same.
        r = TerminalTheme.resolve(sessionThemeId: "dracula", defaults: defaults)
        XCTAssertEqual(r.theme.id, "dracula")
        XCTAssertEqual(r.background, TerminalTheme.theme("dracula")?.background)
        // An unknown id (a theme retired later) falls back, never traps.
        r = TerminalTheme.resolve(sessionThemeId: "gone", defaults: defaults)
        XCTAssertEqual(r.theme.id, "solarized-dark")
        // Hex round-trips.
        XCTAssertEqual(TerminalTheme.RGB(hex: "#1c1c1c")?.hex, "#1C1C1C")
        XCTAssertNil(TerminalTheme.RGB(hex: "nope"))
    }
}
