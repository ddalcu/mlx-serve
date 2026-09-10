import Foundation

/// A terminal color scheme: the 16 ANSI slots plus the default text and
/// ground. SwiftTerm ships none (one dim xterm palette), so these are ours.
/// Pure — the SwiftTerm mapping lives in `EmbeddedTerminalView`.
struct TerminalTheme: Identifiable, Equatable {

    struct RGB: Equatable {
        let r: UInt8, g: UInt8, b: UInt8

        init(_ r: UInt8, _ g: UInt8, _ b: UInt8) { self.r = r; self.g = g; self.b = b }

        /// "#RRGGBB" (case-insensitive, `#` optional); nil for anything else.
        init?(hex: String) {
            var h = hex.trimmingCharacters(in: .whitespaces)
            if h.hasPrefix("#") { h.removeFirst() }
            guard h.count == 6, let v = UInt32(h, radix: 16) else { return nil }
            self.init(UInt8(v >> 16 & 0xff), UInt8(v >> 8 & 0xff), UInt8(v & 0xff))
        }

        var hex: String { String(format: "#%02X%02X%02X", r, g, b) }
    }

    let id: String
    let name: String
    let ansi: [RGB]      // black, red, green, yellow, blue, magenta, cyan, white, then the bright eight
    let foreground: RGB
    let background: RGB

    static let defaultId = "dark"

    static func theme(_ id: String) -> TerminalTheme? { all.first { $0.id == id } }

    /// What a session paints: its own pick, else the Settings default. The
    /// background override applies to the DEFAULT only — a per-session theme
    /// keeps its own ground, or every theme would look the same.
    static func resolve(sessionThemeId: String?,
                        defaults: UserDefaults = .standard) -> (theme: TerminalTheme, background: RGB) {
        if let id = sessionThemeId, let t = theme(id) { return (t, t.background) }
        let t = theme(defaults.string(forKey: InterfacePrefKey.terminalTheme) ?? "")
            ?? theme(defaultId)!
        let override = defaults.string(forKey: InterfacePrefKey.terminalBackground).flatMap(RGB.init(hex:))
        return (t, override ?? t.background)
    }

    // MARK: - Catalogue

    private static func p(_ hexes: [String]) -> [RGB] { hexes.map { RGB(hex: $0)! } }

    static let all: [TerminalTheme] = [
        TerminalTheme(id: "dark", name: "Dark",
                      ansi: p(["#000000", "#E05252", "#23D18B", "#E5E510", "#3B8EEA", "#BC3FBC", "#29B8DB", "#E5E5E5",
                               "#808080", "#F14C4C", "#23D18B", "#F5F543", "#3B8EEA", "#D670D6", "#29B8DB", "#FFFFFF"]),
                      foreground: RGB(hex: "#E5E5E5")!, background: RGB(hex: "#1C1C1C")!),
        TerminalTheme(id: "pro", name: "Pro (black)",
                      ansi: p(["#000000", "#FF6B68", "#A8FF60", "#FFFFB6", "#96CBFE", "#FF73FD", "#C6C5FE", "#EEEEEE",
                               "#7C7C7C", "#FF8785", "#B6FFB2", "#FFFFCC", "#B5DCFE", "#FF9CFE", "#DFDFFE", "#FFFFFF"]),
                      foreground: RGB(hex: "#F2F2F2")!, background: RGB(hex: "#000000")!),
        TerminalTheme(id: "solarized-dark", name: "Solarized Dark",
                      ansi: p(["#073642", "#DC322F", "#859900", "#B58900", "#268BD2", "#D33682", "#2AA198", "#EEE8D5",
                               "#586E75", "#CB4B16", "#93A1A1", "#657B83", "#839496", "#6C71C4", "#93A1A1", "#FDF6E3"]),
                      foreground: RGB(hex: "#93A1A1")!, background: RGB(hex: "#002B36")!),
        TerminalTheme(id: "solarized-light", name: "Solarized Light",
                      ansi: p(["#073642", "#DC322F", "#859900", "#B58900", "#268BD2", "#D33682", "#2AA198", "#EEE8D5",
                               "#586E75", "#CB4B16", "#586E75", "#657B83", "#839496", "#6C71C4", "#93A1A1", "#FDF6E3"]),
                      foreground: RGB(hex: "#586E75")!, background: RGB(hex: "#FDF6E3")!),
        TerminalTheme(id: "dracula", name: "Dracula",
                      ansi: p(["#21222C", "#FF5555", "#50FA7B", "#F1FA8C", "#BD93F9", "#FF79C6", "#8BE9FD", "#F8F8F2",
                               "#6272A4", "#FF6E6E", "#69FF94", "#FFFFA5", "#D6ACFF", "#FF92DF", "#A4FFFF", "#FFFFFF"]),
                      foreground: RGB(hex: "#F8F8F2")!, background: RGB(hex: "#282A36")!),
        TerminalTheme(id: "one-dark", name: "One Dark",
                      ansi: p(["#282C34", "#E06C75", "#98C379", "#E5C07B", "#61AFEF", "#C678DD", "#56B6C2", "#ABB2BF",
                               "#5C6370", "#E06C75", "#98C379", "#E5C07B", "#61AFEF", "#C678DD", "#56B6C2", "#FFFFFF"]),
                      foreground: RGB(hex: "#ABB2BF")!, background: RGB(hex: "#282C34")!),
        TerminalTheme(id: "github-light", name: "GitHub Light",
                      ansi: p(["#24292F", "#CF222E", "#116329", "#4D2D00", "#0969DA", "#8250DF", "#1B7C83", "#6E7781",
                               "#57606A", "#A40E26", "#1A7F37", "#633C01", "#218BFF", "#A475F9", "#3192AA", "#8C959F"]),
                      foreground: RGB(hex: "#24292F")!, background: RGB(hex: "#FFFFFF")!),
    ]
}
