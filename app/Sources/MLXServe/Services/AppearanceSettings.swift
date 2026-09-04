import SwiftUI
import AppKit

/// Client-only display preferences (Settings ▸ Interface). `@AppStorage`-
/// backed rather than on `ServerOptions`: these never touch a launch flag,
/// so they don't belong in the CLI-args mirroring rule that field owns.

/// The one spelling of each UserDefaults key — `ChatMetrics`, the Settings
/// rows and the appearance modifier all read/write through these, so a typo
/// can't split a reader from its writer.
enum InterfacePrefKey {
    static let appearanceMode = "appearanceMode"
    static let accentColor = "accentColorName"
    static let textSize = "chatTextSize"
    static let compactMode = "compactMode"
    static let chatColumn = "chatColumnWidth"
    /// Default `TerminalTheme` id for new sandbox terminals; a session can
    /// override it from its row's context menu.
    static let terminalTheme = "terminalTheme"
    /// "#RRGGBB" ground painted under the default terminal theme; "" = the
    /// theme's own.
    static let terminalBackground = "terminalBackground"
}

/// How wide a conversation reads, as a fraction of the SCREEN rather than of
/// the window.
///
/// Measuring the screen is what makes the setting mean something: the reading
/// width becomes a physical constant the eye can learn, and resizing the window
/// eats the margins rather than reflowing every line. Shrink the window past
/// the setting's own width and the text starts wrapping to the window instead,
/// because there is nothing left to give.
enum ChatColumnWidth: String, CaseIterable, Identifiable {
    case narrow, medium, wide
    var id: String { rawValue }
    var label: String {
        switch self {
        case .narrow: return "Narrow"
        case .medium: return "Medium"
        case .wide: return "Wide"
        }
    }
    /// The reading width in POINTS, or nil for "let the window decide"
    /// (`ChatMetrics.contentWidthFraction` of it, not the whole thing).
    ///
    /// Points rather than a fraction of anything: a reading measure is a
    /// physical property of text, not of the furniture around it. 840 is
    /// roughly 60 characters at the default size, 1260 half again as much.
    var proseWidth: CGFloat? {
        switch self {
        case .narrow: return 840
        case .medium: return 1260
        case .wide: return nil
        }
    }

    /// How wide your own turn may get before it wraps.
    ///
    /// Narrower than the column it sits in, because a right-aligned bubble that
    /// reaches the same left edge as the reply below it stops reading as one
    /// side of a conversation. It stops growing at Medium: past that the
    /// question is already a paragraph, and a wider one only makes the ragged
    /// left edge harder to follow back to.
    var userBubbleWidth: CGFloat {
        switch self {
        case .narrow: return 700
        case .medium, .wide: return 900
        }
    }

    /// F1 / F2 / F3, narrowest first, in View ▸ Interface.
    ///
    /// Spelled as the raw function-key scalars because `KeyEquivalent` has no
    /// named cases for them; these are AppKit's `NSF1FunctionKey` and its two
    /// successors. Unmodified, since the menu is where they are discoverable
    /// and nothing else in the app claims a bare function key.
    var menuShortcut: KeyEquivalent {
        switch self {
        case .narrow: return KeyEquivalent("\u{F704}")
        case .medium: return KeyEquivalent("\u{F705}")
        case .wide: return KeyEquivalent("\u{F706}")
        }
    }

    static var current: ChatColumnWidth {
        ChatColumnWidth(rawValue: UserDefaults.standard.string(forKey: InterfacePrefKey.chatColumn) ?? "") ?? .wide
    }
}

enum AppAppearanceMode: String, CaseIterable, Identifiable {
    case system, light, dark
    var id: String { rawValue }
    var label: String {
        switch self {
        case .system: return "System"
        case .light: return "Light"
        case .dark: return "Dark"
        }
    }
    /// nil = follow the system appearance (no override).
    var colorScheme: ColorScheme? {
        switch self {
        case .system: return nil
        case .light: return .light
        case .dark: return .dark
        }
    }
    /// The AppKit twin, for windows whose chrome SwiftUI does not own (the
    /// Quick Launcher's NSPanel + its NSVisualEffectView material — forced-dark
    /// content over a system-light vibrancy reads as a broken half-theme).
    var nsAppearance: NSAppearance? {
        switch self {
        case .system: return nil
        case .light: return NSAppearance(named: .aqua)
        case .dark: return NSAppearance(named: .darkAqua)
        }
    }

    static var current: AppAppearanceMode {
        AppAppearanceMode(rawValue: UserDefaults.standard.string(forKey: InterfacePrefKey.appearanceMode) ?? "") ?? .system
    }
}

enum ChatTextSize: String, CaseIterable, Identifiable {
    case small, medium, large, xlarge
    var id: String { rawValue }
    var label: String {
        switch self {
        case .small: return "Small"
        case .medium: return "Default"
        case .large: return "Large"
        case .xlarge: return "Extra Large"
        }
    }
    /// Prose size (`ChatMetrics.transcriptFontSize`). `.medium` is the size
    /// this shipped with before the setting existed — changing it moves
    /// everyone, changing the others doesn't.
    var proseSize: CGFloat {
        switch self {
        case .small: return 12
        case .medium: return 14
        case .large: return 16
        case .xlarge: return 19
        }
    }
    /// Fenced/inline code size (`ChatMetrics.transcriptCodeFontSize`) — kept
    /// 1–2pt under prose at every step (mono glyphs run wide, so code at
    /// prose size reads larger than the sentence around it).
    var codeSize: CGFloat {
        switch self {
        case .small: return 11
        case .medium: return 13
        case .large: return 15
        case .xlarge: return 17
        }
    }
}

enum AppAccentColor: String, CaseIterable, Identifiable {
    case system, blue, purple, pink, red, orange, yellow, green, graphite
    var id: String { rawValue }
    var label: String { self == .system ? "System" : rawValue.capitalized }
    /// nil = follow the system accent color (no `.tint` override).
    var color: Color? {
        switch self {
        case .system: return nil
        case .blue: return .blue
        case .purple: return .purple
        case .pink: return .pink
        case .red: return .red
        case .orange: return .orange
        case .yellow: return .yellow
        case .green: return .green
        case .graphite: return .gray
        }
    }
}

/// Applied to every window scene's root content in `MLXCoreApp.body` — BY
/// HAND per scene, so a new scene CAN forget it; the scan in
/// `AppearanceSettingsTests` is what catches that, the same reasoning as the
/// window-injection rules in app/CLAUDE.md.
struct AppAppearance: ViewModifier {
    @AppStorage(InterfacePrefKey.appearanceMode) private var modeRaw = AppAppearanceMode.system.rawValue
    @AppStorage(InterfacePrefKey.accentColor) private var accentRaw = AppAccentColor.system.rawValue

    func body(content: Content) -> some View {
        let mode = AppAppearanceMode(rawValue: modeRaw) ?? .system
        let accent = AppAccentColor(rawValue: accentRaw) ?? .system
        content
            .preferredColorScheme(mode.colorScheme)
            .tint(accent.color)
    }
}

extension View {
    func appAppearance() -> some View { modifier(AppAppearance()) }
}
