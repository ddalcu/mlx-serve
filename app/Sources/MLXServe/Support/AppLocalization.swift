import Foundation

/// Localizes strings that are produced by model types rather than written
/// directly inside a `Text`/`Button` literal, which SwiftUI resolves on its own.
///
/// Both paths go through `Bundle.main`, so the app's own language resolution —
/// the user's preferred languages, `CFBundleLocalizations`, and the per-app
/// language override in System Settings — decides the language exactly once.
enum L10n {
    static func text(_ key: String) -> String {
        Bundle.main.localizedString(forKey: key, value: key, table: nil)
    }

    /// `key` is the English sentence with %-placeholders, so it doubles as the
    /// English fallback and as the format string for every translation.
    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: text(key), locale: Locale.current, arguments: arguments)
    }

    /// `format`, with numbers kept as plain digits.
    ///
    /// The locale-aware path groups them — 1536 px comes out "1,536 px" — while
    /// the values these strings quote (a resolution grid, a step count) are
    /// printed ungrouped everywhere else in the app, and a test pins the range
    /// note at "256 … 1536". One number, one spelling.
    static func formatUngrouped(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: text(key), arguments: arguments)
    }
}
