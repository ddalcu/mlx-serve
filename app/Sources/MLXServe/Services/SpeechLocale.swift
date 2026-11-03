import Foundation

enum SpeechLocale {
    /// The language the user speaks: their first preferred language. `fallback` (bundle-mangled `Locale.current`)
    /// stands in only for empty preferences.
    static func reportingLocale(preferredLanguages: [String], fallback: Locale) -> Locale {
        guard let first = preferredLanguages.first else { return fallback }
        return Locale(identifier: first)
    }
}
