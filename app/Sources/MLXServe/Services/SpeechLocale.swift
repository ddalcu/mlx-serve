import Foundation

enum SpeechLocale {
    enum Resolution: Equatable {
        case available(Locale)
        case unavailable(Locale)
        case unsupported(Locale)

        var locale: Locale {
            switch self {
            case .available(let locale), .unavailable(let locale), .unsupported(let locale):
                return locale
            }
        }

        var isAvailable: Bool {
            if case .available = self { return true }
            return false
        }
    }

    static func resolve(preferredLanguages: [String],
                        supportedLocales: [Locale],
                        isAvailable: (Locale) -> Bool,
                        fallback: Locale) -> Resolution {
        let supported = supportedLocales.sorted { $0.identifier < $1.identifier }
        var firstMatch: Locale?

        for identifier in preferredLanguages {
            let preferred = Locale(identifier: identifier)
            let matches = supported
                .filter { sameLanguage(preferred, $0) }
                .sorted { rank($0, for: preferred) < rank($1, for: preferred) }
            if firstMatch == nil { firstMatch = matches.first }
            if let available = matches.first(where: isAvailable) {
                return .available(available)
            }
        }

        if let firstMatch { return .unavailable(firstMatch) }
        return .unsupported(reportingLocale(preferredLanguages: preferredLanguages, fallback: fallback))
    }

    /// Report-only locale when nothing matched: the language the user is
    /// trying to speak. `fallback` (bundle-mangled `Locale.current`) stands in
    /// only for empty preferences — never build a recognizer from the result.
    static func reportingLocale(preferredLanguages: [String], fallback: Locale) -> Locale {
        guard let first = preferredLanguages.first else { return fallback }
        return Locale(identifier: first)
    }

    private static func sameLanguage(_ lhs: Locale, _ rhs: Locale) -> Bool {
        guard let lhsCode = lhs.language.languageCode,
              let rhsCode = rhs.language.languageCode else { return false }
        return lhsCode == rhsCode
    }

    private static func rank(_ candidate: Locale, for preferred: Locale) -> (Int, Int, Int, String) {
        let equivalent = preferred.language.isEquivalent(to: candidate.language) ? 0 : 1
        let script = preferred.language.script
        let scriptRank = script == nil || script == candidate.language.script ? 0 : 1
        let region = preferred.region
        let regionRank = region == nil || region == candidate.region ? 0 : 1
        return (equivalent, scriptRank, regionRank, candidate.identifier)
    }
}
