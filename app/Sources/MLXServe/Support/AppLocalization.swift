import Foundation
import Observation

/// The invalidation source for the `L10n` lookup path.
///
/// `L10n.text` resolves to a plain `String` while a view body is built, so a
/// swap of the resolved bundle is invisible to SwiftUI unless something the
/// body READ changes. `L10n.text` reads `generation` below, and SwiftUI wraps
/// every `body` evaluation in observation tracking, so the read registers a
/// dependency: bumping the generation re-runs exactly the bodies that built
/// `L10n` copy — in every open window — while their `@State` survives. Keying a
/// scene root on the preference (`.id(languageRaw)`) refreshes the same copy
/// but destroys the subtree, taking a half-typed composer draft and every
/// scroll position with it.
@Observable
final class LanguageLookupRevision {
    static let shared = LanguageLookupRevision()
    private(set) var generation = 0

    /// Called by `BundleLanguageOverride.apply` when the chosen bundle
    /// actually changes; a reapplied language is not an invalidation.
    func bump() { generation &+= 1 }
}

/// Localizes strings that are produced by model types rather than written
/// directly inside a `Text`/`Button` literal, which SwiftUI resolves on its own.
///
/// Both paths go through `Bundle.main`, so the app's own language resolution —
/// the user's preferred languages, `CFBundleLocalizations`, and the per-app
/// language override in System Settings — decides the language exactly once.
enum L10n {
    static func text(_ key: String) -> String {
        // Reading the revision is what makes an `L10n` label re-resolve when
        // the language changes: inside a `body` this read is tracked, so the
        // swap that bumped the generation re-runs that body. Outside a body
        // (model copy, notifications) it is an ordinary read.
        _ = LanguageLookupRevision.shared.generation
        return Bundle.main.localizedString(forKey: key, value: key, table: nil)
    }

    /// `key` is the English sentence with %-placeholders, so it doubles as the
    /// English fallback and as the format string for every translation.
    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: text(key), locale: Locale.current, arguments: arguments)
    }
}
