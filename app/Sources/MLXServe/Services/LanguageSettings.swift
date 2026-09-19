import Foundation

/// The app's own UI language (Settings ▸ Interface ▸ Language).
///
/// macOS already resolves a language from the system setting and the per-app
/// override in System Settings; this exists because reaching that override is
/// a trip to another app, and comparing two languages side by side there is
/// awkward. `.system` keeps the OS answer, which is also the default — a
/// user who never opens the row sees exactly what the app did before it had
/// one.
///
/// Two lookup paths have to agree on the answer, so both read this type:
///  - SwiftUI copy (`Text("…")`, `Label`, `Button`) resolves a literal against
///    the `\.locale` environment, which `AppChrome` sets on every scene root.
///  - Everything that goes through `Bundle.main` instead — `L10n.text` /
///    `L10n.format` (copy produced by models and services) and AppKit's own
///    `NSLocalizedString` (menus, window titles) — is what
///    `BundleLanguageOverride` points at the chosen `.lproj`.
enum AppLanguage: String, CaseIterable, Identifiable {
    case system
    case english = "en"
    case simplifiedChinese = "zh-Hans"

    var id: String { rawValue }

    /// Shown in the picker. A language names itself: only "System" is
    /// translated, because "English" and "简体中文" are the same words in
    /// every UI language, which is the point of listing them that way.
    var label: String {
        switch self {
        case .system: return "System"
        case .english: return "English"
        case .simplifiedChinese: return "简体中文"
        }
    }

    /// The `.lproj` code to resolve against; nil = let the system decide.
    var code: String? { self == .system ? nil : rawValue }

    /// nil = follow the system (no `\.locale` override).
    var locale: Locale? { code.map(Locale.init(identifier:)) }

    static var current: AppLanguage {
        AppLanguage(rawValue: UserDefaults.standard.string(forKey: InterfacePrefKey.language) ?? "") ?? .system
    }

    /// Applies `language` and then records the choice, in that order.
    ///
    /// The write is what re-renders the surface that shows the new language, and
    /// the `L10n` lookups run inside that render — so the bundle has to be
    /// swapped before it, not after, or the strings built on the way through
    /// stay in the language the user just left.
    static func select(_ language: AppLanguage, into defaults: UserDefaults = .standard, in host: Bundle = .main) {
        BundleLanguageOverride.apply(language, in: host)
        defaults.set(language.rawValue, forKey: InterfacePrefKey.language)
    }
}

/// Points `Bundle.main` string lookups at one `.lproj`.
///
/// A selected language has to reach copy the app resolved before SwiftUI sees
/// it — `L10n.text` in the models and services, and AppKit's own menu titles —
/// and those all read a bundle, not the environment. Swizzling
/// `Bundle.localizedString(forKey:value:table:)` is the one place they meet.
///
/// Scoped to `Bundle.main`: every other bundle keeps the system's answer, so
/// a framework's own strings are never rewritten. `nil` (`.system`, or a code
/// this build ships no `.lproj` for) restores the untouched lookup, which is
/// what keeps an unknown preference value harmless.
enum BundleLanguageOverride {
    /// The bundle the override resolves against; nil = pass through.
    private(set) static var languageBundle: Bundle?

    /// Exchanges the lookup once, on first use. `Bundle` is a class cluster,
    /// but `localizedString(forKey:value:table:)` is a real instance method on
    /// `Bundle` itself, so one exchange covers `Bundle.main` and every other
    /// bundle — and the override only rewrites the call for `Bundle.main`.
    private static let installOnce: Void = {
        guard let original = class_getInstanceMethod(Bundle.self, #selector(Bundle.localizedString(forKey:value:table:))),
              let replacement = class_getInstanceMethod(Bundle.self, #selector(Bundle.mlx_localizedString(forKey:value:table:)))
        else { return }
        method_exchangeImplementations(original, replacement)
    }()

    /// Resolves `code` inside `bundle`'s resources. Separated from `apply` so
    /// the lookup rule is testable without touching `Bundle.main`.
    static func bundle(for code: String, in bundle: Bundle = .main) -> Bundle? {
        guard let path = bundle.path(forResource: code, ofType: "lproj") else { return nil }
        return Bundle(path: path)
    }

    /// Applies `language` from here on. Cheap and idempotent — every scene
    /// root calls it, and the Settings row calls it the moment the picker
    /// moves.
    ///
    /// `host` is the bundle whose resources are searched: the app always
    /// passes `Bundle.main`, and tests resolve through the source tree, which
    /// SwiftPM excludes from the test bundle (`LocalizationTests` does the
    /// same thing for the catalog).
    static func apply(_ language: AppLanguage, in host: Bundle = .main) {
        _ = installOnce
        let resolved = language.code.flatMap { bundle(for: $0, in: host) }
        // A reapply of the same language is not an invalidation: every scene
        // calls this on appear, and bumping then would re-run every `L10n`
        // body for nothing. Only a real swap is news.
        guard languageBundle?.bundlePath != resolved?.bundlePath else { return }
        languageBundle = resolved
        LanguageLookupRevision.shared.bump()
    }
}

extension Bundle {
    /// The swizzled twin of `localizedString(forKey:value:table:)`: resolves
    /// against the selected language for the app's own bundle, and forwards
    /// everything else to the implementation it exchanged with.
    @objc func mlx_localizedString(forKey key: String, value: String?, table tableName: String?) -> String {
        if self == Bundle.main, let languageBundle = BundleLanguageOverride.languageBundle {
            return languageBundle.localizedString(forKey: key, value: value, table: tableName)
        }
        return mlx_localizedString(forKey: key, value: value, table: tableName)
    }
}
