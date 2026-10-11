import Foundation

/// Picks the catalog the whole app reads (Settings ▸ Interface ▸ Language).
///
/// `AppleLanguages` in the app's OWN domain is the per-app language override
/// macOS resolves a bundle from — the same key System Settings writes — and
/// it is the only lever that reaches every reader at once: `L10n`, and the
/// `Text("…")`/`Label`/`Button` literals SwiftUI resolves through CFBundle.
/// (Swapping `Bundle.main`'s class reaches `L10n` and nothing else, which
/// ships a screen that is half one language and half the other.)
///
/// A named catalog is written as that one code. Following the Mac writes
/// NOTHING — and undoes only what this app wrote before, which the companion
/// records. Writing a fixed catalog on every launch instead, even for
/// "follow the Mac", takes the language away from the Mac itself: a Korean
/// reader on a Korean Mac gets English from the first launch after updating,
/// and a per-app language picked in System Settings is overridden. macOS's
/// own per-app value lives in the same key of the same domain, so it is left
/// to macOS to resolve — and when it did move under us, the stamp goes, not
/// the value.
enum AppLanguageOverride {
    private static let preferredLanguagesKey = "AppleLanguages"

    /// The catalog the stored preference names, or nil to write nothing.
    /// A code this build no longer ships resolves to nil, so a preference
    /// left by a build with more languages degrades to following the Mac.
    static func catalog(preference: String?, localizations: [String]) -> String? {
        guard let preference, localizations.contains(preference) else { return nil }
        return preference
    }

    /// Reads the stored preference and writes the override. Runs at the top
    /// of `main()`, before the first string of the launch is localized; a
    /// catalog resolves once per process.
    static func install(preference: String?, localizations: [String],
                        defaults: UserDefaults = .standard) {
        if let catalog = catalog(preference: preference, localizations: localizations) {
            defaults.set([catalog], forKey: preferredLanguagesKey)
            defaults.set(catalog, forKey: InterfacePrefKey.languageApplied)
            return
        }
        // Following the Mac: take back the override this app wrote, and only
        // while it is still exactly that. A value this app never wrote is
        // System Settings' per-app language or a manual `defaults write`,
        // and deleting it loses the user's choice instead of following it.
        let applied = defaults.string(forKey: InterfacePrefKey.languageApplied)
        guard let applied else { return }  // nothing this app ever wrote
        let current = (defaults.array(forKey: preferredLanguagesKey) as? [String])?.first
        if current == applied {
            defaults.removeObject(forKey: preferredLanguagesKey)
        }
        // The value moved under us — System Settings changed it, or a manual
        // `defaults write` did. Drop the stamp either way: a value that is
        // not ours stays where its owner put it, and a later System Settings
        // choice of the code we once stamped must not be mistaken for ours.
        defaults.removeObject(forKey: InterfacePrefKey.languageApplied)
    }
}
