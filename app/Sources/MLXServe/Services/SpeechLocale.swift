import Foundation

/// Resolves the locale voice mode recognizes against: the user's own speech
/// language, taken from `Locale.preferredLanguages` (the user's ordered speech
/// languages, bundle-independent). `Locale.current` is unsuitable here because
/// macOS derives it from the bundle's shipped localizations, which can resolve
/// to a pseudo-locale (e.g. `en_JP`) with no on-device dictation model.
///
/// Pure → unit-testable without the Speech framework; the live
/// `SFSpeechRecognizer` probe lives in `SpeechRecognizing.swift`.
enum SpeechLocale {
    /// Canonical Apple speech-recognizer key from a BCP-47 tag.
    ///
    /// Two steps, both required: `-`→`_` so `Locale(identifier:)` parses the
    /// region (`ja-JP` is otherwise left un-canonicalized), then read
    /// `Locale(identifier:).identifier`, which collapses a script+region tag to
    /// Apple's key (`zh-Hans-CN` → `zh_CN`). A naive `-`→`_` alone leaves the
    /// non-key `zh_Hans_CN`. A script-only tag (`zh-Hans`) has no region to key
    /// on and stays as-is, then gets filtered out because no recognizer binds.
    static func canonicalKey(from bcp47: String) -> String {
        Locale(identifier: bcp47.replacingOccurrences(of: "-", with: "_")).identifier
    }

    /// Ordered, de-duplicated canonical candidate keys for the user's preferred
    /// languages, highest priority first. Empty input yields an empty list.
    static func candidateKeys(preferredLanguages: [String]) -> [String] {
        var seen = Set<String>()
        var keys: [String] = []
        for lang in preferredLanguages {
            let key = canonicalKey(from: lang)
            if !key.isEmpty, seen.insert(key).inserted { keys.append(key) }
        }
        return keys
    }

    /// The locale to recognize against: the first candidate whose on-device model
    /// is installed (`supportsOnDevice` returns true). If none is installed, the
    /// first candidate anyway — so the caller's "dictation unavailable" card names
    /// the user's own language. If there are no candidates, `fallback` (the caller
    /// passes `Locale.current`). `supportsOnDevice` is injected so this stays pure.
    static func resolve(preferredLanguages: [String],
                        supportsOnDevice: (Locale) -> Bool,
                        fallback: Locale) -> Locale {
        let keys = candidateKeys(preferredLanguages: preferredLanguages)
        for key in keys {
            let locale = Locale(identifier: key)
            if supportsOnDevice(locale) { return locale }
        }
        if let first = keys.first { return Locale(identifier: first) }
        return fallback
    }
}
