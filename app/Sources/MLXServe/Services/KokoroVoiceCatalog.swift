import Foundation

/// Pure presentation logic for Kokoro's 54 built-in voices.
///
/// The wire ids (`af_bella`, `zm_yunjian`) are what the server wants and what a
/// blend spec is made of, so they are never rewritten — but a menu of 54 raw
/// ids is unreadable, and their two-letter prefix already encodes language and
/// gender. This decodes that for display and groups the list, so the tray menu
/// and Settings render the same names from one place.
enum KokoroVoiceCatalog {

    /// Language, keyed by the FIRST letter of a voice id. Order is the order
    /// sections appear in the menu: the two English variants first, since the
    /// model is strongest there and most users want them.
    static let languages: [(code: Character, name: String)] = [
        ("a", "American English"),
        ("b", "British English"),
        ("e", "Spanish"),
        ("f", "French"),
        ("h", "Hindi"),
        ("i", "Italian"),
        ("j", "Japanese"),
        ("p", "Portuguese"),
        ("z", "Chinese"),
    ]

    static func languageName(for id: String) -> String? {
        guard let first = id.first else { return nil }
        return languages.first { $0.code == first }?.name
    }

    /// "f" → "female", "m" → "male", from the SECOND letter.
    static func gender(for id: String) -> String? {
        let chars = Array(id)
        guard chars.count > 1 else { return nil }
        switch chars[1] {
        case "f": return "female"
        case "m": return "male"
        default: return nil
        }
    }

    /// The bare given name, capitalised: `af_bella` → "Bella".
    static func shortName(for id: String) -> String {
        guard let underscore = id.firstIndex(of: "_") else { return id }
        let raw = String(id[id.index(after: underscore)...])
        guard let f = raw.first else { return id }
        return raw.replacingOccurrences(of: "_", with: " ")
            .replacingCharacters(in: raw.startIndex..<raw.index(after: raw.startIndex),
                                 with: String(f).uppercased())
    }

    /// Menu-row label: "Bella — American English, female". Falls back to the
    /// raw id for anything that does not parse, so an unrecognised voice is
    /// still selectable rather than showing as blank.
    static func displayName(for id: String) -> String {
        guard let lang = languageName(for: id), let g = gender(for: id) else { return id }
        return "\(shortName(for: id)) — \(lang), \(g)"
    }

    /// Compact label for a possibly-blended spec: "Bella + Sky". Used where
    /// space is tight (the tray's collapsed label), and it must never be sent
    /// back to the server as a voice.
    static func blendDisplayName(for spec: String) -> String {
        let parts = spec.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        if parts.isEmpty { return "Kokoro" }
        if parts.count == 1 { return shortName(for: parts[0]) }
        return parts.map { shortName(for: $0) }.joined(separator: " + ")
    }

    /// True when the spec names more than one voice (i.e. it is a blend).
    static func isBlend(_ spec: String) -> Bool {
        spec.split(separator: ",").filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }.count > 1
    }

    /// Voices grouped by language, preserving `languages` order and dropping
    /// empty groups. Voices whose prefix does not parse land in a trailing
    /// "Other" group rather than vanishing from the menu.
    static func grouped(_ ids: [String] = AudioModelPreset.kokoroVoices) -> [(language: String, voices: [String])] {
        var out: [(language: String, voices: [String])] = []
        var claimed = Set<String>()
        for lang in languages {
            let group = ids.filter { $0.first == lang.code && gender(for: $0) != nil }
            if group.isEmpty { continue }
            claimed.formUnion(group)
            out.append((lang.name, group))
        }
        let rest = ids.filter { !claimed.contains($0) }
        if !rest.isEmpty { out.append(("Other", rest)) }
        return out
    }

    /// The sentence a voice preview speaks.
    ///
    /// Deliberately short — a preview is synthesized on demand and the point is
    /// to hear the timbre, not to wait. It names the voice so a run of previews
    /// stays distinguishable, and it carries a comma so the listener hears the
    /// model's prosody rather than a flat single clause.
    static func previewSentence(for spec: String) -> String {
        let who = blendDisplayName(for: spec)
        if isBlend(spec) {
            return "This is \(who), a blend of two voices."
        }
        return "Hi, I'm \(who). This is how I sound."
    }
}
