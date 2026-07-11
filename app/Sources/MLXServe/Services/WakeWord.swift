import Foundation

/// Pure wake-word detector for the hands-free voice assistant. Given a finalized
/// speech transcript, decide whether it is addressed to the assistant — i.e. it
/// opens with the wake phrase ("Hey Loki") — and, if so, return the remaining
/// query verbatim (which may be empty when the user spoke only the wake phrase).
/// Returns `nil` when no wake phrase is present, so the caller can ignore ambient
/// speech.
///
/// STT transcripts of an unusual proper noun are noisy, so matching is tolerant:
/// case- and punctuation-insensitive, accepting common greetings before the name
/// ("hey", "hi", "ok", …) and the usual mis-hearings of "Loki". To keep everyday
/// speech from waking the assistant, the looser mis-hearings ("low key", "loci",
/// …) are honored *only* when a greeting precedes them; a bare, greeting-less
/// match requires the exact name.
enum WakeWord {
    static let defaultPhrase = "hey loki"

    /// Normalize a user-typed phrase from Settings ("  Hey,  JARVIS! " →
    /// "hey jarvis"). Returns nil when no word survives — callers fall back
    /// to `defaultPhrase` so a blank field can never produce a gate that
    /// matches nothing.
    static func normalizePhrase(_ raw: String) -> String? {
        let toks = tokenize(raw).map(\.norm)
        return toks.isEmpty ? nil : toks.joined(separator: " ")
    }

    /// Title-cased phrase for UI labels and prompts ("hey jarvis" → "Hey Jarvis").
    static func display(_ phrase: String) -> String {
        phrase.split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    /// The assistant's name — the phrase's last word ("hey jarvis" → "Jarvis").
    static func assistantName(_ phrase: String) -> String {
        display(phrase).split(separator: " ").last.map(String.init) ?? display(phrase)
    }

    /// Stand-alone greetings tolerated before the name, so "Loki", "Hi Loki" and
    /// "OK Loki" all open the assistant just like "Hey Loki".
    private static let greetings = ["hey", "hi", "hello", "ok", "okay", "yo"]

    /// Distinctive-name mis-hearings, allowed only with a leading greeting.
    private static let homophones: [String: [String]] = [
        "loki": ["loki", "low key", "lowkey", "loci", "lokey", "lokie", "loaky"]
    ]

    /// Detect the wake phrase at the start of `transcript`. Returns the trimmed
    /// remaining query (possibly `""` for a bare wake phrase), or `nil` if absent.
    static func strip(_ transcript: String, phrase: String = defaultPhrase) -> String? {
        let toks = tokenize(transcript)
        guard !toks.isEmpty else { return nil }
        let norms = toks.map(\.norm)

        for prefix in acceptedPrefixes(for: phrase) where prefix.count <= norms.count {
            guard Array(norms.prefix(prefix.count)) == prefix else { continue }
            if prefix.count == toks.count { return "" }      // only the wake phrase
            let start = toks[prefix.count].start
            return String(transcript[start...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return nil
    }

    // MARK: - internals

    private struct Token { let norm: String; let start: String.Index }

    /// Split into lowercased alphanumeric tokens, remembering where each token
    /// begins in the ORIGINAL string so the returned query keeps its real text
    /// (punctuation, casing) instead of the normalized form.
    private static func tokenize(_ s: String) -> [Token] {
        var out: [Token] = []
        var startIdx: String.Index?
        var cur = ""
        var i = s.startIndex
        while i < s.endIndex {
            let ch = s[i]
            if ch.isLetter || ch.isNumber {
                if startIdx == nil { startIdx = i }
                cur.append(contentsOf: ch.lowercased())
            } else if let st = startIdx {
                out.append(Token(norm: cur, start: st)); startIdx = nil; cur = ""
            }
            i = s.index(after: i)
        }
        if let st = startIdx { out.append(Token(norm: cur, start: st)) }
        return out
    }

    /// Accepted wake prefixes as normalized token arrays, longest first so a
    /// greeting+name match strips the greeting too.
    private static func acceptedPrefixes(for phrase: String) -> [[String]] {
        let parts = tokenize(phrase).map(\.norm)
        guard let name = parts.last else { return [] }
        let phraseGreeting = Array(parts.dropLast())

        let looseNames = homophones[name] ?? [name]
        var greetingSets: [[String]] = greetings.map { [$0] }
        if !phraseGreeting.isEmpty { greetingSets.append(phraseGreeting) }

        var prefixes: [[String]] = []
        for g in greetingSets {
            for n in looseNames { prefixes.append(g + n.split(separator: " ").map(String.init)) }
        }
        prefixes.append([name])      // bare exact name only (no loose homophones)

        // Dedup, longest first.
        var seen = Set<String>(), uniq: [[String]] = []
        for p in prefixes.sorted(by: { $0.count > $1.count }) where seen.insert(p.joined(separator: " ")).inserted {
            uniq.append(p)
        }
        return uniq
    }
}
