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

    /// The phrase hands-free mode should listen for right now.
    ///
    /// An agent's own phrase wins over the Settings one: voice launched from a
    /// chat adopts that chat's agent — persona, voice, tools — and the phrase
    /// has to come along, or an agent that introduces itself by name still only
    /// answers to "hey loki" and the "Say “…”" hint tells the user to say the
    /// wrong thing. Both sides go through `normalizePhrase`, so a half-saved or
    /// whitespace-only agent field defers to the global setting instead of
    /// producing a gate that never matches, and an empty global still yields
    /// `defaultPhrase` rather than "" (which `strip` would match on everything).
    static func activePhrase(agentPhrase: String?, global: String) -> String {
        agentPhrase.flatMap(normalizePhrase)
            ?? normalizePhrase(global)
            ?? defaultPhrase
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

    /// Multi-agent detection: which of several phrases opened the utterance, and
    /// the remaining query. Every agent with a wake phrase is listening at once
    /// (plus the app's own phrase), so the ordering matters — LONGEST phrase
    /// first, or "hey loki" eats "hey loki coder" and the specific agent can
    /// never be reached by voice. Blank phrases are ignored rather than matching
    /// everything.
    static func match(_ transcript: String,
                      phrases: [(id: UUID, phrase: String)]) -> (id: UUID, query: String)? {
        let candidates = phrases
            .compactMap { entry -> (id: UUID, phrase: String, length: Int)? in
                guard let norm = normalizePhrase(entry.phrase) else { return nil }
                return (entry.id, norm, tokenize(norm).count)
            }
            .sorted { $0.length > $1.length }

        for candidate in candidates {
            if let query = strip(transcript, phrase: candidate.phrase) {
                return (candidate.id, query)
            }
        }
        return nil
    }

    /// True when `phrase` can't be told apart from one of `others` — checked when
    /// the user saves an agent, because a colliding phrase makes BOTH agents
    /// unreachable by voice and there's nothing to see until you try talking.
    ///
    /// The test is the phrase's NAME (its last word), not the whole phrase:
    /// greetings are universal ("hey X", "ok X" and a bare "X" all open the same
    /// gate), so two phrases ending in the same word are one gate.
    static func collides(_ phrase: String, with others: [String]) -> Bool {
        guard let name = normalizePhrase(phrase)?.split(separator: " ").last else { return false }
        return others.contains { other in
            normalizePhrase(other)?.split(separator: " ").last == name
        }
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
