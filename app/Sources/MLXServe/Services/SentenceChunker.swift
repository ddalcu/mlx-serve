import Foundation

/// Splits a *growing* answer string into complete sentences so the synthesizer
/// can start speaking the first sentence while the rest is still being
/// generated. `ingest(fullText:)` is called with the whole answer-so-far on each
/// streaming update and returns only the sentences that became complete since
/// the previous call; `flush()` releases the trailing partial as a final chunk
/// when generation ends. Pure value type → unit-testable.
struct SentenceChunker {
    private var emitted = 0
    private var lastText = ""
    /// The trailing partial was released by `flush()`. Tracked instead of
    /// zeroing `emitted`/`lastText`, which rewound the chunker: voice mode
    /// re-ingests the finished answer whenever the session list republishes
    /// while TTS is still speaking, and a rewound chunker re-emitted the whole
    /// response — spoken twice.
    private var tailFlushed = false

    init() {}

    /// Newly-completed sentences since the last `ingest`. A sentence is complete
    /// once its terminator is followed by whitespace (or a newline appears) — a
    /// terminator at the very end is held back (it may still grow) until `flush`.
    mutating func ingest(fullText: String) -> [String] {
        lastText = fullText
        let sentences = Self.split(fullText).sentences
        guard sentences.count > emitted else { return [] }
        let fresh = Array(sentences[emitted...])
        emitted = sentences.count
        return fresh
    }

    /// Everything not yet spoken: any complete sentences missed plus the trailing
    /// partial. Idempotent — calling twice yields nothing the second time.
    mutating func flush() -> [String] {
        let (sentences, remainder) = Self.split(lastText)
        var result: [String] = []
        if sentences.count > emitted { result.append(contentsOf: sentences[emitted...]) }
        emitted = sentences.count
        let tail = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty, !tailFlushed {
            result.append(tail)
            tailFlushed = true   // consumed; don't re-emit on a later flush/ingest
        }
        return result
    }

    // MARK: - Pure splitter

    /// Splits `text` into complete sentences and the trailing (incomplete) remainder.
    /// Boundaries are `.?!` runs followed by whitespace, and hard newlines. Common
    /// abbreviations and single-letter initials don't trigger a split.
    static func split(_ text: String) -> (sentences: [String], remainder: String) {
        let chars = Array(text)
        var sentences: [String] = []
        var current = ""
        var i = 0

        func commit() {
            let t = current.trimmingCharacters(in: .whitespaces)
            if !t.isEmpty { sentences.append(t) }
            current = ""
        }

        while i < chars.count {
            let ch = chars[i]

            if ch == "\n" {           // hard boundary, newline itself dropped
                commit()
                i += 1
                continue
            }

            current.append(ch)

            if ch == "." || ch == "!" || ch == "?" {
                var j = i + 1
                // Absorb a run of terminators ("?!", "...") and trailing closers.
                while j < chars.count, chars[j] == "." || chars[j] == "!" || chars[j] == "?" {
                    current.append(chars[j]); j += 1
                }
                while j < chars.count, "\"')]}”’".contains(chars[j]) {
                    current.append(chars[j]); j += 1
                }
                if j < chars.count {
                    let next = chars[j]
                    if next == " " || next == "\t" || next == "\n" {
                        if !endsWithAbbreviation(current) { commit() }
                    }
                    i = j           // resume after the absorbed punctuation/closers
                    continue
                } else {
                    break           // terminator at end of text → hold as remainder
                }
            }
            i += 1
        }

        return (sentences, current)
    }

    private static let abbreviations: Set<String> = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "inc",
        "ltd", "co", "corp", "fig", "dept", "gen", "col", "sgt", "lt", "capt", "cmdr",
        "gov", "sen", "rep", "rev", "hon", "univ", "mt", "ave", "blvd", "approx",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    ]

    /// True if the token ending `s` (ignoring trailing dots/closers) is a known
    /// abbreviation or a single-letter initial (e.g. "Dr.", "U.S.", "John Q.").
    private static func endsWithAbbreviation(_ s: String) -> Bool {
        var chars = Array(s)
        while let last = chars.last, !last.isLetter && !last.isNumber { chars.removeLast() }
        var token = ""
        while let last = chars.last, last.isLetter || last.isNumber {
            token.insert(last, at: token.startIndex)
            chars.removeLast()
        }
        if token.count == 1 { return true }
        return abbreviations.contains(token.lowercased())
    }
}
