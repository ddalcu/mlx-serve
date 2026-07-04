import Foundation

/// Incremental sentence segmenter for the avatar's sentence-level TTS
/// pipelining (plan §7.5): stream LLM deltas in with `feed`, get back COMPLETE
/// sentences ready to synthesize while the next sentence is still decoding.
///
/// Pure value type — the queue discipline (TTS jobs interleaving with chat
/// decode) lives in the caller; this only decides sentence boundaries.
///
/// Rules (deliberately KISS — a TTS voice forgives a rare bad split):
/// - A sentence ends at `.` `!` `?` `…` followed by whitespace/end, unless the
///   `.` sits between digits (decimals, versions: "3.14", "v2.5").
/// - A newline closes any non-empty fragment (list items, headings).
/// - Fragments shorter than `minChars` are held and merged with the next
///   sentence (avoids machine-gun "Ok." clips); `flush()` always releases.
/// - Markdown emphasis/backticks are stripped — the avatar SPEAKS the text.
struct SentenceStreamer {
    private var buffer: String = ""
    private var held: String = ""
    let minChars: Int

    init(minChars: Int = 0) {
        self.minChars = minChars
    }

    /// Feed a streaming delta; returns any sentences completed by it.
    mutating func feed(_ delta: String) -> [String] {
        buffer += delta
        var out: [String] = []
        while let boundary = nextBoundary() {
            let raw = String(buffer[..<boundary.end])
            buffer.removeSubrange(..<boundary.end)
            let cleaned = Self.cleanForSpeech(raw)
            guard !cleaned.isEmpty else { continue }
            let candidate = held.isEmpty ? cleaned : held + " " + cleaned
            if candidate.count < minChars {
                held = candidate
            } else {
                held = ""
                out.append(candidate)
            }
        }
        return out
    }

    /// Release whatever is left (end of the LLM turn). Nil when nothing
    /// speakable remains.
    mutating func flush() -> String? {
        let rest = Self.cleanForSpeech(buffer)
        buffer = ""
        let candidate = held.isEmpty ? rest : (rest.isEmpty ? held : held + " " + rest)
        held = ""
        return candidate.isEmpty ? nil : candidate
    }

    // MARK: - Internals

    private struct Boundary {
        let end: String.Index // exclusive end of the sentence (incl. terminator)
    }

    /// Find the earliest complete sentence boundary in `buffer`. A terminator
    /// only counts once we can SEE the next character (or a newline) — a
    /// trailing "." might still be "3." awaiting "14".
    private func nextBoundary() -> Boundary? {
        var i = buffer.startIndex
        while i < buffer.endIndex {
            let ch = buffer[i]
            if ch == "\n" {
                // Consume the newline; cleanForSpeech trims it from the text.
                return Boundary(end: buffer.index(after: i))
            }
            if ch == "!" || ch == "?" || ch == "…" || ch == "." {
                let next = buffer.index(after: i)
                guard next < buffer.endIndex else { return nil } // need lookahead
                if ch == "." {
                    // Digit on both sides → decimal/version, not a boundary.
                    let prevIsDigit = i > buffer.startIndex && buffer[buffer.index(before: i)].isNumber
                    if prevIsDigit && buffer[next].isNumber {
                        i = next
                        continue
                    }
                }
                if buffer[next].isWhitespace {
                    return Boundary(end: next)
                }
            }
            i = buffer.index(after: i)
        }
        return nil
    }

    /// Strip markdown noise + trim for TTS.
    static func cleanForSpeech(_ s: String) -> String {
        var t = s
        t = t.replacingOccurrences(of: "**", with: "")
        t = t.replacingOccurrences(of: "`", with: "")
        // Single asterisks used for emphasis (leave arithmetic like "2 * 3").
        t = t.replacingOccurrences(of: #"\*(\S[^*]*)\*"#, with: "$1", options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
