import Foundation

/// Drops a reference marker (`<Picture 2>`, `<Audio 1>`) into the prompt where
/// the caret is, with the spaces a person would have typed around it: one
/// before unless the marker starts a line, one after unless what follows is
/// punctuation or the end. Runs of spaces on either side collapse to that one.
/// The caret ends up after the marker — and after its trailing space, when
/// there is one, so typing on continues the sentence.
enum PromptMarkerInsert {
    struct Result: Equatable {
        var text: String
        /// Where the caret goes afterwards, in Characters from the start.
        var cursor: Int
    }

    /// Punctuation that hugs the word before it: no space goes between the
    /// marker and one of these.
    private static let hugging: Set<Character> = [",", ".", ";", ":", "!", "?", ")", "]", "}"]

    /// Horizontal whitespace only. A line break is a boundary to keep, never a
    /// gap to close.
    private static func isGap(_ c: Character) -> Bool { c == " " || c == "\t" }

    /// `range` is what the marker replaces — a caret is an empty range. Both
    /// ends are clamped to the text, so a stale selection cannot crash this.
    static func insert(_ marker: String, into text: String, replacing range: Range<Int>) -> Result {
        let chars = Array(text)
        let lo = max(0, min(range.lowerBound, chars.count))
        let hi = max(lo, min(range.upperBound, chars.count))
        var before = chars[..<lo]
        var after = chars[hi...]
        while let c = before.last, isGap(c) { before.removeLast() }
        while let c = after.first, isGap(c) { after.removeFirst() }

        let spaceBefore = !before.isEmpty && before.last != "\n"
        let spaceAfter: Bool
        if let next = after.first {
            spaceAfter = next != "\n" && !hugging.contains(next)
        } else {
            spaceAfter = false
        }

        var out = String(before)
        if spaceBefore { out += " " }
        out += marker
        if spaceAfter { out += " " }
        let cursor = out.count
        out += String(after)
        return Result(text: out, cursor: cursor)
    }

    /// The unfocused case: on the end, one space away.
    static func append(_ marker: String, to text: String) -> Result {
        let n = text.count
        return insert(marker, into: text, replacing: n..<n)
    }
}
