import Foundation

/// Folding a user turn that runs past the window.
///
/// A pasted log or file is one plain `Text` several screens tall that the
/// transcript lays out whole — so the fold is a `lineLimit`, which bounds
/// that layout, not a clip over it.
enum LongUserTurn {

    /// Lines kept while folded.
    static let collapsedLineLimit = 15

    /// Lines a turn must clear the limit by to earn a "show more". The count
    /// below is an estimate, and a control that reveals nothing is worse than
    /// a long bubble.
    static let foldMargin = 3

    /// Laid-out lines the text will take, counting the wrap of every hard line.
    static func estimatedLines(of text: String, charsPerLine: Int) -> Int {
        guard charsPerLine > 0 else { return 1 }
        return text.split(separator: "\n", omittingEmptySubsequences: false)
            .reduce(0) { $0 + max(1, ($1.count + charsPerLine - 1) / charsPerLine) }
    }

    static func isCollapsible(_ text: String, charsPerLine: Int,
                              limit: Int = collapsedLineLimit) -> Bool {
        estimatedLines(of: text, charsPerLine: charsPerLine) >= limit + foldMargin
    }

    /// Characters across one line of the bubble. `0.5` is the system font's
    /// rough average advance — the real one depends on the glyphs, which only a
    /// layout pass knows, and paying for that pass is what the fold avoids.
    static func charsPerLine(textWidth: CGFloat, fontSize: CGFloat) -> Int {
        guard textWidth > 0, fontSize > 0 else { return 1 }
        return max(1, Int(textWidth / (fontSize * 0.5)))
    }
}

/// Which turns the reader has unfolded, remembered beside the transcript.
///
/// Deliberately not observable: the fold is row state, so folding stays a
/// one-row redraw instead of re-evaluating every message in the conversation.
/// This only answers the question again after the transcript is rebuilt — a
/// density or text-size change gives it a new identity and discards every row —
/// and it is emptied when the conversation changes, because unfolding is an act
/// of reading that belongs to the visit.
@MainActor
final class FoldStore {
    private var expanded: Set<UUID> = []

    func isExpanded(_ id: UUID) -> Bool { expanded.contains(id) }

    func set(_ id: UUID, expanded value: Bool) {
        if value { expanded.insert(id) } else { expanded.remove(id) }
    }

    func clear() { expanded.removeAll() }
}
