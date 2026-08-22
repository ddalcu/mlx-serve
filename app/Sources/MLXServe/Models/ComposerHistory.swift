import Foundation

/// ↑ in an empty composer brings back what you last said, ↓ walks forward
/// again — the recall every shell and every chat app has, and the thing you
/// reach for the moment you send a message with a typo in it.
///
/// The whole design problem is that the arrow keys already mean something:
/// move the caret. Three rules keep both meanings, and each one exists because
/// the obvious version of this feature breaks something:
///
/// 1. The walk arms only from the EDGE of the text — ↑ with the caret at the
///    very start, ↓ at the very end. Inside a multi-line recalled message the
///    arrows go back to being arrows, which is the only way to edit line two
///    of what came back.
/// 2. It arms only on an EMPTY draft. ↑ at the start of a paragraph you are
///    writing must move the caret; replacing that with an old message would
///    destroy work, and there is no undo for a swallowed keystroke.
/// 3. It ends the instant you edit what came back. The field holds your text
///    now, so ↑ is a caret key again rather than something that discards it.
///
/// Distinct from Edit & Resend, which rewrites the message that is already in
/// the transcript and drops everything after it. Recall only fills the field:
/// what you send is a new turn, and the history is untouched.
enum ComposerHistory {

    /// Where a recall currently sits: an index into `entries`, or nil when the
    /// field holds the user's own draft rather than a recalled message.
    struct Walk: Equatable {
        var index: Int?
        static let idle = Walk(index: nil)
    }

    enum Direction { case up, down }

    enum Action: Equatable {
        /// Not ours — AppKit moves the caret, which is what the key is for.
        case pass
        case recall(text: String, walk: Walk)
    }

    /// The user's own messages, oldest first.
    ///
    /// Consecutive repeats collapse: sending the same thing twice (a retry)
    /// should cost one ↑ and not two, since the second press would look like it
    /// did nothing. Blank turns are dropped — an image-only message has no
    /// words to bring back, and the pictures are not re-attached either.
    static func entries(_ messages: [ChatMessage]) -> [String] {
        var out: [String] = []
        for message in messages where message.role == .user {
            let text = message.content
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
            guard out.last != text else { continue }
            out.append(text)
        }
        return out
    }

    static func up(draft: String, caretAtStart: Bool, walk: Walk, entries: [String]) -> Action {
        guard caretAtStart else { return .pass }
        if let index = current(walk, draft: draft, entries: entries) {
            // The end of the history is the end. Wrapping to the newest would
            // read as a different message arriving — the same reason the
            // revision pager refuses to wrap.
            guard index > 0 else { return .pass }
            return .recall(text: entries[index - 1], walk: Walk(index: index - 1))
        }
        guard draft.isEmpty, let newest = entries.indices.last else { return .pass }
        return .recall(text: entries[newest], walk: Walk(index: newest))
    }

    static func down(draft: String, caretAtEnd: Bool, walk: Walk, entries: [String]) -> Action {
        guard caretAtEnd, let index = current(walk, draft: draft, entries: entries) else { return .pass }
        if index + 1 < entries.count {
            return .recall(text: entries[index + 1], walk: Walk(index: index + 1))
        }
        // Past the newest is the empty composer the walk started from.
        return .recall(text: "", walk: .idle)
    }

    /// The entry the field is showing, or nil when this is not a live walk.
    ///
    /// Two ways it is not: the draft no longer matches (rule 3 — you edited
    /// it), or the index no longer names an entry. The second is not
    /// hypothetical: a walk survives a reply landing mid-conversation and a
    /// switch to another tab, and reading it out of bounds would trap.
    private static func current(_ walk: Walk, draft: String, entries: [String]) -> Int? {
        guard let index = walk.index, entries.indices.contains(index),
              entries[index] == draft else { return nil }
        return index
    }
}
