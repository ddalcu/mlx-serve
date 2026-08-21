import Foundation

/// One generated version of an assistant reply.
///
/// Regenerating used to DELETE the reply it replaced, so a better first answer
/// was gone the moment you asked for a second opinion. A revision keeps each
/// version so the pager under the bubble can step back to it.
///
/// Reasoning rides along because it belongs to the version that produced it —
/// showing revision 2's answer under revision 1's thinking is worse than
/// showing no thinking at all.
struct MessageRevision: Codable, Equatable {
    var content: String
    var reasoningContent: String?

    init(content: String, reasoningContent: String? = nil) {
        self.content = content
        self.reasoningContent = reasoningContent
    }
}

/// The pure rules behind the revision pager: what the arrows do, what the
/// counter reads, and when it is shown at all.
///
/// A message with no revisions is the ordinary case and must stay exactly as
/// it was — one reply, no chrome. The list only becomes non-empty when a reply
/// is regenerated, which is also the moment the version being replaced has to
/// be captured, because `regenerate` truncates the transcript to the last user
/// message and the old reply goes with it.
enum MessageRevisions {

    /// Fewer than two versions is not a choice, so nothing is drawn.
    static func isPagerVisible(_ revisions: [MessageRevision]) -> Bool {
        revisions.count > 1
    }

    /// "2/3" — 1-based, because it is read by a person.
    static func label(index: Int, count: Int) -> String {
        guard count > 0 else { return "" }
        let clamped = min(max(index, 0), count - 1)
        return "\(clamped + 1)/\(count)"
    }

    static func canGoBack(index: Int) -> Bool { index > 0 }
    static func canGoForward(index: Int, count: Int) -> Bool { index < count - 1 }

    /// Step, refusing to wrap. Wrapping would make the two arrows do the same
    /// thing at the ends, and the counter jump from 3/3 to 1/3 reads as a
    /// different reply arriving.
    static func step(index: Int, by delta: Int, count: Int) -> Int {
        guard count > 0 else { return 0 }
        return min(max(index + delta, 0), count - 1)
    }

    /// The list a regeneration starts from: whatever the message already had,
    /// or — the first time — the reply about to be replaced, captured as
    /// version 1 so it is not lost.
    ///
    /// Empty prior content yields an empty list: a failed or empty reply is not
    /// a version worth stepping back to, and seeding one would show a pager
    /// whose first page is blank.
    static func seeding(prior: MessageRevision, existing: [MessageRevision]) -> [MessageRevision] {
        if !existing.isEmpty { return existing }
        return prior.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? [] : [prior]
    }

    /// Write an in-place edit into the version currently being read.
    ///
    /// Without this, editing a reply that has a pager loses the edit the moment
    /// you step away and back — stepping reloads `content` from the stored
    /// revision, which would still hold the text before the edit. An unpaged
    /// reply has nothing to sync and keeps its empty list.
    static func applyingEdit(_ text: String,
                             to revisions: [MessageRevision],
                             at index: Int) -> [MessageRevision] {
        guard index >= 0, index < revisions.count else { return revisions }
        var next = revisions
        next[index].content = text
        return next
    }

    /// Apply a regeneration's held seed and record the reply it produced, in
    /// one step — the whole transaction a finished turn performs.
    ///
    /// The seed is HELD from the moment the regeneration is asked for because
    /// the reply it belongs to does not exist yet. On the plain path that gap
    /// is one statement wide; on the agent path the streaming placeholder is
    /// appended from inside a Task, once per tool round, and the reply the
    /// pager belongs to is the LAST of them. Writing the seed at the start
    /// therefore landed it on the user's own message and the role guard
    /// silently dropped it, so with Tools on the pager never appeared at all.
    ///
    /// An `existing` list outranks the seed: that message is already on its
    /// third version, and the seed describes a reply two regenerations ago.
    ///
    /// A finished reply with no text is not a version, for the same reason
    /// `seeding` refuses an empty prior — a turn that failed before streaming
    /// anything would otherwise add a blank page to step onto. The seed is
    /// still installed, so the earlier reply is not lost with it.
    static func finishing(seed: [MessageRevision]?,
                          existing: [MessageRevision],
                          finished: MessageRevision) -> (revisions: [MessageRevision], index: Int) {
        let base = existing.isEmpty ? (seed ?? []) : existing
        guard !finished.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return (base, max(base.count - 1, 0))
        }
        return committing(finished, into: base)
    }

    /// Record a freshly finished reply as the newest version and select it.
    ///
    /// Only meaningful when a regeneration seeded the list — an ordinary first
    /// reply leaves `revisions` empty so nothing about the message changes.
    /// A repeat of the version already at the end is not appended: a
    /// deterministic model (temperature 0) would otherwise grow an unbounded
    /// list of identical pages.
    static func committing(_ finished: MessageRevision,
                           into revisions: [MessageRevision]) -> (revisions: [MessageRevision], index: Int) {
        guard !revisions.isEmpty else { return ([], 0) }
        if let last = revisions.last, last.content == finished.content {
            return (revisions, revisions.count - 1)
        }
        var next = revisions
        next.append(finished)
        return (next, next.count - 1)
    }
}
