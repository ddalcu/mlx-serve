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
