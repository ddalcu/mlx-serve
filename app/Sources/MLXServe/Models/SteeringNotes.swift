import Foundation

/// A note typed while a chat's turn runs, one per session, handed to the
/// agent as the next user message. Never persisted.
struct SteeringNotes: Equatable {
    private var notes: [UUID: String] = [:]

    func note(for session: UUID) -> String? { notes[session] }

    /// Adds the text after what is already there; blank text changes nothing.
    mutating func append(_ text: String, for session: UUID) {
        let merged = Self.joined(notes[session] ?? "", text)
        if merged.isEmpty { notes.removeValue(forKey: session) } else { notes[session] = merged }
    }

    mutating func clear(for session: UUID) {
        notes.removeValue(forKey: session)
    }

    /// The note to send now, removed from the store.
    mutating func take(for session: UUID) -> String? {
        notes.removeValue(forKey: session)
    }

    /// Exactly one blank line between the trimmed texts; an empty side
    /// yields the other.
    static func joined(_ first: String, _ second: String) -> String {
        let a = first.trimmingCharacters(in: .whitespacesAndNewlines)
        let b = second.trimmingCharacters(in: .whitespacesAndNewlines)
        if a.isEmpty { return b }
        if b.isEmpty { return a }
        return a + "\n\n" + b
    }
}
