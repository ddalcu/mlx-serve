import Foundation

/// A note typed while a chat's turn runs, one per session, handed to the
/// agent as the next user message. Never persisted.
struct SteeringNotes: Equatable {
    private var notes: [UUID: String] = [:]
    private var restoring: [UUID: String] = [:]

    func note(for session: UUID) -> String? { notes[session] }
    func restoringNote(for session: UUID) -> String? { restoring[session] }

    func composerNote(for session: UUID) -> String? {
        let text = Self.joined(restoring[session] ?? "", notes[session] ?? "")
        return text.isEmpty ? nil : text
    }

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

    /// Paused text belongs to the composer, even before the IME lets it adopt it.
    mutating func pause(for session: UUID) {
        guard let text = take(for: session) else { return }
        restoring[session] = Self.joined(restoring[session] ?? "", text)
    }

    mutating func restore(for session: UUID, into draft: String,
                          hasMarkedText: Bool) -> String? {
        guard !hasMarkedText, let text = restoring.removeValue(forKey: session) else { return nil }
        return Self.joined(text, draft)
    }

    mutating func retain(only sessions: Set<UUID>) {
        notes = notes.filter { sessions.contains($0.key) }
        restoring = restoring.filter { sessions.contains($0.key) }
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
