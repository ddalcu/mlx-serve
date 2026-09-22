import Foundation

/// Notes typed while a chat's turn runs, one per session. A note is handed
/// to the agent at its next step boundary (after a tool round, or when the
/// turn ends) as the next user message, and fires once. Never persisted: it
/// is meant for the run in progress.
struct SteeringNotes: Equatable {
    private var notes: [UUID: String] = [:]

    func note(for session: UUID) -> String? { notes[session] }

    /// Stores the trimmed text, replacing any earlier note; blank text clears.
    mutating func set(_ text: String, for session: UUID) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { notes.removeValue(forKey: session) } else { notes[session] = trimmed }
    }

    mutating func clear(for session: UUID) {
        notes.removeValue(forKey: session)
    }

    /// The note to send now, removed from the store.
    mutating func take(for session: UUID) -> String? {
        notes.removeValue(forKey: session)
    }
}
