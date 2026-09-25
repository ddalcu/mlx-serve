import Foundation

/// What each chat's in-flight turn is doing right now, at the granularity the
/// sidebar shows: the model is producing tokens, or the app is running the
/// tools the model asked for.
enum TurnPhase: Equatable {
    case generating
    case tool
}

/// Per-session turn phase plus the chats whose turn ended while nobody was
/// looking. In memory only; a fresh launch starts empty.
struct SidebarActivity: Equatable {
    /// How a turn ended: cleanly, or with an error card the user has to read.
    enum Outcome: Equatable {
        case finished
        case attention
    }

    /// The mark a sidebar row draws. The two live phases animate; `finished`
    /// and `attention` are the static "ended, not yet opened" marks.
    enum Dot: Equatable {
        case generating
        case tool
        case finished
        case attention
    }

    private var phases: [UUID: TurnPhase] = [:]
    private var endedUnseen: [UUID: Outcome] = [:]

    func phase(for session: UUID) -> TurnPhase? { phases[session] }

    /// Starting a turn drops the session's stale finished mark: the row is live
    /// again, and the grey dot will be re-earned when this turn ends.
    mutating func setPhase(_ phase: TurnPhase, for session: UUID) {
        phases[session] = phase
        endedUnseen.removeValue(forKey: session)
    }

    /// The turn is over. `seen` is whether the chat is on screen right now; an
    /// ended turn nobody watched earns the outcome's mark.
    mutating func end(for session: UUID, seen: Bool, outcome: Outcome = .finished) {
        phases.removeValue(forKey: session)
        if seen { endedUnseen.removeValue(forKey: session) } else { endedUnseen[session] = outcome }
    }

    mutating func markSeen(_ session: UUID) {
        endedUnseen.removeValue(forKey: session)
    }

    /// Sessions carrying an ended mark; the sidebar clears the selected ones
    /// through `markSeen` as soon as it draws them.
    var unseen: Set<UUID> { Set(endedUnseen.keys) }

    /// The dot for one row. A live phase always shows; an ended mark only on
    /// a row that is not selected.
    func dot(for session: UUID, isSelected: Bool) -> Dot? {
        switch phases[session] {
        case .generating: return .generating
        case .tool: return .tool
        case nil:
            guard !isSelected, let outcome = endedUnseen[session] else { return nil }
            return outcome == .attention ? .attention : .finished
        }
    }
}
