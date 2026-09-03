import Foundation

/// ⌘1…⌘9 over the sidebar's conversation list: which row wears which number,
/// and which chat a digit reaches.
///
/// The ordering is DERIVED from `SidebarSessionGroups.split`, never rebuilt
/// here. The sidebar draws agent threads above plain chats, so numbering the
/// raw `visibleChatSessions` would put ⌘1 on whichever thread is newest — a
/// badge that names a shortcut landing somewhere else. One ordering, and a
/// section added to the sidebar carries the numbers with it.
///
/// The nine numbers go to rows you can SEE. The conversation list scrolls under
/// the pinned destination block, so a scrolled sidebar had ⌘1…⌘5 sitting behind
/// frosted glass — five of nine shortcuts spent on rows the user cannot read,
/// while the rows in front of them started at 6. `numbering` is that filter:
/// the caller passes the rows currently in the clear and the numbers land on
/// them in the same top-to-bottom order.
/// Where one conversation row sits vertically, measured in the sidebar
/// column's own coordinate space.
struct SidebarRowSpan: Equatable {
    let top: CGFloat
    let bottom: CGFloat
}

enum ChatQuickSwitch {

    /// The keyboard has nine of these. A tenth row draws no badge rather than
    /// one promising a shortcut nothing can press.
    static let maxSlots = 9

    /// Which rows are in the clear, from the measurements alone.
    ///
    /// The band runs from the bottom of the pinned destination block (rows
    /// scroll UNDER it) to the bottom of the column, so its height is simply
    /// how much window there is — a taller window numbers more rows, with no
    /// count written down anywhere.
    ///
    /// - Returns: nil for "nothing has been measured yet", which numbers from
    ///   the top; never an empty set for that case, because an empty set means
    ///   the opposite (see `numbered`).
    static func numbering(rowSpans: [UUID: SidebarRowSpan],
                          clearBandTop: CGFloat,
                          clearBandBottom: CGFloat) -> Set<UUID>? {
        guard !rowSpans.isEmpty, clearBandBottom > clearBandTop else { return nil }
        return Set(rowSpans.filter {
            // Fully inside the band. A half-clipped row wearing a number reads
            // as an answer to "which one is 3?" that you then have to scroll to
            // check — and the half point of tolerance is because layout
            // arithmetic lands fractionally off, not because the edge is
            // approximate.
            $0.value.top >= clearBandTop - 0.5 && $0.value.bottom <= clearBandBottom + 0.5
        }.keys)
    }

    /// The conversation rows top to bottom, exactly as the sidebar draws them.
    static func ordered(_ sessions: [ChatSession]) -> [ChatSession] {
        let groups = SidebarSessionGroups.split(sessions)
        return groups.agents + groups.chats
    }

    /// The rows that get a number, top to bottom.
    ///
    /// - Parameter numbering: the rows eligible for a number — those clear of
    ///   the sidebar's frosted overlay. `nil` means "no visibility information",
    ///   which numbers everything: that is the honest answer before the first
    ///   layout has reported, and it is what a caller with no overlay wants.
    ///   An EMPTY set is different and means what it says — nothing is in the
    ///   clear, so nothing is numbered.
    static func numbered(_ sessions: [ChatSession], numbering: Set<UUID>? = nil) -> [ChatSession] {
        let rows = ordered(sessions)
        let ids = numbered(visible: rows.map(\.id), numbering: numbering)
        return ids.compactMap { id in rows.first { $0.id == id } }
    }

    /// The general form: `visible` is the panel's rows top to bottom — chats
    /// AND terminals, in their dragged order — which is the only list the
    /// numbers can honestly follow.
    static func numbered(visible: [UUID], numbering: Set<UUID>? = nil) -> [UUID] {
        guard let numbering else { return Array(visible.prefix(maxSlots)) }
        return Array(visible.filter { numbering.contains($0) }.prefix(maxSlots))
    }

    /// The 1-based number drawn on a row, or nil when it gets none.
    static func slot(for id: UUID, in sessions: [ChatSession],
                     numbering: Set<UUID>? = nil) -> Int? {
        slot(for: id, visible: ordered(sessions).map(\.id), numbering: numbering)
    }

    static func slot(for id: UUID, visible: [UUID], numbering: Set<UUID>? = nil) -> Int? {
        guard let index = numbered(visible: visible, numbering: numbering).firstIndex(of: id)
        else { return nil }
        return index + 1
    }

    /// The row a digit reaches, or nil when no row wears that number.
    ///
    /// Reads the same `numbered` list the badge was drawn from, so the digit
    /// cannot land on a different row than the one showing it.
    static func id(forSlot slot: Int, in sessions: [ChatSession],
                   numbering: Set<UUID>? = nil) -> UUID? {
        id(forSlot: slot, visible: ordered(sessions).map(\.id), numbering: numbering)
    }

    static func id(forSlot slot: Int, visible: [UUID], numbering: Set<UUID>? = nil) -> UUID? {
        guard slot >= 1, slot <= maxSlots else { return nil }
        let rows = numbered(visible: visible, numbering: numbering)
        guard slot <= rows.count else { return nil }
        return rows[slot - 1]
    }

    /// Where a digit lands: a conversation (with the selection maths) or a
    /// terminal (shown as is — there is nothing to range on one).
    enum Target {
        case chat(SidebarMultiSelect.Outcome)
        case terminal(UUID)
    }

    /// `visible` is the whole panel; `chats` is the conversation subset in the
    /// same order, which is what a range runs over.
    static func target(slot: Int,
                       visible: [UUID],
                       chats: [UUID],
                       numbering: Set<UUID>?,
                       selection: Set<UUID>,
                       anchor: UUID?,
                       active: UUID?,
                       extend: Bool) -> Target? {
        guard let hit = id(forSlot: slot, visible: visible, numbering: numbering) else { return nil }
        guard chats.contains(hit) else { return .terminal(hit) }
        return .chat(SidebarMultiSelect.click(
            hit, ordered: chats, selection: selection, anchor: anchor ?? active,
            active: active, command: false, shift: extend))
    }

    /// What ⌘\<digit\> and ⇧⌘\<digit\> do, as one decision.
    ///
    /// TWO lists are in play and keeping them apart is the point of this
    /// function existing. The digit resolves against the NUMBERED rows — the
    /// ones in the clear, wearing badges. The range then runs over the FULL
    /// list, so ⇧⌘ selects every conversation between the two ends including
    /// the ones scrolled under the frost that never wore a number: a selection
    /// is a contiguous run of the list, not of the badges.
    ///
    /// Ranging is `SidebarMultiSelect.click(shift:)` verbatim rather than a
    /// second range rule, so the keyboard and a shift-click cannot disagree —
    /// including the part where the anchor stays put and repeated ranges
    /// re-range from one origin. `active` is the fallback origin: before
    /// anything has been clicked, "from the current conversation" is what the
    /// user means.
    ///
    /// - Returns: nil when no row wears that number — the shortcut does
    ///   nothing rather than guessing at a neighbour.
    static func outcome(slot: Int,
                        sessions: [ChatSession],
                        numbering: Set<UUID>?,
                        selection: Set<UUID>,
                        anchor: UUID?,
                        active: UUID?,
                        extend: Bool) -> SidebarMultiSelect.Outcome? {
        // Never `command` (inside `target`): ⌘ is physically down for every one
        // of these, but the gesture being described is a plain click or a
        // shift-click. Passing it through would make ⌘\<digit\> TOGGLE the row
        // into a multi-selection instead of going to it.
        let chats = ordered(sessions).map(\.id)
        guard case .chat(let outcome)? = target(slot: slot, visible: chats, chats: chats,
                                                numbering: numbering, selection: selection,
                                                anchor: anchor, active: active, extend: extend)
        else { return nil }
        return outcome
    }
}

/// Whether the reply at the end of a transcript can be handed back to the
/// model to finish.
///
/// A continuation streams into the LAST message, so every refusal here is a
/// message that would be the wrong target — caught as a disabled button rather
/// than discovered as text landing in the wrong bubble.
enum ContinueReply {
    /// - Parameter engine: which backend is serving the model. The embedded
    ///   ds4 engine renders its chat template INSIDE the engine, where there is
    ///   nowhere to append a prefill, so the server refuses a continuation
    ///   there by name (`continuationRejectReason`). Offering the button anyway
    ///   spends a click to earn a 400 rendered as an error card — the
    ///   dead-control class, and the same rule as a locked composer disc: never
    ///   offer what the resolver will refuse. `nil` is "no model info yet",
    ///   which the `serverRunning` gate already covers.
    static func isEligible(_ messages: [ChatMessage],
                           serverRunning: Bool,
                           busy: Bool,
                           engine: ServerEngine? = nil) -> Bool {
        guard engine != .dsv4 else { return false }
        guard serverRunning, !busy, let last = messages.last else { return false }
        guard last.role == .assistant else { return false }
        // Already being written.
        guard !last.isStreaming else { return false }
        // Machinery and our own error cards are not the model's prose: a
        // tool-call summary has no sentence to finish, and continuing an error
        // notice asks the model to complete OUR sentence about a failure.
        guard !last.isAgentSummary, !last.failedRetry, last.errorNotice == nil else { return false }
        guard last.toolCalls == nil || last.toolCalls?.isEmpty == true else { return false }
        // The server trims a prefill of trailing whitespace and renders an
        // ordinary turn if nothing is left, so an empty reply would make the
        // button quietly do something other than what it says.
        return !last.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
