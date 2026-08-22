import Foundation

/// Planning for the two "take it from here" actions: Regenerate and Edit &
/// Resend. Both rewind the transcript to a user turn and re-run it, so both
/// share one decision core — where the human's prompt actually is.
///
/// The subtlety is synthetic rows. A truncated tool-call round leaves a
/// `[System: …]` USER message behind (see `ChatTurnEngine.truncatedToolCallNudge`
/// and the malformed-tag nudge), and regenerating from one of those would send
/// the nudge back with no tool results in front of it — a prompt nobody typed.
enum ChatRewind {

    struct Plan: Equatable {
        /// The real user prompt to re-run.
        let userIdx: Int
        /// First index of the tail to drop (`userIdx + 1` when nothing follows).
        let removeFrom: Int
    }

    /// A user row the ENGINE appended mid-turn, never typed by a human. Both
    /// engine nudges open with this marker; keep new ones on the marker or add
    /// them here, or regenerate will rewind to them.
    static func isSyntheticNudge(_ text: String) -> Bool {
        text.hasPrefix("[System:")
    }

    /// The last row a human actually sent, skipping synthetic nudges and
    /// anything hidden from history.
    static func lastRealUserIndex(in messages: [ChatMessage]) -> Int? {
        for idx in messages.indices.reversed() {
            let m = messages[idx]
            guard m.role == .user, !m.failedRetry, !isSyntheticNudge(m.content) else { continue }
            return idx
        }
        return nil
    }

    /// What regenerate must do to this transcript: nil when there is no human
    /// prompt anywhere in it (nothing to re-run).
    static func regeneratePlan(in messages: [ChatMessage]) -> Plan? {
        guard let userIdx = lastRealUserIndex(in: messages) else { return nil }
        return Plan(userIdx: userIdx, removeFrom: min(userIdx + 1, messages.count))
    }
}
