import Foundation

/// Policy + text for the "output truncated" notice shown when a turn's reply is
/// cut short — `finish_reason: "length"`, which the server uses for BOTH the
/// per-request `max_tokens` cap and its own degenerate-tail loop cut.
///
/// The notice is shown **at most once per turn**, only at the turn boundary, and
/// never on an intermediate agent iteration that silently retries a truncated
/// tool call. Appending it inside the per-iteration stream loop is what used to
/// stack duplicate banners on a multi-step agent turn — drive the decision
/// through `shouldShow(...)` at the loop's terminal exit instead.
enum TruncationNotice {
    /// WHY the reply was cut. "length" is the only OpenAI value for either, so
    /// the two are told apart by the server's sibling `finish_details` field —
    /// without it, a repetition loop reads as an output limit nobody set (live
    /// 2026-08-05: a pi session shown "maximum output token limit" with two
    /// thirds of its context free and no max_tokens on either side).
    enum Cause: Equatable {
        case maxTokens
        case repetitionLoop
    }

    /// The user-facing banner. The max-tokens text names the cap that was hit
    /// and the two ways out; the loop text must NOT mention a cap, because
    /// naming one sends people to raise a setting that was never the problem
    /// (same rule as `ChatErrorNotice` keeping a diagnosis it can't support out
    /// of the card).
    static func text(cause: Cause, maxTokens: Int) -> String {
        switch cause {
        case .maxTokens:
            return "\n\n⚠️ *Output truncated — max tokens (\(maxTokens)) reached. Try breaking the task into smaller steps, or raise “max tokens” in Settings.*"
        case .repetitionLoop:
            return "\n\n⚠️ *Stopped — the model started repeating itself and the server cut the reply. Try rephrasing, or ask for a smaller piece of the task.*"
        }
    }

    /// Whether to surface the notice now. True only when the turn is ending (no
    /// further tool calls / retries queued) AND the last response actually hit
    /// the cap AND we're not about to silently recover a truncated tool call.
    static func shouldShow(maxTokensHit: Bool, turnEnding: Bool, willRetry: Bool) -> Bool {
        maxTokensHit && turnEnding && !willRetry
    }

    /// Whether the agent loop may continue after this cut.
    ///
    /// A repetition loop must END the turn: the loop's content is already in
    /// the transcript (a streamed delta cannot be retracted, so the server's
    /// own trim never reaches us), the next round would send it back as
    /// history, and the model resumes its own loop reading it — five
    /// server-side cuts in a row, each firing sooner than the last, is what
    /// that looks like from the server. Continuing costs a full round to
    /// arrive back here faster. The error-echo class, with our own transcript
    /// as the error.
    ///
    /// A max_tokens cut is the opposite: the reply was going fine and simply
    /// ran out of room, so the existing recovery paths (chunk-and-retry,
    /// truncated-tool-call nudge) still apply.
    static func endsTurn(cause: Cause?) -> Bool {
        cause == .repetitionLoop
    }
}
