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
    enum Cause: String, Codable, Equatable {
        case maxTokens
        case repetitionLoop
    }

    /// The notice as DATA on a message (`ChatMessage.truncationNotice`): the
    /// transcript renders it as a footnote under the bubble, and because it
    /// never touches `content`, the history builders cannot send it back to
    /// the model — the error-echo class, closed structurally (the old
    /// append-into-content banner rode back as assistant prose every turn).
    struct Notice: Codable, Equatable {
        let cause: Cause
        let maxTokens: Int

        /// Footnote text, ⚠️ included; plain (rendered as a styled view).
        var text: String { "⚠️ " + TruncationNotice.footnote(cause: cause, maxTokens: maxTokens) }
    }

    /// The user-facing sentence. The max-tokens text names the cap that was
    /// hit and the two ways out; the loop text must NOT mention a cap, because
    /// naming one sends people to raise a setting that was never the problem
    /// (same rule as `ChatErrorNotice` keeping a diagnosis it can't support out
    /// of the card).
    static func footnote(cause: Cause, maxTokens: Int) -> String {
        switch cause {
        case .maxTokens:
            return "Output truncated — max tokens (\(maxTokens)) reached. Try breaking the task into smaller steps, or raise “max tokens” in Settings."
        case .repetitionLoop:
            return "Stopped — the model started repeating itself and the server cut the reply. Try rephrasing, or ask for a smaller piece of the task."
        }
    }

    /// The LEGACY in-content banner shape — builds before 2026-08-11 appended
    /// this to `message.content`. Kept as the one source of truth for
    /// `stripped(from:)`, which scrubs it out of saved sessions at history
    /// build time.
    static func text(cause: Cause, maxTokens: Int) -> String {
        "\n\n⚠️ *\(footnote(cause: cause, maxTokens: maxTokens))*"
    }

    /// Removes a legacy banner an older build appended INTO content, so
    /// history rebuilt from sessions saved before the notice became data
    /// stops teaching the model the warning text. The banner was always
    /// appended at the end of the turn, so everything from its marker on is
    /// cut. Markers derive from `text(...)` — they cannot drift from it.
    static func stripped(from content: String) -> String {
        let loopMarker = text(cause: .repetitionLoop, maxTokens: 0)   // no cap interpolated — fixed string
        let capFull = text(cause: .maxTokens, maxTokens: 0)
        let capMarker = String(capFull[..<capFull.range(of: "(0")!.lowerBound])
        for marker in [loopMarker, capMarker] {
            if let r = content.range(of: marker) {
                return String(content[..<r.lowerBound])
            }
        }
        return content
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
