import Foundation

/// Policy + text for the "output truncated" notice shown when a turn's reply is
/// cut short by the per-request `max_tokens` cap (`finish_reason: "length"`).
///
/// The notice is shown **at most once per turn**, only at the turn boundary, and
/// never on an intermediate agent iteration that silently retries a truncated
/// tool call. Appending it inside the per-iteration stream loop is what used to
/// stack duplicate banners on a multi-step agent turn — drive the decision
/// through `shouldShow(...)` at the loop's terminal exit instead.
enum TruncationNotice {
    /// The user-facing banner. Names the cap that was hit and points at the two
    /// ways out: shorter steps, or a higher cap in Settings.
    static func text(maxTokens: Int) -> String {
        "\n\n⚠️ *Output truncated — max tokens (\(maxTokens)) reached. Try breaking the task into smaller steps, or raise “max tokens” in Settings.*"
    }

    /// Whether to surface the notice now. True only when the turn is ending (no
    /// further tool calls / retries queued) AND the last response actually hit
    /// the cap AND we're not about to silently recover a truncated tool call.
    static func shouldShow(maxTokensHit: Bool, turnEnding: Bool, willRetry: Bool) -> Bool {
        maxTokensHit && turnEnding && !willRetry
    }
}
