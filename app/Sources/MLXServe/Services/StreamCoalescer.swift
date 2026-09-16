import Foundation

/// Time-coalesces streamed token deltas so the chat UI is updated at a bounded
/// cadence instead of once per token.
///
/// **Why this exists — the tray-popover freeze.** The chat/agent engine writes
/// streamed tokens into `AppState.chatSessions`, which is `@Published`; every
/// write fires `AppState.objectWillChange`. While the `MenuBarExtra(.window)`
/// tray popover is open during the assistant's answer (the hands-free voice
/// case), that per-token churn re-renders the whole popover dozens-to-hundreds
/// of times a second. In this `LSUIElement` accessory app the popover lives in a
/// non-key panel, so a saturated main run loop starves SwiftUI `Button`
/// hit-testing — the voice **Stop** button goes dead while the AppKit `Menu`
/// (voice picker) and `Picker` (model) keep working from their own NSMenu
/// event-tracking loop. That's the same "tray locks up but the dropdown still
/// opens" wedge the static `VoiceTrayDot` fixed for animations — here the source
/// is data churn, not a `TimelineView`. Throttling the deltas to ~20 Hz leaves
/// the run loop idle between flushes, so events get dispatched and the buttons
/// respond again. Final content is byte-identical; only the *cadence* changes.
///
/// The clock is injected via `now` (seconds, any monotonic source) so the
/// batching is deterministic and unit-testable with no real time — see
/// `StreamCoalescerTests`.
struct StreamCoalescer {
    /// Max UI-update cadence. 50 ms (~20 Hz): fluid streaming text while leaving
    /// the main run loop mostly idle between flushes, so `Button` hit-testing in
    /// the tray popover isn't starved during a long answer.
    static let defaultInterval: TimeInterval = 0.05

    let interval: TimeInterval
    private var pendingContent = ""
    private var pendingReasoning = ""
    private var lastFlush: TimeInterval?

    init(interval: TimeInterval = StreamCoalescer.defaultInterval) {
        self.interval = interval
    }

    /// Accumulate a streamed delta. Returns the batched `(content, reasoning)` to
    /// apply right now when the flush interval has elapsed since the last flush —
    /// or on the very first delta, so text starts appearing immediately —
    /// otherwise `nil` (keep buffering).
    mutating func add(content: String = "", reasoning: String = "",
                      now: TimeInterval) -> (content: String, reasoning: String)? {
        pendingContent += content
        pendingReasoning += reasoning
        if let last = lastFlush, now - last < interval { return nil }
        lastFlush = now
        return take()
    }

    /// Flush whatever is still buffered — call at stream end and before any
    /// terminal/out-of-band event (tool calls, usage, truncation notice) so
    /// content stays ordered and complete for code that reads it back. Returns
    /// `nil` when nothing is pending.
    mutating func drain() -> (content: String, reasoning: String)? {
        guard !pendingContent.isEmpty || !pendingReasoning.isEmpty else { return nil }
        return take()
    }

    private mutating func take() -> (content: String, reasoning: String) {
        defer { pendingContent = ""; pendingReasoning = "" }
        return (pendingContent, pendingReasoning)
    }
}
