import Foundation

/// Copy and number formatting for Settings → Context size.
///
/// Extracted from the SwiftUI row because that row surfaces THREE different
/// token counts and it is easy to describe the wrong one:
///
///   * **Model max** — `max_position_embeddings` from `config.json` (262,144 on
///     Qwen3.6-27B). Architectural, never memory-aware.
///   * **GPU-safe max** — the largest context this Mac's free memory could hold
///     for this model right now (`/props` `maxSafeContext`, ~93K).
///   * **In use** — what the running server actually pinned and enforces
///     (`/v1/models` `meta.context_length`, 78,848 = 85% of the GPU-safe max).
///     This is the number agent CLIs are told about, so it must be visible.
///
/// "Auto" resolves to the third of those, NOT the first — the shipped help text
/// claimed otherwise.
enum ContextSizeDisplay {

    /// "Auto" for 0; otherwise a 1024-based K/M abbreviation.
    static func formatTokens(_ n: Int) -> String {
        if n == 0 { return "Auto" }
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    /// Body copy under the slider. Shared with `ServerOptions.serverFlagFields`
    /// so the two descriptions of "Auto" cannot drift.
    /// One literal, not a `+` chain: a catalog key is the whole finished
    /// sentence, and a translator cannot be handed three fragments whose word
    /// order Chinese does not keep.
    static let helpText = String(localized:
        "Maximum prompt + completion tokens. \"Auto\" fits the model to available memory when it loads — well under the model max — and holds that value until the server restarts. Higher values use more memory.")

    /// The context the RUNNING server settled on, formatted for the pill.
    /// nil until the server has reported a loaded model.
    static func inUseValue(contextLength: Int?) -> String? {
        guard let contextLength, contextLength > 0 else { return nil }
        return formatTokens(contextLength)
    }
}
