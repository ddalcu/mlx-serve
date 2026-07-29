import Foundation

/// A failed turn, in the shape the transcript renders it.
///
/// Errors used to land as `[Error: …]` inside the assistant's own text, which
/// made a server failure look like something the model said and buried the one
/// case the user can act on. Context overflow is that case, so it is classified
/// explicitly and carries the server's real counts; everything else stays
/// `.generic` and shows verbatim what the server reported. Guessing a specific
/// diagnosis from an unrecognized error would send people to change settings
/// that were never the problem.
struct ChatErrorNotice: Codable, Equatable {

    enum Kind: String, Codable {
        case contextOverflow
        case generic
    }

    var kind: Kind
    /// The underlying text, kept verbatim so a generic failure is still
    /// diagnosable and so the raw server wording is never lost.
    var message: String
    /// Tokens the rejected request needed, when the server reported it.
    var neededTokens: Int?
    /// The context window it was measured against.
    var contextLength: Int?

    // MARK: - Classification

    /// Phrases that mean "the prompt didn't fit". Ours is the first; the rest
    /// are wordings from OpenAI-compatible servers people point the app at.
    private static let overflowPhrases = [
        "exceeds maximum context length",
        "context length exceeded",
        "prompt too long",
        "maximum context",
    ]

    static func from(_ error: Error) -> ChatErrorNotice {
        guard case let APIError.badStatus(code, detail) = error else {
            return ChatErrorNotice(kind: .generic, message: error.localizedDescription,
                                   neededTokens: nil, contextLength: nil)
        }
        let trimmed = detail.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        guard overflowPhrases.contains(where: { lower.contains($0) }) else {
            return ChatErrorNotice(
                kind: .generic,
                message: trimmed.isEmpty ? "HTTP \(code) from mlx-serve" : "HTTP \(code): \(trimmed)",
                neededTokens: nil, contextLength: nil)
        }
        let counts = parseCounts(trimmed)
        return ChatErrorNotice(kind: .contextOverflow,
                               message: trimmed.isEmpty ? "Prompt exceeds maximum context length" : trimmed,
                               neededTokens: counts?.needed, contextLength: counts?.limit)
    }

    /// Pull the two figures out of the server's sentence. Missing numbers are
    /// not a parse failure — a server built before they were added sends the
    /// bare phrase, and the card is still worth showing without them.
    private static func parseCounts(_ text: String) -> (needed: Int, limit: Int)? {
        let pattern = #"maximum context length:\s*(\d+)\s*tokens requested,\s*(\d+)\s*available"#
        guard let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
              m.numberOfRanges == 3,
              let nr = Range(m.range(at: 1), in: text), let lr = Range(m.range(at: 2), in: text),
              let needed = Int(text[nr]), let limit = Int(text[lr])
        else { return nil }
        return (needed, limit)
    }

    // MARK: - Card copy

    var headline: String {
        switch kind {
        case .contextOverflow: "Model ran out of context size"
        case .generic: "Something went wrong"
        }
    }

    var detail: String {
        switch kind {
        case .contextOverflow:
            guard let neededTokens, let contextLength else {
                // Pre-counts server: say what happened without inventing figures.
                return "This request was larger than the model's context window."
            }
            return "This request needed \(Self.grouped(neededTokens)) tokens, "
                + "but the model's context window holds only \(Self.grouped(contextLength))."
        case .generic:
            return message
        }
    }

    /// Whether to offer "Increase Context Size". Only overflow — a button that
    /// can't fix the error it sits under is worse than no button.
    var offersContextAction: Bool { kind == .contextOverflow }

    /// Thousands separators, so 4108 reads at a glance next to 4096. Pinned to
    /// en_US rather than the user's locale because the sentence around it is an
    /// English literal — a comma-grouped number in a period-grouping locale is
    /// odd, but an English sentence with a German number is worse, and the copy
    /// is not localized.
    private static func grouped(_ n: Int) -> String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        f.usesGroupingSeparator = true
        f.groupingSeparator = ","
        f.groupingSize = 3
        f.locale = Locale(identifier: "en_US")
        return f.string(from: NSNumber(value: n)) ?? "\(n)"
    }
}
