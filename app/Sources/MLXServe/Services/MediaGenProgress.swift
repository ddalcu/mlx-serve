import Foundation

/// Live state of an in-chat media generation.
///
/// These block chat decode on the one GPU, so a chat window that sits there with
/// no feedback for two minutes reads as a hang rather than as work. A single
/// value is enough: the inference thread serializes media generation, so there
/// is never more than one running.
struct MediaGenProgress: Equatable {
    var kind: MediaKind
    var step: Int
    var total: Int
    /// What the engine says it is doing right now ("Loading model", "Composing").
    var message: String
    var startedAt: Date

    var title: String { kind.progressTitle }

    /// Bar position, or nil for an INDETERMINATE bar. Nil is the honest answer
    /// whenever the server reports no total — TTS length is model-determined, so
    /// it sends a growing frame count against total 0, and a determinate bar
    /// drawn at 0/0 sits empty and looks stuck.
    var fraction: Double? {
        guard total > 0 else { return nil }
        return min(1.0, max(0.0, Double(step) / Double(total)))
    }

    /// One line under the title. The step count only appears when there are
    /// steps to count.
    var detailText: String {
        guard total > 0 else { return message }
        return "\(message) — step \(step) of \(total)"
    }

    /// `m:ss` since the generation started. Never negative: a startedAt in the
    /// future (clock change mid-run) reads 0:00 rather than "-0:03".
    func elapsedText(now: Date = Date()) -> String {
        let seconds = max(0, Int(now.timeIntervalSince(startedAt)))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

/// Failures the awaitable agent entry points surface. Shared by all four
/// services so the chat sees one taxonomy no matter which modality ran.
enum MediaGenError: LocalizedError {
    case emptyInput(String)
    case notDownloaded(String)
    case server(String)

    var errorDescription: String? {
        switch self {
        case .emptyInput(let what):  return "\(what) is empty."
        case .notDownloaded(let n):  return "\(n) is not downloaded."
        case .server(let m):         return m
        }
    }
}

/// The media-generation SSE shape, in one place.
///
/// All four backends stream the same `{type: "progress" | "complete" | "error"}`
/// envelope, and all four services used to parse it separately — four copies
/// that could each drift on a missing field. The `complete` payload stays with
/// the caller (each modality decodes something different out of it); what is
/// shared is deciding WHICH event this is.
enum MediaSSE {
    enum Event: Equatable {
        case progress(step: Int, total: Int, stage: String)
        case complete
        case failed(String)
        case ignored
    }

    static func classify(_ ev: [String: Any]) -> Event {
        switch ev["type"] as? String {
        case "progress":
            return .progress(step: ev["step"] as? Int ?? 0,
                             total: ev["total"] as? Int ?? 0,
                             stage: ev["stage"] as? String ?? "Generating")
        case "complete":
            return .complete
        case "error":
            // A typed error with no message is still an error — never silently
            // ignored, or the stream just ends and the caller reports "no data".
            return .failed(ev["message"] as? String ?? "Generation failed.")
        default:
            return .ignored
        }
    }

    /// JPEG bytes from an opt-in video `progress` event. The key is `preview`
    /// (`preview.formatProgressJson` is the only emitter). Missing or
    /// undecodable → nil, never a thrown error.
    static func previewJPEG(_ ev: [String: Any]) -> Data? {
        guard let b64 = ev["preview"] as? String else { return nil }

        return Data(base64Encoded: b64)
    }

    /// Readable label for the stage names the engines emit. An unknown stage is
    /// passed through verbatim — inventing a translation would mislabel it.
    static func stageLabel(_ stage: String) -> String {
        switch stage {
        case "encode":  return "Encoding prompt"
        case "prefill": return "Encoding prompt"
        case "frames":  return "Composing frames"
        case "diffuse": return "Composing"
        case "decode":  return "Rendering audio"
        default:        return stage
        }
    }
}
