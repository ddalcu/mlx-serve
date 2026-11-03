import Foundation

/// Pure voice prerequisite checks and actionable notices; the recognizer supplies its live status through
/// `preflight()`.
enum VoicePreflight {
    /// A point-in-time read of the two things voice mode needs.
    struct Snapshot: Equatable {
        var micAuthorized: Bool
        /// Qwen3-ASR is on disk, whole (`VoiceASR.modelPath`): what turns speech into words.
        var asrReady: Bool
    }

    /// The single missing prerequisite to tell the user about: the mic first, then the speech model.
    enum Issue: Equatable {
        case microphoneDenied
        case asrMissing
    }

    /// Qwen3-ASR counts as ready where no words are asked for (the chat window, a model that hears the recording).
    static func asrReady(wantsWords: Bool, onDisk: Bool) -> Bool { !wantsWords || onDisk }

    static func firstIssue(_ s: Snapshot) -> Issue? {
        if !s.micAuthorized { return .microphoneDenied }
        if !s.asrReady { return .asrMissing }
        return nil
    }

    /// One-line status for the voice panel's status row.
    static func shortMessage(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied: return "Microphone access is off"
        case .asrMissing:       return "Voice needs its speech model"
        }
    }

    /// Full, actionable explanation for the notice card.
    static func detail(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied:
            return L10n.text("Voice mode can't hear you — Microphone access is off for MLX-Serve. " +
                "Turn it on in System Settings → Privacy & Security → Microphone, then enable Voice again.")
        case .asrMissing:
            return L10n.text("Voice mode turns what you say into words on this Mac with Qwen3-ASR (about 1 GB). " +
                "Download it, then enable Voice again.")
        }
    }

    /// `x-apple.systempreferences:` deep link to the pane that fixes the issue; nil when the fix is a download.
    static func settingsURLString(for issue: Issue) -> String? {
        switch issue {
        case .microphoneDenied: return "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        case .asrMissing:       return nil
        }
    }

    static func actionLabel(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied: return "Open Privacy Settings"
        case .asrMissing:       return "Download Qwen3-ASR"
        }
    }
}
