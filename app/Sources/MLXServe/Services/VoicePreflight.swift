import Foundation

/// Voice mode's three prerequisites and the friendly, actionable notice shown
/// when one is missing — surfaced non-invasively as a card in the voice panel
/// (no blocking modal) the moment the user enables Voice. Pure → unit-testable
/// without the Speech/AVFoundation frameworks; the live status read lives in
/// `BaseSpeechRecognizer.preflight()`.
enum VoicePreflight {
    /// A point-in-time read of the three things voice mode needs.
    struct Snapshot: Equatable {
        var micAuthorized: Bool
        var speechAuthorized: Bool
        /// On-device dictation model installed for the locale (voice forces
        /// on-device recognition, so this is required — not just nice-to-have).
        var onDeviceAvailable: Bool
        var locale: String
    }

    /// The single missing prerequisite to tell the user about. Ordered by the
    /// pipeline: you need the mic, then permission to transcribe, then the model.
    enum Issue: Equatable {
        case microphoneDenied
        case speechDenied
        /// On-device dictation can't transcribe: the Dictation switch is OFF, or
        /// the language model isn't installed. Both are fixed the same way
        /// (Keyboard → Dictation), and they're indistinguishable to us —
        /// `supportsOnDeviceRecognition` stays true with the switch off — so we
        /// treat them as one issue.
        case dictationUnavailable(locale: String)
    }

    /// First blocking prerequisite from a pre-flight snapshot, or nil when voice
    /// looks good to go. (The Dictation-switch-off case can't be seen here — the
    /// snapshot's `onDeviceAvailable` only reflects the installed model — so it's
    /// caught at runtime instead; see `VoiceModeController`.)
    static func firstIssue(_ s: Snapshot) -> Issue? {
        if !s.micAuthorized { return .microphoneDenied }
        if !s.speechAuthorized { return .speechDenied }
        if !s.onDeviceAvailable { return .dictationUnavailable(locale: s.locale) }
        return nil
    }

    /// One-line status for the voice panel's status row.
    static func shortMessage(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied:     return "Microphone access is off"
        case .speechDenied:         return "Speech Recognition is off"
        case .dictationUnavailable: return "On-device dictation isn't available"
        }
    }

    /// Full, actionable explanation for the notice card.
    static func detail(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied:
            return "Voice mode can't hear you — Microphone access is off for MLX-Serve. " +
                "Turn it on in System Settings → Privacy & Security → Microphone, then enable Voice again."
        case .speechDenied:
            return "Voice mode needs Speech Recognition access (a separate permission from Microphone). " +
                "Turn it on in System Settings → Privacy & Security → Speech Recognition, then enable Voice again."
        case .dictationUnavailable(let locale):
            return "Voice mode transcribes on-device — your audio never leaves this Mac — but on-device " +
                "dictation isn't available for \(locale): it's either turned off or the language isn't " +
                "installed. Turn on System Settings → Keyboard → Dictation (and add your language), then " +
                "enable Voice again."
        }
    }

    /// `x-apple.systempreferences:` deep link to the pane that fixes the issue.
    static func settingsURLString(for issue: Issue) -> String {
        switch issue {
        case .microphoneDenied:
            return "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
        case .speechDenied:
            return "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition"
        case .dictationUnavailable:
            return "x-apple.systempreferences:com.apple.preference.keyboard?Dictation"
        }
    }

    static func actionLabel(for issue: Issue) -> String {
        switch issue {
        case .dictationUnavailable: return "Open Keyboard Settings"
        default:                    return "Open Privacy Settings"
        }
    }
}
