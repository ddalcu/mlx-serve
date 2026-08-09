import Foundation

/// The composer's "make me one of these" mode.
enum ChatCreateMode: String, CaseIterable, Identifiable {
    case image, video, audio

    var id: String { rawValue }

    /// Decode a persisted value. Unknown (or absent) ⇒ no mode, never a throw.
    static func from(_ raw: String?) -> ChatCreateMode? {
        guard let raw else { return nil }
        return ChatCreateMode(rawValue: raw)
    }

    /// The generator this mode drives — the same catalogue the tray tiles, the
    /// discovery chips and the Create pages iterate, so a mode can never point
    /// at a generator that isn't offered.
    var experiment: GenExperiment {
        switch self {
        case .image: return .image
        case .video: return .video
        case .audio: return .audio
        }
    }

    /// Which progress card the transcript draws while this mode runs.
    var progressKind: MediaKind {
        switch self {
        case .image: return .image
        case .video: return .video
        case .audio: return .speech
        }
    }

    /// Shown in the composer's mode banner.
    var title: String {
        switch self {
        case .image: return "Creating images"
        case .video: return "Creating video"
        case .audio: return "Creating speech"
        }
    }

    /// What the composer's placeholder asks for. It has to say that the text is
    /// NOT going to the model — a prompt field that looks like chat but isn't is
    /// the one thing this mode could get badly wrong.
    var placeholder: String {
        switch self {
        case .image: return "Describe the image to generate…"
        case .video: return "Describe the video to generate…"
        case .audio: return "Type the words to speak…"
        }
    }

    /// The send button's verb — "Send" would be a lie: nothing is sent to anyone.
    var actionVerb: String { "Generate" }

    /// Whether an attached image is a SOURCE for this mode (image editing /
    /// image-to-video), rather than something to talk about.
    var usesSourceImage: Bool {
        switch self {
        case .image, .video: return true
        case .audio:         return false
        }
    }

}

/// A create-mode prompt typed before its model was on disk. Held (never
/// dropped) while the download runs, then generated — see `ChatCreateSend`.
/// The attachments ride along: giving only the words back on Cancel, or
/// generating without the source photo, is half the message lost.
struct HeldCreatePrompt: Equatable {
    let prompt: String
    let sourceImages: [ChatImage]?
    /// True once the transfer is under way, so the offer row can stop
    /// offering a second Download press.
    var downloading: Bool = false
}

/// What pressing Generate does, given whether the chosen model is on disk.
enum ChatCreateSend: Equatable {
    /// Ready — run it now.
    case generate
    /// The model isn't on disk. Offer the download; the prompt is held, not
    /// thrown away, and runs when the transfer finishes.
    case offerDownload
    /// Nothing to do (empty prompt, or a generation already running).
    case ignore

    static func decide(prompt: String, modelReady: Bool, busy: Bool) -> ChatCreateSend {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, !busy else {
            return .ignore
        }
        return modelReady ? .generate : .offerDownload
    }
}
