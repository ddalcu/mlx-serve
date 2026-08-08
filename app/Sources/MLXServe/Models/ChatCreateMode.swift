import Foundation

/// The composer's "make me one of these" mode.
///
/// There were two ways to create an asset and they sat at opposite ends of the
/// app: the Create panes (you hold every control, full quality, no
/// conversation) and the chat tools (the model interprets your sentence and
/// decides, at preview settings). Clicking "Create Image" in the chat threw you
/// out of the chat and into a form.
///
/// This is the third shape, and it is the one the chip should have done all
/// along: the CHAT surface, driven directly. Your prompt goes straight to the
/// generator — no model reading it, no tool call, no turn — the composer's
/// existing attach button supplies a source image, the result lands in the
/// transcript, and the handful of settings that matter hide behind a
/// disclosure. Same window, same input, no form.
///
/// So the three ways now read as one spectrum of WHO IS DRIVING:
/// * **Chat, plain** — the model decides everything (it may not generate at all).
/// * **Chat, create mode** — you decide the prompt, the app decides the rest.
/// * **Create pane** — you decide everything.
///
/// Persisted per session as a RAW STRING (`ChatSession.createMode`), the same
/// tolerance the per-chat tool switches use: a mode retired in a later build
/// leaves an unknown name behind rather than failing the whole session's decode.
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

/// What pressing Generate does, given whether the chosen model is on disk.
///
/// The missing-model case is NOT a wall in front of the composer: you type the
/// prompt, press Generate, and the app then asks whether to fetch the model and
/// runs what you wrote as soon as it lands. Making someone go and download
/// something before they are allowed to describe what they want is the same
/// "answer a question you can't answer yet" problem the model picker had.
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
