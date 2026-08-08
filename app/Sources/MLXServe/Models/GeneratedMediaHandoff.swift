import Foundation

/// Moving a Create-pane result into a conversation.
///
/// There are two ways to make an asset in this app and they are deliberately
/// different things:
///
/// * **Create** is the workshop. You hold the controls — model, steps, size,
///   references, LoRA — it runs at full quality, and you iterate: ten renders to
///   get one. Results pile up in the pane's `recent` strip and on disk. Nothing
///   here is a conversation.
/// * **Chat** is asking someone to make it. You describe it in a sentence, the
///   model picks the tool and writes the prompt, and it runs at PREVIEW settings
///   (`MediaChatDefaults`) with one generation per turn — because it is sharing
///   the GPU with the reply you are waiting for. The result is part of the
///   thread, so "now make it winter" has something to refer to.
///
/// Whoever holds the controls is the whole distinction, and the consequences —
/// quality vs. context, iteration vs. conversation — follow from it.
///
/// This is the bridge in one direction: something you made in the workshop,
/// carried into a conversation so you can talk about it. It lands in a NEW chat
/// rather than whichever one happened to be open — a render appearing in the
/// middle of an unrelated thread is the surprise, and Create is iterative, so
/// "the open chat" is rarely the one you meant.
enum GeneratedMediaHandoff {

    /// Which transcript attachment a generator's output becomes. 3D is
    /// deliberately absent: `ChatMediaRef.Kind` has no mesh case and the
    /// transcript has no viewer for one, so a `.glb` stays in the pane that
    /// made it rather than becoming a row the chat can only fail to draw.
    static func kind(for experiment: GenExperiment) -> ChatMediaRef.Kind? {
        switch experiment {
        case .image:   return .image
        case .video:   return .video
        case .audio:   return .audio
        case .model3d: return nil
        }
    }

    /// The message that opens the new conversation.
    ///
    /// A USER message: the model in that thread did not make this and has never
    /// seen it, so filing it as assistant output would be the same class of lie
    /// as rendering an error as something the model said. It also leaves the
    /// turn to the user — the handoff starts a conversation, it doesn't take one.
    static func message(path: String, prompt: String,
                        kind: ChatMediaRef.Kind) -> ChatMessage {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        var message = ChatMessage(role: .user,
                                  content: trimmed.isEmpty ? fallbackTitle(kind) : trimmed)
        // The ref's caption keeps the REAL prompt, empty or not: the stand-in
        // above is ours, and echoing it into the caption would put words in the
        // user's mouth in a transcript they keep.
        message.media = [ChatMediaRef(kind: kind, path: path, prompt: trimmed)]
        return message
    }

    private static func fallbackTitle(_ kind: ChatMediaRef.Kind) -> String {
        switch kind {
        case .image: return "Generated image"
        case .video: return "Generated video"
        case .audio: return "Generated audio"
        }
    }
}
