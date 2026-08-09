import Foundation

/// Moving a Create-pane result into a conversation. Image, video and audio
/// only — the 3D pane offers no "Send to Chat": `ChatMediaRef.Kind` has no
/// mesh case and the transcript has no viewer for one, so a `.glb` stays in
/// the pane that made it rather than becoming a row the chat can't draw.
enum GeneratedMediaHandoff {

    /// The message that opens the new conversation.
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
