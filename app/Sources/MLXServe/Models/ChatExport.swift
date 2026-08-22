import Foundation

/// Chat → Markdown export. The transcript as it READS: roles labelled, code
/// fences verbatim, generated media noted by filename — and none of the
/// machinery. Hidden tool results, failure cards and the model's reasoning
/// scratchpad stay out: an export is a record of the conversation, not of the
/// plumbing that produced it.
enum ChatExport {

    /// A filesystem-safe stem from the chat's title. Separators and whitespace
    /// both split into `-`-joined parts ("a/b: c?" → "a-b-c"), so nothing
    /// unreadable reaches Finder.
    static func suggestedFilename(title: String) -> String {
        let bad = CharacterSet(charactersIn: "/\\:?%*|\"<>").union(.whitespacesAndNewlines)
        let parts = title.unicodeScalars.split(whereSeparator: { bad.contains($0) })
        let name = parts.map(String.init).joined(separator: "-")
        return (name.isEmpty ? "chat" : name) + ".md"
    }

    static func markdown(title: String, messages: [ChatMessage], dateText: String) -> String {
        var out = "# \(title)\n\n"
        out += "_Exported from mlx-serve on \(dateText)._\n\n"
        for message in messages where isExportable(message) {
            out += "---\n\n"
            out += "**\(message.role == .user ? "You" : "Assistant")**\n\n"
            if let images = message.images, !images.isEmpty {
                out += "_Attached image\(images.count == 1 ? "" : "s")_\n\n"
            }
            if let clips = message.audio, !clips.isEmpty {
                out += "_Attached audio clip\(clips.count == 1 ? "" : "s")_\n\n"
            }
            if let media = message.media, !media.isEmpty {
                for ref in media {
                    out += "_Generated \(ref.kind.rawValue): \(ref.filename)_\n\n"
                }
            }
            let body = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            out += body.isEmpty ? "_(no text)_\n" : body + "\n"
            out += "\n"
        }
        return out
    }

    private static func isExportable(_ message: ChatMessage) -> Bool {
        guard !message.failedRetry, message.errorNotice == nil else { return false }
        // Tool-result rows are hidden in the transcript; they stay hidden here.
        return message.toolCallId == nil
    }
}
