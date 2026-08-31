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
        out += chatSection(messages: messages)
        return out
    }

    /// Several chats in one file: ONE exported-at line up top, then each
    /// conversation under its own heading. This is the multi-selection export's
    /// serializer — the single-chat path above is the N=1 case.
    static func markdown(sessions: [(title: String, messages: [ChatMessage])],
                         dateText: String) -> String {
        var out = "# mlx-serve export\n\n_Exported from mlx-serve on \(dateText). " +
                  "\(sessions.count) conversation\(sessions.count == 1 ? "" : "s")._\n\n"
        for session in sessions {
            out += "---\n\n# \(session.title)\n\n"
            out += chatSection(messages: session.messages)
        }
        return out
    }

    /// The labelled transcript body shared by both markdown shapes.
    private static func chatSection(messages: [ChatMessage]) -> String {
        var out = ""
        var first = true
        for message in messages where isExportable(message) {
            if !first { out += "---\n\n" }
            first = false
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

    // MARK: - JSON (re-importable)

    /// The re-import shape: every visible row with role, content, timestamp and
    /// the reasoning scratchpad (a record that drops it cannot reproduce the
    /// turn). Machine rows stay out, same rule as markdown.
    static func jsonData(title: String, messages: [ChatMessage], dateText: String) -> Data? {
        var rows: [[String: Any]] = []
        for message in messages where isExportable(message) {
            var row: [String: Any] = [
                "role": message.role.rawValue,
                "content": message.content,
                "timestamp": Self.exportTimestamp.string(from: message.timestamp),
            ]
            if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                row["reasoning"] = reasoning
            }
            if let images = message.images, !images.isEmpty { row["images"] = images.count }
            if let clips = message.audio, !clips.isEmpty { row["audio"] = clips.count }
            if let media = message.media, !media.isEmpty {
                row["media"] = media.map { ["kind": $0.kind.rawValue, "path": $0.path] }
            }
            rows.append(row)
        }
        let doc: [String: Any] = [
            "title": title,
            "exportedAt": dateText,
            "application": "mlx-serve",
            "messages": rows,
        ]
        return try? JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
    }

    private static let exportTimestamp: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static func isExportable(_ message: ChatMessage) -> Bool {
        guard !message.failedRetry, message.errorNotice == nil else { return false }
        // Tool-result rows are hidden in the transcript; they stay hidden here.
        return message.toolCallId == nil
    }
}
