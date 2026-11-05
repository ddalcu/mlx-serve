import Foundation

/// Shared async mechanism. No transcript mutation until every summary chunk
/// finishes; callers commit the returned checkpoint after cancellation checks.
enum ContextCompactionRunner {
    static func countingInput(_ messages: [[String: Any]]) -> (messages: [[String: Any]], mediaTokens: Int) {
        var mediaTokens = 0
        let sanitized = messages.map { message -> [String: Any] in
            var message = message
            if let blocks = message["content"] as? [[String: Any]] {
                message["content"] = blocks.map { block -> [String: Any] in
                    switch block["type"] as? String {
                    case "image_url":
                        mediaTokens += 8192
                        return ["type": "text", "text": "[image]"]
                    case "video_url":
                        let frames = (block["video_url"] as? [String: Any])?["frames"] as? [String] ?? []
                        mediaTokens += max(1, frames.count) * 8192
                        return ["type": "text", "text": "[video]"]
                    case "input_audio":
                        let encoded = (block["input_audio"] as? [String: Any])?["data"] as? String ?? ""
                        mediaTokens += max(1024, encoded.utf8.count / 300)
                        return ["type": "text", "text": "[audio]"]
                    default: return block
                    }
                }
            }
            return message
        }
        return (sanitized, mediaTokens)
    }

    static func measuredTokens(api: APIClient, port: UInt16, messages: [[String: Any]], tools: String?) async throws -> Int {
        let input = countingInput(messages)
        let data = try JSONSerialization.data(withJSONObject: input.messages, options: [.sortedKeys, .withoutEscapingSlashes])
        let text = (String(data: data, encoding: .utf8) ?? "") + (tools ?? "")
        let count = try await api.contextTokenCount(port: port, text: text)
        // JSON syntax is generally more expensive than template delimiters,
        // but this is still an estimate; reserve additional per-message space.
        return count + messages.count * 16 + 1024 + input.mediaTokens
    }

    static func summarize(source: [ChatMessage], boundary: Int, api: APIClient,
                          port: UInt16, model: String?, context: Int) async throws -> ChatMessage {
        let raw = ContextCompaction.transcript(source)
        let coveredIDs = Set(raw.prefix(boundary + 1).map(\.id))
        // Reuse an earlier valid checkpoint, but include only the newly covered
        // tail. The synthesized summary has a fresh ID and belongs in the input.
        let rawIDs = Set(raw.map(\.id))
        let prefix = ContextCompaction.activeMessages(source).filter {
            coveredIDs.contains($0.id) || !rawIDs.contains($0.id)
        }
        let text = ContextCompaction.summaryText(prefix)
        // UTF-8 bytes bound tokenizer input conservatively, even for code and
        // Chinese. Split on Character boundaries; never silently truncate data.
        let byteLimit = min(160_000, max(1024, context / 3))
        var chunks: [String] = [], chunk = "", bytes = 0
        for character in text {
            let part = String(character)
            if bytes + part.utf8.count > byteLimit, !chunk.isEmpty {
                chunks.append(chunk); chunk = ""; bytes = 0
            }
            chunk += part; bytes += part.utf8.count
        }
        if !chunk.isEmpty { chunks.append(chunk) }
        var summary = ""
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let messages: [[String: Any]] = [
                ["role": "system", "content": ContextCompaction.summaryInstruction],
                ["role": "user", "content": "Prior handoff:\n\(summary)\n\nConversation data chunk \(index + 1)/\(chunks.count):\n\(chunk)"]
            ]
            var next = "", cut = false, finished = false
            for try await event in api.streamChat(port: port, messages: messages,
                maxTokens: min(4096, context / 4), temperature: 0,
                enableThinking: false, modelId: model) {
                try Task.checkCancellation()
                switch event {
                case .content(let delta): next += delta
                case .truncated: cut = true
                case .done: finished = true
                default: break
                }
            }
            try Task.checkCancellation()
            if cut { throw ContextCompaction.Failure.truncatedSummary }
            guard finished, !next.contains("[Server Error:") else { throw ContextCompaction.Failure.incompleteSummary }
            summary = next.trimmingCharacters(in: .whitespacesAndNewlines)
            if summary.isEmpty { throw ContextCompaction.Failure.emptySummary }
        }
        return ContextCompaction.checkpoint(summary: summary, source: source, boundary: boundary)
    }
}
