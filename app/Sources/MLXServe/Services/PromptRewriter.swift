import Foundation

/// Prompts for the "Enhance…" wand on the Image and Video panes (Music has its
/// own in `MusicPromptRewriter`): turns what the user typed into a prompt
/// shaped like the selected model's built-in examples. Pure — the sheet
/// streams the reply via `AgentComposer`.
enum PromptRewriter {

    struct Request: Equatable {
        let system: String
        let user: String
    }

    static func image(text: String, editing: Bool, groups: [ImagePromptExampleGroup]) -> Request {
        // An edit repertoire has up to eight groups: two each keeps the system prompt short.
        let examples = groups.flatMap { $0.examples.prefix(2) }.map(\.body)
        return editing
            ? request(noun: "image edit instruction",
                      shape: "Write ONE imperative instruction about the attached picture, like the examples: what to change and what must stay the same.",
                      examples: examples, text: text)
            : request(noun: "image prompt",
                      shape: "Write one or two sentences of plain prose like the examples: subject, setting, lighting, composition, medium. Quote any text that must appear in the picture.",
                      examples: examples, text: text)
    }

    static func video(text: String, format: VideoPromptFormat, seconds: Int) -> Request {
        let labels = H3PromptExamples.sections(for: format)
        let shape = labels.isEmpty
            ? "Write ONE paragraph of 4-8 sentences like the examples: subject, action, camera movement, lighting, setting, sound. Keep spoken dialogue in double quotes."
            : "Write the prompt in the exact labelled format of the examples, with these labels in order, each on its own line: \(labels.joined(separator: ", ")). Keep any <Picture N>, <Video N> or <Audio N> references verbatim."
        return request(noun: "video prompt", shape: shape,
                       examples: H3PromptExamples.examples(for: format).prefix(3).map(\.body), text: text,
                       note: "The clip is \(seconds) seconds long. Describe only what happens in that time: no more action than fits.")
    }

    private static func request(noun: String, shape: String, examples: [String], text: String, note: String = "") -> Request {
        Request(system: """
                You rewrite \(noun)s for a generative model. \(shape) \
                Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten \(noun), no preamble, no quotes, no markdown.

                Examples of the expected format:

                \(examples.joined(separator: "\n\n---\n\n"))
                """,
                user: "Rewrite this \(noun):\n\n\(text)" + (note.isEmpty ? "" : "\n\n\(note)"))
    }

    /// Model replies sometimes wear a fence or quotes; the editor gets the bare text.
    static func clean(_ reply: String) -> String {
        AgentWriter.stripFences(reply).trimmingCharacters(in: CharacterSet(charactersIn: "\"\u{201C}\u{201D}"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
