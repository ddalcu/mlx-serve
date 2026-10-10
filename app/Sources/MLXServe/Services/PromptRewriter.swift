import Foundation

/// Prompts for the "Enhance…" wand on the Image and Video panes (Music has its
/// own in `MusicPromptRewriter`): turns what the user typed into a prompt
/// shaped like the selected model's built-in examples, or for a video longer
/// than one shot, into a storyboard. Pure — the sheet
/// streams the reply via `AgentComposer`.
enum PromptRewriter {

    struct Request: Equatable {
        let system: String
        let user: String
        var maxTokens = 2048
        /// The clip's first frame, shown to a chat model that can see it.
        var firstFrame: Data? = nil
        /// What the model must do with the picture; said only when it is sent.
        var firstFrameRule = ""

        /// The user message, with the picture's rule when the picture goes along.
        func user(seeingImage: Bool) -> String {
            seeingImage && firstFrame != nil ? user + "\n\n" + firstFrameRule : user
        }
    }

    /// The video's frame 0 is the picture and the rest follows the text, so a
    /// prompt describing another look makes H3 and LTX cut away from it.
    private static let firstFrameRule = """
        The video starts EXACTLY on the attached picture: it is the first frame. \
        Describe what it shows as it is (the same people, clothes, setting, lighting and camera framing) \
        and continue the action from there. Never contradict it: a prompt that describes a different look \
        makes the video cut away from the picture.
        """

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

    static func video(text: String, format: VideoPromptFormat, seconds: Int, firstFrame: Data? = nil) -> Request {
        let labels = H3PromptExamples.sections(for: format)
        let shape = labels.isEmpty
            ? "Write ONE paragraph of 4-8 sentences like the examples: subject, action, camera movement, lighting, setting, sound. Keep spoken dialogue in double quotes."
            : "Write the prompt in the exact labelled format of the examples, with these labels in order, each on its own line: \(labels.joined(separator: ", ")). Keep any <Picture N>, <Video N> or <Audio N> references verbatim."
        var r = request(noun: "video prompt", shape: shape,
                        examples: H3PromptExamples.examples(for: format).prefix(3).map(\.body), text: text,
                        note: "The clip is \(seconds) seconds long. Describe only what happens in that time: no more action than fits.")
        r.firstFrame = firstFrame
        r.firstFrameRule = firstFrameRule
        return r
    }

    /// A storyboard plan: shots headed `=== SHOT n | Ns ===` (`Storyboard.parse`).
    /// Each shot is generated alone from the previous shot's last frame, which
    /// is why every shot restates the cast and keeps one continuous place.
    static func storyboard(idea: String, format: VideoPromptFormat, totalSeconds: Int,
                           shotSeconds: ClosedRange<Int>, firstFrame: Data? = nil) -> Request {
        let labels = H3PromptExamples.sections(for: format)
        let shape = labels.isEmpty
            ? "one paragraph of 4-8 sentences: subject, action, camera movement, lighting, setting, sound. Keep spoken dialogue in double quotes."
            : "the exact labelled format of the examples, with these labels in order, each on its own line: \(labels.joined(separator: ", "))."
        let shots = max(1, Int((Double(totalSeconds) / Double(shotSeconds.upperBound)).rounded(.up)))
        return Request(system: """
            You plan a long video as a storyboard of shots for a generative video model. \
            Each shot is generated on its own: it starts from the last frame of the shot before it and the model sees ONLY that shot's prompt. So:
            - Describe every recurring character, outfit, place and visual style again in EVERY shot, in the same words.
            - Shots flow continuously: a shot begins exactly where the previous one ended, so move the camera or the action rather than cutting to a new scene.
            - Keep the sound and music descriptions the same from shot to shot unless the story changes them.
            - Give each shot one beat of action, no more than fits in its length.
            - Each shot lasts between \(shotSeconds.lowerBound) and \(shotSeconds.upperBound) seconds. Prefer long shots: every join is a seam.
            Write each shot's prompt as \(shape)
            Reply with ONLY the shots. Head each one with a line exactly like `=== SHOT 1 | \(shotSeconds.upperBound)s ===` (its number and its length in seconds), then its prompt. No preamble, no markdown.

            Examples of one shot's prompt format:

            \(H3PromptExamples.examples(for: format).prefix(2).map(\.body).joined(separator: "\n\n---\n\n"))
            """,
            user: "Story:\n\n\(idea)\n\nTotal length: \(totalSeconds) seconds, about \(shots) shots.",
            // A labelled shot runs to a few hundred tokens, and a long story is dozens of shots.
            maxTokens: 1024 + 512 * shots,
            firstFrame: firstFrame,
            firstFrameRule: firstFrameRule + " Shot 1 opens on it, and every later shot keeps that look.")
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
