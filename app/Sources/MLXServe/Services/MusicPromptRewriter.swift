import Foundation

/// Prompts for the Music pane's "Rewrite with LLM" wand: turns what the user
/// typed into a style caption or lyrics shaped like the CURRENT family's
/// built-in examples. Pure — the sheet streams the reply via `AgentComposer`.
enum MusicPromptRewriter {

    enum Kind: String, Identifiable { case style, lyrics; var id: String { rawValue } }

    struct Request: Equatable {
        let system: String
        let user: String
    }

    static func request(_ kind: Kind, text: String, family: MusicEngineFamily,
                        other: String, instrumental: Bool, language: String) -> Request {
        let lang = MusicOptions.languages.first { $0.code == language }?.label ?? language
        switch kind {
        case .style:
            let examples = MusicPrompt.builtinStyles(for: family).map(\.body).joined(separator: "\n\n---\n\n")
            let shape = family == .minimaxMusic3
                ? "Write the caption in the exact three-block format of the examples (Global Metadata / Vocal Details / Arrangement) with the same labelled lines, including bpm, key and scale."
                : "Write ONE paragraph of plain prose like the examples: genre, mood, instruments, production. No tempo, key or time signature (those are separate controls). No headings, no lists."
            let system = """
                You rewrite music style prompts for a text-to-music model. \(shape) \
                Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten prompt, no preamble, no quotes, no markdown.

                Examples of the expected format:

                \(examples)
                """
            var user = "Rewrite this style prompt:\n\n\(text)"
            if instrumental { user += "\n\nThe track is instrumental: no vocals." }
            let lyricsNote = other.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lyricsNote.isEmpty, !instrumental { user += "\n\nIt will sing these lyrics (for mood and language):\n\(lyricsNote)" }
            return Request(system: system, user: user)
        case .lyrics:
            let examples = MusicPrompt.builtinLyrics.map(\.body).joined(separator: "\n\n---\n\n")
            let system = """
                You write song lyrics for a text-to-music model. Use section tags on their own lines, exactly like the examples: \(MusicOptions.sectionTagHint). \
                Keep the user's theme and any lines they wrote; tighten rhythm and rhyme. Reply with ONLY the lyrics, no title, no preamble, no quotes, no markdown.

                Examples of the expected format:

                \(examples)
                """
            var user = "Rewrite these lyrics in \(lang):\n\n\(text)"
            let style = other.trimmingCharacters(in: .whitespacesAndNewlines)
            if !style.isEmpty { user += "\n\nThe music style is:\n\(style)" }
            return Request(system: system, user: user)
        }
    }

    /// Model replies sometimes wear a fence or quotes; the editor gets the bare text.
    static func clean(_ reply: String) -> String {
        AgentWriter.stripFences(reply).trimmingCharacters(in: CharacterSet(charactersIn: "\"\u{201C}\u{201D}"))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
