import Foundation

/// Prompts for the Music pane's "Rewrite with LLM" wand: turns what the user
/// typed into a style caption or lyrics shaped like the CURRENT family's
/// built-in examples. Pure — the sheet streams the reply via `AgentComposer`.
enum MusicPromptRewriter {

    enum Kind: String, Identifiable { case style, lyrics; var id: String { rawValue } }

    static func request(_ kind: Kind, text: String, family: MusicEngineFamily,
                        other: String, instrumental: Bool, language: String) -> PromptRewriter.Request {
        let lang = MusicOptions.languages.first { $0.code == language }?.label ?? language
        switch kind {
        case .style:
            let system = styleSystem(for: family)
            var user = "Rewrite this style prompt:\n\n\(text)"
            if instrumental { user += "\n\nThe track is instrumental: no vocals." }
            let lyricsNote = other.trimmingCharacters(in: .whitespacesAndNewlines)
            if !lyricsNote.isEmpty, !instrumental { user += "\n\nIt will sing these lyrics (for mood and language):\n\(lyricsNote)" }
            return PromptRewriter.Request(system: system, user: user)
        case .lyrics:
            let system = lyricsSystem(for: family)
            var user = "Rewrite these lyrics in \(lang):\n\n\(text)"
            let style = other.trimmingCharacters(in: .whitespacesAndNewlines)
            if !style.isEmpty { user += "\n\nThe music style is:\n\(style)" }
            return PromptRewriter.Request(system: system, user: user)
        }
    }

    /// The system prompt for lyrics: the section-tag grammar plus the family's examples.
    private static func lyricsSystem(for family: MusicEngineFamily) -> String {
        let examples = MusicPrompt.builtinLyrics(for: family).map(\.body).joined(separator: "\n\n---\n\n")
        return """
            You write song lyrics for a text-to-music model. Use section tags on their own lines, exactly like the examples: \(MusicOptions.sectionTagHint(for: family)). \
            Keep the user's theme and any lines they wrote; tighten rhythm and rhyme. Reply with ONLY the lyrics, no title, no preamble, no quotes, no markdown.

            Examples of the expected format:

            \(examples)
            """
    }

    /// The system prompt for a style caption: the family's format rule plus its
    /// own built-in examples.
    private static func styleSystem(for family: MusicEngineFamily) -> String {
        let examples = MusicPrompt.builtinStyles(for: family).map(\.body).joined(separator: "\n\n---\n\n")
        let shape: String
        switch family {
        case .minimaxMusic3:
            shape = "Write the caption in the exact three-block format of the examples (Global Metadata / Vocal Details / Arrangement) with the same labelled lines, including bpm, key and scale."
        case .yue2:
            shape = "Write ONE line of comma-separated tags like the examples: language or genre first, then instruments, mood and the lead vocal. No sentences, no tempo or key (the score carries those)."
        case .acestep:
            shape = "Write ONE paragraph of plain prose like the examples: genre, mood, instruments, production. No tempo, key or time signature (those are separate controls). No headings, no lists."
        }
        return """
            You rewrite music style prompts for a text-to-music model. \(shape) \
            Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten prompt, no preamble, no quotes, no markdown.

            Examples of the expected format:

            \(examples)
            """
    }

    /// AI Radio: the style prompt for the NEXT track of an endless station. The
    /// station's theme never changes, so each request names the recent tracks
    /// and a nudge toward something else, or the model writes the same song again.
    static func radioRequest(theme: String, recent: [String], nudge: String,
                             family: MusicEngineFamily, instrumental: Bool) -> PromptRewriter.Request {
        let system = styleSystem(for: family) + """


            You are the programme director of an endless radio station with ONE fixed theme. Each request asks for the style prompt of the NEXT track. \
            Stay inside the theme, but make this track clearly different from the recent ones: change at least two of tempo, mood, lead instrument, sub-genre, \
            era or production, rhythm, energy. Never reuse the instrument and mood combination of a recent track. \
            Keep any language, culture or genre the theme names (for example "Romanian") in the prompt. \
            Treat the "lean toward" line as a hard requirement and fold it into the prompt naturally. Write a fresh prompt, do not edit an earlier one.
            """
        var user = "Station theme: \(theme)"
        let shown = recent.suffix(radioRecentShown)
        if !shown.isEmpty {
            user += "\n\nRecent tracks, oldest first (this one must not sound like them):\n" + shown.map { "- \($0)" }.joined(separator: "\n")
        }
        user += "\n\nThis track should lean toward: \(nudge)."
        if instrumental { user += "\n\nThe track is instrumental: no vocals." }
        return PromptRewriter.Request(system: system, user: user)
    }

    /// AI Radio, for a model that always sings: fresh lyrics for the next track. The
    /// language is the one the theme names; the pane's language only decides when it names none.
    static func radioLyricsRequest(theme: String, style: String, nudge: String, recent: [String],
                                   family: MusicEngineFamily, fallbackLanguage: String) -> PromptRewriter.Request {
        let fallback = MusicOptions.languages.first { $0.code == fallbackLanguage }?.label ?? fallbackLanguage
        var user = """
            Write original lyrics for the next song on a radio station.

            Station theme: \(theme)

            Write the lyrics in the language the station theme names or implies (a Romanian theme gets Romanian lyrics); \
            if it names none, write them in \(fallback).

            The music style is:
            \(style)

            Mood and feel: \(nudge).

            The song ends by itself after the last line, so keep it to two verses and a chorus: at most 16 sung lines.
            """
        let shown = recent.suffix(radioRecentShown)
        if !shown.isEmpty {
            user += "\n\nEarlier songs on the station had these styles; choose a different subject and imagery than they would suggest:\n"
                + shown.map { "- \($0)" }.joined(separator: "\n")
        }
        return PromptRewriter.Request(system: lyricsSystem(for: family), user: user)
    }

    /// How many earlier prompts the model is shown.
    static let radioRecentShown = 6
}
