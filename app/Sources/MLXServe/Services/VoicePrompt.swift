import Foundation

/// System-prompt guidance applied only while a turn originates from Voice mode,
/// so the model answers in a way that's pleasant to *hear*: it knows who it is
/// (Loki), speaks briefly and conversationally, uses no Markdown, and never
/// reads tool calls or raw output aloud. Pure → unit-testable.
enum VoicePrompt {
    /// Identity + speaking style for the given wake phrase — the assistant is
    /// named by the phrase's last word ("hey jarvis" → "You are Jarvis").
    /// Tolerates raw user input from Settings; blank falls back to the
    /// default. No date here — that's injected separately via
    /// `SystemGrounding` so it stays fresh and isn't duplicated when an agent
    /// system prompt already carries it.
    static func speakingStyle(phrase rawPhrase: String) -> String {
        let phrase = WakeWord.normalizePhrase(rawPhrase) ?? WakeWord.defaultPhrase
        let name = WakeWord.assistantName(phrase)
        let display = WakeWord.display(phrase)
        return """
        You are \(name), a friendly hands-free voice assistant. The user talks to you by saying "\(display)", and everything you say is read aloud by text-to-speech, so talk like a person would out loud:
        - Be brief and conversational — usually one to three sentences. Lead with the answer; skip preamble, filler, and sign-offs.
        - Plain spoken prose only. No Markdown, bullet or numbered lists, headings, tables, code blocks, asterisks, or emoji.
        - Never read out URLs, email addresses, file paths, or long IDs/hashes — don't say "http", "slash", or "dot com". Refer to them in words instead, e.g. "I've put the link in the chat".
        - When you use a tool, do it silently and then just tell me the result in plain words — never read the tool call, the command, the code, or the raw output aloud (don't say things like "shell command date"). Just answer with what you found.
        - Don't recite long lists. Give the few things that matter and offer to go deeper if asked.
        - Say numbers, dates, and units the natural way you'd speak them.
        - If something is inherently visual or long (code, a table, a big list), describe it in a sentence and say the details are in the chat rather than reading it out.
        """
    }

    /// Default-phrase style, kept as a property for callers/tests that don't
    /// thread a phrase.
    static var speakingStyle: String { speakingStyle(phrase: WakeWord.defaultPhrase) }

    /// Full voice system prompt for plain (non-agent) voice chat: current
    /// date/time grounding followed by the identity + speaking style.
    static func systemPrompt(now: Date = Date(), phrase: String = WakeWord.defaultPhrase) -> String {
        SystemGrounding.dateTimeLine(now: now) + "\n\n" + speakingStyle(phrase: phrase)
    }

    /// Decorate an existing system prompt (agent mode) with the voice identity +
    /// speaking style. The agent prompt already carries the date line (injected
    /// by the turn engine), so we deliberately don't repeat it here. The voice
    /// guidance goes last so it takes precedence on any conflict.
    static func decorate(_ base: String?, phrase: String = WakeWord.defaultPhrase) -> String {
        let style = speakingStyle(phrase: phrase)
        guard let base, !base.isEmpty else { return style }
        return base + "\n\n# Voice mode\n" + style
    }
}
