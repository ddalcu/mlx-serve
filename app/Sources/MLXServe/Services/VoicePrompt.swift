import Foundation

/// System-prompt guidance applied only while a turn originates from Voice mode,
/// so the model answers in a way that's pleasant to *hear*: it speaks briefly and
/// conversationally, uses no Markdown, and never reads tool calls or raw output
/// aloud. Pure → unit-testable.
///
/// Two identity modes, and the distinction is load-bearing. With NO agent the
/// prompt also names the assistant after the wake phrase ("hey jarvis" → "You are
/// Jarvis") — that's the app's own voice persona. With an AGENT selected it must
/// assert no identity at all: this guidance is appended LAST so its style rules
/// win any conflict, which meant the wake-phrase name overrode the agent sitting
/// above it (live: an agent introduced itself as Jarvis). The persona owns who it
/// is; this owns how it sounds.
enum VoicePrompt {

    /// The delivery rules — identical in both identity modes, so they can't drift
    /// apart.
    private static let deliveryRules = """
    - Be brief and conversational — usually one to three sentences. Lead with the answer; skip preamble, filler, and sign-offs.
    - Plain spoken prose only. No Markdown, bullet or numbered lists, headings, tables, code blocks, asterisks, or emoji.
    - Never read out URLs, email addresses, file paths, or long IDs/hashes — don't say "http", "slash", or "dot com". Refer to them in words instead, e.g. "I've put the link in the chat".
    - When you use a tool, do it silently and then just tell me the result in plain words — never read the tool call, the command, the code, or the raw output aloud (don't say things like "shell command date"). Just answer with what you found.
    - Don't recite long lists. Give the few things that matter and offer to go deeper if asked.
    - Say numbers, dates, and units the natural way you'd speak them.
    - If something is inherently visual or long (code, a table, a big list), describe it in a sentence and say the details are in the chat rather than reading it out.
    """

    /// Identity + speaking style for the given wake phrase — the assistant is
    /// named by the phrase's last word ("hey jarvis" → "You are Jarvis").
    /// Tolerates raw user input from Settings; blank falls back to the
    /// default. No date here — that's injected separately via
    /// `SystemGrounding` so it stays fresh and isn't duplicated when an agent
    /// system prompt already carries it.
    ///
    /// `hasPersona` = an agent's system prompt is already in this request: the
    /// name and the "you are a voice assistant" framing are then dropped, and the
    /// model is told to stay in character instead.
    static func speakingStyle(phrase rawPhrase: String, hasPersona: Bool = false) -> String {
        if hasPersona {
            return """
            Everything you say is read aloud by text-to-speech. Stay exactly the assistant described above — same name, same character — and deliver it the way a person would out loud:
            \(deliveryRules)
            """
        }
        let phrase = WakeWord.normalizePhrase(rawPhrase) ?? WakeWord.defaultPhrase
        let name = WakeWord.assistantName(phrase)
        let display = WakeWord.display(phrase)
        return """
        You are \(name), a friendly hands-free voice assistant. The user talks to you by saying "\(display)", and everything you say is read aloud by text-to-speech, so talk like a person would out loud:
        \(deliveryRules)
        """
    }

    /// Default-phrase style, kept as a property for callers/tests that don't
    /// thread a phrase.
    static var speakingStyle: String { speakingStyle(phrase: WakeWord.defaultPhrase) }

    /// Full voice system prompt for plain (non-agent-loop) voice chat: current
    /// date/time grounding followed by the identity + speaking style.
    static func systemPrompt(now: Date = Date(),
                             phrase: String = WakeWord.defaultPhrase,
                             hasPersona: Bool = false) -> String {
        SystemGrounding.dateTimeLine(now: now) + "\n\n"
            + speakingStyle(phrase: phrase, hasPersona: hasPersona)
    }

    /// Decorate an existing system prompt (tool loop) with the voice speaking
    /// style. The agent prompt already carries the date line (injected by the turn
    /// engine), so we deliberately don't repeat it here. The voice guidance goes
    /// last so it takes precedence on any conflict — which is exactly why it must
    /// not claim an identity when `hasPersona` is true.
    static func decorate(_ base: String?,
                         phrase: String = WakeWord.defaultPhrase,
                         hasPersona: Bool = false) -> String {
        let style = speakingStyle(phrase: phrase, hasPersona: hasPersona)
        guard let base, !base.isEmpty else { return style }
        return base + "\n\n# Voice mode\n" + style
    }
}
