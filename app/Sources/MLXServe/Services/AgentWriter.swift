import Foundation

/// The pure half of "describe an agent, get a system prompt": the instructions
/// handed to the writing model, the tolerant parse of its reply, and the
/// cleaners. It never calls a model — the Agents window runs one turn through
/// `ChatTurnEngine` and hands the reply here — so all of it is unit-tested.
///
/// Ported verbatim from the iPhone app (mlx-iphone `Models/Agent.swift`), tests
/// included, so a fix on either side transplants cleanly.
enum AgentWriter {

    /// What the writing model gives back.
    struct Draft: Equatable {
        var name: String
        var systemPrompt: String
    }

    /// Hard caps. A system prompt rides on EVERY turn, so an essay here costs
    /// context for the life of the chat.
    static let maxPromptCharacters = 1200
    static let maxNameCharacters = 40

    /// Instructions for the model that writes the prompt. The two-line tagged
    /// format is deliberate: a small model emits malformed JSON often enough to
    /// matter, but keeps a `TAG: value` shape reliably — and when it doesn't,
    /// `parse` still salvages the prose.
    static let instructions = """
    You write system prompts for AI assistants. The user describes the \
    assistant they want; you return the prompt that will make a model behave \
    that way.

    Reply with exactly two lines and nothing else:
    NAME: a 2-4 word name for the assistant
    PROMPT: the system prompt itself

    Write the prompt as instructions addressed TO the assistant ("You are…", \
    "You always…"), never as a description of the user. Cover its role and \
    expertise, how it should behave, what it should prioritise, and its tone. \
    Include one instruction to answer concisely — lead with the answer, no \
    padding or restating the question. Keep it under 120 words, in plain prose. \
    Do not include greetings, sample dialogue, markdown, or any text outside \
    those two lines.
    """

    /// The length instruction an AI-written prompt gets when the model didn't
    /// write its own.
    ///
    /// Written INTO the prompt rather than kept as a setting: it shows up in the
    /// editor, so it can be reworded or deleted like any other sentence, and a
    /// prompt the user typed themselves is never touched. A persona says nothing
    /// about length on its own, which is why agents answered at essay length.
    static let brevityLine = "Answer in as few words as the question needs — usually one to three sentences. Lead with the answer, skip preamble and sign-offs, and add detail only when asked."

    /// Does this prompt already tell the model how long to answer? Covers the
    /// phrasings a writing model actually produces, so the backstop line isn't
    /// bolted onto a prompt that already says it.
    static func mentionsBrevity(_ prompt: String) -> Bool {
        let p = prompt.lowercased()
        let cues = ["concise", "concisely", "brief", "briefly", "short answer",
                    "short replies", "keep replies short", "keep answers short",
                    "few words", "verbose", "to the point", "one to three sentences",
                    "as few words"]
        return cues.contains { p.contains($0) }
    }

    /// A draft as the macOS app STORES it: same words, with the length
    /// instruction guaranteed. Applied by `AgentComposer.draftAgent` — i.e. only
    /// when a model wrote the prompt — so `parse` and `fallbackDraft` stay
    /// byte-identical to the iPhone's and the port keeps transplanting.
    static func concise(_ draft: Draft) -> Draft {
        Draft(name: draft.name, systemPrompt: withBrevity(draft.systemPrompt))
    }

    /// The model's prose, trimmed to leave room, with `brevityLine` appended when
    /// it didn't say it itself. The line is added AFTER the cap is applied to the
    /// prose, so a model that fills the whole budget can't push it off the end.
    static func withBrevity(_ prompt: String) -> String {
        let cleaned = cleanPrompt(prompt)
        guard !cleaned.isEmpty, !mentionsBrevity(cleaned) else { return cleaned }
        let room = maxPromptCharacters - brevityLine.count - 1
        guard room > 0 else { return cleaned }
        var prose = cleaned
        if prose.count > room {
            let clipped = String(prose.prefix(room))
            prose = clipped.lastIndex(where: { ".!?".contains($0) }).map { String(clipped[...$0]) } ?? clipped
        }
        return prose + " " + brevityLine
    }

    /// The user-side message for the writing turn.
    static func request(brief: String) -> String {
        "Write the system prompt for this assistant: \(brief.trimmingCharacters(in: .whitespacesAndNewlines))"
    }

    // MARK: - Parsing

    /// Pull a draft out of whatever the model returned. Tolerant by design:
    /// missing tags, a chatty preamble, markdown fences and quoting are all
    /// things small models do, and none of them should cost the user their
    /// agent. Returns nil only when there's no usable prose at all.
    static func parse(_ reply: String, brief: String) -> Draft? {
        let body = stripFences(reply)
        let lines = body.components(separatedBy: .newlines)

        var name: String?
        var promptLines: [String] = []
        var collectingPrompt = false

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let value = value(of: "NAME", in: trimmed), !collectingPrompt {
                name = value
                continue
            }
            if let value = value(of: "PROMPT", in: trimmed) {
                collectingPrompt = true
                if !value.isEmpty { promptLines.append(value) }
                continue
            }
            // Everything after a PROMPT: tag belongs to the prompt (models
            // wrap onto several lines); before one, it's a preamble we keep
            // only if no tag ever shows up.
            promptLines.append(trimmed)
        }

        // No PROMPT: tag at all — treat the whole reply (minus a NAME line) as
        // the prompt rather than failing.
        let raw = promptLines.joined(separator: "\n")
        let prompt = cleanPrompt(raw)
        guard !prompt.isEmpty else { return nil }
        return Draft(
            name: cleanName(name ?? "") ?? fallbackName(brief: brief),
            systemPrompt: prompt
        )
    }

    /// `TAG: value` (case-insensitive), tolerating markdown bold around the
    /// tag — `**PROMPT:**` is a common small-model flourish.
    private static func value(of tag: String, in line: String) -> String? {
        let bare = line.replacingOccurrences(of: "*", with: "")
            .replacingOccurrences(of: "#", with: "")
            .trimmingCharacters(in: .whitespaces)
        guard bare.count >= tag.count + 1 else { return nil }
        let prefix = String(bare.prefix(tag.count + 1))
        guard prefix.lowercased() == "\(tag.lowercased()):" else { return nil }
        return String(bare.dropFirst(tag.count + 1)).trimmingCharacters(in: .whitespaces)
    }

    // MARK: - Cleaners

    /// Drop a ``` fence wrapper if the model returned one.
    static func stripFences(_ text: String) -> String {
        var out = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard out.hasPrefix("```") else { return out }
        // Drop the opening fence line (```/```json) and any closing fence.
        if let firstNewline = out.firstIndex(of: "\n") {
            out = String(out[out.index(after: firstNewline)...])
        }
        if let fence = out.range(of: "```", options: .backwards) {
            out = String(out[out.startIndex..<fence.lowerBound])
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The prompt as it will be stored: no markdown, no wrapping quotes, blank
    /// lines collapsed, and capped — cut at a sentence end so it never trails
    /// off mid-word.
    static func cleanPrompt(_ raw: String) -> String {
        var text = stripFences(raw)
        text = text.replacingOccurrences(of: #"^\s*#{1,6}\s*"#, with: "",
                                         options: [.regularExpression])
        text = text.replacingOccurrences(of: "**", with: "")
        text = text.replacingOccurrences(of: #"\n{3,}"#, with: "\n\n",
                                         options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        text = trimWrappingQuotes(text)
        guard text.count > maxPromptCharacters else { return text }

        let clipped = String(text.prefix(maxPromptCharacters))
        if let end = clipped.lastIndex(where: { ".!?".contains($0) }) {
            return String(clipped[...end])
        }
        return clipped
    }

    /// A short, plain name. nil when nothing usable is left.
    static func cleanName(_ raw: String) -> String? {
        var text = raw.replacingOccurrences(of: "*", with: "")
            .replacingOccurrences(of: "#", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        text = trimWrappingQuotes(text)
        guard !text.isEmpty else { return nil }
        return String(text.prefix(maxNameCharacters))
    }

    private static func trimWrappingQuotes(_ text: String) -> String {
        var out = text
        let quotes: Set<Character> = ["\"", "'", "\u{201C}", "\u{201D}", "\u{2018}", "\u{2019}"]
        while let first = out.first, quotes.contains(first) { out.removeFirst() }
        while let last = out.last, quotes.contains(last) { out.removeLast() }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A name derived from the user's own words, for when the model gave none.
    /// Creating an agent must never dead-end on a naming failure.
    static func fallbackName(brief: String) -> String {
        let words = brief
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .prefix(4)
        let joined = words.joined(separator: " ")
        guard !joined.isEmpty else { return "New Agent" }
        return String(joined.prefix(maxNameCharacters))
    }

    /// The agent to save when the writing model fails outright (no model loaded,
    /// a refusal, a dead peer). The user's own description IS a serviceable
    /// system prompt — far better than losing what they typed.
    static func fallbackDraft(brief: String) -> Draft {
        let cleaned = brief.trimmingCharacters(in: .whitespacesAndNewlines)
        return Draft(
            name: fallbackName(brief: cleaned),
            systemPrompt: cleaned.isEmpty
                ? "You are a helpful assistant."
                : "You are an assistant with this purpose: \(cleaned)"
        )
    }
}

/// An avatar glyph guessed from the agent's own words, so a list of agents
/// doesn't read as a column of identical discs. Pure — unit-tested.
enum AgentSymbol {
    /// First match wins, so put the specific words before the general ones.
    static let rules: [(keywords: [String], symbol: String)] = [
        (["code", "program", "developer", "bug", "review", "engineer"], "chevron.left.forwardslash.chevron.right"),
        (["cook", "recipe", "chef", "food", "meal", "kitchen"], "fork.knife"),
        (["travel", "trip", "flight", "holiday", "vacation", "itinerary"], "airplane"),
        (["money", "budget", "finance", "invest", "tax", "account"], "banknote"),
        (["teach", "tutor", "learn", "study", "school", "explain", "math"], "graduationcap"),
        (["write", "editor", "copy", "story", "novel", "blog"], "pencil.and.outline"),
        (["fitness", "workout", "train", "run", "gym", "health"], "figure.run"),
        (["doctor", "medical", "symptom", "nurse"], "cross.case"),
        (["law", "legal", "contract", "lawyer"], "building.columns"),
        (["music", "song", "guitar", "band"], "music.note"),
        (["news", "research", "search", "web"], "globe"),
        (["game", "play", "dungeon", "rpg"], "gamecontroller"),
        (["plant", "garden", "grow"], "leaf"),
    ]

    static func pick(for text: String) -> String {
        let haystack = text.lowercased()
        for rule in rules where rule.keywords.contains(where: haystack.contains) {
            return rule.symbol
        }
        return "sparkles"
    }

    /// Glyphs offered by the editor's symbol picker: the rule table's own
    /// symbols, so the picker can never offer something the guesser wouldn't.
    static var pickerChoices: [String] {
        var seen = Set<String>()
        return (["sparkles"] + rules.map(\.symbol)).filter { seen.insert($0).inserted }
    }
}
