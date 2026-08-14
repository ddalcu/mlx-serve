import Foundation

/// One skill as the composer's menu sees it.
struct SkillSummary: Identifiable, Equatable {
    let name: String
    let description: String
    var id: String { name }
}

/// The composer's "/" menu, and the `/name` invocation it writes.
///
/// Pure: what the typed text is asking of the menu, which skills answer it,
/// and what accepting one does to the field. The view owns only selection and
/// drawing, so the rules are testable without a window.
enum SlashCommands {

    /// The name fragment being typed, or nil when the menu must not open.
    /// Leading "/" only (`src/main.zig` is a path, not a command), and the
    /// token ends at the first whitespace — once the user types a space the
    /// command is chosen and the menu's job is done.
    static func query(in text: String) -> String? {
        guard text.hasPrefix("/") else { return nil }
        let rest = text.dropFirst()
        guard !rest.contains(where: { $0.isWhitespace }) else { return nil }
        return String(rest)
    }

    /// Skills answering `query`, prefix hits before substring hits. An empty
    /// query (a bare "/") lists everything in its own order.
    static func matches(query: String, in skills: [SkillSummary]) -> [SkillSummary] {
        let q = query.lowercased()
        guard !q.isEmpty else { return skills }
        let prefixed = skills.filter { $0.name.lowercased().hasPrefix(q) }
        let rest = skills.filter { !$0.name.lowercased().hasPrefix(q) && $0.name.lowercased().contains(q) }
        return prefixed + rest
    }

    /// The field's text after accepting `name`. The menu is only open while
    /// the whole text IS the half-typed token, so it is replaced wholesale —
    /// and a trailing space puts the caret where the message starts.
    static func accept(_ name: String, in text: String) -> String {
        _ = text
        return "/\(name) "
    }

    /// The skill a message invokes explicitly, lowercased. Typing the name is
    /// the invocation — the skill's own triggers are bypassed.
    static func invokedSkillName(in message: String) -> String? {
        guard message.hasPrefix("/") else { return nil }
        let token = message.dropFirst().prefix { !$0.isWhitespace }
        return token.isEmpty ? nil : token.lowercased()
    }
}
