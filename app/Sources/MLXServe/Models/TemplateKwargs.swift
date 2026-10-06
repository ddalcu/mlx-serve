import Foundation

/// `chat_template_kwargs` rows: the server hands the object to the model's Jinja
/// template verbatim, so keys are the template's own variable names and values
/// keep their JSON type. Known keys get a picker; anything else is free text.
enum TemplateKwargs {
    struct Known {
        let key: String
        let choices: [Any]
        let hint: String
    }

    /// Request-decided keys (enable_thinking, reasoning_effort) only fill in when
    /// the client sends nothing; preserve_thinking keeps the template's default.
    static let known: [Known] = [
        Known(key: "preserve_thinking", choices: [true, false],
              hint: "Prior-turn reasoning in the prompt. Default: the template's (Qwen3.8 keeps it)."),
        Known(key: "enable_thinking", choices: [true, false],
              hint: "Only when the request does not decide thinking itself."),
        Known(key: "reasoning_effort", choices: ["low", "medium", "high"],
              hint: "Only when the request sends no reasoning_effort."),
    ]

    static func hint(for key: String) -> String? { known.first { $0.key == key }?.hint }
    static func choices(for key: String) -> [Any]? { known.first { $0.key == key }?.choices }

    /// Type from the spelling: true/false, integer, decimal, else string. Blank = nothing.
    static func parse(_ text: String) -> Any? {
        let t = text.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return nil }
        switch t.lowercased() {
        case "true": return true
        case "false": return false
        default: break
        }
        if let i = Int(t) { return i }
        if let d = Double(t), d.isFinite { return d }
        return t
    }

    static func display(_ value: Any) -> String {
        if let b = value as? Bool { return b ? "true" : "false" }
        if let s = value as? String { return s }
        if let n = value as? NSNumber { return n.stringValue }
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]),
           let s = String(data: data, encoding: .utf8) { return s }
        return String(describing: value)
    }
}
