import Foundation

/// What a tool-call card shows, derived from `SerializedToolCall` rather than
/// the engine's `**name**(key: value…)` summary text (values cut at 80 chars).
enum ToolCallDisplay {

    struct Argument: Equatable, Identifiable {
        let name: String
        let value: String
        var id: String { name }
    }

    /// Longest value shown: a `writeFile` carries the whole file in `content`.
    static let valueLimit = 1200

    /// Arguments in the order the model sent them, recovered from the raw JSON
    /// (a dictionary has none; alphabetical puts a file body above its path).
    static func arguments(fromJSON json: String) -> [Argument] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }

        return orderedKeys(in: json, among: Set(object.keys)).map { key in
            Argument(name: key, value: flatten(object[key]))
        }
    }

    /// A value as one line: whitespace collapsed, cut at `valueLimit`.
    static func flatten(_ value: Any?) -> String {
        let text: String
        switch value {
        case let s as String: text = s
        case let n as NSNumber:
            // JSON `true` arrives as an NSNumber with `stringValue` "1".
            text = CFGetTypeID(n) == CFBooleanGetTypeID()
                ? (n.boolValue ? "true" : "false")
                : n.stringValue
        case .none, is NSNull: text = "null"
        default:
            text = (try? JSONSerialization.data(withJSONObject: value as Any, options: [.fragmentsAllowed]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? String(describing: value ?? "")
        }

        let collapsed = text
            .components(separatedBy: CharacterSet.newlines.union(.whitespaces))
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        return collapsed.count > valueLimit
            ? String(collapsed.prefix(valueLimit)) + "…"
            : collapsed
    }

    /// The result without the `**name** → ` prefix; a summary without the
    /// marker is shown as is.
    static func resultBody(_ summary: String) -> String {
        guard let range = summary.range(of: "** → ") else {
            return summary.replacingOccurrences(of: "**", with: "")
        }
        return String(summary[range.upperBound...])
    }

    /// The argument each tool is about, shown in the header. A list, not a
    /// rule: an unlisted tool shows nothing extra rather than a wrong guess.
    static let headlineArgument: [String: String] = [
        "shell": "command",
        "cwd": "path",
        "writeFile": "path",
        "readFile": "path",
        "editFile": "path",
        "searchFiles": "pattern",
        "listFiles": "path",
        "browse": "url",
        "webSearch": "query",
        "searchDocuments": "query",
        "saveMemory": "memory",
        "createTask": "goal",
        "readProcessOutput": "handle",
        "killProcess": "handle",
        "generate_image": "prompt",
        "generate_speech": "text",
        "generate_music": "prompt",
        "generate_video": "prompt",
    ]

    /// Tools whose behaviour is chosen by one argument, shown as part of the
    /// name (`browse:click`).
    static let variantArgument: [String: String] = [
        "browse": "action",
    ]

    static func variant(toolName: String, arguments: [Argument]) -> String? {
        let bare = toolName.components(separatedBy: "__").last ?? toolName
        guard let key = variantArgument[toolName] ?? variantArgument[bare],
              let value = arguments.first(where: { $0.name == key })?.value,
              !value.isEmpty
        else { return nil }
        return value
    }

    static let headlineLimit = 90

    /// What follows the tool's name in the header. An MCP tool is looked up
    /// under its bare name too.
    static func headline(toolName: String, arguments: [Argument]) -> String? {
        let bare = MCPManager.parseNamespacedName(toolName)?.tool ?? toolName
        // `browse`: the action decides which argument matters.
        if bare == "browse" {
            for key in ["selector", "script", "url"] {
                if let value = arguments.first(where: { $0.name == key })?.value, !value.isEmpty {
                    return value.count > headlineLimit
                        ? String(value.prefix(headlineLimit)) + "…"
                        : value
                }
            }
            return nil
        }
        guard let key = headlineArgument[toolName] ?? headlineArgument[bare],
              let value = arguments.first(where: { $0.name == key })?.value,
              !value.isEmpty
        else { return nil }
        return value.count > headlineLimit
            ? String(value.prefix(headlineLimit)) + "…"
            : value
    }

    /// What came of the call, from its result text. Nil for a tool with no
    /// rule or when the output does not match: a changed message goes quiet
    /// rather than showing a wrong number.
    static func resultHeadline(toolName: String, result: String) -> String? {
        let bare = toolName.components(separatedBy: "__").last ?? toolName
        switch bare {
        case "writeFile":
            // "Wrote 1234 characters to x" / "Appended 1234 characters to x"
            guard let match = result.range(of: "\\d+(?= characters)", options: .regularExpression)
            else { return nil }
            return "\(result[match]) chars"
        case "readFile":
            // Counted off the result (a partial read reports what was handed
            // over); the `[File: …]` metadata header is not a line.
            let lines = result
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.hasPrefix("[File: ") }
            guard !lines.isEmpty, !result.isEmpty else { return nil }
            return "\(lines.count) line\(lines.count == 1 ? "" : "s")"
        case "editFile":
            // Line mode returns "replaced lines 4-9" (clamped to the file);
            // text mode has no count to give.
            guard let range = result.range(of: "replaced lines \\d+-\\d+",
                                           options: .regularExpression) else { return nil }
            let bounds = result[range]
                .replacingOccurrences(of: "replaced lines ", with: "")
                .split(separator: "-")
                .compactMap { Int($0) }
            guard bounds.count == 2, bounds[1] >= bounds[0] else { return nil }
            let count = bounds[1] - bounds[0] + 1
            return "\(count) line\(count == 1 ? "" : "s")"
        case "searchFiles":
            // Hits are `path:line:text`; context lines are `path-line-text`.
            if result.hasPrefix("No matches found") { return "nothing found" }
            var files = Set<String>()
            var hits = 0
            for line in result.split(separator: "\n") {
                guard let sep = line.range(of: ":\\d+:", options: .regularExpression)
                else { continue }
                hits += 1
                files.insert(String(line[line.startIndex..<sep.lowerBound]))
            }
            guard hits > 0 else { return nil }
            return "\(hits) occurrence\(hits == 1 ? "" : "s") in "
                + "\(files.count) file\(files.count == 1 ? "" : "s")"
        case "listFiles":
            if result.hasPrefix("No files found") { return "nothing found" }
            let truncated = result.contains("[... truncated at 200 entries]")
            let entries = result
                .split(separator: "\n", omittingEmptySubsequences: true)
                .filter { !$0.hasPrefix("[... truncated") }
            guard !entries.isEmpty else { return nil }
            return truncated
                ? "200+ files found"
                : "\(entries.count) file\(entries.count == 1 ? "" : "s") found"
        default:
            return nil
        }
    }

    /// The handle a `shell` result announced (`… as bg1 (pid 123)`).
    /// `processHandles` is per round, so this is the only per-call association.
    static func backgroundHandle(inResult result: String) -> String? {
        guard let range = result.range(of: " as bg\\d+ \\(", options: .regularExpression)
        else { return nil }
        return result[range]
            .trimmingCharacters(in: CharacterSet(charactersIn: " ("))
            .replacingOccurrences(of: "as ", with: "")
    }

    /// `server__tool` shown as `server/tool`, split the way dispatch splits it.
    static func displayName(_ toolName: String) -> String {
        guard let (server, tool) = MCPManager.parseNamespacedName(toolName) else { return toolName }
        return "\(server)/\(tool)"
    }

    /// From the structured record, else from the `**name**(args)` summary.
    static func title(calls: [SerializedToolCall], summary: String) -> String {
        if let first = calls.first, !first.name.isEmpty {
            return calls.count > 1 ? "\(first.name) +\(calls.count - 1)" : first.name
        }
        guard let open = summary.range(of: "**"),
              let close = summary.range(of: "**", range: open.upperBound..<summary.endIndex)
        else { return summary.replacingOccurrences(of: "**", with: "") }
        return String(summary[open.upperBound..<close.lowerBound])
    }

    /// Keys in the order they appear in the raw JSON.
    private static func orderedKeys(in json: String, among keys: Set<String>) -> [String] {
        var found: [(offset: Int, key: String)] = []
        for key in keys {
            // Quoted and followed by a colon, or it matches inside a value.
            guard let range = json.range(of: "\"\(key)\"\\s*:", options: .regularExpression) else {
                found.append((Int.max, key))
                continue
            }
            found.append((json.distance(from: json.startIndex, to: range.lowerBound), key))
        }
        return found.sorted { ($0.offset, $0.key) < ($1.offset, $1.key) }.map(\.key)
    }
}
