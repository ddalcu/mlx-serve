import Foundation

/// What a tool-call card shows, derived from the STRUCTURED record rather than
/// from the summary text.
///
/// The transcript used to render `call.content`, which the engine had just
/// built as `**name**(key: value, key: value)` with each value truncated at 80
/// characters — so the card stripped the bold markers off a string that had
/// been assembled purely to be read back. `SerializedToolCall` carries the name
/// and the arguments as JSON, and the result rides its own message, so nothing
/// here has to be parsed out of prose.
enum ToolCallDisplay {

    /// One argument, ready to draw.
    struct Argument: Equatable, Identifiable {
        let name: String
        let value: String
        var id: String { name }
    }

    /// Longest value shown. A `writeFile` carries an entire file in `content`,
    /// and the panel is a summary of what was asked, not a second copy of the
    /// document — the file itself is one `readFile` away.
    static let valueLimit = 1200

    /// Arguments in the order the model sent them.
    ///
    /// `JSONSerialization` hands back a dictionary, which has no order, so the
    /// keys are recovered from the raw JSON by where they appear in it. A model
    /// writes `path` before `content` for a reason, and re-sorting them
    /// alphabetically would put a 4 KB file body above the path it was written
    /// to.
    static func arguments(fromJSON json: String) -> [Argument] {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [] }

        return orderedKeys(in: json, among: Set(object.keys)).map { key in
            Argument(name: key, value: flatten(object[key]))
        }
    }

    /// A value as one line: every kind of line break becomes a space, runs of
    /// whitespace collapse, and a long one is cut with an ellipsis.
    ///
    /// Newlines are what make an argument panel unreadable — a file body turns
    /// a two-line card into a two-screen one — and the panel's job is to let
    /// you see WHAT was passed at a glance.
    static func flatten(_ value: Any?) -> String {
        let text: String
        switch value {
        case let s as String: text = s
        case let n as NSNumber:
            // JSON's `true` arrives as an NSNumber whose `stringValue` is "1".
            // Only its CoreFoundation type tells a boolean from the number one,
            // and `run_in_background: 1` is not what the model sent.
            text = CFGetTypeID(n) == CFBooleanGetTypeID()
                ? (n.boolValue ? "true" : "false")
                : n.stringValue
        case .none, is NSNull: text = "null"
        default:
            text = (try? JSONSerialization.data(withJSONObject: value as Any, options: [.fragmentsAllowed]))
                .flatMap { String(data: $0, encoding: .utf8) } ?? String(describing: value ?? "")
        }

        // Split on whitespace rather than matching it: `CharacterSet.newlines`
        // already covers CR, LF, CRLF and the two Unicode separators, and an
        // ICU pattern spelling them out needs ` `, not Swift's `\u{2028}` —
        // which makes the whole pattern invalid, so nothing is replaced at all
        // and a file body reaches the panel with its newlines intact.
        let collapsed = text
            .components(separatedBy: CharacterSet.newlines.union(.whitespaces))
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        return collapsed.count > valueLimit
            ? String(collapsed.prefix(valueLimit)) + "…"
            : collapsed
    }

    /// The result, without the `**name** → ` the summary was built with.
    ///
    /// Kept tolerant: a summary that does not carry the marker (an older
    /// history, a shape that changes upstream) is shown as it is rather than
    /// silently emptied.
    static func resultBody(_ summary: String) -> String {
        guard let range = summary.range(of: "** → ") else {
            return summary.replacingOccurrences(of: "**", with: "")
        }
        return String(summary[range.upperBound...])
    }

    /// The argument each tool is ABOUT, shown beside its name in the header so
    /// a settled call says what it did without being opened.
    ///
    /// One entry per tool rather than a rule ("show the first argument"): the
    /// interesting one is not always first, and for several tools it is not the
    /// one a rule would pick — `searchFiles` is about its `pattern`, not the
    /// `path` it searched, and `editFile` is about the file, not the text.
    ///
    /// A tool that is not listed shows nothing extra, which is what every tool
    /// did before this existed. So a new tool degrades to the old behaviour
    /// rather than to a wrong guess.
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

    /// Tools whose behaviour is really chosen by ONE argument, so that argument
    /// reads as part of the name: `browse:click` rather than `browse` with a
    /// `click` buried in the panel.
    ///
    /// `browse` is the whole list today. Its seven actions (navigate, readText,
    /// extractText, readHTML, click, executeJS, screenshot) share nothing but a
    /// browser — they could each have been a tool of their own.
    static let variantArgument: [String: String] = [
        "browse": "action",
    ]

    /// The variant part of a call's name, or nil for a tool that has none.
    static func variant(toolName: String, arguments: [Argument]) -> String? {
        let bare = toolName.components(separatedBy: "__").last ?? toolName
        guard let key = variantArgument[toolName] ?? variantArgument[bare],
              let value = arguments.first(where: { $0.name == key })?.value,
              !value.isEmpty
        else { return nil }
        return value
    }

    /// Longest headline. Short, because it shares a line with the tool's name
    /// and a chevron: it is a reminder of what the call was about, and the
    /// panel below holds the rest.
    static let headlineLimit = 90

    /// What follows the tool's name in the header, or nil when this tool has no
    /// headline argument or the call did not carry it.
    ///
    /// An MCP tool (`<server>__<tool>`) is looked up under its bare name too,
    /// so a server exposing `readFile` gets the same treatment as the built-in.
    static func headline(toolName: String, arguments: [Argument]) -> String? {
        let bare = MCPManager.parseNamespacedName(toolName)?.tool ?? toolName
        // `browse` has no single interesting argument: its action decides which
        // one matters (a selector for click, a script for executeJS, a URL only
        // for navigate). Take whichever is present, in that order of interest —
        // `url` is last because the browser keeps its page between calls, so
        // most actions carry none.
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

    /// A second headline piece, taken from what the call RETURNED.
    ///
    /// Separate from `headline` because it answers a different question: the
    /// argument says what the call was about, this says what came of it. For
    /// `writeFile` that is the size actually written — which is the number
    /// worth seeing without opening the panel, and is not knowable from the
    /// arguments (the value shown there is truncated, and an append writes only
    /// its own chunk).
    ///
    /// Nil while the call is still running, for a tool with no rule, or when
    /// the output does not match — a changed message must go quiet rather than
    /// show a wrong number.
    static func resultHeadline(toolName: String, result: String) -> String? {
        let bare = toolName.components(separatedBy: "__").last ?? toolName
        switch bare {
        case "writeFile":
            // "Wrote 1234 characters to x" / "Appended 1234 characters to x"
            guard let match = result.range(of: "\\d+(?= characters)", options: .regularExpression)
            else { return nil }
            return "\(result[match]) chars"
        case "readFile":
            // Lines, not characters: the tool numbers every line it returns
            // (`42| let x = 1`) precisely because the model works in lines —
            // `editFile` takes a startLine/endLine — so that is the unit the
            // call is measured in. Counted off the result rather than the file,
            // so a partial read (startLine/endLine, or a long file truncated on
            // its way to the model) reports what was actually handed over.
            //
            // The metadata header a large file carries is one of those lines,
            // so it is dropped rather than counted.
            let lines = result
                .split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.hasPrefix("[File: ") }
            guard !lines.isEmpty, !result.isEmpty else { return nil }
            return "\(lines.count) line\(lines.count == 1 ? "" : "s")"
        case "editFile":
            // Two modes, and only one of them can be counted. Line mode
            // returns "Edited x (replaced lines 4-9)" — the range AFTER it was
            // clamped to the file, so a model asking for 4-999 in a 50-line
            // file reports what it really replaced. Text mode (find/replace)
            // returns "Edited x" and nothing else: `find` can be part of one
            // line, so there is no line count to give.
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
            // The output is one line per hit, `path:line:text`, with context
            // lines around it spelled `path-line-text` — the separator is what
            // tells a match from its context, so only the colon form is
            // counted. Both numbers matter and mean different things: forty
            // hits in one file is a busy file, forty across twenty is a
            // codebase-wide name.
            if result.hasPrefix("No matches found") { return "nothing found" }
            var files = Set<String>()
            var hits = 0
            for line in result.split(separator: "\n") {
                // The FIRST `:<digits>:` ends the path. A path may contain a
                // colon, so the search is for the whole separator, not for a
                // colon on its own.
                guard let sep = line.range(of: ":\\d+:", options: .regularExpression)
                else { continue }
                hits += 1
                files.insert(String(line[line.startIndex..<sep.lowerBound]))
            }
            guard hits > 0 else { return nil }
            return "\(hits) occurrence\(hits == 1 ? "" : "s") in "
                + "\(files.count) file\(files.count == 1 ? "" : "s")"
        case "listFiles":
            // One entry per line, or "No files found in x". The listing stops
            // at 200 and says so on its own line, which is not an entry — and
            // the count then reads "200+", because the real number is unknown.
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

    /// The background handle a `shell` result announced, if it announced one.
    ///
    /// `processHandles` is collected across the whole ROUND and pinned to the
    /// one summary message, so nothing in the model records which call started
    /// which process — with two parallel `shell` calls there were two kill
    /// buttons and no way to tell which was which. The handle IS in the result
    /// text, in all three spellings the tool produces (host, sandbox, and a
    /// foreground command that outlived its timeout): `… as bg1 (pid 123)`.
    static func backgroundHandle(inResult result: String) -> String? {
        guard let range = result.range(of: " as bg\\d+ \\(", options: .regularExpression)
        else { return nil }
        return result[range]
            .trimmingCharacters(in: CharacterSet(charactersIn: " ("))
            .replacingOccurrences(of: "as ", with: "")
    }

    /// The name as a reader should see it: an MCP tool's `<server>__<tool>` is
    /// a wire format, and `perry-memory/get_context` says the same thing in the
    /// notation people already read as "this thing, over there".
    ///
    /// Split by `MCPManager.parseNamespacedName`, the function dispatch itself
    /// uses, so the card can never disagree with where the call went: the
    /// server half cannot contain `__` (`namespacedName` collapses it), and
    /// everything after the first separator is the tool's own name, `__`
    /// included. A name that is not of that shape is shown untouched.
    static func displayName(_ toolName: String) -> String {
        guard let (server, tool) = MCPManager.parseNamespacedName(toolName) else { return toolName }
        return "\(server)/\(tool)"
    }

    /// The tool's name for the header, from the structured record when there is
    /// one and from the summary text otherwise (a history written before the
    /// calls were recorded still has `**name**(args)` to read).
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
            // The key as JSON writes it: quoted, followed by a colon. Searching
            // for the bare word would match it inside a VALUE.
            guard let range = json.range(of: "\"\(key)\"\\s*:", options: .regularExpression) else {
                found.append((Int.max, key))
                continue
            }
            found.append((json.distance(from: json.startIndex, to: range.lowerBound), key))
        }
        return found.sorted { ($0.offset, $0.key) < ($1.offset, $1.key) }.map(\.key)
    }
}
