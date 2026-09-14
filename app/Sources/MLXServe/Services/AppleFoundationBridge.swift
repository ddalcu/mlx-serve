import Foundation

/// One entry of the transcript handed to Apple's on-device model, in our own
/// terms. The framework types (`Transcript.Entry`, `DynamicGenerationSchema`)
/// are a mechanical translation of these — kept separate so the mapping from
/// OpenAI-shaped history is testable without a session.
enum AppleChatEntry: Equatable {
    case instructions(String)
    case prompt(String)
    case response(String)
    case toolCalls([AppleToolCallRef])
    case toolOutput(callId: String, name: String, text: String)
}

struct AppleToolCallRef: Equatable {
    let id: String
    let name: String
    let argumentsJSON: String
}

/// A tool parameter's type, as much of JSON Schema as the on-device model's
/// schema language expresses.
indirect enum AppleSchemaNode: Equatable {
    case string(choices: [String])
    case number
    case integer
    case boolean
    case array(AppleSchemaNode)
    case object([AppleToolProperty])
}

struct AppleToolProperty: Equatable {
    let name: String
    let description: String?
    let node: AppleSchemaNode
    let isOptional: Bool
}

struct AppleToolDef: Equatable {
    let name: String
    let description: String
    let properties: [AppleToolProperty]
}

/// The pure half of the Apple Foundation Models chat route.
enum AppleFoundationBridge {

    // MARK: - History

    /// Split OpenAI-shaped history into a transcript and the prompt to answer.
    ///
    /// The framework has no "continue" call: every response answers a prompt.
    /// A turn that follows tool results therefore takes the RESULTS as its
    /// prompt — the call that produced them stays in the transcript, so the
    /// model still sees what it asked for.
    static func plan(messages: [[String: Any]]) -> (entries: [AppleChatEntry], prompt: String) {
        var instructions: [String] = []
        var body: [AppleChatEntry] = []
        var callNames: [String: String] = [:]   // tool_call_id → tool name
        var tail: [AppleChatEntry] = []         // trailing tool outputs / final user turn

        for msg in messages {
            let role = msg["role"] as? String ?? ""
            let text = joinedText(msg["content"])
            switch role {
            case "system":
                if !text.isEmpty { instructions.append(text) }
            case "user":
                body.append(contentsOf: tail); tail = []
                tail = [.prompt(text)]
            case "tool":
                let id = msg["tool_call_id"] as? String ?? ""
                tail.append(.toolOutput(callId: id, name: callNames[id] ?? "tool", text: text))
            case "assistant":
                body.append(contentsOf: tail); tail = []
                if !text.isEmpty { body.append(.response(text)) }
                let calls = toolCalls(msg["tool_calls"])
                if !calls.isEmpty {
                    for c in calls { callNames[c.id] = c.name }
                    body.append(.toolCalls(calls))
                }
            default:
                break
            }
        }

        var entries: [AppleChatEntry] = []
        if !instructions.isEmpty { entries.append(.instructions(instructions.joined(separator: "\n\n"))) }
        entries.append(contentsOf: body)

        // The tail is either the final user turn or the tool results that
        // followed the last call; both become the prompt.
        var promptParts: [String] = []
        for entry in tail {
            switch entry {
            case .prompt(let t): promptParts.append(t)
            case .toolOutput(_, let name, let text):
                promptParts.append("Tool \(name) returned:\n\(text)")
            default: break
            }
        }
        return (entries, promptParts.joined(separator: "\n\n"))
    }

    /// A content field is a string or an array of typed blocks. The on-device
    /// model is text-only, so image blocks drop and the text joins in order —
    /// the same "join, never last-wins" rule the server applies.
    private static func joinedText(_ content: Any?) -> String {
        if let s = content as? String { return s }
        guard let parts = content as? [[String: Any]] else { return "" }
        return parts.compactMap { $0["text"] as? String }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private static func toolCalls(_ raw: Any?) -> [AppleToolCallRef] {
        guard let arr = raw as? [[String: Any]] else { return [] }
        return arr.compactMap { tc in
            guard let fn = tc["function"] as? [String: Any],
                  let name = fn["name"] as? String else { return nil }
            return AppleToolCallRef(id: tc["id"] as? String ?? UUID().uuidString,
                                    name: name,
                                    argumentsJSON: fn["arguments"] as? String ?? "{}")
        }
    }

    // MARK: - Tools

    /// Parse the OpenAI tool array we already build for the server.
    static func tools(fromJSON json: String) -> [AppleToolDef] {
        guard let data = json.data(using: .utf8),
              let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
        else { return [] }
        return arr.compactMap { entry in
            guard let fn = entry["function"] as? [String: Any],
                  let name = fn["name"] as? String else { return nil }
            let params = fn["parameters"] as? [String: Any] ?? [:]
            let props = params["properties"] as? [String: Any] ?? [:]
            let required = Set(params["required"] as? [String] ?? [])
            // JSON objects are unordered; the schema's own `required` list and
            // then alphabetical order keep the parameter order STABLE, so the
            // model sees the same tool twice in a row.
            let names = props.keys.sorted { a, b in
                let ra = required.contains(a), rb = required.contains(b)
                return ra == rb ? a < b : ra
            }
            return AppleToolDef(
                name: name,
                description: fn["description"] as? String ?? "",
                properties: names.map { key in
                    let spec = props[key] as? [String: Any] ?? [:]
                    return AppleToolProperty(name: key,
                                             description: spec["description"] as? String,
                                             node: node(spec),
                                             isOptional: !required.contains(key))
                })
        }
    }

    /// An unknown or missing type reads as a string: a parameter the model
    /// cannot fill is worse than one typed loosely.
    private static func node(_ spec: [String: Any]) -> AppleSchemaNode {
        let choices = (spec["enum"] as? [Any])?.compactMap { $0 as? String } ?? []
        switch spec["type"] as? String {
        case "number":  return .number
        case "integer": return .integer
        case "boolean": return .boolean
        case "array":   return .array(node(spec["items"] as? [String: Any] ?? [:]))
        case "object":
            let props = spec["properties"] as? [String: Any] ?? [:]
            let required = Set(spec["required"] as? [String] ?? [])
            return .object(props.keys.sorted().map { key in
                AppleToolProperty(name: key,
                                  description: (props[key] as? [String: Any])?["description"] as? String,
                                  node: node(props[key] as? [String: Any] ?? [:]),
                                  isOptional: !required.contains(key))
            })
        default: return .string(choices: choices)
        }
    }

    // MARK: - Streaming

    /// The framework streams CUMULATIVE snapshots; our events are deltas. A
    /// snapshot that is not an extension of the last one is emitted whole
    /// rather than diffed into nonsense.
    static func delta(previous: String, cumulative: String) -> String {
        guard cumulative.hasPrefix(previous) else { return cumulative }
        return String(cumulative.dropFirst(previous.count))
    }
}
