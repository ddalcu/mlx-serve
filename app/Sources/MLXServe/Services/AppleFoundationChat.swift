import Foundation
import FoundationModels

/// Apple's on-device model as a chat source, answering the same
/// `AsyncThrowingStream<SSEEvent, Error>` the server does — so the turn engine,
/// the agent loop, approvals and the transcript are the ONE path for both.
///
/// Nothing is loaded, downloaded or served: the model is the system's, so this
/// route needs no mlx-serve process at all.
enum AppleFoundationChat {

    /// What the picker calls it, and the tag `ChatModelSelection` round-trips.
    static let displayName = "Apple Intelligence"

    /// The on-device model's context window: 4096 tokens shared between the
    /// prompt and the reply, fixed by the system. There is no API to raise it,
    /// so everything we send has to be budgeted against THIS, not the app's
    /// context setting.
    static let contextTokens = 4096
    /// What the reply may take out of that window. The rest is prompt: system
    /// text, tool definitions and history.
    static let maxResponseTokens = 1024

    /// The only tools this route advertises. The full definition blob is
    /// roughly 3.8k tokens against a 4096-token window, so "every tool" is not
    /// a thing that can fit; these two are the ones worth the room, and the
    /// model has no other way to reach past its training data.
    static let allowedTools: Set<AgentToolKind> = [.browse, .webSearch]

    static func responseTokens(requested: Int) -> Int {
        max(1, min(requested, maxResponseTokens))
    }

    enum Availability: Equatable {
        case available
        /// Why it cannot answer, in the user's terms.
        case unavailable(String)

        var isAvailable: Bool { self == .available }
    }

    static var availability: Availability {
        switch SystemLanguageModel.default.availability {
        case .available: return .available
        case .unavailable(let reason):
            switch reason {
            case .appleIntelligenceNotEnabled:
                return .unavailable("Apple Intelligence is off — turn it on in System Settings")
            case .modelNotReady:
                return .unavailable("Apple Intelligence is still downloading its model")
            case .deviceNotEligible:
                return .unavailable("This Mac does not support Apple Intelligence")
            @unknown default:
                return .unavailable("Apple Intelligence is unavailable")
            }
        }
    }

    /// A tool the model asked for. The framework runs tools itself; we take
    /// the call OUT instead (`ToolCallIntercepted`) so it rides the same agent
    /// loop as every other model's — approvals, logging and repetition guards
    /// included.
    private struct ToolCallIntercepted: Error {}

    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [APIClient.ToolCall] = []
        func record(_ call: APIClient.ToolCall) {
            lock.lock(); defer { lock.unlock() }
            calls.append(call)
        }
        var drained: [APIClient.ToolCall] {
            lock.lock(); defer { lock.unlock() }
            return calls
        }
    }

    private struct BridgedTool: Tool {
        typealias Arguments = GeneratedContent
        typealias Output = String
        let name: String
        let description: String
        let parameters: GenerationSchema
        let recorder: Recorder

        func call(arguments: GeneratedContent) async throws -> String {
            recorder.record(APIClient.ToolCall(
                id: "apple-\(UUID().uuidString.prefix(8))",
                name: name,
                arguments: AppleFoundationChat.stringArgs(arguments),
                rawArguments: arguments.jsonString))
            throw ToolCallIntercepted()
        }
    }

    /// The engine's tool executor takes flat string arguments; anything that
    /// is not a scalar rides as its JSON text.
    private static func stringArgs(_ content: GeneratedContent) -> [String: String] {
        guard case .structure(let props, let order) = content.kind else { return [:] }
        var out: [String: String] = [:]
        for key in (order.isEmpty ? Array(props.keys) : order) {
            guard let value = props[key] else { continue }
            switch value.kind {
            case .string(let s): out[key] = s
            case .bool(let b):   out[key] = b ? "true" : "false"
            case .number(let n): out[key] = n == n.rounded() ? String(Int(n)) : String(n)
            default:             out[key] = value.jsonString
            }
        }
        return out
    }

    // MARK: - Streaming

    static func stream(messages: [[String: Any]],
                       toolsJSON: String? = nil,
                       temperature: Double? = nil,
                       maxTokens: Int? = nil) -> AsyncThrowingStream<SSEEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if case .unavailable(let why) = availability {
                        throw AppleFoundationError.unavailable(why)
                    }
                    let plan = AppleFoundationBridge.plan(messages: messages)
                    let recorder = Recorder()
                    let tools = buildTools(toolsJSON, recorder: recorder)
                    // BOTH halves, always: the transcript carries the tool
                    // DEFINITIONS (what the model may call) and the session
                    // carries the IMPLEMENTATIONS (what dispatch finds).
                    // Definitions alone and the model answers "I can't do
                    // that" without ever calling anything — measured.
                    let session = LanguageModelSession(
                        tools: tools,
                        transcript: transcript(plan.entries, tools: tools))
                    let options = GenerationOptions(
                        temperature: temperature,
                        maximumResponseTokens: responseTokens(requested: maxTokens ?? maxResponseTokens))

                    var emitted = ""
                    do {
                        let responses = session.streamResponse(to: plan.prompt, options: options)
                        for try await partial in responses {
                            try Task.checkCancellation()
                            let text = partial.content
                            let delta = AppleFoundationBridge.delta(previous: emitted, cumulative: text)
                            emitted = text
                            if !delta.isEmpty { continuation.yield(.content(delta)) }
                        }
                    } catch {
                        // A tool call aborts generation by design; anything
                        // else with no recorded call is a real failure.
                        if recorder.drained.isEmpty { throw Self.mapped(error) }
                    }
                    let calls = recorder.drained
                    if !calls.isEmpty { continuation.yield(.toolCalls(calls)) }
                    continuation.yield(.done)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func buildTools(_ json: String?, recorder: Recorder) -> [BridgedTool] {
        guard let json, !json.isEmpty else { return [] }
        return AppleFoundationBridge.tools(fromJSON: json).compactMap { def in
            let root = DynamicGenerationSchema(
                name: def.name,
                description: def.description.isEmpty ? nil : def.description,
                properties: def.properties.map(property))
            guard let schema = try? GenerationSchema(root: root, dependencies: []) else { return nil }
            return BridgedTool(name: def.name,
                               description: def.description,
                               parameters: schema,
                               recorder: recorder)
        }
    }

    private static func property(_ p: AppleToolProperty) -> DynamicGenerationSchema.Property {
        DynamicGenerationSchema.Property(name: p.name, description: p.description,
                                        schema: schema(p.node, name: p.name),
                                        isOptional: p.isOptional)
    }

    private static func schema(_ node: AppleSchemaNode, name: String) -> DynamicGenerationSchema {
        switch node {
        case .string(let choices):
            return choices.isEmpty
                ? DynamicGenerationSchema(type: String.self)
                : DynamicGenerationSchema(name: name, anyOf: choices)
        case .number:  return DynamicGenerationSchema(type: Double.self)
        case .integer: return DynamicGenerationSchema(type: Int.self)
        case .boolean: return DynamicGenerationSchema(type: Bool.self)
        case .array(let item):
            return DynamicGenerationSchema(arrayOf: schema(item, name: name + "Item"))
        case .object(let props):
            return DynamicGenerationSchema(name: name, properties: props.map(property))
        }
    }

    private static func transcript(_ entries: [AppleChatEntry],
                                   tools: [BridgedTool]) -> Transcript {
        let definitions = tools.map { Transcript.ToolDefinition(tool: $0) }
        var out: [Transcript.Entry] = []
        var sawInstructions = false
        for entry in entries {
            switch entry {
            case .instructions(let text):
                sawInstructions = true
                out.append(.instructions(.init(segments: [.text(.init(content: text))],
                                               toolDefinitions: definitions)))
            case .prompt(let text):
                out.append(.prompt(.init(segments: [.text(.init(content: text))])))
            case .response(let text):
                out.append(.response(.init(assetIDs: [], segments: [.text(.init(content: text))])))
            case .toolCalls(let calls):
                out.append(.toolCalls(.init(calls.map {
                    Transcript.ToolCall(id: $0.id, toolName: $0.name,
                                        arguments: (try? GeneratedContent(json: $0.argumentsJSON))
                                            ?? GeneratedContent($0.argumentsJSON))
                })))
            case .toolOutput(let id, let name, let text):
                out.append(.toolOutput(.init(id: id, toolName: name,
                                             segments: [.text(.init(content: text))])))
            }
        }
        // Tool definitions ride the instructions entry, so a tool-bearing turn
        // with no system prompt still needs one.
        if !sawInstructions && !definitions.isEmpty {
            out.insert(.instructions(.init(segments: [], toolDefinitions: definitions)), at: 0)
        }
        return Transcript(entries: out)
    }
}

extension AppleFoundationChat {
    /// The window is 4k and cannot be raised, so the overflow has to say what
    /// the user can actually do about it.
    static func mapped(_ error: Error) -> Error {
        guard case LanguageModelSession.GenerationError.exceededContextWindowSize = error else {
            return error
        }
        return AppleFoundationError.contextFull
    }
}

enum AppleFoundationError: LocalizedError {
    case unavailable(String)
    case contextFull
    var errorDescription: String? {
        switch self {
        case .unavailable(let why): return why
        case .contextFull:
            return "Apple Intelligence has a fixed \(AppleFoundationChat.contextTokens)-token window and this turn does not fit. Turn off some tools, shorten the agent's instructions, or start a new chat."
        }
    }
}
