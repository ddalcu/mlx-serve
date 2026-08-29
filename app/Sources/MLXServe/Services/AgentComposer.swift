import Foundation

/// One-shot completions against whichever model is currently answering chat.
/// Used by the Agents window to turn a description into a system prompt.
///
/// It exists because every other generation path in the app streams into a
/// visible chat bubble, and writing an agent's prompt must not create a
/// conversation. It routes exactly the way `ChatTurnEngine`'s plain path does —
/// same `APIClient`, same port, same `chatModelId`, same headless hot-load — so
/// an agent is written by the model that will run it.
@MainActor
enum AgentComposer {

    enum ComposerError: LocalizedError {
        case noModel

        var errorDescription: String? {
            switch self {
            case .noModel:
                return "No chat model is running yet — start the server (or pick a model) and try again."
            }
        }
    }

    /// Run one prompt to completion and return the whole reply.
    static func complete(userText: String, systemPrompt: String,
                         appState: AppState, maxTokens: Int = 512) async throws -> String {
        var reply = ""
        for try await delta in try await stream(userText: userText, systemPrompt: systemPrompt,
                                                appState: appState, maxTokens: maxTokens) {
            reply += delta
        }
        return reply
    }

    /// Same request, content deltas as they arrive.
    static func stream(userText: String, systemPrompt: String,
                       appState: AppState, maxTokens: Int = 512) async throws -> AsyncThrowingStream<String, Error> {
        guard appState.server.status == .running else { throw ComposerError.noModel }
        await appState.server.ensureDefaultChatModel(selectedModelPath: appState.selectedModelPath)

        let messages: [[String: Any]] = [
            ["role": "system", "content": systemPrompt],
            ["role": "user", "content": userText],
        ]
        let stream = APIClient().streamChat(
            port: appState.server.port,
            messages: messages,
            maxTokens: maxTokens,
            temperature: 0.7,
            defaults: APIClient.RequestDefaults.from(appState.serverOptions),
            modelId: appState.server.chatModelId)

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in stream {
                        if case .content(let delta) = event { continuation.yield(delta) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Describe an agent, get back a name and a system prompt.
    ///
    /// Never throws for a model that answered badly: an unparseable reply falls
    /// back to the user's own words, because losing what they typed to a small
    /// model's bad day is the worse outcome. It DOES throw when there's no model
    /// at all, so the window can say so.
    static func draftAgent(brief: String, appState: AppState) async throws -> AgentWriter.Draft {
        let reply = try await complete(userText: AgentWriter.request(brief: brief),
                                      systemPrompt: AgentWriter.instructions,
                                      appState: appState)
        let draft = AgentWriter.parse(reply, brief: brief) ?? AgentWriter.fallbackDraft(brief: brief)
        // AI-written prompts carry a length instruction — the model's own when it
        // wrote one, ours appended otherwise. A prompt the user types (or edits)
        // is never touched, and there's no setting to find: the line is right
        // there in the editor.
        return AgentWriter.concise(draft)
    }
}
