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
        for try await event in events(userText: userText, systemPrompt: systemPrompt,
                                      appState: appState, maxTokens: maxTokens) {
            if case .content(let delta) = event { reply += delta }
        }
        return reply
    }

    /// What a request goes through, as it happens: a slow load or a long think
    /// is normal, and a sheet that shows nothing meanwhile reads as a hang.
    enum Event: Equatable {
        /// The chat model is being hot-loaded first (its name).
        case loading(String)
        /// The request is with the server: reading the prompt, or a cold load there.
        case sent
        case reasoning(String)
        case content(String)
        /// The reply was cut at its token budget.
        case truncated
    }

    /// Whether the model that will answer can see a picture: the resident
    /// chat model's live capabilities, else the picked model's config.
    static func seesImages(appState: AppState) -> Bool {
        appState.server.chatModelInfo?.supportsVision
            ?? appState.localModels.first { $0.path == appState.selectedModelPath }?.hasVision
            ?? false
    }

    /// Same request, every stage as it arrives. `image` rides the user turn.
    static func events(userText: String, systemPrompt: String, image: Data? = nil, appState: AppState,
                       maxTokens: Int = 512) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let server = appState.server
                    guard server.status == .running else { throw ComposerError.noModel }
                    let path = appState.selectedModelPath
                    if server.chatLoadNeeded(selectedModelPath: path) {
                        continuation.yield(.loading((path as NSString).lastPathComponent))
                    }
                    await server.ensureDefaultChatModel(selectedModelPath: path)
                    continuation.yield(.sent)
                    let user: Any = image.map {
                        MultimodalContent.build(text: userText, images: [ChatImage(data: $0)],
                                                serverPreprocess: MultimodalContent.wantsServerPreprocess(
                                                    architecture: server.chatModelInfo?.architecture ?? ""))
                    } ?? userText
                    let messages: [[String: Any]] = [
                        ["role": "system", "content": systemPrompt],
                        ["role": "user", "content": user],
                    ]
                    let stream = APIClient().streamChat(
                        port: server.port,
                        messages: messages,
                        maxTokens: maxTokens,
                        temperature: 0.7,
                        defaults: APIClient.RequestDefaults.from(appState.serverOptions),
                        modelId: server.chatRequestModelId(selectedPath: path))
                    for try await event in stream {
                        switch event {
                        case .content(let delta): continuation.yield(.content(delta))
                        case .reasoning(let delta): continuation.yield(.reasoning(delta))
                        case .truncated: continuation.yield(.truncated)
                        default: break
                        }
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
