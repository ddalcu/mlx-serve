import Foundation
import Combine

/// Bridges a Telegram bot to the local model. Long-polls Telegram for incoming
/// messages, runs each through the app's *canonical* chat/agent engine against a
/// hidden per-chat session, then sends the reply back. Communication is
/// outbound HTTPS only (Telegram long-poll), so it works behind home NAT with no
/// public URL, tunnel, or port-forward.
///
/// Lifecycle is driven by `ServerOptions.telegram` through `AppState`:
/// `reconcile()` starts / stops / restarts the poll loop when the token or the
/// enabled flag changes. `agentMode`, `enableThinking`, and the allow-list are
/// read live per message, so changing them needs no restart.
///
/// Reuse, not reimplementation: a *dedicated* `ChatTurnEngine` instance runs the
/// same loop the chat window and voice assistant use (tools, MCP-off, memory,
/// thinking, truncation/retry recovery). Using a separate instance means a
/// phone-driven turn never cancels — or is cancelled by — the user's in-app
/// turn; both still serialize on the one model server-side, which is correct.
@MainActor
final class TelegramBridge: ObservableObject {
    enum Status: Equatable {
        case off
        case connecting
        case listening(username: String?)
        case error(String)

        /// Short human label for the Settings status pill.
        var label: String {
            switch self {
            case .off: return "Off"
            case .connecting: return "Connecting…"
            case .listening(let u): return u.map { "Listening as @\($0)" } ?? "Listening"
            case .error(let m): return m
            }
        }

        var isHealthy: Bool { if case .listening = self { return true }; return false }
    }

    @Published private(set) var status: Status = .off
    /// True while a Telegram-driven turn is generating. `TaskScheduler.drain()`
    /// reads this so a createTask spawned from a Telegram turn waits for that turn
    /// to finish instead of running a second engine against the model concurrently.
    @Published private(set) var isProcessing = false

    /// Owning app state. `unowned` because the bridge is a `lazy var` on
    /// `AppState` — it never outlives its owner.
    unowned let appState: AppState

    /// Dedicated engine — isolates Telegram turns from the user's in-app chat
    /// engine (see the type doc).
    private lazy var engine = ChatTurnEngine(appState: appState)

    private var pollTask: Task<Void, Never>?
    /// Debounces `reconcile()`: every Settings keystroke mutates `serverOptions`
    /// (→ didSet → reconcile), so applying immediately would tear down + restart
    /// the poll loop on each character of the bot token — hammering getMe with a
    /// partial token (401 thrash). We coalesce to the last change after a quiet gap.
    private var reconcileDebounce: Task<Void, Never>?
    /// Config the running loop was started with. Lets `reconcile()` no-op on the
    /// unrelated `ServerOptions` mutations that fire on every Settings keystroke.
    private var appliedConfig: ServerOptions.TelegramConfig?
    /// Telegram chat id → hidden session id. In-memory only: conversations reset
    /// when the app restarts (they live on the phone, not the chat sidebar).
    private var sessions: [Int64: UUID] = [:]

    /// Telegram long-poll hold, seconds. The URLSession request timeout must
    /// comfortably exceed this.
    private let pollTimeout = 25
    private let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 60
        c.timeoutIntervalForResource = 120
        c.waitsForConnectivity = true
        return URLSession(configuration: c)
    }()

    /// Sandbox the Telegram agent's file tools land in (agent mode only).
    static let agentWorkspace: String = {
        let p = NSString(string: "~/.mlx-serve/telegram-workspace").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: p, withIntermediateDirectories: true)
        return p
    }()

    init(appState: AppState) { self.appState = appState }

    // MARK: - Lifecycle

    /// Start / stop / restart the poll loop to match the current config. Cheap
    /// no-op when the fields that affect the connection (token, enabled) are
    /// unchanged — everything else is read live per message.
    func reconcile() {
        // Debounce: rapid serverOptions mutations (typing/pasting the token) each
        // fire didSet → reconcile; coalesce to the last change after a quiet gap so
        // the poll loop isn't torn down + restarted per keystroke against a partial
        // token (which would 401-thrash and write the prefs file every character).
        reconcileDebounce?.cancel()
        reconcileDebounce = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 800_000_000)
            guard !Task.isCancelled, let self else { return }
            self.reconcileDebounce = nil
            self.applyReconcile()
        }
    }

    private func applyReconcile() {
        let cfg = appState.serverOptions.telegram

        // Already running with the same connection params → nothing to restart.
        if cfg.isRunnable, pollTask != nil, let applied = appliedConfig,
           applied.enabled == cfg.enabled, applied.trimmedToken == cfg.trimmedToken {
            appliedConfig = cfg
            return
        }

        stop()
        appliedConfig = cfg
        guard cfg.isRunnable else { status = .off; return }
        status = .connecting
        let token = cfg.trimmedToken
        pollTask = Task { [weak self] in await self?.runLoop(token: token) }
    }

    func stop() {
        reconcileDebounce?.cancel()
        reconcileDebounce = nil
        pollTask?.cancel()
        pollTask = nil
        status = .off
    }

    // MARK: - Poll loop

    private func runLoop(token: String) async {
        // Best-effort getMe so the Settings pill can show "Listening as @bot".
        let username = await fetchUsername(token: token)
        if Task.isCancelled { return }
        status = .listening(username: username)

        // Resume from the last persisted offset so a restart doesn't re-fetch
        // (and re-answer) updates Telegram hadn't yet had confirmed.
        var offset: Int64 = Self.loadOffset(token: token)
        var backoffSeconds = 1
        while !Task.isCancelled {
            guard let url = TelegramAPI.getUpdatesURL(token: token, offset: offset, timeout: pollTimeout) else {
                status = .error("Malformed bot token.")
                return
            }
            do {
                let (data, response) = try await session.data(from: url)
                let httpStatus = (response as? HTTPURLResponse)?.statusCode ?? 200
                if httpStatus == 401 {
                    status = .error("Invalid bot token (401) — re-check it in @BotFather.")
                    return
                }
                if httpStatus != 200 {
                    // 429 / 409 / 5xx return instantly (no long-poll hold). Falling
                    // through would parse {"ok":false} → no offset advance → tight
                    // busy-loop hammering the API. Back off and retry instead.
                    status = .error("Telegram API \(httpStatus) — retrying…")
                    try? await Task.sleep(nanoseconds: UInt64(backoffSeconds) * 1_000_000_000)
                    backoffSeconds = min(backoffSeconds * 2, 30)
                    continue
                }
                // Recovered from a prior transient error.
                if !status.isHealthy { status = .listening(username: username) }
                backoffSeconds = 1

                let (updates, nextOffset) = TelegramAPI.parseUpdates(data)
                if let nextOffset { offset = nextOffset }
                for update in updates {
                    if Task.isCancelled { return }
                    await handle(update, token: token)
                }
                // Persist AFTER handling: a crash mid-batch leaves the offset on the
                // prior value so nothing is lost (at worst a handled item replays),
                // while a clean restart resumes past the batch — no duplicate replies.
                if let nextOffset { Self.saveOffset(nextOffset, token: token) }
            } catch {
                if Task.isCancelled { return }
                // Transient network issue — surface briefly, back off, retry.
                status = .error("Reconnecting…")
                try? await Task.sleep(nanoseconds: UInt64(backoffSeconds) * 1_000_000_000)
                backoffSeconds = min(backoffSeconds * 2, 30)
            }
        }
    }

    // Persisted poll offset, keyed by the bot id (the non-secret part of the token
    // before ":") so changing tokens doesn't resume at a stale offset.
    private static func offsetKey(token: String) -> String {
        let botId = token.split(separator: ":").first.map(String.init) ?? "default"
        return "telegramPollOffset.\(botId)"
    }
    private static func loadOffset(token: String) -> Int64 {
        Int64(UserDefaults.standard.integer(forKey: offsetKey(token: token)))
    }
    private static func saveOffset(_ offset: Int64, token: String) {
        UserDefaults.standard.set(offset, forKey: offsetKey(token: token))
    }

    // MARK: - Per-message handling

    private func handle(_ update: TelegramUpdate, token: String) async {
        let text = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let chatId = update.chatId
        // Either text or a media attachment is needed to act (parse already
        // dropped empty/unsupported messages, but be defensive).
        guard !text.isEmpty || update.attachment != nil else { return }

        // Access gate — read live so allow-list edits in Settings take effect at
        // once. This is the bot's only protection against a stranger driving the
        // local model, so it runs before anything else. Applies to attachment-
        // only messages too (a photo is enough to adopt a first chat).
        switch appState.serverOptions.telegram.access(forChatId: chatId) {
        case .rejected:
            await send(token: token, chatId: chatId, text: "⛔️ This bot is locked to another chat.")
            return
        case .adopt:
            // Trust-on-first-use: the first chat to message becomes the owner.
            // Mutating serverOptions persists it (AppState.didSet) and harmlessly
            // re-enters reconcile() as a no-op (token/enabled unchanged).
            appState.serverOptions.telegram.allowedChatIds.append(chatId)
            await send(token: token, chatId: chatId,
                       text: "✅ Locked to this chat. I'll relay your messages to the local model on \(hostName()). Send /new anytime to start a fresh conversation.")
            // Fall through and answer this first message (unless it's a command).
        case .allowed:
            break
        }

        // Commands — only a plain text message (no attachment) is a command.
        if update.attachment == nil {
            switch text {
            case "/start":
                await send(token: token, chatId: chatId, text: startHelp())
                return
            case "/new", "/reset":
                // Reap the old session's background processes before dropping the
                // mapping — otherwise the orphaned ChatSession's processes survive
                // untracked (no chat card, no /new can ever reach them again).
                if let oldSessionId = sessions[chatId] {
                    appState.processRegistry.killSession(oldSessionId)
                }
                sessions[chatId] = nil
                await send(token: token, chatId: chatId, text: "🧹 Started a new conversation.")
                return
            default:
                break
            }
        }

        guard appState.server.status == .running else {
            await send(token: token, chatId: chatId,
                       text: "⚠️ No model is loaded right now. Open MLX Core, start a model, then message me again.")
            return
        }

        // Route on the attachment kind and the LOADED model's live capabilities
        // (`supportsVision`/`supportsAudio` drop out under `--no-vision` / on a
        // text-only model, which is exactly what gates the refusal messages).
        let info = appState.server.modelInfo
        let action = TelegramAPI.decideAttachmentAction(update,
            supportsVision: info?.supportsVision ?? false,
            supportsAudio: info?.supportsAudio ?? false)

        switch action {
        case .imageUnsupported:
            await send(token: token, chatId: chatId,
                       text: "🚫 This model can't see images. Load a vision model (e.g. Gemma 4) in MLX Core and try again.")
        case .unsupported(let reason):
            await send(token: token, chatId: chatId, text: reason)
        case .textOnly, .image, .audio:
            // Everything that generates a reply runs under the "typing…"
            // indicator for the whole download → decode → generate span.
            await withTyping(token: token, chatId: chatId) {
                await self.handleGenerating(action: action, update: update, token: token, text: text)
            }
        }
    }

    /// Download / decode / transcribe an attachment as needed, run the turn, and
    /// send the reply. Called only for the generating actions, inside `withTyping`.
    private func handleGenerating(action: TelegramAttachmentAction, update: TelegramUpdate,
                                  token: String, text: String) async {
        let chatId = update.chatId
        switch action {
        case .textOnly:
            let reply = await generateReply(chatId: chatId, senderName: update.senderName, text: text)
            await sendReply(token: token, chatId: chatId, reply: reply)

        case .image(let fileId):
            guard let bytes = await download(token: token, fileId: fileId) else {
                await send(token: token, chatId: chatId, text: "⚠️ I couldn't download that image. Try sending it again.")
                return
            }
            let reply = await generateReply(chatId: chatId, senderName: update.senderName,
                                            text: text, images: [ChatImage(data: bytes)])
            await sendReply(token: token, chatId: chatId, reply: reply)

        case .audio(let fileId, let transcribe):
            guard let bytes = await download(token: token, fileId: fileId) else {
                await send(token: token, chatId: chatId, text: "⚠️ I couldn't download that voice clip. Try sending it again.")
                return
            }
            guard let decoded = VoicePreprocessor.decode(
                bytes,
                oggOpus: Self.attachmentIsOggOpus(update.attachment),
                sourceExtension: Self.attachmentAudioExtension(update.attachment)
            ) else {
                await send(token: token, chatId: chatId, text: "⚠️ I couldn't decode that voice clip.")
                return
            }
            defer { try? FileManager.default.removeItem(at: decoded.fileURL) }

            if transcribe {
                // Model can't hear — transcribe on-device and send the words.
                switch await VoiceTranscriber.transcribe(fileURL: decoded.fileURL) {
                case .success(let transcript):
                    let combined = text.isEmpty ? "[voice] \(transcript)" : "\(text)\n\n[voice] \(transcript)"
                    let reply = await generateReply(chatId: chatId, senderName: update.senderName, text: combined)
                    await sendReply(token: token, chatId: chatId, reply: reply)
                case .failure(let failure):
                    await send(token: token, chatId: chatId, text: Self.transcriptionMessage(for: failure))
                }
            } else {
                // Model is audio-capable — feed the decoded PCM so it "hears" it.
                let clip = ChatAudio(name: "voice.m4a", pcm: decoded.pcm)
                let reply = await generateReply(chatId: chatId, senderName: update.senderName,
                                                text: text, audio: [clip])
                await sendReply(token: token, chatId: chatId, reply: reply)
            }

        case .imageUnsupported, .unsupported:
            break   // handled by the caller before reaching here
        }
    }

    private func sendReply(token: String, chatId: Int64, reply: String) async {
        for chunk in TelegramAPI.splitForTelegram(reply) {
            await send(token: token, chatId: chatId, text: chunk)
        }
    }

    /// Run one turn through the canonical engine against the chat's hidden
    /// session and return the final assistant text. `images`/`audio` are the
    /// decoded attachments (when the loaded model can consume them).
    private func generateReply(chatId: Int64, senderName: String, text: String,
                               images: [ChatImage]? = nil, audio: [ChatAudio]? = nil) async -> String {
        let cfg = appState.serverOptions.telegram
        let sessionId = sessionId(for: chatId, senderName: senderName, agentMode: cfg.agentMode)
        let workspace = Self.agentWorkspace
        let turnConfig = ChatTurnEngine.TurnConfig(
            agentMode: cfg.agentMode,
            mcpMode: cfg.useMCP,
            enableThinking: cfg.enableThinking,
            voiceStyle: false,
            workingDirectory: (cfg.agentMode || cfg.useMCP) ? workspace : nil,
            telegramChatId: chatId   // so the agent's createTask reports back here
        )

        engine.runTurn(
            sessionId: sessionId,
            userText: text,
            images: images,
            audio: audio,
            config: turnConfig,
            approval: { tc in
                // Agent-over-Telegram has no interactive approval surface, so
                // reuse the pure, tested ApprovalPolicy at fullAuto: read-only +
                // shell + workspace-confined writes auto-allow; out-of-workspace
                // writes and unknown tools are denied (the loop adapts). The
                // allow-list is the real gate keeping strangers out.
                ApprovalPolicy.decide(
                    tool: tc.name, autonomy: .fullAuto,
                    arguments: tc.arguments, rawArguments: tc.rawArguments,
                    workingDirectory: workspace
                ) == .allow
            }
        )
        isProcessing = true
        defer { isProcessing = false }
        await awaitEngineIdle()
        return lastAssistantText(sessionId: sessionId)
    }

    // MARK: - Session bookkeeping

    private func sessionId(for chatId: Int64, senderName: String, agentMode: Bool) -> UUID {
        if let existing = sessions[chatId],
           appState.chatSessions.contains(where: { $0.id == existing }) {
            return existing
        }
        var s = ChatSession(title: "\(senderName) (Telegram)")
        s.isExternalBridge = true
        if agentMode {
            s.mode = .agent
            s.workingDirectory = Self.agentWorkspace
        }
        appState.chatSessions.append(s)   // hidden from the sidebar; never persisted
        sessions[chatId] = s.id
        return s.id
    }

    /// Await the dedicated engine returning to idle. `runTurn` sets
    /// `isGenerating = true` synchronously when it starts, so by the time we
    /// subscribe the value is either still `true` (wait for the `false`
    /// transition) or already `false` (turn finished — return at once).
    private func awaitEngineIdle() async {
        for await generating in engine.$isGenerating.values {
            if !generating { return }
        }
    }

    private func lastAssistantText(sessionId: UUID) -> String {
        let msgs = appState.chatSessions.first { $0.id == sessionId }?.messages ?? []
        let content = msgs.last {
            $0.role == .assistant && !$0.isAgentSummary && !$0.failedRetry
        }?.content ?? ""
        let cleaned = content
            .replacingOccurrences(of: "<pad>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty
            ? "⚠️ I couldn't generate a response. Try /new and rephrase."
            : cleaned
    }

    // MARK: - Task result delivery (called by TaskScheduler.finalize)

    /// Push a finished task's result to the Telegram chat that created it (via the
    /// agent's `createTask` tool). Runs ALONGSIDE the desktop notification, not
    /// instead of it. Best-effort and independent of the poll loop — it sends as
    /// long as a token is configured, so a scheduled task still reports even if
    /// the bridge happens to be toggled off.
    func deliverTaskResult(chatId: Int64, task: ScheduledTask, run: TaskRun) {
        let token = appState.serverOptions.telegram.trimmedToken
        guard !token.isEmpty else { return }
        // Send the FULL final answer, not `run.summary` — that field is capped at
        // 280 chars for the in-app timeline row, so relaying it is exactly the
        // truncation users see on the phone. `splitForTelegram` chunks the rest.
        let full = TaskScheduler.fullLastAssistantText(
            appState.taskScheduler.transcript(taskId: task.id, runId: run.id))
        let text = Self.taskResultText(title: task.title, completed: run.status == .completed,
                                       body: full ?? run.summary)
        Task { [weak self] in
            guard let self else { return }
            for chunk in TelegramAPI.splitForTelegram(text) {
                await self.send(token: token, chatId: chatId, text: chunk)
            }
        }
    }

    /// Format a finished task's report: status header + the full result body
    /// (when there is one). Pure/testable; the caller chunks it for Telegram.
    nonisolated static func taskResultText(title: String, completed: Bool, body: String?) -> String {
        let header = completed ? "✅ Task “\(title)” finished" : "⚠️ Task “\(title)” failed"
        return header + (body.map { "\n\n\($0)" } ?? "")
    }

    // MARK: - Telegram I/O

    @discardableResult
    private func send(token: String, chatId: Int64, text: String) async -> Bool {
        guard let url = TelegramAPI.sendMessageURL(token: token) else { return false }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = TelegramAPI.sendMessageBody(chatId: chatId, text: text)
        do {
            let (_, response) = try await session.data(for: req)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func fetchUsername(token: String) async -> String? {
        guard let url = TelegramAPI.getMeURL(token: token),
              let (data, _) = try? await session.data(from: url) else { return nil }
        return TelegramAPI.parseUsername(data)
    }

    /// POST a single `sendChatAction` (typing). Best-effort — failures are
    /// invisible to the user (the indicator just doesn't show), so we ignore them.
    private func sendChatAction(token: String, chatId: Int64) async {
        guard let url = TelegramAPI.sendChatActionURL(token: token) else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = TelegramAPI.sendChatActionBody(chatId: chatId)
        _ = try? await session.data(for: req)
    }

    /// Show Telegram's "typing…" indicator for the whole duration of `body`.
    /// The action expires after ~5 s, so a side task re-posts it every 4 s and
    /// is cancelled the instant `body` returns.
    private func withTyping<T>(token: String, chatId: Int64, _ body: () async -> T) async -> T {
        let typing = Task { [weak self] in
            while !Task.isCancelled {
                await self?.sendChatAction(token: token, chatId: chatId)
                try? await Task.sleep(nanoseconds: 4_000_000_000)
            }
        }
        defer { typing.cancel() }
        return await body()
    }

    /// Resolve a `file_id` via `getFile` and download its bytes. Caps at
    /// Telegram's 20 MB bot-download limit; returns nil on any failure.
    private func download(token: String, fileId: String,
                          maxBytes: Int = 20 * 1024 * 1024) async -> Data? {
        guard let metaURL = TelegramAPI.getFileURL(token: token, fileId: fileId),
              let (metaData, metaResp) = try? await session.data(from: metaURL),
              (metaResp as? HTTPURLResponse)?.statusCode == 200,
              let filePath = TelegramAPI.parseFilePath(metaData),
              let dlURL = TelegramAPI.fileDownloadURL(token: token, filePath: filePath),
              let (bytes, dlResp) = try? await session.data(from: dlURL),
              (dlResp as? HTTPURLResponse)?.statusCode == 200,
              !bytes.isEmpty, bytes.count <= maxBytes
        else { return nil }
        return bytes
    }

    // MARK: - Attachment helpers (pure)

    /// Whether an attachment is Ogg/Opus (needs SwiftOGG) vs. an AVFoundation-
    /// readable container. Telegram voice notes are always Ogg/Opus.
    private static func attachmentIsOggOpus(_ attachment: TelegramUpdate.Attachment?) -> Bool {
        switch attachment {
        case .voice:
            return true
        case .document(_, let mime, let name):
            let m = mime.lowercased(), n = name.lowercased()
            return m.contains("ogg") || m.contains("opus") || n.hasSuffix(".oga") || n.hasSuffix(".ogg")
        case .audio, .photo, .none:
            return false   // `audio` (music) is virtually always mp3/m4a
        }
    }

    /// File extension hint for a non-Ogg audio attachment, so the temp file we
    /// hand AVFoundation is named sensibly (`m4a` when unknown).
    private static func attachmentAudioExtension(_ attachment: TelegramUpdate.Attachment?) -> String {
        if case .document(_, _, let name) = attachment {
            let ext = (name as NSString).pathExtension
            if !ext.isEmpty { return ext }
        }
        return "m4a"
    }

    private static func transcriptionMessage(for failure: VoiceTranscriber.Failure) -> String {
        switch failure {
        case .unavailable(let message): return "🎙️ \(message)"
        case .noSpeech: return "🎙️ I couldn't make out any speech in that clip. Try recording again."
        case .failed(let message): return "🎙️ I couldn't transcribe that voice clip: \(message)"
        }
    }

    // MARK: - Copy

    private func startHelp() -> String {
        let mode = appState.serverOptions.telegram.agentMode
            ? "agent mode (it can run shell commands and edit files on the Mac)"
            : "chat mode"
        return """
        👋 Connected to MLX Core in \(mode).
        Send a message and I'll relay it to your local model.
        You can also send 📷 photos (vision models will look at them) and 🎙️ voice notes (audio models hear them; other models read an on-device transcript).
        Commands:
        /new — start a fresh conversation
        """
    }

    private func hostName() -> String { Host.current().localizedName ?? "this Mac" }
}
