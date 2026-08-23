import Foundation
import Combine

/// The single source of truth for running a chat/agent turn. Both the text chat
/// window and the hands-free voice controller submit turns through here, so the
/// two never drift behaviourally. The engine is app-level (`AppState` owns it for
/// the app's lifetime) and window-independent: it appends messages, streams via
/// `APIClient`, runs the tool-calling loop, and writes everything into
/// `AppState.chatSessions`. Views observe `isGenerating` for spinners.
///
/// It conforms to `TurnRunning` so the voice controller can be unit-tested with a
/// fake runner instead of the concrete `APIClient`.
@MainActor
protocol TurnRunning: AnyObject {
    var isGenerating: Bool { get }
    /// Run one user turn against `sessionId`. Cancels any in-flight turn FOR
    /// THAT SESSION first (other sessions' turns are untouched — the engine is
    /// multi-turn). `approval` is invoked before every tool dispatch (agent
    /// mode) — returning false denies the call. Plain chat never calls it.
    func runTurn(sessionId: UUID,
                 userText: String,
                 images: [ChatImage]?,
                 videos: [ChatVideo]?,
                 audio: [ChatAudio]?,
                 config: ChatTurnEngine.TurnConfig,
                 approval: @escaping (APIClient.ToolCall) async -> Bool)
    // (No default for `videos` here — Swift protocol requirements can't carry
    // default argument values; every call site names it explicitly.)
    /// Cancel every in-flight turn (the legacy app-wide stop).
    func stop()
    /// Cancel one session's in-flight turn, leaving the others running.
    func stop(sessionId: UUID)
}

/// A caller that only knows "stop" (test fakes, single-session drivers) gets
/// the app-wide stop for free; the real engine overrides with a scoped one.
extension TurnRunning {
    func stop(sessionId: UUID) { stop() }
}

/// Pure bookkeeping for concurrent per-session turns. Each turn is identified
/// by a TOKEN minted at `begin` — cleanup paths must present it, so a
/// superseded task's async unwind can never clear its successor's slot (the
/// supersede race: cancel old task → begin new turn → old task's cleanup runs
/// later). One turn per session: `begin` on a busy session replaces the entry.
struct TurnLedger {
    struct Turn {
        let token: UUID
        var liveTokens: Int = 0
    }

    private(set) var turns: [UUID: Turn] = [:]

    var isBusy: Bool { !turns.isEmpty }
    var activeSessionIds: Set<UUID> { Set(turns.keys) }

    /// Start (or supersede) a turn for `session`; returns its cleanup token.
    mutating func begin(session: UUID) -> UUID {
        let token = UUID()
        turns[session] = Turn(token: token)
        return token
    }

    /// End the turn only if `token` still owns the session's slot. Returns
    /// false (and touches nothing) when a newer turn superseded this one.
    @discardableResult
    mutating func end(session: UUID, token: UUID) -> Bool {
        guard turns[session]?.token == token else { return false }
        turns.removeValue(forKey: session)
        return true
    }

    /// Unconditionally clear the session's slot (the user-facing Stop).
    mutating func endAll(session: UUID) {
        turns.removeValue(forKey: session)
    }

    mutating func setLiveTokens(_ n: Int, session: UUID) {
        turns[session]?.liveTokens = n
    }

    func liveTokens(session: UUID) -> Int {
        turns[session]?.liveTokens ?? 0
    }

    /// Turns whose session no longer exists (the ghost-turn class): deleting
    /// one chat must stop ONLY that chat's turn.
    func orphaned(existingSessions: Set<UUID>) -> Set<UUID> {
        Set(turns.keys.filter { !existingSessions.contains($0) })
    }
}

/// Per-session memory for the "Allow all tools this session" decision. Keyed by
/// chat-session id so each tab remembers its own choice: the text-chat window
/// reuses one `ChatDetailView` across `sessionId` changes, so a single `Bool`
/// here is shared by every tab and was wiped on every switch. A `Set` keyed by
/// id has nothing to wipe on a switch — switching away and back preserves the
/// grant — and re-arming (Agent toggled off) clears only that one session.
struct SessionToolAllowList {
    private var allowed: Set<UUID> = []
    func allowsAll(_ id: UUID) -> Bool { allowed.contains(id) }
    mutating func allowAll(_ id: UUID) { allowed.insert(id) }
    /// Re-prompt this session on the next tool call (leaves other tabs alone).
    mutating func rearm(_ id: UUID) { allowed.remove(id) }
}

@MainActor
final class ChatTurnEngine: ObservableObject, TurnRunning {
    /// Owning app state. `unowned` because the engine lives exactly as long as
    /// `AppState` does (it's a `lazy var` on it) — there is never a window where
    /// the engine outlives its owner.
    unowned let appState: AppState

    /// True while ANY session's turn is in flight. Kept as the coarse signal
    /// for surfaces that only care about GPU business (TaskScheduler's defer
    /// gate, TelegramBridge's own-instance wait); per-chat UI reads
    /// `composerState(for:)` / `activeTurnSessionIds` instead.
    @Published private(set) var isGenerating = false

    /// The sessions with a turn in flight — the per-chat composer re-renders
    /// off this, and the voice controller scopes its end-of-turn detection to
    /// its own session with it.
    @Published private(set) var activeTurnSessionIds: Set<UUID> = []

    /// Live count of tokens each in-flight reply has produced — for the chat
    /// composer's live "gen:" readout and growing context bar. Counted by tallying
    /// streamed `.content`/`.reasoning` deltas (one per token for this server) and
    /// PUBLISHED only at the StreamCoalescer's ~20 Hz flush cadence, never per
    /// token — per-token @Published churn is exactly what StreamCoalescer exists to
    /// avoid. Reset at the start of each streamed round; reconciled to the
    /// authoritative `usage.completion_tokens` when the stream reports usage.
    @Published private(set) var liveTokensBySession: [UUID: Int] = [:]

    func liveCompletionTokens(for sessionId: UUID) -> Int {
        liveTokensBySession[sessionId] ?? 0
    }

    /// The turn table: token-identified turn per session (see `TurnLedger`)
    /// plus the driving Task, cancelled by `stop(sessionId:)`.
    private var ledger = TurnLedger()
    private var tasks: [UUID: Task<Void, Never>] = [:]

    /// The in-flight media generation, or nil. ONE value is still right — the
    /// inference thread serializes media gen, so two never RUN at once — but
    /// with concurrent turns it needs an owner, so only the chat that asked
    /// renders the card. Published because these block chat decode for anything
    /// from six seconds (an image) to minutes (a clip) — a window with no
    /// feedback for that long reads as a hang.
    @Published private(set) var mediaProgress: MediaGenProgress?
    @Published private(set) var mediaProgressSessionId: UUID?

    /// ONE media generation per user turn, keyed by the turn token its driver
    /// passes. See `MediaTurnBudget` for why it's a token and not a reset call.
    private var mediaBudget = MediaTurnBudget()

    /// What the composer's primary button should show for a given chat. The
    /// engine runs one turn per session, so the only states are "this chat is
    /// answering" and "free to send" — another chat's turn blocks nothing.
    enum ComposerState: Equatable {
        case idle             // free to send (subject to having content)
        case generatingHere   // this chat owns an in-flight turn → show Stop
    }

    /// Pure decision for `ComposerState`; the instance accessor below feeds it
    /// the live engine state. `nonisolated` because it touches no actor state —
    /// just its arguments — so views and tests can call it freely.
    nonisolated static func composerState(activeTurnSessionIds: Set<UUID>,
                                          for sessionId: UUID) -> ComposerState {
        activeTurnSessionIds.contains(sessionId) ? .generatingHere : .idle
    }

    func composerState(for sessionId: UUID) -> ComposerState {
        Self.composerState(activeTurnSessionIds: activeTurnSessionIds, for: sessionId)
    }

    /// Re-derive the published mirrors from the ledger. Called after every
    /// ledger mutation; cheap (set compare) and keeps view updates minimal.
    private func publishTurnState() {
        let ids = ledger.activeSessionIds
        if activeTurnSessionIds != ids { activeTurnSessionIds = ids }
        let busy = ledger.isBusy
        if isGenerating != busy { isGenerating = busy }
    }

    /// A turn whose session no longer exists is a GHOST: every append/update
    /// no-ops, the empty-response check reads "" from the missing session and
    /// pad-retries with full generations, and the slot stays busy forever —
    /// no server restart can clear it (the turn is app-side). Live capture
    /// 2026-07-03: a deleted agent chat kept a 27B model generating for 10+
    /// minutes across a server restart. Multi-turn corollary: deleting one
    /// chat stops ONLY that chat's turn (`TurnLedger.orphaned`). Called by
    /// `AppState.deleteSession` right after removal (the only runtime
    /// session-removal site); the agent loop also re-checks per iteration as
    /// defense in depth for any future removal path.
    func stopIfOrphaned() {
        let existing = Set(appState.chatSessions.map(\.id))
        for sid in ledger.orphaned(existingSessions: existing) {
            stop(sessionId: sid)
        }
    }

    init(appState: AppState) {
        self.appState = appState
    }

    /// Per-turn configuration. Every field is DECIDED before it gets here —
    /// build it with `TurnConfig.from(_:)` out of a `ResolvedAgentSettings`
    /// rather than reading globals at the call site. Five surfaces start turns
    /// (chat tab, voice tray, scheduled task, Telegram, Quick Launcher) and a
    /// per-surface read is exactly how one of them ends up silently ignoring the
    /// active agent.
    struct TurnConfig {
        var agentMode: Bool
        var mcpMode: Bool
        var enableThinking: Bool
        var voiceStyle: Bool
        var workingDirectory: String?
        /// Per-session document index (mini RAG). Non-nil while the user has a
        /// folder attached — advertises the searchDocuments tool and, with both
        /// Agent and MCP off, routes the turn through the loop in docs-only mode.
        var documentIndex: DocumentIndex? = nil
        /// Set when this turn is driven by the Telegram bridge — the originating
        /// chat id. Threaded into any `createTask` the agent makes so the task's
        /// result is pushed back to that chat. nil for in-app / voice turns.
        var telegramChatId: Int64? = nil

        // MARK: The agent (persona) driving this turn

        /// nil = no agent, i.e. today's behavior in every respect.
        var agentId: UUID? = nil
        /// The persona, prepended to the STABLE system-prompt prefix. "" when
        /// there's no agent.
        var systemPromptPrefix: String = ""
        /// The tools this turn may advertise AND dispatch. Defaults to
        /// everything — full access is what every surface had before agents.
        var tools: Set<AgentToolKind> = Set(AgentToolKind.allCases)
        /// Skip the approval gate for this turn (an agent that declared it).
        var autoApprove: Bool = false
        /// Sampling overrides; nil = the path's own default (the tool loop and
        /// plain chat do NOT share one).
        var temperature: Double? = nil
        var maxTokens: Int? = nil
        /// The rest of the sampling surface; nil = the user's saved default
        /// (`ServerOptions`). Applied by `requestDefaults(from:)`.
        var topP: Double? = nil
        var topK: Int? = nil
        var repeatPenalty: Double? = nil
        var presencePenalty: Double? = nil
        var reasoningBudget: Int? = nil
        /// The surface's `reasoning_effort` pick, sent only while thinking is
        /// on (see `reasoningEffortParam`).
        var reasoningEffort: ReasoningEffort = .low
        /// The agent's own voice for this turn; nil = follow Settings.
        var voice: AgentVoice? = nil
        /// The spoken name this turn answers to (the agent's phrase when it has
        /// one). nil = the app's own phrase.
        var wakePhrase: String? = nil

        /// The one builder every turn site goes through.
        static func from(_ r: ResolvedAgentSettings,
                         voiceStyle: Bool = false,
                         documentIndex: DocumentIndex? = nil,
                         telegramChatId: Int64? = nil) -> TurnConfig {
            TurnConfig(
                agentMode: r.toolsEnabled,
                mcpMode: r.mcpEnabled,
                enableThinking: r.thinkingEnabled,
                voiceStyle: voiceStyle,
                workingDirectory: r.workingDirectory,
                documentIndex: documentIndex,
                telegramChatId: telegramChatId,
                agentId: r.agentId,
                systemPromptPrefix: r.systemPromptPrefix,
                tools: r.tools,
                autoApprove: r.autoApprove,
                temperature: r.temperatureOverride,
                maxTokens: r.maxTokensOverride,
                topP: r.topPOverride,
                topK: r.topKOverride,
                repeatPenalty: r.repeatPenaltyOverride,
                presencePenalty: r.presencePenaltyOverride,
                reasoningBudget: r.reasoningBudgetOverride,
                reasoningEffort: r.reasoningEffort,
                voice: r.voiceOverride,
                wakePhrase: r.wakePhrase
            )
        }

        /// The wire value for `reasoning_effort`, gated on the turn's EFFECTIVE
        /// thinking state: the server reads the field as a thinking opt-in, so
        /// sending it with the toggle off would silently turn thinking on.
        func reasoningEffortParam(thinking: Bool) -> String? {
            thinking ? reasoningEffort.rawValue : nil
        }

        /// The per-request defaults for this turn: the user's saved sampling
        /// with the agent's overrides laid on top. An override REPLACES the
        /// saved value — including with the canonical "off" (top_k 0, repeat
        /// 1.0, presence 0.0, budget -1), which clears the global rather than
        /// leaving it standing, mapped to an omitted field exactly as
        /// `RequestDefaults.from` maps it.
        func requestDefaults(from opts: ServerOptions) -> APIClient.RequestDefaults {
            var d = APIClient.RequestDefaults.from(opts)
            if let v = topP { d.topP = v }
            if let v = topK { d.topK = v > 0 ? v : nil }
            if let v = repeatPenalty { d.repeatPenalty = v != 1.0 ? v : nil }
            if let v = presencePenalty { d.presencePenalty = v != 0.0 ? v : nil }
            if let v = reasoningBudget { d.reasoningBudget = v >= 0 ? v : nil }
            return d
        }

        /// The tools to ADVERTISE: none unless the loop is actually running.
        /// (`tools` itself stays the dispatch allow-list, which must keep
        /// `searchDocuments` for docs-only turns.)
        var advertisedTools: Set<AgentToolKind> { agentMode ? tools : [] }
    }

    // MARK: - Per-turn sampling

    /// The tool loop's own temperature. Deliberately NOT the user's default chat
    /// temperature — a tool-calling loop wants tighter sampling, and this number
    /// predates agents, so changing it would move every existing install.
    static let agentLoopTemperature = 0.7

    /// Token cap for this turn: the agent's override, else the app default.
    private func turnMaxTokens(_ config: TurnConfig) -> Int {
        config.maxTokens ?? appState.maxTokens
    }

    /// Temperature for this turn: the agent's override, else the calling path's
    /// own default (they differ — see `agentLoopTemperature`).
    private func turnTemperature(_ config: TurnConfig, default fallback: Double) -> Double {
        config.temperature ?? fallback
    }

    // MARK: - Convenience accessors

    private var server: ServerManager { appState.server }
    private var mcpManager: MCPManager { appState.mcpManager }
    private func session(_ id: UUID) -> ChatSession? {
        appState.chatSessions.first { $0.id == id }
    }

    /// End a failed turn with a notice ROW rather than `[Error: …]` appended to
    /// the assistant's own text.
    ///
    /// Two things follow from it being its own message. Whatever the model had
    /// already streamed stays intact and readable instead of gaining a bracketed
    /// tail that looks like something it said. And the notice is marked
    /// `failedRetry`, so it renders in the transcript but is excluded from the
    /// history sent back to the model — an error card fed to the next turn as
    /// assistant output is how a model starts apologising for a server failure.
    /// The spinner is cleared first, or `GeneratingIndicator` keeps animating on
    /// the orphaned streaming bubble.
    private func appendErrorNotice(_ error: Error, to sessionId: UUID) {
        appState.updateLastMessage(in: sessionId, streaming: false)
        // Before the notice row is appended, so a regeneration's pager lands on
        // the partial reply and not on our own error card.
        appState.finishRevisions(in: sessionId)
        var msg = ChatMessage(role: .assistant, content: "")
        msg.isStreaming = false
        msg.failedRetry = true
        msg.errorNotice = ChatErrorNotice.from(error)
        appState.appendMessage(to: sessionId, message: msg)
    }

    /// Apply a coalesced streaming batch into the session. Streamed tokens are
    /// batched (see `StreamCoalescer`) rather than written one at a time so the
    /// per-token `AppState.objectWillChange` churn can't re-render — and wedge —
    /// the open tray popover during the assistant's answer.
    private func applyStreamBatch(_ batch: (content: String, reasoning: String)?,
                                  to sessionId: UUID) {
        guard let batch else { return }
        if !batch.content.isEmpty { appState.updateLastMessage(in: sessionId, content: batch.content) }
        if !batch.reasoning.isEmpty { appState.updateLastMessage(in: sessionId, reasoning: batch.reasoning) }
    }

    /// Begin counting a fresh streamed reply (one per round). Zeroes the live
    /// token tally so the composer's "gen:" restarts from 0.
    private func beginLiveTokenCount(for sessionId: UUID) {
        ledger.setLiveTokens(0, session: sessionId)
        liveTokensBySession[sessionId] = 0
    }

    /// Stream one text/reasoning delta into the session and tally it toward the
    /// live token count. The published dict advances only when the coalescer
    /// actually flushes (≤20 Hz), so the live readout never adds per-token churn.
    private func streamDelta(content: String = "", reasoning: String = "",
                             coalescer: inout StreamCoalescer, to sessionId: UUID) {
        ledger.setLiveTokens(ledger.liveTokens(session: sessionId) + 1, session: sessionId)
        if let batch = coalescer.add(content: content, reasoning: reasoning,
                                     now: Date().timeIntervalSinceReferenceDate) {
            applyStreamBatch(batch, to: sessionId)
            liveTokensBySession[sessionId] = ledger.liveTokens(session: sessionId)
        }
    }

    /// Reconcile the live readout to the stream's authoritative usage count.
    private func setLiveTokens(_ n: Int, for sessionId: UUID) {
        ledger.setLiveTokens(n, session: sessionId)
        liveTokensBySession[sessionId] = n
    }

    // MARK: - Public API (TurnRunning)

    /// The app-wide stop: every session's turn. Kept for callers that mean
    /// "silence everything"; per-chat Stop buttons use `stop(sessionId:)`.
    func stop() {
        for sid in ledger.activeSessionIds { stop(sessionId: sid) }
    }

    /// Stop one session's in-flight turn; other sessions keep streaming.
    func stop(sessionId: UUID) {
        guard ledger.activeSessionIds.contains(sessionId) else { return }
        tasks[sessionId]?.cancel()
        tasks[sessionId] = nil
        ledger.endAll(session: sessionId)
        liveTokensBySession.removeValue(forKey: sessionId)
        // Belt and braces on the meter: the cancelled generation's own `defer`
        // clears it as it unwinds, but a card left behind on a stopped turn is a
        // permanent fake progress bar.
        if mediaProgressSessionId == sessionId {
            mediaProgress = nil
            mediaProgressSessionId = nil
        }
        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.finishRevisions(in: sessionId)
        appState.saveChatHistory()
        publishTurnState()
    }

    /// End a turn from its own task's unwind. Token-gated: a superseded task
    /// presents a stale token and touches NOTHING (its successor owns the slot).
    private func endTurn(sessionId: UUID, token: UUID) {
        guard ledger.end(session: sessionId, token: token) else { return }
        tasks[sessionId] = nil
        liveTokensBySession.removeValue(forKey: sessionId)
        if mediaProgressSessionId == sessionId {
            mediaProgress = nil
            mediaProgressSessionId = nil
        }
        // The ONE exit both paths complete through. A regeneration's held seed
        // is applied here rather than when the turn started, because on the
        // agent path the reply it belongs to is the last of several appended
        // from inside the Task — see AppState.pendingRevisionSeed.
        appState.finishRevisions(in: sessionId)
        appState.saveChatHistory()
        publishTurnState()
    }

    func runTurn(sessionId: UUID,
                 userText: String,
                 images: [ChatImage]?,
                 videos: [ChatVideo]? = nil,
                 audio: [ChatAudio]?,
                 config: TurnConfig,
                 approval: @escaping (APIClient.ToolCall) async -> Bool) {
        let text = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty || images != nil || videos != nil || audio != nil,
              server.status == .running else { return }

        // A new submission to the SAME session supersedes its in-flight turn.
        // Other sessions' turns are untouched — the engine is multi-turn.
        stop(sessionId: sessionId)
        let token = ledger.begin(session: sessionId)
        publishTurnState()

        // Publish the answering agent's voice for THIS turn — but only when the
        // turn can actually be the one speaking: a voice-driven turn, or the
        // chat the user is looking at. A BACKGROUND tab's agent finishing later
        // must not hijack the speaking voice mid-utterance (multi-turn). nil
        // restores the live per-utterance Settings read.
        if config.voiceStyle || appState.activeChatId == sessionId {
            ActiveAgentVoice.set(config.voice)
        }

        if config.agentMode || config.mcpMode || config.documentIndex != nil {
            runAgentTurn(sessionId: sessionId, text: text, images: images, videos: videos, audio: audio,
                         config: config, token: token, approval: approval)
        } else {
            runPlainTurn(sessionId: sessionId, text: text, images: images, videos: videos, audio: audio,
                         config: config, token: token)
        }
    }

    /// Regenerate the last reply: drop the last user turn (and whatever
    /// followed it — the old assistant reply, any tool-call chain) and
    /// resubmit that same user text as a fresh turn. Reuses `runTurn` rather
    /// than duplicating the plain/agent branching, so a regenerated turn is
    /// indistinguishable from a freshly sent one.
    func regenerate(sessionId: UUID, config: TurnConfig,
                     approval: @escaping (APIClient.ToolCall) async -> Bool) {
        guard let msgs = session(sessionId)?.messages,
              let lastUserIdx = msgs.lastIndex(where: { $0.role == .user })
        else { return }
        let text = msgs[lastUserIdx].content
        let images = msgs[lastUserIdx].images
        let audio = msgs[lastUserIdx].audio
        // The reply about to be destroyed. `truncateMessages` drops everything
        // from the last user turn onward, so this is the only moment it can be
        // captured — and a regeneration that silently threw away a better first
        // answer is the whole reason the pager exists.
        let replaced = msgs[(lastUserIdx + 1)...].last { $0.role == .assistant && !$0.content.isEmpty }
        appState.truncateMessages(in: sessionId, keepingFirst: lastUserIdx)
        runTurn(sessionId: sessionId, userText: text, images: images, audio: audio,
                config: config, approval: approval)
        // AFTER runTurn, which opens with `stop(sessionId:)` — and stop is a
        // turn exit, so a seed placed before it would be spent immediately.
        // The seed is HELD from here and applied when this turn ends: the
        // reply it belongs to does not exist yet, and on the agent path it is
        // the last of several appended from inside the Task.
        if let replaced { appState.seedRevisions(in: sessionId, from: replaced) }
    }

    /// Extend the reply at the end of the transcript instead of answering
    /// after it — the model is handed its own unfinished text and resumes.
    ///
    /// Deliberately NOT routed through `runTurn`: that appends a user message
    /// and a fresh placeholder, which is exactly what a continuation must not
    /// do. It also never takes the agent path — a tool loop mid-reply would
    /// have to splice a call into text the user has already read.
    func continueReply(sessionId: UUID, config: TurnConfig) {
        guard ContinueReply.isEligible(session(sessionId)?.messages ?? [],
                                       serverRunning: server.status == .running,
                                       busy: composerState(for: sessionId) != .idle,
                                       engine: server.chatModelInfo?.engine)
        else { return }
        stop(sessionId: sessionId)
        let token = ledger.begin(session: sessionId)
        // AFTER stop, which is a turn exit and would consume the mark — the
        // same ordering hazard `regenerate` has with its seed. It tells the
        // turn exit to EXTEND the version being read rather than file the
        // continuation as a new one.
        appState.markContinuing(sessionId)
        publishTurnState()
        runPlainTurn(sessionId: sessionId, text: "", images: nil, audio: nil,
                     config: config, token: token, continuing: true)
    }

    // MARK: - Plain chat

    /// - Parameter continuing: extend the reply already at the end of the
    ///   transcript instead of answering after it. No user message is added and
    ///   no placeholder is appended — the trailing assistant message IS the
    ///   placeholder, and the server is told to treat it as a prefill.
    private func runPlainTurn(sessionId: UUID, text: String,
                              images: [ChatImage]?, videos: [ChatVideo]? = nil, audio: [ChatAudio]?,
                              config: TurnConfig, token: UUID,
                              continuing: Bool = false) {
        if !continuing {
            var userMsg = ChatMessage(role: .user, content: text)
            userMsg.images = images
            userMsg.videos = videos
            userMsg.audio = audio
            appState.appendMessage(to: sessionId, message: userMsg)
        }

        let api = APIClient()

        // Build the request from the session (its source of truth). We append
        // the streaming placeholder AFTER this so it never lands in the
        // request — same pattern the agent loop uses. Image/video handling:
        // only the latest user message's attachments are sent (older turns'
        // are stripped for bandwidth).
        let sessionMsgs = session(sessionId)?.messages ?? []
        let lastUserIdx = sessionMsgs.lastIndex { $0.role == .user }
        let useServerPreprocess = wantsServerImagePreprocess
        let history: [[String: Any]] = sessionMsgs.enumerated().map { i, msg in
            if i == lastUserIdx, msg.role == .user {
                let imgs = msg.images ?? []
                let vids = msg.videos ?? []
                let clips = msg.audio ?? []
                if !imgs.isEmpty || !vids.isEmpty || !clips.isEmpty {
                    return ["role": "user", "content": Self.buildMultimodalContent(text: msg.content, images: imgs, videos: vids, audio: clips, serverPreprocess: useServerPreprocess)]
                }
            }
            return Self.plainHistoryDict(msg)
        }
        // Plain chat: no synthesized system message (see the long note in the
        // original ChatView implementation — a "formatNudge" system message was
        // routinely read by the model AS the user's input). Two exceptions, both
        // things the user asked for explicitly: an agent's persona, and voice
        // mode's style guidance (so spoken answers stay short and Markdown-free).
        // They share ONE system message, persona first.
        var messagesArray = history
        var plainSystemBits: [String] = []
        let persona = config.systemPromptPrefix.trimmingCharacters(in: .whitespacesAndNewlines)
        if !persona.isEmpty { plainSystemBits.append(persona) }
        if config.voiceStyle {
            // `hasPersona`: with an agent above it, the voice guidance must not
            // name the assistant after the app's wake phrase — it's appended last,
            // so that name would override the persona (live: an agent said it was
            // called Jarvis).
            plainSystemBits.append(VoicePrompt.systemPrompt(
                phrase: config.wakePhrase ?? appState.serverOptions.wakePhrase,
                hasPersona: !persona.isEmpty))
        }
        // Third explicit ask: a skill the user invoked by NAME (`/music3 …`).
        // Plain chat builds its own system message, so this is a SECOND
        // construction site — the agent loop's injection does not cover it
        // (live: /music3 with Tools off answered from the model's own head).
        let invokedSkill = AgentPrompt.skillManager.invokedSkill(for: text)
        if !invokedSkill.isEmpty { plainSystemBits.append(invokedSkill) }
        if !plainSystemBits.isEmpty {
            messagesArray.insert(["role": "system",
                                  "content": plainSystemBits.joined(separator: "\n\n")],
                                 at: 0)
        }

        // Streaming placeholder for the UI — appended AFTER the request body is
        // built so it doesn't show up in the prompt. A continuation has one
        // already: the reply being extended. Appending a second would stream
        // the rest of the sentence into a NEW bubble under the one it belongs
        // to, and `updateLastMessage` (which appends) writes to the last
        // message either way, so reusing it needs no other change.
        if continuing {
            // The notice said the reply was cut. It is being un-cut.
            appState.clearTruncationNotice(in: sessionId)
            appState.updateLastMessage(in: sessionId, streaming: true)
        } else {
            var assistantMsg = ChatMessage(role: .assistant, content: "")
            assistantMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: assistantMsg)
        }

        tasks[sessionId] = Task { [weak self] in
            await self?.streamPlainResponse(api: api, sessionId: sessionId,
                                            messages: messagesArray, config: config,
                                            token: token, continuing: continuing)
        }
    }

    private func streamPlainResponse(api: APIClient, sessionId: UUID,
                                     messages: [[String: Any]], config: TurnConfig,
                                     token: UUID, continuing: Bool = false) async {
        do {
            // A media-first server runs headless (no default model) — hot-load
            // the selected chat model once so the request below resolves.
            await server.ensureDefaultChatModel(selectedModelPath: appState.selectedModelPath)
            // Pin the request to the active model (server-resolved default if
            // nil) so hot-switch can finish in-flight requests on the old model.
            let thinking = config.enableThinking || appState.serverOptions.defaultEnableThinking
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: turnMaxTokens(config),
                temperature: turnTemperature(config, default: appState.serverOptions.defaultTemperature),
                enableThinking: thinking,
                reasoningEffort: config.reasoningEffortParam(thinking: thinking),
                defaults: config.requestDefaults(from: appState.serverOptions),
                modelId: server.chatModelId,
                continueFinalMessage: continuing
            )
            var coalescer = StreamCoalescer()
            beginLiveTokenCount(for: sessionId)
            for try await event in stream {
                try Task.checkCancellation()
                switch event {
                case .content(let text):
                    streamDelta(content: text, coalescer: &coalescer, to: sessionId)
                case .reasoning(let text):
                    streamDelta(reasoning: text, coalescer: &coalescer, to: sessionId)
                case .usage(let usage):
                    applyStreamBatch(coalescer.drain(), to: sessionId)
                    // A continuation streams into a message that already
                    // carries a generation's worth of tokens, so the counts
                    // ADD — the footnote describes the reply, and the reply is
                    // both halves.
                    appState.updateLastMessage(in: sessionId, usage: usage,
                                               addingCompletionTokens: continuing)
                    setLiveTokens(usage.completionTokens, for: sessionId)   // reconcile to the authoritative count
                case .toolCalls:
                    break
                case .truncated(let cause):
                    // Plain chat is a single, always-terminal response — show the
                    // notice immediately (no agent loop to stack it). It rides the
                    // message as DATA, never content: content is what history
                    // builders send back, and the old in-content banner taught
                    // the model its own warning text.
                    applyStreamBatch(coalescer.drain(), to: sessionId)
                    appState.updateLastMessage(in: sessionId, truncation: .init(cause: cause, maxTokens: turnMaxTokens(config)))
                case .done:
                    break
                }
            }
            applyStreamBatch(coalescer.drain(), to: sessionId)   // flush the trailing batch
        } catch is CancellationError {
            // Stopped by user (`stop(sessionId:)`) or superseded by a new
            // submission — either way that path already cleared the streaming
            // flag, and with a SUCCESSOR turn possibly mid-stream in this same
            // session, touching the last message here would hit ITS placeholder.
            endTurn(sessionId: sessionId, token: token)
            return
        } catch {
            print("[ChatTurnEngine] Chat error: \(error)")
            try? "Chat error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
            appendErrorNotice(error, to: sessionId)
        }
        appState.updateLastMessage(in: sessionId, streaming: false)
        appState.saveChatHistory()
        endTurn(sessionId: sessionId, token: token)
    }

    // MARK: - Agent mode (native tool calling)

    private func runAgentTurn(sessionId: UUID, text: String,
                              images: [ChatImage]?, videos: [ChatVideo]? = nil, audio: [ChatAudio]?,
                              config: TurnConfig, token: UUID,
                              approval: @escaping (APIClient.ToolCall) async -> Bool) {
        var userMsg = ChatMessage(role: .user, content: text)
        userMsg.images = images
        userMsg.videos = videos
        userMsg.audio = audio
        appState.appendMessage(to: sessionId, message: userMsg)

        let api = APIClient()
        let workDir = config.workingDirectory
        // Under the App Sandbox a user-picked working folder outside the
        // container is unreachable after a relaunch until its security-scoped
        // bookmark is resolved. This is the single seam every agent turn passes
        // through (chat window, Quick Launcher, voice), so shell/file tools and
        // MCP servers below always run with the grant live. No bookmark stored
        // (default workspace, DMG build) → no-op.
        SecurityScopedBookmark.startAccessOnce(name: SecurityScopedBookmark.workingFolderName(sessionId))
        // Sessions inheriting the DEFAULT workspace have no per-session slot;
        // a custom default picked in Settings rides this global bookmark.
        SecurityScopedBookmark.startAccessOnce(name: SecurityScopedBookmark.defaultWorkspaceName)

        tasks[sessionId] = Task { [weak self] in
            guard let self else { return }
            // Lazy-spawn MCP servers if MCP mode is on. Idempotent — already-connected servers are skipped.
            if config.mcpMode {
                // Inherit the chat's working directory so filesystem/shell MCP servers anchor at the
                // same dir the agent's built-in tools use. Per-entry `cwd` in mcp.json still wins.
                self.mcpManager.defaultCwd = config.workingDirectory
                await self.mcpManager.startEnabled()
                // Surface startup failures inline in chat — otherwise they're hidden behind the
                // marketplace gear icon and the user just sees "MCP doesn't seem to do anything".
                if !self.mcpManager.startErrors.isEmpty {
                    let lines = self.mcpManager.startErrors
                        .sorted(by: { $0.key < $1.key })
                        .map { "• **\($0.key)**: \($0.value)" }
                        .joined(separator: "\n")
                    let hint = self.mcpManager.sessions.isEmpty
                        ? "No MCP servers are connected — the model has no MCP tools available for this turn. Open the gear icon on the MCP pill to fix or disable broken servers."
                        : "Some MCP servers couldn't start. The model will only see tools from the ones that did connect."
                    let warning = ChatMessage(
                        role: .assistant,
                        content: "⚠️ MCP startup issues:\n\n\(lines)\n\n\(hint)"
                    )
                    self.appState.appendMessage(to: sessionId, message: warning)
                }
            }
            do {
                try await self.runAgentLoop(api: api, sessionId: sessionId, config: config,
                                            workingDirectory: workDir, approval: approval)
            } catch is CancellationError {
                // Stopped by user or superseded — `stop(sessionId:)` already
                // cleared the streaming flag, and a successor turn may be
                // streaming into this session already. Release only OUR slot.
                self.endTurn(sessionId: sessionId, token: token)
                return
            } catch {
                print("[ChatTurnEngine] Agent error: \(error)")
                try? "Agent error: \(error)\n".write(toFile: NSString(string: "~/.mlx-serve/debug.log").expandingTildeInPath, atomically: true, encoding: .utf8)
                self.appendErrorNotice(error, to: sessionId)
            }
            self.appState.saveChatHistory()
            self.endTurn(sessionId: sessionId, token: token)
        }
    }

    /// Agent loop: call model with tools (streaming), execute tool calls, feed results back, repeat.
    /// Stops when the model responds with content (no tool calls) or after 150 iterations.
    private func runAgentLoop(api: APIClient, sessionId: UUID, config: TurnConfig,
                              workingDirectory initialWorkDir: String?,
                              approval: @escaping (APIClient.ToolCall) async -> Bool) async throws {
        var workingDirectory = initialWorkDir
        // One media generation per USER TURN, not per round — a round cap can't
        // bound a model that just calls again next round. This token identifies
        // the turn for every round below.
        let mediaTurn = UUID()
        if mediaProgressSessionId == sessionId {
            mediaProgress = nil
            mediaProgressSessionId = nil
        }
        let maxIterations = 150
        let padRetryPolicy = RetryPolicy.aggressive
        let repetition = AgentEngine.RepetitionTracker()
        // Bail out if the model spends several rounds where every tool call
        // fails/blocks (e.g. an unresolvable tool name) instead of grinding to
        // `maxIterations` achieving nothing.
        var stuck = AgentEngine.StuckDetector()
        // Budget for recoverable failures: ghost/malformed tool calls, truncated
        // tool-call args, empty/pad responses. CONSECUTIVE, not cumulative — a
        // real tool round resets it (recordProgress below), so an isolated late
        // failure doesn't end a long, productive turn.
        var retry = AgentEngine.AgentRetryBudget()

        // Resolve the LAN IP once per turn (not per iteration): it's a getifaddrs
        // enumeration that won't change mid-turn, and the agent loop rebuilds the
        // system prompt on every tool round.
        let lanIP = SystemGrounding.localIPAddress()

        for iteration in 0..<maxIterations {
            try Task.checkCancellation()

            // Session deleted mid-turn → the turn is orphaned. Bail before
            // issuing another request: with the session gone every append
            // no-ops and the pad-retry path would burn full generations
            // against an empty history (see `stopIfOrphaned`).
            guard session(sessionId) != nil else { return }

            // Build message history for API
            let turnMax = turnMaxTokens(config)
            let contextLength = AgentEngine.effectiveContextLength(
                appContextSize: appState.contextSize,
                modelContextLength: server.chatModelInfo?.contextLength
            )
            let useServerPreprocess = wantsServerImagePreprocess
            var history = AgentEngine.buildAgentHistory(
                messages: session(sessionId)?.messages ?? [],
                contextLength: contextLength,
                maxTokens: turnMax,
                buildMultimodalContent: { text, images in
                    Self.buildMultimodalContent(text: text, images: images, serverPreprocess: useServerPreprocess)
                }
            )
            let userMsg = history.last { ($0["role"] as? String) == "user" }?["content"] as? String ?? ""
            let mcpToolsJSON = config.mcpMode ? mcpManager.toolDefinitionsJSON() : nil
            let mcpListing = config.mcpMode ? mcpManager.toolListingForPrompt() : ""
            var systemPrompt: String
            // Volatile context that changes mid-session — kept OUT of the
            // stable prefix and appended at the very end (see composeSystemPrompt).
            var agentVolatileTail = ""
            if config.agentMode {
                let skills = AgentPrompt.skillManager.matchingSkills(for: userMsg)
                // Stable, cacheable core: base instructions + execution
                // environment + memory instructions + MCP listing. The model's
                // tool block (rendered by the chat template) sits in front of
                // all of this, so as long as this prefix stays byte-identical
                // the server reuses the whole tool+instruction KV across
                // requests. The environment section is stable within a session
                // (it only changes when the user flips the Agent Sandbox
                // setting — one KV re-prefill, then cached again).
                systemPrompt = AgentPrompt.systemPrompt
                    + AgentPrompt.executionEnvironmentSection(sandboxed: AgentSandbox.shared.isEnabled)
                    + AgentPrompt.memory
                if !mcpListing.isEmpty {
                    systemPrompt += "\n\n# MCP Tools\nIn addition to the built-in tools above, the user has connected these MCP servers. Their tools are namespaced as `<server>__<tool>`:\n\n\(mcpListing)"
                }
                // Volatile context to the tail, ordered big-listing-then-tiny-
                // snippet so a shell command (which rewrites the recent-commands
                // snippet every turn) only re-prefills the snippet, not the
                // working-dir listing: matched skills (per message), the
                // working-dir listing (changes as files change), then the
                // learned recent-dirs/commands snippet (changes per command).
                // Most-stable volatile item first (it only changes when the
                // user switches music model), so a per-message skill hit
                // re-prefills less.
                if config.advertisedTools.contains(.generateMusic) {
                    agentVolatileTail += AgentPrompt.musicEngineNote(
                        MusicGenSettings.load().resolvedModel(models: server.allModels))
                }
                agentVolatileTail += skills
                if let wd = workingDirectory {
                    agentVolatileTail += AgentEngine.workingDirectoryContext(wd)
                }
                agentVolatileTail += appState.agentMemory.contextSnippet()
            } else if config.mcpMode {
                // MCP-only mode: minimal system prompt focused on MCP tool use, no shell/file rules.
                systemPrompt = AgentPrompt.mcpOnlySystemPrompt(toolListing: mcpListing)
            } else {
                // Docs-only mode: plain chat with a document folder attached.
                let index = config.documentIndex
                systemPrompt = AgentPrompt.docsOnlySystemPrompt(
                    folderName: index?.folderName ?? "documents",
                    fileCount: indexedFileCount(index))
            }
            // A skill invoked by NAME (`/music3 …`) works in every mode: the
            // user asked for it explicitly, so it does not wait for the agent
            // loop's trigger matching (which also covers it, above).
            if !config.agentMode {
                agentVolatileTail += AgentPrompt.skillManager.invokedSkill(for: userMsg)
            }
            // Attached-docs section for the modes whose base prompt doesn't
            // already explain the searchDocuments tool.
            if let index = config.documentIndex, config.agentMode || config.mcpMode {
                systemPrompt += AgentPrompt.attachedDocumentsSection(
                    folderName: index.folderName, fileCount: indexedFileCount(index))
            }
            // Ground the turn in the wall clock (so "what day is it" is answered
            // from reality) and surface the Mac's LAN IP (so the agent hands out
            // reachable URLs). Date-ONLY (no clock time) and at the very END:
            // a per-minute timestamp at the front changed the first tokens every
            // request, so the KV prefix cache missed at token 0 and every agent
            // turn cold-re-prefilled the whole prompt — the cause of slow TTFB.
            var grounding = SystemGrounding.dateLine()
            let ipLine = SystemGrounding.localIPLine(ip: lanIP)
            if !ipLine.isEmpty { grounding += " " + ipLine }
            // Stable prefix first, all volatile content (skills / working-dir /
            // memory) + grounding last — so a mid-session change only re-prefills
            // the short tail, not the cached tool+instruction block.
            // Make the model aware of its actual per-response token cap so it
            // chunks large writes BEFORE truncating (a static "~200 lines" hint
            // gets ignored). Stable within the session → stays in the volatile
            // tail, before the date grounding, so the KV prefix still hits.
            systemPrompt = Self.composeSystemPrompt(persona: config.systemPromptPrefix,
                                                    stable: systemPrompt,
                                                    volatileTail: agentVolatileTail + AgentPrompt.outputBudgetGuidance(maxTokens: turnMaxTokens(config), contextLength: contextLength),
                                                    grounding: grounding)
            // Voice mode: tools/thinking run silently; only the final answer is
            // spoken, so steer it to a short, speakable reply (no URLs/Markdown).
            if config.voiceStyle {
                systemPrompt = VoicePrompt.decorate(
                    systemPrompt,
                    phrase: config.wakePhrase ?? appState.serverOptions.wakePhrase,
                    hasPersona: !config.systemPromptPrefix.isEmpty)
            }
            var messages: [[String: Any]] = [["role": "system", "content": systemPrompt]]
            // Some models (e.g. Gemma 4 E4B) can't generate after tool results without
            // a user message. Add a nudge so the model knows to synthesize a response —
            // asks explicitly for a short plain-text summary when finished so the user
            // never sees a conversation that ends on a bare tool-call echo.
            if let lastRole = history.last?["role"] as? String, lastRole == "tool" {
                history.append(["role": "user", "content": "Continue. If the task is complete, reply with a short plain-text summary for the user (what got done, where it lives, any caveats) — no tool calls, no JSON. If more work is needed, make the next tool call."])
            }
            messages.append(contentsOf: history)

            AgentEngine.dumpDebugRequest(messages: messages, maxTokens: turnMax)

            // Add streaming assistant message
            var streamMsg = ChatMessage(role: .assistant, content: "")
            streamMsg.isStreaming = true
            appState.appendMessage(to: sessionId, message: streamMsg)

            // Stream model response with tools
            var receivedToolCalls: [APIClient.ToolCall] = []
            var maxTokensHit = false
            var truncationCause: TruncationNotice.Cause? = nil
            let combinedToolsJSON = Self.combinedToolsJSON(
                tools: config.advertisedTools,
                mcpToolsJSON: mcpToolsJSON,
                docsToolJSON: config.documentIndex != nil ? AgentPrompt.searchDocumentsToolJSON : nil
            )
            // Headless (media-first) server → ensure the selected chat model
            // is loaded + promoted before the alias-addressed request.
            await server.ensureDefaultChatModel(selectedModelPath: appState.selectedModelPath)
            let stream = api.streamChat(
                port: server.port,
                messages: messages,
                maxTokens: turnMaxTokens(config),
                temperature: turnTemperature(config, default: Self.agentLoopTemperature),
                enableThinking: config.enableThinking,
                reasoningEffort: config.reasoningEffortParam(thinking: config.enableThinking),
                toolsJSON: combinedToolsJSON,
                defaults: config.requestDefaults(from: appState.serverOptions),
                modelId: server.chatModelId
            )

            // No client-side stream watchdog: long generations (large
            // contexts, big batches, slow sampling on big MoE) can legitimately
            // sit silent for minutes between events. The user keeps the Stop
            // button as the manual cancel; URLSession's own resource timeout
            // (set in APIClient) handles a truly broken socket.
            let streamTask = Task<(tcs: [APIClient.ToolCall], maxHit: Bool, cause: TruncationNotice.Cause?), Error> {
                var tcs: [APIClient.ToolCall] = []
                var maxHit = false
                var cause: TruncationNotice.Cause? = nil
                var coalescer = StreamCoalescer()
                self.beginLiveTokenCount(for: sessionId)
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .content(let text):
                        self.streamDelta(content: text, coalescer: &coalescer, to: sessionId)
                    case .reasoning(let text):
                        self.streamDelta(reasoning: text, coalescer: &coalescer, to: sessionId)
                    case .usage(let usage):
                        self.applyStreamBatch(coalescer.drain(), to: sessionId)
                        appState.updateLastMessage(in: sessionId, usage: usage)
                        self.setLiveTokens(usage.completionTokens, for: sessionId)   // reconcile to the authoritative count
                    case .toolCalls(let calls):
                        tcs = calls
                    case .truncated(let c):
                        // Just record it. The notice is surfaced once at the
                        // turn's terminal exit (see below) — appending here, per
                        // iteration, is what stacked duplicate banners on a
                        // multi-step agent turn.
                        maxHit = true
                        cause = c
                    case .done:
                        break
                    }
                }
                // Flush the trailing batch so the message content is complete
                // before the post-stream truncation/pad checks read it back.
                self.applyStreamBatch(coalescer.drain(), to: sessionId)
                return (tcs, maxHit, cause)
            }
            // Wire the user's Stop button through to the inner stream task.
            do {
                let result = try await withTaskCancellationHandler {
                    try await streamTask.value
                } onCancel: {
                    streamTask.cancel()
                }
                receivedToolCalls = result.tcs
                maxTokensHit = result.maxHit
                truncationCause = result.cause
            } catch is CancellationError {
                throw CancellationError()
            }
            appState.updateLastMessage(in: sessionId, streaming: false)

            // A repetition-loop cut ENDS the turn, ahead of every recovery path
            // below. The loop's text is already in the transcript — a streamed
            // delta cannot be retracted, so the server's own trim never reaches
            // a streaming client — and the next round would send it back as
            // history for the model to read and resume. That is the error-echo
            // class with our own transcript as the error, and from the server it
            // looks like five cuts in a row, each firing sooner than the last
            // (live 2026-08-05, under pi). Every other "length" cause keeps
            // its recovery: the reply was fine and simply ran out of room.
            if TruncationNotice.endsTurn(cause: truncationCause) {
                appState.updateLastMessage(in: sessionId, truncation: .init(cause: .repetitionLoop, maxTokens: turnMaxTokens(config)))
                return
            }

            // Truncation recovery: if max_tokens was hit AND tool calls were received,
            // the tool call args are likely truncated (incomplete JSON). Don't execute them —
            // mark the broken message as non-replayable (preserves reasoning in the UI)
            // and nudge the model to try again more concisely.
            if maxTokensHit && !receivedToolCalls.isEmpty && retry.allowTruncationRetry() {
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   !appState.chatSessions[sIdx].messages.isEmpty {
                    let mIdx = appState.chatSessions[sIdx].messages.count - 1
                    appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    appState.chatSessions[sIdx].messages[mIdx].toolCalls = nil
                }
                let nudge = ChatMessage(role: .user, content: Self.truncatedToolCallNudge)
                appState.appendMessage(to: sessionId, message: nudge)
                continue
            }

            // Check for pad-only or empty responses — retry limited times.
            // Mark the empty message as failedRetry so it's hidden from API history
            // but its reasoning (if any) stays visible in the UI.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""
                let cleaned = lastContent
                    .replacingOccurrences(of: "<pad>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if cleaned.isEmpty {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    retry.pad += 1
                    if retry.pad <= padRetryPolicy.maxRetries {
                        let delay = padRetryPolicy.delay(for: retry.pad)
                        try? await Task.sleep(nanoseconds: delay)
                        continue
                    }
                    let errorMsg = ChatMessage(role: .assistant, content: "The model couldn't generate a response. Try rephrasing or starting a new chat.")
                    appState.appendMessage(to: sessionId, message: errorMsg)
                    return
                }
            }

            // If no tool calls, we're done — but make sure the user sees a
            // clean completion text. The model sometimes exits with a ghost
            // tool call (malformed <|tool_call>...<tool_call|> or <tool_call>
            // with bad args that didn't parse) as its final content; that's
            // ugly and uninformative. When we detect one, mark the garbled
            // turn as failedRetry (hidden from API history) and ask the model
            // for a plain-text summary before returning control to the user.
            if receivedToolCalls.isEmpty {
                let lastContent = appState.chatSessions
                    .first(where: { $0.id == sessionId })?.messages.last?.content ?? ""

                // Truncation, NOT a ghost. When the cap was hit AND the content
                // still carries a tool-call opener with no matching close, the
                // call was cut off mid-emission and the server's parser couldn't
                // recover it (Step 1 recovers Hermes/XML, but a future format
                // could escape it). This is a *truncation* — route it to the
                // chunk/append nudge (budget 2) instead of falling through to the
                // ghost path's "call it with proper JSON" (useless: the JSON was
                // fine, just too long). Must precede the ghost check, which also
                // matches `<function=`/`<tool_call>`.
                if maxTokensHit && Self.hasUnclosedToolCallOpener(lastContent) && retry.allowTruncationRetry() {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    let nudge = ChatMessage(role: .user, content: Self.truncatedToolCallNudge)
                    appState.appendMessage(to: sessionId, message: nudge)
                    continue
                }
                // Match the full `<tool…` family — `<tool_call>`, `<tool_call name=…>`,
                // `<tool_calls>` wrapper, `<tool name=… arguments=…/>` self-closing,
                // Gemma 4 `<|tool_call>`/`<tool_call|>`, and `<function=` legacy. The
                // server-side `parseToolCalls` already handles all of these; this
                // check is the defense-in-depth that fires the retry nudge when
                // a new model variant slips through the parser before we recognize
                // it (the symptom: hundreds of completion_tokens but the assistant
                // turn ends with markup-as-content and no parsed tool_calls).
                let looksLikeGhostToolCall = lastContent.contains("<|tool_call>") ||
                    lastContent.contains("<tool_call>") ||
                    lastContent.contains("<tool_call ") ||
                    lastContent.contains("<tool_calls>") ||
                    lastContent.contains("<tool_calls ") ||
                    lastContent.contains("<tool_call|>") ||
                    lastContent.contains("<tool name=") ||
                    lastContent.contains("<function=")
                if looksLikeGhostToolCall && retry.allowGhostRetry() {
                    if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                       !appState.chatSessions[sIdx].messages.isEmpty {
                        let mIdx = appState.chatSessions[sIdx].messages.count - 1
                        appState.chatSessions[sIdx].messages[mIdx].failedRetry = true
                    }
                    let nudge = ChatMessage(role: .user, content: "[System: your last response contained a malformed tool-call tag. If you meant to call a tool, call it with proper JSON. If the task is complete, respond with a short plain-text summary of what you did — no tool tags, no JSON — just a sentence or two for the user.]")
                    appState.appendMessage(to: sessionId, message: nudge)
                    continue
                }
                // Terminal exit: a final answer with no more tool calls. If it
                // was cut off by the cap, surface the truncation notice exactly
                // once here — not per iteration in the stream loop above.
                if TruncationNotice.shouldShow(maxTokensHit: maxTokensHit, turnEnding: true, willRetry: false) {
                    appState.updateLastMessage(in: sessionId, truncation: .init(cause: truncationCause ?? .maxTokens, maxTokens: turnMaxTokens(config)))
                }
                return
            }

            // A round with real, parseable tool calls = progress. Reset the
            // recoverable-failure budget so an isolated *later* ghost/truncated
            // tool call gets its own retry instead of ending the turn (the budget
            // counts consecutive failures, not lifetime-of-turn ones).
            retry.recordProgress()

            // Track repetition for this round
            repetition.track(toolCalls: receivedToolCalls)

            // Store tool calls on the assistant message for history replay
            if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               !appState.chatSessions[sIdx].messages.isEmpty {
                let mIdx = appState.chatSessions[sIdx].messages.count - 1
                appState.chatSessions[sIdx].messages[mIdx].toolCalls = receivedToolCalls.map { tc in
                    let argsJson = (try? JSONSerialization.data(withJSONObject: tc.arguments))
                        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                    return SerializedToolCall(id: tc.id, name: tc.name, arguments: argsJson)
                }
            }

            // Show tool call summary as display-only message. Mark streaming so the GeneratingIndicator
            // keeps rendering underneath while tools execute — otherwise a slow / hung MCP tool looks
            // like the chat just froze with no feedback.
            let callSummary = receivedToolCalls.map { tc in
                let args = tc.arguments.map { "\($0.key): \($0.value.prefix(80))" }.joined(separator: ", ")
                let display = args.isEmpty ? tc.rawArguments.prefix(200) : args[...]
                return "**\(tc.name)**(\(display))"
            }.joined(separator: "\n")
            var summaryMsg = ChatMessage(role: .assistant, content: callSummary)
            summaryMsg.isAgentSummary = true
            summaryMsg.isStreaming = true
            let summaryId = summaryMsg.id
            appState.appendMessage(to: sessionId, message: summaryMsg)
            // Stop the spinner on the summary regardless of how we leave the loop (success, throw, cancel).
            defer {
                if let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
                   let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == summaryId }) {
                    appState.chatSessions[sIdx].messages[mIdx].isStreaming = false
                }
            }

            // Execute each tool call. MCP-namespaced names (`<server>__<tool>`) route to MCPManager;
            // everything else flows through the existing AgentEngine dispatch.
            // Tool-approval gate: before every dispatch, ask via the injected
            // `approval` closure (the chat window's sheet, or the voice
            // controller's auto-approve/inline card). Deny short-circuits to a
            // fabricated error result so the agent loop can react and the
            // user's intent is visible in the transcript.
            var roundOutputs: [String] = []
            var roundHandles: [String] = []
            for tc in receivedToolCalls {
                try Task.checkCancellation()

                // An agent that declared auto-approve skips the gate entirely —
                // one decision, made when the agent was configured, instead of a
                // dialog per call.
                let approved = config.autoApprove ? true : await approval(tc)
                guard approved else {
                    let denied = AgentEngine.ToolResult(
                        id: tc.id,
                        name: tc.name,
                        output: "Error: user denied this tool call. Do not retry this command; ask the user how to proceed or try a different approach."
                    )
                    // A user denial is a deliberate stop, not a stuck loop — don't
                    // count it toward the no-progress tally.
                    var deniedMsg = ChatMessage(role: .assistant, content: "**\(tc.name)** → denied by user")
                    deniedMsg.isAgentSummary = true
                    appState.appendMessage(to: sessionId, message: deniedMsg)
                    var toolMsg = ChatMessage(role: .system, content: denied.output)
                    toolMsg.toolCallId = denied.id
                    toolMsg.toolName = denied.name
                    appState.appendMessage(to: sessionId, message: toolMsg)
                    continue
                }

                // One execution path for built-in *and* MCP tools — the shared
                // repetition guard applies to both, so an MCP tool can no longer
                // loop forever (it routes to `mcpManager` internally via
                // `MCPToolRouting`).
                let result = await AgentEngine.executeToolCall(
                    tc, workingDirectory: &workingDirectory,
                    repetition: repetition, iteration: iteration,
                    agentMemory: appState.agentMemory,
                    mcpRouter: mcpManager,
                    mcpEnabled: config.mcpMode,
                    documentIndex: config.documentIndex,
                    createTask: { [weak self] goal, schedule in
                        await self?.createTaskFromAgent(
                            goal: goal, schedule: schedule,
                            telegramChatId: config.telegramChatId
                        ) ?? "Error: task creation unavailable."
                    },
                    generateMedia: { [weak self] kind, mediaArgs in
                        await self?.generateMediaFromAgent(kind: kind, args: mediaArgs, turn: mediaTurn,
                                                           session: sessionId)
                            ?? "Error: media generation unavailable."
                    },
                    processRegistry: appState.processRegistry,
                    sessionId: sessionId,
                    allowedTools: config.tools
                )
                roundOutputs.append(result.output)
                if let handle = result.backgroundHandle { roundHandles.append(handle) }

                // Build the model-facing tool message FIRST, so the visible
                // summary can mirror it 1:1 (same content the model receives —
                // no separate, smaller display cap).
                var toolMsg = ChatMessage(role: .system, content: "")
                toolMsg.toolCallId = result.id
                toolMsg.toolName = result.name

                // Inline images. `browse` screenshots attach to the (hidden) tool
                // message as VISION INPUT for the next turn. `generate_image`
                // instead surfaces the rendered image to the USER as a separate
                // visible assistant message (the tool message is hidden, and only
                // a `.message` bubble renders images — ChatView ~1793); its
                // model-facing content keeps just the caption/path, never the
                // multi-KB base64 (which would blow up the context).
                var pendingInlineImage: ChatImage? = nil
                // A produced track or clip rides back as a PATH, not bytes — see
                // ChatMediaRef. Same split-and-hide shape as the image marker:
                // the model gets the caption, the user gets a player.
                var pendingMediaRef: ChatMediaRef? = nil
                if (result.name == "browse" || result.name == "generate_image")
                    && result.output.contains(AgentMediaInline.jpegDataURIMarker) {
                    let (_, jpeg) = AgentMediaInline.splitInlineImage(result.output)
                    let chatImage = jpeg.map { ChatImage(data: $0) }
                    if result.name == "generate_image" {
                        // A generated image ships BOTH markers, so the caption
                        // comes from the ref split (which stops before the ref
                        // line) rather than the image split (which would keep
                        // it). The ref is what gives the picture the same
                        // Reveal-in-Finder button as a track or a clip.
                        let (caption, ref) = AgentMediaInline.splitMediaRef(
                            result.output, prompt: tc.arguments["prompt"] ?? "")
                        toolMsg.content = caption.isEmpty ? "[image generated]" : caption
                        pendingInlineImage = chatImage
                        pendingMediaRef = ref
                    } else if let chatImage {
                        toolMsg.images = [chatImage]
                        toolMsg.content = "[screenshot captured]"
                    } else {
                        toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                    }
                } else if result.output.contains(AgentMediaInline.mediaRefMarker) {
                    let asked = tc.arguments["prompt"] ?? tc.arguments["text"] ?? ""
                    let (caption, ref) = AgentMediaInline.splitMediaRef(result.output, prompt: asked)
                    toolMsg.content = caption.isEmpty ? "[media generated]" : caption
                    pendingMediaRef = ref
                } else {
                    toolMsg.content = AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name)
                }

                // Visible summary (display-only) — exactly what the model sees.
                var resultMsg = ChatMessage(role: .assistant,
                    content: AgentEngine.toolResultSummary(name: result.name, modelContent: toolMsg.content))
                resultMsg.isAgentSummary = true
                appState.appendMessage(to: sessionId, message: resultMsg)
                appState.appendMessage(to: sessionId, message: toolMsg)

                // Render the generated media inline AFTER the tool-call card, via
                // a visible assistant `.message` (the only row that displays it).
                // ONE message even when both are set — a generated image is its
                // bytes AND its file, and splitting them would put the caption
                // and its Reveal button in a separate row from the picture.
                if pendingInlineImage != nil || pendingMediaRef != nil {
                    var mediaMsg = ChatMessage(role: .assistant, content: "")
                    mediaMsg.images = pendingInlineImage.map { [$0] }
                    mediaMsg.media = pendingMediaRef.map { [$0] }
                    appState.appendMessage(to: sessionId, message: mediaMsg)
                }
            }

            // Attach any background-process handles this round started to the
            // call-summary message (located by the captured summaryId) so the
            // tool-call card can render a kill X for each live process.
            if !roundHandles.isEmpty,
               let sIdx = appState.chatSessions.firstIndex(where: { $0.id == sessionId }),
               let mIdx = appState.chatSessions[sIdx].messages.firstIndex(where: { $0.id == summaryId }) {
                var handles = appState.chatSessions[sIdx].messages[mIdx].processHandles ?? []
                handles.append(contentsOf: roundHandles)
                appState.chatSessions[sIdx].messages[mIdx].processHandles = handles
            }

            // Stop if the model made no progress for several consecutive rounds
            // (every tool call failed/blocked) rather than grinding to the cap.
            stuck.record(outputs: roundOutputs)
            if stuck.isStuck {
                let msg = ChatMessage(role: .assistant, content: "Stopped: the last \(AgentEngine.StuckDetector.limit) tool-call rounds all failed without making progress (often an unrecognized tool name). Tell me how you'd like to proceed.")
                appState.appendMessage(to: sessionId, message: msg)
                return
            }
        }

        // Max iterations reached
        let msg = ChatMessage(role: .assistant, content: "(Agent stopped after \(maxIterations) tool call rounds)")
        appState.appendMessage(to: sessionId, message: msg)
    }

    // MARK: - createTask tool (agent-scheduled background tasks)

    /// Backs the agent's `createTask` tool: build a `ScheduledTask` from a goal +
    /// optional natural-language schedule and hand it to the `TaskScheduler`. A
    /// one-shot ("now" / omitted) is created disabled and run immediately; a
    /// recurring schedule is added enabled and fires on its own. `telegramChatId`
    /// (set on Telegram-driven turns) is stamped on the task so each finished run
    /// reports back to that chat. Returns a short confirmation / error string for
    /// the model to relay.
    func createTaskFromAgent(goal: String, schedule: String?, telegramChatId: Int64?) async -> String {
        let trimmedGoal = goal.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedGoal.isEmpty else {
            return "Error: createTask needs a non-empty \"goal\" — the full instruction the task should carry out (it has no memory of this conversation)."
        }
        let scheduler = appState.taskScheduler
        let title = TaskScheduler.deriveTitle(from: trimmedGoal)
        let willNotify = telegramChatId != nil ? "message you here" : "notify you on the desktop"
        // Inherit MCP exposure from the originating context: the bot's MCP toggle
        // for Telegram-created tasks, the app's MCP mode for in-app ones.
        let useMCP = telegramChatId != nil ? appState.serverOptions.telegram.useMCP : appState.mcpMode

        switch TaskScheduler.scheduleIntent(schedule) {
        case .invalid:
            return "Couldn't understand the schedule “\(schedule ?? "")”. Use a phrase like “every day at 9am”, “every hour”, or “Mon Wed Fri at 8am” — or omit it / say “now” to run once immediately."
        case .once:
            let now = Date()
            let cal = Calendar.current
            let task = ScheduledTask(
                title: title, goal: trimmedGoal,
                trigger: .dailyAt(hour: cal.component(.hour, from: now),
                                  minute: cal.component(.minute, from: now)),
                scheduleText: "once", autonomy: .fullAuto, useMCP: useMCP, enabled: false,
                originTelegramChatId: telegramChatId, deleteAfterRun: true)
            scheduler.addTask(task)
            scheduler.runNow(task)
            return "✅ Created and started task “\(title)”. I'll \(willNotify) with the result when it finishes."
        case .recurring(let trigger):
            let task = ScheduledTask(
                title: title, goal: trimmedGoal, trigger: trigger,
                scheduleText: schedule?.trimmingCharacters(in: .whitespacesAndNewlines),
                autonomy: .fullAuto, useMCP: useMCP, enabled: true,
                originTelegramChatId: telegramChatId)
            scheduler.addTask(task)
            return "✅ Scheduled task “\(title)” to run \(ScheduleParser.describe(trigger)). I'll \(willNotify) with each result."
        }
    }

    // MARK: - Media tools (agent-generated image / speech / music / video)

    /// Backs all four `generate_*` tools. One entry point because they share
    /// everything that matters: the per-turn budget, the "is it downloaded"
    /// gate, the progress meter, and the caption+marker result shape the loop
    /// splits. What differs — which arguments are read and how they're clamped —
    /// lives in `MediaToolArgs`, which is pure and tested.
    ///
    /// The model NEVER picks the model or the quality: those come from the
    /// user's saved settings for that modality, with the step/duration knobs
    /// forced to `MediaChatDefaults` because a chat generation is a preview that
    /// blocks chat decode while it runs.
    /// `turn` identifies the user turn this call belongs to — mint one per turn
    /// in the driving loop and pass the same value for every round of it.
    func generateMediaFromAgent(kind: MediaKind, args: [String: String],
                                turn: UUID, session: UUID? = nil) async -> String {
        if let refusal = mediaBudget.claim(kind, turn: turn) { return refusal }
        mediaProgress = nil
        mediaProgressSessionId = session
        defer { mediaProgress = nil; mediaProgressSessionId = nil }
        // Ownership rides every progress event, not just the claim: with two
        // concurrent turns generating, the later claim would otherwise strand
        // the earlier generation's card in the wrong tab.
        let onProgress: (MediaGenProgress) -> Void = { [weak self] p in
            self?.mediaProgress = p
            self?.mediaProgressSessionId = session
        }
        do {
            switch kind {
            case .image:  return try await runImageTool(args, onProgress: onProgress)
            case .speech: return try await runSpeechTool(args, onProgress: onProgress)
            case .music:  return try await runMusicTool(args, onProgress: onProgress)
            case .video:  return try await runVideoTool(args, onProgress: onProgress)
            }
        } catch let missing as MediaToolArgs.MissingArgument {
            return missing.localizedDescription
        } catch {
            return "Error generating \(kind.rawValue): \(error.localizedDescription)"
        }
    }

    /// Don't kick off a silent multi-GB download from a chat turn: if the saved
    /// model for this modality isn't present, say so and point at the window
    /// that has a progress bar and a RAM gate. Returns nil when it IS usable
    /// (present locally, or a LAN id whose host has the weights).
    private func notDownloadedNotice(repo: String, name: String, approxGB: String,
                                     window: String, lanId: String?) -> String? {
        guard lanId == nil, ServerManager.resolveModelDir(repo: repo) == nil else { return nil }
        return "The \(window.lowercased()) model “\(name)” isn't downloaded yet, so I can't make that. Open the \(window) generation window once (menu-bar tray ▸ \(window)) to download it (~\(approxGB) GB), then ask me again."
    }

    private func runImageTool(_ args: [String: String],
                              onProgress: @escaping (MediaGenProgress) -> Void) async throws -> String {
        let s = ImageGenSettings.load()
        let model = s.resolvedModel(models: appState.server.allModels)
        // A LAN model picked in the Image pane needs no local download — the
        // hosting Mac has the weights.
        let lanId = LanPick.lanId(s.modelId)
        if let notice = notDownloadedNotice(repo: model.repo, name: model.name,
                                            approxGB: "\(model.approxDownloadGB)",
                                            window: "Image", lanId: lanId) { return notice }
        let req = try MediaToolArgs.image(args, model: model,
                                          saved: s.concreteResolution(for: model),
                                          seed: s.seed,
                                          keepResident: s.keepResident, lanId: lanId)
        let path = try await appState.imageGen.generateForAgent(req, server: appState.server,
                                                                onProgress: onProgress)
        // Caption FIRST, base64 LAST, the file reference BETWEEN them. The image
        // is the one modality that ships both: the JPEG bytes the transcript
        // displays, and the PNG path its Reveal-in-Finder button opens. That
        // order is what lets `splitMediaRef` take a clean caption (everything
        // before the ref line) while `splitInlineImage` still finds the payload;
        // a ref line after the base64 would put the whole data URI in the
        // caption the model reads.
        let caption = "Generated a \(req.width)×\(req.height) image for: \(req.prompt). Saved to \(path)."
        let ref = AgentMediaInline.mediaRefLine(kind: .image, path: path)
        guard let dataURI = AgentMediaInline.pngFileToJpegDataURI(path) else {
            return "\(caption)\n\(ref)"
        }
        return "\(caption)\n\(ref)\n\(dataURI)"
    }

    private func runSpeechTool(_ args: [String: String],
                               onProgress: @escaping (MediaGenProgress) -> Void) async throws -> String {
        let s = AudioGenSettings.load()
        let model = s.resolvedModel(models: appState.server.allModels)
        let lanId = LanPick.lanId(s.modelId)
        if let notice = notDownloadedNotice(repo: model.repo, name: model.name,
                                            approxGB: String(format: "%.1f", model.approxDownloadGB),
                                            window: "Audio", lanId: lanId) { return notice }
        var req = try MediaToolArgs.speech(args, model: model,
                                           keepResident: s.keepResident, lanId: lanId)
        // Model's own voice, deliberately: `generate_speech` takes no voice
        // argument, and the Audio window's reference clip is transient view
        // state, so there is nothing here that could honestly be cloned from.
        req.temperature = s.temperature
        let path = try await appState.audioGen.generateForAgent(req, server: appState.server,
                                                                onProgress: onProgress)
        let caption = "Spoke: “\(req.text)”. Saved to \(path)."
        return "\(caption)\n\(AgentMediaInline.mediaRefLine(kind: .audio, path: path))"
    }

    private func runMusicTool(_ args: [String: String],
                              onProgress: @escaping (MediaGenProgress) -> Void) async throws -> String {
        let s = MusicGenSettings.load()
        let model = s.resolvedModel(models: appState.server.allModels)
        let lanId = LanPick.lanId(s.modelId)
        if let notice = notDownloadedNotice(repo: model.repo, name: model.name,
                                            approxGB: String(format: "%.1f", model.approxDownloadGB),
                                            window: "Music", lanId: lanId) { return notice }
        let req = try MediaToolArgs.music(args, model: model, language: s.vocalLanguage,
                                          keepResident: s.keepResident, lanId: lanId)
        let path = try await appState.musicGen.generateForAgent(req, server: appState.server,
                                                                onProgress: onProgress)
        let caption = "Generated a \(req.durationSeconds)s track for: \(req.prompt). Saved to \(path)."
        return "\(caption)\n\(AgentMediaInline.mediaRefLine(kind: .audio, path: path))"
    }

    private func runVideoTool(_ args: [String: String],
                              onProgress: @escaping (MediaGenProgress) -> Void) async throws -> String {
        let s = VideoGenSettings.load()
        let model = s.resolvedModel(models: appState.server.allModels)
        let lanId = LanPick.lanId(s.modelId)
        if let notice = notDownloadedNotice(repo: model.repo, name: model.name,
                                            approxGB: "\(model.approxFirstRunDownloadGB)",
                                            window: "Video", lanId: lanId) { return notice }
        let req = try MediaToolArgs.video(args, model: model,
                                          saved: s.concreteResolution(for: model),
                                          keepResident: s.keepResident, lanId: lanId)
        let path = try await appState.videoGen.generateForAgent(req, server: appState.server,
                                                                onProgress: onProgress)
        let seconds = Double(req.numFrames) / Double(max(req.fps, 1))
        let caption = String(format: "Generated a %.1fs %d×%d clip for: %@. Saved to %@.",
                             seconds, req.width, req.height, req.prompt, path)
        return "\(caption)\n\(AgentMediaInline.mediaRefLine(kind: .video, path: path))"
    }

    // MARK: - Tool JSON + multimodal content (relocated from ChatDetailView)

    /// Build the JSON tools array sent to the model. Concatenates agent tools (when agent mode is on),
    /// MCP tools (when MCP mode is on), and the searchDocuments tool (when a folder is attached).
    /// Returns nil when no tools should be advertised.
    /// `nonisolated` so it stays a pure helper callable off the main actor (unit tests).
    /// Assemble the system prompt so cacheable content stays first and volatile
    /// content lasts. `stable` (base instructions + memory instructions + MCP +
    /// attached-docs, with the template's tool block rendered in front of all of
    /// it) must be byte-identical across a session for the server's KV prefix
    /// cache to reuse the tool+instruction block. `volatileTail` (matched skills,
    /// working-dir listing, learned recent-dirs/commands) and `grounding` (date +
    /// LAN IP) change mid-session, so they go LAST — a change there re-prefills
    /// only the short tail, not the big cached prefix. Pure → unit-tested.
    /// `persona` (the active agent's system prompt) REPLACES the whole
    /// composition: an agent's prompt is the entire system prompt, so the
    /// normal instructions never ride along to compete with it. The
    /// agent-prompt body opens with its own identity claim ("You are an
    /// autonomous agent…"), which sat right after the persona and overrode it
    /// (live 2026-07-29: Laguna answered "who are you?" with "I'm poolside
    /// Malibu" under an Elon Musk persona) — the composeSystemPrompt instance
    /// of the voice-prompt "Jarvis" class. Tools still ride the request's
    /// tools JSON, so tool dispatch is unaffected; the agent's prompt has to
    /// carry anything else it needs (matching plain chat, where a persona is
    /// already the whole system message). Persona is "" when there's no agent,
    /// and the result is then byte-identical to what this produced before
    /// agents existed.
    nonisolated static func composeSystemPrompt(persona: String = "",
                                                stable: String,
                                                volatileTail: String,
                                                grounding: String) -> String {
        if !persona.isEmpty {
            return persona.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var p = stable + volatileTail
        if !grounding.isEmpty { p += "\n\n" + grounding }
        return p
    }

    /// Nudge for a tool call cut off by the token cap (the call was NOT
    /// executed). Shared by the two truncation paths: when the server still
    /// parsed the (truncated) call (args incomplete) and when it dropped it
    /// entirely (a format the parser couldn't recover, leaving only the opener
    /// in content). Steers the model to chunk + append rather than retry the
    /// same one-shot write or switch to an equally-capped heredoc.
    nonisolated static let truncatedToolCallNudge = "[System: Your last response was cut off because the output was too long, so the tool call was NOT executed. Write shorter responses: for a large file, write it in chunks — writeFile a first part, then writeFile again with append:\"true\" for each remaining chunk. (A shell heredoc has the same length limit, so don't switch to that.)]"

    /// True when `content` carries a tool-call OPENER with no matching close —
    /// the signature of a call cut off by the token cap (a *truncation*, not a
    /// malformed/"ghost" call). Routes maxTokensHit-with-empty-calls to the
    /// chunk/append nudge instead of the useless "call it with proper JSON"
    /// ghost nudge. Covers Hermes `<function=`→`</function>`, the
    /// `<tool_call>`/`<tool_calls>` wrappers→`</…>`, and Gemma 4
    /// `<|tool_call>`→`<tool_call|>`. Pure → unit-tested.
    nonisolated static func hasUnclosedToolCallOpener(_ content: String) -> Bool {
        func openNoClose(_ open: String, _ close: String) -> Bool {
            content.contains(open) && !content.contains(close)
        }
        return openNoClose("<function=", "</function>")
            || openNoClose("<tool_call>", "</tool_call>")
            || openNoClose("<tool_calls>", "</tool_calls>")
            || openNoClose("<|tool_call>", "<tool_call|>")
    }

    /// `tools` is the agent's resolved allow-list (empty = the loop's own tools
    /// are off entirely), not a bare on/off flag — the advertised list has to
    /// match what dispatch will actually run.
    nonisolated static func combinedToolsJSON(tools: Set<AgentToolKind>, mcpToolsJSON: String?,
                                              docsToolJSON: String? = nil) -> String? {
        // Strip each array's outer brackets, drop empties, re-wrap as one array.
        let agentTools = tools.isEmpty ? nil : AgentPrompt.toolDefinitionsJSON(allowing: tools)
        let parts = [agentTools, mcpToolsJSON, docsToolJSON]
            .compactMap { $0 }
            .map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            .filter { !$0.isEmpty }
        guard !parts.isEmpty else { return nil }
        return "[\(parts.joined(separator: ","))]"
    }

    /// File count for prompt text — falls back to indexing-time totals so the
    /// prompt is sensible even if a turn races the tail of indexing.
    private func indexedFileCount(_ index: DocumentIndex?) -> Int {
        switch index?.state {
        case .ready(let files, _): return files
        case .indexing(_, let total): return total
        case .preparing, .failed, nil: return 0
        }
    }

    /// Plain-chat history dict for one message. Assistant reasoning the app
    /// received is round-tripped as `reasoning_content` — the server forwards
    /// it to the chat template, and templates that persist reasoning across
    /// turns (laguna) render every prior turn as the empty <think></think>
    /// nothink signature without it, so the model stops thinking from turn 2.
    /// Templates that strip history reasoning (Qwen, Gemma) never read it.
    nonisolated static func plainHistoryDict(_ msg: ChatMessage) -> [String: Any] {
        // `truncationNotice` is a field, so it never rides here by construction;
        // the strip covers sessions saved when the banner lived IN content.
        let content = msg.role == .assistant
            ? TruncationNotice.stripped(from: msg.content) : msg.content
        var d: [String: Any] = ["role": msg.role.rawValue, "content": content]
        if msg.role == .assistant && content.isEmpty { d.removeValue(forKey: "content") }
        if msg.role == .assistant, let rc = msg.reasoningContent, !rc.isEmpty {
            d["reasoning_content"] = rc
        }
        return d
    }

    /// Build OpenAI-style content blocks for a message with images (and,
    /// optionally, video/audio). Delegates to the pure, unit-tested
    /// `MultimodalContent` builder. Two overloads so the `buildAgentHistory`
    /// closure (images only — the agent tool-loop doesn't send video/audio
    /// attachments to the model, same as audio today) and the plain-chat path
    /// (images + video + audio) can both reference it.
    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage], serverPreprocess: Bool = false) -> Any {
        MultimodalContent.build(text: text, images: images, videos: [], audio: [], serverPreprocess: serverPreprocess)
    }

    nonisolated static func buildMultimodalContent(text: String, images: [ChatImage], videos: [ChatVideo] = [], audio: [ChatAudio], serverPreprocess: Bool = false) -> Any {
        MultimodalContent.build(text: text, images: images, videos: videos, audio: audio, serverPreprocess: serverPreprocess)
    }

    var wantsServerImagePreprocess: Bool {
        MultimodalContent.wantsServerPreprocess(architecture: server.chatModelInfo?.architecture ?? "")
    }
}
