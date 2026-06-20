import Foundation
import Combine

/// Runs agentic tasks unattended, reusing the same `ChatTurnEngine` agent loop as
/// interactive chat but with no chat window. v1 (Phase 1) supports on-demand runs
/// ("Run now") with `yolo`/`fullAuto` autonomy auto-approved; recurrence and the
/// pause-for-approval flow land in later phases (see the plan).
///
/// Isolation: runs drive a *dedicated* engine instance so a background run never
/// cancels — or freezes the spinner state of — the user's interactive chat. Runs
/// are serialized through a small FIFO queue that also yields to interactive
/// generation (interactive wins; a scheduled run waits for the chat to go idle).
@MainActor
final class TaskScheduler: ObservableObject {
    unowned let appState: AppState

    /// Dedicated agent engine for unattended runs (independent `isGenerating`).
    private lazy var runEngine = ChatTurnEngine(appState: appState)

    @Published private(set) var tasks: [ScheduledTask] = []
    /// Per-task run history (newest first), lazy-loaded from disk on demand.
    @Published private(set) var runsByTask: [UUID: [TaskRun]] = [:]
    /// The run currently executing, for live UI.
    @Published private(set) var activeRun: TaskRun?

    private var running = false
    /// Serialized work items (a fresh run, or a resume). Interactive chat wins the
    /// single generation slot; queued items wait for it to go idle.
    private var queue: [() async -> Void] = []
    /// Tool calls that paused a run waiting on the user, keyed by run id.
    private var pendingForRun: [UUID: PendingApproval] = [:]
    /// Runs the user cancelled mid-generation. `driveRun` checks this when the
    /// engine goes idle so a user-stopped run finalizes as `.cancelled` instead of
    /// being clobbered with the `.completed`/`.failed` it would otherwise get.
    private var cancelledRunIds: Set<UUID> = []
    /// Single coalescing timer armed to the soonest `nextFireAt` across enabled tasks.
    private var timer: DispatchSourceTimer?
    private var started = false

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = .prettyPrinted
        return e
    }()
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    init(appState: AppState) {
        self.appState = appState
        loadCatalog()
    }

    // MARK: - Catalog CRUD

    func addTask(_ task: ScheduledTask) {
        var t = task
        t.nextFireAt = Self.nextFire(for: t.trigger, after: Date(), calendar: .current)
        tasks.insert(t, at: 0)
        saveCatalog()
        armTimer()
    }

    func updateTask(_ task: ScheduledTask) {
        guard let idx = tasks.firstIndex(where: { $0.id == task.id }) else { return }
        var t = task
        t.nextFireAt = t.enabled ? Self.nextFire(for: t.trigger, after: Date(), calendar: .current) : nil
        tasks[idx] = t
        saveCatalog()
        armTimer()
    }

    func setEnabled(_ id: UUID, _ enabled: Bool) {
        guard let idx = tasks.firstIndex(where: { $0.id == id }) else { return }
        tasks[idx].enabled = enabled
        tasks[idx].nextFireAt = enabled
            ? Self.nextFire(for: tasks[idx].trigger, after: Date(), calendar: .current)
            : nil
        saveCatalog()
        armTimer()
    }

    func deleteTask(_ id: UUID) {
        tasks.removeAll { $0.id == id }
        runsByTask[id] = nil
        saveCatalog()
        armTimer()
        try? FileManager.default.removeItem(atPath: TaskPaths.taskDir(id))
    }

    /// Derive a short title from a free-text goal (first line, ~48 chars).
    nonisolated static func deriveTitle(from goal: String) -> String {
        let firstLine = goal.split(whereSeparator: \.isNewline).first.map(String.init) ?? goal
        let trimmed = firstLine.trimmingCharacters(in: .whitespaces)
        return trimmed.count > 48 ? String(trimmed.prefix(48)) + "…" : (trimmed.isEmpty ? "Untitled task" : trimmed)
    }

    /// What a free-text schedule string means for a programmatically-created task
    /// (the `createTask` agent tool). Pure/testable so the classification isn't
    /// buried in an actor-isolated method.
    enum ScheduleIntent: Equatable {
        case once                      // run a single time, now (no recurrence)
        case recurring(TaskTrigger)    // a parsed recurring schedule
        case invalid                   // provided but unparseable
    }

    /// Classify the `schedule` argument from the createTask tool. Empty / "now" /
    /// "once" (and friends) → run once immediately; anything `ScheduleParser`
    /// understands → recurring; anything else → invalid (so the model gets a
    /// helpful error instead of a silently-dropped schedule).
    nonisolated static func scheduleIntent(_ raw: String?) -> ScheduleIntent {
        let trimmed = (raw ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = trimmed.lowercased()
        let onceWords: Set<String> = ["", "now", "once", "immediately", "asap",
                                      "right now", "right away", "one-off", "one off"]
        if onceWords.contains(lower) { return .once }
        if let trigger = ScheduleParser.parse(trimmed) { return .recurring(trigger) }
        return .invalid
    }

    // MARK: - Scheduling (recurrence)

    /// Soonest time `trigger` should fire strictly after `date`. Pure/testable.
    /// `cron` is handled by `CronSchedule` (Phase 4); returns nil if unparseable.
    nonisolated static func nextFire(for trigger: TaskTrigger, after date: Date, calendar: Calendar) -> Date? {
        switch trigger {
        case .interval(let seconds):
            return date.addingTimeInterval(TimeInterval(max(1, seconds)))
        case .dailyAt(let hour, let minute):
            var comps = DateComponents()
            comps.hour = hour; comps.minute = minute
            return calendar.nextDate(after: date, matching: comps, matchingPolicy: .nextTime)
        case .weekly(let weekdays, let hour, let minute):
            let candidates = weekdays.compactMap { wd -> Date? in
                var comps = DateComponents()
                comps.weekday = wd; comps.hour = hour; comps.minute = minute
                return calendar.nextDate(after: date, matching: comps, matchingPolicy: .nextTime)
            }
            return candidates.min()
        case .cron(let expr):
            return CronSchedule.nextFire(expr, after: date, calendar: calendar)
        }
    }

    /// On wake/launch, decide whether a task whose slot passed while the app was
    /// asleep should run once now, and what its next fire should be. Never replays
    /// every missed interval — at most one catch-up run. Pure/testable.
    nonisolated static func catchUpDecision(task: ScheduledTask, now: Date, calendar: Calendar)
        -> (runNow: Bool, nextFire: Date?) {
        guard task.enabled else { return (false, nil) }
        if let nf = task.nextFireAt, nf <= now {
            // A scheduled slot was missed while we were down.
            let next = nextFire(for: task.trigger, after: now, calendar: calendar)
            return (task.catchUpMissed, next)
        }
        // Not missed (or never scheduled): keep the existing fire or compute a fresh one.
        let next = task.nextFireAt ?? nextFire(for: task.trigger, after: now, calendar: calendar)
        return (false, next)
    }

    /// Start background scheduling: run catch-up for missed slots, then arm the
    /// timer. Called once from AppState.init. Idempotent.
    func start() {
        guard !started else { return }
        started = true
        TaskNotifier.shared.requestAuthorization()
        let now = Date()
        let cal = Calendar.current
        for i in tasks.indices {
            // Heal stale runs first: any run persisted as `.running` was interrupted
            // by an app quit/crash (nothing is executing it now — activeRun is nil at
            // launch), so flip it to `.failed` instead of leaving a forever-spinner.
            let current = runs(for: tasks[i].id)
            let swept = Self.reconcileStaleRuns(current, now: now)
            if swept != current {
                runsByTask[tasks[i].id] = swept
                saveRuns(tasks[i].id, swept)
            }
            let decision = Self.catchUpDecision(task: tasks[i], now: now, calendar: cal)
            tasks[i].nextFireAt = decision.nextFire
            if decision.runNow { enqueue(tasks[i], reason: "catch-up") }
        }
        saveCatalog()
        armTimer()
    }

    private func armTimer() {
        timer?.cancel()
        timer = nil
        let soonest = tasks.filter { $0.enabled }.compactMap { $0.nextFireAt }.min()
        guard let soonest else { return }
        let delay = max(1, soonest.timeIntervalSinceNow)
        let t = DispatchSource.makeTimerSource(queue: .main)
        t.schedule(deadline: .now() + delay)
        t.setEventHandler { [weak self] in
            Task { @MainActor in self?.fireDueTasks() }
        }
        t.resume()
        timer = t
    }

    private func fireDueTasks() {
        let now = Date()
        let cal = Calendar.current
        for i in tasks.indices where tasks[i].enabled {
            guard let nf = tasks[i].nextFireAt, nf <= now else { continue }
            enqueue(tasks[i], reason: "scheduled")
            tasks[i].nextFireAt = Self.nextFire(for: tasks[i].trigger, after: now, calendar: cal)
        }
        saveCatalog()
        armTimer()
    }

    // MARK: - Run history access (lazy)

    func runs(for taskId: UUID) -> [TaskRun] {
        if let cached = runsByTask[taskId] { return cached }
        let loaded = loadRuns(taskId)
        runsByTask[taskId] = loaded
        return loaded
    }

    /// Load a finished run's transcript for the detail viewer.
    func transcript(taskId: UUID, runId: UUID) -> [ChatMessage] {
        let path = TaskPaths.transcriptFile(taskId, runId)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let msgs = try? decoder.decode([ChatMessage].self, from: data) else { return [] }
        return msgs
    }

    // MARK: - Run management (stop / delete)

    /// Stop a run. For the live run this signals `driveRun` to finalize it as
    /// `.cancelled` (so the engine's in-flight completion can't overwrite it); for
    /// a non-terminal run that ISN'T currently executing — a `needsApproval` pause,
    /// or a stale `.running` left by a crash — it flips the status directly. A
    /// terminal run is left alone.
    func cancelRun(taskId: UUID, runId: UUID) {
        if activeRun?.id == runId {
            cancelledRunIds.insert(runId)
            runEngine.stop()   // wakes driveRun's await; it finalizes as .cancelled
            return
        }
        pendingForRun[runId] = nil
        guard var run = runs(for: taskId).first(where: { $0.id == runId }),
              !run.status.isTerminal else { return }
        run.status = .cancelled
        run.finishedAt = Date()
        run.pendingApproval = nil
        persistRun(run, taskId: taskId)
        if activeRun?.id == runId { activeRun = nil }
    }

    /// Delete one run: drop it from the timeline index and remove its on-disk
    /// folder (transcript + artifacts). Refuses to delete the live run — cancel it
    /// first (the UI gates this; the guard is defense in depth against a race).
    func deleteRun(taskId: UUID, runId: UUID) {
        guard activeRun?.id != runId else { return }
        pendingForRun[runId] = nil
        cancelledRunIds.remove(runId)
        var list = runs(for: taskId)
        let before = list.count
        list.removeAll { $0.id == runId }
        guard list.count != before else { return }
        runsByTask[taskId] = list
        saveRuns(taskId, list)
        try? FileManager.default.removeItem(atPath: TaskPaths.runDir(taskId, runId))
    }

    /// Delete every terminal run for a task (keeping any live or awaiting-approval
    /// run), removing each one's on-disk folder. Returns the count removed.
    @discardableResult
    func clearFinishedRuns(taskId: UUID) -> Int {
        let (keep, removed) = Self.runsAfterClearingFinished(runs(for: taskId), activeRunId: activeRun?.id)
        guard !removed.isEmpty else { return 0 }
        runsByTask[taskId] = keep
        saveRuns(taskId, keep)
        for r in removed {
            pendingForRun[r.id] = nil
            try? FileManager.default.removeItem(atPath: TaskPaths.runDir(taskId, r.id))
        }
        return removed.count
    }

    /// Partition a run list into (kept, removed) for a "clear finished" sweep:
    /// terminal runs that aren't the live one are removed; everything else is kept.
    /// Pure/testable — the instance method just applies it + does the disk I/O.
    nonisolated static func runsAfterClearingFinished(_ runs: [TaskRun], activeRunId: UUID?)
        -> (keep: [TaskRun], removed: [TaskRun]) {
        var keep: [TaskRun] = []
        var removed: [TaskRun] = []
        for r in runs {
            if r.status.isTerminal && r.id != activeRunId { removed.append(r) } else { keep.append(r) }
        }
        return (keep, removed)
    }

    /// Heal stale runs at launch: a run persisted as `.running` was interrupted by
    /// an app quit/crash (nothing is executing it now), so mark it `.failed`.
    /// `.needsApproval` (durable, resumable) and already-terminal runs are left as
    /// is. Pure/testable.
    nonisolated static func reconcileStaleRuns(_ runs: [TaskRun], now: Date) -> [TaskRun] {
        runs.map { run in
            guard run.status == .running else { return run }
            var r = run
            r.status = .failed
            r.finishedAt = r.finishedAt ?? now
            if r.summary == nil { r.summary = "Interrupted — the app quit while this run was in progress." }
            return r
        }
    }

    // MARK: - Running

    /// Fire a task on demand. Returns immediately; the run executes asynchronously
    /// behind the serial queue.
    func runNow(_ task: ScheduledTask) {
        enqueue(task, reason: "manual")
    }

    private func enqueue(_ task: ScheduledTask, reason: String) {
        enqueueWork { [weak self] in await self?.execute(task, reason: reason) }
    }

    private func enqueueWork(_ work: @escaping () async -> Void) {
        queue.append(work)
        drain()
    }

    private func drain() {
        guard !running, !queue.isEmpty else { return }
        // Interactive turns win the single generation slot — wait for them to go
        // idle. That's the in-app chat engine AND the Telegram bridge's engine
        // (e.g. a createTask spawned from a Telegram turn must not run a second
        // engine against the model while that turn is still generating).
        if appState.chatEngine.isGenerating || appState.telegramBridge.isProcessing {
            Task { [weak self] in
                try? await Task.sleep(nanoseconds: 500_000_000)
                self?.drain()
            }
            return
        }
        running = true
        let work = queue.removeFirst()
        Task { [weak self] in
            await work()
            self?.running = false
            self?.drain()
        }
    }

    /// YOLO passes a nil working directory so `ToolExecutor.resolveAndConfine` does
    /// not guard — the documented "no confinement" lever. Other levels confine to
    /// the per-run folder.
    private func workDir(for task: ScheduledTask, runId: UUID) -> String? {
        task.autonomy == .yolo ? nil : TaskPaths.runDir(task.id, runId)
    }

    private func execute(_ task: ScheduledTask, reason: String) async {
        var run = TaskRun(taskId: task.id, status: .running, triggerReason: reason)
        activeRun = run
        appendRun(run, taskId: task.id)

        guard await ensureServerReady(modelPath: task.modelPath) else {
            run.status = .failed
            run.finishedAt = Date()
            run.summary = "No server/model available to run this task."
            finalize(run, taskId: task.id)
            return
        }

        let dir = workDir(for: task, runId: run.id)
        try? FileManager.default.createDirectory(atPath: TaskPaths.runDir(task.id, run.id),
                                                 withIntermediateDirectories: true)

        // Transient hidden session — the agent loop reads/appends through AppState.
        var session = ChatSession(title: "Task: \(task.title)")
        session.mode = .agent
        session.workingDirectory = dir
        session.taskRunId = run.id
        appState.chatSessions.insert(session, at: 0)

        await driveRun(run: run, task: task, sessionId: session.id, workDir: dir, userText: task.goal)
    }

    /// Run a (possibly resumed) session to completion, observing the dedicated
    /// engine's generation flag, then harvest the transcript. If a tool call paused
    /// the run, persist it as `needsApproval` instead of finishing.
    private func driveRun(run: TaskRun, task: ScheduledTask, sessionId: UUID,
                          workDir: String?, userText: String) async {
        var run = run
        let approval = makeApproval(runId: run.id, autonomy: task.autonomy, workDir: workDir)
        let config = ChatTurnEngine.TurnConfig(
            agentMode: true, mcpMode: task.useMCP, enableThinking: false,
            voiceStyle: false, workingDirectory: workDir
        )
        runEngine.runTurn(sessionId: sessionId, userText: userText,
                          images: nil, audio: nil, config: config, approval: approval)
        if runEngine.isGenerating {
            for await generating in runEngine.$isGenerating.values where !generating { break }
        }

        let messages = appState.chatSessions.first { $0.id == sessionId }?.messages ?? []

        // Cancelled by the user mid-run? (cancelRun set the flag + stopped the
        // engine, which is what woke the await above.) Cancel wins over a pending
        // approval — finalize as `.cancelled` so it isn't overwritten as completed.
        if cancelledRunIds.remove(run.id) != nil {
            pendingForRun[run.id] = nil
            saveTranscript(taskId: task.id, runId: run.id, messages: messages)
            appState.chatSessions.removeAll { $0.id == sessionId }
            run.status = .cancelled
            run.finishedAt = Date()
            run.summary = Self.lastAssistantText(messages) ?? "Cancelled by user."
            finalize(run, taskId: task.id)
            return
        }

        // Paused for approval?
        if let pending = pendingForRun[run.id] {
            pendingForRun[run.id] = nil
            let cleaned = Self.stripTrailingDenial(messages, toolCallId: pending.toolCallId)
            saveTranscript(taskId: task.id, runId: run.id, messages: cleaned)
            appState.chatSessions.removeAll { $0.id == sessionId }
            run.status = .needsApproval
            run.pendingApproval = pending
            run.finishedAt = nil
            persistRun(run, taskId: task.id)
            if activeRun?.id == run.id { activeRun = nil }
            TaskNotifier.shared.notifyNeedsApproval(task: task, run: run)
            return
        }

        saveTranscript(taskId: task.id, runId: run.id, messages: messages)
        appState.chatSessions.removeAll { $0.id == sessionId }
        run.status = messages.isEmpty ? .failed : .completed
        run.finishedAt = Date()
        run.summary = Self.lastAssistantText(messages) ?? (run.status == .failed ? "Run produced no output." : nil)
        finalize(run, taskId: task.id)
    }

    /// The headless approval gate: auto-decide per the task's autonomy, and on
    /// `.ask` record a durable pause, cancel the run, and let `driveRun` persist
    /// `needsApproval`. Returning false here is what makes the loop unwind; the
    /// trailing "denied" pair it leaves behind is stripped on save.
    private func makeApproval(runId: UUID, autonomy: TaskAutonomy, workDir: String?)
        -> (APIClient.ToolCall) async -> Bool {
        return { [weak self] tc in
            guard let self else { return false }
            switch ApprovalPolicy.decide(tool: tc.name, autonomy: autonomy,
                                         arguments: tc.arguments, rawArguments: tc.rawArguments,
                                         workingDirectory: workDir) {
            case .allow:
                return true
            case .deny:
                return false
            case .ask(let reason):
                self.pendingForRun[runId] = PendingApproval(
                    toolCallId: tc.id, toolName: tc.name, arguments: tc.arguments,
                    rawArguments: tc.rawArguments, reason: reason, requestedAt: Date())
                self.runEngine.stop()   // unwind the loop; driveRun persists the pause
                return false
            }
        }
    }

    /// Resume a paused run: apply the user's Approve/Deny to the pending tool, then
    /// continue the agent loop on the rehydrated transcript.
    func resume(runId: UUID, approved: Bool) {
        guard let (taskId, run) = findRun(runId),
              run.status == .needsApproval,
              let pending = run.pendingApproval,
              let task = tasks.first(where: { $0.id == taskId }) else { return }
        enqueueWork { [weak self] in await self?.performResume(task: task, run: run, pending: pending, approved: approved) }
    }

    private func performResume(task: ScheduledTask, run: TaskRun,
                               pending: PendingApproval, approved: Bool) async {
        var run = run
        run.status = .running
        run.pendingApproval = nil
        activeRun = run
        persistRun(run, taskId: task.id)

        guard await ensureServerReady(modelPath: task.modelPath) else {
            run.status = .failed; run.finishedAt = Date()
            run.summary = "No server/model available to resume this task."
            finalize(run, taskId: task.id)
            return
        }

        let dir = workDir(for: task, runId: run.id)
        var session = ChatSession(title: "Task: \(task.title)")
        session.mode = .agent
        session.workingDirectory = dir
        session.taskRunId = run.id
        session.messages = transcript(taskId: task.id, runId: run.id)
        appState.chatSessions.insert(session, at: 0)
        let sessionId = session.id

        // Apply the user's decision to the pending tool, feeding the result back as
        // if the loop had produced it, then continue.
        let tc = APIClient.ToolCall(id: pending.toolCallId, name: pending.toolName,
                                    arguments: pending.arguments, rawArguments: pending.rawArguments)
        // The approved tool runs here, before the loop (which would normally start
        // MCP) takes over — so start MCP servers first if this is an MCP task.
        if approved, task.useMCP {
            appState.mcpManager.defaultCwd = dir
            await appState.mcpManager.startEnabled()
        }
        if approved {
            var wd = dir
            let repetition = AgentEngine.RepetitionTracker()
            let result = await AgentEngine.executeToolCall(
                tc, workingDirectory: &wd, repetition: repetition, iteration: 0,
                agentMemory: appState.agentMemory, mcpRouter: appState.mcpManager)
            appendToolResult(sessionId: sessionId, id: result.id, name: result.name,
                             display: "**\(result.name)** → \(String(result.output.prefix(500)))",
                             content: AgentEngine.truncateWithOverflow(result.output, toolCallId: result.id, toolName: result.name))
        } else {
            appendToolResult(sessionId: sessionId, id: tc.id, name: tc.name,
                             display: "**\(tc.name)** → denied by user",
                             content: "Error: user denied this tool call. Do not retry; try a different approach or stop.")
        }

        await driveRun(run: run, task: task, sessionId: sessionId, workDir: dir,
                       userText: "Continue. Use the tool result above. If the task is complete, reply with a short plain-text summary — no tool calls.")
    }

    /// Append a tool result to a run session (display summary + tool-role message),
    /// mirroring how the agent loop records executed tools.
    private func appendToolResult(sessionId: UUID, id: String, name: String, display: String, content: String) {
        var summary = ChatMessage(role: .assistant, content: display)
        summary.isAgentSummary = true
        appState.appendMessage(to: sessionId, message: summary)
        var toolMsg = ChatMessage(role: .system, content: content)
        toolMsg.toolCallId = id
        toolMsg.toolName = name
        appState.appendMessage(to: sessionId, message: toolMsg)
    }

    /// Strip the trailing "denied by user" pair the loop leaves when a pause unwinds.
    nonisolated static func stripTrailingDenial(_ messages: [ChatMessage], toolCallId: String) -> [ChatMessage] {
        var m = messages
        while let last = m.last {
            if last.role == .system, last.toolCallId == toolCallId { m.removeLast(); continue }
            if last.role == .assistant, last.isAgentSummary, last.content.contains("denied by user") { m.removeLast(); continue }
            break
        }
        return m
    }

    private func findRun(_ runId: UUID) -> (taskId: UUID, run: TaskRun)? {
        for task in tasks {
            if let run = runs(for: task.id).first(where: { $0.id == runId }) {
                return (task.id, run)
            }
        }
        return nil
    }

    /// Latest non-empty assistant message, in FULL — the task's actual result.
    /// Used where the whole answer matters (e.g. relaying to Telegram, which
    /// chunks long replies itself). The `prefix(280)` cap in `lastAssistantText`
    /// is purely for the one-line timeline row, so never reuse it here.
    nonisolated static func fullLastAssistantText(_ messages: [ChatMessage]) -> String? {
        for msg in messages.reversed() where msg.role == .assistant {
            let trimmed = msg.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return nil
    }

    /// Latest non-empty assistant message, capped — the task's "result" line in the
    /// timeline. For the full text (relaying, export) use `fullLastAssistantText`.
    nonisolated static func lastAssistantText(_ messages: [ChatMessage]) -> String? {
        fullLastAssistantText(messages).map { String($0.prefix(280)) }
    }

    /// Make sure the server is running with the right model before a run:
    /// - server stopped → start it with the task's pinned model (or the selected one);
    /// - server running a *different* model than the task pins → switch to it
    ///   (hot-swap when enabled, else restart);
    /// - server already running the right model → proceed.
    private func ensureServerReady(modelPath: String?) async -> Bool {
        if appState.server.status == .running {
            // A task can pin a model; if the server is on a different one, switch.
            if let pinned = modelPath, !pinned.isEmpty,
               modelId(appState.server.currentModelPath) != modelId(pinned) {
                return await switchModel(to: pinned)
            }
            return true
        }
        let path = modelPath ?? appState.selectedModelPath
        guard !path.isEmpty else { return false }
        appState.server.start(modelPath: path, options: appState.serverOptions)
        return await awaitHealthy()
    }

    /// Switch the running server to `path`, mirroring AppState's picker: hot-swap
    /// in place when enabled, falling back to a restart (and on stopped servers).
    private func switchModel(to path: String) async -> Bool {
        if appState.hotSwitchEnabled {
            do {
                let id = (path as NSString).lastPathComponent
                let drafter = appState.downloads.recommendedDrafterFromPath(path)?.url.path
                _ = try await appState.server.loadModel(id: id, drafterPath: drafter)
                return true
            } catch {
                // Hot-swap failed (e.g. model not under --model-dir) — restart.
            }
        }
        appState.server.stop()
        appState.server.start(modelPath: path, options: appState.serverOptions)
        return await awaitHealthy()
    }

    private func awaitHealthy() async -> Bool {
        let api = APIClient()
        for _ in 0..<120 {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            if appState.server.status == .running { return true }
            if let ok = try? await api.checkHealth(port: appState.server.port), ok {
                appState.server.forceRunning()
                return true
            }
        }
        return false
    }

    /// Model identity = the directory/file name, matching how `loadModel` keys models.
    private func modelId(_ path: String) -> String {
        (path as NSString).lastPathComponent
    }

    // MARK: - Run persistence

    private func appendRun(_ run: TaskRun, taskId: UUID) {
        var list = runs(for: taskId)
        list.insert(run, at: 0)
        runsByTask[taskId] = list
        saveRuns(taskId, list)
    }

    /// Upsert a run into its task's history without finalize semantics.
    private func persistRun(_ run: TaskRun, taskId: UUID) {
        var list = runs(for: taskId)
        if let idx = list.firstIndex(where: { $0.id == run.id }) {
            list[idx] = run
        } else {
            list.insert(run, at: 0)
        }
        runsByTask[taskId] = list
        saveRuns(taskId, list)
        if activeRun?.id == run.id { activeRun = run }
    }

    private func finalize(_ run: TaskRun, taskId: UUID) {
        persistRun(run, taskId: taskId)
        if let tIdx = tasks.firstIndex(where: { $0.id == taskId }) {
            tasks[tIdx].lastRunAt = run.finishedAt ?? Date()
            saveCatalog()
        }
        if activeRun?.id == run.id { activeRun = nil }
        if let task = tasks.first(where: { $0.id == taskId }) {
            switch run.status {
            case .completed: TaskNotifier.shared.notifyCompleted(task: task, run: run)
            case .failed: TaskNotifier.shared.notifyFailed(task: task, run: run)
            default: break
            }
            // Bridge-created tasks also report back to the Telegram chat that made
            // them — in ADDITION to the desktop notification above.
            if let chatId = task.originTelegramChatId,
               run.status == .completed || run.status == .failed {
                appState.telegramBridge.deliverTaskResult(chatId: chatId, task: task, run: run)
            }
            // One-shot ("run now") tasks created by the agent clean themselves up so
            // they don't pile up as disabled entries in the Tasks list — the result
            // already went to Telegram + the desktop notification above.
            if task.deleteAfterRun { deleteTask(taskId) }
        }
    }

    private func saveTranscript(taskId: UUID, runId: UUID, messages: [ChatMessage]) {
        let dir = TaskPaths.runDir(taskId, runId)
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        guard let data = try? encoder.encode(messages) else { return }
        try? data.write(to: URL(fileURLWithPath: TaskPaths.transcriptFile(taskId, runId)))
    }

    // MARK: - Catalog persistence

    private func loadCatalog() {
        let path = TaskPaths.catalogFile
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let loaded = try? decoder.decode([ScheduledTask].self, from: data) else { return }
        tasks = loaded
    }

    private func saveCatalog() {
        guard let data = try? encoder.encode(tasks) else { return }
        try? data.write(to: URL(fileURLWithPath: TaskPaths.catalogFile))
    }

    private func loadRuns(_ taskId: UUID) -> [TaskRun] {
        let path = TaskPaths.runsFile(taskId)
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let loaded = try? decoder.decode([TaskRun].self, from: data) else { return [] }
        return loaded
    }

    private func saveRuns(_ taskId: UUID, _ runs: [TaskRun]) {
        try? FileManager.default.createDirectory(atPath: TaskPaths.taskDir(taskId), withIntermediateDirectories: true)
        guard let data = try? encoder.encode(runs) else { return }
        try? data.write(to: URL(fileURLWithPath: TaskPaths.runsFile(taskId)))
    }
}
