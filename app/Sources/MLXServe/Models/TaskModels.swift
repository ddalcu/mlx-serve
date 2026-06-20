import Foundation

// MARK: - Scheduled Tasks data model
//
// A "scheduled task" is an agentic goal the user wants run unattended — on a
// schedule and/or on demand. It reuses the same agent loop as interactive chat
// (`ChatTurnEngine`), but with no chat window: runs are driven by `TaskScheduler`
// and their transcripts are persisted out of line. See the plan in
// ~/.claude/plans for the full design.
//
// Storage layout under ~/.mlx-serve/tasks/:
//   tasks.json                       -> [ScheduledTask]  (the catalog)
//   <taskId>/runs.json               -> [TaskRun]        (per-task timeline index, metadata only)
//   <taskId>/<runId>/transcript.json -> [ChatMessage]    (lazy-loaded when a run is opened)
//   <taskId>/<runId>/                -> the run's artifact folder + confinement root

/// When a task should fire. v1 is time-based only; `fileWatch`/`webhook` are v2
/// (declared as a note, not implemented) so the model can grow without a migration.
enum TaskTrigger: Codable, Equatable {
    /// Every N seconds (the UI offers minutes/hours, stored as seconds).
    case interval(seconds: Int)
    /// Every day at a wall-clock time.
    case dailyAt(hour: Int, minute: Int)
    /// On the given weekdays (1 = Sunday … 7 = Saturday, `Calendar` convention) at a time.
    case weekly(weekdays: Set<Int>, hour: Int, minute: Int)
    /// A raw 5-field cron expression (advanced; parsed by ScheduleParser).
    case cron(expression: String)
    // v2: case fileWatch(path: String), case webhook(token: String)
}

/// How much the task is trusted to act without asking. Ascending autonomy.
enum TaskAutonomy: String, Codable, CaseIterable {
    /// Browse/search/read only. Any write or shell call pauses for approval.
    case readOnly
    /// Auto-approve anything that stays inside the run's folder; escapes pause.
    case workspace
    /// Auto-approve all tools (shell included) while file writes stay confined.
    case fullAuto
    /// Unrestricted: every tool auto-approved, no path confinement, never asks.
    case yolo
}

/// Lifecycle of a single run.
enum RunStatus: String, Codable {
    case scheduled      // created, not yet started
    case running        // generation in flight
    case completed      // finished, produced a final answer
    case failed         // errored (server down, exception, etc.)
    case needsApproval  // paused mid-run waiting on the user (Phase 3)
    case cancelled      // user-stopped

    /// A finished state: the run is no longer (and never again) executing, so it's
    /// safe to delete and shouldn't show a spinner. `running`/`scheduled` are live;
    /// `needsApproval` is paused-but-resumable, so neither counts as terminal.
    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled: return true
        case .scheduled, .running, .needsApproval: return false
        }
    }
}

/// A user-defined recurring/agentic task. Codable for the `tasks.json` catalog.
struct ScheduledTask: Identifiable, Codable, Equatable {
    let id: UUID
    var title: String                 // short label, derived from goal but editable
    var goal: String                  // free-text prompt -> the run's first user message
    var trigger: TaskTrigger
    var scheduleText: String?         // verbatim NL the user typed, for re-display/edit
    var autonomy: TaskAutonomy
    var modelPath: String?            // nil = use AppState.selectedModelPath
    var useMCP: Bool                  // expose the user's enabled MCP servers to this task
    var enabled: Bool
    var catchUpMissed: Bool           // run once on wake if a slot was missed while asleep
    var createdAt: Date
    var lastRunAt: Date?
    var nextFireAt: Date?             // cached soonest fire; recomputed by the scheduler
    /// nil = each run gets its own isolated folder (default). Non-nil = a persistent
    /// shared workspace across runs (Phase 4 toggle).
    var workingDirectory: String?
    /// Telegram chat id that created this task via the bot's `createTask` tool.
    /// Non-nil → each finished run is also pushed back to that chat (in addition
    /// to the desktop notification). nil for tasks created in the app UI.
    var originTelegramChatId: Int64?
    /// One-shot "run now" tasks (the createTask tool with no schedule) set this so
    /// the scheduler removes them from the catalog once their run finishes, instead
    /// of leaving a disabled entry cluttering the Tasks list.
    var deleteAfterRun: Bool

    init(id: UUID = UUID(),
         title: String,
         goal: String,
         trigger: TaskTrigger,
         scheduleText: String? = nil,
         autonomy: TaskAutonomy = .workspace,
         modelPath: String? = nil,
         useMCP: Bool = false,
         enabled: Bool = true,
         catchUpMissed: Bool = true,
         createdAt: Date = Date(),
         lastRunAt: Date? = nil,
         nextFireAt: Date? = nil,
         workingDirectory: String? = nil,
         originTelegramChatId: Int64? = nil,
         deleteAfterRun: Bool = false) {
        self.id = id
        self.title = title
        self.goal = goal
        self.trigger = trigger
        self.scheduleText = scheduleText
        self.autonomy = autonomy
        self.modelPath = modelPath
        self.useMCP = useMCP
        self.enabled = enabled
        self.catchUpMissed = catchUpMissed
        self.createdAt = createdAt
        self.lastRunAt = lastRunAt
        self.nextFireAt = nextFireAt
        self.workingDirectory = workingDirectory
        self.originTelegramChatId = originTelegramChatId
        self.deleteAfterRun = deleteAfterRun
    }

    enum CodingKeys: String, CodingKey {
        case id, title, goal, trigger, scheduleText, autonomy, modelPath, useMCP
        case enabled, catchUpMissed, createdAt, lastRunAt, nextFireAt, workingDirectory
        case originTelegramChatId, deleteAfterRun
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        goal = try c.decode(String.self, forKey: .goal)
        trigger = try c.decode(TaskTrigger.self, forKey: .trigger)
        scheduleText = try c.decodeIfPresent(String.self, forKey: .scheduleText)
        autonomy = try c.decodeIfPresent(TaskAutonomy.self, forKey: .autonomy) ?? .workspace
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath)
        useMCP = try c.decodeIfPresent(Bool.self, forKey: .useMCP) ?? false
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        catchUpMissed = try c.decodeIfPresent(Bool.self, forKey: .catchUpMissed) ?? true
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        lastRunAt = try c.decodeIfPresent(Date.self, forKey: .lastRunAt)
        nextFireAt = try c.decodeIfPresent(Date.self, forKey: .nextFireAt)
        workingDirectory = try c.decodeIfPresent(String.self, forKey: .workingDirectory)
        originTelegramChatId = try c.decodeIfPresent(Int64.self, forKey: .originTelegramChatId)
        deleteAfterRun = try c.decodeIfPresent(Bool.self, forKey: .deleteAfterRun) ?? false
    }
}

/// A tool call that tripped the autonomy ceiling and is waiting on the user.
struct PendingApproval: Codable, Equatable {
    let toolCallId: String    // original APIClient.ToolCall.id, to reconstruct on resume
    let toolName: String
    let arguments: [String: String]
    let rawArguments: String
    let reason: String        // why it tripped (from ApprovalPolicy)
    let requestedAt: Date
}

/// One execution of a task. Persisted in `<taskId>/runs.json` — metadata only;
/// the message transcript lives in `<taskId>/<runId>/transcript.json`.
struct TaskRun: Identifiable, Codable, Equatable {
    let id: UUID
    let taskId: UUID
    var startedAt: Date
    var finishedAt: Date?
    var status: RunStatus
    var triggerReason: String     // "scheduled" | "manual" | "catch-up"
    var summary: String?          // final assistant text, for the timeline row
    var pendingApproval: PendingApproval?

    init(id: UUID = UUID(),
         taskId: UUID,
         startedAt: Date = Date(),
         finishedAt: Date? = nil,
         status: RunStatus = .scheduled,
         triggerReason: String = "manual",
         summary: String? = nil,
         pendingApproval: PendingApproval? = nil) {
        self.id = id
        self.taskId = taskId
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.status = status
        self.triggerReason = triggerReason
        self.summary = summary
        self.pendingApproval = pendingApproval
    }
}

// MARK: - On-disk paths

/// Centralizes the ~/.mlx-serve/tasks/ layout so the scheduler, store, and UI agree.
enum TaskPaths {
    static let root: String = {
        let path = NSString(string: "~/.mlx-serve/tasks").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }()

    static var catalogFile: String { (root as NSString).appendingPathComponent("tasks.json") }

    static func taskDir(_ taskId: UUID) -> String {
        (root as NSString).appendingPathComponent(taskId.uuidString)
    }

    static func runsFile(_ taskId: UUID) -> String {
        (taskDir(taskId) as NSString).appendingPathComponent("runs.json")
    }

    static func runDir(_ taskId: UUID, _ runId: UUID) -> String {
        (taskDir(taskId) as NSString).appendingPathComponent(runId.uuidString)
    }

    static func transcriptFile(_ taskId: UUID, _ runId: UUID) -> String {
        (runDir(taskId, runId) as NSString).appendingPathComponent("transcript.json")
    }
}
