import SwiftUI

/// The Tasks surface: a list of scheduled/on-demand agent tasks, and the
/// selected task's run history + transcript. The unattended "claw" surface —
/// create a goal, give it autonomy and (optionally) a schedule, and let it run
/// in the background.
///
/// It has no shell of its own. It was a `Window`, then briefly a two-pane HStack
/// inside the chat window's detail column; it is now the CONTENT and DETAIL
/// columns of that window's own three-column split (`ChatView.tasksSplitView`),
/// because a list of tasks is navigation and belongs beside the app's sidebar
/// rather than inside the area it navigates. Selection therefore lives on
/// `AppState` — neither column can own the other's state.
///
/// **The two columns are two VIEW TYPES, and that is load-bearing** (live crash
/// 2026-08-08): they used to be `taskList` / `taskDetail` computed properties on
/// one `TasksView`, which the split view read as `TasksView().taskList`. That
/// constructs a view VALUE and immediately evaluates a property that touches
/// `@EnvironmentObject` — but SwiftUI populates that storage when it INSTALLS a
/// view in the hierarchy, and this instance never was one, so the first click on
/// Tasks trapped in `TasksView.$appState.getter`. A `.environmentObject(…)` at
/// the call site cannot save it: the modifier decorates the view the property
/// already returned, long after the property read the empty box. An environment
/// reader has to BE the column, not produce it.

/// The middle column: the task list.
struct TaskListPane: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var scheduler: TaskScheduler

    @State private var showNewTask = false

    var body: some View {
        VStack(spacing: 0) {
            header
            taskListBody
        }
        .sheet(isPresented: $showNewTask) {
            NewTaskSheet { newTask in
                scheduler.addTask(newTask)
                appState.selectedTaskId = newTask.id
            }
        }
        // A tapped task notification lands here (the pane is always present
        // while Tasks is up; the detail column may be showing nothing yet).
        .onChange(of: appState.pendingTaskDeepLink) { _, taskId in
            if let taskId { appState.selectedTaskId = taskId; appState.pendingTaskDeepLink = nil }
        }
        .onAppear {
            if let taskId = appState.pendingTaskDeepLink {
                appState.selectedTaskId = taskId; appState.pendingTaskDeepLink = nil
            }
        }
    }

    /// The column's own title row. The create control rides it rather than a
    /// ToolbarItem: this is a middle column of the chat window's split, and that
    /// window's toolbar belongs to the chat.
    ///
    /// A plain row ABOVE the list, not a `safeAreaInset` over it. The inset
    /// needed an opaque backdrop so rows didn't scroll through the title, that
    /// backdrop was `.bar`, and a `.bar` draws a separator along its edge — the
    /// horizontal rule under the title. Nothing scrolls under a sibling, so the
    /// backdrop and its rule are both simply unnecessary.
    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Tasks")
                .font(.title3.weight(.semibold))
            Spacer()
            NewTaskButton { showNewTask = true }
        }
        .padding(.leading, 16)
        .padding(.trailing, 12)
        .padding(.top, 14)
        .padding(.bottom, 10)
    }

    private var taskListBody: some View {
        List(selection: $appState.selectedTaskId) {
                if scheduler.tasks.isEmpty {
                    Text("No tasks yet.\nTap + to create one.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 24)
                        .listRowSeparator(.hidden)
                }
                ForEach(scheduler.tasks) { task in
                    TaskRow(task: task, isRunning: scheduler.activeRun?.taskId == task.id)
                        .tag(task.id)
                }
            }
            .scrollContentBackground(.hidden)
    }
}

/// The `+` in the Tasks column header.
///
/// `.borderless` gave a bare glyph with no target to aim at and nothing under
/// the pointer on hover. This is the shape the rest of the app's icon controls
/// use (a `.plain` button drawing its own fill), so it reads as a control and
/// answers the click before it happens — and a bordered style is declined for
/// the reason `ChatMetrics` records: a bordered control keeps its intrinsic
/// size and merely centers inside whatever frame it is given.
private struct NewTaskButton: View {
    let action: () -> Void
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            Image(systemName: "plus")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(Color.primary)
                .frame(width: 24, height: 24)
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.primary.opacity(hovering ? 0.14 : 0.07)))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .help("New task")
    }
}

/// The detail column: the selected task's runs and transcript.
struct TaskDetailPane: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var scheduler: TaskScheduler

    var body: some View {
        if let id = appState.selectedTaskId, let task = scheduler.tasks.first(where: { $0.id == id }) {
            TaskDetailView(task: task)
        } else {
            ContentUnavailableView("Select a task",
                                   systemImage: "clock.badge.checkmark",
                                   description: Text("Pick a task to see its runs, or create a new one."))
        }
    }
}

private struct TaskRow: View {
    let task: ScheduledTask
    let isRunning: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                if isRunning {
                    ProgressView().controlSize(.small)
                }
                Text(task.title)
                    .font(.body.weight(.medium))
                    .lineLimit(1)
                Spacer()
                if !task.enabled {
                    Image(systemName: "pause.circle").foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 6) {
                Image(systemName: "clock").font(.caption2)
                Text(ScheduleParser.describe(task.trigger))
                    .font(.caption)
                AutonomyBadge(autonomy: task.autonomy)
            }
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .padding(.vertical, 2)
    }
}

private struct AutonomyBadge: View {
    let autonomy: TaskAutonomy
    var body: some View {
        Text(autonomy.shortLabel)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 5).padding(.vertical, 1)
            .background(autonomy.tint.opacity(0.18), in: Capsule())
            .foregroundStyle(autonomy.tint)
    }
}

// MARK: - Detail (header + run history)

private struct TaskDetailView: View {
    @EnvironmentObject var scheduler: TaskScheduler
    @EnvironmentObject var server: ServerManager
    let task: ScheduledTask

    @State private var showEdit = false

    private var runs: [TaskRun] { scheduler.runs(for: task.id) }
    private var isRunning: Bool { scheduler.activeRun?.taskId == task.id }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text(task.title).font(.title2.weight(.semibold))
                    Text(task.goal)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    HStack(spacing: 10) {
                        Label(ScheduleParser.describe(task.trigger), systemImage: "clock")
                        AutonomyBadge(autonomy: task.autonomy)
                        if task.useMCP {
                            Label("MCP", systemImage: "puzzlepiece.extension")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                // Actions
                HStack(spacing: 10) {
                    Button {
                        scheduler.runNow(task)
                    } label: {
                        Label(isRunning ? "Running…" : "Run now", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRunning)

                    Toggle("Enabled", isOn: Binding(
                        get: { task.enabled },
                        set: { scheduler.setEnabled(task.id, $0) }
                    ))
                    .toggleStyle(.switch)
                    .controlSize(.small)

                    Spacer()

                    Button { showEdit = true } label: {
                        Label("Edit", systemImage: "pencil")
                    }
                    .help("Edit task")

                    Button(role: .destructive) {
                        scheduler.deleteTask(task.id)
                    } label: { Image(systemName: "trash") }
                    .help("Delete task")
                }

                if server.status != .running {
                    Label("The server isn't running — the task will start it on its first run.",
                          systemImage: "info.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Divider()

                // Run history
                HStack {
                    Text("Runs").font(.headline)
                    Spacer()
                    if runs.contains(where: { $0.status.isTerminal && scheduler.activeRun?.id != $0.id }) {
                        Button("Clear finished") { scheduler.clearFinishedRuns(taskId: task.id) }
                            .buttonStyle(.link)
                            .font(.caption)
                            .help("Delete all completed, failed and cancelled runs")
                    }
                }
                if runs.isEmpty {
                    Text("No runs yet. Tap Run now to try it.")
                        .font(.callout).foregroundStyle(.secondary)
                } else {
                    ForEach(runs) { run in
                        RunRow(task: task, run: run)
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .id(task.id)
        .sheet(isPresented: $showEdit) {
            NewTaskSheet(existing: task) { updated in
                scheduler.updateTask(updated)
            }
        }
    }
}

// MARK: - One run (expandable to its transcript)

private struct RunRow: View {
    @EnvironmentObject var scheduler: TaskScheduler
    let task: ScheduledTask
    let run: TaskRun

    @State private var expanded = false
    @State private var transcript: [ChatMessage] = []

    /// The live, currently-executing run can't be deleted out from under the engine.
    private var isLive: Bool { scheduler.activeRun?.id == run.id }

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                if run.status == .needsApproval, let pending = run.pendingApproval {
                    ApprovalCard(task: task, run: run, pending: pending)
                }
                ForEach(transcript) { msg in
                    MessageBubble(message: msg)
                }
                HStack {
                    Button {
                        NSWorkspace.shared.open(URL(fileURLWithPath: TaskPaths.runDir(task.id, run.id)))
                    } label: {
                        Label("Reveal artifacts in Finder", systemImage: "folder")
                    }
                    .buttonStyle(.link)
                    Spacer()
                    if !run.status.isTerminal {
                        Button(role: .destructive) {
                            scheduler.cancelRun(taskId: task.id, runId: run.id)
                        } label: { Label("Stop", systemImage: "stop.circle") }
                        .buttonStyle(.link)
                    }
                    Button(role: .destructive) {
                        scheduler.deleteRun(taskId: task.id, runId: run.id)
                    } label: { Label("Delete", systemImage: "trash") }
                    .buttonStyle(.link)
                    .disabled(isLive)
                    .help(isLive ? "Stop the run before deleting it" : "Delete this run and its artifacts")
                }
                .font(.caption)
            }
            .padding(.top, 6)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: run.status.iconName)
                    .foregroundStyle(run.status.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text(run.summary ?? run.status.label)
                        .font(.callout)
                        .lineLimit(2)
                    Text("\(run.startedAt.formatted(date: .abbreviated, time: .shortened)) · \(run.triggerReason)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if run.status == .running { ProgressView().controlSize(.small) }
            }
        }
        .contextMenu {
            if !run.status.isTerminal {
                Button(role: .destructive) {
                    scheduler.cancelRun(taskId: task.id, runId: run.id)
                } label: { Label("Stop run", systemImage: "stop.circle") }
            }
            Button(role: .destructive) {
                scheduler.deleteRun(taskId: task.id, runId: run.id)
            } label: { Label("Delete run", systemImage: "trash") }
            .disabled(isLive)
        }
        .onChange(of: expanded) { _, now in
            if now, transcript.isEmpty {
                transcript = scheduler.transcript(taskId: task.id, runId: run.id)
            }
        }
        .onAppear { if run.status == .needsApproval { expanded = true } }
    }
}

/// Inline Approve/Deny card shown for a paused run.
private struct ApprovalCard: View {
    @EnvironmentObject var scheduler: TaskScheduler
    let task: ScheduledTask
    let run: TaskRun
    let pending: PendingApproval

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Wants to run “\(pending.toolName)”", systemImage: "hand.raised.fill")
                .font(.subheadline.weight(.semibold))
            Text(pending.reason).font(.caption).foregroundStyle(.secondary)
            if !pending.arguments.isEmpty {
                Text(pending.arguments.map { "\($0.key): \($0.value)" }.sorted().joined(separator: "\n"))
                    .font(.caption.monospaced())
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
            HStack {
                Button("Deny") { scheduler.resume(runId: run.id, approved: false) }
                Button("Approve") { scheduler.resume(runId: run.id, approved: true) }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(10)
        .background(Color.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Presentation helpers

extension TaskAutonomy {
    var shortLabel: String {
        switch self {
        case .readOnly: "read-only"
        case .workspace: "workspace"
        case .fullAuto: "full auto"
        case .yolo: "YOLO"
        }
    }
    var tint: Color {
        switch self {
        case .readOnly: .green
        case .workspace: .blue
        case .fullAuto: .orange
        case .yolo: .red
        }
    }
    var blurb: String {
        switch self {
        case .readOnly: "Can browse, search and read. Pauses before changing anything."
        case .workspace: "Can also create and edit files inside the task's own folder."
        case .fullAuto: "Can run shell commands too. File writes still stay in the folder."
        case .yolo: "Never asks — every tool auto-approved, shell can run anything. Files go to your default agent workspace. Use with care."
        }
    }
}

extension RunStatus {
    var label: String {
        switch self {
        case .scheduled: "Scheduled"
        case .running: "Running…"
        case .completed: "Completed"
        case .failed: "Failed"
        case .needsApproval: "Waiting for approval"
        case .cancelled: "Cancelled"
        }
    }
    var iconName: String {
        switch self {
        case .scheduled: "clock"
        case .running: "play.circle"
        case .completed: "checkmark.circle.fill"
        case .failed: "xmark.octagon.fill"
        case .needsApproval: "hand.raised.circle.fill"
        case .cancelled: "minus.circle"
        }
    }
    var tint: Color {
        switch self {
        case .completed: .green
        case .failed: .red
        case .needsApproval: .orange
        case .running: .blue
        default: .secondary
        }
    }
}
