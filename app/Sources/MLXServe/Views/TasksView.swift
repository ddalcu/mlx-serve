import SwiftUI

/// The Tasks window: a list of scheduled/on-demand agent tasks on the left, and the
/// selected task's run history + transcript on the right. The unattended "claw"
/// surface — create a goal, give it autonomy and (optionally) a schedule, and let it
/// run in the background.
struct TasksView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var scheduler: TaskScheduler

    @State private var selectedTaskId: UUID?
    @State private var showNewTask = false

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedTaskId) {
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
            .navigationSplitViewColumnWidth(min: 240, ideal: 300)
            .toolbar {
                ToolbarItem {
                    Button { showNewTask = true } label: { Image(systemName: "plus") }
                        .help("New task")
                }
            }
        } detail: {
            if let id = selectedTaskId, let task = scheduler.tasks.first(where: { $0.id == id }) {
                TaskDetailView(task: task)
            } else {
                ContentUnavailableView("Select a task",
                                       systemImage: "clock.badge.checkmark",
                                       description: Text("Pick a task to see its runs, or create a new one."))
            }
        }
        .sheet(isPresented: $showNewTask) {
            NewTaskSheet { newTask in
                scheduler.addTask(newTask)
                selectedTaskId = newTask.id
            }
        }
        .onChange(of: appState.pendingTaskDeepLink) { _, taskId in
            if let taskId { selectedTaskId = taskId; appState.pendingTaskDeepLink = nil }
        }
        .onAppear {
            if let taskId = appState.pendingTaskDeepLink {
                selectedTaskId = taskId; appState.pendingTaskDeepLink = nil
            }
        }
    }
}

// MARK: - Sidebar row

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
        case .yolo: "Unrestricted — runs any command anywhere and never asks. Use with care."
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
