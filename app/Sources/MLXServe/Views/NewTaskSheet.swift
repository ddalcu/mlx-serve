import SwiftUI

/// Create-a-task sheet. Plain-language first: type a goal, type when it should run
/// ("every weekday at 8am") or tap a preset, and the parsed schedule is echoed back
/// live. Pick an autonomy level, then Save. (Run a saved task on demand with the
/// Run-now button in the Tasks window.)
struct NewTaskSheet: View {
    /// The task being edited, or nil to create a new one.
    var existing: ScheduledTask?
    let onSubmit: (ScheduledTask) -> Void

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState

    @State private var goal: String
    @State private var scheduleText: String
    @State private var autonomy: TaskAutonomy
    @State private var modelPath: String?   // nil = use the currently-selected model
    @State private var useMCP: Bool
    @State private var agentId: UUID?

    init(existing: ScheduledTask? = nil, onSubmit: @escaping (ScheduledTask) -> Void) {
        self.existing = existing
        self.onSubmit = onSubmit
        _goal = State(initialValue: existing?.goal ?? "")
        _scheduleText = State(initialValue: existing?.scheduleText
            ?? existing.map { ScheduleParser.describe($0.trigger) }
            ?? "every day at 9am")
        _autonomy = State(initialValue: existing?.autonomy ?? .workspace)
        _modelPath = State(initialValue: existing?.modelPath)
        _useMCP = State(initialValue: existing?.useMCP ?? false)
        _agentId = State(initialValue: existing?.agentId)
    }

    private var baseModels: [LocalModel] {
        appState.localModels.filter { $0.isChatPickable }
    }

    private var isEditing: Bool { existing != nil }

    private var parsedTrigger: TaskTrigger? { ScheduleParser.parse(scheduleText) }
    private var canSave: Bool {
        !goal.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && parsedTrigger != nil
    }

    private let presets: [(String, String)] = [
        ("Hourly", "every hour"),
        ("Daily", "every day at 9am"),
        ("Weekdays", "every weekday at 8am"),
        ("Weekly", "every monday at 9am"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(isEditing ? "Edit Task" : "New Task").font(.title2.weight(.semibold))

            // Goal
            VStack(alignment: .leading, spacing: 6) {
                Text("What should I do?").font(.subheadline.weight(.medium))
                TextEditor(text: $goal)
                    .font(.body)
                    .frame(minHeight: 80)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                if goal.isEmpty {
                    Text("e.g. “Check Hacker News and write me the top AI stories”")
                        .font(.caption).foregroundStyle(.tertiary)
                }
            }

            // Schedule
            VStack(alignment: .leading, spacing: 6) {
                Text("When?").font(.subheadline.weight(.medium))
                HStack(spacing: 6) {
                    ForEach(presets, id: \.0) { preset in
                        Button(preset.0) { scheduleText = preset.1 }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                    }
                }
                TextField("every weekday at 8am · or a cron expression", text: $scheduleText)
                    .textFieldStyle(.roundedBorder)
                if let trigger = parsedTrigger {
                    Label("\(ScheduleParser.describe(trigger))", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                } else {
                    Label("I couldn't read that schedule — try “every day at 8am”.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption).foregroundStyle(.orange)
                }
            }

            // Autonomy
            VStack(alignment: .leading, spacing: 6) {
                Text("How much can it do on its own?").font(.subheadline.weight(.medium))
                Picker("", selection: $autonomy) {
                    ForEach(TaskAutonomy.allCases, id: \.self) { level in
                        Text(level.shortLabel).tag(level)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                Label(autonomy.blurb, systemImage: autonomy == .yolo ? "exclamationmark.octagon.fill" : "info.circle")
                    .font(.caption)
                    .foregroundStyle(autonomy == .yolo ? Color.red : .secondary)
            }

            // MCP tools
            VStack(alignment: .leading, spacing: 2) {
                Toggle(isOn: $useMCP) {
                    Text("Use MCP tools").font(.subheadline.weight(.medium))
                }
                .toggleStyle(.switch)
                Text("Give this task your enabled MCP servers (configure them in Chat → MCP). Outside Read-only/Workspace, MCP calls pause for approval.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            // Agent (optional persona; supplies the model + workspace below)
            VStack(alignment: .leading, spacing: 6) {
                Text("Agent").font(.subheadline.weight(.medium))
                Picker("", selection: $agentId) {
                    Text("None (app defaults)").tag(UUID?.none)
                    ForEach(appState.agents.allAgents) { agent in
                        Text(agent.name).tag(UUID?.some(agent.id))
                    }
                }
                .labelsHidden()
                Text("Run this task as one of your agents — its prompt, tools, model and workspace.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            // Model (optional pin)
            if !baseModels.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Model").font(.subheadline.weight(.medium))
                    Picker("", selection: $modelPath) {
                        Text("Use current model").tag(String?.none)
                        ForEach(baseModels) { model in
                            Text(model.name).tag(String?.some(model.path))
                        }
                    }
                    .labelsHidden()
                }
            }

            Spacer(minLength: 0)

            // Actions
            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") { submit() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canSave)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 480, height: 660)
    }

    private func submit() {
        guard let trigger = parsedTrigger else { return }
        let trimmedGoal = goal.trimmingCharacters(in: .whitespacesAndNewlines)
        let task: ScheduledTask
        if let existing {
            // Preserve identity, history and run state; update editable fields.
            task = ScheduledTask(
                id: existing.id,
                title: TaskScheduler.deriveTitle(from: trimmedGoal),
                goal: trimmedGoal,
                trigger: trigger,
                scheduleText: scheduleText,
                autonomy: autonomy,
                modelPath: modelPath,
                agentId: agentId,
                useMCP: useMCP,
                enabled: existing.enabled,
                catchUpMissed: existing.catchUpMissed,
                createdAt: existing.createdAt,
                lastRunAt: existing.lastRunAt,
                nextFireAt: nil,           // recomputed by the scheduler on update
                workingDirectory: existing.workingDirectory
            )
        } else {
            task = ScheduledTask(
                title: TaskScheduler.deriveTitle(from: trimmedGoal),
                goal: trimmedGoal,
                trigger: trigger,
                scheduleText: scheduleText,
                autonomy: autonomy,
                modelPath: modelPath,
                agentId: agentId,
                useMCP: useMCP
            )
        }
        onSubmit(task)
        dismiss()
    }
}
