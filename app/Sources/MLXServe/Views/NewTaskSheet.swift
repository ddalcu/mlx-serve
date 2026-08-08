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
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    // The two things a task IS. Everything below is a default
                    // you can leave alone, which is why they are separated by a
                    // rule rather than stacked at equal weight — the old sheet
                    // gave six sections the same visual priority and read as a
                    // form to fill in rather than two questions to answer.
                    field("What should I do?",
                          hint: "e.g. “Check Hacker News and write me the top AI stories”") {
                        TextEditor(text: $goal)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .padding(6)
                            .frame(minHeight: 84)
                            .background(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .fill(Color.primary.opacity(0.05)))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8, style: .continuous)
                                    .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1))
                    }

                    field("When?") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack(spacing: 6) {
                                ForEach(presets, id: \.0) { preset in
                                    Button(preset.0) { scheduleText = preset.1 }
                                        .buttonStyle(.bordered)
                                        .controlSize(.small)
                                }
                            }
                            TextField("every weekday at 8am · or a cron expression",
                                      text: $scheduleText)
                                .textFieldStyle(.roundedBorder)
                            // The echo is the whole point of typing a schedule in
                            // English: it is the only confirmation that what you
                            // meant is what will run.
                            if let trigger = parsedTrigger {
                                Label(ScheduleParser.describe(trigger),
                                      systemImage: "checkmark.circle.fill")
                                    .font(.caption).foregroundStyle(.green)
                            } else {
                                Label("I couldn't read that schedule — try “every day at 8am”.",
                                      systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption).foregroundStyle(.orange)
                            }
                        }
                    }

                    Divider()

                    field("How much can it do on its own?") {
                        VStack(alignment: .leading, spacing: 8) {
                            Picker("", selection: $autonomy) {
                                ForEach(TaskAutonomy.allCases, id: \.self) { level in
                                    Text(level.shortLabel).tag(level)
                                }
                            }
                            .labelsHidden()
                            .pickerStyle(.segmented)
                            Label(autonomy.blurb,
                                  systemImage: autonomy == .yolo ? "exclamationmark.octagon.fill" : "info.circle")
                                .font(.caption)
                                .foregroundStyle(autonomy == .yolo ? Color.red : .secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }

                    field("Run as", hint: "An agent brings its own prompt, tools, model and workspace.") {
                        Picker("", selection: $agentId) {
                            Text("None (app defaults)").tag(UUID?.none)
                            ForEach(appState.agents.allAgents) { agent in
                                Text(agent.name).tag(UUID?.some(agent.id))
                            }
                        }
                        .labelsHidden()
                    }

                    if !baseModels.isEmpty {
                        field("Model") {
                            Picker("", selection: $modelPath) {
                                Text("Use current model").tag(String?.none)
                                ForEach(baseModels) { model in
                                    Text(model.name).tag(String?.some(model.path))
                                }
                            }
                            .labelsHidden()
                        }
                    }

                    field("MCP tools",
                          hint: "Your enabled MCP servers (configure them in Chat ▸ MCP). Outside Read-only/Workspace, MCP calls pause for approval.") {
                        Toggle(isOn: $useMCP) {
                            Text("Available to this task").font(.subheadline)
                        }
                        .toggleStyle(.switch)
                    }
                }
                .padding(22)
            }
            Divider()
            footer
        }
        .frame(width: 520, height: 640)
    }

    private var header: some View {
        HStack {
            Text(isEditing ? "Edit Task" : "New Task")
                .font(.headline)
            Spacer()
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 14)
    }

    /// Bottom-trailing default button, as every macOS sheet places it.
    private var footer: some View {
        HStack {
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button(isEditing ? "Save" : "Create Task") { submit() }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave)
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 14)
    }

    /// One labelled field: title, control, optional explainer UNDER it. The old
    /// sheet mixed all three orders — some hints above the control, some below,
    /// some inside a Label — which is most of why it read as cluttered.
    @ViewBuilder
    private func field<Content: View>(_ title: String, hint: String? = nil,
                                      @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.semibold))
            content()
            if let hint {
                Text(hint)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
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
