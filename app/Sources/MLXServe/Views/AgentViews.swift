import SwiftUI

// MARK: - Plan Card

struct PlanCardView: View {
    let plan: AgentPlan
    let results: [StepResult]
    let currentStepIndex: Int?
    let onApprove: () -> Void
    let onReject: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "list.bullet.clipboard")
                    .foregroundStyle(.blue)
                Text("Plan (\(plan.steps.count) steps)")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                statusBadge
            }

            Divider()

            ForEach(Array(plan.steps.enumerated()), id: \.element.id) { index, step in
                HStack(alignment: .top, spacing: 8) {
                    stepIndicator(for: index)
                    Image(systemName: step.tool.icon)
                        .frame(width: 16)
                        .foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(step.description)
                            .font(.caption)
                        Text(step.tool.displayName)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }

            if plan.status == .pending {
                HStack(spacing: 12) {
                    Button(action: onApprove) {
                        Label("Approve", systemImage: "checkmark.circle.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .controlSize(.small)

                    Button(action: onReject) {
                        Label("Reject", systemImage: "xmark.circle.fill")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.small)
                }
                .padding(.top, 4)
            }
        }
        .padding(12)
        .background(Color(.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(borderColor, lineWidth: 1)
        )
    }

    private var borderColor: Color {
        switch plan.status {
        case .pending: .blue.opacity(0.3)
        case .approved, .executing: .orange.opacity(0.3)
        case .completed: .green.opacity(0.3)
        case .rejected, .failed: .red.opacity(0.3)
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        let (text, color): (String, Color) = switch plan.status {
        case .pending: ("Pending", .blue)
        case .approved: ("Approved", .orange)
        case .executing: ("Running", .orange)
        case .completed: ("Done", .green)
        case .rejected: ("Rejected", .red)
        case .failed: ("Failed", .red)
        }
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }

    @ViewBuilder
    private func stepIndicator(for index: Int) -> some View {
        let result = results.first { $0.stepId == plan.steps[index].id }
        ZStack {
            Circle()
                .fill(stepColor(for: index, result: result))
                .frame(width: 18, height: 18)
            if let result {
                Image(systemName: result.status == .success ? "checkmark" : "xmark")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.white)
            } else if currentStepIndex == index {
                ProgressView()
                    .controlSize(.mini)
            } else {
                Text("\(index + 1)")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.white)
            }
        }
    }

    private func stepColor(for index: Int, result: StepResult?) -> Color {
        if let result {
            return result.status == .success ? .green : .red
        }
        if currentStepIndex == index { return .orange }
        return .secondary
    }
}

// MARK: - Tool Result Block

struct ToolResultBlockView: View {
    let step: PlanStep
    let result: StepResult
    @State private var isExpanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            ScrollView {
                Text(truncatedOutput)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 300)

            if let error = result.error, !error.isEmpty {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: step.tool.icon)
                    .foregroundStyle(.secondary)
                Text(step.description)
                    .font(.caption)
                    .lineLimit(1)
                Spacer()
                if result.durationMs > 0 {
                    Text("\(result.durationMs)ms")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Image(systemName: result.status == .success ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(result.status == .success ? .green : .red)
                    .font(.caption)
            }
        }
        .padding(8)
        .background(Color(.controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var truncatedOutput: String {
        let lines = result.output.components(separatedBy: "\n")
        if lines.count > 50 {
            return lines.prefix(50).joined(separator: "\n") + "\n... (\(lines.count - 50) more lines)"
        }
        return result.output
    }
}

// MARK: - Agent Mode Toggle

struct AgentModeToggle: View {
    @Binding var isAgentMode: Bool

    var body: some View {
        Toggle(isOn: $isAgentMode) {
            Image(systemName: "wrench")
                .foregroundStyle(isAgentMode ? .orange : .secondary)
        }
        .toggleStyle(.button)
        .buttonStyle(.borderless)
        .help("Agent mode")
    }
}

// MARK: - Workspace folder picker

/// The workspace-folder picker behind the Agent pill's folder icon (the old
/// name-sized toolbar chip is gone — a variable-width chip in the toolbar
/// either wasted a fixed slot or re-tripped the » eviction bug; the current
/// folder shows in the pill's tooltip instead).
enum WorkspacePicker {
    /// `@MainActor`: a modal panel only runs on the main thread (already true in
    /// practice — `AppActivation` just makes it explicit).
    @MainActor
    static func pickDirectory() -> String? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Select Working Directory"
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return nil }
        return url.path
    }
}
