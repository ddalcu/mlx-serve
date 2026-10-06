import SwiftUI

/// The wand chip beside a prompt field's Templates menu. Disabled until there
/// is something to rewrite.
struct PromptEnhanceButton: View {
    let disabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: "wand.and.sparkles")
                Text("Enhance…").font(.app(.body))
            }
            .modifier(PaneChip())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        // A `.plain` button over our own background does not dim itself.
        .opacity(disabled ? 0.4 : 1)
        .help("Rewrite with the chat model")
    }
}

extension String {
    var isBlank: Bool { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

/// The wand sheet: streams the chat model's rewrite into an editable box;
/// Apply hands the edited text back, Try again re-asks, Cancel keeps the
/// original untouched.
struct PromptRewriteSheet: View {
    let title: String
    /// Video only: the clip-length slider beside Try again, seconds. nil = no slider.
    var clip: (initial: Int, max: Int)? = nil
    let request: (Int) -> PromptRewriter.Request
    /// Called on Apply with the slider's seconds, only when it was moved.
    var onApplyClip: ((Int) -> Void)? = nil
    let onApply: (String) -> Void
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var text: String = ""
    @State private var clipSeconds = 0
    @State private var isWriting = false
    @State private var error: String? = nil
    @State private var job: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(L10n.text(title)).font(.app(.headline))
                Spacer()
                if isWriting { ProgressView().controlSize(.small) }
            }
            TextEditor(text: $text)
                .font(.app(.body))
                .frame(minHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            if let error {
                Text(error).font(.app(.caption)).foregroundStyle(.red)
            } else {
                Text("Edit the result, then Apply to replace your text.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
            if let clip {
                HStack {
                    Text("Clip length").font(.app(.caption))
                    Slider(value: Binding(get: { Double(clipSeconds) }, set: { clipSeconds = Int($0.rounded()) }),
                           in: 1...Double(max(2, clip.max)), step: 1)
                    Text(L10n.format("%lld s", Int64(clipSeconds)))
                        .font(.app(.caption).monospacedDigit()).foregroundStyle(.secondary)
                }
                .disabled(isWriting)
            }
            HStack {
                Button { start() } label: { Text("Try again")
                    .font(.app(.body)) }.disabled(isWriting)
                Spacer()
                Button { dismiss() } label: { Text("Cancel")
                    .font(.app(.body)) }.keyboardShortcut(.cancelAction)
                Button {
                    onApply(text)
                    if let clip, clipSeconds != clip.initial { onApplyClip?(clipSeconds) }
                    dismiss()
                } label: { Text("Apply")
                    .font(.app(.body)) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(isWriting || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 520)
        .onAppear { clipSeconds = clip?.initial ?? 0; start() }
        .onDisappear { job?.cancel() }
    }

    private func start() {
        job?.cancel()
        text = ""
        error = nil
        isWriting = true
        job = Task {
            defer { isWriting = false }
            let req = request(clipSeconds)
            do {
                let stream = try await AgentComposer.stream(userText: req.user, systemPrompt: req.system,
                                                            appState: appState, maxTokens: 2048)
                for try await delta in stream {
                    if Task.isCancelled { return }
                    text += delta
                }
                text = PromptRewriter.clean(text)
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
