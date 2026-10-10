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
    /// Video only: the clip-length slider beside Try again, in seconds.
    struct ClipLength {
        var initial: Int
        var range: ClosedRange<Int>
        /// A line under the slider for a length (the video pane: when it plans a storyboard).
        var note: (Int) -> String? = { _ in nil }
    }

    let title: String
    var clip: ClipLength? = nil
    let request: (Int) -> PromptRewriter.Request
    /// Called on Apply with the slider's seconds, only when it was moved.
    var onApplyClip: ((Int) -> Void)? = nil
    /// Why the text at this length cannot be applied, or nil.
    var applyError: ((String, Int) -> String?)? = nil
    let onApply: (String) -> Void
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var text: String = ""
    @State private var clipSeconds = 0
    /// The length the text in the box was written for.
    @State private var writtenSeconds = 0
    @State private var isWriting = false
    @State private var progress = RewriteProgress()
    /// Set when there is a first frame the chat model cannot see.
    @State private var blindToFirstFrame = false
    @State private var startedAt = Date()
    @State private var error: String? = nil
    @State private var job: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(L10n.text(title)).font(.app(.headline))
                Spacer()
                if isWriting {
                    ProgressView().controlSize(.small)
                    TimelineView(.periodic(from: startedAt, by: 1)) { context in
                        Text(verbatim: progress.label + " " + Duration.seconds(context.date.timeIntervalSince(startedAt))
                                .formatted(.time(pattern: .minuteSecond)))
                            .font(.app(.caption).monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
            }
            TextEditor(text: $text)
                .font(.app(.body))
                .frame(minHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
                .overlay(alignment: .topLeading) {
                    // The think, until the answer starts: proof the model is working.
                    if isWriting, text.isEmpty, !progress.thought.isEmpty {
                        Text(verbatim: String(progress.thought.suffix(1200)))
                            .font(.app(.caption)).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                            .padding(8)
                            .clipped()
                            .allowsHitTesting(false)
                    }
                }
            if let error = error ?? (isWriting ? nil : applyError?(text, writtenSeconds)) {
                Text(error).font(.app(.caption)).foregroundStyle(.red)
            } else if blindToFirstFrame {
                Text("Your chat model can't see images, so this was written without your first frame and the video may cut away from it. A vision chat model fixes that.")
                    .font(.app(.caption)).foregroundStyle(.orange)
            } else {
                Text("Edit the result, then Apply to replace your text.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
            if let clip {
                HStack {
                    Text("Clip length").font(.app(.caption))
                    Slider(value: Binding(get: { Double(clipSeconds) }, set: { clipSeconds = Int($0.rounded()) }),
                           in: Double(clip.range.lowerBound)...Double(max(clip.range.lowerBound + 1, clip.range.upperBound)),
                           step: 1)
                    Text(verbatim: Storyboard.lengthLabel(clipSeconds))
                        .font(.app(.caption).monospacedDigit()).foregroundStyle(.secondary)
                }
                .disabled(isWriting)
                if let note = clip.note(clipSeconds) {
                    Text(note).font(.app(.caption2)).foregroundStyle(.secondary)
                }
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
                    .disabled(isWriting || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || applyError?(text, writtenSeconds) != nil)
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
        progress = RewriteProgress()
        startedAt = Date()
        isWriting = true
        job = Task {
            defer { isWriting = false }
            writtenSeconds = clipSeconds
            let req = request(clipSeconds)
            let sees = req.firstFrame != nil && AgentComposer.seesImages(appState: appState)
            blindToFirstFrame = req.firstFrame != nil && !sees
            do {
                for try await event in AgentComposer.events(userText: req.user(seeingImage: sees), systemPrompt: req.system,
                                                            image: sees ? req.firstFrame : nil,
                                                            appState: appState, maxTokens: req.maxTokens) {
                    if Task.isCancelled { return }
                    progress.apply(event)
                    if case .content = event { text = progress.text }
                }
                if Task.isCancelled { return }
                text = PromptRewriter.clean(text)
                error = progress.emptyReason
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}

/// What the Enhance sheet says while the chat model works on a rewrite.
struct RewriteProgress: Equatable {
    enum Stage: Equatable { case loading(String), waiting, thinking, writing }

    var stage: Stage = .waiting
    var text = ""
    var thought = ""
    var truncated = false

    mutating func apply(_ event: AgentComposer.Event) {
        switch event {
        case .loading(let name): stage = .loading(name)
        case .sent: stage = .waiting
        case .reasoning(let delta): thought += delta; stage = .thinking
        case .content(let delta):
            text += delta
            // A blank lead-in is not the answer starting.
            if !text.isBlank { stage = .writing }
        case .truncated: truncated = true
        }
    }

    var label: String {
        switch stage {
        case .loading(let name): return L10n.format("Loading %@…", name)
        case .waiting: return L10n.text("Waiting for the model…")
        case .thinking: return L10n.text("Thinking…")
        case .writing: return L10n.text("Writing…")
        }
    }

    /// Why a finished reply left nothing to apply, or nil.
    var emptyReason: String? {
        guard PromptRewriter.clean(text).isEmpty else { return nil }
        return truncated && !thought.isEmpty
            ? L10n.text("The model used its whole token budget thinking and wrote nothing. Try again.")
            : L10n.text("The model finished without writing anything. Try again.")
    }
}
