import SwiftUI
import AppKit

/// Menu-bar tray content for the persistent voice assistant. Rendered as its own
/// section in the status popover and bound to the app-level
/// `VoiceModeController`, so it works with **no chat window open**. When this
/// popover is closed, feedback is audio-only — open it for status + controls.
struct VoiceTrayPanel: View {
    @ObservedObject var voice: VoiceModeController
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header

            // Non-invasive setup notice: shown when enabling Voice hit a missing
            // prerequisite. Visible even though `isActive` is false (the toggle
            // bounces back off), so the user learns why + can jump to Settings.
            if let issue = voice.setupIssue {
                setupNotice(issue)
            }

            if voice.isActive {
                statusLine
                chips
                if voice.agentMode || voice.mcpMode {
                    autoApproveRow
                }
                controls
                if let req = voice.pendingApproval {
                    approvalCard(req)
                }
            }
        }
    }

    // MARK: Header (master toggle)

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(voice.isActive ? Color.accentColor : .secondary)
            Text("Voice")
                .font(.subheadline.weight(.medium))
            Spacer()
            Toggle("", isOn: activeBinding)
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                .disabled(appState.server.status != .running)
        }
        .help("Hands-free voice assistant — talk to the model with no chat window required. Speech-to-text and text-to-speech run locally on your Mac. When this popover is closed, feedback is audio-only; reopen it for status and controls.")
    }

    /// On → ensure a session exists, then start listening. Off → tear down.
    private var activeBinding: Binding<Bool> {
        Binding(
            get: { voice.isActive },
            set: { on in
                if on {
                    if appState.activeChatId == nil
                        || !appState.chatSessions.contains(where: { $0.id == appState.activeChatId }) {
                        _ = appState.newChatSession()
                    }
                    Task { _ = await voice.begin() }
                } else {
                    voice.end()
                }
            }
        )
    }

    // MARK: Setup notice (missing prerequisite)

    @ViewBuilder
    private func setupNotice(_ issue: VoicePreflight.Issue) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                Text(VoicePreflight.shortMessage(for: issue))
                    .font(.caption.weight(.semibold))
            }
            Text(VoicePreflight.detail(for: issue))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button(VoicePreflight.actionLabel(for: issue)) {
                if let url = URL(string: VoicePreflight.settingsURLString(for: issue)) {
                    NSWorkspace.shared.open(url)
                }
            }
            .controlSize(.small)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.10))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: Status line

    private var statusLine: some View {
        HStack(spacing: 8) {
            // STATIC dot — color alone encodes the state. Do NOT animate this in
            // the tray: a continuously-redrawing view (a `repeatForever`
            // animation *or* a running `TimelineView(.animation)`) inside this
            // LSUIElement app's MenuBarExtra(.window) popover starves SwiftUI
            // Button hit-testing and wedges every tray button, while the model
            // Picker / voice Menu keep working from their own NSMenu tracking
            // loop. The breathe lives only in the in-window orb. See `VoiceTrayDot`.
            Circle()
                .fill(dotColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.caption.weight(.medium))
                .foregroundStyle(isError ? .red : .secondary)
            if !voice.partialTranscript.isEmpty {
                Text("“\(voice.partialTranscript)”")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: Mode chips

    private var chips: some View {
        HStack(spacing: 6) {
            chip("Hey Loki", system: "mic.circle", on: voice.requireWakeWord) {
                voice.requireWakeWord.toggle()
            }
            chip("Agent", system: "wrench", on: voice.agentMode) {
                voice.agentMode.toggle()
                if voice.agentMode { voice.enableThinking = false }
            }
            chip("MCP", system: "puzzlepiece.extension", on: voice.mcpMode) {
                voice.mcpMode.toggle()
            }
            chip("Think", system: "brain", on: voice.enableThinking) {
                voice.enableThinking.toggle()
            }
            Spacer(minLength: 0)
        }
        .help("“Hey Loki”: only send a request when you address the assistant by name — it ignores other talk. Turn off for always-on listening.")
    }

    private var autoApproveRow: some View {
        Toggle(isOn: $voice.autoApproveTools) {
            Text("Auto-approve tools (hands-free)")
                .font(.caption)
        }
        .toggleStyle(.switch)
        .controlSize(.mini)
        .help("On: tool calls run without asking — required for true hands-free use. Off: each tool call waits for you to Allow or Deny below.")
    }

    // MARK: Control row

    private var controls: some View {
        HStack(spacing: 10) {
            controlButton(system: "plus.bubble", label: "New", help: "Start a fresh chat session") {
                appState.chatEngine.stop()
                voice.bargeIn()
                _ = appState.newChatSession()
            }
            controlButton(system: "stop.fill", label: "Stop",
                          tint: voice.canInterrupt ? .red : nil,
                          help: "Stop the assistant and listen again — cut off a long answer and move on") {
                voice.bargeIn()
            }
            .disabled(!voice.canInterrupt)
            controlButton(system: voice.isMuted ? "mic.slash.fill" : "mic.fill",
                          label: voice.isMuted ? "Unmute" : "Mute",
                          tint: voice.isMuted ? .orange : nil,
                          help: voice.isMuted ? "Resume listening" : "Stop listening without ending voice") {
                voice.toggleMute()
            }
            Spacer(minLength: 0)
            voicePicker
        }
    }

    private var voicePicker: some View {
        Menu {
            ForEach(voice.availableVoices) { v in
                Button {
                    voice.selectVoice(v.id)
                } label: {
                    if v.id == voice.selectedVoiceId {
                        Label(v.displayName, systemImage: "checkmark")
                    } else {
                        Text(v.displayName)
                    }
                }
            }
            if voice.availableVoices.isEmpty {
                Text("No voices installed")
            }
            Divider()
            Button("Download more voices…") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpokenContent") {
                    NSWorkspace.shared.open(url)
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "speaker.wave.2.fill").font(.caption2)
                Text(selectedVoiceName).font(.caption).lineLimit(1)
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 8))
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Choose the speech voice. Add higher-quality voices in System Settings → Accessibility → Spoken Content.")
    }

    private var selectedVoiceName: String {
        voice.availableVoices.first { $0.id == voice.selectedVoiceId }?.name ?? "Voice"
    }

    // MARK: Inline approval card (auto-approve off)

    private func approvalCard(_ req: ToolApprovalRequest) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "shield.lefthalf.filled")
                    .foregroundStyle(.orange)
                Text("Allow this tool call?")
                    .font(.caption.weight(.semibold))
            }
            Text(req.toolName)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
            if !req.rawArguments.isEmpty {
                Text(req.rawArguments)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
                    .truncationMode(.tail)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack(spacing: 6) {
                Button(role: .destructive) { voice.resolve(.deny) } label: {
                    Text("Deny").frame(maxWidth: .infinity)
                }
                Button { voice.resolve(.allow, allowAll: true) } label: {
                    Text("Always").frame(maxWidth: .infinity)
                }
                Button { voice.resolve(.allow) } label: {
                    Text("Allow").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }
            .controlSize(.small)
        }
        .padding(8)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: Small components

    private func chip(_ title: String, system: String, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 3) {
                Image(systemName: system).font(.system(size: 10, weight: .medium))
                Text(title).font(.caption2.weight(.medium))
            }
            .foregroundStyle(on ? .white : .secondary)
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(on ? Color.accentColor : Color.secondary.opacity(0.15))
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private func controlButton(system: String, label: String, tint: Color? = nil,
                               help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: system).font(.system(size: 11, weight: .medium))
                Text(label).font(.caption2.weight(.medium))
            }
            .foregroundStyle(tint ?? .secondary)
        }
        .buttonStyle(.plain)
        .help(help)
    }

    // MARK: State → presentation

    private var isError: Bool { if case .error = voice.state { return true }; return false }

    private var statusText: String {
        switch voice.state {
        case .idle:        return "Starting…"
        case .listening:   return voice.isMuted ? "Muted" : voice.listeningPrompt
        case .recognizing: return "Listening…"
        case .thinking:    return "Thinking…"
        case .speaking:    return "Speaking…"
        case .error(let m): return m
        }
    }

    /// Static color for the status dot, mapped from the pure `VoiceTrayDot`
    /// presentation (which is time-free by design — see the freeze regression).
    private var dotColor: Color {
        switch VoiceTrayDot.tint(for: voice.state) {
        case .active:   return .cyan
        case .thinking: return .purple
        case .speaking: return .green
        case .error:    return .red
        case .idle:     return .gray
        }
    }
}
