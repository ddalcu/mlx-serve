import SwiftUI
import AppKit

/// Menu-bar tray content for the persistent voice assistant. Rendered as its own
/// section in the status popover and bound to the app-level
/// `VoiceModeController`, so it works with **no chat window open**. When this
/// popover is closed, feedback is audio-only — open it for status + controls.
struct VoiceTrayPanel: View {
    @ObservedObject var voice: VoiceModeController
    @EnvironmentObject var appState: AppState
    /// Opens the Agents window — the tray can't reach SwiftUI's `openWindow`.
    var openAgents: () -> Void = {}

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
                // Who am I talking to — and nothing else. The old row of chips
                // (wake phrase / Agent / MCP / Think), the auto-approve toggle
                // and the voice picker are all things an AGENT decides now, so
                // the panel is down to: who, what it's doing, and the three
                // transport buttons. With no agent picked it behaves exactly as
                // it always did, on the app's own settings.
                agentPicker
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

    /// On → open the selected agent's thread (creating it on first use), then
    /// start listening. Off → tear down.
    private var activeBinding: Binding<Bool> {
        Binding(
            get: { voice.isActive },
            set: { on in
                if on {
                    appState.sessionForAgent(appState.defaultAgentId)
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

    // MARK: Who am I talking to

    /// The agent driving hands-free turns. Picking one also switches its model,
    /// workspace and voice (`AppState.applyAgentSelection`); an agent whose model
    /// isn't downloaded is shown as unavailable rather than silently answered by
    /// whoever was active.
    private var agentPicker: some View {
        HStack(spacing: 6) {
            Image(systemName: activeAgent?.symbol ?? "person.crop.circle")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Picker("", selection: Binding(get: { appState.defaultAgentId },
                                          set: { appState.defaultAgentId = $0 })) {
                Text("None (app defaults)").tag(UUID?.none)
                ForEach(appState.agents.allAgents) { agent in
                    Text(agentLabel(agent)).tag(UUID?.some(agent.id))
                }
            }
            .labelsHidden()
            .controlSize(.small)
            .fixedSize()
            Spacer(minLength: 0)
            Button("Manage…") { openAgents() }
                .buttonStyle(.link)
                .font(.caption2)
        }
        .help("Who you're talking to. An agent brings its own prompt, voice, tools, workspace and model; “None” uses the app's own settings, exactly as before. Say another agent's wake phrase to hand the conversation over mid-session.")
    }

    private var activeAgent: Agent? { appState.agents.agent(id: appState.defaultAgentId) }

    private func agentLabel(_ agent: Agent) -> String {
        AgentModelSwitch.isSelectable(appState.agentModelDecision(for: agent))
            ? agent.name
            : "\(agent.name) — model not downloaded"
    }

    // MARK: Control row

    private var controls: some View {
        HStack(spacing: 10) {
            controlButton(system: "plus.bubble", label: "New",
                          help: "Start a fresh conversation with this agent — the old one stays in the chat list") {
                // Scoped: only the voice conversation's turn. Other chat tabs'
                // streams keep running (multi-turn engine).
                if let sid = appState.activeChatId {
                    appState.chatEngine.stop(sessionId: sid)
                }
                voice.bargeIn()
                // A fresh thread FOR THE SAME AGENT, so the next turn doesn't get
                // routed straight back into the old one. The previous thread is
                // left in the sidebar, as it always has been.
                _ = appState.newChatSession(agentId: appState.defaultAgentId)
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
        }
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
