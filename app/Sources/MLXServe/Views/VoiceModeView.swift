import SwiftUI
import AppKit

/// ChatGPT-style hands-free voice screen presented over the chat. A single
/// animated orb reflects the turn state (listening / thinking / speaking) and
/// pulses with the live mic level; the user's partial transcript shows beneath
/// it. The model/agent pipeline is reused as-is — this view only renders state
/// from `VoiceModeController` and forwards the agent/thinking/MCP toggles.
struct VoiceModeView: View {
    @ObservedObject var controller: VoiceModeController
    var onClose: () -> Void

    var body: some View {
        ZStack {
            backdrop.ignoresSafeArea()

            VStack(spacing: 20) {
                Spacer(minLength: 8)

                Text(statusText)
                    .font(.title3.weight(.medium))
                    .foregroundStyle(isError ? .red : .secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .transition(.opacity)

                orb
                    .onTapGesture { if controller.state == .speaking { controller.bargeIn() } }
                    .help(controller.state == .speaking ? "Tap to interrupt" : "")

                transcript

                voicePicker

                Spacer(minLength: 8)

                toggles
                controls
                    .padding(.bottom, 12)
            }
            .padding(24)
        }
        .frame(minWidth: 560, minHeight: 660)
        .animation(.easeInOut(duration: 0.25), value: controller.state)
        .overlay { approvalOverlay }
    }

    // MARK: Tool-approval overlay (agent mode)

    @ViewBuilder
    private var approvalOverlay: some View {
        if let req = controller.pendingApproval {
            ZStack {
                Color.black.opacity(0.6).ignoresSafeArea()
                    .onTapGesture { }   // swallow taps to the orb behind
                ToolApprovalSheet(request: req,
                                  onAllow: { controller.resolve(.allow) },
                                  onDeny: { controller.resolve(.deny) },
                                  onAllowAll: { controller.resolve(.allow, allowAll: true) })
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
                    .shadow(radius: 30)
                    .padding(24)
            }
            .transition(.opacity)
        }
    }

    // MARK: Orb

    private var orb: some View {
        // Breathe via TimelineView + `VoicePulse` instead of a `repeatForever`
        // animation kicked off in `.onAppear` (see `VoicePulse` for why that
        // pattern wedges the menu-bar popover).
        TimelineView(.animation) { tl in
            let breathe = VoicePulse.orbBreathe(
                animating: true, at: tl.date.timeIntervalSinceReferenceDate)
            let scale = 0.94 + CGFloat(controller.level) * 0.5 + CGFloat(breathe)
            ZStack {
                Circle()
                    .fill(RadialGradient(colors: orbColors.map { $0.opacity(0.9) },
                                         center: .center, startRadius: 4, endRadius: 150))
                    .frame(width: 220, height: 220)
                    .shadow(color: orbColors.first?.opacity(0.6) ?? .clear, radius: 40)
                Circle()
                    .stroke(orbColors.first?.opacity(0.35) ?? .clear, lineWidth: 2)
                    .frame(width: 252, height: 252)
                    .scaleEffect(1 + CGFloat(controller.level) * 0.25)
                if controller.state == .thinking {
                    ProgressView()
                        .controlSize(.large)
                        .tint(.white)
                }
            }
            .scaleEffect(scale)
            .animation(.easeOut(duration: 0.12), value: controller.level)
        }
    }

    private var transcript: some View {
        Text(controller.partialTranscript.isEmpty ? " " : controller.partialTranscript)
            .font(.body)
            .foregroundStyle(.primary)
            .multilineTextAlignment(.center)
            .lineLimit(3)
            .frame(maxWidth: 420, minHeight: 56, alignment: .top)
            .padding(.horizontal, 16)
    }

    // MARK: Voice picker

    private var voicePicker: some View {
        Menu {
            ForEach(controller.availableVoices) { v in
                Button {
                    controller.selectVoice(v.id)
                } label: {
                    if v.id == controller.selectedVoiceId {
                        Label(v.displayName, systemImage: "checkmark")
                    } else {
                        Text(v.displayName)
                    }
                }
            }
            if controller.availableVoices.isEmpty {
                Text("No voices installed")
            }
            Divider()
            Button("Download more voices…") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpokenContent") {
                    NSWorkspace.shared.open(url)
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "speaker.wave.2.fill")
                Text(selectedVoiceName).lineLimit(1)
                Image(systemName: "chevron.up.chevron.down").font(.caption2)
            }
            .font(.subheadline)
            .padding(.horizontal, 14).padding(.vertical, 7)
            .background(.thinMaterial, in: Capsule())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Choose the speech voice. Add higher-quality voices in System Settings → Accessibility → Spoken Content.")
    }

    private var selectedVoiceName: String {
        controller.availableVoices.first { $0.id == controller.selectedVoiceId }?.name ?? "Voice"
    }

    // MARK: Toggles + controls

    private var toggles: some View {
        // Inline VStack (not an offset overlay) so the auto-approve row takes
        // real layout space — otherwise it overlaps the control row below and
        // pushes the circle buttons off the bottom of the sheet.
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                chip("Hey Loki", system: "mic.circle", on: controller.requireWakeWord) { controller.requireWakeWord.toggle() }
                chip("Think", system: "brain", on: controller.enableThinking) { controller.enableThinking.toggle() }
                chip("Agent", system: "wrench", on: controller.agentMode) {
                    controller.agentMode.toggle()
                    if controller.agentMode { controller.enableThinking = false }
                }
                chip("MCP", system: "puzzlepiece.extension", on: controller.mcpMode) { controller.mcpMode.toggle() }
            }
            if controller.agentMode || controller.mcpMode {
                Toggle("Auto-approve tools (hands-free)", isOn: $controller.autoApproveTools)
                    .toggleStyle(.switch)
                    .controlSize(.mini)
                    .font(.caption)
                    .fixedSize()
            }
        }
    }

    private var controls: some View {
        // No "new session" button — voice mode is bound to the chat session it was
        // launched from (its toggles are seeded from that session on open).
        HStack(spacing: 36) {
            circleButton(system: "stop.fill",
                         tint: controller.canInterrupt ? .red : .secondary,
                         help: "Stop the assistant and listen again") {
                controller.bargeIn()
            }
            .disabled(!controller.canInterrupt)
            circleButton(system: controller.isMuted ? "mic.slash.fill" : "mic.fill",
                         tint: controller.isMuted ? .orange : .secondary,
                         help: controller.isMuted ? "Unmute" : "Mute") {
                controller.toggleMute()
            }
            circleButton(system: "xmark", tint: .red, help: "End voice mode") { onClose() }
        }
    }

    private func chip(_ title: String, system: String, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: system).font(.system(size: 11, weight: .medium))
                Text(title).font(.caption.weight(.medium))
            }
            .foregroundStyle(on ? .white : .secondary)
            .padding(.horizontal, 10).padding(.vertical, 5)
            .background(on ? Color.accentColor : Color.secondary.opacity(0.15))
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private func circleButton(system: String, tint: Color, help: String,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: system)
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(tint == .secondary ? Color.primary : .white)
                .frame(width: 56, height: 56)
                .background(tint == .secondary ? AnyShapeStyle(.thinMaterial) : AnyShapeStyle(tint))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .help(help)
    }

    // MARK: State → presentation

    private var isError: Bool { if case .error = controller.state { return true }; return false }

    private var statusText: String {
        switch controller.state {
        case .idle:        return "Starting…"
        case .listening:   return controller.isMuted ? "Muted — tap the mic to resume" : controller.listeningPrompt
        case .recognizing: return "Listening…"
        case .thinking:    return "Thinking…"
        case .speaking:    return "Speaking… (tap the orb to interrupt)"
        case .error(let m): return m
        }
    }

    private var orbColors: [Color] {
        switch controller.state {
        case .listening, .recognizing: return [.cyan, .blue]
        case .thinking:                return [.purple, .indigo]
        case .speaking:                return [.green, .teal]
        case .error:                   return [.red, .orange]
        case .idle:                    return [.gray, Color(white: 0.4)]
        }
    }

    private var backdrop: some View {
        LinearGradient(colors: [Color(white: 0.06), Color(white: 0.12)],
                       startPoint: .top, endPoint: .bottom)
    }
}
