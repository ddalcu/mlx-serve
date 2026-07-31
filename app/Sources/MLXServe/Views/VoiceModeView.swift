import SwiftUI
import AppKit

/// The in-window face of hands-free voice: a compact talking orb rendered
/// INLINE just above the composer. It replaced the full-window sheet — the
/// sheet covered the transcript (the one thing a chat window is for) and its
/// toggle row duplicated controls the composer already carries. The orb
/// reflects the turn state (listening / thinking / speaking) via color +
/// motion and pulses with the live mic level; the caption beneath carries the
/// status or the user's partial transcript. Renders nothing while voice is
/// off, so mounting it unconditionally costs no layout.
///
/// The model/agent pipeline is reused as-is — this view only renders state
/// from the app-level `VoiceModeController` (which the tray shares, so voice
/// started from either surface shows here).
struct VoiceOrbView: View {
    @ObservedObject var controller: VoiceModeController
    /// The chat column this orb renders in. Voice is ONE instance bound to ONE
    /// session (`controller.boundSessionId`) — only that session's tab shows
    /// the orb; every other chat renders nothing.
    var sessionId: UUID?

    /// Orb diameter. 128 by design — big enough to read the state animation,
    /// small enough that the transcript stays the window's main content.
    static let orbSize: CGFloat = 128

    @ViewBuilder
    var body: some View {
        if VoiceModeController.voiceOwnedHere(isActive: controller.isActive,
                                              boundSessionId: controller.boundSessionId,
                                              sessionId: sessionId) {
            VStack(spacing: 6) {
                orb
                    .onTapGesture { if controller.state == .speaking { controller.bargeIn() } }
                    .help(controller.state == .speaking ? "Tap to interrupt" : statusText)
                caption
                // Tool approval (agent mode, auto-approve off). Inline next to
                // the orb: there is no sheet to host it any more, and the
                // window-level approval sheet belongs to typed chat turns.
                if let req = controller.pendingApproval {
                    ToolApprovalSheet(request: req,
                                      onAllow: { controller.resolve(.allow) },
                                      onDeny: { controller.resolve(.deny) },
                                      onAllowAll: { controller.resolve(.allow, allowAll: true) })
                        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)
            .animation(.easeInOut(duration: 0.25), value: controller.state)
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
                                         center: .center, startRadius: 3,
                                         endRadius: Self.orbSize * 0.68))
                    .frame(width: Self.orbSize, height: Self.orbSize)
                    .shadow(color: orbColors.first?.opacity(0.5) ?? .clear, radius: 22)
                Circle()
                    .stroke(orbColors.first?.opacity(0.35) ?? .clear, lineWidth: 1.5)
                    .frame(width: Self.orbSize * 1.14, height: Self.orbSize * 1.14)
                    .scaleEffect(1 + CGFloat(controller.level) * 0.25)
                if controller.state == .thinking {
                    ProgressView()
                        .tint(.white)
                }
            }
            .scaleEffect(scale)
            .animation(.easeOut(duration: 0.12), value: controller.level)
        }
        // Reserve the ring's overshoot so the pulse never lands on the
        // composer border below.
        .frame(width: Self.orbSize * 1.2, height: Self.orbSize * 1.2)
    }

    /// One line under the orb: the partial transcript while the user is
    /// talking, otherwise the state — which is also where a mic-permission
    /// failure becomes visible (the old sheet's status text did that job).
    private var caption: some View {
        Text(controller.partialTranscript.isEmpty ? statusText : controller.partialTranscript)
            .font(.caption)
            .foregroundStyle(isError ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary))
            .multilineTextAlignment(.center)
            .lineLimit(2)
            .padding(.horizontal, 16)
    }

    // MARK: State → presentation

    private var isError: Bool { if case .error = controller.state { return true }; return false }

    private var statusText: String {
        switch controller.state {
        case .idle:        return "Starting…"
        case .listening:   return controller.isMuted ? "Muted — unmute in the menu bar" : controller.listeningPrompt
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
}

/// The composer-row voice toggle (between the context gauge and Send). Its own
/// observing view so the on/off tint follows the app-level controller even
/// when voice starts from the tray or dies on its own — ChatView itself does
/// not observe the controller.
struct VoiceComposerToggle: View {
    @ObservedObject var controller: VoiceModeController
    /// The chat column this toggle sits in — the on-tint shows only in the
    /// session voice is BOUND to (see `VoiceModeController.boundSessionId`).
    var sessionId: UUID?
    var disabled: Bool
    /// Starts voice with the surrounding chat's toggles/agent (ChatView's
    /// `startVoiceMode`). Clicking the (off-tinted) toggle in another chat
    /// while voice runs elsewhere MOVES voice here: end + start, adopting this
    /// chat's agent/toggles.
    var start: () -> Void

    private var ownedHere: Bool {
        VoiceModeController.voiceOwnedHere(isActive: controller.isActive,
                                           boundSessionId: controller.boundSessionId,
                                           sessionId: sessionId)
    }

    var body: some View {
        Button {
            if ownedHere {
                controller.end()
            } else {
                if controller.isActive { controller.end() }   // move: release the mic first
                start()
            }
        } label: {
            Image(systemName: "waveform")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(ownedHere ? Color.white : Color.secondary)
                .frame(width: ChatMetrics.composerIconSize, height: ChatMetrics.composerIconSize)
                .background(ownedHere ? Color.accentColor : Color.secondary.opacity(0.15))
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .frame(width: ChatMetrics.composerControlSize, height: ChatMetrics.composerControlSize)
        .disabled(disabled)
        .help("Voice mode (\(ownedHere ? "ON in this chat" : controller.isActive ? "ON in another chat — click to move it here" : "OFF")) — talk to the model hands-free. Speech-to-text and text-to-speech run locally on your Mac; the model only handles text (and tools/thinking if enabled).")
    }
}
