import SwiftUI

/// The voice picker shared by the tray voice panel (compact) and the chat
/// voice overlay (pill). The cloned voice is a first-class entry beside the
/// Apple system voices:
///
///   Your voice      My voice — morgan.mp3   ✓   (selecting re-enables clone)
///                   Choose audio file…           (pick + normalize a clip)
///   System voices   Samantha — Premium           (selecting disables clone,
///                   …                             the clip is kept)
///
/// The collapsed label names what will ACTUALLY speak: the clip while the
/// clone is active, the Apple voice otherwise — including when the Qwen3-TTS
/// model isn't downloaded, in which case the clone rows are disabled with a
/// pointer at the Audio tile (decisions in `VoiceCloneMenuModel`).
struct VoiceSelectorMenu: View {
    @ObservedObject var voice: VoiceModeController
    @EnvironmentObject var appState: AppState
    /// true = tray caption styling; false = the overlay's material pill.
    let compact: Bool

    /// Disk check cached per appearance — the menu body re-evaluates on every
    /// controller publish (~20 Hz while speaking), too often for a stat call.
    @State private var ttsDownloaded = false
    @State private var pickError: String?

    var body: some View {
        Menu {
            cloneSection
            systemSection
        } label: {
            collapsedLabelView
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Choose the speech voice — your cloned voice or a system voice. Add higher-quality system voices in System Settings → Accessibility → Spoken Content.")
        .onAppear { ttsDownloaded = VoiceCloneMenuModel.ttsModelDownloaded() }
        .alert("Couldn't use that audio file",
               isPresented: Binding(get: { pickError != nil },
                                    set: { if !$0 { pickError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(pickError ?? "")
        }
    }

    // MARK: Menu sections

    @ViewBuilder private var cloneSection: some View {
        Section("Your voice") {
            if !clipPath.isEmpty {
                Button {
                    appState.serverOptions.voiceCloneEnabled = true
                } label: {
                    if cloneActive {
                        Label(VoiceCloneMenuModel.cloneItemTitle(label: cloneLabel),
                              systemImage: "checkmark")
                    } else {
                        Text(VoiceCloneMenuModel.cloneItemTitle(label: cloneLabel))
                    }
                }
                .disabled(!ttsDownloaded)
            }
            Button(clipPath.isEmpty ? "Choose audio file to clone…" : "Choose different audio file…") {
                pickCloneFile()
            }
            .disabled(!ttsDownloaded)
            if let reason = VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: ttsDownloaded) {
                Text(reason)
            }
        }
    }

    @ViewBuilder private var systemSection: some View {
        Section("System voices") {
            ForEach(voice.availableVoices) { v in
                Button {
                    // Switching to an Apple voice turns the clone off but
                    // keeps the clip — "My voice" stays one click away.
                    appState.serverOptions.voiceCloneEnabled = false
                    voice.selectVoice(v.id)
                } label: {
                    if v.id == voice.selectedVoiceId && !cloneActive {
                        Label(v.displayName, systemImage: "checkmark")
                    } else {
                        Text(v.displayName)
                    }
                }
            }
            if voice.availableVoices.isEmpty {
                Text("No voices installed")
            }
            Button("Download more voices…") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.universalaccess?SpokenContent") {
                    NSWorkspace.shared.open(url)
                }
            }
        }
    }

    // MARK: Collapsed label

    @ViewBuilder private var collapsedLabelView: some View {
        if compact {
            HStack(spacing: 4) {
                Image(systemName: collapsedIcon).font(.caption2)
                Text(collapsedTitle).font(.caption).lineLimit(1)
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 8))
            }
        } else {
            HStack(spacing: 6) {
                Image(systemName: collapsedIcon)
                Text(collapsedTitle).lineLimit(1)
                Image(systemName: "chevron.up.chevron.down").font(.caption2)
            }
            .font(.subheadline)
            .padding(.horizontal, 14).padding(.vertical, 7)
            .background(.thinMaterial, in: Capsule())
        }
    }

    private var collapsedTitle: String {
        VoiceCloneMenuModel.collapsedLabel(
            clipPath: clipPath,
            cloneEnabled: appState.serverOptions.voiceCloneEnabled,
            ttsModelDownloaded: ttsDownloaded,
            cloneLabel: cloneLabel,
            systemVoiceName: voice.availableVoices.first { $0.id == voice.selectedVoiceId }?.name)
    }

    private var collapsedIcon: String {
        cloneActive ? "person.wave.2.fill" : "speaker.wave.2.fill"
    }

    // MARK: State helpers

    private var clipPath: String { appState.serverOptions.voiceClonePath }
    private var cloneLabel: String { appState.serverOptions.voiceCloneLabel }
    private var cloneActive: Bool {
        VoiceCloneMenuModel.cloneIsActive(clipPath: clipPath,
                                          cloneEnabled: appState.serverOptions.voiceCloneEnabled,
                                          ttsModelDownloaded: ttsDownloaded)
    }

    private func pickCloneFile() {
        do {
            guard let picked = try VoiceCloneMenuModel.pickAndPersistClip() else { return }
            appState.serverOptions.voiceClonePath = picked.path
            appState.serverOptions.voiceCloneLabel = picked.label
            appState.serverOptions.voiceCloneEnabled = true
        } catch {
            pickError = error.localizedDescription
        }
    }
}
