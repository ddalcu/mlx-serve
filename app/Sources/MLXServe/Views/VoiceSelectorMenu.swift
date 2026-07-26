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
    @EnvironmentObject private var downloads: DownloadManager
    /// true = tray caption styling; false = the overlay's material pill.
    let compact: Bool

    /// Disk check cached per appearance — the menu body re-evaluates on every
    /// controller publish (~20 Hz while speaking), too often for a stat call.
    @State private var ttsDownloaded = false
    @State private var kokoroDownloaded = false
    @State private var pickError: String?
    /// Previews the voice as it is picked, so the tray is auditionable too.
    @StateObject private var previewer = VoicePreviewer()

    var body: some View {
        Menu {
            kokoroSection
            cloneSection
            systemSection
        } label: {
            collapsedLabelView
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Choose the speech voice — your cloned voice or a system voice. Add higher-quality system voices in System Settings → Accessibility → Spoken Content.")
        .onAppear {
            restatVoiceModels()
            previewer.attach(server: appState.server)
        }
        // A download that finishes while the tray is open would otherwise leave
        // Kokoro looking unavailable until the menu is reopened. Re-stat when a
        // download STATE publishes — not per render, which is what the
        // snapshots above exist to avoid.
        .onChange(of: voiceModelDownloadStatuses) { _, _ in restatVoiceModels() }
        .alert("Couldn't use that audio file",
               isPresented: Binding(get: { pickError != nil },
                                    set: { if !$0 { pickError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(pickError ?? "")
        }
    }

    // MARK: Menu sections

    /// Kokoro's 54 voices, grouped by language. Selecting one switches the
    /// engine AND plays a short sample, so picking a voice in the tray is the
    /// same gesture as in Settings.
    @ViewBuilder private var kokoroSection: some View {
        Section("Kokoro voices") {
            if let reason = VoiceCloneMenuModel.kokoroUnavailableReason(kokoroDownloaded: kokoroDownloaded) {
                Text(reason)
            } else {
                ForEach(KokoroVoiceCatalog.grouped(), id: \.language) { group in
                    Menu(group.language) {
                        ForEach(group.voices, id: \.self) { v in
                            Button {
                                appState.serverOptions.voiceEngine = .kokoro
                                appState.serverOptions.kokoroVoice = v
                                previewer.preview(v)
                            } label: {
                                if kokoroActive && appState.serverOptions.kokoroVoice == v {
                                    Label(KokoroVoiceCatalog.displayName(for: v), systemImage: "checkmark")
                                } else {
                                    Text(KokoroVoiceCatalog.displayName(for: v))
                                }
                            }
                        }
                    }
                }
                // A blend is set in Settings (it needs free text); surface it
                // here as a tickable row when one is active so the tray never
                // shows a plain voice while a blend is what speaks.
                if kokoroActive && KokoroVoiceCatalog.isBlend(appState.serverOptions.kokoroVoice) {
                    Label(KokoroVoiceCatalog.blendDisplayName(for: appState.serverOptions.kokoroVoice),
                          systemImage: "checkmark")
                }
            }
        }
    }

    @ViewBuilder private var cloneSection: some View {
        Section("Your voice") {
            if !clipPath.isEmpty {
                Button {
                    appState.serverOptions.voiceEngine = .clone
                    appState.serverOptions.voiceCloneEnabled = true
                    previewer.stop()
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
                    appState.serverOptions.voiceEngine = .system
                    appState.serverOptions.voiceCloneEnabled = false
                    previewer.stop()
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
            engine: appState.serverOptions.voiceEngine,
            clipPath: clipPath,
            cloneEnabled: appState.serverOptions.voiceCloneEnabled,
            ttsModelDownloaded: ttsDownloaded,
            kokoroDownloaded: kokoroDownloaded,
            kokoroVoice: appState.serverOptions.kokoroVoice,
            cloneLabel: cloneLabel,
            systemVoiceName: voice.availableVoices.first { $0.id == voice.selectedVoiceId }?.name)
    }

    private var collapsedIcon: String {
        if kokoroActive { return "waveform" }
        return cloneActive ? "person.wave.2.fill" : "speaker.wave.2.fill"
    }

    // MARK: State helpers

    /// Every tracked repo's status, projected to something `onChange` can
    /// compare. A dictionary of a handful of enums is far cheaper than the two
    /// disk stats it gates, and watching ALL repos (not just Kokoro's) means a
    /// finished Qwen3-TTS download un-greys the clone rows too.
    private var voiceModelDownloadStatuses: [String: DownloadManager.DownloadState.Status] {
        downloads.downloads.mapValues(\.status)
    }

    private func restatVoiceModels() {
        ttsDownloaded = VoiceCloneMenuModel.ttsModelDownloaded()
        kokoroDownloaded = VoiceCloneMenuModel.kokoroModelDownloaded()
    }

    private var clipPath: String { appState.serverOptions.voiceClonePath }
    private var cloneLabel: String { appState.serverOptions.voiceCloneLabel }
    private var cloneActive: Bool {
        appState.serverOptions.voiceEngine == .clone &&
        VoiceCloneMenuModel.cloneIsActive(clipPath: clipPath,
                                          cloneEnabled: appState.serverOptions.voiceCloneEnabled,
                                          ttsModelDownloaded: ttsDownloaded)
    }
    private var kokoroActive: Bool {
        VoiceCloneMenuModel.kokoroIsActive(engine: appState.serverOptions.voiceEngine,
                                          kokoroDownloaded: kokoroDownloaded)
    }

    private func pickCloneFile() {
        do {
            guard let picked = try VoiceCloneMenuModel.pickAndPersistClip() else { return }
            appState.serverOptions.voiceClonePath = picked.path
            appState.serverOptions.voiceCloneLabel = picked.label
            appState.serverOptions.voiceCloneEnabled = true
            appState.serverOptions.voiceEngine = .clone
        } catch {
            pickError = error.localizedDescription
        }
    }
}
