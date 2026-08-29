import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Audio generation window — two tabs over one window: **Voice** (neural TTS
/// with zero-shot voice cloning, Qwen3-TTS) and **Music** (prompt-driven music
/// generation, ACE-Step). Each tab is its own self-contained pane; this
/// container only hosts the segmented switcher.
struct AudioGenView: View {
    enum Tab: String, CaseIterable {
        case voice = "Voice"
        case music = "Music"
    }

    /// The left menu's "Audio & Music" row must reopen on the tab you left it
    /// on. `ChatView.createPane` UNMOUNTS this view on navigation, so plain
    /// `@State` reset to Voice on every visit, not only across launches.
    /// The stored raw values are a persistence contract — renaming the display
    /// text would silently send everyone back to Voice.
    @AppStorage("audioGenTab") private var tab: Tab = .voice

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $tab) {
                ForEach(Tab.allCases, id: \.self) { t in
                    Text(t.rawValue).tag(t)
                }
            }
            .pickerStyle(.segmented)
            .controlSize(.large)
            .labelsHidden()
            .frame(width: 280)
            .padding(.top, 10)
            .padding(.bottom, 14)

            switch tab {
            case .voice: VoiceGenView()
            case .music: MusicGenView()
            }
        }
    }
}

/// The style prompt a track was made from, read back out of the `<track>.txt`
/// sidecar the gen services write beside every WAV.
///
/// A history row carries a path and nothing else, so without this a track sent
/// to chat arrives captioned "Generated audio" — true, and useless in a
/// transcript you keep. Best-effort by design: an older track, a hand-copied
/// file or a missing sidecar just falls back to that.
enum AudioSidecar {
    static func prompt(forTrack path: String) -> String {
        let txt = (path as NSString).deletingPathExtension + ".txt"
        guard let body = try? String(contentsOfFile: txt, encoding: .utf8) else { return "" }
        guard let range = body.range(of: "# Style prompt\n") else { return "" }
        let rest = body[range.upperBound...]
        // The sidecar's sections are separated by a blank line.
        let end = rest.range(of: "\n\n")?.lowerBound ?? rest.endIndex
        return String(rest[..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Vertical list of past generations, shared by the Voice and Music tabs.
/// Every file the service ever wrote stays listed (newest first, uncapped);
/// clicking a row plays it through the owning tab's player, replacing
/// whatever was playing.
struct AudioHistoryShelf: View {
    let title: String
    let paths: [String]
    let playingPath: String?
    let onPlay: (String) -> Void
    let onStop: () -> Void
    /// Send this track to a new conversation. Optional so a shelf without a
    /// chat to send to still compiles; both audio tabs pass it.
    var onSendToChat: ((String) -> Void)? = nil

    var body: some View {
        Group {
            if !paths.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    ScrollView {
                        VStack(spacing: 2) {
                            ForEach(paths, id: \.self) { path in
                                row(path)
                            }
                        }
                    }
                    .frame(maxHeight: 150)
                }
            }
        }
    }

    private func row(_ path: String) -> some View {
        let playing = playingPath == path
        return HStack(spacing: 8) {
            Image(systemName: playing ? "speaker.wave.2.fill" : "waveform")
                .foregroundStyle(playing ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                .frame(width: 16)
            Text(URL(fileURLWithPath: path).lastPathComponent)
                .font(.caption)
                .lineLimit(1).truncationMode(.middle)
                .help(path)
            Spacer()
            Button {
                playing ? onStop() : onPlay(path)
            } label: {
                Image(systemName: playing ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.borderless)
            .help(playing ? "Stop" : "Play")
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            } label: { Image(systemName: "folder") }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Reveal in Finder")
            // Reaching a conversation should not depend on a track being the
            // one this session happened to make last: the completed panel is
            // transient, the shelf is every track you have.
            if let send = onSendToChat {
                Button { send(path) } label: {
                    Image(systemName: "bubble.left.and.text.bubble.right")
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Send to Chat — opens a new conversation with this attached")
            }
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .background(
            RoundedRectangle(cornerRadius: 5)
                .fill(playing ? Color.accentColor.opacity(0.12) : Color.clear)
        )
        .contentShape(Rectangle())
        .onTapGesture { playing ? onStop() : onPlay(path) }
    }
}

/// Voice tab — neural TTS with zero-shot voice cloning, run natively by the
/// embedded mlx-serve server. Same shell as ImageGen/VideoGen: a model
/// picker, the text to speak, a reference-voice section (record or pick a file)
/// with an optional transcript, and a player for the result. (Extracted
/// verbatim from the pre-tabs AudioGenView — zero behavior change.)
struct VoiceGenView: View {
    @EnvironmentObject var service: AudioGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    /// For "Send to Chat" — the hand-off opens a new conversation and switches
    /// the window to it (`AppState.sendGeneratedMediaToNewChat`).
    @EnvironmentObject var appState: AppState

    @StateObject private var recorder = AudioRecorder()

    @State private var text: String = ""
    @State private var model: AudioModelPreset = .qwen3TTS06B8bit
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var refAudioURL: URL? = nil
    @State private var refText: String = ""
    @State private var speed: Double = 1.0
    @State private var temperature: Double = 0.7
    @State private var showAdvanced: Bool = false

    @State private var refError: String? = nil
    /// True while a drag carrying a file hovers the reference-voice section —
    /// drives its dashed-border highlight (see `MediaDropTarget`).
    @State private var isDropTargeted: Bool = false
    /// Dictation into the text editor: the voice-mode recognizer emits one
    /// finalized utterance per silence gap; each is appended via `Dictation`.
    /// Created lazily on first use — never at launch (audio-graph TCC rule).
    @State private var dictation: (any SpeechRecognizing)? = nil
    @State private var dictating: Bool = false
    @State private var dictationPartial: String = ""
    @State private var dictationError: String? = nil
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: AudioGenRequest? = nil
    @StateObject private var clipPlayer = AudioClipPlayer()
    /// Keep the model resident after generating (default off → unload).
    @State private var keepResident: Bool = false
    /// Hydration guard — see ImageGenView for the full rationale.
    @State private var hydrating: Bool = false
    @State private var didHydrate: Bool = false

    var body: some View {
        // No window-sized floor — see ImageGenView: pages shrink their
        // preview side, they don't overflow the detail column.
        readyView
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                DispatchQueue.main.async { hydrating = false }
            }
            // Freshen the network-model list so LAN entries are current in
            // the picker (discovery lands seconds after the server boots).
            if server.status == .running { Task { await server.refreshModels() } }
        }
        .onDisappear { stopDictation() }
        .onChange(of: model) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: stickySnapshot) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: service.phase) { _, phase in
            // A new generation stops whatever is still playing.
            if case .running = phase { stopPlayback() }
            if case .completed(let path) = phase { play(path) }
        }
    }

    private func play(_ path: String) {
        clipPlayer.play(path)
    }

    private func stopPlayback() {
        clipPlayer.stop()
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    textSection
                    modelSection
                    referenceSection
                    if showAdvanced { advancedSection } else { advancedToggle }
                    actionRow
                }
                .padding(16)
            }
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                AudioHistoryShelf(
                    title: "History",
                    paths: service.recent,
                    playingPath: clipPlayer.playingPath,
                    onPlay: { play($0) },
                    onStop: { stopPlayback() },
                    onSendToChat: { path in
                        appState.sendGeneratedMediaToNewChat(
                            path: path, prompt: AudioSidecar.prompt(forTrack: path), kind: .audio)
                    }
                )
                outputFolderLink
            }
            .padding(16)
            // The preview gives way in a small window.
            .frame(minWidth: 280)
        }
        .alert("Model exceeds your Mac's RAM", isPresented: $showRAMWarning) {
            Button("Cancel", role: .cancel) { pendingRequest = nil }
            Button("Generate Anyway", role: .destructive) {
                if let req = pendingRequest { service.generate(req, server: server) }
                pendingRequest = nil
            }
        } message: {
            Text(ramWarningMessage)
        }
    }

    // MARK: - Sections

    private var textSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Text to speak").font(.subheadline.weight(.semibold))
                Spacer()
                Button { toggleDictation() } label: {
                    Image(systemName: dictating ? "mic.fill" : "mic")
                        .foregroundStyle(dictating ? AnyShapeStyle(Color.red) : AnyShapeStyle(.secondary))
                }
                .buttonStyle(.borderless)
                .help(dictating ? "Stop dictation" : "Dictate the text to speak")
            }
            TextEditor(text: $text)
                .font(.body)
                .frame(height: 120)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
            if dictating {
                Text(dictationPartial.isEmpty ? "Listening…" : dictationPartial)
                    .font(.caption2).foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            if let err = dictationError {
                Text(err).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    // MARK: - Dictation

    private func toggleDictation() {
        dictating ? stopDictation() : startDictation()
    }

    private func startDictation() {
        dictationError = nil
        Task {
            let rec = dictation ?? makeSpeechRecognizer()
            dictation = rec
            guard await rec.requestAuthorization() else {
                dictationError = "Microphone or speech-recognition access is off. Enable both in System Settings ▸ Privacy & Security."
                return
            }
            rec.onPartialTranscript = { dictationPartial = $0 }
            rec.onFinalTranscript = {
                text = Dictation.appending($0, to: text)
                dictationPartial = ""
            }
            rec.onError = { msg in
                dictationError = msg
                stopDictation()
            }
            do {
                try rec.start()
                dictating = true
            } catch {
                dictationError = error.localizedDescription
            }
        }
    }

    private func stopDictation() {
        // Words spoken but not yet finalized by the silence gap land too —
        // stopping mid-sentence must not eat them.
        if !dictationPartial.isEmpty { text = Dictation.appending(dictationPartial, to: text) }
        dictationPartial = ""
        dictation?.stop()
        dictating = false
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`.
    private var modelSection: some View {
        MediaModelChooser.pane(
            all: AudioModelPreset.all,
            onThisMac: CustomMediaModels.audioPresets(from: server.allModels),
            // "speech", not "audio": see ModelInfo.lanAdvertises — a peer's
            // music model advertises "audio" too.
            capability: "speech",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.audioPreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; persist() }
    }

    private var referenceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Reference voice").font(.subheadline.weight(.semibold))
                Spacer()
                Text("~\(model.recommendedRefSeconds)s recommended")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if let url = refAudioURL {
                HStack(spacing: 8) {
                    Image(systemName: "waveform.circle.fill").foregroundStyle(.green)
                    Text(url.lastPathComponent)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    if clipPlayer.playingPath == url.path {
                        Button { clipPlayer.stop() } label: { Image(systemName: "stop.circle.fill") }
                            .buttonStyle(.borderless).help("Stop preview")
                    } else {
                        Button { playReference(url) } label: { Image(systemName: "play.circle") }
                            .buttonStyle(.borderless).help("Preview reference")
                    }
                    Button { clearReference() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary).help("Clear reference")
                }
            } else if recorder.isRecording {
                HStack(spacing: 10) {
                    Image(systemName: "mic.fill").foregroundStyle(.red)
                    ProgressView(value: Double(recorder.level)).frame(width: 120)
                    Text(String(format: "%.1fs", recorder.duration))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    Spacer()
                    Button { stopRecording() } label: {
                        Label("Stop", systemImage: "stop.circle.fill")
                    }
                    .buttonStyle(.bordered)
                }
            } else {
                HStack(spacing: 8) {
                    Button { chooseReferenceFile() } label: {
                        Label("Choose file…", systemImage: "folder")
                            .font(.caption).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    Button { startRecording() } label: {
                        Label("Record", systemImage: "mic")
                            .font(.caption).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                }
            }

            if refAudioURL != nil {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Transcript of reference (optional)").font(.caption)
                    TextField("", text: $refText,
                              prompt: Text("Optional — the reference audio alone clones the voice"))
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                }
            } else {
                Text("Pick, record or drag in ~\(model.recommendedRefSeconds) seconds of the voice to clone. Without a reference, the model's default voice is used.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            if let err = refError {
                Text(err).font(.caption2).foregroundStyle(.orange)
            }
        }
        // One clip slot, so a drop replaces what's there. Routed through
        // `acceptReference` so a dropped file is transcoded exactly like a
        // picked one — see `MediaDropTarget`.
        .mediaDrop(.audio, isTargeted: $isDropTargeted) { urls in
            if let url = urls.first { acceptReference(url) }
        }
    }

    private var advancedToggle: some View {
        Button {
            withAnimation { showAdvanced = true }
        } label: {
            Label("Advanced options", systemImage: "chevron.right").font(.caption)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Advanced").font(.caption.weight(.semibold))
                Spacer()
                Button { withAnimation { showAdvanced = false } } label: { Image(systemName: "chevron.down") }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Speed (\(String(format: "%.2fx", speed)))").font(.caption)
                Slider(value: $speed, in: 0.5...2.0, step: 0.05)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Temperature (\(String(format: "%.2f", temperature)))").font(.caption)
                Slider(value: $temperature, in: 0.1...1.5, step: 0.05)
                Text("Higher = more expressive and varied.").font(.caption2).foregroundStyle(.secondary)
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        }
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
            HStack {
                if service.isRunning {
                    Button(role: .destructive) { service.cancel() } label: {
                        Label("Cancel", systemImage: "stop.circle").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button { tryGenerate() } label: {
                        Label("Generate", systemImage: "waveform").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (lanModel == nil && !downloads.bundleReady(model.bundle)))
                }
            }
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No audio yet", systemImage: "waveform",
                                           description: Text("Enter text, add a reference voice, and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        // Audio length is unknown until the model stops (total==0)
                        // → indeterminate bar; encode/decode stages are determinate.
                        if total == 0 {
                            ProgressView().frame(width: 240)
                        } else {
                            ProgressView(value: Double(step), total: max(1, Double(total)))
                                .progressViewStyle(.linear).frame(width: 240)
                        }
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                case .completed(let path):
                    completedPreview(path: path)
                case .failed(let msg):
                    ContentUnavailableView {
                        Label("Failed", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(msg)
                    } actions: {
                        Button("Show log") { showLogWindow() }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func completedPreview(path: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 64)).foregroundStyle(.tint)
            HStack(spacing: 10) {
                Button { clipPlayer.play(path) } label: {
                    Label("Play", systemImage: "play.fill")
                }
                .buttonStyle(.bordered)
                Button { clipPlayer.pause() } label: {
                    Label("Pause", systemImage: "pause.fill")
                }
                .buttonStyle(.bordered)
            }
            HStack(spacing: 8) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless).help("Reveal in Finder")
                // The one bridge from the workshop to a conversation. It opens
                // a NEW chat and switches to it — see
                // `AppState.sendGeneratedMediaToNewChat`.
                Button {
                    appState.sendGeneratedMediaToNewChat(
                        path: path, prompt: text, kind: .audio)
                } label: { Image(systemName: "bubble.left.and.text.bubble.right") }
                .buttonStyle(.borderless)
                .help("Send to Chat — opens a new conversation with this attached")
            }
        }
        .padding(16)
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.audiosRoot)])
        } label: {
            Label("Open output folder in Finder", systemImage: "folder").font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.audiosRoot)
    }

    // MARK: - Reference actions

    private func chooseReferenceFile() {
        refError = nil
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        acceptReference(url)
    }

    /// The one way a FILE becomes the reference clip, whether it was picked or
    /// dropped: a dropped file gets the same 24 kHz mono transcode, and a
    /// transcode failure the same visible reason rather than a file that
    /// silently doesn't attach.
    private func acceptReference(_ url: URL) {
        refError = nil
        do {
            refAudioURL = try AudioReference.normalizedReferenceWav(fromFile: url)
        } catch {
            refError = error.localizedDescription
        }
    }

    private func startRecording() {
        refError = nil
        stopDictation() // one mic user at a time
        Task {
            guard await AudioRecorder.requestPermission() else {
                refError = "Microphone access denied. Enable it in System Settings ▸ Privacy ▸ Microphone."
                return
            }
            do { try recorder.start() }
            catch { refError = error.localizedDescription }
        }
    }

    private func stopRecording() {
        guard let data = recorder.stop() else {
            refError = "Nothing was recorded."
            return
        }
        do {
            refAudioURL = try AudioReference.normalizedReferenceWav(fromRecordedPCM: data)
        } catch {
            refError = error.localizedDescription
        }
    }

    private func clearReference() {
        if let url = refAudioURL { try? FileManager.default.removeItem(at: url) }
        refAudioURL = nil
        refText = ""
    }

    private func playReference(_ url: URL) {
        clipPlayer.play(url.path)
    }

    // MARK: - Sticky settings

    private func hydrate() {
        let s = AudioGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        speed = s.speed
        temperature = s.temperature
        keepResident = s.keepResident
        text = s.text
        refText = s.refText
        // The clip is a temp transcode; only restore it while it still exists.
        refAudioURL = s.refAudioPath.flatMap { FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil }
    }

    /// Every sticky field (knobs AND the typed draft) as one `Equatable`
    /// blob, so a single `onChange` persists all of it.
    private var stickySnapshot: AudioGenSettings {
        var s = AudioGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.speed = speed
        s.temperature = temperature
        s.keepResident = keepResident
        s.text = text
        s.refText = refText
        s.refAudioPath = refAudioURL?.path
        return s
    }

    private func persist() { stickySnapshot.save() }

    // MARK: - Generate

    private func tryGenerate() {
        stopDictation() // the open mic would pick up the played result
        let req = AudioGenRequest(
            model: model,
            text: text,
            refAudioPath: refAudioURL?.path,
            refText: refText,
            speed: speed,
            temperature: temperature,
            keepResident: keepResident,
            lanModelId: lanModel
        )
        persist()
        let total = RAMChecker.totalGB
        let needed = model.approxRAMGB
        if total < needed {
            ramWarningMessage = "This model needs about \(needed) GB of RAM, but your Mac has \(total) GB total. It may run very slowly or fail. Continue?"
            pendingRequest = req
            showRAMWarning = true
            return
        }
        service.generate(req, server: server)
    }

    private func showLogWindow() {
        let logText = server.combinedGenLog(own: service.log)
        let alert = NSAlert()
        alert.messageText = "Audio generation log"
        alert.informativeText = logText.isEmpty ? "(no output)" : logText
        alert.runModal()
    }
}
