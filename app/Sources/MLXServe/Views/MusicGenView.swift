import SwiftUI
import AppKit

/// Music tab — prompt-driven music generation ("in the style of…"), run
/// natively by the embedded mlx-serve server (ACE-Step v1.5 XL Turbo). Same
/// visual language as VoiceGenView/Model3DGenView: prompt + optional lyrics,
/// model picker, duration, advanced section, and a player for the result.
/// A menu wearing the chip. `.borderlessButton` hands the label to AppKit,
/// which keeps the text and throws the background away, and draws its own
/// indicator; `.button` renders the label as a real button, so the chip
/// survives and the chevron can live inside it.
private struct PaneChipMenu: ViewModifier {
    func body(content: Content) -> some View {
        content
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
    }
}

struct MusicGenView: View {
    @EnvironmentObject var service: MusicGenService
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject var downloads: DownloadManager
    /// For the model row's Download button (`MediaModelChooser`) — a completed
    /// transfer has to re-scan the models directory.
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    @State private var lyrics: String = ""
    /// Height of the lyrics editor — dragged by `lyricsResizeHandle`, sticky.
    @State private var lyricsHeight: Double = PromptEditorHeight.defaultHeight
    /// Height when the current drag began (nil = not dragging); `translation`
    /// is cumulative, so applying it to the live height compounds.
    @State private var lyricsHeightAtDragStart: Double? = nil
    @State private var model: MusicModelPreset = .acestepXLTurbo8bit
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var durationSeconds: Double = 60
    @State private var vocalLanguage: String = "en"
    @State private var bpm: Int? = nil
    @State private var keyscale: String = ""
    @State private var timesignature: String = ""
    @State private var seed: Int = -1
    @State private var steps: Int? = nil
    @State private var instrumental: Bool = false
    /// Open by default — see MusicGenSettings.showAdvanced.
    @State private var showAdvanced: Bool = true
    @StateObject private var library = MusicPromptLibrary()
    @State private var showSaveStyle = false
    @State private var showSaveLyrics = false
    @State private var saveTitle = ""
    @State private var rewriteKind: MusicPromptRewriter.Kind? = nil

    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: MusicGenRequest? = nil
    // The app-wide singleton, not a per-view instance: this view unmounts on
    // navigation (see the tab-persistence note on AudioGenView), and a
    // private player left playing when that happens is a leaked NSSound with
    // no reachable Stop button — see .onDisappear below.
    @ObservedObject private var clipPlayer = AudioClipPlayer.shared
    // Kept across preset switches like the video pane's first frame; the
    // SERVICE gates the field on `supportsReferenceAudio`.
    @State private var refAudioURL: URL? = nil
    @State private var refError: String? = nil
    @State private var isRefDropTargeted: Bool = false
    @State private var refBusy: Bool = false
    // Source-audio tasks (ACE-Step only). The SERVICE gates every field on
    // `supportsSourceAudio` + the task, so a clip left behind by a model
    // switch never reaches a server that refuses it.
    @State private var task: MusicTask = .text2music
    @State private var srcAudioURL: URL? = nil
    /// The clip's length, read from its header once when it arrives — the file
    /// runs to ~100 MB, so this is never re-read while the pane redraws.
    @State private var srcSeconds: Double? = nil
    @State private var srcError: String? = nil
    @State private var isSrcDropTargeted: Bool = false
    @State private var srcBusy: Bool = false
    @State private var coverStrength: Double = 1.0
    @State private var coverNoiseStrength: Double = 0.0
    @State private var trackClasses: [String] = []
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
        .onDisappear { stopPlayback() }
        .onChange(of: model) { _, m in
            guard !hydrating else { return }
            durationSeconds = min(max(durationSeconds, m.durationRange.lowerBound), m.durationRange.upperBound)
            persist()
        }
        // Everything the pane shows is sticky, the typed draft included — it
        // all used to reset on every navigation away from the Audio page,
        // since the view unmounts. ONE onChange on the snapshot: a chain of
        // fourteen is where the type-checker gives up.
        .onChange(of: stickySnapshot) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: service.phase) { _, phase in
            // A new generation stops whatever is still playing.
            if case .running = phase { stopPlayback() }
            if case .completed(let path) = phase { play(path) }
        }
        .alert("Save style prompt", isPresented: $showSaveStyle) {
            TextField("Name", text: $saveTitle)
            Button("Save") { library.saveStyle(title: saveTitle, body: prompt) }
                .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Give this style a name to reuse it from the Templates menu.")
        }
        .alert("Save lyrics", isPresented: $showSaveLyrics) {
            TextField("Name", text: $saveTitle)
            Button("Save") { library.saveLyrics(title: saveTitle, body: lyrics) }
                .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Give these lyrics a name to reuse them from the Templates menu.")
        }
        .sheet(item: $rewriteKind) { kind in
            PromptRewriteSheet(
                title: kind == .style ? "Rewrite style prompt" : "Rewrite lyrics",
                request: MusicPromptRewriter.request(
                    kind, text: kind == .style ? prompt : lyrics, family: model.family,
                    other: kind == .style ? lyrics : prompt,
                    instrumental: instrumental, language: vocalLanguage),
                onApply: { if kind == .style { prompt = $0 } else { lyrics = $0 } })
            .environmentObject(appState)
        }
    }

    private func play(_ path: String) { clipPlayer.play(path) }
    private func stopPlayback() { clipPlayer.stop() }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                // The model decides what the rest of the pane means — whether
                // there are source-audio modes at all, what Advanced holds,
                // how long a track may be — so it is read first.
                VStack(alignment: .leading, spacing: 14) {
                    modelSection
                    if model.supportsSourceAudio { modeSection }
                    if sourceTask { sourceSection }
                    promptSection
                    lyricsSection
                    if model.supportsReferenceAudio { referenceSection }
                    // No Duration in a source task: the clip is the length, and
                    // the Source well already says how long that is.
                    if !sourceTask { durationSection }
                    advancedSection.padding(.top, 8)
                    // Generate stands apart from the settings it acts on.
                    actionRow.padding(.top, 14)
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
                    onStop: { stopPlayback() }
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
                if let req = pendingRequest { service.generate(req, server: server, downloads: downloads) }
                pendingRequest = nil
            }
        } message: {
            Text(ramWarningMessage)
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Style prompt").font(.subheadline.weight(.semibold))
                Spacer()
                rewriteButton(.style, text: prompt)
                styleExamplesMenu
            }
            TextEditor(text: $prompt)
                .font(.body)
                .frame(height: 80)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
            Text("Genre, mood, instruments — e.g. \"upbeat synthwave with driving bass and dreamy pads\".")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var lyricsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Lyrics").font(.subheadline.weight(.semibold))
                // The instrumental switch. Both engines can make a wordless
                // track, but only ACE-Step ever said so (empty lyrics) and
                // Music 3 refused outright — the server 400s an empty lyric
                // block there, so this needed the `instrumental` field before a
                // checkbox could work.
                Toggle("Instrumental", isOn: $instrumental)
                    .font(.caption)
                    .fixedSize()
                    .help(model.family == .minimaxMusic3
                          ? "Asks for a track with no singing. This model has no dedicated instrumental switch, so it is requested in text — it may still add wordless vocals."
                          : "Generate music with no singing. The lyrics below are not used.")
                Spacer()
                rewriteButton(.lyrics, text: lyrics)
                lyricsExamplesMenu
            }
            if instrumental {
                // The box goes away, the words do not: `lyrics` is untouched
                // and still persisted, so turning the switch back off returns
                // the verse. Deleting it here is the failure mode the server's
                // named 400 exists to prevent. Music 3 is hedged because its
                // open weights expose no `is_instrumental` equivalent and every
                // text arm tried so far still produced wordless vocal texture;
                // ACE-Step's marker is its own documented convention.
                Text(model.family == .minimaxMusic3
                     ? "In instrumental mode, lyrics are not used. This model has no instrumental switch of its own, so it is asked in text and may still add wordless vocals. This is an experimental feature."
                     : "In instrumental mode, lyrics are not used.")
                    .font(.caption2).foregroundStyle(.secondary)
            } else {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $lyrics)
                        .font(.body)
                        .frame(height: lyricsHeight)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                        )
                    if lyrics.isEmpty {
                        Text(model.requiresLyrics
                             ? "This model sings your lyrics. Section tags go on their own lines: \(MusicOptions.sectionTagHint)"
                             : "Leave empty, or tick Instrumental, for a track with no vocals. Section tags: \(MusicOptions.sectionTagHint)")
                            .font(.body)
                            .foregroundStyle(.secondary.opacity(0.6))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 8)
                            .allowsHitTesting(false)
                    }
                }
                lyricsResizeHandle
            }
        }
    }

    /// Drag strip under the lyrics editor: a verse is a document, and 90pt of
    /// it is a keyhole. Full width so it is easy to grab; the height sticks,
    /// clamped both ways so one dragged on a tall window cannot come back
    /// unusable.
    private var lyricsResizeHandle: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.35))
            .frame(width: 36, height: 4)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 3)
            .contentShape(Rectangle())
            .gesture(
                // GLOBAL, never the default `.local`: the handle sits under the
                // box it resizes, so growing the box moves the handle, and a
                // translation measured in a space that moved subtracts its own
                // effect — the box then tracks at half the cursor's speed and
                // re-solves every frame.
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { v in
                        let base = lyricsHeightAtDragStart ?? lyricsHeight
                        if lyricsHeightAtDragStart == nil { lyricsHeightAtDragStart = base }
                        lyricsHeight = PromptEditorHeight.clamp(base + v.translation.height)
                    }
                    .onEnded { _ in
                        lyricsHeightAtDragStart = nil
                        persist()
                    }
            )
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
            }
            .help("Drag to resize the lyrics box.")
    }

    /// The task actually in force: the mode control only exists on models
    /// with source-audio tasks, so elsewhere the sticky value is inert.
    private var sourceTask: Bool { model.supportsSourceAudio && task.needsSource }

    /// What the track is made from. Cover keeps the source's melody and
    /// structure under a new caption; Vocal to BGM arranges around a vocal
    /// stem. Both take their length from the clip, so Duration disappears.
    private var modeSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("", selection: $task) {
                // The Cover segment declares a missing tokenizer in its OWN
                // label, so the reason is readable before the mode is even
                // selected — the whole point of #269.
                ForEach(MusicTask.allCases, id: \.self) { t in
                    Text(CoverWeightsFetch.modeLabel(t, decision: t == .cover ? coverWeights : .ready)).tag(t)
                }
            }
            .labelsHidden().pickerStyle(.segmented)
            Text(task == .cover
                 ? "Re-sings an existing track in the style you describe: melody and structure stay, the caption and lyrics decide the rest."
                 : (task == .complete
                    ? "Builds an arrangement around a vocal stem (or any part): pick the instruments to add, or leave them all off to let the model decide."
                    : "A new track from the style prompt and lyrics."))
                .font(.caption2).foregroundStyle(.secondary)
            coverWeightsNotice
        }
    }

    /// Whether Cover can run against the pack on THIS Mac. Purely local — the
    /// pack dir resolves across every served root, so an LM Studio or custom
    /// folder answers too, and no generation has to be set up to find out.
    private var coverWeights: CoverWeightsFetch.Decision {
        CoverWeightsFetch.decide(
            task: .cover,
            modelSupportsSourceAudio: model.supportsSourceAudio,
            isRemote: lanModel != nil,
            packDir: lanModel == nil ? ServerManager.resolveModelDir(repo: model.repo) : nil,
            fetching: downloads.isFetchingPackFile(repoId: model.repo))
    }

    /// Says what is missing, by name, and offers the ONE file — never the 5 GB
    /// pack again. The engine stats for the tokenizer on every cover request
    /// (`acestep.Engine.fsqAvailable`), so a resident model picks the file up
    /// as soon as it lands: no reload, no app restart, nothing to warn about.
    @ViewBuilder
    private var coverWeightsNotice: some View {
        if task == .cover, let text = CoverWeightsFetch.notice(coverWeights) {
            VStack(alignment: .leading, spacing: 6) {
                Text(text).font(.caption2).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
                switch coverWeights {
                case .fetch:
                    Button {
                        downloads.startPackFile(repoId: model.repo,
                                                fileName: CoverWeightsFetch.fileName) {
                            appState.refreshModels()
                        }
                    } label: {
                        Label("Download \(CoverWeightsFetch.fileName) (\(CoverWeightsFetch.approxMB) MB)",
                              systemImage: "arrow.down.circle")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                case .downloading:
                    HStack(spacing: 8) {
                        ProgressView(value: downloads.downloads[model.repo]?.progress ?? 0)
                            .progressViewStyle(.linear)
                        Text(downloads.downloads[model.repo]?.percentFormatted ?? "")
                            .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        // Cancels ONLY this single-file fetch — a full pack
                        // download for the same repo is never touched.
                        Button("Cancel") { downloads.cancelPackFile(repoId: model.repo) }
                            .buttonStyle(.borderless).font(.caption)
                    }
                default:
                    EmptyView()
                }
            }
            .padding(8)
            .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
        }
    }

    /// The source clip for Cover / Vocal to BGM: full length (up to 10 minutes),
    /// because the track is made exactly as long as it. Same pick / drop /
    /// preview / clear shape as the reference well below.
    private var sourceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(task == .cover ? "Source track" : "Source stem").font(.subheadline.weight(.semibold))
                Spacer()
            }
            if let url = srcAudioURL {
                MediaDropWellFilled(isTargeted: isSrcDropTargeted) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Image(systemName: "waveform.circle.fill").foregroundStyle(.blue)
                            Text(url.lastPathComponent)
                                .font(.caption).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            if clipPlayer.playingPath == url.path {
                                Button { clipPlayer.stop() } label: { Image(systemName: "stop.circle.fill") }
                                    .buttonStyle(.borderless).help("Stop preview")
                            } else {
                                Button { clipPlayer.play(url.path) } label: { Image(systemName: "play.circle") }
                                    .buttonStyle(.borderless).help("Preview source")
                            }
                            Button { clearSource() } label: { Image(systemName: "xmark.circle.fill") }
                                .buttonStyle(.borderless).foregroundStyle(.secondary).help("Clear source")
                        }
                        if let seconds = srcSeconds {
                            Text("The new track will be \(SourceTrackLength.spoken(seconds: seconds)) long.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            } else if srcBusy {
                MediaDropWellFilled(isTargeted: isSrcDropTargeted) { converting }
            } else {
                MediaDropWell(title: "Choose file…",
                              systemImage: "waveform.badge.plus",
                              caption: "10 seconds to 10 minutes. The new track will have the same length.",
                              isTargeted: isSrcDropTargeted,
                              action: chooseSourceFile)
            }
            if let err = srcError {
                Text(err).font(.caption2).foregroundStyle(.orange)
            }
            if task == .cover {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Cover strength").font(.caption)
                        Spacer()
                        Text(String(format: "%.2f", coverStrength)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Slider(value: $coverStrength, in: 0...1, step: 0.05)
                    Text("How many of the steps follow the source. 1 keeps it all the way; lower lets the caption take over for the last steps.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                // The knobs under the clip are their own step, not a caption
                // of the well above them.
                .padding(.top, 6)
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Noise strength").font(.caption)
                        Spacer()
                        Text(String(format: "%.2f", coverNoiseStrength)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                    Slider(value: $coverNoiseStrength, in: 0...1, step: 0.05)
                    Text("0 starts from pure noise (the default). Higher starts closer to the original audio, so more of it comes through.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                .padding(.top, 6)
            }
            if task == .complete {
                Text("Add").font(.caption).padding(.top, 6)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 110), alignment: .leading)], alignment: .leading, spacing: 4) {
                    ForEach(MusicTask.trackClasses, id: \.self) { name in
                        Toggle(name.replacingOccurrences(of: "_", with: " "),
                               isOn: Binding(get: { trackClasses.contains(name) },
                                             set: { on in
                                                 if on { if !trackClasses.contains(name) { trackClasses.append(name) } }
                                                 else { trackClasses.removeAll { $0 == name } }
                                             }))
                            .font(.caption)
                    }
                }
            }
        }
        .mediaDrop(.audio, isTargeted: $isSrcDropTargeted) { urls in
            if let url = urls.first { acceptSource(url) }
        }
    }

    private func chooseSourceFile() {
        srcError = nil
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        acceptSource(url)
    }

    private func acceptSource(_ url: URL) {
        guard !srcBusy else { return }
        srcError = nil
        srcBusy = true
        transcode(url, maxSeconds: 600) { result in
            srcBusy = false
            switch result {
            case .success(let wav):
                srcAudioURL = wav
                srcSeconds = SourceTrackLength.seconds(ofWavAt: wav)
            case .failure(let err): srcError = err.localizedDescription
            }
        }
    }

    /// Decode + resample + write off the main thread: a 10-minute clip took
    /// seconds, and the drop animation sat frozen for all of them.
    private func transcode(_ url: URL, maxSeconds: Double, done: @escaping @MainActor (Result<URL, Error>) -> Void) {
        Task.detached(priority: .userInitiated) {
            let result = Result { try AudioReference.referenceWav(fromFile: url, maxSeconds: maxSeconds) }
            await done(result)
        }
    }

    private func clearSource() {
        if let url = srcAudioURL { try? FileManager.default.removeItem(at: url) }
        srcAudioURL = nil
        srcSeconds = nil
    }

    /// ACE-Step's timbre slot (#259): a clip whose style the track follows.
    /// Same pick / drop / preview / clear shape as the Audio pane's voice
    /// reference, minus recording — a mic clip is no style reference.
    private var referenceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Reference audio (optional)").font(.subheadline.weight(.semibold))
                Spacer()
            }
            if let url = refAudioURL {
                MediaDropWellFilled(isTargeted: isRefDropTargeted) {
                    HStack(spacing: 8) {
                        Image(systemName: "waveform.circle.fill").foregroundStyle(.blue)
                        Text(url.lastPathComponent)
                            .font(.caption).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        if clipPlayer.playingPath == url.path {
                            Button { clipPlayer.stop() } label: { Image(systemName: "stop.circle.fill") }
                                .buttonStyle(.borderless).help("Stop preview")
                        } else {
                            Button { clipPlayer.play(url.path) } label: { Image(systemName: "play.circle") }
                                .buttonStyle(.borderless).help("Preview reference")
                        }
                        Button { clearReference() } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.borderless).foregroundStyle(.secondary).help("Clear reference")
                    }
                }
            } else if refBusy {
                MediaDropWellFilled(isTargeted: isRefDropTargeted) { converting }
            } else {
                MediaDropWell(title: "Choose file…",
                              systemImage: "music.note",
                              caption: "A light touch: the model takes the overall feel and timbre from up to 30 seconds of the clip, it does not recreate the song.",
                              isTargeted: isRefDropTargeted,
                              action: chooseReferenceFile)
            }
            if let err = refError {
                Text(err).font(.caption2).foregroundStyle(.orange)
            }
        }
        .mediaDrop(.audio, isTargeted: $isRefDropTargeted) { urls in
            if let url = urls.first { acceptReference(url) }
        }
    }

    private var converting: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Converting…").font(.caption).foregroundStyle(.secondary)
        }
    }

    private func chooseReferenceFile() {
        refError = nil
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio, .aiff]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        acceptReference(url)
    }

    /// Picked or dropped, a file becomes the reference the same way: a 48 kHz
    /// stereo transcode, or a visible reason instead of a clip that silently
    /// doesn't attach.
    private func acceptReference(_ url: URL) {
        guard !refBusy else { return }
        refError = nil
        refBusy = true
        transcode(url, maxSeconds: 30) { result in
            refBusy = false
            switch result {
            case .success(let wav): refAudioURL = wav
            case .failure(let err): refError = err.localizedDescription
            }
        }
    }

    private func clearReference() {
        if let url = refAudioURL { try? FileManager.default.removeItem(at: url) }
        refAudioURL = nil
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`. The transfer
    /// bar and residency both belong to the model, not to the track, so they
    /// sit with it rather than beside Generate.
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            modelChooser
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                // Local-only models have no HF download yet — steer the user to
                // the on-device conversion instead of a Download button.
                if model.isLocalOnly { convertHint } else { BundleDownloadBar(bundle: model.bundle, showsStartButton: false) }
            }
        }
    }

    /// Residency rides the switcher's row: it is a property of the model, and
    /// the only thing about it the pane still has to say once it is picked.
    private var keepResidentToggle: AnyView {
        AnyView(
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .controlSize(.small)
                .fixedSize()
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        )
    }

    private var modelChooser: some View {
        MediaModelChooser.pane(
            all: MusicModelPreset.all,
            onThisMac: CustomMediaModels.musicPresets(from: server.allModels),
            capability: "music",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.musicPreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist,
            accessory: keepResidentToggle)
    }

    private var durationSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            // The box belongs NEXT to the label it edits. Right-justifying it
            // against the pane margin puts a number and its name at opposite
            // ends of a wide row with nothing between them to tie the two.
            HStack(spacing: 6) {
                Text("Duration").font(.subheadline.weight(.semibold))
                // Typed entry beside the slider: the slider steps by 5 and
                // landing on 95 s by dragging is not a thing anyone should do.
                NumberField(range: durationRangeInt,
                            value: Binding(get: { Int(durationSeconds) },
                                           set: { durationSeconds = Double($0) }),
                            width: 52,
                            help: "Seconds. \(durationRangeInt.lowerBound)–\(durationRangeInt.upperBound) for this model.")
                Text("sec · \(formattedDuration)").font(.caption2).foregroundStyle(.secondary)
                Spacer()
            }
            Slider(value: $durationSeconds, in: model.durationRange, step: 5)
            if model.family == .minimaxMusic3 {
                Text("An upper bound — the model may end the song earlier.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    /// Label over control, pinned to the cell's leading edge.
    private func advancedCell<C: View>(_ label: String, @ViewBuilder control: () -> C) -> some View {
        // No `maxWidth: .infinity`: that stretched every cell across its whole
        // grid column, which reads as an arbitrary gap between two controls.
        // The cell hugs its content and the grid sets it at the leading edge.
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            control()
        }
    }

    /// The model's server-valid duration bounds as integers, for the typed box.
    private var durationRangeInt: ClosedRange<Int> {
        Int(model.durationRange.lowerBound)...Int(model.durationRange.upperBound)
    }

    /// The shared seed component (dice + forgiving paste). The pane's old bare
    /// TextField parsed with `Int(...)`, so a pasted "Seed: 3,847,592" fell
    /// through to -1 and rolled a RANDOM seed — the user believed they had
    /// reproduced a track and had not.
    private var seedControl: some View {
        SeedField(label: "Seed", placeholder: "Random",
                  range: -1...Int(UInt32.max), value: $seed)
    }

    private var formattedDuration: String {
        let s = Int(durationSeconds)
        return String(format: "%d:%02d", s / 60, s % 60)
    }

    /// One header in both states: same words, same chevron on the same side,
    /// and the whole row is the control.
    private var advancedHeader: some View {
        Button {
            withAnimation { showAdvanced.toggle() }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: showAdvanced ? "chevron.down" : "chevron.right")
                Text("Advanced options")
                Spacer()
            }
            .font(.subheadline.weight(.semibold))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.primary)
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            advancedHeader
            if showAdvanced { advancedBody }
        }
    }

    private var advancedBody: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Refinement passes. Music 3 only: ACE-Step Turbo is distillation-
            // fixed at 8 and the server ignores the field, so a control there
            // would visibly do nothing.
            if model.supportsSteps {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Quality passes").font(.caption)
                        NumberField(range: model.stepsRange,
                                    value: Binding(get: { steps ?? model.fixedSteps },
                                                   set: { steps = $0 }),
                                    width: 52,
                                    help: "Flow-matching steps, \(model.stepsRange.lowerBound)–\(model.stepsRange.upperBound). Default \(model.fixedSteps).")
                        Spacer()
                    }
                    Slider(value: Binding(get: { Double(steps ?? model.fixedSteps) },
                                          set: { steps = Int($0.rounded()) }),
                           in: Double(model.stepsRange.lowerBound)...Double(model.stepsRange.upperBound),
                           step: 1)
                    Text("More passes means more detail and a slower render.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            // Dropdowns only — every choice is a value the server accepts,
            // "Auto" leaves the decision to the model (field omitted). The
            // whole musical-metadata knob set is ACE-Step's; Music 3 has no
            // equivalent (the server names each field a 400), so the controls
            // disappear with it — and `requestBody` gates the FIELDS too.
            // Tempo and key are supported by BOTH engines — conditioning fields
            // on ACE-Step, caption text on Music 3 — so they sit OUTSIDE the
            // acestep-only block. Hiding them on Music 3 read as "this model
            // cannot do tempo", which its own model card contradicts.
            // Tempo, key and seed share a row when the pane is wide enough and
            // WRAP when it is not: a fixed-width HStack here set the left
            // pane's minimum width (~540 pt) above what the default window
            // gives it, and the HSplitView pushed the whole column under the
            // sidebar (live 2026-08-22, Music tab clipped at default size).
            FlowLayout(spacing: 14, rowSpacing: 10) {
                if model.supportsTempoAndKey {
                    advancedCell("Tempo (BPM)") {
                        // Typed, because the server takes 30–300 and the ten
                        // anchors in the menu could not express 92. The menu stays
                        // as a shortcut for people who think in genres, not numbers.
                        HStack(spacing: 4) {
                            OptionalNumberField(range: MusicOptions.bpmRange, value: $bpm,
                                                placeholder: "Auto", width: 64,
                                                help: "\(MusicOptions.bpmRange.lowerBound)–\(MusicOptions.bpmRange.upperBound), or leave empty to let the model decide.")
                            Menu {
                                Button("Auto") { bpm = nil }
                                ForEach(MusicOptions.bpms, id: \.bpm) { opt in
                                    Button(opt.label) { bpm = opt.bpm }
                                }
                            } label: {
                                Image(systemName: "chevron.down").modifier(PaneChip(square: true))
                            }
                            .modifier(PaneChipMenu())
                            .help("Common tempos")
                        }
                    }
                    advancedCell("Key") {
                        Picker("", selection: $keyscale) {
                            Text("Auto").tag("")
                            ForEach(MusicOptions.keyscales, id: \.self) { key in
                                Text(MusicOptions.keyLabel(key)).tag(key)
                            }
                        }
                        .labelsHidden().pickerStyle(.menu).fixedSize()
                    }
                }
                if model.supportsMusicalMeta {
                    advancedCell("Vocal language") {
                        Picker("", selection: $vocalLanguage) {
                            ForEach(MusicOptions.languages, id: \.code) { opt in
                                Text(opt.label).tag(opt.code)
                            }
                        }
                        .labelsHidden().pickerStyle(.menu).fixedSize()
                    }
                    advancedCell("Time signature") {
                        Picker("", selection: $timesignature) {
                            Text("Auto").tag("")
                            ForEach(MusicOptions.timeSignatures, id: \.value) { opt in
                                Text(opt.label).tag(opt.value)
                            }
                        }
                        .labelsHidden().pickerStyle(.menu).fixedSize()
                    }
                }
                seedControl
            }
            if model.supportsTempoAndKey && model.family == .minimaxMusic3 {
                Text("Tempo and key are written into the style prompt for this model — it has no separate fields for them.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Text("Same seed + prompt reproduces the track.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var actionRow: some View {
        HStack {
            if service.isRunning {
                Button(role: .destructive) { service.cancel() } label: {
                    Label("Cancel", systemImage: "stop.circle").frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            } else {
                Button { tryGenerate() } label: {
                    Label("Generate", systemImage: "music.note").frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || !MusicGenRequest.lyricsSatisfied(model: model, lyrics: lyrics,
                                                              instrumental: instrumental)
                          || (sourceTask && srcAudioURL == nil)
                          || (lanModel == nil && !downloads.bundleReady(model.bundle)))
            }
        }
    }

    private var convertHint: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label("Weights not found", systemImage: "wrench.and.screwdriver")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text("\(model.name) has no download yet — convert the weights on-device with the matching script in the repo (see its README). They install to ~/.mlx-serve/models/local/.")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No music yet", systemImage: "music.note",
                                           description: Text("Describe a style, optionally add lyrics, and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
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
        // One control, two states: pausing left the shelf lit under a clip
        // that had stopped making sound, so there is no pause any more.
        let playing = clipPlayer.playingPath == path
        return VStack(spacing: 12) {
            trackGlyph(playing: playing)
            Button {
                playing ? clipPlayer.stop() : clipPlayer.play(path)
            } label: {
                Label(playing ? "Stop" : "Play", systemImage: playing ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.bordered)
            // The name and the way to reach the file belong together, centred
            // under the track they describe.
            HStack(spacing: 8) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless).help("Reveal in Finder")
            }
        }
        .padding(16)
    }

    /// There is no note-in-a-circle symbol, so the disc is drawn and the note
    /// is PUNCHED out of it: `.destinationOut` clears what it covers, which
    /// needs the stack to composite as one layer first. The hole is what
    /// bounces.
    private func trackGlyph(playing: Bool) -> some View {
        ZStack {
            Circle()
                .fill(.tint)
                .frame(width: 64, height: 64)
            Image(systemName: "music.note.list")
                .font(.system(size: 32))
                .blendMode(.destinationOut)
                .symbolEffect(.bounce.down.byLayer, options: .repeat(.continuous), isActive: playing)
        }
        .compositingGroup()
    }

    private func showLogWindow() {
        AppActivation.openWindow(id: "serverLog", using: openWindow)
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.musicRoot)])
        } label: {
            Label("Open output folder in Finder", systemImage: "folder").font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.musicRoot)
    }

    // MARK: - Sticky settings

    private func hydrate() {
        let s = MusicGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        durationSeconds = Double(s.durationSeconds)
        vocalLanguage = s.vocalLanguage
        keepResident = s.keepResident
        bpm = s.bpm
        keyscale = s.keyscale
        timesignature = s.timesignature
        seed = s.seed
        steps = s.steps
        instrumental = s.instrumental
        showAdvanced = s.showAdvanced
        prompt = s.prompt
        lyrics = s.lyrics
        lyricsHeight = PromptEditorHeight.clamp(s.lyricsHeight)
        refAudioURL = s.refAudioPath.flatMap { FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil }
        task = s.task
        srcAudioURL = s.srcAudioPath.flatMap { FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil }
        srcSeconds = srcAudioURL.flatMap { SourceTrackLength.seconds(ofWavAt: $0) }
        coverStrength = s.coverStrength
        coverNoiseStrength = s.coverNoiseStrength
        trackClasses = s.trackClasses
    }

    /// Every sticky field, as the blob it would persist to — `Equatable`, so
    /// one `onChange` covers all of them.
    private var stickySnapshot: MusicGenSettings {
        var s = MusicGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.durationSeconds = Int(durationSeconds)
        s.vocalLanguage = vocalLanguage
        s.keepResident = keepResident
        s.bpm = bpm
        s.keyscale = keyscale
        s.timesignature = timesignature
        s.seed = seed
        s.steps = steps
        s.instrumental = instrumental
        s.showAdvanced = showAdvanced
        s.prompt = prompt
        s.lyrics = lyrics
        s.lyricsHeight = PromptEditorHeight.clamp(lyricsHeight)
        s.refAudioPath = refAudioURL?.path
        s.task = task
        s.srcAudioPath = srcAudioURL?.path
        s.coverStrength = coverStrength
        s.coverNoiseStrength = coverNoiseStrength
        s.trackClasses = trackClasses
        return s
    }

    private func persist() { stickySnapshot.save() }


    // MARK: - Examples

    /// The wand: asks the chat model to rewrite the field like the current
    /// family's examples. Disabled until there is something to rewrite.
    private func rewriteButton(_ kind: MusicPromptRewriter.Kind, text: String) -> some View {
        let off = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || (kind == .lyrics && instrumental)
        return Button { rewriteKind = kind } label: {
            HStack(spacing: 5) {
                Image(systemName: "wand.and.sparkles")
                Text("Enhance…")
            }
            .modifier(PaneChip())
        }
        .buttonStyle(.plain)
        .disabled(off)
        // A `.plain` button over our own background does not dim itself.
        .opacity(off ? 0.4 : 1)
        .help("Rewrite with the chat model")
    }

    /// Style-prompt Templates menu: Save current + your saved styles (with a
    /// Delete submenu) + the built-in genre starters.
    private var styleExamplesMenu: some View {
        Menu {
            Button("Save current…") {
                saveTitle = MusicPromptStore.autoTitle(from: prompt)
                showSaveStyle = true
            }
            .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if !library.savedStyles.isEmpty {
                Section("Saved") {
                    ForEach(library.savedStyles) { p in
                        Button(p.title) { prompt = p.body }
                    }
                }
                Menu("Delete saved…") {
                    ForEach(library.savedStyles) { p in
                        Button(role: .destructive) { library.deleteStyle(title: p.title) } label: {
                            Label(p.title, systemImage: "trash")
                        }
                    }
                }
            }
            Section("Example templates") {
                ForEach(MusicPrompt.builtinStyles(for: model.family)) { p in
                    Button(p.title) { prompt = p.body }
                }
            }
        } label: {
            templatesChip
        }
        .modifier(PaneChipMenu())
    }

    /// The menu's own button face: the system indicator is hidden so the
    /// chevron sits INSIDE the chip, beside the word.
    private var templatesChip: some View {
        HStack(spacing: 5) {
            Text("Templates")
            Image(systemName: "chevron.down")
        }
        .modifier(PaneChip())
    }

    /// Lyrics Templates menu: Save current + your saved lyrics (with a Delete
    /// submenu) + built-in ORIGINAL lyric templates to start from.
    private var lyricsExamplesMenu: some View {
        Menu {
            Button("Save current…") {
                saveTitle = MusicPromptStore.autoTitle(from: lyrics)
                showSaveLyrics = true
            }
            .disabled(lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            if !library.savedLyrics.isEmpty {
                Section("Saved") {
                    ForEach(library.savedLyrics) { p in
                        Button(p.title) { lyrics = p.body }
                    }
                }
                Menu("Delete saved…") {
                    ForEach(library.savedLyrics) { p in
                        Button(role: .destructive) { library.deleteLyrics(title: p.title) } label: {
                            Label(p.title, systemImage: "trash")
                        }
                    }
                }
            }
            Section("Example templates") {
                ForEach(MusicPrompt.builtinLyrics) { p in
                    Button(p.title) { lyrics = p.body }
                }
            }
        } label: {
            templatesChip
        }
        .modifier(PaneChipMenu())
    }

    // MARK: - Generate

    private func tryGenerate() {
        let req = MusicGenRequest(
            model: model,
            prompt: prompt,
            lyrics: lyrics,
            instrumental: instrumental,
            vocalLanguage: vocalLanguage,
            bpm: bpm,
            keyscale: keyscale,
            timesignature: timesignature,
            durationSeconds: Int(durationSeconds),
            seed: seed,
            steps: steps,
            keepResident: keepResident,
            refAudioPath: refAudioURL?.path,
            task: task,
            srcAudioPath: srcAudioURL?.path,
            coverStrength: coverStrength,
            coverNoiseStrength: coverNoiseStrength,
            trackClasses: trackClasses,
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
        service.generate(req, server: server, downloads: downloads)
    }
}

// MARK: - Rewrite with LLM

/// The wand sheet: streams the chat model's rewrite into an editable box;
/// Apply hands the edited text back, Try again re-asks, Cancel keeps the
/// original untouched.
struct PromptRewriteSheet: View {
    let title: String
    let request: MusicPromptRewriter.Request
    let onApply: (String) -> Void
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var text: String = ""
    @State private var isWriting = false
    @State private var error: String? = nil
    @State private var job: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title).font(.headline)
                Spacer()
                if isWriting { ProgressView().controlSize(.small) }
            }
            TextEditor(text: $text)
                .font(.body)
                .frame(minHeight: 220)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            if let error {
                Text(error).font(.caption).foregroundStyle(.red)
            } else {
                Text("Edit the result, then Apply to replace your text.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            HStack {
                Button("Try again") { start() }.disabled(isWriting)
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Apply") { onApply(text); dismiss() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(isWriting || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 520)
        .onAppear { start() }
        .onDisappear { job?.cancel() }
    }

    private func start() {
        job?.cancel()
        text = ""
        error = nil
        isWriting = true
        job = Task {
            defer { isWriting = false }
            do {
                let stream = try await AgentComposer.stream(userText: request.user, systemPrompt: request.system,
                                                            appState: appState, maxTokens: 1024)
                for try await delta in stream {
                    if Task.isCancelled { return }
                    text += delta
                }
                text = MusicPromptRewriter.clean(text)
            } catch is CancellationError {
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
