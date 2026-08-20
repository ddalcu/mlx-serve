import SwiftUI
import AppKit

/// Music tab — prompt-driven music generation ("in the style of…"), run
/// natively by the embedded mlx-serve server (ACE-Step v1.5 XL Turbo). Same
/// visual language as VoiceGenView/Model3DGenView: prompt + optional lyrics,
/// model picker, duration, advanced section, and a player for the result.
struct MusicGenView: View {
    @EnvironmentObject var service: MusicGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    /// For the model row's Download button (`MediaModelChooser`) — a completed
    /// transfer has to re-scan the models directory.
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    @State private var lyrics: String = ""
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

    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: MusicGenRequest? = nil
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
        .onChange(of: model) { _, m in
            guard !hydrating else { return }
            durationSeconds = min(max(durationSeconds, m.durationRange.lowerBound), m.durationRange.upperBound)
            persist()
        }
        .onChange(of: durationSeconds) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: vocalLanguage) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        // Everything the pane shows is sticky now — these all used to reset on
        // every navigation away from the Audio page, since the view unmounts.
        .onChange(of: bpm) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keyscale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: timesignature) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: instrumental) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: showAdvanced) { _, _ in guard !hydrating else { return }; persist() }
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
            Text("Give this style a name to reuse it from the Examples menu.")
        }
        .alert("Save lyrics", isPresented: $showSaveLyrics) {
            TextField("Name", text: $saveTitle)
            Button("Save") { library.saveLyrics(title: saveTitle, body: lyrics) }
                .keyboardShortcut(.defaultAction)
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Give these lyrics a name to reuse them from the Examples menu.")
        }
    }

    private func play(_ path: String) { clipPlayer.play(path) }
    private func stopPlayback() { clipPlayer.stop() }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    promptSection
                    lyricsSection
                    modelSection
                    durationSection
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

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Style prompt").font(.subheadline.weight(.semibold))
                Spacer()
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
                Text(model.requiresLyrics ? "Lyrics" : "Lyrics (optional)")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                lyricsExamplesMenu
            }
            // The instrumental switch. Both engines can make a wordless track,
            // but only ACE-Step ever said so (empty lyrics) and Music 3 refused
            // outright — the server 400s an empty lyric block there, so this
            // needed the `instrumental` field before a checkbox could work.
            // Honest label on Music 3: the open weights expose no
            // `is_instrumental` equivalent, and every text arm tried so far
            // still produced wordless vocal texture. ACE-Step's marker is its
            // own documented convention and does work, so it is not hedged.
            Toggle(model.family == .minimaxMusic3
                   ? "Instrumental (no vocals) — experimental"
                   : "Instrumental (no vocals)", isOn: $instrumental)
                .font(.caption)
                .help(model.family == .minimaxMusic3
                      ? "Asks for a track with no singing. This model has no dedicated instrumental switch, so it is requested in text — it may still add wordless vocals."
                      : "Generate music with no singing. The lyrics below are not used.")
            TextEditor(text: $lyrics)
                .font(.body)
                .frame(height: 90)
                .disabled(instrumental)
                .opacity(instrumental ? 0.45 : 1)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
            // Say the words are being ignored rather than deleting them: a
            // sticky checkbox silently discarding a typed verse is the failure
            // mode the server's named 400 exists to prevent.
            Text(instrumental
                 ? "Not used while Instrumental is on. Your lyrics are kept if you turn it off."
                 : (model.requiresLyrics
                    ? "This model sings your lyrics. Section tags go on their own lines: \(MusicOptions.sectionTagHint)"
                    : "Leave empty, or tick Instrumental, for a track with no vocals. Section tags: \(MusicOptions.sectionTagHint)"))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`.
    private var modelSection: some View {
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
            persist: persist)
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

    /// Every dropdown in Advanced is this wide. They were 110 / 130 / 90,
    /// which made a row of three menus look like a mistake, and Key needed the
    /// room once its labels carried the key's character. Sized for the longest
    /// entry it will ever hold ("C# major", "A minor — plain sad") with slack,
    /// because a menu that truncates its own options is worse than a bare one —
    /// `.frame(width:)` on a Picker clips, it does not wrap or shrink.
    private var menuWidth: CGFloat { 210 }

    /// The model's server-valid duration bounds as integers, for the typed box.
    private var durationRangeInt: ClosedRange<Int> {
        Int(model.durationRange.lowerBound)...Int(model.durationRange.upperBound)
    }

    /// The shared seed component (dice + forgiving paste). The pane's old bare
    /// TextField parsed with `Int(...)`, so a pasted "Seed: 3,847,592" fell
    /// through to -1 and rolled a RANDOM seed — the user believed they had
    /// reproduced a track and had not.
    private var seedControl: some View {
        SeedField(label: "Seed", placeholder: "random",
                  range: -1...Int(UInt32.max), value: $seed)
    }

    private var formattedDuration: String {
        let s = Int(durationSeconds)
        return String(format: "%d:%02d", s / 60, s % 60)
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
            // Tempo, key and seed share one row — three short controls that
            // each owned a full-width line, in a pane whose whole problem is
            // that its options are hard to find.
            if model.supportsTempoAndKey {
            HStack(alignment: .bottom, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Tempo (BPM)").font(.caption)
                    // Typed, because the server takes 30–300 and the ten
                    // anchors in the menu could not express 92. The menu stays
                    // as a shortcut for people who think in genres, not numbers.
                    HStack(spacing: 2) {
                        OptionalNumberField(range: MusicOptions.bpmRange, value: $bpm,
                                            placeholder: "Auto", width: 52,
                                            help: "\(MusicOptions.bpmRange.lowerBound)–\(MusicOptions.bpmRange.upperBound), or leave empty to let the model decide.")
                        Menu {
                            Button("Auto") { bpm = nil }
                            ForEach(MusicOptions.bpms, id: \.bpm) { opt in
                                Button(opt.label) { bpm = opt.bpm }
                            }
                        } label: { Image(systemName: "chevron.down") }
                        .menuStyle(.borderlessButton).fixedSize()
                        .help("Common tempos")
                    }
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Key").font(.caption)
                    Picker("", selection: $keyscale) {
                        Text("Auto").tag("")
                        ForEach(MusicOptions.keyscales, id: \.self) { key in
                            Text(MusicOptions.keyLabel(key)).tag(key)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: menuWidth)
                }
                seedControl
                Spacer()
            }
            if model.family == .minimaxMusic3 {
                Text("Tempo and key are written into the style prompt for this model — it has no separate fields for them.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            }
            if model.supportsMusicalMeta {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Vocal language").font(.caption)
                    Picker("", selection: $vocalLanguage) {
                        ForEach(MusicOptions.languages, id: \.code) { opt in
                            Text(opt.label).tag(opt.code)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: menuWidth)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Time signature").font(.caption)
                    Picker("", selection: $timesignature) {
                        Text("Auto").tag("")
                        ForEach(MusicOptions.timeSignatures, id: \.value) { opt in
                            Text(opt.label).tag(opt.value)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: menuWidth)
                }
            }
            }
            Text("Same seed + prompt reproduces the track.")
                .font(.caption2).foregroundStyle(.secondary)
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        }
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                // Local-only models have no HF download yet — steer the user to
                // the on-device conversion instead of a Download button.
                if model.isLocalOnly { convertHint } else { BundleDownloadBar(bundle: model.bundle, showsStartButton: false) }
            }
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
                              || (lanModel == nil && !downloads.bundleReady(model.bundle)))
                }
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
                        // The Voice tab has had this; music's failure state
                        // offered nothing. `combinedGenLog` falls back to the
                        // server tail, which is where a model that failed to
                        // LOAD leaves its reason.
                        Button("Show log") { showLogWindow() }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func completedPreview(path: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "music.note.list")
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
                // The one bridge from the workshop to a conversation — the
                // Voice tab has had it, music did not. Chat renders a real
                // player for an `.audio` ref (ChatMediaAttachmentView).
                Button {
                    appState.sendGeneratedMediaToNewChat(
                        path: path, prompt: prompt, kind: .audio)
                } label: { Image(systemName: "bubble.left.and.text.bubble.right") }
                .buttonStyle(.borderless)
                .help("Send to Chat — opens a new conversation with this attached")
            }
        }
        .padding(16)
    }

    private func showLogWindow() {
        let logText = server.combinedGenLog(own: service.log)
        let alert = NSAlert()
        alert.messageText = "Music generation log"
        alert.informativeText = logText.isEmpty ? "(no output)" : logText
        alert.runModal()
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
    }

    private func persist() {
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
        s.save()
    }

    // MARK: - Examples

    /// Style-prompt Examples menu: Save current + your saved styles (with a
    /// Delete submenu) + the built-in genre starters.
    private var styleExamplesMenu: some View {
        Menu("Examples") {
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
                        Button(p.title, role: .destructive) { library.deleteStyle(title: p.title) }
                    }
                }
            }
            Section("Examples") {
                ForEach(MusicPrompt.builtinStyles(for: model.family)) { p in
                    Button(p.title) { prompt = p.body }
                }
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .font(.caption)
    }

    /// Lyrics Examples menu: Save current + your saved lyrics (with a Delete
    /// submenu) + built-in ORIGINAL lyric templates to start from.
    private var lyricsExamplesMenu: some View {
        Menu("Examples") {
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
                        Button(p.title, role: .destructive) { library.deleteLyrics(title: p.title) }
                    }
                }
            }
            Section("Templates") {
                ForEach(MusicPrompt.builtinLyrics) { p in
                    Button(p.title) { lyrics = p.body }
                }
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .font(.caption)
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
}
