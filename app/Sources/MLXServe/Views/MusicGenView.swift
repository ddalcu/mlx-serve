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

    @State private var prompt: String = ""
    @State private var lyrics: String = ""
    @State private var model: MusicModelPreset = .acestepXLTurbo8bit
    @State private var durationSeconds: Double = 60
    @State private var vocalLanguage: String = "en"
    @State private var bpm: Int? = nil
    @State private var keyscale: String = ""
    @State private var timesignature: String = ""
    @State private var seedText: String = ""
    @State private var showAdvanced: Bool = false
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
        readyView
        .frame(minWidth: 820, minHeight: 600)
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                DispatchQueue.main.async { hydrating = false }
            }
        }
        .onChange(of: model) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: durationSeconds) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: vocalLanguage) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: service.phase) { _, phase in
            // A new generation stops whatever is still playing.
            if case .running = phase { stopPlayback() }
            if case .completed(let path) = phase { play(path) }
        }
        .alert("Save style prompt", isPresented: $showSaveStyle) {
            TextField("Name", text: $saveTitle)
            Button("Save") { library.saveStyle(title: saveTitle, body: prompt) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Give this style a name to reuse it from the Examples menu.")
        }
        .alert("Save lyrics", isPresented: $showSaveLyrics) {
            TextField("Name", text: $saveTitle)
            Button("Save") { library.saveLyrics(title: saveTitle, body: lyrics) }
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
                    onStop: { stopPlayback() }
                )
                outputFolderLink
            }
            .padding(16)
            .frame(minWidth: 420)
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
                Text("Lyrics (optional)").font(.subheadline.weight(.semibold))
                Spacer()
                lyricsExamplesMenu
            }
            TextEditor(text: $lyrics)
                .font(.body)
                .frame(height: 90)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
            Text("Leave empty for an instrumental track.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(MusicModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            Text("~\(model.approxRAMGB) GB RAM • \(model.fixedSteps)-step distilled")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var durationSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Duration (\(formattedDuration))").font(.subheadline.weight(.semibold))
            Slider(value: $durationSeconds, in: 10...600, step: 5)
        }
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
            // Dropdowns only — every choice is a value the server accepts,
            // "Auto" leaves the decision to the model (field omitted).
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Vocal language").font(.caption)
                    Picker("", selection: $vocalLanguage) {
                        ForEach(MusicOptions.languages, id: \.code) { opt in
                            Text(opt.label).tag(opt.code)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: 110)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Tempo").font(.caption)
                    Picker("", selection: $bpm) {
                        Text("Auto").tag(Int?.none)
                        ForEach(MusicOptions.bpms, id: \.bpm) { opt in
                            Text(opt.label).tag(Int?.some(opt.bpm))
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: 160)
                }
            }
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Key").font(.caption)
                    Picker("", selection: $keyscale) {
                        Text("Auto").tag("")
                        ForEach(MusicOptions.keyscales, id: \.self) { key in
                            Text(key).tag(key)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: 130)
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text("Time signature").font(.caption)
                    Picker("", selection: $timesignature) {
                        Text("Auto").tag("")
                        ForEach(MusicOptions.timeSignatures, id: \.value) { opt in
                            Text(opt.label).tag(opt.value)
                        }
                    }
                    .labelsHidden().pickerStyle(.menu).frame(width: 90)
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Seed").font(.caption)
                TextField("random", text: $seedText)
                    .textFieldStyle(.roundedBorder).font(.caption).frame(width: 140)
                Text("Same seed + prompt reproduces the track.").font(.caption2).foregroundStyle(.secondary)
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        }
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if !downloads.bundleReady(model.bundle) {
                // Local-only models have no HF download yet — steer the user to
                // the on-device conversion instead of a Download button.
                if model.isLocalOnly { convertHint } else { BundleDownloadBar(bundle: model.bundle) }
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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !downloads.bundleReady(model.bundle))
                }
            }
        }
    }

    private var convertHint: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label("Weights not found", systemImage: "wrench.and.screwdriver")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text("ACE-Step 1.5 has no download yet — convert the weights on-device with tests/convert_acestep_weights.py (see the repo README). They install to ~/.mlx-serve/models/local/.")
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
            }
        }
        .padding(16)
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
        model = s.resolvedModel
        durationSeconds = Double(s.durationSeconds)
        vocalLanguage = s.vocalLanguage
        keepResident = s.keepResident
    }

    private func persist() {
        var s = MusicGenSettings()
        s.modelId = model.id
        s.durationSeconds = Int(durationSeconds)
        s.vocalLanguage = vocalLanguage
        s.keepResident = keepResident
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
                ForEach(MusicPrompt.builtinStyles) { p in
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
            vocalLanguage: vocalLanguage,
            bpm: bpm,
            keyscale: keyscale,
            timesignature: timesignature,
            durationSeconds: Int(durationSeconds),
            seed: Int(seedText.trimmingCharacters(in: .whitespaces)) ?? -1,
            keepResident: keepResident
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
