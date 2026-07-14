import SwiftUI
import AppKit
import AVKit
import AVFoundation
import UniformTypeIdentifiers

/// Video generation window — LTX-Video 2.3, run natively by the mlx-serve server.
/// Uses the same Quality / Resolution preset shape as ImageGen, plus a
/// Frames dropdown clamped to LTX's `8N+1` ladder and the user's RAM
/// budget.
struct VideoGenView: View {
    @EnvironmentObject var service: VideoGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    @State private var prompt: String = ""
    @State private var showAdvanced: Bool = false
    @State private var model: VideoModelPreset = .ltx23Q4
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = VideoModelPreset.ltx23Q4.defaultResolution
    @State private var numFrames: Int = 97
    @State private var fps: Int = 24
    @State private var mode: VideoPipelineMode = .oneStage
    @State private var steps: Int = 12
    @State private var cfgScale: Double = 1.0
    @State private var stgScale: Double = 0.0
    @State private var seed: Int = 42
    /// Style LoRA (Advanced): .safetensors adapter path ("" = none).
    @State private var loraPath: String = ""
    @State private var firstFrameImageURL: URL? = nil
    // ── Speech & sound (audio-to-video) ──
    /// Where the conditioning clip comes from. `.none` → the model invents a
    /// soundtrack from the prompt; `.file`/`.speech` freeze a real clip.
    enum A2VSource: String, CaseIterable, Identifiable {
        case none = "None"
        case file = "Audio file"
        case speech = "Speak text"
        var id: String { rawValue }
    }
    @State private var audioSource: A2VSource = .none
    /// The attached clip (picked file or TTS output). Transient, like the
    /// first-frame image.
    @State private var audioURL: URL? = nil
    @State private var audioDuration: Double? = nil
    @State private var speechText: String = ""
    @State private var audioPlayer: AVAudioPlayer? = nil
    /// Local TTS runner — chains Qwen3-TTS (load → speak → unload) on the same
    /// server, then attaches the WAV as the a2vid clip.
    @StateObject private var tts = AudioGenService()
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: VideoGenRequest? = nil
    @State private var player: AVPlayer?
    /// Keep the model resident after generating (default off → unload).
    @State private var keepResident: Bool = false
    /// Hydration guard — see ImageGenView for the full rationale.
    @State private var hydrating: Bool = false
    @State private var didHydrate: Bool = false

    var body: some View {
        readyView
        .frame(minWidth: 880, minHeight: 660)
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                DispatchQueue.main.async { hydrating = false }
            }
        }
        // Persist the fields not owned by the model/quality/resolution sections.
        .onChange(of: numFrames) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: fps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: mode) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: cfgScale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: stgScale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: service.phase) { _, phase in
            if case .completed(let path) = phase {
                player = AVPlayer(url: URL(fileURLWithPath: path))
                player?.play()
            }
            // Load/unload just happened (or a cancel left the model resident)
            // — reflect it in the residency row right away.
            let repo = model.repo
            Task { await service.refreshResidency(repo: repo, server: server) }
        }
        // Slow residency poll while the window is open: is the model loaded,
        // and how much GPU memory the server holds. Never starts the server.
        .task {
            while !Task.isCancelled {
                await service.refreshResidency(repo: model.repo, server: server)
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
        }
        // TTS finished → attach the spoken line as the a2vid clip.
        .onChange(of: tts.phase) { _, phase in
            if case .completed(let path) = phase, audioSource == .speech {
                attachAudio(URL(fileURLWithPath: path))
            }
        }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    promptSection
                    modelSection
                    qualitySection
                    resolutionSection
                    framesSection
                    firstFrameSection
                    speechSection
                    if showAdvanced { advancedSection } else { advancedToggle }
                    actionRow
                }
                .padding(16)
            }
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                outputFolderLink
            }
            .padding(16)
            .frame(minWidth: 460)
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
                Text("Prompt").font(.subheadline.weight(.semibold))
                Spacer()
                Menu("Examples") {
                    ForEach(Self.examplePrompts, id: \.title) { ex in
                        Button(ex.title) { prompt = ex.body }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .font(.caption)
                Link(destination: URL(string: "https://docs.ltx.video/api-documentation/prompting-guide")!) {
                    Label("Prompt tips", systemImage: "arrow.up.right.square")
                        .font(.caption)
                }
            }
            ZStack(alignment: .topLeading) {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(height: 110)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                    )
                if prompt.isEmpty {
                    Text("Describe your shot like a cinematographer — subject, action, camera movement, lighting, setting. 4–8 sentences. Put spoken dialogue in quotes to make characters talk. Click Examples above for a starting point.")
                        .font(.body)
                        .foregroundStyle(.secondary.opacity(0.6))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            if let hint = promptLengthHint {
                Text(hint).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    /// Soft warning when the prompt is too short for LTX's taste. Official
    /// guidance is 4–8 sentences (~80–180 words); anything under ~15 words
    /// reliably produces incoherent motion.
    private var promptLengthHint: String? {
        let words = prompt.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        guard words > 0, words < 15 else { return nil }
        return "LTX-Video performs best with detailed 4–8 sentence prompts. Try Examples or Prompt tips above."
    }

    /// Canonical LTX-style example prompts seeded into the Examples menu.
    /// Dense, cinematographer-style, covering four common shot types. The
    /// dialogue example matters most: LTX only generates speech when the
    /// spoken words appear in quotes (short phrases, acting directions
    /// between them, per the official prompting guide) — without it the
    /// soundtrack is ambient noise only.
    private static let examplePrompts: [(title: String, body: String)] = [
        ("Talking character (dialogue)",
         "Medium close-up of a woman in her thirties with short auburn hair, seated at a kitchen table in warm morning light. She looks into the camera and says warmly, \"Good morning. I made coffee — it's still hot.\" She pauses, glancing toward the window, then adds with a small smile, \"Come sit with me for a minute.\" Her voice is clear and natural, speaking English. Soft room tone with a faint clink of a cup. The camera holds steady at eye level."),
        ("Cinematic character",
         "Medium shot of a young woman with dark curly hair and freckles, wearing a beige wool coat, walking slowly down a rain-slicked cobblestone street at dusk. She holds a folded paper map in one hand and glances up at the glowing shop windows. The camera tracks her from the side at eye level, then slowly dollies in as she stops. Warm amber light spills from the windows onto the wet stones, contrasting with the deep blue-grey sky. Light rain falls continuously, catching the light."),
        ("Nature aerial",
         "A wide aerial shot sweeps low over a pine forest at sunrise, mist clinging to the treetops in thick white ribbons. The camera glides forward steadily, revealing a narrow river cutting through the valley below, its surface catching the gold of the early sun. A flock of birds lifts off in a loose spiral. Lighting is soft, warm, and directional from the right. Colors are saturated emerald greens and amber golds."),
        ("Product close-up",
         "Close-up of hands in a sunlit kitchen kneading bread dough on a floured wooden counter. The camera holds steady at a low angle, focused tight on the rhythmic press-and-fold motion. Flour dust rises and catches in the shaft of morning light from a window on the left. The hands belong to an older man in a rolled blue shirt, skin weathered and dusted white. Warm natural backlight, muted earth tones."),
    ]

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(VideoModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults(); persist() }
            Text("~\(model.approxRAMGB) GB RAM • Includes audio")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Quality").font(.subheadline.weight(.semibold))
            Picker("", selection: $quality) {
                ForEach(QualityPreset.allCases) { q in
                    Text(q.label).tag(q)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .onChange(of: quality) { _, _ in guard !hydrating else { return }; applyQualityDefaults(); persist() }
            Text(qualityHint)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var qualityHint: String {
        let s = model.settings(quality)
        let durationSec = Double(s.numFrames) / Double(model.fps)
        // With a clip attached, a one-stage preset runs two-stage on the wire
        // (audio-to-video requires it) — say so instead of lying "1-stage".
        let label = (audioURL != nil && s.mode == .oneStage) ? "2-stage (audio-to-video)" : modeLabel(s.mode)
        return "\(label), \(s.steps) steps, \(s.numFrames) frames (~\(String(format: "%.1f", durationSec))s)"
    }

    private func modeLabel(_ m: VideoPipelineMode) -> String {
        switch m {
        case .oneStage:   return "1-stage"
        case .twoStage:   return "2-stage"
        case .twoStageHQ: return "2-stage HQ"
        }
    }

    private var resolutionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Resolution").font(.subheadline.weight(.semibold))
            Picker("", selection: $resolution) {
                ForEach(model.resolutions) { r in
                    Text(r.label).tag(r)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: resolution) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
        }
    }

    private var framesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Frames").font(.subheadline.weight(.semibold))
                Spacer()
                Text("\(numFrames) frames · ~\(String(format: "%.1f", Double(numFrames) / Double(fps)))s")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            // Snap through LTX's valid `8N+1` frame ladder by index, so the
            // slider can only land on generatable lengths (9, 17, 25, … maxFrames).
            frameSlider
            if let warn = frameRAMWarning {
                Text(warn).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    private var frameSlider: some View {
        let opts = availableFrameOptions
        let maxIdx = max(1, opts.count - 1)
        return Slider(
            value: Binding(
                get: {
                    // Live index of the current frame count on the ladder.
                    let i = opts.firstIndex(of: numFrames)
                        ?? opts.lastIndex(where: { $0 <= numFrames })
                        ?? 0
                    return Double(i)
                },
                set: { newVal in
                    let idx = min(opts.count - 1, max(0, Int(newVal.rounded())))
                    numFrames = opts[idx]
                }
            ),
            in: 0...Double(maxIdx),
            step: 1
        )
        .help("Clip length. LTX only generates \(opts.first ?? 9)–\(opts.last ?? 193) frames on its 8N+1 ladder; the slider snaps to valid counts.")
    }

    /// Always show every option up to the model's hard cap. The user can
    /// pick longer than RAM suggests — we just hint at it in the warning
    /// below the dropdown rather than removing the option.
    private var availableFrameOptions: [Int] {
        model.frameOptions
    }

    /// Soft hint when the chosen length looks too aggressive for the Mac's
    /// total RAM at the current resolution. Doesn't block — the user might
    /// know better (e.g. they just freed memory).
    private var frameRAMWarning: String? {
        let cap = RAMChecker.safeFrameCap(
            model: model,
            width: resolution.width,
            height: resolution.height,
            available: RAMChecker.totalGB
        )
        if numFrames > cap {
            return "May exceed your Mac's RAM (\(RAMChecker.totalGB) GB total) at this length."
        }
        return nil
    }

    // Image-to-video is always available: the native mlx-serve engine supports
    // first-frame conditioning in every pipeline mode (the server VAE-encodes
    // the image and pins it as the clean first latent frame), and gracefully
    // falls back to text-to-video if the VAE encoder isn't downloaded — so the
    // picker is never disabled.
    private var firstFrameSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("First frame").font(.subheadline.weight(.semibold))
                Spacer()
                Text("optional — I2V")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let url = firstFrameImageURL {
                HStack(spacing: 8) {
                    if let img = NSImage(contentsOf: url) {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 64, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button {
                        firstFrameImageURL = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Clear first frame")
                }
            } else {
                Button {
                    chooseFirstFrameImage()
                } label: {
                    Label("Choose image...", systemImage: "photo.on.rectangle.angled")
                        .font(.caption)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .help("Select an image to use as the first frame of the video.")
            }
        }
    }

    // ── Speech & sound (audio-to-video) ──
    // Attach real speech/audio and the model generates the video AGAINST it:
    // voices, lip sync and performance follow the clip, and the clip itself
    // becomes the mp4's soundtrack (guaranteed words — no hoping the joint
    // model nails quoted dialogue). Two sources: any audio file, or a line
    // synthesized by the local Qwen3-TTS voice right from this pane.
    private var speechSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Speech & sound").font(.subheadline.weight(.semibold))
                Spacer()
                Text("optional — audio-to-video")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Picker("", selection: $audioSource) {
                ForEach(A2VSource.allCases) { s in Text(s.rawValue).tag(s) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .onChange(of: audioSource) { _, s in
                if s == .none { clearAudio() }
            }

            switch audioSource {
            case .none:
                Text("The model invents a soundtrack from your prompt. Attach speech to make characters say exact words.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            case .file:
                if audioURL == nil {
                    Button {
                        chooseAudioFile()
                    } label: {
                        Label("Choose audio…", systemImage: "waveform.badge.plus")
                            .font(.caption)
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .help("WAV, MP3, M4A or AAC. The clip drives the performance and becomes the video's soundtrack.")
                }
            case .speech:
                speechComposer
            }

            if audioURL != nil {
                attachedAudioChip
                Text("Voices, lip sync and timing follow this clip — it becomes the video's soundtrack. Runs on the 2-stage pipeline.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// The "Speak text" composer: a line + Create speech via local Qwen3-TTS.
    @ViewBuilder
    private var speechComposer: some View {
        let ttsPreset = AudioModelPreset.all.first { ServerManager.resolveModelDir(repo: $0.repo) != nil }
        TextField("Line to speak — e.g. Good morning. Coffee's ready.", text: $speechText, axis: .vertical)
            .textFieldStyle(.roundedBorder)
            .lineLimit(2...4)
            .font(.body)
        if let preset = ttsPreset {
            HStack(spacing: 8) {
                if tts.isRunning {
                    ProgressView().controlSize(.small)
                    if case .running(_, _, let msg) = tts.phase {
                        Text(msg).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Cancel") { tts.cancel() }
                        .font(.caption)
                } else {
                    Button {
                        tts.generate(AudioGenRequest(model: preset, text: speechText), server: server)
                    } label: {
                        Label(audioURL == nil ? "Create speech" : "Recreate speech", systemImage: "waveform")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                    .disabled(speechText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Text(preset.name).font(.caption2).foregroundStyle(.secondary)
                    Spacer()
                }
            }
            if case .failed(let msg) = tts.phase {
                Text(msg).font(.caption2).foregroundStyle(.orange)
            }
        } else {
            Text("Download a voice first — open the Audio window and grab Qwen3-TTS, then come back.")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    /// Attached-clip chip: name, duration, preview play/stop, clear.
    private var attachedAudioChip: some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform")
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(audioURL?.lastPathComponent ?? "")
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                if let d = audioDuration {
                    Text(String(format: "%.1fs%@", d, clipOutlastsVideo ? " — trimmed to the video length" : ""))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Button {
                togglePreview()
            } label: {
                Image(systemName: audioPlayer?.isPlaying == true ? "stop.fill" : "play.fill")
            }
            .buttonStyle(.borderless)
            .help("Preview the clip")
            Button {
                clearAudio()
            } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Remove the clip")
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
    }

    /// Whether the attached clip is longer than the selected video length.
    private var clipOutlastsVideo: Bool {
        guard let d = audioDuration else { return false }
        return d > Double(numFrames) / Double(fps) + 0.05
    }

    private func chooseAudioFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            attachAudio(url)
        }
    }

    /// Attach a clip and snap the frame count up to cover it (capped at the
    /// model max; the server trims a longer clip to the video).
    private func attachAudio(_ url: URL) {
        audioPlayer?.stop()
        audioPlayer = nil
        audioURL = url
        audioDuration = Self.audioDuration(of: url)
        if let d = audioDuration, let f = model.framesCovering(durationSeconds: d) {
            numFrames = f
        }
    }

    private func clearAudio() {
        audioPlayer?.stop()
        audioPlayer = nil
        audioURL = nil
        audioDuration = nil
    }

    private func togglePreview() {
        if audioPlayer?.isPlaying == true {
            audioPlayer?.stop()
            audioPlayer = nil
            return
        }
        guard let url = audioURL, let p = try? AVAudioPlayer(contentsOf: url) else { return }
        audioPlayer = p
        p.play()
    }

    static func audioDuration(of url: URL) -> Double? {
        guard let f = try? AVAudioFile(forReading: url) else { return nil }
        let sr = f.processingFormat.sampleRate
        guard sr > 0 else { return nil }
        return Double(f.length) / sr
    }

    private var advancedToggle: some View {
        Button {
            withAnimation { showAdvanced = true }
        } label: {
            Label("Advanced options", systemImage: "chevron.right")
                .font(.caption)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Advanced (overrides Quality preset)").font(.caption.weight(.semibold))
                Spacer()
                Button {
                    withAnimation { showAdvanced = false }
                } label: { Image(systemName: "chevron.down") }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            // Steps — more steps = more detail/smoother motion, but slower.
            intSliderRow("Steps", value: $steps, range: 4...50,
                         help: "Denoising steps. More = more detail and smoother motion, but slower. LTX runs well from ~8 (fast) to ~30 (reference quality).")
            Text("More steps refine the video further at the cost of speed. ~8 is fast, ~30 is the reference default.")
                .font(.caption2).foregroundStyle(.secondary)

            // CFG scale — always adjustable; the native engine honors it in
            // every pipeline mode (one-stage and both two-stage variants).
            sliderRow("CFG scale", value: $cfgScale, range: 1...10, step: 0.5,
                      help: "Classifier-free guidance strength. LTX-2 default: 3.0; 1.0 = off (fastest).")
            Text("Guidance strength — how closely the video follows your prompt. 1.0 = off: fastest and most natural-looking. Higher sticks to the prompt more strictly but is slower and can look over-saturated. LTX default is 3.0.")
                .font(.caption2).foregroundStyle(.secondary)

            HStack {
                numberField("Seed", value: $seed, step: 1)
                Spacer()
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
            residencyRow

            Divider()
            Text("Style LoRA").font(.caption.weight(.semibold))
            if loraPath.isEmpty {
                Button {
                    chooseLora()
                } label: {
                    Label("Choose .safetensors…", systemImage: "paintpalette")
                        .font(.caption)
                }
                Text("Apply a LoRA adapter to the video model for a custom style.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                HStack(spacing: 8) {
                    Image(systemName: "paintpalette")
                        .foregroundStyle(.secondary)
                    Text(URL(fileURLWithPath: loraPath).lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(loraPath)
                    Spacer()
                    Button {
                        loraPath = ""
                        persist()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Remove the LoRA")
                }
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
            }
        }
    }

    /// Live "is the model resident, and what does the GPU hold" line under the
    /// keep-loaded toggle — fed by the slow `/v1/models` poll.
    private var residencyRow: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(service.residency?.loaded == true ? Color.green : Color.secondary.opacity(0.4))
                .frame(width: 7, height: 7)
            Text(residencyText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .help("Live server state: whether this model is loaded, and the total memory held by all loaded models.")
    }

    private var residencyText: String {
        guard server.status == .running, let r = service.residency else {
            return "Model not loaded"
        }
        let gpu = MemoryInfo.format(r.gpuResidentBytes)
        if r.loaded {
            return "Model loaded · GPU memory \(gpu)"
        }
        // Other models resident without this one → say who holds the GPU
        // (a chat model, or another pane's model).
        if r.gpuResidentBytes > (1 << 29) {
            return "Model not loaded · GPU memory \(gpu) in use"
        }
        return "Model not loaded"
    }

    private func chooseLora() {
        let panel = NSOpenPanel()
        if let st = UTType(filenameExtension: "safetensors") {
            panel.allowedContentTypes = [st]
        }
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            loraPath = url.path
            persist()
        }
    }

    private func chooseFirstFrameImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            firstFrameImageURL = url
        }
    }

    private func numberField(_ label: String, value: Binding<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            Stepper(value: value, step: step) {
                Text(String(value.wrappedValue))
            }
        }
    }

    /// Labeled slider for a `Double` setting, with a live value readout on the
    /// right and an optional hover tooltip.
    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double, help: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text(String(format: "%.1f", value.wrappedValue))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
        }
        .help(help ?? "")
    }

    /// Labeled slider for an `Int` setting (bridges to a `Double` slider).
    private func intSliderRow(_ label: String, value: Binding<Int>, range: ClosedRange<Int>, help: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text("\(value.wrappedValue)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { Double(value.wrappedValue) },
                    set: { value.wrappedValue = Int($0.rounded()) }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound),
                step: 1
            )
        }
        .help(help ?? "")
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle)
            }
            HStack {
                if service.isRunning {
                    Button(role: .destructive) {
                        service.cancel()
                    } label: {
                        Label("Cancel", systemImage: "stop.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button {
                        tryGenerate()
                    } label: {
                        Label("Generate", systemImage: "wand.and.stars")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !downloads.bundleReady(model.bundle))
                }
            }
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No generation yet", systemImage: "film", description: Text("Enter a prompt and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                case .completed(let path):
                    completedPreview(path: path)
                case .cancelled:
                    ContentUnavailableView("Cancelled", systemImage: "stop.circle", description: Text("Generation was cancelled."))
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
        VStack(spacing: 8) {
            if let player {
                AVPlayerViewRepresentable(player: player)
                    .frame(minHeight: 240)
            }
            HStack(spacing: 8) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless)
                .help("Reveal in Finder")
            }
        }
        .padding(8)
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting(
                [URL(fileURLWithPath: MediaStorage.videosRoot)]
            )
        } label: {
            Label("Open output folder in Finder", systemImage: "folder")
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.videosRoot)
    }

    // MARK: - Sticky settings

    private func hydrate() {
        let s = VideoGenSettings.load()
        model = s.resolvedModel
        quality = s.quality
        resolution = s.resolvedResolution(for: model)
        numFrames = s.numFrames
        fps = s.fps
        mode = s.mode
        // Clamp into the slider ranges — a value persisted by the old wider
        // steppers (Steps unbounded, CFG 0…20) would otherwise sit off-scale.
        steps = min(50, max(4, s.steps))
        cfgScale = min(10, max(1, s.cfgScale))
        stgScale = s.stgScale
        seed = s.seed
        keepResident = s.keepResident
        loraPath = s.loraPath
        // The LoRA file may have moved since last session — drop a stale path.
        if !loraPath.isEmpty && !FileManager.default.fileExists(atPath: loraPath) {
            loraPath = ""
        }
        clampFramesToRAM()
    }

    private func persist() {
        var s = VideoGenSettings()
        s.modelId = model.id
        s.quality = quality
        s.resolutionId = resolution.id
        s.numFrames = numFrames
        s.fps = fps
        s.mode = mode
        s.steps = steps
        s.cfgScale = cfgScale
        s.stgScale = stgScale
        s.seed = seed
        s.keepResident = keepResident
        s.loraPath = loraPath
        s.save()
    }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        resolution = model.defaultResolution
        fps = model.fps
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        let s = model.settings(quality)
        mode = s.mode
        steps = s.steps
        cfgScale = s.cfgScale
        stgScale = s.stgScale
        numFrames = s.numFrames
        clampFramesToRAM()
        // Keep firstFrameImageURL across preset changes so users can swap
        // Quality tiers without losing their attached image — every pipeline
        // mode supports first-frame conditioning.
    }

    /// Resolution change still snaps frame count down to the model's hard
    /// cap (`8N+1` ladder) — but no RAM-based clamping anymore. The user
    /// gets a soft warning instead.
    private func clampFramesToRAM() {
        if numFrames > model.maxFrames,
           let snap = model.frameOptions.last(where: { $0 <= model.maxFrames }) {
            numFrames = snap
        }
    }

    /// Soft gate: only warn when the model needs more RAM than the Mac has
    /// total. macOS's "available" reading is misleading on unified memory
    /// (idle apps get paged out under pressure) — using it as a hard gate
    /// blocked legitimate runs, so we let the user override.
    private func tryGenerate() {
        let req = VideoGenRequest(
            model: model,
            prompt: prompt,
            seed: seed,
            width: resolution.width,
            height: resolution.height,
            numFrames: numFrames,
            fps: fps,
            mode: mode,
            steps: steps,
            cfgScale: cfgScale,
            stgScale: stgScale,
            firstFrameImagePath: firstFrameImageURL?.path,
            audioPath: audioURL?.path,
            keepResident: keepResident,
            loraPath: loraPath.isEmpty ? nil : loraPath
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
        let text = service.log.joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "Video generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}

// MARK: - AVPlayerView wrapper

/// Direct `NSViewRepresentable` around AVKit's `AVPlayerView`. We use this
/// instead of SwiftUI's generic `VideoPlayer<VideoOverlay>` because on
/// macOS 26.4 the Swift runtime fatal-aborts while resolving VideoPlayer's
/// generic metadata when it's mounted via a state-driven transition
/// (phase `.running` → `.completed`), crashing the whole app.
private struct AVPlayerViewRepresentable: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}
