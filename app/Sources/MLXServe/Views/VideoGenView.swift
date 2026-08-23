import SwiftUI
import AppKit
import AVKit
import AVFoundation
import UniformTypeIdentifiers

/// Video generation window — LTX-Video (2.3 / 2.5) and MiniMax-H3, run
/// natively by the mlx-serve server. Uses the same Quality / Resolution preset
/// shape as ImageGen, plus a Frames slider clamped to LTX's `8N+1` ladder and
/// the user's RAM budget. Which controls are offered is decided by the
/// preset's declared capabilities, never by its id — the two LTX releases
/// share every one of them.
struct VideoGenView: View {
    @EnvironmentObject var service: VideoGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    /// For "Send to Chat" — the hand-off opens a new conversation and switches
    /// the window to it (`AppState.sendGeneratedMediaToNewChat`).
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    /// Height of the prompt editor — dragged by `promptResizeHandle`, sticky.
    @State private var promptHeight: Double = PromptEditorHeight.defaultHeight
    /// Height when the current drag began (nil = not dragging); `translation`
    /// is cumulative from the gesture's start, so it needs an anchor.
    @State private var promptHeightAtDragStart: Double? = nil
    @State private var showAdvanced: Bool = false
    @State private var model: VideoModelPreset = .ltx23Q4
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = VideoModelPreset.ltx23Q4.defaultResolution
    // Held as text so a half-typed size is allowed while editing.
    @State private var customWidthText: String = "704"
    @State private var customHeightText: String = "448"
    @State private var numFrames: Int = 97
    @State private var fps: Int = 24
    @State private var mode: VideoPipelineMode = .oneStage
    @State private var steps: Int = 12
    @State private var cfgScale: Double = 1.0
    @State private var stgScale: Double = 0.0
    // 0 = Auto (the server's own "all 3"), so the control has an explicit Auto
    // position rather than a 0 that reads as "no refine at all".
    @State private var stage2Steps: Int = 0
    @State private var cfgAudioScale: Double = 7.0
    @State private var chainWindows: Int = 1
    @State private var seed: Int = 42
    /// Style LoRAs (Advanced): stacked `.safetensors` adapters ([] = none).
    /// Several can attach at once — their effects sum, so order doesn't matter.
    @State private var loras: [LoraAdapter] = []
    @State private var firstFrameImageURL: URL? = nil
    // The other half of fl2va. Kept across preset changes like the first
    // frame; the SERVICE gates the field on `supportsLastFrame`, so a leftover
    // pick can never reach a backend without the anchor.
    @State private var lastFrameImageURL: URL? = nil
    @State private var isLastFrameDropTargeted = false
    // ref2va references, kept across preset changes like firstFrameImageURL —
    // `requestBody` gates the fields on the pack's own capability, so state
    // left behind by a preset switch can never reach an FL2VA request.
    @State private var refImageURLs: [URL] = []
    @State private var refVideoURLs: [URL] = []
    @State private var refAudioURLs: [URL] = []
    @State private var refImageSize: RefImageSizing = .match
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
    @State private var bestQuality: Bool = false
    @State private var diffusionDecoder: Bool = false
    /// Turbo distillation LoRA (H3 fl2va): 4-step sampling, recipe off.
    @State private var turbo: Bool = false
    /// Hydration guard — see ImageGenView for the full rationale.
    @State private var hydrating: Bool = false
    @State private var didHydrate: Bool = false
    /// True while a drag carrying a file hovers the first-frame section —
    /// drives that section's dashed-border highlight (see `MediaDropTarget`).
    @State private var isDropTargeted: Bool = false
    /// The same, for the ref2va References section, which is its own target.
    @State private var isRefDropTargeted: Bool = false

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
        // Persist the fields not owned by the model/quality/resolution sections.
        .onChange(of: numFrames) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: customWidthText) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
        .onChange(of: customHeightText) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
        .onChange(of: fps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: mode) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: cfgScale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: stgScale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: stage2Steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: cfgAudioScale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: chainWindows) { _, _ in guard !hydrating else { return }; persist() }
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
                    lastFrameSection
                    referencesSection
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
            // The preview gives way in a small window; the player scales.
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
                Text("Prompt").font(.subheadline.weight(.semibold))
                Spacer()
                Menu("Examples") {
                    ForEach(examplePrompts, id: \.title) { ex in
                        Button(ex.title) { prompt = ex.body }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .font(.caption)
                Link(destination: H3PromptExamples.tipsURL(for: model.promptFormat)) {
                    Label("Prompt tips", systemImage: "arrow.up.right.square")
                        .font(.caption)
                }
            }
            ZStack(alignment: .topLeading) {
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(height: promptHeight)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                    )
                if prompt.isEmpty {
                    Text(H3PromptExamples.placeholder(for: model.promptFormat))
                        .font(.body)
                        .foregroundStyle(.secondary.opacity(0.6))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            promptResizeHandle
            if let hint = promptHint {
                Text(hint).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    /// Drag strip under the prompt editor. H3's format is a multi-section
    /// document, so 110pt is a keyhole — full width so it's easy to grab, and
    /// the height sticks (clamped on the way in and out, so a value dragged on
    /// a taller window can't come back unusable).
    private var promptResizeHandle: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.35))
            .frame(width: 36, height: 4)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 3)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 1)
                    .onChanged { v in
                        let base = promptHeightAtDragStart ?? promptHeight
                        if promptHeightAtDragStart == nil { promptHeightAtDragStart = base }
                        promptHeight = PromptEditorHeight.clamp(base + v.translation.height)
                    }
                    .onEnded { _ in
                        promptHeightAtDragStart = nil
                        persist()
                    }
            )
            .onHover { inside in
                if inside { NSCursor.resizeUpDown.push() } else { NSCursor.pop() }
            }
            .help("Drag to resize the prompt box.")
    }

    /// Soft caption under the prompt field. Per-BACKEND: LTX's "4–8 sentences"
    /// is advice for a different engine on an H3 model, where the thing the
    /// user cannot guess is the section labels. Pure logic in
    /// `H3PromptExamples` so it is testable.
    private var promptHint: String? {
        H3PromptExamples.hint(for: model.promptFormat, prompt: prompt)
    }

    /// Example prompts for the selected model's format — LTX prose, H3's
    /// three-field base format, or H3 REF2VA's six sections.
    private var examplePrompts: [VideoPromptExample] {
        H3PromptExamples.examples(for: model.promptFormat)
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`.
    private var modelSection: some View {
        MediaModelChooser.pane(
            all: VideoModelPreset.all,
            onThisMac: CustomMediaModels.videoPresets(from: server.allModels),
            capability: "video",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.videoPreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults(); persist() }
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
        // Gated on the capability so a stale clip can't make H3 claim it.
        let label = (model.supportsAudioInput && audioURL != nil && s.mode == .oneStage)
            ? "2-stage (audio-to-video)" : modeLabel(s.mode)
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
                ForEach(model.resolutionOptions()) { r in
                    Text(r.label).tag(r)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: resolution) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
            if resolution.isCustom { customResolutionFields }
            // A two-stage tier denoises at HALF this canvas and upscales, so on
            // a small pick "Quality" is softer than the one-stage tiers above
            // it in the same menu. Only shown when that tier is selected — the
            // note is about the combination, not the resolution.
            if model.settings(quality).mode != .oneStage,
               let note = model.twoStageCanvasNote(width: effectiveSize.width, height: effectiveSize.height) {
                Text(note).font(.caption2).foregroundStyle(.orange)
            }
        }
    }

    /// Width/height for the Custom… row. The server REFUSES an off-grid video
    /// canvas outright (unlike the image path, which quietly rewrites it), so
    /// this is the difference between a hint and a failed generation.
    @ViewBuilder
    private var customResolutionFields: some View {
        let verdict = customResolutionVerdict
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                labelledSizeField("Width", text: $customWidthText)
                Text("×").foregroundStyle(.secondary)
                labelledSizeField("Height", text: $customHeightText)
            }
            if let hint = verdict.hint {
                Label(hint, systemImage: verdict.isValid ? "wand.and.stars" : "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(verdict.isValid ? Color.secondary : Color.orange)
            }
        }
    }

    private func labelledSizeField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: 80)
        }
    }

    /// A two-stage tier denoises at HALF the canvas, so the server tightens its
    /// refusal to /64 there. The grid follows the SELECTED tier, or the hint
    /// would name a step the request will not be judged by.
    private var customResolutionVerdict: CustomResolution {
        model.resolutionGrid(twoStage: model.settings(quality).mode != .oneStage)
            .resolve(width: Int(customWidthText) ?? 0, height: Int(customHeightText) ?? 0)
    }

    private var customSizeValid: Bool {
        !resolution.isCustom || customResolutionVerdict.isValid
    }

    /// The canvas the request should carry.
    private var effectiveSize: (width: Int, height: Int) {
        guard resolution.isCustom else { return (resolution.width, resolution.height) }
        return customResolutionVerdict.size ?? (resolution.width, resolution.height)
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
        model.frameOptions(width: effectiveSize.width, height: effectiveSize.height)
    }

    /// Soft hint when the chosen length looks too aggressive for the Mac's
    /// total RAM at the current resolution. Doesn't block — the user might
    /// know better (e.g. they just freed memory).
    private var turboEngaged: Bool { turbo && model.supportsTurbo }
    /// What the server's recipe actually runs: turbo forces it off, so every
    /// plan/estimate call reads THIS, never `!bestQuality` alone.
    private var effectiveFast: Bool { !bestQuality && !turboEngaged }
    /// The steps slider under turbo offers the LoRA's own trained range.
    private var effectiveStepsRange: ClosedRange<Int> {
        turboEngaged ? 4...16 : model.stepsRange
    }

    /// guess.
    private var frameRAMWarning: String? {
        let total = RAMChecker.totalGB
        let cap = RAMChecker.safeFrameCap(
            model: model,
            width: effectiveSize.width,
            height: effectiveSize.height,
            available: total,
            fast: effectiveFast
        )
        guard model.backend == .minimaxH3 else {
            guard numFrames > cap else { return nil }
            return "May exceed your Mac's RAM (\(total) GB total) at this length."
        }
        // Ask whether THIS configuration fits, not whether it is longer than
        // the cap: the cap has a floor, so "cap == 124" means both "124 frames
        // fits" and "nothing fits", and a Mac too small to load the pack saw
        // no warning at all at exactly 124 frames.
        guard !H3Plan.fits(model: model, width: effectiveSize.width, height: effectiveSize.height,
                           frames: numFrames, fast: effectiveFast, turbo: turboEngaged, availableGB: total) else { return nil }
        let gib = 1024.0 * 1024.0 * 1024.0
        let need = Double(H3Plan.peakBytes(model: model, width: effectiveSize.width, height: effectiveSize.height,
                                           frames: numFrames, fast: effectiveFast, turbo: turboEngaged)) / gib
        var out = String(format: "Needs about %.0f GB at this size and length; your Mac has %d GB. ", need, total)
        let floorFits = H3Plan.fits(model: model, width: effectiveSize.width, height: effectiveSize.height,
                                    frames: cap, fast: effectiveFast, turbo: turboEngaged, availableGB: total)
        out += floorFits && cap < numFrames ? "About \(cap) frames fits here" : "Try a smaller resolution"
        if effectiveFast {
            let slow = Double(H3Plan.peakBytes(model: model, width: effectiveSize.width, height: effectiveSize.height,
                                               frames: numFrames, fast: false)) / gib
            if slow < need - 2 {
                out += String(format: ", or turn on Max quality to drop the step cache (%.0f GB, but several times slower)", slow)
            }
        }
        return out + "."
    }

    /// "about 50 min — estimated for M4 Max", under the Generate button.
    private var timeEstimate: String? {
        guard model.backend == .minimaxH3, lanModel == nil else { return nil }
        return H3TimeEstimate.describeBest(
            model: model, width: effectiveSize.width, height: effectiveSize.height,
            frames: numFrames, steps: steps, fast: effectiveFast
        )
    }

    // Image-to-video is always available: the native mlx-serve engine supports
    // first-frame conditioning in every pipeline mode (the server VAE-encodes
    // the image and pins it as the clean first latent frame), and gracefully
    // falls back to text-to-video if the VAE encoder isn't downloaded — so the
    // picker is never disabled.
    private var firstFrameSection: some View {
        keyframeWell(title: "First frame",
                     note: model.supportsLastFrame ? "optional — starts here" : "optional — I2V",
                     url: $firstFrameImageURL,
                     isTargeted: $isDropTargeted,
                     help: "Select an image to use as the first frame of the video.")
    }

    // The second anchor (H3 fl2va, LTX both pipelines). Hidden rather than
    // offered-and-ignored on a backend without it (the `pipeline`-on-H3
    // rule): ref2va has no keyframe row to land on.
    @ViewBuilder
    private var lastFrameSection: some View {
        if model.supportsLastFrame {
            VStack(alignment: .leading, spacing: 6) {
                keyframeWell(title: "Last frame",
                             note: "optional — ends here",
                             url: $lastFrameImageURL,
                             isTargeted: $isLastFrameDropTargeted,
                             help: "Select the image the clip should land on. The first frame sets the size; this one is fitted to it.")
                if lastFrameImageURL != nil && firstFrameImageURL == nil {
                    Text("With only a last frame the model invents the opening and works toward it. Add a first frame to pin both ends.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// One keyframe slot: thumbnail + clear when filled, drop well when empty.
    /// Both anchors draw the same shape so they read as a pair.
    private func keyframeWell(title: String, note: String, url: Binding<URL?>,
                              isTargeted: Binding<Bool>, help: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(title).font(.subheadline.weight(.semibold))
                Spacer()
                Text(note)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let picked = url.wrappedValue {
                HStack(spacing: 8) {
                    if let img = NSImage(contentsOf: picked) {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 64, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    Text(picked.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button {
                        url.wrappedValue = nil
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Clear \(title.lowercased())")
                }
            } else {
                // The same well the Image and 3D panes' empty states draw —
                // one shape for "a picture goes here" across the four panes.
                MediaDropWell(title: "Choose image...",
                              systemImage: "photo.on.rectangle.angled",
                              isTargeted: isTargeted.wrappedValue) { chooseKeyframeImage(into: url) }
                    .help(help)
            }
        }
        // One image slot, so a drop REPLACES whatever is there — same as
        // picking again. Drops land on this section rather than the whole
        // window; see `MediaDropTarget`.
        .mediaDrop(.image, isTargeted: isTargeted) { urls in
            if let dropped = urls.first { url.wrappedValue = dropped }
        }
    }

    // ── ref2va references ──
    // Only the REF2VA pack has the DiT for this; FL2VA would generate while
    // ignoring every reference, so the whole section is hidden rather than
    // offered and refused. Declared by the preset, never inferred from the id.
    @ViewBuilder
    private var referencesSection: some View {
        if model.supportsReferences {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("References").font(.subheadline.weight(.semibold))
                    Spacer()
                    Text("optional — the generation follows them")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("Refer to them in the prompt as <Picture 1>, <Video 1>, <Audio 1> — the numbering is per type, in the order below.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("Drag files in — each one joins the list for its own type.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                refList(title: "Images", limit: H3RefLimits.images, urls: $refImageURLs,
                        addLabel: "Add image...", systemImage: "photo.on.rectangle.angled") {
                    chooseRefFiles(types: [.image, .png, .jpeg, .heic],
                                   limit: refRemaining(perType: H3RefLimits.images, current: refImageURLs.count),
                                   into: $refImageURLs)
                }
                refList(title: "Clips", limit: H3RefLimits.videos, urls: $refVideoURLs,
                        addLabel: "Add clip...", systemImage: "film") {
                    chooseRefFiles(types: [.movie, .mpeg4Movie, .quickTimeMovie],
                                   limit: refRemaining(perType: H3RefLimits.videos, current: refVideoURLs.count),
                                   into: $refVideoURLs)
                }
                refList(title: "Audio", limit: H3RefLimits.audios, urls: $refAudioURLs,
                        addLabel: "Add audio...", systemImage: "waveform") {
                    chooseRefFiles(types: [.audio, .mp3, .wav, .mpeg4Audio],
                                   limit: refRemaining(perType: H3RefLimits.audios, current: refAudioURLs.count),
                                   into: $refAudioURLs)
                }
                if let note = H3RefLimits.totalNote(attached: refFilesAttached) {
                    Text(note).font(.caption2).foregroundStyle(.orange)
                }

                if !refImageURLs.isEmpty {
                    Picker("Image detail", selection: $refImageSize) {
                        ForEach(RefImageSizing.allCases, id: \.self) { s in
                            Text(s.label).tag(s)
                        }
                    }
                    .font(.caption)
                    // Reference tokens ride through EVERY sampling step, so
                    // this is a real time cost, not a quality knob.
                    .help("How large each reference image is fed to the model. Maximum detail keeps identity better and is several times slower — every reference token is re-read on every sampling step.")
                }
            }
            // ONE target for the whole section rather than three: the lists are
            // rows a few points tall, and which list a file belongs to is
            // already knowable from the file itself. `H3RefDrop` spends the
            // per-type caps and the combined budget the Add buttons follow.
            .mediaDropAnyKind(limit: H3RefLimits.remaining(perType: H3RefLimits.total,
                                                           current: 0,
                                                           totalAttached: refFilesAttached),
                              isTargeted: $isRefDropTargeted) { urls in
                let routed = H3RefDrop.route(urls, images: refImageURLs,
                                             videos: refVideoURLs, audios: refAudioURLs)
                refImageURLs = routed.images
                refVideoURLs = routed.videos
                refAudioURLs = routed.audios
            }
        }
    }

    /// One removable reference list. Same shape for all three types so a
    /// fourth cannot pick up different behaviour by accident.
    @ViewBuilder
    private func refList(title: String, limit: Int, urls: Binding<[URL]>,
                         addLabel: String, systemImage: String,
                         add: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.caption.weight(.medium)).foregroundStyle(.secondary)
                Spacer()
                Text("\(urls.wrappedValue.count)/\(limit)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            ForEach(Array(urls.wrappedValue.enumerated()), id: \.element) { idx, url in
                HStack(spacing: 8) {
                    Text("\(idx + 1).").font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button {
                        urls.wrappedValue.removeAll { $0 == url }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Remove this reference")
                }
            }
            // Room is the tighter of this list's cap and the COMBINED 12-file
            // budget, so a full set hides Add on an empty list too — with the
            // note under the section saying why.
            if refRemaining(perType: limit, current: urls.wrappedValue.count) > 0 {
                Button(action: add) {
                    Label(addLabel, systemImage: systemImage)
                        .font(.caption)
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    /// Files attached across all three reference lists.
    private var refFilesAttached: Int {
        refImageURLs.count + refVideoURLs.count + refAudioURLs.count
    }

    private func refRemaining(perType: Int, current: Int) -> Int {
        H3RefLimits.remaining(perType: perType, current: current, totalAttached: refFilesAttached)
    }

    // ── Speech & sound (audio-to-video) ──
    // Attach real speech/audio and the model generates the video AGAINST it:
    @ViewBuilder
    private var speechSection: some View {
        // A backend that GENERATES its soundtrack takes no audio input, so the
        // whole section is hidden rather than offered and refused. Declared by
        // the preset, never inferred from the model id.
        if !model.supportsAudioInput {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Sound").font(.subheadline.weight(.semibold))
                    Spacer()
                    Text(model.generatesAudio ? "generated with the video" : "not supported")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if model.generatesAudio {
                    Text("This model writes its own soundtrack. Describe it in the prompt after \"overall_soundscape:\" (and \"non_diegetic_music:\" for score).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        } else {
            audioInputSection
        }
    }

    private var audioInputSection: some View {
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
            intSliderRow("Steps", value: $steps, range: effectiveStepsRange,
                         help: "Denoising steps. More = more detail and smoother motion, but slower.")
            Text(turboEngaged ? "4 steps is sharp on this adapter and is the floor; more steps still help a little. If the picture shows over-sharp grain, drop the LoRA scale to 0.8-0.95; if it ghosts, raise it to 1.05-1.2." : model.stepsHelp)
                .font(.caption2).foregroundStyle(.secondary)

            // CFG is honored in every LTX pipeline mode, but a CFG-DISTILLED
            // backend has no guidance pass to scale — showing the slider there
            // would be a dead control (the Mage-Flow class).
            if model.supportsCFG {
                sliderRow("CFG scale", value: $cfgScale, range: 1...10, step: 0.5,
                          help: "Classifier-free guidance strength. LTX-2 default: 3.0; 1.0 = off (fastest).")
                Text("Guidance strength — how closely the video follows your prompt. 1.0 = off: fastest and most natural-looking. Higher sticks to the prompt more strictly but is slower and can look over-saturated. LTX default is 3.0.")
                    .font(.caption2).foregroundStyle(.secondary)

                // STG was sent on every LTX request from the day the wire was
                // fixed, with nothing to set it — so it sat at whatever was in
                // storage. A field on the wire with no control is worse than an
                // absent one: the request looks right.
                sliderRow("STG scale", value: $stgScale, range: 0...4, step: 0.5,
                          help: "Spatio-temporal guidance. 0 = off (the default). Steadies motion and structure at the cost of speed.")
                Text("Steadies motion and shape by re-running part of the model with its attention perturbed. 0 = off, which is the default. Around 1.0 helps wobbly motion; higher costs time and can flatten detail.")
                    .font(.caption2).foregroundStyle(.secondary)

                // Audio guidance belongs to the a2vid guider, so it only shows
                // with a clip attached — otherwise it is a knob on something
                // that never runs.
                if model.supportsAudioInput, audioURL != nil {
                    sliderRow("Audio guidance", value: $cfgAudioScale, range: 1...12, step: 0.5,
                              help: "How closely the picture follows the attached soundtrack. LTX default: 7.0.")
                    Text("How hard the video is pushed to match your clip — lip sync, timing, performance. 7.0 is the LTX default. Lower drifts from the audio; higher locks to it and can look stiff.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            // Two-stage refine. One-stage has no second pass, so the control is
            // hidden rather than shown doing nothing.
            if model.supportsPipelineModes, mode != .oneStage {
                Picker("Refine steps", selection: $stage2Steps) {
                    Text("Auto").tag(0)
                    ForEach(1...6, id: \.self) { Text("\($0)").tag($0) }
                }
                .pickerStyle(.menu)
                .font(.caption)
                .help("Steps in the second, full-resolution pass. Auto uses the reference schedule (3).")
                Text("The second pass sharpens the upscaled frames. Auto is the reference schedule. More steps clean up detail and cost time; fewer are faster and softer.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            // Chained windows. Already wired end to end — this is the control
            // that never existed, which is why long clips were unreachable.
            if model.supportsChainedWindows {
                Stepper(value: $chainWindows, in: 1...8) {
                    Text("Chained windows: \(chainWindows)").font(.caption)
                }
                .help("Join several generations end to end, each starting from the last frame of the one before.")
                Text(chainWindows > 1
                     ? "\(chainWindows) windows joined end to end — about \(numFrames * chainWindows) frames, and roughly \(chainWindows)x the time of a single window."
                     : "Joins several generations end to end for a longer clip. Each window costs another full generation.")
                    .font(.caption2).foregroundStyle(.secondary)
            }

            HStack {
                SeedField(label: "Seed", placeholder: "42", range: 0...Int.max, value: $seed,
                          help: "Same seed + same settings reproduces the clip. Paste one to rerun someone else's.")
                Spacer()
            }
            if model.supportsTurbo {
                Toggle("Turbo (distilled 4-step sampling)", isOn: $turbo)
                    .font(.caption)
                    .help("Runs the Turbo distillation LoRA: 4 steps instead of 30, about twice as fast end to end. Slightly softer detail and harder light than a full render. The adapter ships with the model; packs downloaded before it existed fetch it once, on the first run with this on.")
                    .onChange(of: turbo) { _, on in
                        guard !hydrating else { return }
                        // Snap steps into the mode's own range: 4 is what the
                        // adapter is distilled for, 30 the full render default.
                        steps = on ? 4 : min(model.stepsRange.upperBound, max(model.stepsRange.lowerBound, 30))
                        persist()
                        // Fetch the adapter the moment it is asked for rather
                        // than at Generate: 744 MB discovered 30 seconds into
                        // a job reads as a hang, and this way the Downloads
                        // pane shows it while the user finishes their prompt.
                        // The off-flip CANCELS an in-flight fetch — without
                        // that, a briefly-ticked box still downloads 744 MB
                        // in the background with nothing on screen saying so.
                        if on, turboFetchDecision == .fetch {
                            downloads.startTurboLora(repoId: model.repo)
                        } else if !on {
                            downloads.cancelTurboLora(repoId: model.repo)
                        }
                    }
                if turboFetchDecision == .fetch {
                    Text("Turbo needs a \(TurboLoraFetch.approxMB) MB adapter this pack predates — it downloads once, and Generate waits for it.")
                        .font(.caption2).foregroundStyle(.secondary)
                } else if turboFetchDecision == .unavailableRemotely {
                    Text("Turbo runs on the Mac hosting this model; it needs the adapter in ITS copy of the pack.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            if model.supportsDiffusionDecoder {
                Toggle("Diffusion decoder (sharper, slower)", isOn: $diffusionDecoder)
                    .font(.caption)
                    .help("Off (default): the plain convolutional decoder. On: LTX's own diffusion decoder — the one their published clips use. It denoises the frames instead of interpolating them, so fine texture and edges come out sharper. The decode itself takes about 21 s for a 97-frame clip at 768x512; measured end to end against the plain decoder in the same session the difference was inside run-to-run variance.")
            }
            if model.supportsFastRecipe {
                Toggle("Max quality (slower)", isOn: $bestQuality)
                    .font(.caption)
                    .help("Off (default): the fast recipe — step caching + attention reuse, about 2.8x faster at 768p. On: every denoising step is fully computed; marginally better detail for final renders.")
                    // Under turbo the recipe is already off server-side; a
                    // toggle that could not change anything is a dead control.
                    .disabled(turboEngaged)
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
            residencyRow

            if model.supportsLoRA { loraSection }
        }
    }

    @ViewBuilder
    private var loraSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Divider()
            HStack {
                Text("Style LoRAs").font(.caption.weight(.semibold))
                Spacer()
                Button {
                    chooseLora()
                } label: {
                    Image(systemName: "plus.circle")
                }
                .buttonStyle(.borderless)
                .disabled(loras.count >= maxLoras)
                .help(loras.count >= maxLoras ? "Maximum \(maxLoras) LoRAs" : "Add another LoRA")
            }
            if loras.isEmpty {
                Button {
                    chooseLora()
                } label: {
                    Label("Choose .safetensors…", systemImage: "paintpalette")
                        .font(.caption)
                }
                Text("Apply one or more LoRA adapters to the video model for a custom style. Several can stack at once.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(loras.enumerated()), id: \.element.id) { index, lora in
                    HStack(spacing: 8) {
                        Image(systemName: "paintpalette")
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(URL(fileURLWithPath: lora.path).lastPathComponent)
                                .font(.caption)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .help(lora.path)
                            Stepper(value: $loras[index].scale, in: 0...2, step: 0.05) {
                                Text("scale \(String(format: "%.2f", lora.scale))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .onChange(of: loras[index].scale) { _, _ in guard !hydrating else { return }; persist() }
                        }
                        Spacer()
                        Button {
                            loras.remove(at: index)
                            persist()
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .help("Remove this LoRA")
                    }
                    .padding(6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
                }
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

    /// Max simultaneously-attached LoRAs — mirrors the server's `lora.MAX_LORAS`.
    private let maxLoras = 8

    private func chooseLora() {
        guard loras.count < maxLoras else { return }
        let panel = NSOpenPanel()
        if let st = UTType(filenameExtension: "safetensors") {
            panel.allowedContentTypes = [st]
        }
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        if AppActivation.runModal(panel) == .OK {
            for url in panel.urls.prefix(maxLoras - loras.count) {
                loras.append(LoraAdapter(path: url.path))
            }
            persist()
        }
    }

    /// Append picked files to a reference list, never past its cap — the
    /// server rejects an over-cap set by name, and a picker that lets you build
    /// one only to fail at generate time is a worse version of the same 400.
    /// `room` is how many more files may be added — the tighter of this list's
    /// own cap and what is left of the combined 12-file budget, not the list's
    /// absolute limit. A multi-select that overshoots is truncated here rather
    /// than earning a 400 at generate time.
    private func chooseRefFiles(types: [UTType], limit room: Int, into urls: Binding<[URL]>) {
        guard room > 0 else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        guard AppActivation.runModal(panel) == .OK else { return }
        var added = 0
        for url in panel.urls where added < room {
            if !urls.wrappedValue.contains(url) {
                urls.wrappedValue.append(url)
                added += 1
            }
        }
    }

    private func chooseKeyframeImage(into slot: Binding<URL?>) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            slot.wrappedValue = url
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
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (lanModel == nil && !downloads.bundleReady(model.bundle)) || !customSizeValid)
                }
            }
            if !service.isRunning, let est = timeEstimate {
                Text(est)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Estimated from measured runs and this Mac's GPU. Actual time varies with what else is using the GPU.")
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
                // The one bridge from the workshop to a conversation. It opens
                // a NEW chat and switches to it — see
                // `AppState.sendGeneratedMediaToNewChat`.
                Button {
                    appState.sendGeneratedMediaToNewChat(
                        path: path, prompt: prompt, kind: .video)
                } label: { Image(systemName: "bubble.left.and.text.bubble.right") }
                .buttonStyle(.borderless)
                .help("Send to Chat — opens a new conversation with this attached")
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
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        quality = s.quality
        resolution = s.resolvedResolution(for: model)
        customWidthText = String(s.customWidth)
        customHeightText = String(s.customHeight)
        numFrames = s.numFrames
        fps = s.fps
        mode = s.mode
        // Turbo restores BEFORE the steps clamp: its range reaches below the
        // preset's floor, and clamping first would bounce a saved 8 up to 16.
        turbo = s.turbo && model.supportsTurbo
        // Clamp into the slider ranges — a value persisted by the old wider
        // steppers (Steps unbounded, CFG 0…20) would otherwise sit off-scale.
        steps = min(effectiveStepsRange.upperBound, max(effectiveStepsRange.lowerBound, s.steps))
        cfgScale = min(10, max(1, s.cfgScale))
        stgScale = s.stgScale
        // Clamped like the sliders above: a value saved before a range moved
        // would otherwise sit off-scale, and 0 stays Auto.
        stage2Steps = min(6, max(0, s.stage2Steps))
        cfgAudioScale = min(12, max(1, s.cfgAudioScale))
        chainWindows = model.supportsChainedWindows ? min(8, max(1, s.chainWindows)) : 1
        seed = s.seed
        keepResident = s.keepResident
        bestQuality = s.bestQuality
        diffusionDecoder = s.diffusionDecoder
        promptHeight = PromptEditorHeight.clamp(s.promptHeight)
        loras = s.loras
        // A LoRA file may have moved since last session — drop stale entries.
        loras.removeAll { !FileManager.default.fileExists(atPath: $0.path) }
        clampFramesToRAM()
    }

    private func persist() {
        var s = VideoGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.quality = quality
        s.resolutionId = resolution.id
        s.customWidth = Int(customWidthText) ?? VideoGenSettings().customWidth
        s.customHeight = Int(customHeightText) ?? VideoGenSettings().customHeight
        s.numFrames = numFrames
        s.fps = fps
        s.mode = mode
        s.steps = steps
        s.cfgScale = cfgScale
        s.stgScale = stgScale
        s.stage2Steps = stage2Steps
        s.cfgAudioScale = cfgAudioScale
        s.chainWindows = chainWindows
        s.seed = seed
        s.keepResident = keepResident
        s.bestQuality = bestQuality
        s.diffusionDecoder = diffusionDecoder
        s.turbo = turbo
        s.promptHeight = PromptEditorHeight.clamp(promptHeight)
        s.loras = loras
        s.save()
    }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        resolution = model.recommendedResolution(totalGB: RAMChecker.totalGB)
        fps = model.fps
        // A clip attached under LTX must not survive a switch to a backend
        // that takes no audio input: the section hides, so the user can't
        // clear it, the quality hint claims "audio-to-video", and an
        // unreadable file still hard-errors a generate it wouldn't reach.
        // (The first-frame image deliberately survives — every backend
        // supports keyframe conditioning.)
        if !model.supportsAudioInput {
            clearAudio()
            audioSource = .none
        }
        // The DiffVAE is the decoder LTX's own published clips use, and only
        // the 8-bit pack ships it — a pack chosen FOR quality. Following the
        // preset here (rather than defaulting the stored setting) also clears
        // it on a switch to a pack that cannot serve it, so the toggle's state
        // can never outlive the capability.
        diffusionDecoder = model.supportsDiffusionDecoder
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        let s = model.settings(quality)
        mode = s.mode
        // A quality tier describes a FULL render — its step counts are the
        // non-turbo schedule's — so picking one turns turbo off rather than
        // clamping the tier's 30 steps into turbo's 16-step ceiling.
        turbo = false
        // Clamp into the backend's range: a preset switch carrying a value the
        // new model's slider cannot show leaves the control off-scale.
        steps = min(model.stepsRange.upperBound, max(model.stepsRange.lowerBound, s.steps))
        cfgScale = s.cfgScale
        stgScale = s.stgScale
        // A tier does not describe chaining, but a preset switch can land on a
        // partition without it — collapse rather than carry a dead value.
        if !model.supportsChainedWindows { chainWindows = 1 }
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
        // The ladder is per-CANVAS now (the whole clip rides back as one
        // base64 blob), so a length saved at 768×512 must snap down when the
        // user moves to 1920×1088 — the slider only reads `numFrames`.
        guard let lo = availableFrameOptions.first, let hi = availableFrameOptions.last else { return }
        if numFrames > hi {
            numFrames = availableFrameOptions.last(where: { $0 <= hi }) ?? hi
        } else if numFrames < lo {
            // Stale persisted value below a raised floor (e.g. H3's 5→124) —
            // the slider can't self-correct since it only reads `numFrames`.
            numFrames = lo
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
            width: effectiveSize.width,
            height: effectiveSize.height,
            numFrames: numFrames,
            fps: fps,
            mode: mode,
            steps: steps,
            cfgScale: cfgScale,
            stgScale: stgScale,
            firstFrameImagePath: firstFrameImageURL?.path,
            lastFrameImagePath: lastFrameImageURL?.path,
            // Belt-and-braces with the requestBody gate: a clip must never
            // reach the transcode (whose failure is a hard error) on a
            // backend that generates its own soundtrack.
            audioPath: model.supportsAudioInput ? audioURL?.path : nil,
            keepResident: keepResident,
            bestQuality: bestQuality,
            // Belt-and-braces with the requestBody gate: the toggle's state
            // survives a preset switch, and only one pack ships the decoder.
            diffusionDecoder: model.supportsDiffusionDecoder && diffusionDecoder,
            lanModelId: lanModel,
            loras: loras,
            // Belt-and-braces with the requestBody gate (turbo state survives
            // preset switches, like the reference files below).
            turbo: turboEngaged,
            // Gated here too: the stepper's value survives a preset switch,
            // and only the fl2va partition has a keyframe row to chain.
            chainWindows: model.supportsChainedWindows ? chainWindows : 1,
            stage2Steps: stage2Steps,
            cfgAudioScale: cfgAudioScale,
            // Belt-and-braces with the requestBody gate, same as `audioPath`
            // above: reference files must never reach the reader (whose
            // failure is a hard error) on a pack that cannot use them.
            refImagePaths: model.supportsReferences ? refImageURLs.map(\.path) : [],
            refVideoPaths: model.supportsReferences ? refVideoURLs.map(\.path) : [],
            refAudioPaths: model.supportsReferences ? refAudioURLs.map(\.path) : [],
            refImageSize: refImageSize
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

        // Belt-and-braces with the toggle's own fetch: turbo can be persisted
        // ON from a previous session, or the fetch it started can still be in
        // flight. `startTurboLora` attaches to a running transfer rather than
        // starting a second one, so both paths converge on one download.
        if turboFetchDecision == .fetch {
            downloads.startTurboLora(repoId: model.repo) {
                service.generate(req, server: server)
            }
            return
        }

        service.generate(req, server: server)
    }

    /// Whether this pane's current selection needs the Turbo adapter fetched.
    /// The file lives in the pack, so a LAN model's is not ours to complete.
    private var turboFetchDecision: TurboLoraFetch.Decision {
        TurboLoraFetch.decide(
            turboRequested: turbo,
            backendSupportsTurbo: model.supportsTurbo,
            isRemote: lanModel != nil,
            fileOnDisk: TurboLoraFetch.isOnDisk(modelDir: ServerManager.resolveModelDir(repo: model.repo))
        )
    }

    private func showLogWindow() {
        let text = server.combinedGenLog(own: service.log)
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
// `AVPlayerViewRepresentable` moved to Views/ChatMediaAttachmentView.swift when
// the chat transcript grew video attachments. SwiftUI's own `VideoPlayer`
// fatal-aborts under state transitions on macOS 26.4, so there must be exactly
// ONE AVPlayerView wrapper — a second copy is a second thing to get wrong.
