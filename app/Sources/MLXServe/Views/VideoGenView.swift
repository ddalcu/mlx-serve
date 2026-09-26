import SwiftUI
import AppKit
import AVKit
import AVFoundation
import ImageIO
import UniformTypeIdentifiers

/// One row of the "Set by starting frame" menu: a canvas and what to call it.
private struct ClipSizeChoice: Identifiable {
    let canvas: AspectCanvas
    let name: String?
    var isSourceSize: Bool = false
    var id: String { canvas.id }
}

/// Video generation window — LTX-Video (2.3 / 2.5) and MiniMax-H3, run
/// natively by the mlx-serve server. Uses the same Quality / Resolution preset
/// shape as ImageGen, plus a Frames slider clamped to LTX's `8N+1` ladder and
/// the user's RAM budget. Which controls are offered is decided by the
/// preset's declared capabilities, never by its id — the two LTX releases
/// share every one of them.
struct VideoGenView: View {
    @EnvironmentObject var service: VideoGenService
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject var downloads: DownloadManager
    /// For "Send to Chat" — the hand-off opens a new conversation and switches
    /// the window to it (`AppState.sendGeneratedMediaToNewChat`).
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    /// The editor's caret or selection, for dropping a reference marker where
    /// the user is typing. nil until the editor has had focus.
    @State private var promptSelection: TextSelection? = nil
    @FocusState private var promptFocused: Bool
    /// Height of the prompt editor — dragged by `promptResizeHandle`, sticky.
    @State private var promptHeight: Double = PromptEditorHeight.defaultHeight
    /// The prompt header row in global space: the hover bubble sizes itself
    /// against its width and is kept inside it.
    @State private var promptRow: CGRect = .zero
    @State private var showAdvanced: Bool = false
    /// The media-input block starts open: on most models it holds the first
    /// frame, which is the thing people reach for straight after the prompt.
    @State private var showMediaInputs: Bool = true
    @State private var model: VideoModelPreset = .ltx23Q4
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var quality: QualityPreset = .good
    /// Pixel size of the picked first frame, read from the file's metadata
    /// when it arrives. The shape it gives is what the "Set by starting frame"
    /// menu offers canvases for.
    @State private var firstFrameSize: (width: Int, height: Int)? = nil
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
    /// Which way into the clip slot the user took — the well draws a different
    /// state for each. Persisted with the draft (`VideoAudioSource`).
    @State private var audioSource: VideoAudioSource = .none
    /// The attached clip: a picked file, or the TTS output under
    /// `~/.mlx-serve/generations/audio` — a real file either way, so its path
    /// rides the draft and comes back after a relaunch.
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
    /// Per-step latent previews on the SSE stream (issue #208).
    @State private var livePreview: Bool = false
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
    /// And for the audio-to-video clip slot.
    @State private var isAudioDropTargeted: Bool = false
    /// The References heading row in global space (quantised, like the grid's
    /// rect): the budget counter's bubble is kept inside it.
    @State private var refHeaderRow: CGRect = .zero
    /// Whether the media-inputs block is wide enough for the two keyframe
    /// wells to share a row. False until its first `onGeometryChange`, which
    /// stacks them for one frame — the narrow answer either way.
    @State private var keyframesSideBySide: Bool = false
    /// Whether the form is wide enough for the Quality tiers as segments.
    /// Measured on the section, never judged by `ViewThatFits`: a segmented
    /// picker accepts any width and squeezes, so it always "fits".
    @State private var qualityFitsSegments: Bool = true

    /// Set when `hydrate` dropped a reference whose file is gone: the tiles
    /// after it renumbered, so a prompt that names them now points elsewhere.
    /// Cleared by the first edit to the prompt or to the references — either
    /// means the user has looked.
    @State private var refsDroppedOnHydrate: Bool = false
    /// See `resolveSpeechPreset`.
    @State private var speechPreset: AudioModelPreset? = nil

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
            speechPreset = Self.resolveSpeechPreset()
            // Freshen the network-model list so LAN entries are current in
            // the picker (discovery lands seconds after the server boots).
            if server.status == .running { Task { await server.refreshModels() } }
        }
        // ONE observation of the whole blob (the Music pane's mechanism):
        // anything in `stickySnapshot` is sticky by construction.
        .onChange(of: stickySnapshot) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: prompt) { _, _ in guard !hydrating else { return }; refsDroppedOnHydrate = false }
        .onChange(of: refFilesAttached) { _, _ in guard !hydrating else { return }; refsDroppedOnHydrate = false }
        // Windows multiply the DELIVERED frames (`w*n - (w-1)` in one
        // response), so raising them shortens the ladder under a set length.
        .onChange(of: chainWindows) { _, _ in guard !hydrating else { return }; clampFramesToRAM() }
        // The two size fields are separate because they do more than persist.
        .onChange(of: customWidthText) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
        .onChange(of: customHeightText) { _, _ in guard !hydrating else { return }; clampFramesToRAM(); persist() }
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
        // The menu offers canvases for the picture's shape, so the shape has
        // to be read when the picture changes, and forgotten when it goes.
        .onChange(of: firstFrameImageURL) { _, url in
            firstFrameSize = url.flatMap { VideoGenView.pixelSize(of: $0) }
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
                // The model decides what the rest of the pane means — which
                // anchors exist, whether there are references or a soundtrack
                // to attach, what Advanced holds — so it is read first.
                VStack(alignment: .leading, spacing: 14) {
                    modelSection
                    promptSection
                    mediaInputsSection
                    clipSizeSection
                    qualitySection
                    framesSection
                    advancedSection
                    // Generate stands apart from the settings it acts on.
                    actionRow.padding(.top, 14)
                }
                // Full-width, leading-aligned frame OUTSIDE the padding: a
                // child that will not compress otherwise makes the stack
                // oversized, and the ScrollView centres the overflow.
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
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
            Text(L10n.text(ramWarningMessage))
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Prompt").font(.app(.subheadline).weight(.semibold))
                Spacer()
                if let hint = promptHint { promptWarning(hint) }
                templatesMenu
            }
            // The header's hover bubble reaches over the editor below it, and
            // zIndex only orders SIBLINGS: without this the row is painted
            // first and the editor lands on top of what it opened.
            .zIndex(1)
            // Quantised: only `minX` and `width` are read, and an exact rect
            // changes on every frame of a drag.
            .onGeometryChange(for: CGRect.self) { proxy in
                let r = proxy.frame(in: .global)
                return CGRect(x: (r.minX / 8).rounded() * 8, y: 0,
                              width: (r.width / 8).rounded() * 8, height: 0)
            } action: { promptRow = $0 }
            ZStack(alignment: .topLeading) {
                // Selection and focus are read by the reference tiles: a click
                // on one drops its marker where the caret is, or on the end
                // when the editor is not the one being typed into.
                TextEditor(text: $prompt, selection: $promptSelection)
                    .focused($promptFocused)
                    .font(.app(.body))
                    .frame(height: promptHeight)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                    )
                if prompt.isEmpty {
                    Text(L10n.text(H3PromptExamples.placeholder(for: model.promptFormat)))
                        .font(.app(.body))
                        .foregroundStyle(.secondary.opacity(0.6))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 8)
                        .allowsHitTesting(false)
                }
            }
            promptResizeHandle
        }
    }

    /// Four fifths of the column, floored so a narrow pane still gets a
    /// readable paragraph, capped so a wide one does not get one long line,
    /// and never wider than the column it has to stay inside.
    private var bubbleWidth: CGFloat {
        guard promptRow.width > 0 else { return 320 }
        return min(min(max(promptRow.width * 0.8, 320), 640), promptRow.width)
    }

    /// One badge for every format complaint; the sentence that fired floats
    /// over the pointer.
    private func promptWarning(_ hint: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: "info.triangle.fill")
            Text("Prompt not optimal. Look at templates or prompt tips.")
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .font(.app(.caption))
        .foregroundStyle(.orange)
        .hoverReveal(placement: .pointerClamped(width: bubbleWidth, container: promptRow)) {
            Self.hoverBubble(L10n.text(hint))
        }
    }

    /// The pane's floating-bubble surface, in one place: the prompt's format
    /// advice and a reference tile's filename are the same kind of thing said
    /// over the pointer, and two copies would drift. Static, so the tile —
    /// its own view — can draw it too.
    static func hoverBubble(_ text: String) -> some View {
        hoverBubble(text) { EmptyView() }
    }

    /// A picture's size fitted into a square of `side`, never upscaled: a
    /// small reference shown larger than it is would only be blurry. nil for
    /// a size that is not a size.
    static func previewSize(for size: CGSize, within side: CGFloat) -> CGSize? {
        guard size.width > 0, size.height > 0, side > 0 else { return nil }
        let scale = min(side / size.width, side / size.height, 1)
        return CGSize(width: size.width * scale, height: size.height * scale)
    }

    /// The same bubble with something ABOVE the sentence — a reference tile
    /// puts the whole picture there, uncropped, since the tile shows a square
    /// cut from it.
    static func hoverBubble<Above: View>(_ text: String,
                                         @ViewBuilder above: () -> Above) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            above()
            Text(text)
                .font(.app(.caption))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(8)
        .background(Color(nsColor: .textBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .shadow(color: .black.opacity(0.18), radius: 3, y: 1)
    }

    /// H3's format is a multi-section document, so 110pt is a keyhole. The
    /// height sticks (clamped on the way in and out, so a value dragged on a
    /// taller window can't come back unusable).
    private var promptResizeHandle: some View {
        EditorResizeHandle(height: $promptHeight, onCommit: persist,
                           help: "Drag to resize the prompt box.")
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

    /// The prompt's starting points, and the place to read about writing one.
    /// The tips link lives IN the menu rather than beside it: it is the same
    /// kind of thing as the templates (help with the prompt).
    private var templatesMenu: some View {
        let title = H3PromptExamples.templatesTitle(for: model.promptFormat)
        return Menu {
            Section(L10n.text(title)) {
                ForEach(examplePrompts, id: \.title) { ex in
                    Button(L10n.text(ex.title)) { prompt = ex.body }
                }
            }
            Divider()
            // No tint: AppKit draws menu item titles in the system colour and
            // a `foregroundStyle` here is a modifier that does nothing.
            Link(destination: H3PromptExamples.tipsURL(for: model.promptFormat)) {
                Label("Prompt tips…", systemImage: "arrow.up.forward.square")
            }
        } label: {
            HStack(spacing: 5) {
                Text("Templates")
                Image(systemName: "chevron.down")
            }
            .modifier(PaneChip())
        }
        .modifier(PaneChipMenu())
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`. The transfer
    /// bar, the residency line and the residency SETTING all belong to the
    /// model, not to the clip, so they sit with it.
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            modelChooser
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
        }
    }

    /// Residency rides the switcher's row: it is a property of the model, and
    /// the only thing about it the pane still has to say once it is picked.
    private var keepResidentToggle: AnyView {
        AnyView(
            // No `fixedSize()`: the row it rides also carries the switcher and
            // the residency line, and a label that refuses to compress makes
            // the whole column wider than the pane can offer.
            Toggle(isOn: $keepResident) {
                // One line that ellipsises, like the model name above it:
                // wrapping to two lines makes the row taller than the switcher
                // beside it. The full sentence is in the tooltip.
                Text("Keep model loaded after generating")
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
                .font(.app(.caption))
                .controlSize(.small)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        )
    }

    private var modelChooser: some View {
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
            onDownloadFinished: {
                appState.refreshModels()
                speechPreset = Self.resolveSpeechPreset()
            },
            persist: persist,
            status: AnyView(residencyRow),
            accessory: keepResidentToggle)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults(); persist() }
    }

    /// What the switcher shows. `custom` exists only while it is SELECTED, so
    /// it is never something to pick — there is nothing to pick it back from.
    private enum QualitySelection: Hashable {
        case preset(QualityPreset)
        case custom
    }

    /// The tier the live values mean, or nil for Custom. `quality` is the last
    /// tier the user explicitly picked and only settles an ambiguity — see
    /// `VideoQualityMatch`.
    private var matchedQuality: QualityPreset? {
        VideoQualityMatch.match(
            VideoQualityMatch.Resolved(mode: mode, steps: steps, cfgScale: cfgScale,
                                       stgScale: stgScale, numFrames: numFrames,
                                       turbo: turboEngaged),
            model: model, width: effectiveSize.width, height: effectiveSize.height,
            chainWindows: chainWindows, preferring: quality)
    }

    /// What five segments need: the four tier names plus Custom, at the
    /// segmented control's own per-segment padding. "Super Quality" is the
    /// wide one, and shortening it is not on the table — the tiers are named
    /// the same in every Create pane.
    private static let qualitySegmentsMinWidth: CGFloat = 380

    /// Reads the DERIVED tier and writes by applying one. Custom is unwritable
    /// by construction, so the guard is a formality rather than a policy.
    private var qualitySelection: Binding<QualitySelection> {
        Binding(
            get: { matchedQuality.map(QualitySelection.preset) ?? .custom },
            set: { sel in
                guard case .preset(let q) = sel else { return }
                quality = q
                applyQualityDefaults()
                persist()
            })
    }

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Quality").font(.app(.subheadline).weight(.semibold))
            // Measured, not `ViewThatFits`: see `qualityFitsSegments`. Five
            // segments degrade to a menu rather than shortening the tier names
            // this pane shares with every other Create pane.
            qualityPicker(segmented: qualityFitsSegments)
            Text(qualityHint)
                .font(.app(.caption))
                .foregroundStyle(.secondary)
        }
        // Measure the SECTION at the column's width, never the picker: the
        // menu variant is `fixedSize` and would never re-fit.
        .frame(maxWidth: .infinity, alignment: .leading)
        .onGeometryChange(for: Bool.self) { $0.size.width >= Self.qualitySegmentsMinWidth }
            action: { qualityFitsSegments = $0 }
    }

    @ViewBuilder
    private func qualityPicker(segmented: Bool) -> some View {
        let picker = Picker("", selection: qualitySelection) {
            ForEach(QualityPreset.allCases) { q in
                Text(L10n.text(q.label)).tag(QualitySelection.preset(q))
            }
            if matchedQuality == nil {
                Text("Custom").tag(QualitySelection.custom)
            }
        }
        .labelsHidden()
        if segmented {
            picker.pickerStyle(.segmented)
        } else {
            picker.pickerStyle(.menu).fixedSize()
        }
    }

    /// The pipeline the request will RUN: audio-to-video is two-stage only, so
    /// `requestBody` upgrades a one-stage request that carries a clip. Controls
    /// DESCRIBING the request read this; the stored `mode` stays the user's
    /// (the switcher reads that). Capability-gated so a stale clip cannot make
    /// H3 claim it.
    private var effectiveMode: VideoPipelineMode {
        (model.supportsAudioInput && audioURL != nil && mode == .oneStage) ? .twoStage : mode
    }
    private var modeUpgradedForAudio: Bool { effectiveMode != mode }

    /// The Mode menu: reads the effective mode, writes the stored one. While a
    /// clip forces two stages the menu is disabled, so the setter is only ever
    /// reached when the two agree.
    private var modeSelection: Binding<VideoPipelineMode> {
        Binding(get: { effectiveMode }, set: { mode = $0 })
    }

    /// What the three guidance sliders read while a clip forces two stages:
    /// `requestBody` omits them then, and the server applies its own set.
    private var guidanceLockedReadout: String? {
        modeUpgradedForAudio ? L10n.text("server default") : nil
    }

    private var modeHint: String {
        if modeUpgradedForAudio {
            return "Audio-to-video runs on two stages, so the clip sets this while it is attached: the first stage denoises at half the clip size and the second refines at full size, which is why the clip size has to be a multiple of 64. Guidance (CFG, STG, audio) follows the server's two-stage defaults meanwhile, because the tier's one-stage values would run the first stage unguided. Refine steps is that second stage: Auto is the reference schedule. Remove the clip to choose again."
        }
        return effectiveMode == .oneStage
            ? "One stage is the fastest and has no refine pass, so the slider is off. The two-stage modes denoise at half the clip size and refine at full size — that is where their detail comes from, and why they want a larger canvas and a clip size in multiples of 64."
            : "Denoises at half the clip size and refines at full size — that is where the detail comes from, and why this mode wants a larger canvas and a clip size in multiples of 64. Refine steps is that second stage: Auto is the reference schedule, more steps clean up detail and cost time, fewer are faster and softer."
    }

    private var qualityHint: String {
        let durationSec = Double(numFrames) / Double(fps)
        let mode = L10n.text(modeLabel(effectiveMode))
        let label = modeUpgradedForAudio ? L10n.format("%@ (audio-to-video)", mode) : mode
        // Turbo replaces the schedule the step count belongs to, so a bare
        // "4 steps" would read as a slow render nobody asked for.
        let turboNote = turboEngaged ? L10n.text(" (Turbo)") : ""
        return L10n.format("%@, %lld steps%@, %lld frames (~%.1fs)",
                           label, Int64(steps), turboNote, Int64(numFrames), durationSec)
    }

    private func modeLabel(_ m: VideoPipelineMode) -> String {
        switch m {
        case .oneStage:   return "1-stage"
        case .twoStage:   return "2-stage"
        case .twoStageHQ: return "2-stage HQ"
        }
    }

    /// A reference tile was clicked: its marker goes where the caret is, or on
    /// the end when the editor is not the one being typed into. The caret
    /// follows the marker so typing on continues the sentence. Focus is the
    /// gate, not the selection: the editor keeps a selection after focus has
    /// moved on, and dropping a marker into last week's caret position is
    /// not what a click on a tile means.
    private func insertMarker(_ marker: String) {
        let result: PromptMarkerInsert.Result
        if promptFocused, let selection = promptSelection,
           case .selection(let range) = selection.indices {
            let lo = prompt.distance(from: prompt.startIndex, to: range.lowerBound)
            let hi = prompt.distance(from: prompt.startIndex, to: range.upperBound)
            result = PromptMarkerInsert.insert(marker, into: prompt, replacing: lo..<hi)
        } else {
            result = PromptMarkerInsert.append(marker, to: prompt)
        }
        prompt = result.text
        let caret = result.text.index(result.text.startIndex,
                                      offsetBy: min(result.cursor, result.text.count))
        promptSelection = TextSelection(insertionPoint: caret)
    }

    /// The line under the size fields. The grid's own correction note names the
    /// step but not WHY it is that step; here the pane knows the mode the
    /// request will run and says so, since the step follows from it.
    private func clipSizeHint(_ verdict: CustomResolution) -> String? {
        switch verdict {
        case .ok:
            return nil
        case let .corrected(w, h, _):
            let step = model.resolutionGrid(twoStage: effectiveMode != .oneStage).alignment
            let why = model.supportsPipelineModes
                ? L10n.format("model will run in %@ mode and sample in %lldpx steps",
                              L10n.text(modeLabel(effectiveMode)), Int64(step))
                : L10n.format("model samples in %lldpx steps", Int64(step))
            return L10n.format("Will be rounded to %lld × %lld. With current settings, %@.",
                               Int64(w), Int64(h), why)
        case let .invalid(message):
            return L10n.text(message)
        }
    }

    /// The clip's canvas. The two fields are the ONE source of truth for what
    /// the request carries; the two menus only write into them. The server
    /// REFUSES an off-grid video canvas outright (unlike the image path, which
    /// quietly rewrites it), so the verdict under the fields is the difference
    /// between a hint and a failed generation.
    private var clipSizeSection: some View {
        let verdict = customResolutionVerdict
        return VStack(alignment: .leading, spacing: 6) {
            // Bottom, not centre: the fields carry a heading above them, and
            // centring the row puts the button halfway up that heading.
            HStack(alignment: .bottom, spacing: 8) {
                clipSizeFields
                Spacer(minLength: 8)
                presetsMenu
            }
            if let hint = clipSizeHint(verdict) {
                Label(hint, systemImage: verdict.isValid ? "wand.and.stars" : "exclamationmark.triangle")
                    .font(.app(.caption2))
                    .foregroundStyle(verdict.isValid ? Color.secondary : Color.orange)
            }
            // A two-stage tier denoises at HALF this canvas and upscales, so on
            // a small canvas "Quality" is softer than the one-stage tiers.
            if effectiveMode != .oneStage,
               let note = model.twoStageCanvasNote(width: effectiveSize.width, height: effectiveSize.height) {
                Text(L10n.text(note)).font(.app(.caption2)).foregroundStyle(.orange)
            }
        }
    }

    private var clipSizeFields: some View {
        HStack(alignment: .bottom, spacing: 8) {
            labelledSizeField("Clip width", text: $customWidthText)
            // Centred on the fields, not on the pair of labels above them.
            Image(systemName: "multiply")
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .frame(height: 24)
            labelledSizeField("Clip height", text: $customHeightText)
        }
    }

    private func labelledSizeField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            // A section heading like First frame and Last frame. `fixedSize`
            // because a squeezed HStack proposes less than its widest child
            // and the TEXT is what gives first: two words on two lines.
            Text(L10n.text(title))
                .font(.app(.subheadline).weight(.semibold))
                .fixedSize()
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: 80)
        }
    }

    /// Canvases matching the starting frame's shape. A submenu of Presets
    /// rather than a button of its own: it answers the same question, and one
    /// control leaves the row room it does not have to fight for.
    @ViewBuilder
    private var startingFrameMenu: some View {
        let canvases = startingFrameCanvases
        if firstFrameImageURL == nil {
            // Disabled as a plain ITEM, not as a disabled submenu: a submenu
            // still opens on hover, and an empty one that opens reads as a
            // bug rather than as "pick a picture first".
            Button("Set by starting frame…") {}
                .disabled(true)
        } else {
            Menu {
                if canvases.isEmpty {
                    // Two rows, because `NSMenu` renders a title on one line
                    // and drops the newline.
                    Button("Selected first frame's picture does not fit this model.") {}
                        .disabled(true)
                    Button("Consider its cropping or adding a letterbox.") {}
                        .disabled(true)
                } else {
                    Section("Matching \(startingFrameRatio ?? "the starting frame")") {
                        // The source's own size is a different kind of answer
                        // from the spread below it: it does not rescale.
                        ForEach(canvases.filter(\.isSourceSize)) { choice in
                            Button(choiceLabel(choice)) {
                                setClipSize(width: choice.canvas.width, height: choice.canvas.height)
                            }
                        }
                        if canvases.contains(where: \.isSourceSize) { Divider() }
                        ForEach(canvases.filter { !$0.isSourceSize }) { choice in
                            Button(choiceLabel(choice)) {
                                setClipSize(width: choice.canvas.width, height: choice.canvas.height)
                            }
                        }
                    }
                }
            } label: {
                Text("Set by starting frame…")
            }
        }
    }

    /// The model's own curated sizes, grouped by orientation and largest first.
    private var presetsMenu: some View {
        Menu {
            ForEach([ResolutionOption.Orientation.landscape, .square, .portrait], id: \.self) { o in
                let rows = model.resolutions
                    .filter { $0.orientation == o }
                    .sorted { $0.width * $0.height > $1.width * $1.height }
                if !rows.isEmpty {
                    Section(orientationName(o)) {
                        ForEach(rows) { r in
                            Button(presetLabel(r)) { setClipSize(width: r.width, height: r.height) }
                        }
                    }
                }
            }
            Divider()
            startingFrameMenu
        } label: {
            clipMenuLabel("Presets")
        }
        .modifier(PaneChipMenu())
        .help("Sizes this model ships with, and sizes that match the starting frame.")
    }

    private func clipMenuLabel(_ title: String) -> some View {
        HStack(spacing: 5) {
            Text(title)
            Image(systemName: "chevron.down")
        }
        // Body, not caption: this sits beside the size fields rather than
        // above a text box. The height is the fields' own, so the two line up
        // instead of the chip hugging its text a few points shorter.
        .font(.app(.body))
        .modifier(PaneChip(height: 24))
    }

    private func orientationName(_ o: ResolutionOption.Orientation) -> String {
        switch o {
        case .landscape: return "Landscape"
        case .square:    return "Square"
        case .portrait:  return "Portrait"
        }
    }

    private func presetLabel(_ r: ResolutionOption) -> String {
        var out = "\(r.width) × \(r.height)"
        if let ratio = r.ratio { out += " (\(ratio))" }
        if let note = r.note { out += " - \(note)" }
        return out
    }

    /// No ratio per row: every row has the same one, and the section heading
    /// above them already says which.
    private func choiceLabel(_ choice: ClipSizeChoice) -> String {
        var out = "\(choice.canvas.width) × \(choice.canvas.height)"
        if let name = choice.name { out += " - \(name)" }
        return out
    }

    private var startingFrameRatio: String? {
        guard let size = firstFrameSize else { return nil }
        return AspectCanvases.ratioLabel(width: size.width, height: size.height)
    }

    /// The source's own size first (the one option that does not rescale the
    /// picture at all), then the spread named by size. Recomputed with the
    /// tier, because the grid tightens to /64 on the two-stage pipelines, and
    /// with the picture, because it is the picture's shape being matched.
    private var startingFrameCanvases: [ClipSizeChoice] {
        guard let size = firstFrameSize else { return [] }
        let grid = model.resolutionGrid(twoStage: effectiveMode != .oneStage)
        let spread = AspectCanvases.options(sourceWidth: size.width, sourceHeight: size.height, grid: grid)
        // Five names for five sizes; fewer candidates take the ends and the
        // middle, because "Large" among two is not information.
        let names: [String?] = {
            switch spread.count {
            case 5:  return ["largest", "large", "medium", "small", "smallest"]
            case 4:  return ["largest", "large", "small", "smallest"]
            case 3:  return ["largest", "medium", "smallest"]
            case 2:  return ["largest", "smallest"]
            default: return [nil]
            }
        }()
        var out = spread.enumerated().map { i, c in
            ClipSizeChoice(canvas: c, name: i < names.count ? names[i] : nil)
        }
        if let own = AspectCanvases.sourceSize(sourceWidth: size.width, sourceHeight: size.height, grid: grid) {
            out.removeAll { $0.canvas == own }
            out.insert(ClipSizeChoice(canvas: own, name: "source size", isSourceSize: true), at: 0)
        }
        return out
    }

    /// The file's pixel size from its metadata — ImageIO reads the header
    /// without decoding the picture, which a 4000px photo would make a
    /// noticeable pause.
    private static func pixelSize(of url: URL) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              w > 0, h > 0 else { return nil }
        return (w, h)
    }

    /// Written into the fields, over a focused one too: the user picked a size
    /// from a menu, so the box has to show it.
    private func setClipSize(width: Int, height: Int) {
        customWidthText = String(width)
        customHeightText = String(height)
        clampFramesToRAM()
        persist()
    }

    /// A two-stage pipeline denoises at HALF the canvas, so the server tightens
    /// its refusal to /64 there. The grid follows the mode the REQUEST will
    /// carry (`effectiveMode`): a one-stage tier with a clip runs two-stage.
    private var customResolutionVerdict: CustomResolution {
        model.resolutionGrid(twoStage: effectiveMode != .oneStage)
            .resolve(width: Int(customWidthText) ?? 0, height: Int(customHeightText) ?? 0)
    }

    private var customSizeValid: Bool { customResolutionVerdict.isValid }

    /// The length x windows x canvas combination the request would carry.
    private var payloadFits: Bool {
        model.framePayloadFits(width: effectiveSize.width, height: effectiveSize.height,
                               numFrames: numFrames,
                               chainWindows: model.supportsChainedWindows ? chainWindows : 1)
    }

    /// The canvas the request carries. The FIELDS are the source of truth; an
    /// invalid entry falls back to the model's default only for the hints and
    /// estimates below it — Generate is disabled meanwhile.
    private var effectiveSize: (width: Int, height: Int) {
        customResolutionVerdict.size
            ?? (model.defaultResolution.width, model.defaultResolution.height)
    }

    private var framesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Frames").font(.app(.subheadline).weight(.semibold))
                Spacer()
                Text(L10n.format("%lld frames · ~%.1fs", Int64(numFrames), Double(numFrames) / Double(fps)))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            // Snap through LTX's valid `8N+1` frame ladder by index, so the
            // slider can only land on generatable lengths (9, 17, 25, … maxFrames).
            frameSlider
            if let warn = frameRAMWarning {
                Text(L10n.text(warn)).font(.app(.caption2)).foregroundStyle(.orange)
            }
            if let advice = model.framesAdvisory(numFrames) {
                Text(L10n.text(advice)).font(.app(.caption2)).foregroundStyle(.orange)
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
        .help(L10n.format("Clip length. LTX only generates %lld–%lld frames on its 8N+1 ladder; the slider snaps to valid counts.",
                          Int64(opts.first ?? 9), Int64(opts.last ?? 193)))
    }

    /// Always show every option up to the model's hard cap. The user can
    /// pick longer than RAM suggests — we just hint at it in the warning
    /// below the dropdown rather than removing the option.
    private var availableFrameOptions: [Int] {
        model.frameOptions(width: effectiveSize.width, height: effectiveSize.height, chainWindows: chainWindows)
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

    /// Whether a few-step adapter is driving this render: the engine-owned
    /// Turbo toggle, or any attached Style LoRA. The REF2VA pack has no Turbo
    /// toggle at all — a community distillation loaded here is the ONLY way it
    /// samples in 4 steps — so the LoRA list is load-bearing, not a nicety.
    private var distilledSampling: Bool { turboEngaged || !loras.isEmpty }

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
            return L10n.format("May exceed your Mac's RAM (%lld GB total) at this length.", Int64(total))
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
    /// Everything the generation can be given besides the prompt: the two
    /// anchors, the references and the soundtrack. Which of them exist is the
    /// model's business, so on a model that offers none the section folds to
    /// its own header and says as much.
    private var mediaInputsSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            FoldingSectionHeader(title: "Media inputs for generation", isExpanded: $showMediaInputs)
            if showMediaInputs {
                keyframeRow
                referencesSection
                speechSection
                // Closes the block, the way the rule above Style LoRAs opens
                // one: the sections inside are wells, and without it the
                // last well runs straight into the clip-size fields.
                Divider()
            }
        }
        // The ANSWER, not the width: `onGeometryChange` fires only when its
        // value changes, so a Bool fires at the threshold and nowhere else.
        .onGeometryChange(for: Bool.self) { $0.size.width >= Self.keyframePairMinWidth }
            action: { keyframesSideBySide = $0 }
    }

    /// Side by side only where two wells are still wells rather than slots.
    /// 180% of the pane's own 340pt floor, less the form's 16pt gutters,
    /// because what gets measured is the content width, not the pane's.
    private static let keyframePairMinWidth: CGFloat = 340 * 1.8 - 32

    /// The two anchors. One of them alone takes the full width; the pair
    /// splits the row until the row is too narrow to split, then stacks.
    @ViewBuilder
    private var keyframeRow: some View {
        if !model.supportsLastFrame {
            firstFrameSection
        } else if keyframesSideBySide {
            // `.top`: the last-frame column can carry a caption of its own, and
            // the two grey blocks must stay on one line whether it does or not.
            HStack(alignment: .top, spacing: 8) {
                firstFrameSection.frame(maxWidth: .infinity, alignment: .leading)
                lastFrameSection.frame(maxWidth: .infinity, alignment: .leading)
            }
        } else {
            VStack(alignment: .leading, spacing: 14) {
                firstFrameSection
                lastFrameSection
            }
        }
    }

    private var firstFrameSection: some View {
        keyframeWell(title: "First frame",
                     note: model.supportsLastFrame ? "optional, starts here" : "optional, I2V",
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
                             note: "optional, ends here",
                             url: $lastFrameImageURL,
                             isTargeted: $isLastFrameDropTargeted,
                             help: "Select the image the clip should land on. The first frame sets the size; this one is fitted to it.")
                if lastFrameImageURL != nil && firstFrameImageURL == nil {
                    Text("With only a last frame the model invents the opening and works toward it. Add a first frame to pin both ends.")
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                }
            }
        }
    }

    /// One keyframe slot: thumbnail + clear when filled, drop well when empty.
    /// Both anchors draw the same shape so they read as a pair.
    private func keyframeWell(title: String, note: String, url: Binding<URL?>,
                              isTargeted: Binding<Bool>, help: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            // The note reads as part of the heading, so it sits against it
            // rather than across the row from it.
            HStack(spacing: 6) {
                Text(L10n.text(title)).font(.app(.subheadline).weight(.semibold))
                Text(L10n.text(note))
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            if let picked = url.wrappedValue {
                // Same surface and same floor height as the empty well: a
                // picked image must not change where the form sits.
                MediaDropWellFilled(isTargeted: isTargeted.wrappedValue) {
                    HStack(spacing: 8) {
                        if let img = NSImage(contentsOf: picked) {
                            Image(nsImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 64, height: 48)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                        Text(picked.lastPathComponent)
                            .font(.app(.caption))
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
                HStack(spacing: 6) {
                    Text("References").font(.app(.subheadline).weight(.semibold))
                    // A saved reference is gone and the tiles after it
                    // renumbered, so the prompt's `<Picture n>` may now name
                    // another picture. Cleared by the first edit to either.
                    if refsDroppedOnHydrate {
                        Image(systemName: "exclamationmark.circle.fill")
                            .font(.app(.caption))
                            .foregroundStyle(.orange)
                            .hoverReveal(placement: .pointerClamped(width: refCapBubbleWidth,
                                                                    container: refHeaderRow)) {
                                Self.hoverBubble(L10n.text("Some previously added references were not found on disk. Double-check the media identifiers in the prompt and adjust them if necessary."))
                            }
                    }
                    Text("optional, the model follows them")
                        .font(.app(.caption))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 8)
                    // The combined budget, always in view, with the reason on
                    // hover: it is what takes a slot away while that slot's
                    // own type is not full.
                    HStack(spacing: 4) {
                        Text(L10n.format("%lld of %lld", Int64(refFilesAttached), Int64(H3RefLimits.total)))
                        Image(systemName: "info.circle")
                    }
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
                    .hoverReveal(placement: .pointerClamped(width: refCapBubbleWidth,
                                                            container: refHeaderRow)) {
                        Self.hoverBubble(L10n.text(H3RefLimits.combinedCapNote))
                    }
                }
                // Painted over the well below it — see the prompt heading.
                .zIndex(1)
                .onGeometryChange(for: CGRect.self) { proxy in
                    let r = proxy.frame(in: .global)
                    return CGRect(x: (r.minX / 8).rounded() * 8, y: 0,
                                  width: (r.width / 8).rounded() * 8, height: 0)
                } action: { refHeaderRow = $0 }
                refPanel
            }
            // The whole section (heading included) so its left edge matches
            // First frame's. ONE target: `H3RefDrop` routes by file type,
            // spends both caps and refuses duplicates.
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

    /// Everything the references are, in ONE well: what is attached, and the
    /// ways to attach more. The prompt refers to them by the labels under the
    /// tiles, so the tiles ARE the documentation and the two captions that used
    /// to say so are gone.
    private var refPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if !refImageURLs.isEmpty {
                // The title never breaks inside itself; the PAIR breaks
                // instead, putting the menu on its own line.
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) {
                        refGroupTitle("Images", count: refImageURLs.count, limit: H3RefLimits.images)
                        imageDetailPicker
                        Spacer(minLength: 0)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        refGroupTitle("Images", count: refImageURLs.count, limit: H3RefLimits.images)
                        HStack(spacing: 0) {
                            imageDetailPicker
                            Spacer(minLength: 0)
                        }
                    }
                }
                refTileGrid(urls: $refImageURLs, marker: "Picture", kind: .image)
            }
            if !refVideoURLs.isEmpty {
                refGroupTitle("Clips", count: refVideoURLs.count, limit: H3RefLimits.videos)
                refTileGrid(urls: $refVideoURLs, marker: "Video", kind: .icon("film"))
            }
            if !refAudioURLs.isEmpty {
                refGroupTitle("Audio", count: refAudioURLs.count, limit: H3RefLimits.audios)
                refTileGrid(urls: $refAudioURLs, marker: "Audio", kind: .icon("waveform"))
            }
            refFooterSlots
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(MediaDropWellBackground(isTargeted: isRefDropTargeted))
    }

    /// The count rides the title rather than a tile of its own: a tile that is
    /// not a reference still reads as one, and it moved every real tile.
    /// One sentence wide (300), never wider than the row it is clamped inside.
    private var refCapBubbleWidth: CGFloat {
        refHeaderRow.width > 0 ? min(300, refHeaderRow.width) : 300
    }

    private func refGroupTitle(_ title: String, count: Int, limit: Int) -> some View {
        Text(L10n.format("%@ (%lld/%lld)", L10n.text(title), Int64(count), Int64(limit)))
            .font(.app(.caption).weight(.medium))
            .foregroundStyle(.secondary)
            // Four words on four lines is what a squeezed HStack does to the
            // text before it touches the control beside it.
            .fixedSize()
    }

    /// Beside the Images title rather than at the foot of the section: it is a
    /// property of the images, and at the bottom it read as a property of the
    /// whole set.
    private var imageDetailPicker: some View {
        Picker("", selection: $refImageSize) {
            ForEach(RefImageSizing.allCases, id: \.self) { s in
                Text(L10n.text(s.label)).tag(s)
            }
        }
        .labelsHidden()
        .pickerStyle(.menu)
        .font(.app(.caption))
        .fixedSize()
        // Reference tokens ride through EVERY sampling step, so this is a real
        // time cost, not a quality knob.
        .help("How large each reference image is fed to the model. Maximum detail keeps identity better and is several times slower — every reference token is re-read on every sampling step.")
    }

    private enum RefTileKind {
        case image
        case icon(String)

        var isImage: Bool { if case .image = self { return true } else { return false } }
    }

    /// Three per row at the pane's 340pt floor: 340 − 32 form gutters − 12
    /// `MediaDropModifier` padding − 24 well padding = 272, less two 8pt gaps,
    /// over three.
    private static let refTileSide: CGFloat = 84
    private static let refTileSpacing: CGFloat = 8

    private func refTileGrid(urls: Binding<[URL]>, marker: String,
                             kind: RefTileKind) -> some View {
        RefTileGrid(urls: urls, marker: marker, kind: kind, insert: insertMarker)
    }

    /// Fixed cells: `adaptive(minimum:maximum:)` at ONE value packs 84pt
    /// columns leading (a minimum alone stretches them). Its own view so the
    /// rect the bubbles are clamped to is this view's state, not the pane's.
    /// Identity is the FILE (every way in dedupes); the label is the position.
    private struct RefTileGrid: View {
        let urls: Binding<[URL]>
        let marker: String
        let kind: RefTileKind
        /// Drops a tile's `<Marker n>` into the prompt.
        let insert: (String) -> Void

        @State private var rect: CGRect = .zero

        var body: some View {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: VideoGenView.refTileSide,
                                                   maximum: VideoGenView.refTileSide),
                                         spacing: VideoGenView.refTileSpacing, alignment: .top)],
                      alignment: .leading,
                      spacing: VideoGenView.refTileSpacing) {
                ForEach(Array(urls.wrappedValue.enumerated()), id: \.element) { idx, url in
                    RefTile(url: url, index: idx, marker: marker, kind: kind, container: rect,
                            insert: insert) {
                        urls.wrappedValue.removeAll { $0 == url }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            // Quantised to 8pt: only `minX` and `width` are read
            // (`HoverReveal.clampedX`), and an exact rect changes on every
            // frame of a drag.
            .onGeometryChange(for: CGRect.self) { proxy in
                let r = proxy.frame(in: .global)
                return CGRect(x: (r.minX / 8).rounded() * 8, y: 0,
                              width: (r.width / 8).rounded() * 8, height: 0)
            } action: { rect = $0 }
        }
    }

    /// One reference, in the shape of the chat composer's attachment chips.
    /// Hover floats the picture and its filename in the pane's bubble (the
    /// 84pt caption would be dots). The label is the POSITION, so removing
    /// one renumbers the rest — which is what the prompt's `<Picture n>` means.
    private struct RefTile: View {
        let url: URL
        let index: Int
        let marker: String
        let kind: RefTileKind
        let container: CGRect
        let insert: (String) -> Void
        let remove: () -> Void

        /// What the prompt calls this reference.
        private var promptMarker: String { "<\(marker) \(index + 1)>" }

        /// Loaded once per tile, not in `body`: nine full-size photos re-read
        /// on every change in the pane is a resize that drags.
        @State private var image: NSImage?

        /// Half the tile, so a clip or a track reads as an icon, not a mark.
        private static let iconSize: CGFloat = 34
        /// Long enough for a filename on one or two lines, never wider than
        /// the grid it is kept inside.
        private var bubbleWidth: CGFloat {
            container.width > 0 ? min(220, container.width) : 220
        }

        var body: some View {
            VStack(spacing: 4) {
                ZStack(alignment: .topTrailing) {
                    // `contentShape` beside `clipShape`: the clip cuts only the
                    // DRAWING, and a `.fill` image's overflow still takes the
                    // pointer — over the next tile's badge. The badge is a
                    // SIBLING above the face, never a Button inside a Button.
                    Button { insert(promptMarker) } label: {
                        face
                            .frame(width: VideoGenView.refTileSide, height: VideoGenView.refTileSide)
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                            .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    }
                    .buttonStyle(.plain)
                    Button(action: remove) {
                        // The glyph sits on a photograph, so it carries its own
                        // erased ring instead of trusting what is behind it.
                        ZStack {
                            Circle()
                                .fill(Color(nsColor: .windowBackgroundColor))
                                .frame(width: 17, height: 17)
                            Image(systemName: "multiply.circle.fill")
                                .font(.app(.body))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .padding(3)
                }
                Text(promptMarker)
                    .font(.app(.caption2).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .frame(width: VideoGenView.refTileSide)
            .hoverReveal(placement: .pointerClamped(width: bubbleWidth, container: container)) {
                VideoGenView.hoverBubble(url.lastPathComponent) {
                    // The whole picture, fitted. Sized from its own aspect:
                    // an overlay is proposed the tile's 84pt, so
                    // `maxWidth`/`maxHeight` would cap it there.
                    if let image,
                       let fitted = VideoGenView.previewSize(for: image.size,
                                                             within: bubbleWidth - 16) {
                        Image(nsImage: image)
                            .resizable()
                            .frame(width: fitted.width, height: fitted.height)
                            .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                            .frame(maxWidth: .infinity)
                    }
                }
            }
            .onAppear { if kind.isImage { image = NSImage(contentsOf: url) } }
        }

        @ViewBuilder
        private var face: some View {
            switch kind {
            case .image:
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    placeholder("photo")
                }
            case .icon(let name):
                placeholder(name)
            }
        }

        private func placeholder(_ name: String) -> some View {
            ZStack {
                Color.secondary.opacity(0.12)
                Image(systemName: name)
                    .font(.system(size: Self.iconSize))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private struct RefSlot: Identifiable {
        let id: String
        let title: String
        let icon: String
        let action: () -> Void
    }

    /// A slot exists only while there is room for its type — which is the
    /// tighter of that type's cap and what is left of the combined budget, so
    /// a full set of images and clips takes the audio slot away too.
    private var refSlots: [RefSlot] {
        var out: [RefSlot] = []
        if refRemaining(perType: H3RefLimits.images, current: refImageURLs.count) > 0 {
            out.append(RefSlot(id: "image", title: "Choose image...",
                               icon: "photo.on.rectangle.angled") {
                chooseRefFiles(types: [.image, .png, .jpeg, .heic],
                               limit: refRemaining(perType: H3RefLimits.images,
                                                   current: refImageURLs.count),
                               into: $refImageURLs)
            })
        }
        if refRemaining(perType: H3RefLimits.videos, current: refVideoURLs.count) > 0 {
            out.append(RefSlot(id: "video", title: "Choose clip...", icon: "film") {
                chooseRefFiles(types: [.movie, .mpeg4Movie, .quickTimeMovie],
                               limit: refRemaining(perType: H3RefLimits.videos,
                                                   current: refVideoURLs.count),
                               into: $refVideoURLs)
            })
        }
        if refRemaining(perType: H3RefLimits.audios, current: refAudioURLs.count) > 0 {
            out.append(RefSlot(id: "audio", title: "Choose audio...", icon: "waveform") {
                chooseRefFiles(types: [.audio, .mp3, .wav, .mpeg4Audio],
                               limit: refRemaining(perType: H3RefLimits.audios,
                                                   current: refAudioURLs.count),
                               into: $refAudioURLs)
            })
        }
        return out
    }

    @ViewBuilder
    private var refFooterSlots: some View {
        let slots = refSlots
        if !slots.isEmpty {
            HStack(spacing: 0) {
                ForEach(Array(slots.enumerated()), id: \.element.id) { i, slot in
                    // 52, like `MediaDropWellPair`'s: a short rule between the
                    // slots rather than a line across the well.
                    if i > 0 { Divider().frame(height: 52) }
                    Button(action: slot.action) {
                        // Stacked: three titles side by side have no width to
                        // spare at the pane's floor, and wrapping them behind
                        // their icons left the three slots different heights.
                        MediaWellAction(title: slot.title, systemImage: slot.icon,
                                        layout: .stacked)
                            .frame(maxWidth: .infinity)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
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
                HStack(spacing: 6) {
                    Text("Sound").font(.app(.subheadline).weight(.semibold))
                    Text(L10n.text(model.generatesAudio ? "generated with the video" : "not supported"))
                        .font(.app(.caption))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                }
                if model.generatesAudio {
                    // The same surface as the wells above it, so the block
                    // reads as one of them rather than a stray caption.
                    Text("This model writes its own soundtrack. Describe it in the prompt after \"overall_soundscape:\" (and \"non_diegetic_music:\" for score).")
                        .font(.app(.caption2))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(MediaDropWellBackground(isTargeted: false))
                }
            }
            // Not a drop target, but padded like one: every block above it is,
            // and `MediaDropModifier` gives each 6pt of outline room.
            .padding(6)
        } else {
            audioInputSection
        }
    }

    /// One clip slot with two ways in — a file, or a line spoken by the local
    /// TTS — in the well the Voice pane's reference slot draws. `audioSource`
    /// is which way the user took; the slot's contents are `audioURL`.
    private var audioInputSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Text("Speech & sound").font(.app(.subheadline).weight(.semibold))
                Text("optional, audio-to-video")
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                // The way out of the composer, on the heading rather than in
                // the well: it leaves the whole state, not the line. Laid out
                // in EVERY state and hidden in the others: the chip is taller
                // than the heading, so a chip that comes and goes moved the
                // heading with it.
                Button {
                    if tts.isRunning { tts.cancel() }
                    clearAudio()
                    audioSource = .none
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.backward")
                        Text("Back")
                    }
                    .modifier(PaneChip())
                }
                .buttonStyle(.plain)
                .help("Discard the line and the clip")
                .opacity(audioSource == .speech ? 1 : 0)
                .allowsHitTesting(audioSource == .speech)
            }
            audioWell
            if audioURL != nil {
                // The trim note rides here rather than the clip row: the row
                // is the file's name, and a fact about its length against the
                // video is a fact about the request.
                Text(L10n.text(clipOutlastsVideo
                     ? "Voices, lip sync and timing follow this clip — it becomes the video's soundtrack. Runs on the 2-stage pipeline. The clip is longer than the video and is trimmed to it."
                     : "Voices, lip sync and timing follow this clip — it becomes the video's soundtrack. Runs on the 2-stage pipeline."))
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
            } else if audioSource == .none {
                Text("The model invents a soundtrack from your prompt. Attach speech to make characters say exact words.")
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
            }
        }
        // The slot takes a drop in every state but one: while speech is being
        // generated its result is about to land here, so a drop then is
        // refused (room 0) rather than raced against it. A drop always means
        // the FILE way in, whatever the well was showing.
        .mediaDrop(.audio, limit: tts.isRunning ? 0 : 1, isTargeted: $isAudioDropTargeted) { urls in
            if let dropped = urls.first {
                attachAudio(dropped)
                audioSource = .file
            }
        }
    }

    /// The first Qwen3-TTS on disk, in the catalogue's order — the smallest.
    /// Not the Voice pane's pick: this is a line of dialogue for a video, and
    /// the fast voice is the right default for it. Resolved on appear and
    /// after a download, not in `body`: it walks the model roots.
    private static func resolveSpeechPreset() -> AudioModelPreset? {
        AudioModelPreset.all.first { ServerManager.resolveModelDir(repo: $0.repo) != nil }
    }

    @ViewBuilder
    private var audioWell: some View {
        switch audioSource {
        case .none:
            MediaDropWellPair(
                isTargeted: isAudioDropTargeted,
                leading: MediaDropWellOption(
                    title: "Choose audio…",
                    systemImage: "waveform.badge.plus",
                    caption: "or drag one here",
                    action: chooseAudioFile),
                // The voice it would use, named where the choice is made.
                trailing: MediaDropWellOption(
                    title: "Create speech…",
                    systemImage: "text.bubble",
                    caption: speechPreset?.name ?? "no voice downloaded yet",
                    action: { audioSource = .speech }))
        case .file:
            MediaDropWellFilled(isTargeted: isAudioDropTargeted) {
                attachedClipRow(leadWithDuration: false) {
                    clearAudio()
                    audioSource = .none
                }
            }
        case .speech:
            // Tall enough for the line AND the row under it, so the well does
            // not shrink when the row is not there yet; the form sits at the
            // top of it either way.
            MediaDropWellFilled(isTargeted: isAudioDropTargeted,
                                minHeight: Self.speechWellMinHeight,
                                alignment: .topLeading) {
                speechComposer
            }
        }
    }

    /// A two-line field, the gap, and the clip row, plus the well's padding.
    private static let speechWellMinHeight: CGFloat = 100

    /// The "Create speech" way in: a line with its button beside it, and under
    /// them the ONE row that is either the progress while it speaks or the
    /// clip once it has. The way out is Back on the section heading.
    @ViewBuilder
    private var speechComposer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                TextField("Line to speak — e.g. Good morning. Coffee's ready.", text: $speechText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(2...4)
                    .font(.app(.body))
                    .disabled(tts.isRunning)
                // One width for all three titles, content centred in it: the
                // button changes its word as the flow moves and must not
                // change its size with it.
                if let preset = speechPreset {
                    if tts.isRunning {
                        Button { tts.cancel() } label: {
                            Label("Stop", systemImage: "stop.fill")
                                .font(.app(.caption))
                                .frame(width: Self.speechButtonWidth)
                        }
                        .buttonStyle(.bordered)
                    } else {
                        Button {
                            tts.generate(AudioGenRequest(model: preset, text: speechText), server: server)
                        } label: {
                            Label(L10n.text(audioURL == nil ? "Create speech" : "Recreate speech"), systemImage: "waveform")
                                .font(.app(.caption))
                                .frame(width: Self.speechButtonWidth)
                        }
                        .buttonStyle(.bordered)
                        .disabled(speechText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
            if speechPreset == nil {
                Text("Download a voice first — open the Audio window and grab Qwen3-TTS, then come back.")
                    .font(.app(.caption2))
                    .foregroundStyle(.orange)
            }
            if case .failed(let msg) = tts.phase {
                Text(msg).font(.app(.caption2)).foregroundStyle(.orange)
            }
            // The clip's row, holding its place while the clip is being made.
            if tts.isRunning {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    if case .running(_, _, let msg) = tts.phase {
                        Text(msg).font(.app(.caption)).foregroundStyle(.secondary)
                            .lineLimit(1).truncationMode(.tail)
                    }
                    Spacer()
                }
            } else if audioURL != nil {
                // Removing the clip keeps the composer: the line is still
                // there to be spoken again.
                attachedClipRow(leadWithDuration: true) { clearAudio() }
            }
        }
    }

    /// Wide enough for "Recreate speech" with its glyph at `.caption`.
    private static let speechButtonWidth: CGFloat = 116

    /// The clip in the slot, the way the Voice pane draws its reference: icon,
    /// name, preview, clear. A generated clip leads with its length — it is
    /// the one number the user did not choose.
    private func attachedClipRow(leadWithDuration: Bool, clear: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "waveform.circle.fill").foregroundStyle(.blue)
            if leadWithDuration, let d = audioDuration {
                Text(String(format: "%.1fs", d))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
                Text("·").font(.app(.caption)).foregroundStyle(.secondary)
            }
            Text(audioURL?.lastPathComponent ?? "")
                .font(.app(.caption))
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button {
                togglePreview()
            } label: {
                Image(systemName: audioPlayer?.isPlaying == true ? "stop.circle.fill" : "play.circle")
            }
            .buttonStyle(.borderless)
            .help(L10n.text(audioPlayer?.isPlaying == true ? "Stop preview" : "Preview the clip"))
            Button(action: clear) {
                Image(systemName: "multiply.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Remove the clip")
        }
    }

    /// Whether the attached clip is longer than the selected video length.
    private var clipOutlastsVideo: Bool {
        guard let d = audioDuration else { return false }
        return d > Double(numFrames) / Double(fps) + 0.05
    }

    private func chooseAudioFile() {
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.audio, .wav, .mp3, .mpeg4Audio]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            attachAudio(url)
            audioSource = .file
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

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            FoldingSectionHeader(title: "Advanced options", isExpanded: $showAdvanced)
            if showAdvanced { advancedBody }
        }
    }

    private var advancedBody: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Steps — more steps = more detail/smoother motion, but slower.
            intSliderRow("Steps", value: $steps, range: effectiveStepsRange,
                         help: "Denoising steps. More = more detail and smoother motion, but slower.")
            Text(L10n.text(turboEngaged ? "4 steps is sharp on this adapter and is the floor; more steps still help a little. If the picture shows over-sharp grain, drop the LoRA scale to 0.8-0.95; if it ghosts, raise it to 1.05-1.2." : model.stepsHelp))
                .font(.app(.caption2)).foregroundStyle(.secondary)
            // The low end is REACHABLE and only advised against, so the pane
            // can load a community few-step adapter the way the server can.
            if let advice = model.stepsAdvisory(steps: steps, distilled: distilledSampling) {
                Text(L10n.text(advice)).font(.app(.caption2)).foregroundStyle(.orange)
            }

            // CFG is honored in every LTX pipeline mode, but a CFG-DISTILLED
            // backend has no guidance pass to scale — showing the slider there
            // would be a dead control (the Mage-Flow class).
            if model.supportsCFG {
                // A one-stage tier with a clip attached runs two-stage with the
                // SERVER's guidance defaults — `requestBody` drops these three
                // so the reference set (3.0 / 7.0) applies whole. Each slider
                // says so in its own readout; the Mode hint below says why.
                sliderRow("CFG scale", value: $cfgScale, range: 1...10, step: 0.5,
                          help: "Classifier-free guidance strength. LTX-2 default: 3.0; 1.0 = off (fastest).",
                          lockedReadout: guidanceLockedReadout)
                Text("Guidance strength — how closely the video follows your prompt. 1.0 = off: fastest and most natural-looking. Higher sticks to the prompt more strictly but is slower and can look over-saturated. LTX default is 3.0.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)

                // STG was sent on every LTX request from the day the wire was
                // fixed, with nothing to set it — so it sat at whatever was in
                // storage. A field on the wire with no control is worse than an
                // absent one: the request looks right.
                sliderRow("STG scale", value: $stgScale, range: 0...4, step: 0.5,
                          help: "Spatio-temporal guidance. 0 = off (the default). Steadies motion and structure at the cost of speed.",
                          lockedReadout: guidanceLockedReadout)
                Text("Steadies motion and shape by re-running part of the model with its attention perturbed. 0 = off, which is the default. Around 1.0 helps wobbly motion; higher costs time and can flatten detail.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)

                // Audio guidance belongs to the a2vid guider, so it only shows
                // with a clip attached — otherwise it is a knob on something
                // that never runs.
                if model.supportsAudioInput, audioURL != nil {
                    sliderRow("Audio guidance", value: $cfgAudioScale, range: 1...12, step: 0.5,
                              help: "How closely the picture follows the attached soundtrack. LTX default: 7.0.",
                              lockedReadout: guidanceLockedReadout)
                    Text("How hard the video is pushed to match your clip — lip sync, timing, performance. 7.0 is the LTX default. Lower drifts from the audio; higher locks to it and can look stiff.")
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                }
            }

            // The pipeline itself, and the refine pass that belongs to it. Both
            // stay VISIBLE on every LTX tier: the second pass existing at all
            // is the difference between the tiers, so hiding the pair on
            // one-pass hid the thing the user came here to understand. Mode is
            // a tracked value, so touching it lands the switcher on Custom.
            if model.supportsPipelineModes {
                // The pass count and the refine pass are ONE decision, so they
                // share a row and a caption. `firstTextBaseline` aligns the two
                // labels exactly: a menu bezel and a slider track are not the
                // same height, so aligning the controls would step the labels.
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        // `fixedSize` for the reason `labelledSizeField` gives:
                        // a squeezed HStack gives first on the TEXT.
                        Text("Mode").font(.app(.caption)).fixedSize()
                        // Through `modeLabel`, the same three words as the
                        // caption under the Quality switcher. Shows the
                        // EFFECTIVE mode and locks while a clip forces it.
                        Picker("", selection: modeSelection) {
                            Text(L10n.text(modeLabel(.oneStage))).tag(VideoPipelineMode.oneStage)
                            Text(L10n.text(modeLabel(.twoStage))).tag(VideoPipelineMode.twoStage)
                            Text(L10n.text(modeLabel(.twoStageHQ))).tag(VideoPipelineMode.twoStageHQ)
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .font(.app(.caption))
                        .fixedSize()
                        .disabled(modeUpgradedForAudio)
                    }
                    // Disabled rather than hidden at one stage, and the stored
                    // number is left alone: `requestBody` already drops the
                    // field for a one-stage request, so a trip through 1-stage
                    // cannot cost the user the value they chose.
                    labelledIntSliderRow("Refine steps", value: $stage2Steps, range: 0...6,
                                         readout: { $0 == 0 ? "Auto" : "\($0)" },
                                         disabled: effectiveMode == .oneStage,
                                         disabledReadout: "Off",
                                         help: "Steps in the second, full-resolution pass. Auto uses the reference schedule (3).")
                        .frame(maxWidth: .infinity)
                }
                Text(L10n.text(modeHint))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }

            // Chained windows. Already wired end to end — this is the control
            // that never existed, which is why long clips were unreachable.
            if model.supportsChainedWindows {
                intSliderRow("Chained windows", value: $chainWindows, range: 1...6,
                             help: "Join several generations end to end, each starting from the last frame of the one before.")
                // Through `deliveredFrames`: windows SHARE their seam frames,
                // so the joined clip is one frame short per extra window.
                Text(chainWindows > 1
                     ? L10n.format(
                        "%lld windows joined end to end — %lld frames, and roughly %lldx the time of a single window.",
                        Int64(chainWindows),
                        Int64(VideoModelPreset.deliveredFrames(perWindow: numFrames, chainWindows: chainWindows)),
                        Int64(chainWindows))
                     : L10n.text("Joins several generations end to end for a longer clip. Each window costs another full generation."))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }

            HStack {
                SeedField(label: "Seed", placeholder: "42", range: 0...Int.max, value: $seed,
                          help: "Same seed + same settings reproduces the clip. Paste one to rerun someone else's.")
                Spacer()
            }
            if model.supportsTurbo {
                // A Binding, not `onChange`: the snap below is for the USER's
                // flip. `applyQualityDefaults` also turns Turbo off, and an
                // `onChange` would then overwrite the tier's steps with 30 —
                // reading every tier as Custom.
                Toggle("Turbo (distilled 4-step sampling)", isOn: Binding(
                    get: { turbo },
                    set: { on in
                        turbo = on
                        // 4 is what the adapter is distilled for, 30 the full
                        // render default.
                        steps = on ? 4 : min(model.stepsRange.upperBound, max(model.stepsRange.lowerBound, 30))
                        // Cleared as well as disabled: a ticked box gone grey
                        // reads as a bug. It does not tick itself back later.
                        if on { bestQuality = false }
                        persist()
                        // Fetch on the flip, not at Generate, so the Downloads
                        // pane shows the 744 MB while the prompt is written;
                        // the off-flip cancels an in-flight fetch.
                        if on, turboFetchDecision == .fetch {
                            downloads.startTurboLora(repoId: model.repo)
                        } else if !on {
                            downloads.cancelTurboLora(repoId: model.repo)
                        }
                    }))
                    .font(.app(.caption))
                    .help("Runs the Turbo distillation LoRA: 4 steps instead of 30, about twice as fast end to end. Slightly softer detail and harder light than a full render. The adapter ships with the model; packs downloaded before it existed fetch it once, on the first run with this on.")
                if turboFetchDecision == .fetch {
                    Text("Turbo needs a \(TurboLoraFetch.approxMB) MB adapter this pack predates — it downloads once, and Generate waits for it.")
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                } else if turboFetchDecision == .unavailableRemotely {
                    Text("Turbo runs on the Mac hosting this model; it needs the adapter in ITS copy of the pack.")
                        .font(.app(.caption2)).foregroundStyle(.secondary)
                }
            }
            if model.supportsDiffusionDecoder {
                Toggle("Diffusion decoder (sharper, slower)", isOn: $diffusionDecoder)
                    .font(.app(.caption))
                    .help("Off (default): the plain convolutional decoder. On: LTX's own diffusion decoder — the one their published clips use. It denoises the frames instead of interpolating them, so fine texture and edges come out sharper. The decode itself takes about 21 s for a 97-frame clip at 768x512; measured end to end against the plain decoder in the same session the difference was inside run-to-run variance.")
            }
            if model.supportsFastRecipe {
                Toggle("Max quality (slower)", isOn: $bestQuality)
                    .font(.app(.caption))
                    .help("Off (default): the fast recipe — step caching + attention reuse, about 2.8x faster at 768p. On: every denoising step is fully computed; marginally better detail for final renders, and it drops the step cache, which is what makes a long clip fit in memory. Turbo runs without the recipe anyway, so this has nothing to add there.")
                    // Under turbo the recipe is already off server-side; a
                    // toggle that could not change anything is a dead control.
                    .disabled(turboEngaged)
            }
            Toggle("Show live preview while generating", isOn: $livePreview)
                .font(.app(.caption))
                .help("On: each denoising step sends a small still built by projecting the latent straight to RGB — enough to see the shot taking form, but flat and soft compared with the finished clip, which is decoded by the VAE. Off (default): no preview. It is not free — every step solves for the clean latent and copies the previewed frame to the CPU.")

            if model.supportsLoRA { loraSection }
        }
    }

    /// Both LoRA wells take this, so the empty one and a filled one are the
    /// same block rather than two sizes of the same idea. Thinner than the
    /// image wells' 84: this one holds two lines of text, not a thumbnail.
    private static let loraWellMinHeight: CGFloat = 64

    @ViewBuilder
    private var loraSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Divider()
            Text("Style LoRAs").font(.app(.caption).weight(.semibold))
            ForEach(Array(loras.enumerated()), id: \.element.id) { index, lora in
                loraRow(index: index, lora: lora)
            }
            // The way in is the well itself, and it comes back under the last
            // adapter so adding a second one needs no separate control. At the
            // cap there is nothing to offer, so it goes.
            if loras.count < maxLoras { loraAddWell }
        }
    }

    /// Click-only: a LoRA is picked from a file panel, and a well that accepts
    /// a drag is a promise this one does not keep.
    private var loraAddWell: some View {
        Button(action: chooseLora) {
            MediaWellAction(title: "Choose .safetensors…",
                            systemImage: "paintpalette",
                            caption: "Add LoRA adapter for custom style. Several can stack at once.")
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, minHeight: Self.loraWellMinHeight, alignment: .center)
            .background(MediaDropWellBackground(isTargeted: false))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func loraRow(index: Int, lora: LoraAdapter) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: "paintpalette")
                    .foregroundStyle(.secondary)
                Text(URL(fileURLWithPath: lora.path).lastPathComponent)
                    .font(.app(.caption))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(lora.path)
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
            HStack(spacing: 8) {
                Text("Scale").font(.app(.caption))
                Slider(value: $loras[index].scale, in: 0...2, step: 0.05)
                // Fixed width: a readout that sizes to its digits drags the
                // slider's right edge every time the value crosses a width.
                Text(String(format: "%.2f", lora.scale))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 34, alignment: .trailing)
            }
            .onChange(of: loras[index].scale) { _, _ in guard !hydrating else { return }; persist() }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, minHeight: Self.loraWellMinHeight, alignment: .leading)
        .background(MediaDropWellBackground(isTargeted: false))
    }

    /// Live "is the model resident, and what does the GPU hold" line under the
    /// keep-loaded toggle — fed by the slow `/v1/models` poll.
    private var residencyRow: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(service.residency?.loaded == true ? Color.green : Color.secondary.opacity(0.4))
                .frame(width: 7, height: 7)
            Text(L10n.text(residencyText))
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
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
        let panel = OpenPanel.make()
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
        let panel = OpenPanel.make()
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
        let panel = OpenPanel.make()
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
    /// `lockedReadout`: a WORD in place of the number, and the slider disabled.
    /// A disabled slider on macOS looks almost live, so the readout is what
    /// says the value shown is not the one the request will carry.
    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>,
                           step: Double, help: String? = nil, lockedReadout: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.caption))
                Spacer()
                Text(lockedReadout ?? String(format: "%.1f", value.wrappedValue))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
                .disabled(lockedReadout != nil)
                .padding(.top, Self.steppedSliderTrackDrop)
        }
        .help(help ?? "")
    }

    /// Labeled slider for an `Int` setting (bridges to a `Double` slider).
    private func intSliderRow(_ label: String, value: Binding<Int>, range: ClosedRange<Int>, help: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.caption))
                Spacer()
                Text("\(value.wrappedValue)")
                    .font(.app(.caption).monospacedDigit())
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
            .padding(.top, Self.steppedSliderTrackDrop)
        }
        .help(help ?? "")
    }

    /// Slider over a small integer range whose readout is a WORD at some
    /// positions rather than the number. A disabled one sits at its left end
    /// and reads `disabledReadout` WITHOUT writing that value: the control is
    /// inapplicable, not reset, so the user's choice survives a trip through a
    /// mode that has no use for it.
    private func labelledIntSliderRow(_ label: String, value: Binding<Int>,
                                      range: ClosedRange<Int>,
                                      readout: @escaping (Int) -> String,
                                      disabled: Bool = false,
                                      disabledReadout: String? = nil,
                                      help: String? = nil) -> some View {
        let shown = disabled ? (disabledReadout ?? readout(range.lowerBound))
                             : readout(value.wrappedValue)
        return VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.caption))
                Spacer()
                Text(L10n.text(shown))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: Binding(
                    get: { disabled ? Double(range.lowerBound) : Double(value.wrappedValue) },
                    set: { newVal in
                        guard !disabled else { return }
                        value.wrappedValue = Int(newVal.rounded())
                    }
                ),
                in: Double(range.lowerBound)...Double(range.upperBound),
                step: 1
            )
            .disabled(disabled)
            .padding(.top, Self.steppedSliderTrackDrop)
        }
        .help(help ?? "")
    }

    /// A stepped slider reserves a tick row under its track, so the track sits
    /// glued to the caption above; 3pt puts it (and Refine steps) on the Mode
    /// picker's centre line. Frames is exempt: a section title spaces it.
    private static let steppedSliderTrackDrop: CGFloat = 3

    private var actionRow: some View {
        VStack(spacing: 8) {
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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (lanModel == nil && !downloads.bundleReady(model.bundle)) || !customSizeValid || !payloadFits)
                }
            }
            // A disabled button with no reason is worse than no button, and
            // this one guards a combination the controls no longer produce:
            // it is the backstop, not the message people normally see.
            if !payloadFits {
                Text("\(VideoModelPreset.deliveredFrames(perWindow: numFrames, chainWindows: chainWindows)) frames at \(effectiveSize.width) × \(effectiveSize.height) is more than one response can carry. Shorten the clip, drop a window, or use a smaller canvas.")
                    .font(.app(.caption2)).foregroundStyle(.orange)
            }
            if !service.isRunning, let est = timeEstimate {
                Text(L10n.text(est))
                    .font(.app(.caption))
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
                        if let img = service.livePreview {
                            Image(nsImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxHeight: 220)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                        }
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
                        Text(message).font(.app(.footnote)).foregroundStyle(.secondary)
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
                    .font(.app(.caption))
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
                .font(.app(.caption))
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
        // The fields are the canvas now. A blob from a build that stored a
        // PRESET row rather than a typed size opens on that row's numbers, so
        // nobody's saved canvas changes under them.
        let saved = s.resolvedResolution(for: model)
        if saved.isCustom || saved.isMatchSource {
            customWidthText = String(s.customWidth)
            customHeightText = String(s.customHeight)
        } else {
            customWidthText = String(saved.width)
            customHeightText = String(saved.height)
        }
        numFrames = s.numFrames
        fps = s.fps
        // A backend with one pipeline has no Mode control to explain a saved
        // two-stage, so the switcher would read Custom with nothing to click.
        mode = model.supportsPipelineModes ? s.mode : .oneStage
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
        chainWindows = model.supportsChainedWindows ? min(6, max(1, s.chainWindows)) : 1
        seed = s.seed
        keepResident = s.keepResident
        livePreview = s.livePreview
        bestQuality = s.bestQuality
        // The pair could be stored together by a build that only disabled the
        // box, so the launch after that one must not show it ticked and grey.
        if turboEngaged { bestQuality = false }
        diffusionDecoder = s.diffusionDecoder
        promptHeight = PromptEditorHeight.clamp(s.promptHeight)
        loras = s.loras
        // A LoRA file may have moved since last session — drop stale entries.
        loras.removeAll { !FileManager.default.fileExists(atPath: $0.path) }
        clampFramesToRAM()

        // The draft. Files come back only if they are still there. Anchors and
        // references are restored whatever the model, as they survive a
        // preset switch in session (the request gates them); audio follows
        // `applyModelDefaults`, which clears it on a backend without input.
        prompt = s.prompt
        showMediaInputs = s.showMediaInputs
        showAdvanced = s.showAdvanced
        firstFrameImageURL = Self.existingFile(s.firstFramePath)
        lastFrameImageURL = Self.existingFile(s.lastFramePath)
        speechText = s.speechText
        audioSource = model.supportsAudioInput ? s.audioSource : .none
        if let clip = Self.existingFile(s.audioPath), audioSource != .none {
            // Restored, not re-attached: `attachAudio` snaps the length up to
            // cover the clip, and the saved length is the user's own choice
            // made after that snap.
            audioURL = clip
            audioDuration = Self.audioDuration(of: clip)
        } else if audioSource == .file {
            audioSource = .none
        }
        refImageURLs = Self.existingFiles(s.refImagePaths)
        refVideoURLs = Self.existingFiles(s.refVideoPaths)
        refAudioURLs = Self.existingFiles(s.refAudioPaths)
        // Missing, not merely duplicate: only a file that is gone renumbers.
        refsDroppedOnHydrate = [s.refImagePaths, s.refVideoPaths, s.refAudioPaths].joined()
            .contains { Self.existingFile($0) == nil }
        refImageSize = s.refImageSize
    }

    private static func existingFile(_ path: String?) -> URL? {
        guard let path, FileManager.default.fileExists(atPath: path) else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Deduplicated as well as checked: the tiles are identified by file, and
    /// this is a way in like the picker and the drop.
    private static func existingFiles(_ paths: [String]) -> [URL] {
        var seen = Set<URL>()
        return paths.compactMap { existingFile($0) }.filter { seen.insert($0).inserted }
    }

    /// Every sticky field, as the blob it would persist to — `Equatable`, so
    /// one `onChange` covers all of them.
    private var stickySnapshot: VideoGenSettings {
        var s = VideoGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.quality = quality
        // Always the Custom sentinel: the typed pair IS the canvas, and every
        // reader that wants numbers (the chat's `generate_video` among them)
        // resolves the sentinel through `customWidth`/`customHeight`.
        s.resolutionId = ResolutionOption.custom.id
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
        s.livePreview = livePreview
        s.bestQuality = bestQuality
        s.diffusionDecoder = diffusionDecoder
        s.turbo = turbo
        s.promptHeight = PromptEditorHeight.clamp(promptHeight)
        s.loras = loras
        s.prompt = prompt
        s.firstFramePath = firstFrameImageURL?.path
        s.lastFramePath = lastFrameImageURL?.path
        s.audioSource = audioSource
        s.audioPath = audioURL?.path
        s.speechText = speechText
        s.refImagePaths = refImageURLs.map(\.path)
        s.refVideoPaths = refVideoURLs.map(\.path)
        s.refAudioPaths = refAudioURLs.map(\.path)
        s.refImageSize = refImageSize
        s.showMediaInputs = showMediaInputs
        s.showAdvanced = showAdvanced
        return s
    }

    private func persist() { stickySnapshot.save() }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        // Still sized to this Mac, just written into the fields instead of
        // selected in a picker: the largest canvas whose default frame count
        // this much RAM can hold.
        let recommended = model.recommendedResolution(totalGB: RAMChecker.totalGB)
        customWidthText = String(recommended.width)
        customHeightText = String(recommended.height)
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
            refImageSize: refImageSize,
            livePreview: livePreview
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
        AppActivation.openWindow(id: "serverLog", using: openWindow)
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
