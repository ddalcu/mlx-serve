import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Image generation window — native FLUX.2, Krea-2-Turbo and Mage-Flow (no
/// Python). The model picker lists every `ImageModelPreset`; the server
/// auto-routes to the right image backend by the model's `model_type`.
struct ImageGenView: View {
    @EnvironmentObject var service: ImageGenService
    /// The enlarge side. Two services, one pane — see `ImagePanePreview`.
    @EnvironmentObject var restore: RestoreService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    /// For "Send to Chat" — the hand-off opens a new conversation and switches
    /// the window to it (`AppState.sendGeneratedMediaToNewChat`).
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    @State private var showAdvanced: Bool = false
    @State private var model: ImageModelPreset = .flux2Klein4B_Q4
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = ImageModelPreset.flux2Klein4B_Q4.defaultResolution
    // Held as text, not Int: a size field has to be allowed to be empty or
    // half-typed while the user edits it, which a numeric binding fights.
    @State private var customWidthText: String = "1024"
    @State private var customHeightText: String = "1024"
    @State private var steps: Int = 8
    @State private var seed: Int = -1
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: ImageGenRequest? = nil
    /// The enlarge waiting on the same RAM alert. Only one of this and
    /// `pendingRequest` is ever set — the alert's confirm button reads which.
    @State private var pendingEnlarge: String? = nil
    /// Keep the model resident after generating (default off → unload to free
    /// GPU memory). On → the next generation reuses it instantly.
    @State private var keepResident: Bool = false
    /// Apply the NSFW content filter (on by default). Off → sends safety:false so
    /// the server skips it. (The license expects filtering in deployments.)
    @State private var safeMode: Bool = true
    /// Image-to-image source (transient — not persisted, like video's first frame).
    @State private var initImageURL: URL? = nil
    /// Extra in-context references for edit mode (FLUX.2 multi-reference):
    /// "replace the face in image 1 with the face from image 2". Transient,
    /// like the source. The server takes at most 3 beside the source.
    @State private var refImageURLs: [URL] = []
    /// img2img renoise strength: low = stay close to the source, high = mostly prompt.
    @State private var strength: Double = 0.6
    /// What the attached source image is FOR: an instruction edit (FLUX.2
    /// in-context reference, keeps the subject), a renoise variation, or a
    /// SeedVR2 enlargement. Only meaningful while a source IS attached —
    /// `effectiveVerb` is nil otherwise and the pane is text-to-image.
    @State private var sourceVerb: ImageSourceVerb = .edit
    /// Conditioning rebalance (Advanced): global gain on the prompt embeddings.
    @State private var condGain: Double = 1.0
    /// Conditioning rebalance (Advanced): per-tapped-layer weights as typed.
    @State private var condWeightsText: String = ""
    /// Style LoRAs (Advanced): stacked `.safetensors` adapters ([] = none).
    /// Several can attach at once — their effects sum, so order doesn't matter.
    @State private var loras: [LoraAdapter] = []
    /// True while `hydrate()` seeds `@State` from saved settings. Hydrating
    /// `model`/`quality` fires their `.onChange` (applyModelDefaults /
    /// applyQualityDefaults) which would clobber the just-restored
    /// steps/resolution — so every reset + persist is guarded on this.
    @State private var hydrating: Bool = false
    /// Hydrate exactly once per window lifetime (the first `.onAppear`).
    @State private var didHydrate: Bool = false
    /// True while a drag carrying a file is hovering the source-image section
    /// — drives that section's dashed-border highlight and the well's fill.
    @State private var isDropTargeted: Bool = false
    /// Set when a handoff can't proceed (the result file is gone). Shown as an
    /// alert rather than silently doing nothing, which is what a dead button
    /// looks like from the outside.
    @State private var handoffError: String? = nil

    // ── Enlarge (SeedVR2) ───────────────────────────────────────────────
    // A different model family behind the same source image, so it carries its
    // own model pick and scale. Seed and Keep-loaded are deliberately NOT
    // duplicated here: the pane has one Advanced section now.

    @State private var restoreModel: RestoreModelPreset = .seedvr2_3b
    /// Selected network restore model's routing id; nil = local.
    @State private var restoreLanModel: String? = nil
    /// 1 = restore only (same resolution, sharper/cleaner). SeedVR2 has no
    /// upscaling of its own — a factor above 1x is a bicubic resize to that
    /// target canvas BEFORE restoration, so the model fills in real detail at
    /// the larger size instead of the resize just looking soft. Continuous, so
    /// "1.5x" is a drag rather than a compromise between two menu items.
    @State private var scale: Double = 2
    /// The strip row the preview is showing, by path. Set when either service
    /// finishes and when a row is clicked; `ImagePanePreview` turns it plus the
    /// two phases into what is drawn. Never derived from the verb — that is
    /// what used to blank a perfectly good picture the moment the controls
    /// changed.
    @State private var selectedPath: String? = nil
    /// Set when a delete can't go through, so a failed Move to Trash says so
    /// instead of looking like a dead menu item.
    @State private var deleteError: String? = nil

    var body: some View {
        // No window-sized floor: this is a PAGE of the chat window now, and a
        // root minimum wider than the detail column overflows it and clips
        // both edges. Small windows shrink the preview side instead.
        //
        // ONE pane. There is no Create/Upscale switch: enlarging is a verb you
        // apply to a source image, beside Edit and Variation, not a place you
        // travel to and come back from.
        readyView
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                // Clear on the next runloop tick so the cascade of `.onChange`
                // fired by hydration's state writes is ignored.
                DispatchQueue.main.async { hydrating = false }
            }
            // Freshen the network-model list so LAN entries are current in
            // the picker (discovery lands seconds after the server boots).
            if server.status == .running { Task { await server.refreshModels() } }
            downloads.ensureNsfwClassifier() // best-effort: provision the shared content filter
        }
        // Persist every other sticky field on change (model/quality persist in
        // their sections after applying preset defaults).
        .onChange(of: resolution) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: customWidthText) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: customHeightText) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: safeMode) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: scale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: restoreModel) { _, _ in guard !hydrating else { return }; persist() }
        // A model switch can take a verb away (Mage-Flow has no variation
        // path). Re-point it here rather than at request time, so the picker
        // never shows a selection the backend would 400 on.
        .onChange(of: model) { _, m in
            let resolved = ImageSourceVerb.resolve(sourceVerb, for: m)
            if resolved != sourceVerb { sourceVerb = resolved }
        }
        // A finished run selects itself, so it lands in the preview AND is
        // highlighted in the strip — one notion of "what you're looking at".
        .onChange(of: service.phase) { _, phase in
            if case .completed(let path) = phase { selectedPath = path }
        }
        .onChange(of: restore.phase) { _, phase in
            if case .completed(let path) = phase { selectedPath = path }
        }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    // With no source attached this is text-to-image and reads
                    // exactly as it always did: the prompt first. Attaching a
                    // photo and choosing Enlarge is what drops the prompt and
                    // brings the scale up — the source section becomes the
                    // first thing on the page by consequence, not by a mode.
                    if effectiveVerb != .enlarge { promptSection }
                    sourceImageSection
                    if effectiveVerb == .enlarge {
                        scaleSection
                        enlargeModelSection
                    } else {
                        modelSection
                        qualitySection
                        resolutionSection
                    }
                    if showAdvanced { advancedSection } else { advancedToggle }
                    actionRow
                }
                .padding(16)
            }
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                sessionStrip
                outputFolderLink
            }
            .padding(16)
            // The preview is what gives way in a small window — the generated
            // image scales to fit; the controls column keeps its form floor.
            .frame(minWidth: 280)
        }
        .alert("Model exceeds your Mac's RAM", isPresented: $showRAMWarning) {
            Button("Cancel", role: .cancel) { pendingRequest = nil; pendingEnlarge = nil }
            Button(pendingEnlarge != nil ? "Enlarge Anyway" : "Generate Anyway", role: .destructive) {
                if let path = pendingEnlarge { startEnlarge(sourcePath: path) }
                else if let req = pendingRequest { service.generate(req, server: server) }
                pendingRequest = nil
                pendingEnlarge = nil
            }
        } message: {
            Text(ramWarningMessage)
        }
        .alert("Couldn't delete", isPresented: Binding(
            get: { deleteError != nil },
            set: { if !$0 { deleteError = nil } })) {
            Button("OK", role: .cancel) { deleteError = nil }
        } message: {
            Text(deleteError ?? "")
        }
        .alert("Can't enlarge this", isPresented: Binding(
            get: { handoffError != nil },
            set: { if !$0 { handoffError = nil } })) {
            Button("OK", role: .cancel) { handoffError = nil }
        } message: {
            Text(handoffError ?? "")
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Prompt").font(.subheadline.weight(.semibold))
                Spacer()
                // Same idiom as the Video pane. For an EDIT model this menu is
                // the feature discovery surface: the repertoire is prompts, so
                // an unlisted capability may as well not exist.
                Menu("Examples") {
                    ForEach(model.promptExamples(editing: isEditing), id: \.name) { group in
                        Menu(group.name) {
                            ForEach(group.examples, id: \.title) { ex in
                                Button(ex.title) { prompt = ex.body; persist() }
                            }
                        }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .font(.caption)
            }
            TextEditor(text: $prompt)
                .font(.body)
                .frame(height: 110)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
        }
    }

    private var sourceImageSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Source image (optional)").font(.subheadline.weight(.semibold))
                Spacer(minLength: 8)
                // The mode switch belongs to the SECTION, not to the source
                // row: sitting between the source and the references it split
                // a list of identical rows in half and read as a property of
                // the picture above it. In the header it sits beside the name
                // of the thing it modifies, and the pictures below are one
                // uninterrupted list. It only appears where BOTH modes exist —
                // a model with instruction editing but no VAE-encoder
                // variation path (Mage-Flow-Edit) would otherwise offer
                // "Variation" and get a 400 back — and only once there is a
                // source for it to apply to.
                // Three verbs now, not two: Enlarge joins Edit and Variation
                // instead of living behind a top-level mode switch. Shown
                // whenever the model offers more than one — on a txt2img-only
                // preset the list is [.enlarge] alone, and a picker with one
                // item is a label, so the hint below says it in words instead.
                if initImageURL != nil, availableVerbs.count > 1 {
                    Picker("", selection: $sourceVerb) {
                        ForEach(availableVerbs) { v in
                            Text(v.label).tag(v)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .fixedSize()
                    .onChange(of: sourceVerb) { _, _ in guard !hydrating else { return }; persist() }
                }
            }
            if let url = initImageURL {
                // The source is picture 1 and the references follow it, which
                // is the numbering the prompt refers to — so they are one
                // list, drawn by one row.
                imageRow(url, number: numberedImageRows ? 1 : nil,
                         help: "Remove the source image (back to text-to-image)") {
                    initImageURL = nil
                    refImageURLs = []
                }
                if effectiveVerb == .enlarge {
                    if let note = cropNote {
                        Text(note).font(.caption2).foregroundStyle(.secondary)
                    }
                    Text("Enlarges the photo and restores real detail at the new size — not just a blurry resize. Runs SeedVR2, a different model from the one above.")
                        .font(.caption2).foregroundStyle(.secondary)
                } else if effectiveEditMode {
                    ForEach(Array(refImageURLs.enumerated()), id: \.element) { i, ref in
                        imageRow(ref, number: numberedImageRows ? i + 2 : nil,
                                 help: "Remove this reference image") {
                            refImageURLs.removeAll { $0 == ref }
                        }
                    }
                    if refImageURLs.count < maxRefImages {
                        Button {
                            chooseRefImage()
                        } label: {
                            Label("Add reference image…", systemImage: "photo.badge.plus")
                                .font(.caption)
                        }
                    }
                    if refImageURLs.isEmpty {
                        Text("Describe the change in the prompt — “make the hair blue”, “remove the monitor”. The model sees the original and keeps the rest.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Refer to the pictures by number — the source is image 1, references follow in order: “replace the face of the man in image 1 with the face from image 2”.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } else if model.supportsImg2Img {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text("Variation strength").font(.caption)
                            Spacer()
                            Text(String(format: "%.0f%%", strength * 100))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Slider(value: $strength, in: 0.1...1.0, step: 0.05)
                            .onChange(of: strength) { _, _ in guard !hydrating else { return }; persist() }
                        Text("Low = stay close to the source; high = mostly the prompt.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                MediaDropWell(title: sourceImageButtonLabel,
                              systemImage: "photo.badge.plus",
                              isTargeted: isDropTargeted) { chooseSourceImage() }
                Text(sourceImageHint)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        // Drops land on the source-image section rather than the whole window,
        // and because the section GROWS once a source is set, the target grows
        // to cover the thumbnail and reference rows — which is exactly where a
        // later drop is aimed. `ImageDropPlacement` decides which slot it
        // lands in, and states the ROOM it has for one — so a pane with
        // nothing left to fill bounces the file instead of swallowing it.
        .mediaDrop(.image,
                   limit: ImageDropPlacement.room(source: initImageURL,
                                                  editing: effectiveEditMode,
                                                  refs: refImageURLs.count,
                                                  refLimit: maxRefImages),
                   isTargeted: $isDropTargeted) { placeDroppedImages($0) }
    }

    /// One attached picture. The source and every reference draw the SAME row
    /// — they are one numbered list to the model, so they read as one list
    /// here, and a row that differs only in what its ✕ does has no business
    /// being written twice.
    private func imageRow(_ url: URL, number: Int?, help: String,
                          remove: @escaping () -> Void) -> some View {
        HStack(spacing: 8) {
            if let number {
                Text("\(number)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 10, alignment: .trailing)
            }
            if let img = NSImage(contentsOf: url) {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 40, height: 40)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            Text(url.lastPathComponent)
                .font(.caption)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Button(action: remove) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help(help)
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
    }

    /// The numbers are the prompt's own vocabulary ("the face from image 2"),
    /// so they appear exactly when there is more than one picture to tell
    /// apart. A lone "1" beside the only image on screen is decoration.
    private var numberedImageRows: Bool {
        effectiveEditMode && !refImageURLs.isEmpty
    }

    /// What a source image is FOR on this model. Derived from the same verb
    /// list the picker draws, so the sentence and the control can never
    /// disagree about what the backend will accept.
    private var sourceImageButtonLabel: String { "Choose image…" }

    private var sourceImageHint: String {
        switch availableVerbs {
        case [.edit, .variation, .enlarge]:
            return "Edit it with an instruction, remix it as a variation, or enlarge it."
        case [.edit, .enlarge]:
            return "Edit it with an instruction — say what to change and the rest stays put — or enlarge it."
        case [.variation, .enlarge]:
            return "Generate a variation of it guided by the prompt, or enlarge it."
        default:
            // One verb: a picker would be a label, so the sentence carries it.
            return "This model can't edit or remix a photo, so a source image here means one thing: enlarge it and restore detail."
        }
    }

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`.
    private var modelSection: some View {
        MediaModelChooser.pane(
            all: ImageModelPreset.all,
            onThisMac: CustomMediaModels.imagePresets(from: server.allModels),
            capability: "image",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.imagePreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults(); persist() }
    }

    @ViewBuilder
    private var qualitySection: some View {
        // A distilled model has ONE schedule. Offering tiers that only buy time
        // is the same silent-no-op the capability flags exist to kill.
        if model.stepsAreFixed {
            VStack(alignment: .leading, spacing: 6) {
                Text("Quality").font(.subheadline.weight(.semibold))
                Text("Fixed at \(model.fixedSteps) steps — this model is distilled for a \(model.fixedSteps)-step schedule, so more steps cost time without adding detail.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        } else {
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
    }

    /// Tier-specific hint with the actual numbers, so users see the cost up
    /// front. CFG is deliberately absent: no image backend reads a guidance
    /// field, so quoting one would be inventing a knob.
    private var qualityHint: String {
        "\(model.settings(quality).steps) steps"
    }

    private var resolutionSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Resolution").font(.subheadline.weight(.semibold))
            Picker("", selection: $resolution) {
                ForEach(model.resolutionOptions(editMode: isEditing)) { r in
                    Text(r.label).tag(r)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            // Dropping the source image (or leaving edit mode) takes "Match
            // source" off the menu — re-point the selection or the picker shows
            // an empty label.
            .onChange(of: isEditing) { _, editing in
                resolution = model.validResolution(resolution, editMode: editing)
            }
            if resolution.isMatchSource {
                Text("The edit comes back at the source image's own size.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if resolution.isCustom { customResolutionFields }
        }
    }

    /// Width/height for the Custom… row, with the one line of feedback the
    /// grid produced. The server rewrites anything off-grid regardless, so the
    /// point of this is to say so BEFORE the request rather than leave the user
    /// reading an unexpected size off a finished image.
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
                    // A correction is information; a refusal is the reason
                    // Generate is disabled, so only that one is coloured.
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

    /// Only gates while Custom is actually selected — a stale unparseable value
    /// left in the fields must not disable Generate for a fixed bucket.
    private var customSizeValid: Bool {
        !resolution.isCustom || customResolutionVerdict.isValid
    }

    /// What the selected model's grid makes of the typed size. Non-numeric or
    /// empty text reads as 0, which the grid already refuses by name.
    private var customResolutionVerdict: CustomResolution {
        model.resolutionGrid.resolve(width: Int(customWidthText) ?? 0,
                                     height: Int(customHeightText) ?? 0)
    }

    /// The size the request should carry: the grid's corrected numbers while
    /// Custom is selected, the picked bucket otherwise. Generate is gated on
    /// the same verdict, so the `?? ` fallback is never the one that ships.
    private var effectiveSize: (width: Int, height: Int) {
        guard resolution.isCustom else { return (resolution.width, resolution.height) }
        return customResolutionVerdict.size ?? (resolution.width, resolution.height)
    }

    // MARK: - Enlarge (SeedVR2)

    /// The source photo's pixel dimensions, or nil while nothing's picked /
    /// the file can't be decoded.
    private var sourcePixelSize: (width: Int, height: Int)? {
        guard let url = initImageURL, let img = NSImage(contentsOf: url),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        return (cg.width, cg.height)
    }

    /// SeedVR2 needs both pixel dimensions divisible by 16 (`RestoreGeometry`)
    /// — told UP FRONT rather than only discovered from the run log. nil when
    /// the photo's already on-grid, or when scaling up (a resize hits the
    /// target canvas exactly, so no crop is needed).
    private var cropNote: String? {
        guard scale <= 1, let (w, h) = sourcePixelSize else { return nil }
        guard let crop = RestoreGeometry.centeredCrop(width: w, height: h) else { return nil }
        return "Will be center-cropped to \(crop.width) × \(crop.height) — SeedVR2 needs both dimensions divisible by 16."
    }

    private var scaleSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Scale").font(.subheadline.weight(.semibold))
                Spacer()
                Text(scale <= 1 ? "1× (restore only)" : RestoreGeometry.formatFactor(scale))
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            Slider(value: $scale, in: 1...4, step: 0.1) {
                Text("Scale")
            } minimumValueLabel: {
                Text("1×").font(.caption2).foregroundStyle(.secondary)
            } maximumValueLabel: {
                Text("4×").font(.caption2).foregroundStyle(.secondary)
            }
            .labelsHidden()
            if let (w, h) = sourcePixelSize {
                let t = scale > 1
                    ? RestoreGeometry.upscaledTarget(width: w, height: h, factor: scale)
                    : (width: RestoreGeometry.snap(w), height: RestoreGeometry.snap(h))
                Text("\(w) × \(h) → \(t.width) × \(t.height)")
                    .font(.caption2).foregroundStyle(.secondary)
            }
            // At 1x the model only cleans up what is there, which is the least
            // obvious thing it does — so it is said in words, not left to a
            // slider readout that disappears the moment you drag.
            if scale <= 1 {
                Text("Same size, cleaned up: sharpens detail and removes compression artifacts.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private var enlargeModelSection: some View {
        MediaModelChooser.pane(
            all: RestoreModelPreset.all,
            onThisMac: CustomMediaModels.restorePresets(from: server.allModels),
            capability: "restore",
            selected: $restoreModel, lanModel: $restoreLanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.restorePreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist)
    }

    // MARK: - Advanced

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
                } label: {
                    Image(systemName: "chevron.down")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
            // No CFG field and no negative prompt: NO image backend reads either
            // one (`handleImage` parses neither, and the app never sent
            // guidance), so both were pure decoration on every model, not just
            // the distilled ones. Steps stay overridable even where the schedule
            // is fixed — it's the Advanced panel, and the hint says the cost.
            HStack {
                // Steps belong to the image schedule; an enlarge is one step
                // by construction, so the field would be a lie there.
                if effectiveVerb != .enlarge { numberField("Steps", value: $steps, step: 1) }
                // -1 is the random sentinel and renders as an EMPTY box, so the
                // placeholder explains it instead of a literal -1 that reads as
                // a broken value. ONE seed for the pane — a seed is a seed, and
                // two of them was the duplicated shell in miniature.
                SeedField(label: "Seed", placeholder: "random", range: -1...Int.max, value: $seed,
                          help: "Same seed + same settings reproduces the result. Paste one to rerun someone else's; leave it empty for a new one each time.")
            }
            if effectiveVerb != .enlarge, model.stepsAreFixed {
                Text("This model is distilled for \(model.fixedSteps) steps; other values cost time without adding detail.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Toggle("Keep model loaded afterwards", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next run is instant. Off (default): it's unloaded to free GPU memory.")
            // Everything below is a property of the IMAGE model — safety
            // filter, text-encoder rebalance, DiT LoRAs. None of it reaches
            // SeedVR2, and a control the backend ignores is worse than one
            // that isn't there.
            if effectiveVerb == .enlarge { EmptyView() } else {
            Toggle("Safe mode (NSFW content filter)", isOn: $safeMode)
                .font(.caption)
                .help("On (default): generated images are screened by an on-device NSFW classifier and explicit results are blocked. Off: no filtering — you are responsible for the output.")

            // Rebalance scales the TAPPED text-encoder layers. A backend that
            // conditions on a single final hidden state has none to tap
            // (`condWeightCount == 0`), and the panel used to ask for
            // "Layer weights (0 numbers…)".
            if model.condWeightCount > 0 {
                Divider()
                Text("Conditioning rebalance").font(.caption.weight(.semibold))
                VStack(alignment: .leading, spacing: 2) {
                    Text("Global gain").font(.caption)
                    Stepper(value: $condGain, in: 0...4, step: 0.1) {
                        Text(String(format: "%.1f", condGain))
                    }
                    .onChange(of: condGain) { _, _ in guard !hydrating else { return }; persist() }
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Layer weights (\(model.condWeightCount) numbers, comma or space separated)")
                        .font(.caption)
                    TextField("", text: $condWeightsText, prompt: Text(defaultWeightsPlaceholder))
                        .textFieldStyle(.roundedBorder)
                        .font(.caption.monospaced())
                        .onChange(of: condWeightsText) { _, _ in guard !hydrating else { return }; persist() }
                    if !condWeightsValid {
                        Text("Needs exactly \(model.condWeightCount) numbers — one per tapped encoder layer.")
                            .font(.caption2)
                            .foregroundStyle(.red)
                    } else {
                        Text("Scales each tapped text-encoder layer's contribution (1 = neutral). Empty = off.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            // LoRA attaches to the DiT; a backend without that path answers 400.
            if model.supportsLoRA {
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
                Text("Apply one or more LoRA adapters to the image model for a custom style. Several can stack at once.")
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
            } // model.supportsLoRA
            } // image-model-only controls
        }
    }

    /// The verbs this model can serve, in picker order. `enlarge` is always
    /// among them — it runs a different model family, so it is not a
    /// capability of the image preset at all.
    private var availableVerbs: [ImageSourceVerb] { ImageSourceVerb.available(for: model) }

    /// What the pane is set up to do, or nil when there is no source image and
    /// the pane is plain text-to-image. Resolved against the model so a verb it
    /// cannot serve never reaches a request — the old `effectiveEditMode`
    /// rule, generalised to three verbs.
    private var effectiveVerb: ImageSourceVerb? {
        guard initImageURL != nil else { return nil }
        return ImageSourceVerb.resolve(sourceVerb, for: model)
    }

    /// Kept as its own name because the request field and several sections
    /// still ask exactly this question.
    private var effectiveEditMode: Bool { effectiveVerb == .edit }

    /// True when the pane is set up to edit a real source image — the only
    /// situation where "Match source" is a meaningful output size.
    private var isEditing: Bool {
        effectiveEditMode && initImageURL != nil
    }

    /// Placeholder showing the right count for the selected model's backend.
    private var defaultWeightsPlaceholder: String {
        Array(repeating: "1", count: model.condWeightCount).joined(separator: " ")
    }

    /// Empty = feature off = valid; otherwise it must parse to exactly the
    /// backend's tap count (the server 400s on a wrong count).
    private var condWeightsValid: Bool {
        let t = condWeightsText.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { return true }
        return ImageGenRequest.parseCondWeights(t)?.count == model.condWeightCount
    }

    /// `verb` is the one the picked photo should arrive on — the empty
    /// preview's "Enlarge a photo…" is a verb and a file panel in one gesture,
    /// not "attach something and then go find the mode".
    private func chooseSourceImage(verb: ImageSourceVerb? = nil) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        // Through AppActivation, never a raw `panel.runModal()`: this is an
        // accessory app, so a picker opened without bringing it forward first
        // comes up unfocused and swallows the click.
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            initImageURL = url
            if let verb { sourceVerb = verb }
            persist()
        }
    }

    private func chooseRefImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url, refImageURLs.count < maxRefImages {
            refImageURLs.append(url)
        }
    }

    /// Max simultaneously-attached LoRAs — mirrors the server's `lora.MAX_LORAS`.
    private let maxLoras = 8
    /// Reference images an edit takes beside the source.
    private let maxRefImages = 3

    /// Routing lives in `ImageDropPlacement` — one drop can carry several
    /// files, so this is the whole placement (source, then references) applied
    /// at once rather than a per-file decision.
    private func placeDroppedImages(_ urls: [URL]) {
        let placed = ImageDropPlacement.place(
            urls, source: initImageURL, editing: effectiveEditMode,
            refs: refImageURLs, refLimit: maxRefImages)
        initImageURL = placed.source
        refImageURLs = placed.refs
    }

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

    private func numberField(_ label: String, value: Binding<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            Stepper(value: value, step: step) {
                Text(String(value.wrappedValue))
            }
        }
    }

    /// The model the current verb actually runs. Enlarge is SeedVR2, not the
    /// image preset — so the download bar, the ready check and the RAM gate
    /// all have to ask the verb, not the pane.
    private var activeBundleReady: Bool {
        effectiveVerb == .enlarge
            ? (restoreLanModel != nil || downloads.bundleReady(restoreModel.bundle))
            : (lanModel != nil || downloads.bundleReady(model.bundle))
    }

    private var isRunning: Bool {
        effectiveVerb == .enlarge ? restore.isRunning : service.isRunning
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            // Progress only — the Download BUTTON lives on the model row above
            // (`MediaModelChooser`). Two buttons stacked here, one to fetch and
            // one to run, was the pane's most confusing moment.
            //
            // It follows the VERB, because picking Enlarge on a Mac that only
            // has FLUX means a model it hasn't got — and that has to read as an
            // answer to the choice just made, right under it.
            if effectiveVerb == .enlarge {
                if restoreLanModel == nil && !downloads.bundleReady(restoreModel.bundle) {
                    BundleDownloadBar(bundle: restoreModel.bundle, showsStartButton: false)
                }
            } else if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
            HStack {
                if isRunning {
                    Button(role: .destructive) {
                        if effectiveVerb == .enlarge { restore.cancel() } else { service.cancel() }
                    } label: {
                        Label("Cancel", systemImage: "stop.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else if effectiveVerb == .enlarge {
                    Button {
                        tryEnlarge()
                    } label: {
                        Label("Enlarge", systemImage: "arrow.up.left.and.arrow.down.right")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(!activeBundleReady)
                } else {
                    Button {
                        tryGenerate()
                    } label: {
                        Label("Generate", systemImage: "wand.and.stars")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !activeBundleReady || !condWeightsValid || !customSizeValid)
                }
            }
        }
    }

    /// One preview, two services. `ImagePanePreview` decides what it shows
    /// from the two phases plus the focus — never from the current verb, which
    /// is what used to blank a perfectly good picture the moment the controls
    /// changed.
    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.15))
            Group {
                switch ImagePanePreview.resolve(generate: generateRun,
                                                enlarge: enlargeRun,
                                                selected: selectedItem) {
                case .empty:
                    ContentUnavailableView {
                        Label("Nothing made yet", systemImage: "photo")
                    } description: {
                        Text("Write a prompt and press Generate — or bring in a photo you already have.")
                    } actions: {
                        // The other door. A photo from the camera roll has no
                        // result to press Enlarge on, and nobody looks for
                        // "I already have one" in a model picker.
                        Button("Enlarge a photo…") { chooseSourceImage(verb: .enlarge) }
                    }
                case .running(.generated, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(generateStep.step), total: max(1, Double(generateStep.total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                case .running(.enlarged, let message):
                    // No step events on this endpoint — one request, minutes of
                    // GPU — so an indeterminate spinner is the honest shape.
                    VStack(spacing: 12) {
                        ProgressView()
                        Text(message).font(.footnote).foregroundStyle(.secondary)
                    }
                case .result(let origin, let path):
                    completedPreview(path: path, origin: origin)
                case .failed(let origin, let msg):
                    ContentUnavailableView {
                        Label(origin == .enlarged ? "Enlarge failed" : "Generation failed",
                              systemImage: "exclamationmark.triangle")
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

    /// Everything this pane has made, newest first — both services' outputs on
    /// one timeline. Rebuilt from the two `recent` lists, which each survive a
    /// relaunch by scanning the output folders.
    private var sessionItems: [MediaSessionItem] {
        MediaSessionStrip.items(generated: service.recent, enlarged: restore.recent)
    }

    /// The selected row, or nil. A path that has left the strip (deleted
    /// elsewhere, folder cleared) selects nothing rather than drawing a
    /// preview of a file that is gone.
    private var selectedItem: MediaSessionItem? {
        guard let selectedPath else { return nil }
        return sessionItems.first { $0.path == selectedPath }
    }

    /// The two services' phases flattened for `ImagePanePreview`.
    private var generateRun: ImagePanePreview.Run {
        switch service.phase {
        case .idle: return .idle
        case .running(_, _, let message): return .running(message)
        case .completed(let path): return .done(path)
        case .failed(let msg): return .failed(msg)
        }
    }

    private var enlargeRun: ImagePanePreview.Run {
        switch restore.phase {
        case .idle: return .idle
        case .running(let message): return .running(message)
        case .completed(let path): return .done(path)
        case .failed(let msg): return .failed(msg)
        }
    }

    /// The determinate bar's numbers, which only the generate side has.
    private var generateStep: (step: Int, total: Int) {
        if case .running(let step, let total, _) = service.phase { return (step, total) }
        return (0, 1)
    }

    /// One result, one action set — whichever service produced it. The Upscale
    /// pane used to offer Reveal alone, so the same picture got different
    /// options depending on how it was made.
    private func completedPreview(path: String, origin: ImagePanePreview.Origin) -> some View {
        VStack(spacing: 8) {
            if let img = NSImage(contentsOfFile: path) {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFit()
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
                        path: path, prompt: prompt, kind: .image)
                } label: { Image(systemName: "bubble.left.and.text.bubble.right") }
                .buttonStyle(.borderless)
                .help("Send to Chat — opens a new conversation with this attached")
                // The verb belongs next to the picture it applies to. Without
                // it, enlarging what you are looking at means Reveal in Finder
                // or an NSOpenPanel aimed at the app's own output folder.
                Button { enlarge(path) } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                .buttonStyle(.borderless)
                .help(origin == .enlarged
                      ? "Enlarge again — run this result through SeedVR2 once more"
                      : "Enlarge — upscale and restore detail with SeedVR2")
            }
        }
        .padding(8)
    }

    // MARK: - Session strip

    /// Everything made here, as a film strip under the preview. A generated
    /// image used to exist on screen for exactly as long as the next one took
    /// to arrive; this is the pane's memory of its own session.
    @ViewBuilder
    private var sessionStrip: some View {
        let items = sessionItems
        if !items.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                // Lazy: a plain HStack builds all 60 rows on every body pass,
                // and each row reads a file.
                LazyHStack(spacing: 6) {
                    ForEach(items) { item in
                        stripTile(item)
                    }
                }
                .padding(.horizontal, 2)
                .padding(.vertical, 2)
            }
            .frame(height: 66)
        }
    }

    private func stripTile(_ item: MediaSessionItem) -> some View {
        let selected = item.path == selectedPath
        return Button {
            selectedPath = item.path
        } label: {
            ZStack(alignment: .bottomTrailing) {
                if let img = MediaThumbnails.cached(path: item.path) {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 56, height: 56)
                        .clipped()
                } else {
                    // The file went away under us — the tile stays so the row
                    // can still be selected and deleted rather than becoming
                    // an invisible gap.
                    RoundedRectangle(cornerRadius: 5).fill(Color.secondary.opacity(0.15))
                        .frame(width: 56, height: 56)
                        .overlay(Image(systemName: "questionmark").foregroundStyle(.secondary))
                }
                // An enlarged picture is marked, because at thumbnail size it
                // looks exactly like the one it came from.
                if item.origin == .enlarged {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.system(size: 8, weight: .bold))
                        .padding(3)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 4))
                        .padding(3)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 5))
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(selected ? Color.accentColor : Color.secondary.opacity(0.25),
                            lineWidth: selected ? 2 : 0.5)
            )
        }
        .buttonStyle(.plain)
        .help(item.filename)
        .contextMenu {
            Button("Use as Source") { useAsSource(item) }
            Button("Enlarge") { enlarge(item.path) }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: item.path)])
            }
            Divider()
            // Trash, never unlink: this is the user's own picture, and the
            // recoverable verb is the only one an app should offer for it.
            Button("Move to Trash", role: .destructive) { moveToTrash(item) }
        }
    }

    /// Put a strip row back into the controls as the source image, on whatever
    /// verb is currently selected — the strip's other job is being the input
    /// tray, not just the output shelf.
    private func useAsSource(_ item: MediaSessionItem) {
        switch ImageSourceHandoff.resolve(path: item.path, isRunning: isRunning) {
        case .accepted(let url):
            initImageURL = url
            refImageURLs = []
            persist()
        case .missing(let name):
            handoffError = "\(name) is no longer in the output folder."
        case .busy:
            handoffError = "A run is already going. Cancel it first, or wait for it to finish."
        }
    }

    /// Move a result to the Trash and drop it from the strip. The selection
    /// lands on the neighbour rather than nothing — clearing out a run of bad
    /// results is the reason this exists, and blanking the preview after every
    /// one makes the pane feel like it lost its place.
    private func moveToTrash(_ item: MediaSessionItem) {
        let next = MediaSessionStrip.selectionAfterDelete(sessionItems,
                                                          removing: item.path,
                                                          selected: selectedPath)
        do {
            try FileManager.default.trashItem(at: URL(fileURLWithPath: item.path),
                                              resultingItemURL: nil)
        } catch {
            deleteError = "Could not move \(item.filename) to the Trash: \(error.localizedDescription)"
            return
        }
        // The services own their own lists, and each also holds the path in a
        // `.completed` phase — both have to let go or the strip redraws the
        // row from memory and the preview keeps pointing at a trashed file.
        switch item.origin {
        case .generated: service.forget(path: item.path)
        case .enlarged: restore.forget(path: item.path)
        }
        MediaThumbnails.forget(path: item.path)
        selectedPath = next
    }

    /// SeedVR2 output always lands in `upscales/`, whatever it came from — one
    /// rule, so a result's folder never depends on how you got to it. The link
    /// follows the verb so it opens the folder the next run will write to.
    private var outputRoot: String {
        effectiveVerb == .enlarge ? MediaStorage.upscalesRoot : MediaStorage.imagesRoot
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting(
                [URL(fileURLWithPath: outputRoot)]
            )
        } label: {
            Label("Open output folder in Finder", systemImage: "folder")
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(outputRoot)
    }

    // MARK: - Sticky settings

    /// Seed `@State` from the last-used settings. Saved values win; resolution
    /// and quality are revalidated against the restored model so they stay
    /// in-range. Runs under `hydrating == true` so the `.onChange` cascade these
    /// writes trigger doesn't reapply preset defaults over them.
    private func hydrate() {
        let s = ImageGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        quality = s.quality
        resolution = s.resolvedResolution(for: model)
        steps = s.steps
        seed = s.seed
        safeMode = s.safeMode
        keepResident = s.keepResident
        strength = s.strength
        sourceVerb = s.sourceVerb
        condGain = s.condGain
        condWeightsText = s.condWeightsText
        loras = s.loras
        customWidthText = String(s.customWidth)
        customHeightText = String(s.customHeight)
        // A LoRA file may have moved since last session — drop stale entries.
        loras.removeAll { !FileManager.default.fileExists(atPath: $0.path) }
        // The enlarge side keeps its own model and scale; seed and
        // keep-resident are the pane's, above.
        let r = RestoreGenSettings.load()
        restoreModel = r.resolvedModel(models: server.allModels)
        restoreLanModel = LanPick.lanId(r.modelId)
        scale = r.scale
    }

    /// Capture the current controls as the new last-used settings.
    private func persist() {
        var s = ImageGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.quality = quality
        s.resolutionId = resolution.id
        // Persist what the fields HOLD, not the corrected value: rewriting the
        // user's own number under them mid-edit is the thing a hint exists to
        // avoid. Unparseable text keeps the previous saved size.
        s.customWidth = Int(customWidthText) ?? ImageGenSettings().customWidth
        s.customHeight = Int(customHeightText) ?? ImageGenSettings().customHeight
        s.steps = steps
        s.seed = seed
        s.safeMode = safeMode
        s.keepResident = keepResident
        s.strength = strength
        s.sourceVerb = sourceVerb
        s.condGain = condGain
        s.condWeightsText = condWeightsText
        s.loras = loras
        s.save()
        RestoreGenSettings(modelId: LanPick.persisted(lanModel: restoreLanModel, presetId: restoreModel.id),
                           scale: scale).save()
    }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        // Resolution menus are per-model (Mage-Flow offers 2048 and 4:1 shapes
        // FLUX doesn't), so a carried-over selection can be off-menu.
        resolution = model.validResolution(model.defaultResolution, editMode: isEditing)
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        steps = model.settings(quality).steps
    }

    /// Soft gate: only block if the model truly can't fit (needs more RAM
    /// than the Mac physically has) — and even then, just warn so the user
    /// can override. Available-RAM was misleading: macOS aggressively pages
    /// out idle apps under unified-memory pressure, so a "5 GB free" reading
    /// rarely means the system can't allocate the working set.
    private func tryGenerate() {
        let req = ImageGenRequest(
            model: model,
            prompt: prompt,
            seed: seed,
            width: effectiveSize.width,
            height: effectiveSize.height,
            steps: steps,
            keepResident: keepResident,
            lanModelId: lanModel,
            safeMode: safeMode,
            initImagePath: initImageURL?.path,
            strength: strength,
            editMode: effectiveEditMode,
            refImagePaths: effectiveEditMode ? refImageURLs.map(\.path) : [],
            condGain: condGain,
            condWeightsText: condWeightsText,
            loras: loras
        )
        persist()  // final capture — the agent's generate_image reuses these

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

    /// Hand a finished picture to the upscale controls: attach it as the
    /// source and switch. `ImageSourceHandoff` owns the refusals — the file
    /// can be gone, and a run already in flight owns its own source.
    private func enlarge(_ path: String) {
        switch ImageSourceHandoff.resolve(path: path, isRunning: appState.restoreGen.isRunning) {
        case .accepted(let url):
            initImageURL = url
            refImageURLs = []
            sourceVerb = .enlarge
            persist()
        case .missing(let name):
            handoffError = "\(name) is no longer in the output folder, so there is nothing to enlarge."
        case .busy:
            handoffError = "An enlarge is already running. Cancel it first, or wait for it to finish."
        }
    }

    /// Soft gate — same shape as `tryGenerate`, plus the one gate generation
    /// does not need: the TARGET canvas, not the model, is what makes a
    /// restore run out of memory, and the server only discovers that after
    /// the checkpoint has loaded, which is minutes. Say it here, while it is
    /// still one drag of the slider to fix.
    private func tryEnlarge() {
        guard let source = initImageURL else { return }
        persist()

        let total = RAMChecker.totalGB
        let needed = restoreModel.approxRAMGB
        if total < needed {
            ramWarningMessage = "This model needs about \(needed) GB of RAM, but your Mac has \(total) GB total. It may run very slowly or fail. Continue?"
            pendingEnlarge = source.path
            showRAMWarning = true
            return
        }
        if let (w, h) = sourcePixelSize {
            let t = scale > 1
                ? RestoreGeometry.upscaledTarget(width: w, height: h, factor: scale)
                : (width: RestoreGeometry.snap(w), height: RestoreGeometry.snap(h))
            if let msg = RestoreGeometry.memoryWarning(targetWidth: t.width, targetHeight: t.height,
                                                      modelGB: needed, totalRAMGB: total) {
                ramWarningMessage = msg
                pendingEnlarge = source.path
                showRAMWarning = true
                return
            }
        }
        startEnlarge(sourcePath: source.path)
    }

    private func startEnlarge(sourcePath: String) {
        restore.restore(sourcePath: sourcePath, model: restoreModel, lanModelId: restoreLanModel,
                        scale: scale, seed: seed, keepResident: keepResident, server: server)
    }

    private func showLogWindow() {
        let text = server.combinedGenLog(own: service.log)
        let alert = NSAlert()
        alert.messageText = "Image generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}
