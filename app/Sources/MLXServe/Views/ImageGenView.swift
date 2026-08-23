import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Image generation window — native FLUX.2, Krea-2-Turbo and Mage-Flow (no
/// Python). The model picker lists every `ImageModelPreset`; the server
/// auto-routes to the right image backend by the model's `model_type`.
struct ImageGenView: View {
    @EnvironmentObject var service: ImageGenService
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
    /// Keep the model resident after generating (default off → unload to free
    /// GPU memory). On → the next generation reuses it instantly.
    @State private var keepResident: Bool = false
    /// Image-to-image source (transient — not persisted, like video's first frame).
    @State private var initImageURL: URL? = nil
    /// Extra in-context references for edit mode (FLUX.2 multi-reference):
    /// "replace the face in image 1 with the face from image 2". Transient,
    /// like the source. The server takes at most 3 beside the source.
    @State private var refImageURLs: [URL] = []
    /// img2img renoise strength: low = stay close to the source, high = mostly prompt.
    @State private var strength: Double = 0.6
    /// Source-image mode: true = instruction edit (FLUX.2 in-context reference,
    /// keeps the subject), false = variation (renoise remix).
    @State private var editMode: Bool = true
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

    var body: some View {
        // No window-sized floor: this is a PAGE of the chat window now, and a
        // root minimum wider than the detail column overflows it and clips
        // both edges. Small windows shrink the preview side instead.
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
        }
        // Persist every other sticky field on change (model/quality persist in
        // their sections after applying preset defaults).
        .onChange(of: resolution) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: customWidthText) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: customHeightText) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    promptSection
                    sourceImageSection
                    modelSection
                    qualitySection
                    resolutionSection
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
            // The preview is what gives way in a small window — the generated
            // image scales to fit; the controls column keeps its form floor.
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
                if initImageURL != nil && model.supportsReferenceEdit && model.supportsImg2Img {
                    Picker("", selection: $editMode) {
                        Text("Edit").tag(true)
                        Text("Variation").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .controlSize(.small)
                    .fixedSize()
                    .onChange(of: editMode) { _, _ in guard !hydrating else { return }; persist() }
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
                if effectiveEditMode {
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

    /// What a source image is FOR on this model — instruction editing, a
    /// renoise variation, or both. Never offers a mode the backend 400s on.
    private var sourceImageButtonLabel: String {
        model.supportsImg2Img ? "Choose image…" : "Choose image to edit…"
    }

    private var sourceImageHint: String {
        switch (model.supportsReferenceEdit, model.supportsImg2Img) {
        case (true, true):
            return "Edit an existing image with an instruction, or generate a variation of it."
        case (true, false):
            return "Edit an existing image with an instruction — say what to change and the rest stays put."
        default:
            return "Generate a variation of an existing image, guided by the prompt (image-to-image)."
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
                numberField("Steps", value: $steps, step: 1)
                // -1 is the random sentinel and renders as an EMPTY box, so the
                // placeholder explains it instead of a literal -1 that reads as
                // a broken value.
                SeedField(label: "Seed", placeholder: "random", range: -1...Int.max, value: $seed,
                          help: "Same seed + same settings reproduces the image. Paste one to rerun someone else's; leave it empty for a new one each time.")
            }
            if model.stepsAreFixed {
                Text("This model is distilled for \(model.fixedSteps) steps; other values cost time without adding detail.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")

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
        }
    }

    /// Edit mode only applies where the model was trained for it; on models
    /// without that training a source image always means variation. And where
    /// editing is the ONLY thing a source image can do (no img2img path), a
    /// source image means edit regardless of the toggle — the mode picker is
    /// hidden in that case, so a stale `false` would otherwise send a variation
    /// request the backend rejects.
    private var effectiveEditMode: Bool {
        model.supportsReferenceEdit && (editMode || !model.supportsImg2Img)
    }

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

    private func chooseSourceImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            initImageURL = url
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

    private var actionRow: some View {
        VStack(spacing: 8) {
            // Progress only — the Download BUTTON lives on the model row above
            // (`MediaModelChooser`). Two buttons stacked here, one to fetch and
            // one to run, was the pane's most confusing moment.
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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (lanModel == nil && !downloads.bundleReady(model.bundle)) || !condWeightsValid || !customSizeValid)
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
                    ContentUnavailableView("No generation yet", systemImage: "photo", description: Text("Enter a prompt and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
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
            }
        }
        .padding(8)
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting(
                [URL(fileURLWithPath: MediaStorage.imagesRoot)]
            )
        } label: {
            Label("Open output folder in Finder", systemImage: "folder")
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.imagesRoot)
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
        keepResident = s.keepResident
        strength = s.strength
        editMode = s.editMode
        condGain = s.condGain
        condWeightsText = s.condWeightsText
        loras = s.loras
        customWidthText = String(s.customWidth)
        customHeightText = String(s.customHeight)
        // A LoRA file may have moved since last session — drop stale entries.
        loras.removeAll { !FileManager.default.fileExists(atPath: $0.path) }
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
        s.keepResident = keepResident
        s.strength = strength
        s.editMode = editMode
        s.condGain = condGain
        s.condWeightsText = condWeightsText
        s.loras = loras
        s.save()
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

    private func showLogWindow() {
        let text = server.combinedGenLog(own: service.log)
        let alert = NSAlert()
        alert.messageText = "Image generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}
