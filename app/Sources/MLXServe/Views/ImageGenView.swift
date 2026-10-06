import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Image generation window — native FLUX.2, Krea-2-Turbo and Mage-Flow (no
/// Python). The model picker lists every `ImageModelPreset`; the server
/// auto-routes to the right image backend by the model's `model_type`.
struct ImageGenView: View {
    @EnvironmentObject var service: ImageGenService
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    @State private var prompt: String = ""
    @State private var showEnhance = false
    /// Selection and focus are read by the image tiles: a click on one drops
    /// its name where the caret is, or on the end when the editor is not
    /// the one being typed into.
    @State private var promptSelection: TextSelection? = nil
    @FocusState private var promptFocused: Bool
    @State private var showAdvanced: Bool = false
    @State private var model: ImageModelPreset = .flux2Klein4B_Q4
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var quality: QualityPreset = .good
    // The canvas. Held as text, not Int: a size field has to be allowed to be
    // empty or half-typed while the user edits it, which a numeric binding
    // fights. The two fields are the one source of truth for the request; the
    // Presets menu only writes into them.
    @State private var customWidthText: String = "1024"
    @State private var customHeightText: String = "1024"
    @State private var promptHeight: Double = PromptEditorHeight.defaultHeight
    @State private var steps: Int = 8
    @State private var seed: Int = -1
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: ImageGenRequest? = nil
    /// Keep the model resident after generating (default off → unload to free
    /// GPU memory). On → the next generation reuses it instantly.
    @State private var keepResident: Bool = false
    /// Image-to-image source.
    @State private var initImageURL: URL? = nil
    /// Its pixel size, read from the file header when it is set.
    @State private var sourceSize: (width: Int, height: Int)? = nil
    /// Extra in-context references for edit mode (FLUX.2 multi-reference):
    /// "replace the face in image 1 with the face from image 2". The server
    /// takes at most 3 beside the source.
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
    /// Classifier-free guidance (Advanced, `model.supportsGuidance` only):
    /// how strongly to follow the prompt over the unconditional pathway.
    @State private var guidanceScale: Double = 1.0
    /// What to steer away from (Advanced, CFG only).
    @State private var negativePrompt: String = ""
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
    /// Whether the Quality segments fit the column; a menu below that.
    @State private var qualityFitsSegments: Bool = true
    /// A saved picture was gone on open and the rest renumbered, so the
    /// prompt's "image n" may now name another picture. Cleared by the first
    /// edit to either.
    @State private var refsDroppedOnHydrate: Bool = false

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
        // One persistence site for every sticky control: what is captured in
        // `stickySnapshot` is sticky by construction.
        .onChange(of: stickySnapshot) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: initImageURL) { _, url in
            sourceSize = url.flatMap { AspectCanvases.pixelSize(of: $0) }
            guard !hydrating, isEditing else { return }
            adoptSourceSize()
        }
        // An edit starts at the source's own size; the fields can still be
        // typed over afterwards.
        .onChange(of: isEditing) { _, editing in
            guard !hydrating, editing else { return }
            adoptSourceSize()
        }
        .onChange(of: prompt) { _, _ in guard !hydrating else { return }; refsDroppedOnHydrate = false }
        .onChange(of: attachedImages.wrappedValue) { _, _ in guard !hydrating else { return }; refsDroppedOnHydrate = false }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                // The model decides what the rest of the pane offers, so it is read first.
                VStack(alignment: .leading, spacing: 14) {
                    modelSection
                    promptSection
                    sourceImageSection
                    qualitySection
                    canvasSection
                    advancedSection
                    // Generate stands apart from the settings it acts on.
                    actionRow.padding(.top, 14)
                }
                // Full-width, leading-aligned frame OUTSIDE the padding — see
                // AudioGenView.
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
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
            Button(role: .cancel) { pendingRequest = nil } label: { Text("Cancel")
                .font(.app(.body)) }
            Button(role: .destructive) {
                if let req = pendingRequest { service.generate(req, server: server) }
                pendingRequest = nil
            } label: { Text("Generate Anyway")
                .font(.app(.body)) }
        } message: {
            Text(L10n.text(ramWarningMessage)).font(.app(.body))
        }
        .sheet(isPresented: $showEnhance) {
            PromptRewriteSheet(title: "Rewrite image prompt", request: { _ in PromptRewriter.image(text: prompt, editing: isEditing, groups: model.promptExamples(editing: isEditing)) }, onApply: { prompt = $0 })
                .environmentObject(appState)
        }
    }

    // MARK: - Sections

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text("Prompt").font(.app(.headline).weight(.semibold))
                Spacer()
                PromptEnhanceButton(disabled: prompt.isBlank) { showEnhance = true }
                templatesMenu
            }
            TextEditor(text: $prompt, selection: $promptSelection)
                .focused($promptFocused)
                .font(.app(.body))
                .frame(height: promptHeight)
                .overlay(
                    RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3), lineWidth: 0.5)
                )
            EditorResizeHandle(height: $promptHeight, onCommit: persist,
                               help: "Drag to resize the prompt box.")
        }
    }

    /// For an EDIT model this menu is the feature discovery surface: the
    /// repertoire is prompts, so an unlisted capability may as well not exist.
    /// One group lists flat; the edit repertoires have several, one submenu
    /// each.
    private var templatesMenu: some View {
        let groups = model.promptExamples(editing: isEditing)
        return Menu {
            if groups.count == 1, let group = groups.first {
                Section(L10n.text(group.name)) {
                    ForEach(group.examples, id: \.title) { ex in
                        Button(L10n.text(ex.title)) { prompt = ex.body }
                    }
                }
            } else {
                ForEach(groups, id: \.name) { group in
                    Menu(L10n.text(group.name)) {
                        ForEach(group.examples, id: \.title) { ex in
                            Button(L10n.text(ex.title)) { prompt = ex.body }
                        }
                    }
                }
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

    private var takesSourceImage: Bool { model.supportsReferenceEdit || model.supportsImg2Img }

    /// Hidden where the model takes no picture at all (Mage-Flow Turbo), the
    /// same rule as every other capability-gated control on this pane.
    @ViewBuilder
    private var sourceImageSection: some View {
        if takesSourceImage {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text("Source image(s)").font(.app(.subheadline).weight(.semibold))
                    if refsDroppedOnHydrate {
                        Image(systemName: "exclamationmark.circle.fill")
                            .font(.app(.caption))
                            .foregroundStyle(.orange)
                            .help("Some previously added references were not found on disk. Double-check the media identifiers in the prompt and adjust them if necessary.")
                    }
                    Text("optional")
                        .font(.app(.caption))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 8)
                    // The mode switch belongs to the SECTION: it sits beside
                    // the name of the thing it modifies. Only where BOTH modes
                    // exist, and only once there is a source for it to apply
                    // to.
                    if initImageURL != nil && model.supportsReferenceEdit && model.supportsImg2Img {
                        Picker("", selection: $editMode) {
                            Text("Edit").tag(true)
                            Text("Variation").tag(false)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .controlSize(.small)
                        .fixedSize()
                    }
                }
                if initImageURL == nil {
                    MediaDropWell(title: sourceImageButtonLabel,
                                  systemImage: "photo.badge.plus",
                                  isTargeted: isDropTargeted) { chooseSourceImage() }
                    Text(L10n.text(sourceImageHint))
                        .font(.app(.caption2))
                        .foregroundStyle(.secondary)
                } else if effectiveEditMode {
                    imagesPanel
                    if refImageURLs.isEmpty {
                        Text("Describe the change in the prompt — “make the hair blue”, “remove the monitor”. The model sees the original and keeps the rest.")
                            .font(.app(.caption2))
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Refer to the pictures by number — the source is image 1, references follow in order: “replace the face of the man in image 1 with the face from image 2”. Click a picture to drop its name into the prompt.")
                            .font(.app(.caption2))
                            .foregroundStyle(.secondary)
                    }
                } else if let url = initImageURL {
                    // One slot: the same filled well as a video keyframe, not a
                    // grid of one.
                    MediaDropWellFilled(isTargeted: isDropTargeted) {
                        HStack(spacing: 8) {
                            MediaImageView(url: url, maxPixel: 160)
                                .frame(width: 64, height: 48)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            Text(url.lastPathComponent)
                                .font(.app(.caption)).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            Button { initImageURL = nil; refImageURLs = [] } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.borderless).foregroundStyle(.secondary)
                            .help("Remove the source image (back to text-to-image)")
                        }
                    }
                    if model.supportsImg2Img {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Variation strength").font(.app(.caption))
                                Spacer()
                                Text(String(format: "%.0f%%", strength * 100))
                                    .font(.app(.caption))
                                    .foregroundStyle(.secondary)
                            }
                            Slider(value: $strength, in: 0.1...1.0, step: 0.05)
                            Text("Low = stay close to the source; high = mostly the prompt.")
                                .font(.app(.caption2))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            // Drops land on the section rather than the whole window, and the
            // section GROWS once a source is set, so the target covers the
            // tiles, which is exactly where a later drop is aimed.
            // `ImageDropPlacement` decides which slot it lands in, and states
            // the ROOM it has for one, so a pane with nothing left to fill
            // bounces the file instead of swallowing it.
            .mediaDrop(.image,
                       limit: imageRoom,
                       isTargeted: $isDropTargeted) { placeDroppedImages($0) }
        }
    }

    /// The source and the references as ONE list, which is how the prompt
    /// numbers them ("image 1" is the source). Removing a tile renumbers the
    /// rest, so taking image 1 away makes the next picture the source.
    private var attachedImages: Binding<[URL]> {
        Binding(
            get: {
                guard let source = initImageURL else { return [] }
                return [source] + (effectiveEditMode ? refImageURLs : [])
            },
            set: { urls in
                initImageURL = urls.first
                refImageURLs = Array(urls.dropFirst())
            })
    }

    private var imageRoom: Int {
        ImageDropPlacement.room(source: initImageURL, editing: effectiveEditMode,
                                refs: refImageURLs.count, refLimit: maxRefImages)
    }

    /// The tiles, and under them the way to add another while there is room.
    /// Same surface as the empty well, so a picked picture does not move the
    /// form.
    private var imagesPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            RefTileGrid(urls: attachedImages, label: { "image \($0 + 1)" }, kind: .image,
                        insert: insertMarker)
            if imageRoom > 0 {
                Button { chooseRefImage() } label: {
                    MediaWellAction(title: "Choose image…", systemImage: "photo.badge.plus",
                                    caption: "or drag one here")
                        .frame(maxWidth: .infinity)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(MediaDropWellBackground(isTargeted: isDropTargeted))
    }

    /// A tile's name goes where the caret is, or on the end when the editor
    /// is not focused. Shared helper with the Video pane's markers.
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
    /// the Download button ON the model — see `MediaModelChooser`. The transfer
    /// bar belongs to the model, not to the output, so it sits with it rather
    /// than beside Generate.
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            modelChooser
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
        }
    }

    private var modelChooser: some View {
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
            persist: persist,
            accessory: keepResidentToggle)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults() }
    }

    private var keepResidentToggle: AnyView {
        AnyView(
            Toggle(isOn: $keepResident) {
                Text("Keep model loaded after generating")
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
                .font(.app(.caption))
                .controlSize(.small)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
        )
    }

    @ViewBuilder
    private var qualitySection: some View {
        // A distilled model has ONE schedule. Offering tiers that only buy time
        // is the same silent-no-op the capability flags exist to kill.
        if model.stepsAreFixed {
            VStack(alignment: .leading, spacing: 6) {
                Text("Quality").font(.app(.headline).weight(.semibold))
                Text("Fixed at \(model.fixedSteps) steps — this model is distilled for a \(model.fixedSteps)-step schedule, so more steps cost time without adding detail.")
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            }
        } else {
            VStack(alignment: .leading, spacing: 6) {
                Text("Quality").font(.app(.rowTitle).weight(.semibold))
                // Measured, not `ViewThatFits`: the menu variant is `fixedSize`
                // and would never re-fit. Five segments degrade to a menu
                // rather than shortening the tier names the Create panes share.
                qualityPicker(segmented: qualityFitsSegments)
                Text(L10n.text(qualityHint))
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .onGeometryChange(for: Bool.self) { $0.size.width >= Self.qualitySegmentsMinWidth }
                action: { qualityFitsSegments = $0 }
        }
    }

    /// What the switcher shows. `custom` exists only while it is SELECTED, so
    /// it is never something to pick.
    private enum QualitySelection: Hashable {
        case preset(QualityPreset)
        case custom
    }

    /// The tier the live steps mean, or nil for Custom; `quality` is the last
    /// tier picked and only settles a tie (`ImageQualityMatch`).
    private var matchedQuality: QualityPreset? {
        ImageQualityMatch.match(steps: steps, model: model, preferring: quality)
    }

    /// The four tier names plus Custom at the segmented control's own
    /// padding; the Video pane's value.
    private static let qualitySegmentsMinWidth: CGFloat = 380

    /// Reads the DERIVED tier and writes by applying one.
    private var qualitySelection: Binding<QualitySelection> {
        Binding(
            get: { matchedQuality.map(QualitySelection.preset) ?? .custom },
            set: { sel in
                guard case .preset(let q) = sel else { return }
                quality = q
                applyQualityDefaults()
            })
    }

    @ViewBuilder
    private func qualityPicker(segmented: Bool) -> some View {
        let picker = Picker("", selection: qualitySelection) {
            ForEach(QualityPreset.allCases) { q in
                Text(L10n.text(q.label)).font(.app(.body)).tag(QualitySelection.preset(q))
            }
            if matchedQuality == nil {
                Text("Custom").font(.app(.body)).tag(QualitySelection.custom)
            }
        }
        .labelsHidden()
        if segmented {
            picker.pickerStyle(.segmented)
        } else {
            picker.pickerStyle(.menu).fixedSize()
        }
    }

    /// The steps the request will carry, so Custom reads its own number.
    private var qualityHint: String {
        L10n.format("%lld steps", Int64(steps))
    }

    /// The canvas. The two fields are the one source of truth for what the
    /// request carries; the Presets menu only writes into them. The server
    /// rewrites an off-grid image size, so the verdict under the fields says
    /// so BEFORE the request rather than leave the user reading an unexpected
    /// size off a finished image.
    private var canvasSection: some View {
        let verdict = customResolutionVerdict
        return VStack(alignment: .leading, spacing: 6) {
            // Bottom, not centre: the fields carry a heading above them, and
            // centring the row puts the button halfway up that heading.
            HStack(alignment: .bottom, spacing: 8) {
                canvasFields
                Spacer(minLength: 8)
                presetsMenu
            }
            if let hint = verdict.hint {
                Label(hint, systemImage: verdict.isValid ? "wand.and.stars" : "exclamationmark.triangle")
                    .font(.app(.caption2))
                    // A correction is information; a refusal is the reason
                    // Generate is disabled, so only that one is coloured.
                    .foregroundStyle(verdict.isValid ? Color.secondary : Color.orange)
            }
        }
    }

    private var canvasFields: some View {
        HStack(alignment: .bottom, spacing: 8) {
            labelledSizeField("Width", text: $customWidthText)
            // Centred on the fields, not on the pair of labels above them.
            Image(systemName: "multiply")
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .frame(height: 24)
            labelledSizeField("Height", text: $customHeightText)
        }
    }

    private func labelledSizeField(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            // A section heading like Prompt and Quality. `fixedSize` because a
            // squeezed HStack proposes less than its widest child and the TEXT
            // is what gives first.
            Text(L10n.text(title))
                .font(.app(.subheadline).weight(.semibold))
                .fixedSize()
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
                .frame(width: 80).font(.app(.body))
        }
    }

    /// The model's own curated sizes, grouped by orientation and largest
    /// first, then the sizes that match the source picture.
    private var presetsMenu: some View {
        Menu {
            ForEach([ResolutionOption.Orientation.landscape, .square, .portrait], id: \.self) { o in
                let rows = model.resolutions
                    .filter { $0.orientation == o }
                    .sorted { $0.width * $0.height > $1.width * $1.height }
                if !rows.isEmpty {
                    Section(orientationName(o)) {
                        ForEach(rows) { r in
                            Button(L10n.text(r.label)) { setCanvas(width: r.width, height: r.height) }
                        }
                    }
                }
            }
            Divider()
            sourceImageMenu
        } label: {
            HStack(spacing: 5) {
                Text("Presets")
                Image(systemName: "chevron.down")
            }
            // Body, not caption: this sits beside the size fields rather than
            // above a text box. The height is the fields' own.
            .font(.app(.body))
            .modifier(PaneChip(height: 24))
        }
        .modifier(PaneChipMenu())
        .help("Sizes this model ships with, and sizes that match the source image.")
    }

    /// Canvases matching the source picture's shape. An edit keeps the
    /// source's aspect at the requested budget; a variation is cover-cropped
    /// to the canvas, so a matching shape is what keeps its edges.
    @ViewBuilder
    private var sourceImageMenu: some View {
        let choices = sourceCanvases
        if initImageURL == nil || !takesSourceImage {
            // Disabled as a plain ITEM, not as a disabled submenu: a submenu
            // still opens on hover, and an empty one that opens reads as a
            // bug rather than as "pick a picture first".
            Button { } label: { Text("Set by source image…").font(.app(.body)) }
                .disabled(true)
        } else {
            Menu {
                if choices.isEmpty {
                    Button("The source image does not fit this model.") {}
                        .disabled(true)
                } else {
                    Section("Matching \(sourceRatio ?? "the source image")") {
                        ForEach(choices.filter(\.isSourceSize)) { choice in
                            Button(choiceLabel(choice)) {
                                setCanvas(width: choice.canvas.width, height: choice.canvas.height)
                            }
                        }
                        if choices.contains(where: \.isSourceSize) { Divider() }
                        ForEach(choices.filter { !$0.isSourceSize }) { choice in
                            Button(choiceLabel(choice)) {
                                setCanvas(width: choice.canvas.width, height: choice.canvas.height)
                            }
                        }
                    }
                }
            } label: {
                Text("Set by source image…")
            }
        }
    }

    private func orientationName(_ o: ResolutionOption.Orientation) -> String {
        switch o {
        case .landscape: return "Landscape"
        case .square:    return "Square"
        case .portrait:  return "Portrait"
        }
    }

    /// No ratio per row: every row has the same one, and the section heading
    /// above them already says which.
    private func choiceLabel(_ choice: SourceCanvasChoice) -> String {
        var out = "\(choice.canvas.width) × \(choice.canvas.height)"
        if let name = choice.name { out += " - \(name)" }
        return out
    }

    private var sourceRatio: String? {
        guard let size = sourceSize else { return nil }
        return AspectCanvases.ratioLabel(width: size.width, height: size.height)
    }

    private var sourceCanvases: [SourceCanvasChoice] {
        guard let size = sourceSize else { return [] }
        return AspectCanvases.choices(sourceWidth: size.width, sourceHeight: size.height,
                                      grid: model.resolutionGrid)
    }

    /// Written into the fields, over a focused one too: the user picked a size
    /// from a menu, so the box has to show it.
    private func setCanvas(width: Int, height: Int) {
        customWidthText = String(width)
        customHeightText = String(height)
    }

    /// The source's own size on the grid, when it fits; otherwise the fields
    /// keep what they have.
    private func adoptSourceSize() {
        guard let size = sourceSize,
              let own = AspectCanvases.sourceSize(sourceWidth: size.width, sourceHeight: size.height,
                                                  grid: model.resolutionGrid) else { return }
        setCanvas(width: own.width, height: own.height)
    }

    /// Generate is gated on the verdict, never on a stale unparseable value.
    private var customSizeValid: Bool { customResolutionVerdict.isValid }

    /// What the selected model's grid makes of the typed size. Non-numeric or
    /// empty text reads as 0, which the grid already refuses by name.
    private var customResolutionVerdict: CustomResolution {
        model.resolutionGrid.resolve(width: Int(customWidthText) ?? 0,
                                     height: Int(customHeightText) ?? 0)
    }

    /// The size the request should carry: the grid's corrected numbers.
    /// Generate is gated on the same verdict, so the fallback never ships.
    private var effectiveSize: (width: Int, height: Int) {
        customResolutionVerdict.size ?? (model.defaultResolution.width, model.defaultResolution.height)
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            FoldingSectionHeader(title: "Advanced options", isExpanded: $showAdvanced)
            if showAdvanced {
                // Steps stay overridable even where the schedule is fixed —
                // it's the Advanced panel, and the hint says the cost.
                intSliderRow("Steps", value: $steps, range: 1...50)
                if model.stepsAreFixed {
                    Text("This model is distilled for \(model.fixedSteps) steps; other values cost time without adding detail.")
                        .font(.app(.caption2))
                        .foregroundStyle(.secondary)
                }

                // Real CFG — the undistilled base checkpoint only. Every other
                // preset has guidance baked into its weights, so the field
                // would be pure decoration there and stays hidden.
                if model.supportsGuidance {
                    Text("Classifier-free guidance").font(.app(.caption).weight(.semibold))
                    sliderRow("Guidance scale", value: $guidanceScale, range: 1...20, step: 0.5)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Negative prompt").font(.app(.caption))
                        TextField("", text: $negativePrompt, prompt: Text("what to steer away from (optional)"))
                            .textFieldStyle(.roundedBorder)
                            .font(.app(.caption))
                    }
                }
                // -1 is the random sentinel and renders as an EMPTY box, so the
                // placeholder explains it instead of a literal -1 that reads as
                // a broken value.
                SeedField(label: "Seed", placeholder: "random", range: -1...Int.max, value: $seed,
                          help: "Same seed + same settings reproduces the image. Paste one to rerun someone else's; leave it empty for a new one each time.")
                // Rebalance scales the TAPPED text-encoder layers. A backend
                // that conditions on a single final hidden state has none to
                // tap (`condWeightCount == 0`), and the panel used to ask for
                // "Layer weights (0 numbers…)".
                if model.condWeightCount > 0 {
                    Divider()
                    Text("Conditioning rebalance").font(.app(.caption).weight(.semibold))
                    sliderRow("Global gain", value: $condGain, range: 0...4, step: 0.1)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Layer weights (\(model.condWeightCount) numbers, comma or space separated)")
                            .font(.app(.caption))
                        TextField("", text: $condWeightsText, prompt: Text(defaultWeightsPlaceholder))
                            .textFieldStyle(.roundedBorder)
                            .font(.app(.caption).monospaced())
                        if !condWeightsValid {
                            Text("Needs exactly \(model.condWeightCount) numbers — one per tapped encoder layer.")
                                .font(.app(.caption2))
                                .foregroundStyle(.red)
                        } else {
                            Text("Scales each tapped text-encoder layer's contribution (1 = neutral). Empty = off.")
                                .font(.app(.caption2))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                // LoRA attaches to the DiT; a backend without that path answers 400.
                if model.supportsLoRA { loraSection }
            }
        }
    }

    private var loraSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Divider()
            Text("Style LoRAs").font(.app(.caption).weight(.semibold))
            ForEach(Array(loras.enumerated()), id: \.element.id) { index, _ in
                LoraAdapterRow(lora: $loras[index]) { loras.remove(at: index) }
            }
            // The way in is the well itself, and it comes back under the last
            // adapter so adding a second one needs no separate control. At the
            // cap there is nothing to offer, so it goes.
            if loras.count < maxLoras { LoraAddWell(action: chooseLora) }
        }
    }

    /// Labeled slider for a `Double` setting, the value read out on the right.
    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>,
                           step: Double) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.caption))
                Spacer()
                Text(String(format: "%.1f", value.wrappedValue))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
                .padding(.top, Self.steppedSliderTrackDrop)
        }
    }

    /// Labeled slider for an `Int` setting (bridges to a `Double` slider).
    private func intSliderRow(_ label: String, value: Binding<Int>, range: ClosedRange<Int>) -> some View {
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
    }

    /// A stepped slider reserves a tick row under its track, so the track sits
    /// glued to the caption above; 3pt, the Video pane's value.
    private static let steppedSliderTrackDrop: CGFloat = 3

    /// Edit mode only applies where the model was trained for it; on models
    /// without that training a source image always means variation. And where
    /// editing is the ONLY thing a source image can do (no img2img path), a
    /// source image means edit regardless of the toggle — the mode picker is
    /// hidden in that case, so a stale `false` would otherwise send a variation
    /// request the backend rejects.
    private var effectiveEditMode: Bool {
        model.supportsReferenceEdit && (editMode || !model.supportsImg2Img)
    }

    /// True when the pane is set up to edit a real source image: the canvas
    /// then starts from the source's own size, and the templates offer the
    /// edit repertoire.
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
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            initImageURL = url
        }
    }

    /// Adds through the same placement a drop uses: the empty source first,
    /// then references while editing.
    private func chooseRefImage() {
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        if AppActivation.runModal(panel) == .OK {
            placeDroppedImages(Array(panel.urls.prefix(imageRoom)))
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
        }
    }

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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (lanModel == nil && !downloads.bundleReady(model.bundle)) || !condWeightsValid || !customSizeValid)
                }
            }
            .font(.app(.callout))
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No generation yet", systemImage: "photo", description: Text("Enter a prompt and press Generate.").font(.app(.body)))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear)
                            .frame(width: 240)
                        Text(message).font(.app(.footnote)).foregroundStyle(.secondary)
                    }
                case .completed(let path):
                    completedPreview(path: path)
                case .failed(let msg):
                    ContentUnavailableView {
                        Label("Failed", systemImage: "exclamationmark.triangle").font(.app(.body))
                    } description: {
                        Text(msg)
                    } actions: {
                        Button { showLogWindow() } label: { Text("Show log")
                            .font(.app(.body)) }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func completedPreview(path: String) -> some View {
        VStack(spacing: 8) {
            MediaImageView(url: URL(fileURLWithPath: path), maxPixel: 3072,
                           contentMode: .fit)
            // The name and the ways to reach the file belong together, centred
            // under the picture they describe.
            HStack(spacing: 8) {
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
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
                [URL(fileURLWithPath: MediaStorage.imagesRoot)]
            )
        } label: {
            Label("Open output folder in Finder", systemImage: "folder")
                .font(.app(.caption))
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.imagesRoot)
    }

    // MARK: - Sticky settings

    /// Seed `@State` from the last-used settings. Saved values win; quality is
    /// revalidated against the restored model so it stays in-range. Runs under
    /// `hydrating == true` so the `.onChange` cascade these writes trigger
    /// doesn't reapply preset defaults over them.
    private func hydrate() {
        let s = ImageGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        quality = s.quality
        // The fields are the canvas. A blob that stored a PRESET row rather
        // than a typed size opens on that row's numbers, so nobody's saved
        // canvas changes under them.
        let saved = s.resolvedResolution(for: model)
        if saved.isCustom {
            customWidthText = String(s.customWidth)
            customHeightText = String(s.customHeight)
        } else {
            customWidthText = String(saved.width)
            customHeightText = String(saved.height)
        }
        steps = s.steps
        seed = s.seed
        keepResident = s.keepResident
        strength = s.strength
        editMode = s.editMode
        condGain = s.condGain
        condWeightsText = s.condWeightsText
        guidanceScale = s.guidanceScale
        loras = s.loras
        prompt = s.prompt
        negativePrompt = s.negativePrompt
        promptHeight = s.promptHeight
        showAdvanced = s.showAdvanced
        // A file may have moved since last session — drop stale entries.
        loras.removeAll { !FileManager.default.fileExists(atPath: $0.path) }
        let images = ImageDraftImages.restore(sourcePath: s.sourcePath, refPaths: s.refPaths,
                                              exists: { FileManager.default.fileExists(atPath: $0) })
        initImageURL = images.source
        refImageURLs = images.refs
        refsDroppedOnHydrate = images.dropped
        sourceSize = images.source.flatMap { AspectCanvases.pixelSize(of: $0) }
        // The saved canvas of an edit is the model's default (see
        // `stickySnapshot`); the edit itself runs at the source's size.
        if isEditing { adoptSourceSize() }
    }

    /// The current controls as the settings blob. Read by the root `.onChange`
    /// to persist, and by `persist()` to save.
    private var stickySnapshot: ImageGenSettings {
        var s = ImageGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.quality = quality
        // The typed pair IS the canvas, saved as the Custom sentinel that every
        // reader wanting numbers (the chat's `generate_image` among them)
        // resolves through `customWidth`/`customHeight`. While a source is
        // being edited the fields hold that picture's size, which is no
        // default for a text-to-image request elsewhere, so the saved id is
        // the model's default bucket and `hydrate` re-adopts the source.
        s.resolutionId = isEditing ? model.defaultResolution.id : ResolutionOption.custom.id
        // Persist what the fields HOLD, not the corrected value: rewriting the
        // user's own number under them mid-edit is the thing a hint exists to
        // avoid. Unparseable text saves the default size.
        s.customWidth = Int(customWidthText) ?? ImageGenSettings().customWidth
        s.customHeight = Int(customHeightText) ?? ImageGenSettings().customHeight
        s.steps = steps
        s.seed = seed
        s.keepResident = keepResident
        s.strength = strength
        s.editMode = editMode
        s.condGain = condGain
        s.condWeightsText = condWeightsText
        s.guidanceScale = guidanceScale
        s.loras = loras
        s.prompt = prompt
        s.negativePrompt = negativePrompt
        s.promptHeight = promptHeight
        s.showAdvanced = showAdvanced
        s.sourcePath = initImageURL?.path
        s.refPaths = refImageURLs.map(\.path)
        return s
    }

    private func persist() { stickySnapshot.save() }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        // Grids are per-model (Mage-Flow offers 2048 and 4:1 shapes FLUX
        // doesn't), so the canvas restarts from the model's default; an edit
        // in progress keeps following its source.
        if isEditing, sourceSize != nil {
            adoptSourceSize()
        } else {
            setCanvas(width: model.defaultResolution.width, height: model.defaultResolution.height)
        }
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
            // A picture kept from another model stays in the draft but is not
            // sent where the backend takes none (the section is hidden there).
            initImagePath: takesSourceImage ? initImageURL?.path : nil,
            strength: strength,
            editMode: effectiveEditMode,
            refImagePaths: takesSourceImage && effectiveEditMode ? refImageURLs.map(\.path) : [],
            condGain: condGain,
            condWeightsText: condWeightsText,
            // A stack saved on a LoRA-capable model does not ride into one
            // that answers 400 to it (the section is hidden there).
            loras: model.supportsLoRA ? loras : [],
            guidanceScale: model.supportsGuidance ? guidanceScale : 1.0,
            negativePrompt: model.supportsGuidance ? negativePrompt : ""
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
        AppActivation.openWindow(id: "serverLog", using: openWindow)
    }
}
