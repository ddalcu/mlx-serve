import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Image generation window — native FLUX.2 and Krea-2-Turbo (no Python).
/// The model picker lists every `ImageModelPreset`; the server auto-routes to
/// the right image backend by the model's `config.json` `model_type`.
///
/// UI layering: a Quality picker drives steps + CFG, a Resolution picker
/// pins to model-trained buckets, and Advanced lets the user override
/// individual fields if they really want to.
struct ImageGenView: View {
    @EnvironmentObject var service: ImageGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    @State private var prompt: String = ""
    @State private var negative: String = ""
    @State private var showAdvanced: Bool = false
    @State private var model: ImageModelPreset = .flux2Klein4B_Q4
    @State private var quality: QualityPreset = .good
    @State private var resolution: ResolutionOption = ImageModelPreset.flux2Klein4B_Q4.defaultResolution
    @State private var steps: Int = 8
    @State private var guidance: Double = 0.5
    @State private var seed: Int = -1
    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: ImageGenRequest? = nil
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
    /// Source-image mode: true = instruction edit (FLUX.2 in-context reference,
    /// keeps the subject), false = variation (renoise remix).
    @State private var editMode: Bool = true
    /// Conditioning rebalance (Advanced): global gain on the prompt embeddings.
    @State private var condGain: Double = 1.0
    /// Conditioning rebalance (Advanced): per-tapped-layer weights as typed.
    @State private var condWeightsText: String = ""
    /// Style LoRA (Advanced): .safetensors adapter path ("" = none).
    @State private var loraPath: String = ""
    /// True while `hydrate()` seeds `@State` from saved settings. Hydrating
    /// `model`/`quality` fires their `.onChange` (applyModelDefaults /
    /// applyQualityDefaults) which would clobber the just-restored
    /// steps/guidance/resolution — so every reset + persist is guarded on this.
    @State private var hydrating: Bool = false
    /// Hydrate exactly once per window lifetime (the first `.onAppear`).
    @State private var didHydrate: Bool = false

    var body: some View {
        readyView
        .frame(minWidth: 880, minHeight: 640)
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                // Clear on the next runloop tick so the cascade of `.onChange`
                // fired by hydration's state writes is ignored.
                DispatchQueue.main.async { hydrating = false }
            }
            downloads.ensureNsfwClassifier() // best-effort: provision the shared content filter
        }
        // Persist every other sticky field on change (model/quality persist in
        // their sections after applying preset defaults).
        .onChange(of: resolution) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: guidance) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: negative) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: safeMode) { _, _ in guard !hydrating else { return }; persist() }
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
            Text("Prompt").font(.subheadline.weight(.semibold))
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
            Text("Source image (optional)").font(.subheadline.weight(.semibold))
            if let url = initImageURL {
                HStack(spacing: 8) {
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
                    Button {
                        initImageURL = nil
                        refImageURLs = []
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless)
                    .foregroundStyle(.secondary)
                    .help("Remove the source image (back to text-to-image)")
                }
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
                if model.supportsReferenceEdit {
                    Picker("", selection: $editMode) {
                        Text("Edit").tag(true)
                        Text("Variation").tag(false)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .onChange(of: editMode) { _, _ in guard !hydrating else { return }; persist() }
                }
                if effectiveEditMode {
                    ForEach(refImageURLs, id: \.self) { ref in
                        HStack(spacing: 8) {
                            if let img = NSImage(contentsOf: ref) {
                                Image(nsImage: img)
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 40, height: 40)
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                            Text(ref.lastPathComponent)
                                .font(.caption)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer()
                            Button {
                                refImageURLs.removeAll { $0 == ref }
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.borderless)
                            .foregroundStyle(.secondary)
                            .help("Remove this reference image")
                        }
                        .padding(6)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
                    }
                    if refImageURLs.count < 3 {
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
                } else {
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
                Button {
                    chooseSourceImage()
                } label: {
                    Label("Choose image…", systemImage: "photo.badge.plus")
                        .font(.caption)
                }
                Text("Generate a variation of an existing image, guided by the prompt (image-to-image).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(ImageModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .onChange(of: model) { _, _ in guard !hydrating else { return }; applyModelDefaults(); persist() }
            Text("~\(model.approxRAMGB) GB RAM")
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

    /// Tier-specific marketing-ish hint with the actual numbers, so users see
    /// the cost up front (e.g. "Super = 50 steps, CFG 4.5").
    private var qualityHint: String {
        let s = model.settings(quality)
        return "\(s.steps) steps, CFG \(String(format: "%.1f", s.guidance))"
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
        }
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
            HStack {
                numberField("Steps", value: $steps, step: 1)
                guidanceField
            }
            HStack {
                numberField("Seed (-1 = random)", value: $seed, step: 1)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Negative prompt").font(.caption)
                TextField("", text: $negative, prompt: Text("(optional, ignored by Klein)"))
                    .textFieldStyle(.roundedBorder)
            }
            Toggle("Keep model loaded after generating", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next generation is instant. Off (default): it's unloaded to free GPU memory.")
            Toggle("Safe mode (NSFW content filter)", isOn: $safeMode)
                .font(.caption)
                .help("On (default): generated images are screened by an on-device NSFW classifier and explicit results are blocked. Off: no filtering — you are responsible for the output.")

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

            Divider()
            Text("Style LoRA").font(.caption.weight(.semibold))
            if loraPath.isEmpty {
                Button {
                    chooseLora()
                } label: {
                    Label("Choose .safetensors…", systemImage: "paintpalette")
                        .font(.caption)
                }
                Text("Apply a LoRA adapter to the image model for a custom style.")
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

    /// Edit mode only applies where the model was trained for it (FLUX.2);
    /// on other models the source image always means variation.
    private var effectiveEditMode: Bool {
        editMode && model.supportsReferenceEdit
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
        if AppActivation.runModal(panel) == .OK, let url = panel.url, refImageURLs.count < 3 {
            refImageURLs.append(url)
        }
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

    private func numberField(_ label: String, value: Binding<Int>, step: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption)
            Stepper(value: value, step: step) {
                Text(String(value.wrappedValue))
            }
        }
    }

    private var guidanceField: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("CFG / Guidance").font(.caption)
            Stepper(value: $guidance, in: 0...20, step: 0.5) {
                Text(String(format: "%.1f", guidance))
            }
        }
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
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !downloads.bundleReady(model.bundle) || !condWeightsValid)
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
        model = s.resolvedModel
        quality = s.quality
        resolution = s.resolvedResolution(for: model)
        steps = s.steps
        guidance = s.guidance
        seed = s.seed
        negative = s.negativePrompt
        safeMode = s.safeMode
        keepResident = s.keepResident
        strength = s.strength
        editMode = s.editMode
        condGain = s.condGain
        condWeightsText = s.condWeightsText
        loraPath = s.loraPath
        // The LoRA file may have moved since last session — drop a stale path.
        if !loraPath.isEmpty && !FileManager.default.fileExists(atPath: loraPath) {
            loraPath = ""
        }
    }

    /// Capture the current controls as the new last-used settings.
    private func persist() {
        var s = ImageGenSettings()
        s.modelId = model.id
        s.quality = quality
        s.resolutionId = resolution.id
        s.steps = steps
        s.guidance = guidance
        s.seed = seed
        s.negativePrompt = negative
        s.safeMode = safeMode
        s.keepResident = keepResident
        s.strength = strength
        s.editMode = editMode
        s.condGain = condGain
        s.condWeightsText = condWeightsText
        s.loraPath = loraPath
        s.save()
    }

    // MARK: - Actions

    private func applyModelDefaults() {
        quality = model.defaultQuality
        resolution = model.defaultResolution
        applyQualityDefaults()
    }

    private func applyQualityDefaults() {
        let s = model.settings(quality)
        steps = s.steps
        guidance = s.guidance
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
            negativePrompt: negative,
            seed: seed,
            width: resolution.width,
            height: resolution.height,
            steps: steps,
            guidance: guidance,
            keepResident: keepResident,
            safeMode: safeMode,
            initImagePath: initImageURL?.path,
            strength: strength,
            editMode: effectiveEditMode,
            refImagePaths: effectiveEditMode ? refImageURLs.map(\.path) : [],
            condGain: condGain,
            condWeightsText: condWeightsText,
            loraPath: loraPath.isEmpty ? nil : loraPath
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
        let text = service.log.joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "Image generation log"
        alert.informativeText = text.isEmpty ? "(no output)" : text
        alert.runModal()
    }
}
