import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// The Create/Enhance switch shared by both of the Image window's top-level
/// modes — one small segmented control, styled like every other section in
/// the pane (a subheadline label above a plain `.segmented` picker) rather
/// than floating in its own bar above the two-column layout. Living in the
/// LEFT column keeps it the same width as everything else it sits above.
struct ImagePaneModeSwitcher: View {
    @Binding var mode: ImagePaneMode

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("", selection: $mode) {
                ForEach(ImagePaneMode.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }
}

/// The Image Generation window's "Upscale" mode: restore/enlarge an existing
/// photo with SeedVR2, in place of the text-to-image controls. Same shell as
/// the other Create panes (source picker, model picker, action row, preview).
struct RestoreUpscaleView: View {
    @EnvironmentObject var service: RestoreService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    /// Shared with the Create side, so switching back is the same control.
    @Binding var mode: ImagePaneMode

    @State private var sourceURL: URL? = nil
    @State private var model: RestoreModelPreset = .seedvr2_3b
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    /// 1 = restore only (same resolution, sharper/cleaner). SeedVR2 itself
    /// has no built-in upscale — a factor above 1x is a bicubic resize to
    /// that target canvas BEFORE restoration, so the model fills in real
    /// detail at the larger size instead of the resize just looking soft.
    @State private var scale: Int = 2
    @State private var seed: Int = -1
    @State private var keepResident: Bool = false
    @State private var showAdvanced: Bool = false

    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: (path: String, model: RestoreModelPreset)? = nil
    /// Hydration guard — see ImageGenView for the full rationale.
    @State private var hydrating: Bool = false
    @State private var didHydrate: Bool = false
    @State private var isDropTargeted: Bool = false

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    ImagePaneModeSwitcher(mode: $mode)
                    sourceSection
                    scaleSection
                    modelSection
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
            .frame(minWidth: 280)
        }
        .onAppear {
            if !didHydrate {
                hydrating = true
                hydrate()
                didHydrate = true
                DispatchQueue.main.async { hydrating = false }
            }
            if server.status == .running { Task { await server.refreshModels() } }
        }
        .onChange(of: model) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: scale) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: seed) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        .alert("Model exceeds your Mac's RAM", isPresented: $showRAMWarning) {
            Button("Cancel", role: .cancel) { pendingRequest = nil }
            Button("Upscale Anyway", role: .destructive) {
                if let req = pendingRequest {
                    service.restore(sourcePath: req.path, model: req.model, lanModelId: lanModel,
                                    scale: scale, seed: seed, keepResident: keepResident, server: server)
                }
                pendingRequest = nil
            }
        } message: {
            Text(ramWarningMessage)
        }
    }

    // MARK: - Sections

    private var sourceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Photo to upscale").font(.subheadline.weight(.semibold))
            if let url = sourceURL {
                HStack(spacing: 8) {
                    if let img = NSImage(contentsOf: url) {
                        Image(nsImage: img)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 40, height: 40)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    Text(url.lastPathComponent)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button { sourceURL = nil } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless).foregroundStyle(.secondary).help("Remove photo")
                }
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
                if let note = cropNote {
                    Text(note)
                        .font(.caption2).foregroundStyle(.secondary)
                }
            } else {
                MediaDropWell(title: "Choose photo…",
                              systemImage: "wand.and.stars",
                              isTargeted: isDropTargeted) { choosePhoto() }
                Text("Enlarges a photo and restores real detail at the new size — not just a blurry resize.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .mediaDrop(.image, isTargeted: $isDropTargeted) { urls in
            if let url = urls.first { sourceURL = url }
        }
    }

    /// The source photo's pixel dimensions, or nil while nothing's picked /
    /// the file can't be decoded. Read once per source change by both
    /// `cropNote` and `scaleSection`'s target-size caption.
    private var sourcePixelSize: (width: Int, height: Int)? {
        guard let url = sourceURL, let img = NSImage(contentsOf: url),
              let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        return (cg.width, cg.height)
    }

    /// SeedVR2 needs both pixel dimensions divisible by 16 (`RestoreGeometry`)
    /// — told UP FRONT, before Upscale, rather than only discovered from the
    /// run log. nil when the photo's already on-grid, or when scaling up
    /// (a resize hits the target canvas exactly — no crop needed).
    private var cropNote: String? {
        guard scale == 1, let (w, h) = sourcePixelSize else { return nil }
        guard let crop = RestoreGeometry.centeredCrop(width: w, height: h) else { return nil }
        return "Will be center-cropped to \(crop.width) × \(crop.height) — SeedVR2 needs both dimensions divisible by 16."
    }

    private var scaleSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Scale").font(.subheadline.weight(.semibold))
            Picker("", selection: $scale) {
                Text("1× (restore only)").tag(1)
                Text("2×").tag(2)
                Text("4×").tag(4)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            if scale > 1, let (w, h) = sourcePixelSize {
                let t = RestoreGeometry.upscaledTarget(width: w, height: h, factor: scale)
                Text("\(w) × \(h) → \(t.width) × \(t.height)")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    /// Best-per-capability up front, everything else behind "Other Models",
    /// and the Download button ON the model — see `MediaModelChooser`.
    private var modelSection: some View {
        MediaModelChooser.pane(
            all: RestoreModelPreset.all,
            onThisMac: CustomMediaModels.restorePresets(from: server.allModels),
            capability: "restore",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.restorePreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist)
        .onChange(of: model) { _, _ in guard !hydrating else { return }; persist() }
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
            VStack(alignment: .leading, spacing: 2) {
                Text("Seed").font(.caption)
                TextField("Random", value: $seed, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
            }
            Toggle("Keep model loaded after upscaling", isOn: $keepResident)
                .font(.caption)
                .help("On: the model stays resident so the next upscale is instant. Off (default): it's unloaded to free GPU memory.")
        }
    }

    private var actionRow: some View {
        VStack(spacing: 8) {
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                BundleDownloadBar(bundle: model.bundle, showsStartButton: false)
            }
            HStack {
                if service.isRunning {
                    Button(role: .destructive) { service.cancel() } label: {
                        Label("Cancel", systemImage: "stop.circle").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button { tryUpscale() } label: {
                        Label("Upscale", systemImage: "wand.and.stars").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(sourceURL == nil || (lanModel == nil && !downloads.bundleReady(model.bundle)))
                }
            }
        }
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No result yet", systemImage: "wand.and.stars",
                                           description: Text("Choose a photo and press Upscale."))
                case .running(let message):
                    VStack(spacing: 12) {
                        ProgressView()
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
        VStack(spacing: 8) {
            if let img = NSImage(contentsOfFile: path) {
                Image(nsImage: img)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            HStack(spacing: 10) {
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
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.restoredRoot)])
        } label: {
            Label("Open output folder in Finder", systemImage: "folder").font(.caption)
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.restoredRoot)
    }

    // MARK: - Actions

    private func choosePhoto() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { sourceURL = url }
    }

    /// Soft gate — see ImageGenView.tryGenerate for the rationale.
    private func tryUpscale() {
        guard let sourceURL else { return }
        persist()

        let total = RAMChecker.totalGB
        let needed = model.approxRAMGB
        if total < needed {
            ramWarningMessage = "This model needs about \(needed) GB of RAM, but your Mac has \(total) GB total. It may run very slowly or fail. Continue?"
            pendingRequest = (sourceURL.path, model)
            showRAMWarning = true
            return
        }

        service.restore(sourcePath: sourceURL.path, model: model, lanModelId: lanModel,
                        scale: scale, seed: seed, keepResident: keepResident, server: server)
    }

    // MARK: - Persistence

    private func hydrate() {
        let s = RestoreGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        scale = s.scale
        seed = s.seed
        keepResident = s.keepResident
    }

    private func persist() {
        guard !hydrating else { return }
        RestoreGenSettings(modelId: model.id, scale: scale, seed: seed, keepResident: keepResident).save()
    }
}
