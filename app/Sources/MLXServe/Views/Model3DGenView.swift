import SwiftUI
import AppKit
import SceneKit
import ModelIO
import SceneKit.ModelIO
import UniformTypeIdentifiers

/// 3D generation window — single photo → textured mesh, run natively by the
/// embedded mlx-serve server (Hunyuan3D 2.1 shape stage). Same shell as
/// ImageGen/VideoGen/AudioGen: a model picker, a photo well, an Advanced
/// disclosure (steps / guidance / mesh resolution), and a SceneKit preview of
/// the result with a turntable "Animate" toggle.
struct Model3DGenView: View {
    @EnvironmentObject var service: Model3DGenService
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject var downloads: DownloadManager
    /// For the model row's Download button (`MediaModelChooser`) — a completed
    /// transfer has to re-scan the models directory.
    @EnvironmentObject var appState: AppState

    @State private var photoURL: URL? = nil
    @State private var model: Model3DModelPreset = .recommended
    /// Selected network model's routing id (`<model>@<peer>`); nil = local.
    @State private var lanModel: String? = nil
    @State private var steps: Int = 30
    @State private var guidance: Double = 5.0
    @State private var resolution: Int = 384
    @State private var keepResident: Bool = false
    @State private var turntable: Bool = true
    @State private var texture: Bool = false
    @State private var trellisResolution: Int = 1024
    @State private var splatSteps: Int = 50
    @State private var splatGuidance: Double = 3.0
    @State private var splatGaussians: Int = 262_144
    @State private var splatPhoto: Model3DPhotoKind = .render
    @State private var decimationTarget: Int = 300_000
    @State private var textureSize: Int = 2048
    @State private var finish: Model3DFinish = .model
    @State private var showAdvanced: Bool = false

    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: Model3DGenRequest? = nil
    /// Hydration guard — see ImageGenView for the full rationale.
    @State private var hydrating: Bool = false
    @State private var didHydrate: Bool = false
    /// True while a drag carrying a file hovers the photo section — drives its
    /// dashed-border highlight and the well's fill (see `MediaDropTarget`).
    @State private var isDropTargeted: Bool = false

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
        // ONE observation of the whole blob (the Music pane's mechanism):
        // anything in `stickySnapshot` is sticky by construction.
        .onChange(of: stickySnapshot) { _, _ in guard !hydrating else { return }; persist() }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                // The model decides whether the weights are on this Mac at
                // all, so it is read before the photo it acts on.
                VStack(alignment: .leading, spacing: 14) {
                    modelSection
                    photoSection
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
                historyShelf
                outputFolderLink
            }
            .padding(16)
            // The preview gives way in a small window.
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
    }

    // MARK: - Sections

    /// Best-per-capability up front, everything else behind "Other Models", and
    /// the Download button ON the model — see `MediaModelChooser`. The transfer
    /// bar (or the conversion hint for weights with no download) and residency
    /// belong to the model, not to the output, so they sit with it rather than
    /// beside Generate.
    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            modelChooser
            if lanModel == nil && !downloads.bundleReady(model.bundle) {
                if model.isLocalOnly { convertHint } else { BundleDownloadBar(bundle: model.bundle, showsStartButton: false) }
            }
        }
    }

    /// Residency rides the switcher's row: it is a property of the model, and
    /// the only thing about it the pane still has to say once it is picked.
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

    private var modelChooser: some View {
        MediaModelChooser.pane(
            all: Model3DModelPreset.all,
            onThisMac: CustomMediaModels.meshPresets(from: server.allModels),
            capability: "3d",
            selected: $model, lanModel: $lanModel,
            capabilityOf: { $0.capabilityLabel },
            resolveCustom: { [models = server.allModels] in
                CustomMediaModels.meshPreset(for: $0, from: models)
            },
            bundleOf: { $0.bundle },
            downloads: downloads,
            onDownloadFinished: { appState.refreshModels() },
            persist: persist,
            accessory: keepResidentToggle)
    }

    private var photoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Photo").font(.app(.headline).weight(.semibold))
            if let url = photoURL {
                // Same surface and same floor height as the empty well.
                MediaDropWellFilled(isTargeted: isDropTargeted) {
                    HStack(spacing: 8) {
                        MediaImageView(url: url, maxPixel: 192)
                            .frame(width: 64, height: 48)
                            .clipShape(RoundedRectangle(cornerRadius: 4))
                        Text(url.lastPathComponent)
                            .font(.app(.caption)).lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Button { photoURL = nil } label: {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(.borderless).foregroundStyle(.secondary).help("Remove photo")
                    }
                }
            } else {
                MediaDropWell(title: "Choose photo…",
                              systemImage: "photo.badge.plus",
                              isTargeted: isDropTargeted) { choosePhoto() }
                // The well says how to add a photo; what is left to say is
                // which photo works.
                fieldNote(model.family.sendsRawPhoto
                     ? "One object, any background: the model cuts the subject out itself. A cutout with a transparent background is used as is."
                     : model.family.cutoutKeepsAlpha
                     ? "A single, well-lit photo of one object works best. A cutout with a transparent background is used as is; otherwise the subject is auto-cut."
                     : "A single, well-lit photo of one object works best. The subject is auto-cut from its background.")
            }
        }
        // One photo slot, so a drop replaces what's there — see
        // `MediaDropTarget`. The target covers the whole section, which once a
        // photo is set is the filled well you'd aim a replacement at.
        .mediaDrop(.image, isTargeted: $isDropTargeted) { urls in
            if let url = urls.first { photoURL = url }
        }
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            FoldingSectionHeader(title: "Advanced options", isExpanded: $showAdvanced)
            if showAdvanced {
                switch model.family {
                case .hunyuan3d: hunyuanControls
                case .trellis2, .pixal3d: cascadeControls
                case .idol: splatControls
                }
            }
        }
    }

    @ViewBuilder private var hunyuanControls: some View {
        intSliderRow("Steps", value: $steps, range: 10...50,
                     note: "More steps refine the shape further and take longer.")
        sliderRow("Guidance", value: $guidance, range: 1...10, step: 0.5,
                  note: "How strongly the shape is pushed toward your photo. Higher follows it more closely.")
        VStack(alignment: .leading, spacing: 4) {
            Text("Mesh resolution").font(.app(.rowTitle))
            Picker("", selection: $resolution) {
                Text("128 (fast)").tag(128)
                Text("256 (balanced)").tag(256)
                Text("384 (fine)").tag(384)
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            fieldNote("Higher = finer mesh, more memory and time.")
        }
        Toggle("Texture (PBR)", isOn: $texture)
            .font(.app(.caption))
            .help("After the shape stage, paint a full PBR texture (albedo + metallic-roughness) onto the mesh from the same photo. Needs the converted paint weights (~4.6 GB).")
    }

    /// TRELLIS.2 and Pixal3D texture every model and run their published recipe, so they have no steps, guidance or paint toggle.
    @ViewBuilder private var cascadeControls: some View {
        segmentedRow("Resolution", selection: $trellisResolution,
                     options: [(1024, "1024"), (1536, "1536")],
                     note: "1536 gives a finer mesh and needs about 35 GB of memory.")
        segmentedRow("Faces", selection: $decimationTarget,
                     options: Model3DGenSettings.faceOptions.map { ($0, $0 >= 1_000_000 ? "\($0 / 1_000_000)M" : "\($0 / 1000)k") },
                     note: "More faces keep more detail. At 1M faces and above the file passes 60 MB.")
        segmentedRow("Texture size", selection: $textureSize,
                     options: [(1024, "1024"), (2048, "2048"), (4096, "4096")],
                     note: "The size of the picture painted onto the surface. Larger keeps finer surface detail and uses more memory.")
        segmentedRow("Finish", selection: $finish,
                     options: Model3DFinish.allCases.map { ($0, L10n.text($0.title)) },
                     note: "The model decides what is shiny; a finish scales it. A glossier finish shows every bump in the surface.")
    }

    /// Idol draws a Gaussian splat in one published recipe: steps and guidance trade speed for detail and
    /// fidelity to the photo, the count sets how many Gaussians the model places.
    @ViewBuilder private var splatControls: some View {
        segmentedRow("Picture", selection: $splatPhoto,
                     options: Model3DPhotoKind.allCases.map { ($0, L10n.text($0.title)) },
                     note: "How tightly Idol frames the object once it is cut out. Render fills the frame (renders and generated images); Photo leaves a margin around the object (real photographs).")
        intSliderRow("Steps", value: $splatSteps, range: 10...100,
                     note: "How many refinement passes Idol makes. Time grows with the count; 50 is the published setting, and fewer steps finish sooner but refine the result less.")
        sliderRow("Guidance", value: $splatGuidance, range: 0...10, step: 0.5,
                  note: "How strongly Idol is pushed toward your picture. Higher follows it more closely. 0 turns guidance off and makes each step about twice as fast.")
        segmentedRow("Gaussians", selection: $splatGaussians,
                     options: Model3DGenSettings.gaussianOptions.map { ($0, $0 >= 1_000_000 ? "1M" : "\($0 / 1024)k") },
                     note: "The object is a cloud of Gaussians, small coloured 3D blobs. More keep finer detail and make a larger file (about 60 bytes each).")
    }

    /// The line of help under a control.
    private func fieldNote(_ text: String) -> some View {
        Text(L10n.text(text)).font(.app(.caption2)).foregroundStyle(.secondary)
    }

    private func segmentedRow<T: Hashable>(_ label: String, selection: Binding<T>, options: [(T, String)], note: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(L10n.text(label)).font(.app(.rowTitle))
            Picker("", selection: selection) {
                ForEach(options, id: \.0) { Text($0.1).tag($0.0) }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            fieldNote(note)
        }
    }

    /// Labeled slider for a `Double` setting, the value read out on the right.
    private func sliderRow(_ label: String, value: Binding<Double>, range: ClosedRange<Double>,
                           step: Double, note: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.rowTitle))
                Spacer()
                Text(String(format: "%.1f", value.wrappedValue))
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(value: value, in: range, step: step)
                .padding(.top, Self.steppedSliderTrackDrop).font(.app(.body))
            fieldNote(note)
        }
    }

    /// Labeled slider for an `Int` setting (bridges to a `Double` slider).
    private func intSliderRow(_ label: String, value: Binding<Int>, range: ClosedRange<Int>, note: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(L10n.text(label)).font(.app(.rowTitle))
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
            .padding(.top, Self.steppedSliderTrackDrop).font(.app(.body))
            fieldNote(note)
        }
    }

    /// A stepped slider reserves a tick row under its track, so the track sits
    /// glued to the caption above; 3pt, the Video pane's value.
    private static let steppedSliderTrackDrop: CGFloat = 3

    private var actionRow: some View {
        HStack {
            if service.isRunning {
                Button(role: .destructive) { service.cancel() } label: {
                    Label("Cancel", systemImage: "stop.circle").font(.app(.body)).frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
            } else {
                Button { tryGenerate() } label: {
                    Label("Generate", systemImage: "cube.transparent").font(.app(.body)).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(photoURL == nil || (lanModel == nil && !downloads.bundleReady(model.bundle)))
            }
        }
    }

    private var convertHint: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label("Weights not found", systemImage: "wrench.and.screwdriver")
                .font(.app(.caption).weight(.semibold)).foregroundStyle(.secondary)
            Text("Hunyuan3D 2.1 has no download yet — convert the weights on-device with tests/convert_hunyuan3d_weights.py (see the repo README). They install to ~/.mlx-serve/models/local/.")
                .font(.app(.caption2)).foregroundStyle(.secondary)
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
    }

    private var previewArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.15))
            Group {
                switch service.phase {
                case .idle:
                    ContentUnavailableView("No 3D model yet", systemImage: "cube.transparent",
                                           description: Text("Choose a photo and press Generate.").font(.app(.body)))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        LoadingModelsRow()
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear).frame(width: 240)
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
            Model3DSceneView(url: URL(fileURLWithPath: path), animate: turntable)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 10) {
                Toggle("Animate", isOn: $turntable)
                    .toggleStyle(.switch).font(.app(.caption))
                    .help("Idle motion: the model turns on a turntable with a gentle breathing pulse.")
                Spacer()
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.app(.caption)).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                if model.family == .idol {
                    Button { exportPLY(path) } label: { Image(systemName: "square.and.arrow.up") }
                        .buttonStyle(.borderless).help("Save as .ply, the format splat viewers open")
                }
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                } label: { Image(systemName: "folder") }
                .buttonStyle(.borderless).help("Reveal in Finder")
            }
        }
        .padding(16)
    }

    /// Save the splat GLB at `path` as a standard `.ply` (read and converted only now, on the click).
    private func exportPLY(_ path: String) {
        guard let glb = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe),
              let ply = SplatPLYExport.ply(fromGLB: glb) else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent + ".ply"
        panel.allowedContentTypes = [.init(filenameExtension: "ply") ?? .data]
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        try? ply.write(to: url)
    }

    private var outputFolderLink: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.models3dRoot)])
        } label: {
            Label("Open output folder in Finder", systemImage: "folder").font(.app(.caption))
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help(MediaStorage.models3dRoot)
    }

    // MARK: - History shelf

    /// Horizontal strip of recent generations (thumbnails render lazily,
    /// offscreen). Click shows the model in the preview above.
    private var historyShelf: some View {
        Group {
            if !service.recent.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(service.recent, id: \.self) { path in
                            Model3DHistoryThumb(
                                path: path,
                                isCurrent: service.phase == .completed(path: path)
                            ) {
                                service.showHistoryItem(path)
                            }
                        }
                    }
                    .padding(.horizontal, 2)
                }
                .frame(height: 62)
            }
        }
    }

    // MARK: - Photo picker

    private func choosePhoto() {
        let panel = OpenPanel.make()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            photoURL = url
        }
    }

    // MARK: - Sticky settings

    private func hydrate() {
        let s = Model3DGenSettings.load()
        model = s.resolvedModel(models: server.allModels)
        lanModel = LanPick.lanId(s.modelId)
        steps = s.steps
        guidance = s.guidance
        resolution = s.resolution
        keepResident = s.keepResident
        turntable = s.turntable
        texture = s.texture
        trellisResolution = s.trellisResolution
        decimationTarget = s.decimationTarget
        textureSize = s.textureSize
        finish = s.finish
        splatSteps = s.splatSteps
        splatGuidance = s.splatGuidance
        splatGaussians = s.splatGaussians
        splatPhoto = s.splatPhoto
        showAdvanced = s.showAdvanced
        // A photo that has gone since is dropped, as the other panes do.
        photoURL = s.photoPath.flatMap { FileManager.default.fileExists(atPath: $0) ? URL(fileURLWithPath: $0) : nil }
    }

    /// Everything the pane persists, as one value: the body's single
    /// `onChange` watches this rather than each field.
    private var stickySnapshot: Model3DGenSettings {
        var s = Model3DGenSettings()
        s.modelId = LanPick.persisted(lanModel: lanModel, presetId: model.id)
        s.steps = steps
        s.guidance = guidance
        s.resolution = resolution
        s.keepResident = keepResident
        s.turntable = turntable
        s.texture = texture
        s.trellisResolution = trellisResolution
        s.decimationTarget = decimationTarget
        s.textureSize = textureSize
        s.finish = finish
        s.splatSteps = splatSteps
        s.splatGuidance = splatGuidance
        s.splatGaussians = splatGaussians
        s.splatPhoto = splatPhoto
        s.photoPath = photoURL?.path
        s.showAdvanced = showAdvanced
        return s
    }

    private func persist() { stickySnapshot.save() }

    // MARK: - Generate

    private func tryGenerate() {
        guard let photoURL else { return }
        let req = Model3DGenRequest(
            model: model,
            photoPath: photoURL.path,
            steps: steps,
            guidanceScale: guidance,
            octreeResolution: resolution,
            keepResident: keepResident,
            lanModelId: lanModel,
            texture: texture,
            trellisResolution: trellisResolution,
            decimationTarget: decimationTarget,
            textureSize: textureSize,
            finish: finish,
            splatSteps: splatSteps,
            splatGuidance: splatGuidance,
            splatGaussians: splatGaussians,
            splatPhoto: splatPhoto
        )
        persist()
        let total = RAMChecker.totalGB
        let needed = model.approxRAMGB(trellisResolution: trellisResolution)
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

// MARK: - History thumbnail cell

/// One shelf cell: the pre-rendered `<glb>.thumb.png` when available, else a
/// cube placeholder while a lazy offscreen render fills it in (pre-thumbnail
/// generations from older builds get theirs on first display).
private struct Model3DHistoryThumb: View {
    let path: String
    let isCurrent: Bool
    let action: () -> Void

    @State private var image: NSImage? = nil

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.10))
                if let image {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 56, height: 56)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    Image(systemName: "cube.transparent")
                        .font(.app(.title3))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 56, height: 56)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(isCurrent ? Color.accentColor : .clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
        .help((path as NSString).lastPathComponent)
        .task(id: path) {
            let thumb = Model3DGenService.thumbnailPath(for: path)
            let url = URL(fileURLWithPath: thumb)
            if let img = await MediaImage.load(url: url, maxPixel: 160) {
                if !Task.isCancelled { image = img }
                return
            }
            // Lazy render off the main actor, then re-check.
            let rendered = await Task.detached(priority: .utility) { () -> Bool in
                Model3DThumbnailer.ensure(glbPath: path)
                return FileManager.default.fileExists(atPath: thumb)
            }.value
            if rendered, let img = await MediaImage.load(url: url, maxPixel: 160),
               !Task.isCancelled { image = img }
        }
    }
}

// MARK: - Offscreen thumbnail rendering

/// Renders a small offscreen SceneKit snapshot of a GLB next to the file
/// (`<glb>.thumb.png`). Best-effort: any failure just leaves the shelf's
/// placeholder. Untestable GPU surface — the PATH contract is the tested part
/// (`Model3DGenService.thumbnailPath`).
enum Model3DThumbnailer {
    static func ensure(glbPath: String) {
        let out = Model3DGenService.thumbnailPath(for: glbPath)
        let fm = FileManager.default
        guard fm.fileExists(atPath: glbPath), !fm.fileExists(atPath: out) else { return }
        guard let scene = GLBMeshLoader.loadScene(url: URL(fileURLWithPath: glbPath)),
              let device = MTLCreateSystemDefaultDevice() else { return }

        // Frame the model: camera pulled back along +z from the bounding sphere.
        let node = GLBMeshLoader.firstGeometryNode(in: scene) ?? scene.rootNode
        let sphere = node.boundingSphere
        let radius = max(CGFloat(sphere.radius), 0.001)
        let camera = SCNCamera()
        camera.zNear = 0.01
        camera.zFar = Double(radius) * 10
        let camNode = SCNNode()
        camNode.camera = camera
        camNode.position = SCNVector3(
            CGFloat(sphere.center.x),
            CGFloat(sphere.center.y) + radius * 0.35,
            CGFloat(sphere.center.z) + radius * 2.6
        )
        camNode.look(at: SCNVector3(sphere.center.x, sphere.center.y, sphere.center.z))
        scene.rootNode.addChildNode(camNode)

        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene
        renderer.pointOfView = camNode
        renderer.autoenablesDefaultLighting = true
        let img = renderer.snapshot(atTime: 0, with: CGSize(width: 112, height: 112), antialiasingMode: .multisampling4X)

        guard let tiff = img.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else { return }
        try? png.write(to: URL(fileURLWithPath: out))
    }
}

// MARK: - GLB loading (pure, testable)

/// Loads a `.glb` mesh into an `SCNScene`. Factored out (pure) so the load path
/// is unit-testable against a fixture without standing up a view.
enum GLBMeshLoader {
    /// Load the file at `url` into a scene, or nil if it can't be parsed.
    static func loadScene(url: URL) -> SCNScene? {
        if let scene = loadGLB(url: url) { return scene }
        // Fallbacks for formats ModelIO/SceneKit parse natively (obj/usd/…).
        let asset = MDLAsset(url: url)
        if asset.count > 0 {
            let scene = SCNScene(mdlAsset: asset)
            if firstGeometryNode(in: scene) != nil { return scene }
        }
        return try? SCNScene(url: url, options: nil)
    }

    /// The first geometry-bearing node in a scene, searched depth-first — the
    /// node the turntable/breathing animations attach to.
    static func firstGeometryNode(in scene: SCNScene) -> SCNNode? {
        var found: SCNNode?
        scene.rootNode.enumerateHierarchy { node, stop in
            if node.geometry != nil {
                found = node
                stop.pointee = true
            }
        }
        return found
    }

    // MARK: - Minimal glTF-binary (.glb) reader

    private static func loadGLB(url: URL) -> SCNScene? {
        guard let data = try? Data(contentsOf: url), data.count >= 12 else { return nil }
        return data.withUnsafeBytes { raw -> SCNScene? in
            func u32(_ off: Int) -> UInt32? {
                guard off >= 0, off + 4 <= raw.count else { return nil }
                return raw.loadUnaligned(fromByteOffset: off, as: UInt32.self)
            }
            guard u32(0) == 0x4654_6C67 else { return nil }   // "glTF" little-endian
            let total = Int(u32(8) ?? 0)
            var off = 12
            var json: [String: Any]? = nil
            var binRange: Range<Int>? = nil
            while off + 8 <= min(total, raw.count) {
                guard let clen = u32(off).map(Int.init), let ctype = u32(off + 4) else { break }
                let body = off + 8
                guard clen >= 0, body + clen <= raw.count else { break }
                if ctype == 0x4E4F_534A, let base = raw.baseAddress {   // "JSON"
                    let sub = Data(bytes: base.advanced(by: body), count: clen)
                    json = (try? JSONSerialization.jsonObject(with: sub)) as? [String: Any]
                } else if ctype == 0x004E_4942 {                        // "BIN\0"
                    binRange = body..<(body + clen)
                }
                off = body + clen
            }
            guard let gltf = json, let bin = binRange else { return nil }
            return buildScene(gltf: gltf, raw: raw, bin: bin)
        }
    }

    private static func buildScene(gltf: [String: Any], raw: UnsafeRawBufferPointer, bin: Range<Int>) -> SCNScene? {
        guard let accessors = gltf["accessors"] as? [[String: Any]],
              let views = gltf["bufferViews"] as? [[String: Any]],
              let meshes = gltf["meshes"] as? [[String: Any]] else { return nil }

        let scene = SCNScene()
        var built = false
        for mesh in meshes {
            guard let prims = mesh["primitives"] as? [[String: Any]] else { continue }
            for prim in prims {
                // mode 0 (POINTS) is Idol's Gaussian splat (`KHR_gaussian_splatting`), drawn as a coloured
                // point cloud; mode 4 (TRIANGLES, the default) is every mesh the writers emit.
                if (prim["mode"] as? Int) == 0 {
                    if let node = buildPoints(prim: prim, accessors: accessors, views: views, raw: raw, bin: bin) {
                        scene.rootNode.addChildNode(node)
                        built = true
                    }
                    continue
                }
                guard (prim["mode"] as? Int ?? 4) == 4,
                      let attrs = prim["attributes"] as? [String: Any],
                      let posIdx = attrs["POSITION"] as? Int,
                      let positions = readVec3(accessors, views, raw, bin, posIdx) else { continue }
                var sources = [SCNGeometrySource(vertices: positions)]
                if let normIdx = attrs["NORMAL"] as? Int,
                   let normals = readVec3(accessors, views, raw, bin, normIdx),
                   normals.count == positions.count {
                    sources.append(SCNGeometrySource(normals: normals))
                }
                // TEXCOORD_0 (paint stage): VEC2 float UVs.
                if let uvIdx = attrs["TEXCOORD_0"] as? Int,
                   let uvs = readVec2(accessors, views, raw, bin, uvIdx),
                   uvs.count == positions.count {
                    sources.append(SCNGeometrySource(textureCoordinates: uvs))
                }
                let indices: [Int32]
                if let idxAccessor = prim["indices"] as? Int,
                   let read = readIndices(accessors, views, raw, bin, idxAccessor) {
                    indices = read
                } else {
                    indices = Array(0..<Int32(positions.count))   // non-indexed
                }
                let element = SCNGeometryElement(indices: indices, primitiveType: .triangles)
                let geometry = SCNGeometry(sources: sources, elements: [element])
                if let matIdx = prim["material"] as? Int,
                   let material = buildMaterial(gltf: gltf, raw: raw, bin: bin, index: matIdx) {
                    geometry.materials = [material]
                }
                let meshNode = SCNNode(geometry: geometry)
                scene.rootNode.addChildNode(meshNode)
                built = true
            }
        }
        return built ? scene : nil
    }

    /// A POINTS primitive as a node of screen-sized, vertex-coloured points. `COLOR_0` (normalized bytes or floats)
    /// carries the splat's colour and opacity; points that are nearly transparent are left out, and the rest
    /// drawn opaque.
    static func buildPoints(prim: [String: Any], accessors: [[String: Any]], views: [[String: Any]],
                            raw: UnsafeRawBufferPointer, bin: Range<Int>) -> SCNNode? {
        guard let attrs = prim["attributes"] as? [String: Any],
              let posIdx = attrs["POSITION"] as? Int,
              let positions = readVec3(accessors, views, raw, bin, posIdx), !positions.isEmpty else { return nil }
        let colors = (attrs["COLOR_0"] as? Int).flatMap { readColors(accessors, views, raw, bin, $0, count: positions.count) }
        var kept: [SCNVector3] = []
        var rgba: [Float] = []
        kept.reserveCapacity(positions.count)
        for (i, p) in positions.enumerated() {
            if let c = colors {
                if c[i * 4 + 3] < 0.04 { continue }
                // SceneKit reads vertex colours as linear; the splat's are sRGB.
                rgba.append(contentsOf: [powf(c[i * 4], 2.2), powf(c[i * 4 + 1], 2.2), powf(c[i * 4 + 2], 2.2), 1])
            } else {
                rgba.append(contentsOf: [0.8, 0.8, 0.8, 1])
            }
            kept.append(p)
        }
        guard !kept.isEmpty else { return nil }
        // With the extension's rotation and scale, each Gaussian becomes an oriented disc in the plane of its two
        // longest axes (opaque, depth-tested): a solid stand-in for the blended splat that fills like one.
        if let ri = attrs["KHR_gaussian_splatting:ROTATION"] as? Int, let si = attrs["KHR_gaussian_splatting:SCALE"] as? Int,
           let rot = readFloats(accessors, views, raw, bin, ri, comps: 4, count: positions.count),
           let scl = readFloats(accessors, views, raw, bin, si, comps: 3, count: positions.count),
           let node = buildDiscs(positions: positions, rot: rot, scale: scl, rgba: colors, keep: { i in (colors?[i * 4 + 3] ?? 1) >= 0.05 }) {
            return node
        }
        var sources = [SCNGeometrySource(vertices: kept)]
        let colorData = rgba.withUnsafeBytes { Data($0) }
        sources.append(SCNGeometrySource(data: colorData, semantic: .color, vectorCount: kept.count,
                                         usesFloatComponents: true, componentsPerVector: 4,
                                         bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                                         dataStride: 4 * MemoryLayout<Float>.size))
        let element = SCNGeometryElement(indices: Array(0..<Int32(kept.count)), primitiveType: .point)
        element.pointSize = 0.006
        element.minimumPointScreenSpaceRadius = 2.5
        element.maximumPointScreenSpaceRadius = 7
        let geometry = SCNGeometry(sources: sources, elements: [element])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = NSColor.white
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }

    /// One quad per kept Gaussian: centre +- `reach` standard deviations along its two longest axes.
    private static func buildDiscs(positions: [SCNVector3], rot: [Float], scale: [Float], rgba: [Float]?,
                                   keep: (Int) -> Bool) -> SCNNode? {
        let reach: Float = 1.5
        var verts: [SCNVector3] = []
        var cols: [Float] = []
        var idx: [Int32] = []
        for i in 0..<positions.count where keep(i) {
            let x = rot[i * 4], y = rot[i * 4 + 1], z = rot[i * 4 + 2], w = rot[i * 4 + 3]
            let axes: [SIMD3<Float>] = [
                SIMD3(1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
                SIMD3(2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
                SIMD3(2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)),
            ]
            let s = [scale[i * 3], scale[i * 3 + 1], scale[i * 3 + 2]]
            let order = [0, 1, 2].sorted { s[$0] > s[$1] }
            let cap: Float = 0.012   // the object spans about 1 unit: a larger Gaussian is blur the blend would soften
            let a = axes[order[0]] * min(reach * s[order[0]], cap), b = axes[order[1]] * min(reach * s[order[1]], cap)
            let c = SIMD3<Float>(Float(positions[i].x), Float(positions[i].y), Float(positions[i].z))
            let base = Int32(verts.count)
            for (sa, sb) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] as [(Float, Float)] {
                let p = c + a * sa + b * sb
                verts.append(SCNVector3(CGFloat(p.x), CGFloat(p.y), CGFloat(p.z)))
                if let col = rgba {
                    cols += [powf(col[i * 4], 2.2), powf(col[i * 4 + 1], 2.2), powf(col[i * 4 + 2], 2.2), 1]
                } else { cols += [0.8, 0.8, 0.8, 1] }
            }
            idx += [base, base + 1, base + 2, base + 1, base + 3, base + 2]
        }
        guard !verts.isEmpty else { return nil }
        let colorData = cols.withUnsafeBytes { Data($0) }
        let sources = [SCNGeometrySource(vertices: verts),
                       SCNGeometrySource(data: colorData, semantic: .color, vectorCount: verts.count,
                                         usesFloatComponents: true, componentsPerVector: 4,
                                         bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
                                         dataStride: 4 * MemoryLayout<Float>.size)]
        let geometry = SCNGeometry(sources: sources, elements: [SCNGeometryElement(indices: idx, primitiveType: .triangles)])
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.diffuse.contents = NSColor.white
        geometry.materials = [material]
        return SCNNode(geometry: geometry)
    }

    /// Read a float VEC`comps` accessor as `count` x `comps` floats.
    private static func readFloats(_ accessors: [[String: Any]], _ views: [[String: Any]], _ raw: UnsafeRawBufferPointer,
                                   _ bin: Range<Int>, _ idx: Int, comps: Int, count: Int) -> [Float]? {
        guard idx >= 0, idx < accessors.count else { return nil }
        let a = accessors[idx]
        guard a["componentType"] as? Int == 5126, a["count"] as? Int == count,
              let v = a["bufferView"] as? Int, v >= 0, v < views.count else { return nil }
        let start = bin.lowerBound + ((views[v]["byteOffset"] as? Int) ?? 0) + ((a["byteOffset"] as? Int) ?? 0)
        guard start + count * comps * 4 <= bin.upperBound else { return nil }
        return (0..<(count * comps)).map { raw.loadUnaligned(fromByteOffset: start + $0 * 4, as: Float.self) }
    }

    /// Read a VEC4 (or VEC3) `COLOR_0` accessor, unsigned-byte or unsigned-short normalized or float, as `count` x 4 floats.
    private static func readColors(_ accessors: [[String: Any]], _ views: [[String: Any]],
                                   _ raw: UnsafeRawBufferPointer, _ bin: Range<Int>, _ idx: Int, count: Int) -> [Float]? {
        guard idx >= 0, idx < accessors.count else { return nil }
        let acc = accessors[idx]
        guard let type = acc["type"] as? String, type == "VEC4" || type == "VEC3",
              let ct = acc["componentType"] as? Int, acc["count"] as? Int == count,
              let vIdx = acc["bufferView"] as? Int, vIdx >= 0, vIdx < views.count else { return nil }
        let comps = type == "VEC4" ? 4 : 3
        let size = ct == 5126 ? 4 : (ct == 5123 ? 2 : 1)
        guard ct == 5126 || ct == 5123 || ct == 5121 else { return nil }
        let view = views[vIdx]
        let start = bin.lowerBound + ((view["byteOffset"] as? Int) ?? 0) + ((acc["byteOffset"] as? Int) ?? 0)
        let stride = (view["byteStride"] as? Int) ?? comps * size
        var out = [Float](repeating: 1, count: count * 4)
        for i in 0..<count {
            let base = start + i * stride
            guard base + comps * size <= bin.upperBound else { return nil }
            for c in 0..<comps {
                let at = base + c * size
                switch ct {
                case 5126: out[i * 4 + c] = raw.loadUnaligned(fromByteOffset: at, as: Float.self)
                case 5123: out[i * 4 + c] = Float(raw.loadUnaligned(fromByteOffset: at, as: UInt16.self)) / 65535
                default: out[i * 4 + c] = Float(raw.loadUnaligned(fromByteOffset: at, as: UInt8.self)) / 255
                }
            }
        }
        return out
    }

    /// Build an SCNMaterial from a glTF PBR material: embedded-PNG baseColor →
    /// diffuse, metallicRoughness → metalness (B) / roughness (G) via SceneKit's
    /// per-channel texture components, plus the finish it declares (roughness factor, clearcoat extension).
    /// Scoped to what the mlx-serve writer emits.
    private static func buildMaterial(gltf: [String: Any], raw: UnsafeRawBufferPointer,
                                      bin: Range<Int>, index: Int) -> SCNMaterial? {
        guard let materials = gltf["materials"] as? [[String: Any]],
              index >= 0, index < materials.count,
              let pbr = materials[index]["pbrMetallicRoughness"] as? [String: Any] else { return nil }

        func textureImage(_ slot: [String: Any]?) -> NSImage? {
            guard let texIdx = slot?["index"] as? Int,
                  let textures = gltf["textures"] as? [[String: Any]],
                  texIdx >= 0, texIdx < textures.count,
                  let srcIdx = textures[texIdx]["source"] as? Int,
                  let images = gltf["images"] as? [[String: Any]],
                  srcIdx >= 0, srcIdx < images.count,
                  let viewIdx = images[srcIdx]["bufferView"] as? Int,
                  let views = gltf["bufferViews"] as? [[String: Any]],
                  viewIdx >= 0, viewIdx < views.count,
                  let base = raw.baseAddress else { return nil }
            let view = views[viewIdx]
            let off = bin.lowerBound + ((view["byteOffset"] as? Int) ?? 0)
            let len = (view["byteLength"] as? Int) ?? 0
            guard len > 0, off + len <= bin.upperBound else { return nil }
            return NSImage(data: Data(bytes: base.advanced(by: off), count: len))
        }

        let material = SCNMaterial()
        material.lightingModel = .physicallyBased
        if let albedo = textureImage(pbr["baseColorTexture"] as? [String: Any]) {
            material.diffuse.contents = albedo
        }
        if let mr = textureImage(pbr["metallicRoughnessTexture"] as? [String: Any]) {
            // glTF packs G = roughness, B = metallic.
            material.roughness.contents = mr
            material.roughness.textureComponents = .green
            // glTF multiplies the roughness texture by the material's factor, which is how a finish rides in.
            material.roughness.intensity = CGFloat(pbr["roughnessFactor"] as? Double ?? 1.0)
            material.metalness.contents = mr
            material.metalness.textureComponents = .blue
        } else {
            material.metalness.contents = pbr["metallicFactor"] as? Double ?? 0.0
            material.roughness.contents = pbr["roughnessFactor"] as? Double ?? 1.0
        }
        if let coat = ((materials[index]["extensions"] as? [String: Any])?["KHR_materials_clearcoat"] as? [String: Any])?["clearcoatFactor"] as? Double,
           coat > 0 {
            material.clearCoat.contents = coat
            material.clearCoatRoughness.contents = 0.03
        }
        return material
    }

    /// Read a VEC2 FLOAT accessor as `[CGPoint]` (texture coordinates).
    private static func readVec2(_ accessors: [[String: Any]], _ views: [[String: Any]],
                                 _ raw: UnsafeRawBufferPointer, _ bin: Range<Int>, _ idx: Int) -> [CGPoint]? {
        guard idx >= 0, idx < accessors.count else { return nil }
        let acc = accessors[idx]
        guard acc["type"] as? String == "VEC2",
              (acc["componentType"] as? Int) == 5126,   // FLOAT
              let count = acc["count"] as? Int,
              let vIdx = acc["bufferView"] as? Int, vIdx >= 0, vIdx < views.count else { return nil }
        let view = views[vIdx]
        let start = bin.lowerBound + ((view["byteOffset"] as? Int) ?? 0) + ((acc["byteOffset"] as? Int) ?? 0)
        let stride = (view["byteStride"] as? Int) ?? 8   // tightly packed VEC2<float>
        var out: [CGPoint] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            let base = start + i * stride
            guard base + 8 <= bin.upperBound else { return nil }
            let u = raw.loadUnaligned(fromByteOffset: base, as: Float.self)
            let v = raw.loadUnaligned(fromByteOffset: base + 4, as: Float.self)
            out.append(CGPoint(x: CGFloat(u), y: CGFloat(v)))
        }
        return out
    }

    /// Read a VEC3 FLOAT accessor as `[SCNVector3]`.
    private static func readVec3(_ accessors: [[String: Any]], _ views: [[String: Any]],
                                 _ raw: UnsafeRawBufferPointer, _ bin: Range<Int>, _ idx: Int) -> [SCNVector3]? {
        guard idx >= 0, idx < accessors.count else { return nil }
        let acc = accessors[idx]
        guard acc["type"] as? String == "VEC3",
              (acc["componentType"] as? Int) == 5126,   // FLOAT
              let count = acc["count"] as? Int,
              let vIdx = acc["bufferView"] as? Int, vIdx >= 0, vIdx < views.count else { return nil }
        let view = views[vIdx]
        let start = bin.lowerBound + ((view["byteOffset"] as? Int) ?? 0) + ((acc["byteOffset"] as? Int) ?? 0)
        let stride = (view["byteStride"] as? Int) ?? 12   // tightly packed VEC3<float>
        var out: [SCNVector3] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            let base = start + i * stride
            guard base + 12 <= bin.upperBound else { return nil }
            let x = raw.loadUnaligned(fromByteOffset: base, as: Float.self)
            let y = raw.loadUnaligned(fromByteOffset: base + 4, as: Float.self)
            let z = raw.loadUnaligned(fromByteOffset: base + 8, as: Float.self)
            out.append(SCNVector3(CGFloat(x), CGFloat(y), CGFloat(z)))
        }
        return out
    }

    /// Read a SCALAR unsigned-int accessor (uint8/16/32) as `[Int32]`.
    private static func readIndices(_ accessors: [[String: Any]], _ views: [[String: Any]],
                                    _ raw: UnsafeRawBufferPointer, _ bin: Range<Int>, _ idx: Int) -> [Int32]? {
        guard idx >= 0, idx < accessors.count else { return nil }
        let acc = accessors[idx]
        guard acc["type"] as? String == "SCALAR",
              let ct = acc["componentType"] as? Int,
              let count = acc["count"] as? Int,
              let vIdx = acc["bufferView"] as? Int, vIdx >= 0, vIdx < views.count else { return nil }
        let size: Int
        switch ct {
        case 5125: size = 4   // UNSIGNED_INT
        case 5123: size = 2   // UNSIGNED_SHORT
        case 5121: size = 1   // UNSIGNED_BYTE
        default: return nil
        }
        let view = views[vIdx]
        let start = bin.lowerBound + ((view["byteOffset"] as? Int) ?? 0) + ((acc["byteOffset"] as? Int) ?? 0)
        let stride = (view["byteStride"] as? Int) ?? size
        var out: [Int32] = []
        out.reserveCapacity(count)
        for i in 0..<count {
            let base = start + i * stride
            guard base + size <= bin.upperBound else { return nil }
            switch ct {
            case 5125: out.append(Int32(bitPattern: raw.loadUnaligned(fromByteOffset: base, as: UInt32.self)))
            case 5123: out.append(Int32(raw.loadUnaligned(fromByteOffset: base, as: UInt16.self)))
            default:   out.append(Int32(raw.loadUnaligned(fromByteOffset: base, as: UInt8.self)))
            }
        }
        return out
    }
}

// MARK: - SceneKit preview

/// SceneKit preview of the generated GLB. Mirrors VideoGenView's
/// `AVPlayerViewRepresentable` — a thin `NSViewRepresentable` so the media
/// preview lives in AppKit, not a state-driven SwiftUI generic. Reloads only
/// when the file changes; the "Animate" idle is a procedural turntable spin
/// plus a gentle breathing scale pulse (plain `SCNAction`s).
struct Model3DSceneView: NSViewRepresentable {
    /// Turntable revolution period, seconds.
    static let turntablePeriod: TimeInterval = 12
    /// Breathing pulse: ±3 % scale over a 4 s out/back cycle.
    static let breathePeriod: TimeInterval = 4
    static let breatheScale: CGFloat = 1.03

    let url: URL?
    /// Idle motion (the pane's "Animate" toggle): spin + breathe when true.
    var animate: Bool = true

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = .clear
        return view
    }

    func updateNSView(_ nsView: SCNView, context: Context) {
        context.coordinator.animatePref = animate
        if context.coordinator.loadedURL != url {
            context.coordinator.loadedURL = url
            nsView.scene = nil
            context.coordinator.modelNode = nil
            context.coordinator.animating = nil
            // Parsing can outlive a later history selection.
            let token = context.coordinator.nextLoadToken()
            guard let url else { return }
            Task { [coordinator = context.coordinator] in
                let result = await Task.detached(priority: .userInitiated) { () -> (scene: SCNScene, node: SCNNode?)? in
                    guard let scene = GLBMeshLoader.loadScene(url: url) else { return nil }
                    return (scene: scene, node: GLBMeshLoader.firstGeometryNode(in: scene))
                }.value
                guard coordinator.loadToken == token else { return }
                nsView.scene = result?.scene
                coordinator.modelNode = result?.node
                coordinator.setAnimating(coordinator.animatePref)
            }
        }
        context.coordinator.setAnimating(animate)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var loadedURL: URL?
        var modelNode: SCNNode?
        /// nil = fresh load (actions must be (re)installed either way).
        var animating: Bool?
        var animatePref = true
        /// Identity of the newest in-flight scene load; only it may apply.
        private(set) var loadToken = 0

        func nextLoadToken() -> Int { loadToken += 1; return loadToken }

        func setAnimating(_ on: Bool) {
            guard on != animating, let node = modelNode else { return }
            animating = on
            node.removeAllActions()
            guard on else {
                node.scale = SCNVector3(1, 1, 1)
                return
            }
            let spin = SCNAction.rotateBy(x: 0, y: 2 * .pi, z: 0,
                                          duration: Model3DSceneView.turntablePeriod)
            let half = Model3DSceneView.breathePeriod / 2
            let out = SCNAction.scale(to: Model3DSceneView.breatheScale, duration: half)
            out.timingMode = .easeInEaseOut
            let back = SCNAction.scale(to: 1.0, duration: half)
            back.timingMode = .easeInEaseOut
            node.runAction(.repeatForever(spin))
            node.runAction(.repeatForever(.sequence([out, back])))
        }
    }
}
