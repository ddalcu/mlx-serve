import SwiftUI
import AppKit
import SceneKit
import ModelIO
import SceneKit.ModelIO
import UniformTypeIdentifiers

/// 3D generation window — single photo → textured mesh, run natively by the
/// embedded mlx-serve server (Hunyuan3D 2.1 shape stage). Same shell as
/// ImageGen/VideoGen/AudioGen: a model picker, a photo chip, an Advanced
/// disclosure (steps / guidance / mesh resolution), and a SceneKit preview of
/// the result with a turntable "Animate" toggle.
struct Model3DGenView: View {
    @EnvironmentObject var service: Model3DGenService
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    @State private var photoURL: URL? = nil
    @State private var model: Model3DModelPreset = .hunyuan3d21_8bit
    @State private var steps: Int = 30
    @State private var guidance: Double = 5.0
    @State private var resolution: Int = 384
    @State private var keepResident: Bool = false
    @State private var turntable: Bool = true
    @State private var texture: Bool = false
    @State private var showAdvanced: Bool = false

    @State private var showRAMWarning: Bool = false
    @State private var ramWarningMessage: String = ""
    @State private var pendingRequest: Model3DGenRequest? = nil
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
        .onChange(of: steps) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: guidance) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: resolution) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: keepResident) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: turntable) { _, _ in guard !hydrating else { return }; persist() }
        .onChange(of: texture) { _, _ in guard !hydrating else { return }; persist() }
    }

    private var readyView: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    photoSection
                    modelSection
                    if showAdvanced { advancedSection } else { advancedToggle }
                    actionRow
                }
                .padding(16)
            }
            .frame(minWidth: 340, idealWidth: 380)

            VStack(spacing: 12) {
                previewArea
                historyShelf
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

    private var photoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Photo").font(.subheadline.weight(.semibold))
            if let url = photoURL {
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
                    Button { photoURL = nil } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.borderless).foregroundStyle(.secondary).help("Remove photo")
                }
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
            } else {
                Button { choosePhoto() } label: {
                    Label("Choose photo…", systemImage: "photo.badge.plus").font(.caption)
                }
                Text("A single, well-lit photo of one object works best. The subject is auto-cut from its background.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model").font(.subheadline.weight(.semibold))
            Picker("", selection: $model) {
                ForEach(Model3DModelPreset.all) { preset in
                    Text(preset.name).tag(preset)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            Text("~\(model.approxRAMGB) GB RAM • single image → 3D mesh")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
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
                Text("Steps (\(steps))").font(.caption)
                Slider(value: Binding(get: { Double(steps) }, set: { steps = Int($0) }), in: 10...50, step: 1)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("Guidance (\(String(format: "%.1f", guidance)))").font(.caption)
                Slider(value: $guidance, in: 1...10, step: 0.5)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Mesh resolution").font(.caption)
                Picker("", selection: $resolution) {
                    Text("128 (fast)").tag(128)
                    Text("256 (balanced)").tag(256)
                    Text("384 (fine)").tag(384)
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                Text("Higher = finer mesh, more memory and time.").font(.caption2).foregroundStyle(.secondary)
            }
            Toggle("Texture (PBR)", isOn: $texture)
                .font(.caption)
                .help("After the shape stage, paint a full PBR texture (albedo + metallic-roughness) onto the mesh from the same photo. Needs the converted paint weights (~4.6 GB).")
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
                        Label("Generate", systemImage: "cube.transparent").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(photoURL == nil || !downloads.bundleReady(model.bundle))
                }
            }
        }
    }

    private var convertHint: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label("Weights not found", systemImage: "wrench.and.screwdriver")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text("Hunyuan3D 2.1 has no download yet — convert the weights on-device with tests/convert_hunyuan3d_weights.py (see the repo README). They install to ~/.mlx-serve/models/local/.")
                .font(.caption2).foregroundStyle(.secondary)
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
                                           description: Text("Choose a photo and press Generate."))
                case .running(let step, let total, let message):
                    VStack(spacing: 12) {
                        ProgressView(value: Double(step), total: max(1, Double(total)))
                            .progressViewStyle(.linear).frame(width: 240)
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
            Model3DSceneView(url: URL(fileURLWithPath: path), animate: turntable)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            HStack(spacing: 10) {
                Toggle("Animate", isOn: $turntable)
                    .toggleStyle(.switch).font(.caption)
                    .help("Idle motion: the model turns on a turntable with a gentle breathing pulse.")
                Spacer()
                Text(URL(fileURLWithPath: path).lastPathComponent)
                    .font(.caption).foregroundStyle(.secondary)
                    .lineLimit(1).truncationMode(.middle)
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
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: MediaStorage.models3dRoot)])
        } label: {
            Label("Open output folder in Finder", systemImage: "folder").font(.caption)
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
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image, .png, .jpeg, .heic]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            photoURL = url
        }
    }

    // MARK: - Sticky settings

    private func hydrate() {
        let s = Model3DGenSettings.load()
        model = s.resolvedModel
        steps = s.steps
        guidance = s.guidance
        resolution = s.resolution
        keepResident = s.keepResident
        turntable = s.turntable
        texture = s.texture
    }

    private func persist() {
        var s = Model3DGenSettings()
        s.modelId = model.id
        s.steps = steps
        s.guidance = guidance
        s.resolution = resolution
        s.keepResident = keepResident
        s.turntable = turntable
        s.texture = texture
        s.save()
    }

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
            texture: texture
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
        let logText = service.log.joined(separator: "\n")
        let alert = NSAlert()
        alert.messageText = "3D generation log"
        alert.informativeText = logText.isEmpty ? "(no output)" : logText
        alert.runModal()
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
                        .font(.title3)
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
            if let img = NSImage(contentsOfFile: thumb) {
                image = img
                return
            }
            // Lazy render off the main actor, then re-check.
            let rendered = await Task.detached(priority: .utility) { () -> Bool in
                Model3DThumbnailer.ensure(glbPath: path)
                return FileManager.default.fileExists(atPath: thumb)
            }.value
            if rendered { image = NSImage(contentsOfFile: thumb) }
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
///
/// macOS has NO built-in glTF loader — ModelIO / SceneKit only parse obj / usd
/// / stl / … — so we read the GLB container ourselves. Scoped to what the
/// mlx-serve GLB writer emits: a single embedded BIN buffer, POSITION (+ optional
/// NORMAL) VEC3-float attributes, and TRIANGLES indices (uint16/uint32/uint8).
/// Anything outside that shape returns nil and the caller falls back to ModelIO
/// (for the obj/usd formats it does handle).
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
                // mode 4 (TRIANGLES) is the only mode the writer emits; default is 4.
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

    /// Build an SCNMaterial from a glTF PBR material: embedded-PNG baseColor →
    /// diffuse, metallicRoughness → metalness (B) / roughness (G) via SceneKit's
    /// per-channel texture components. Scoped to what the mlx-serve writer emits.
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
            material.metalness.contents = mr
            material.metalness.textureComponents = .blue
        } else {
            material.metalness.contents = pbr["metallicFactor"] as? Double ?? 0.0
            material.roughness.contents = pbr["roughnessFactor"] as? Double ?? 1.0
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
        if context.coordinator.loadedURL != url {
            context.coordinator.loadedURL = url
            let scene = url.flatMap { GLBMeshLoader.loadScene(url: $0) }
            nsView.scene = scene
            context.coordinator.modelNode = scene.flatMap { GLBMeshLoader.firstGeometryNode(in: $0) }
            context.coordinator.animating = nil
        }
        context.coordinator.setAnimating(animate)
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var loadedURL: URL?
        var modelNode: SCNNode?
        /// nil = fresh load (actions must be (re)installed either way).
        var animating: Bool?

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
