import SwiftUI

struct ModelSettingsRequest: Identifiable {
    let path: String
    let title: String
    var id: String { path }
}

/// Per-model context / KV quant / MTP (issue #269). Writes the server's
/// `model-settings.json`, then applies it per `ModelSettingsApply.plan`.
enum ModelSettingsApply {
    enum Plan { case saveOnly, reload, restart }

    /// The startup model restarts the server: a hot unload + load re-bills
    /// it under `--max-resident-mem`, which the launch load never paid.
    static func plan(serverRunning: Bool, loaded: Bool, isStartupModel: Bool) -> Plan {
        guard serverRunning, loaded else { return .saveOnly }
        return isStartupModel ? .restart : .reload
    }

    /// MTP rows only where a head exists (unknown = older server, show);
    /// acceptance only while MTP is not Off and no DFlash drafter forces it exact.
    static func mtpRows(available: Bool?, mtp: Bool?, dflash: Bool = false) -> (mtp: Bool, acceptance: Bool) {
        let show = available ?? true
        return (show, show && mtp != false && !dflash)
    }
}

struct ModelSettingsSheet: View {
    let request: ModelSettingsRequest
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager
    @Environment(\.dismiss) private var dismiss

    @State private var override = ModelOverride()
    @State private var initialOverride = ModelOverride()
    @State private var settingsFile = ModelSettingsFile()
    @State private var inheritedGeneration = GenerationDefaults()
    @State private var addingCustom = false
    @State private var customKey = ""
    @State private var customValue = ""
    @State private var busy = false
    @State private var error: String?
    @State private var socket: DrafterSocket = .automatic
    @State private var initialSocket: DrafterSocket = .automatic
    @State private var modelGB: Double = 0

    private var live: ModelInfo? {
        server.allModels.first { request.path.hasSuffix("/" + $0.name) || $0.name == request.path }
    }

    private var plan: ModelSettingsApply.Plan {
        if !override.changesLoad(from: initialOverride) && socket == initialSocket { return .saveOnly }
        return ModelSettingsApply.plan(serverRunning: server.status == .running,
                                loaded: live?.loaded ?? false,
                                isStartupModel: server.currentModelPath == request.path)
    }

    /// Why the typed alias cannot be saved, if it cannot.
    private var aliasError: String? {
        guard let a = override.alias else { return nil }
        if !ModelOverride.isValidAlias(a) { return "An alias has no spaces, @, / or quotes, and is not \"mlx-serve\"." }
        if let other = settingsFile.pathUsingAlias(a, except: request.path) {
            return "\"\(a)\" is already the alias of \((other as NSString).lastPathComponent)."
        }
        if server.allModels.contains(where: { $0.name == a && $0.name != live?.name }) {
            return "\"\(a)\" is already the id of another model."
        }
        return nil
    }

    /// A running server answers from `/v1/models`; otherwise the app's own disk probe.
    private var mtpAvailable: Bool? {
        if let a = live?.mtpAvailable { return a }
        return appState.localModels.first { $0.path == request.path }?.hasMtpHead
    }

    /// ds4 and llama.cpp read only the context size from model-settings.json.
    private var isGguf: Bool {
        request.path.hasSuffix(".gguf") || appState.localModels.first { $0.path == request.path }?.quantFile != nil
    }

    /// The int8 prefill route exists only on Prism Hadamard packs (Bonsai 2).
    private var hasInt8PrefillRoute: Bool {
        appState.localModels.first { $0.path == request.path }?.modelType == "prism_hadamard_qwen35"
    }

    private var rows: (mtp: Bool, acceptance: Bool) {
        if isGguf { return (false, false) }
        return ModelSettingsApply.mtpRows(available: mtpAvailable, mtp: override.mtp, dflash: bindsDflash)
    }

    private var bindsDflash: Bool {
        socket.bindsDflash(localDrafter: FileManager.default.fileExists(
            atPath: (request.path as NSString).appendingPathComponent(DrafterGems.packFolder + "/config.json")))
    }

    private var repoId: String {
        appState.localModels.first { $0.path == request.path }?.name ?? (request.path as NSString).lastPathComponent
    }


    /// What the server actually loaded, and whether drafts are byte-exact.
    private var specLine: String? {
        guard let live, live.loaded else { return nil }
        let exact = live.specExact.map { $0 ? ", byte-exact" : ", not byte-exact" } ?? ""
        if live.drafterLoaded { return "Loaded: drafter \((live.drafterPath.map { ($0 as NSString).lastPathComponent }) ?? "")\(exact)" }
        if live.mtpLoaded { return "Loaded: MTP head\(exact)" }
        return "Loaded without speculation"
    }

    private var formHeight: CGFloat {
        var n = isGguf ? 2 : 3
        if hasInt8PrefillRoute { n += 1 }
        if !isGguf { n += 2 + (specLine == nil ? 0 : 1) }
        if rows.acceptance { n += 2 }
        if live?.loaded == true { n += 1 }
        if !isGguf { n += 2 + override.templateKwargs.count + (addingCustom ? 1 : 0) }
        return min(650, CGFloat(44 * n + 50 + 10 * 72))
    }

    @ViewBuilder
    private func kwargRow(_ key: String) -> some View {
        let value = override.templateKwargs[key] ?? ""
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(key).font(.app(.body).monospaced())
                if let hint = TemplateKwargs.hint(for: key) {
                    Text(L10n.text(hint)).font(.app(.caption2)).foregroundStyle(.secondary)
                }
            }
            Spacer()
            if let choices = TemplateKwargs.choices(for: key) {
                Picker("", selection: Binding(
                    get: { TemplateKwargs.display(value) },
                    set: { picked in
                        override.templateKwargs[key] = choices.first { TemplateKwargs.display($0) == picked } ?? picked
                    })) {
                    ForEach(choices.map(TemplateKwargs.display), id: \.self) { Text($0).font(.app(.body)).tag($0) }
                }
                .labelsHidden().fixedSize()
            } else {
                TextField("value", text: Binding(
                    get: { TemplateKwargs.display(value) },
                    set: { if let v = TemplateKwargs.parse($0) { override.templateKwargs[key] = v } }))
                    .font(.app(.body).monospaced()).frame(width: 140)
            }
            Button { override.templateKwargs[key] = nil } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain).foregroundStyle(.secondary).font(.app(.body))
        }
    }

    private func commitCustom() {
        let key = customKey.trimmingCharacters(in: .whitespaces)
        guard !key.isEmpty, let v = TemplateKwargs.parse(customValue) else { return }
        override.templateKwargs[key] = v
        customKey = ""; customValue = ""; addingCustom = false
    }

    private var footnote: String {
        switch plan {
        case .saveOnly: return "Generation defaults apply to the next request. Load settings apply when the model loads."
        case .reload: return "Applied when the model loads; the resident model is reloaded now."
        case .restart: return "Applied when the model loads; the server is restarted now."
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Model Settings").font(.app(.title3).weight(.semibold))
                    Text(request.title).font(.app(.caption)).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
            }
            .padding(16)
            Divider()
            Form {
                TextField(text: Binding(
                    get: { override.alias ?? "" },
                    set: { let t = $0.trimmingCharacters(in: .whitespaces); override.alias = t.isEmpty ? nil : t }),
                          prompt: Text("none")) { Text("Alias").font(.app(.body)) }
                    .font(.app(.body))
                Picker("Context size", selection: Binding(
                    get: { override.ctxSize ?? -1 },
                    set: { override.ctxSize = $0 < 0 ? nil : $0 })) {
                    Text("Default").font(.app(.body)).tag(-1)
                    ForEach(ContextSizeDisplay.presets, id: \.self) { n in
                        Text(ContextSizeDisplay.formatTokens(n)).font(.app(.body)).tag(n)
                    }
                }
                if !isGguf {
                Picker("KV cache", selection: Binding(
                    get: { override.kvQuant?.rawValue ?? "" },
                    set: { override.kvQuant = KvQuantChoice(rawValue: $0) })) {
                    Text("Default").tag("").font(.app(.body))
                    ForEach(KvQuantChoice.allCases, id: \.rawValue) { Text(L10n.text($0.label)).tag($0.rawValue) }
                }
                }
                if !isGguf {
                    Section("Speculation") {
                        SpeculationSocketRow(socket: $socket, repoId: repoId,
                                             modelDir: request.path, modelGB: modelGB, mtpAvailable: rows.mtp)
                        if let specLine {
                            Text(specLine).font(.app(.caption)).foregroundStyle(.secondary)
                        }
                    }
                }
                if rows.acceptance {
                Picker("MTP acceptance", selection: Binding(
                    get: { override.mtpAcceptance?.rawValue ?? "" },
                    set: { override.mtpAcceptance = MtpAcceptanceChoice(rawValue: $0) })) {
                    Text("Default").tag("").font(.app(.body))
                    ForEach(MtpAcceptanceChoice.allCases, id: \.rawValue) { Text(L10n.text($0.label)).tag($0.rawValue) }
                }
                }
                if rows.acceptance {
                Picker("MTP greedy tail", selection: Binding(
                    get: { override.mtpGreedyTail.map { $0 ? 1 : 0 } ?? -1 },
                    set: { override.mtpGreedyTail = $0 < 0 ? nil : $0 == 1 })) {
                    Text("Default").font(.app(.body)).tag(-1)
                    Text("On").font(.app(.body)).tag(1)
                    Text("Off").font(.app(.body)).tag(0)
                }
                .help("Sampled requests draft only the first MTP token from the sampler and take the head's argmax after it. Pays with Typical acceptance.")
                }
                if hasInt8PrefillRoute {
                Picker("Int8 prefill (lossy)", selection: Binding(
                    get: { override.int8Prefill.map { $0 ? 1 : 0 } ?? -1 },
                    set: { override.int8Prefill = $0 < 0 ? nil : $0 == 1 })) {
                    Text("Default").font(.app(.body)).tag(-1)
                    Text("On").font(.app(.body)).tag(1)
                    Text("Off").font(.app(.body)).tag(0)
                }
                .help("Faster prompt processing by quantizing activations to int8. Changes numerics; needs an M5-class GPU.")
                }
                Section("Generation defaults") {
                    if override.extra["generation_defaults"] != nil {
                        Text("The stored generation policy contains invalid or unsupported fields. It is preserved; correct model-settings.json before editing it here.")
                            .font(.app(.caption)).foregroundStyle(.orange)
                    }
                    GenerationDefaultsRows(profile: $override.generationDefaults, inheritance: "Global",
                                           inherited: inheritedGeneration)
                        .disabled(override.extra["generation_defaults"] != nil)
                    Text("Applies to the next request without reloading. Model rules replace global values and their client-override policy.")
                        .font(.app(.caption)).foregroundStyle(.secondary)
                }
                if !isGguf {
                    Section {
                        ForEach(override.sortedKwargKeys, id: \.self) { key in
                            kwargRow(key)
                        }
                        if addingCustom {
                            HStack {
                                TextField("key", text: $customKey).font(.app(.body).monospaced())
                                TextField("value", text: $customValue).font(.app(.body).monospaced())
                                    .onSubmit(commitCustom)
                                Button(action: commitCustom, label: { Text("Add")
                                    .font(.app(.body)) })
                                    .disabled(customKey.trimmingCharacters(in: .whitespaces).isEmpty || TemplateKwargs.parse(customValue) == nil)
                            }
                        }
                    } header: {
                        HStack {
                            Text("Chat template kwargs").font(.app(.body))
                            Spacer()
                            Menu {
                                ForEach(TemplateKwargs.known, id: \.key) { k in
                                    Button { override.templateKwargs[k.key] = k.choices[0] } label: { Text(k.key)
                                        .font(.app(.body)) }
                                        .disabled(override.templateKwargs[k.key] != nil)
                                }
                                Divider()
                                Button { addingCustom = true } label: { Text("Custom…")
                                    .font(.app(.body)) }
                            } label: {
                                Label("Add", systemImage: "plus").font(.app(.body))
                            }
                            .menuStyle(.borderlessButton).fixedSize()
                        }
                    } footer: {
                        Text("Forwarded to the model's chat template. Values the request decides (thinking, effort) win.").font(.app(.body))
                    }
                }
                if let live, live.loaded {
                    LabeledContent("Live") {
                        Text(isGguf ? "\(ContextSizeDisplay.formatTokens(live.contextLength)) context"
                             : "\(ContextSizeDisplay.formatTokens(live.contextLength)) context, KV \(live.kvQuant.isEmpty ? "default" : live.kvQuant)")
                            .foregroundStyle(.secondary).font(.app(.body))
                    }
                }
            }
            .formStyle(.grouped)
            // A grouped Form is a scroll view with no ideal height: hosted in a
            // Window it collapsed to nothing.
            .frame(height: formHeight)
            if rows.acceptance {
                Text(L10n.text("Lossy acceptance can loop on repetitive output: in our tests Typical looped 10% of runs, TokenV3 40%."))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
                    .padding(.horizontal, 16).padding(.bottom, 4)
            }
            Text(L10n.text(footnote))
                .font(.app(.caption2)).foregroundStyle(.secondary)
                .padding(.horizontal, 16)
            if let aliasError {
                Text(verbatim: aliasError).font(.app(.caption)).foregroundStyle(.red).padding(.horizontal, 16)
            }
            if let error {
                Text(error).font(.app(.caption)).foregroundStyle(.red).padding(.horizontal, 16)
            }
            HStack {
                Spacer()
                Button { dismiss() } label: { Text("Cancel")
                    .font(.app(.body)) }.keyboardShortcut(.cancelAction)
                Button { Task { await save() } } label: { Text(L10n.text(plan == .restart ? "Save & Restart" : "Save"))
                    .font(.app(.body)) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy || aliasError != nil)
            }
            .padding(16)
        }
        .frame(width: 660)
        .onAppear {
            settingsFile = ModelSettingsFile.load()
            override = settingsFile.override(for: request.path) ?? ModelOverride()
            initialOverride = override
            inheritedGeneration = (try? GenerationDefaultsFile.load()) ?? .init()
            let gems = SpeculationSocketRow.gems(repoId: repoId, modelDir: request.path, mtpAvailable: rows.mtp,
                                                 listing: downloads.packListings[repoId])
            socket = DrafterSocket.read(override, gems: gems) { downloads.gemPath($0, modelDir: request.path) }
            initialSocket = socket
            modelGB = Self.weightsGB(in: request.path)
        }
        .task { await downloads.checkPackUpdate(repoId: repoId, dir: request.path, force: true) }
    }

    private func save() async {
        busy = true
        defer { busy = false }
        if socket != initialSocket {
            let gemPath: String? = if case .gem(let g) = socket { downloads.gemPath(g, modelDir: request.path) } else { nil }
            socket.write(into: &override, gemPath: gemPath)
        }
        if bindsDflash { override.mtpAcceptance = nil; override.mtpGreedyTail = nil }
        var file = ModelSettingsFile.load()
        file.set(override, for: request.path)
        do {
            try file.save()
        } catch {
            self.error = "Could not write model-settings.json: \(error.localizedDescription)"
            return
        }
        switch plan {
        case .saveOnly:
            break
        case .restart:
            server.stop()
            server.start(modelPath: appState.selectedModelPath, options: appState.serverOptions)
        case .reload:
            do {
                try await server.unloadModel(id: live!.name)
                _ = try await server.loadModel(id: request.path, setDefault: request.path == appState.selectedModelPath)
            } catch {
                self.error = "Saved, but the reload failed: \(error.localizedDescription)"
                return
            }
        }
        dismiss()
    }

    private static func weightsGB(in dir: String) -> Double {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        let bytes = files.filter { $0.hasSuffix(".safetensors") }
            .reduce(UInt64(0)) { $0 + DownloadManager.resolvedFileSize((dir as NSString).appendingPathComponent($1)) }
        return Double(bytes) / 1e9
    }
}

/// The speculation socket: one slot, the gems that fit this model. Slotting a
/// gem that is not on disk downloads it; emptying a downloaded one offers to
/// delete its files.
struct SpeculationSocketRow: View {
    @Binding var socket: DrafterSocket
    @EnvironmentObject var downloads: DownloadManager
    let repoId: String
    let modelDir: String
    let modelGB: Double
    let mtpAvailable: Bool
    @State private var removing: DrafterGem?

    private var gems: [DrafterGem] {
        Self.gems(repoId: repoId, modelDir: modelDir, mtpAvailable: mtpAvailable, listing: downloads.packListings[repoId])
    }

    static func gems(repoId: String, modelDir: String, mtpAvailable: Bool, listing: [String: Int64]?) -> [DrafterGem] {
        let local = FileManager.default.fileExists(atPath: (modelDir as NSString).appendingPathComponent(DrafterGems.packFolder + "/config.json"))
        return DrafterGems.gems(forRepoId: repoId, packFiles: listing, localDrafter: local, mtpAvailable: mtpAvailable)
    }

    private var fetching: DrafterGem? { gems.first { downloads.isFetchingGem($0) } }

    var body: some View {
        LabeledContent("Drafter") {
            if let g = fetching {
                HStack(spacing: 8) {
                    ProgressView(value: downloads.downloads[g.repo]?.progress ?? 0).frame(width: 100)
                    Button("Cancel") {
                        downloads.cancelGem(g)
                        socket = .empty
                    }
                }
            } else {
                Menu(Self.label(socket)) {
                    Button("Automatic") { choose(.automatic) }
                    Button("Empty") { choose(.empty) }
                    if !gems.isEmpty { Divider() }
                    ForEach(gems) { g in
                        Button(itemLabel(g)) { pick(g) }.disabled(!onDisk(g) && !fits(g))
                    }
                }
                .fixedSize()
            }
        }
        .alert("Remove drafter", isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
               presenting: removing) { g in
            Button("Keep Files", role: .cancel) {}
            Button("Delete Files", role: .destructive) { downloads.removeGem(g, modelDir: modelDir) }
                .keyboardShortcut(.defaultAction)
        } message: { g in
            Text("Also delete the \(g.label) files (\(SystemMemoryInfo.preciseGB(g.sizeGB)))?").font(.app(.body))
        }
    }

    static func label(_ s: DrafterSocket) -> String {
        switch s {
        case .automatic: "Automatic"
        case .empty: "Empty"
        case .gem(let g): g.label
        case .custom(let p): (p as NSString).lastPathComponent
        }
    }

    private func onDisk(_ g: DrafterGem) -> Bool {
        !g.needsDownload || downloads.gemPath(g, modelDir: modelDir) != nil
    }

    private func fits(_ g: DrafterGem) -> Bool {
        DrafterGems.fits(g, modelGB: modelGB, memory: .current())
    }

    private func itemLabel(_ g: DrafterGem) -> String {
        if onDisk(g) { return g.label }
        let size = SystemMemoryInfo.preciseGB(g.sizeGB)
        return fits(g) ? "\(g.label) (download \(size))" : "\(g.label) (\(size), does not fit in memory)"
    }

    private func pick(_ g: DrafterGem) {
        guard !onDisk(g) else { socket = .gem(g); return }
        downloads.startGem(g, modelDir: modelDir) { ok in if ok { socket = .gem(g) } }
    }

    private func choose(_ s: DrafterSocket) {
        if case .gem(let g) = socket, g.needsDownload, downloads.gemPath(g, modelDir: modelDir) != nil { removing = g }
        socket = s
    }
}
