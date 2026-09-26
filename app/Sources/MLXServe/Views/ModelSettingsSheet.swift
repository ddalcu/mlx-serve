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
    /// acceptance only while MTP is not Off.
    static func mtpRows(available: Bool?, mtp: Bool?) -> (mtp: Bool, acceptance: Bool) {
        let show = available ?? true
        return (show, show && mtp != false)
    }
}

struct ModelSettingsSheet: View {
    let request: ModelSettingsRequest
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.dismiss) private var dismiss

    @State private var override = ModelOverride()
    @State private var addingCustom = false
    @State private var customKey = ""
    @State private var customValue = ""
    @State private var busy = false
    @State private var error: String?

    private var live: ModelInfo? {
        server.allModels.first { request.path.hasSuffix("/" + $0.name) || $0.name == request.path }
    }

    private var plan: ModelSettingsApply.Plan {
        ModelSettingsApply.plan(serverRunning: server.status == .running,
                                loaded: live?.loaded ?? false,
                                isStartupModel: server.currentModelPath == request.path)
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

    private var rows: (mtp: Bool, acceptance: Bool) {
        if isGguf { return (false, false) }
        return ModelSettingsApply.mtpRows(available: mtpAvailable, mtp: override.mtp)
    }

    private var formHeight: CGFloat {
        var n = isGguf ? 1 : 2
        if rows.mtp { n += 1 }
        if rows.acceptance { n += 1 }
        if live?.loaded == true { n += 1 }
        if !isGguf { n += 2 + override.templateKwargs.count + (addingCustom ? 1 : 0) }
        return CGFloat(44 * n + 50)
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
                    ForEach(choices.map(TemplateKwargs.display), id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden().fixedSize()
            } else {
                TextField("value", text: Binding(
                    get: { TemplateKwargs.display(value) },
                    set: { if let v = TemplateKwargs.parse($0) { override.templateKwargs[key] = v } }))
                    .font(.app(.body).monospaced()).frame(width: 140)
            }
            Button { override.templateKwargs[key] = nil } label: { Image(systemName: "xmark") }
                .buttonStyle(.plain).foregroundStyle(.secondary)
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
        case .saveOnly: return "Applied when the model loads."
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
                Picker("Context size", selection: Binding(
                    get: { override.ctxSize ?? -1 },
                    set: { override.ctxSize = $0 < 0 ? nil : $0 })) {
                    Text("Default").tag(-1)
                    ForEach(ContextSizeDisplay.presets, id: \.self) { n in
                        Text(ContextSizeDisplay.formatTokens(n)).tag(n)
                    }
                }
                if !isGguf {
                Picker("KV cache", selection: Binding(
                    get: { override.kvQuant?.rawValue ?? "" },
                    set: { override.kvQuant = KvQuantChoice(rawValue: $0) })) {
                    Text("Default").tag("")
                    ForEach(KvQuantChoice.allCases, id: \.rawValue) { Text($0.label).tag($0.rawValue) }
                }
                }
                if rows.mtp {
                Picker("MTP", selection: Binding(
                    get: { override.mtp.map { $0 ? 1 : 0 } ?? -1 },
                    set: { override.mtp = $0 < 0 ? nil : $0 == 1 })) {
                    Text("Default").tag(-1)
                    Text("On").tag(1)
                    Text("Off").tag(0)
                }
                }
                if rows.acceptance {
                Picker("MTP acceptance", selection: Binding(
                    get: { override.mtpAcceptance?.rawValue ?? "" },
                    set: { override.mtpAcceptance = MtpAcceptanceChoice(rawValue: $0) })) {
                    Text("Default").tag("")
                    ForEach(MtpAcceptanceChoice.allCases, id: \.rawValue) { Text($0.label).tag($0.rawValue) }
                }
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
                                Button("Add", action: commitCustom)
                                    .disabled(customKey.trimmingCharacters(in: .whitespaces).isEmpty || TemplateKwargs.parse(customValue) == nil)
                            }
                        }
                    } header: {
                        HStack {
                            Text("Chat template kwargs")
                            Spacer()
                            Menu {
                                ForEach(TemplateKwargs.known, id: \.key) { k in
                                    Button(k.key) { override.templateKwargs[k.key] = k.choices[0] }
                                        .disabled(override.templateKwargs[k.key] != nil)
                                }
                                Divider()
                                Button("Custom…") { addingCustom = true }
                            } label: {
                                Label("Add", systemImage: "plus")
                            }
                            .menuStyle(.borderlessButton).fixedSize()
                        }
                    } footer: {
                        Text("Forwarded to the model's chat template. Values the request decides (thinking, effort) win.")
                    }
                }
                if let live, live.loaded {
                    LabeledContent("Live") {
                        Text(isGguf ? "\(ContextSizeDisplay.formatTokens(live.contextLength)) context"
                             : "\(ContextSizeDisplay.formatTokens(live.contextLength)) context, KV \(live.kvQuant.isEmpty ? "default" : live.kvQuant)")
                            .foregroundStyle(.secondary)
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
            if let error {
                Text(error).font(.app(.caption)).foregroundStyle(.red).padding(.horizontal, 16)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(L10n.text(plan == .restart ? "Save & Restart" : "Save")) { Task { await save() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(busy)
            }
            .padding(16)
        }
        .frame(width: 440)
        .onAppear { override = ModelSettingsFile.load().override(for: request.path) ?? ModelOverride() }
    }

    private func save() async {
        busy = true
        defer { busy = false }
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
}
