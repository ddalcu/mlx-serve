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

    /// The SSD budget row exists only for a row the server marks `streaming`; every other
    /// model ignores the setting, so offering it would be a knob that does nothing.
    static func ssdBudgetRow(streaming: Bool) -> Bool { streaming }
}

struct ModelSettingsSheet: View {
    let request: ModelSettingsRequest
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.dismiss) private var dismiss

    @State private var override = ModelOverride()
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

    private var showSsdBudget: Bool { ModelSettingsApply.ssdBudgetRow(streaming: live?.streaming ?? false) }

    private var formHeight: CGFloat {
        var n = isGguf ? 1 : 2
        if rows.mtp { n += 1 }
        if rows.acceptance { n += 1 }
        if showSsdBudget { n += 1 }
        if live?.loaded == true { n += 1 }
        return CGFloat(44 * n + 50)
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
                    Text("Model Settings").font(.title3.weight(.semibold))
                    Text(request.title).font(.caption).foregroundStyle(.secondary).lineLimit(1)
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
                if showSsdBudget {
                Picker("SSD budget", selection: Binding(
                    get: { override.ssdBudgetGB ?? -1 },
                    set: { override.ssdBudgetGB = $0 < 0 ? nil : $0 })) {
                    Text("None").tag(-1)
                    ForEach(SsdBudgetAdvice.livePresets, id: \.self) { n in
                        Text(SsdBudgetAdvice.label(n, recommended: SsdBudgetAdvice.liveRecommendedGiB)).tag(n)
                    }
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
            Text(L10n.text(footnote))
                .font(.caption2).foregroundStyle(.secondary)
                .padding(.horizontal, 16)
            if let error {
                Text(error).font(.caption).foregroundStyle(.red).padding(.horizontal, 16)
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
        .onAppear {
            override = ModelSettingsFile.load().override(for: request.path) ?? ModelOverride()
            if showSsdBudget && override.ssdBudgetGB == nil {
                override.ssdBudgetGB = SsdBudgetAdvice.liveRecommendedGiB
            }
        }
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
