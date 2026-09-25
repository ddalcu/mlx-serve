import SwiftUI

struct ModelSettingsRequest: Identifiable {
    let path: String
    let title: String
    var id: String { path }
}

/// Per-model context / KV quant / MTP (issue #269). Writes the server's
/// `model-settings.json`, then applies it per `ModelSettingsApply.plan`.
enum ModelSettingsApply {
    enum Plan { case saveOnly, reload, restart, live }
    enum Field: Hashable { case ctxSize, kvQuant, mtp, mtpAcceptance, steering }

    /// The startup model restarts the server: a hot unload + load re-bills
    /// it under `--max-resident-mem`, which the launch load never paid. A
    /// steering-only edit needs neither: `POST /v1/steering` applies it live.
    static func plan(serverRunning: Bool, loaded: Bool, isStartupModel: Bool, changed: Set<Field>) -> Plan {
        guard serverRunning, loaded else { return .saveOnly }
        if changed.isEmpty { return .saveOnly }
        if changed == [.steering] { return .live }
        return isStartupModel ? .restart : .reload
    }

    static func changedFields(from a: ModelOverride, to b: ModelOverride) -> Set<Field> {
        var out: Set<Field> = []
        if a.ctxSize != b.ctxSize { out.insert(.ctxSize) }
        if a.kvQuant != b.kvQuant { out.insert(.kvQuant) }
        if a.mtp != b.mtp { out.insert(.mtp) }
        if a.mtpAcceptance != b.mtpAcceptance { out.insert(.mtpAcceptance) }
        if a.steering != b.steering { out.insert(.steering) }
        return out
    }

    /// The picker's tag for a steering state ("" = inherit, "off", "name:<bank>").
    static func steeringTag(_ s: SteeringOverride?) -> String {
        switch s {
        case nil: return ""
        case .off?: return "off"
        case .configured(let name, _, _)?: return "name:" + name
        }
    }

    /// Picking a bank keeps the scales already entered; a fresh pick starts at ffn 1.
    static func steering(fromTag tag: String, current: SteeringOverride?) -> SteeringOverride? {
        if tag.isEmpty { return nil }
        if tag == "off" { return .off }
        let name = String(tag.dropFirst("name:".count))
        if case .configured(_, let ffn, let attn)? = current { return .configured(name: name, ffn: ffn, attn: attn) }
        return .configured(name: name, ffn: 1, attn: 0)
    }

    /// Picker rows: every registry bank plus the configured name when it is a path or gone.
    static func steeringChoices(registry: [String], current: SteeringOverride?) -> [String] {
        guard let name = current?.name, !registry.contains(name) else { return registry }
        return registry + [name]
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
    @State private var original = ModelOverride()
    @State private var registry: [String] = []
    @State private var busy = false
    @State private var error: String?

    private var live: ModelInfo? {
        server.allModels.first { request.path.hasSuffix("/" + $0.name) || $0.name == request.path }
    }

    private var plan: ModelSettingsApply.Plan {
        ModelSettingsApply.plan(serverRunning: server.status == .running,
                                loaded: live?.loaded ?? false,
                                isStartupModel: server.currentModelPath == request.path,
                                changed: ModelSettingsApply.changedFields(from: original, to: override))
    }

    /// A running server answers from `/v1/models`; otherwise the app's own disk probe.
    private var mtpAvailable: Bool? {
        if let a = live?.mtpAvailable { return a }
        return appState.localModels.first { $0.path == request.path }?.hasMtpHead
    }

    /// A running server answers from `/v1/models`; otherwise the app's own disk probe.
    private var steeringAvailable: Bool {
        if let a = live?.architecture, !a.isEmpty { return SteeringQuickSet.archSupportsSteering(a) }
        let probed = appState.localModels.first { $0.path == request.path }?.modelType ?? ""
        return SteeringQuickSet.archSupportsSteering(probed)
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
        if !isGguf, steeringAvailable { n += 1 }
        if rows.mtp { n += 1 }
        if rows.acceptance { n += 1 }
        if case .configured? = override.steering { n += 2 }
        if live?.loaded == true { n += 1 }
        return CGFloat(44 * n + 50)
    }

    private var footnote: String {
        switch plan {
        case .saveOnly: return "Applied when the model loads."
        case .reload: return "Applied when the model loads; the resident model is reloaded now."
        case .restart: return "Applied when the model loads; the server is restarted now."
        case .live: return "Steering applies to the resident model now, without a reload."
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
                if !isGguf, steeringAvailable {
                Picker("Steering", selection: Binding(
                    get: { ModelSettingsApply.steeringTag(override.steering) },
                    set: { override.steering = ModelSettingsApply.steering(fromTag: $0, current: override.steering) })) {
                    Text("Default (launch flags)").tag("")
                    Text("Off").tag("off")
                    ForEach(ModelSettingsApply.steeringChoices(registry: registry, current: override.steering), id: \.self) { name in
                        Text(name).tag("name:" + name)
                    }
                }
                .help("Direction banks in ~/.mlx-serve/steering (<name>.f32, one unit-norm row per layer)")
                if case .configured(let name, let ffn, let attn)? = override.steering {
                    // Clamped: the server drops the whole key on an out-of-range scale.
                    TextField("FFN scale", value: Binding(
                        get: { ffn },
                        set: { override.steering = .configured(name: name, ffn: SteeringQuickSet.clamped($0), attn: attn) }), format: .number)
                        .help("Positive removes the direction, negative amplifies it (-100…100)")
                    TextField("Attention scale", value: Binding(
                        get: { attn },
                        set: { override.steering = .configured(name: name, ffn: ffn, attn: SteeringQuickSet.clamped($0)) }), format: .number)
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
            original = override
            registry = SteeringRegistry.names()
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
        case .live:
            do {
                try await server.setSteering(id: live!.name, override.steering)
            } catch {
                self.error = "Saved, but the live apply failed: \(error.localizedDescription)"
                return
            }
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
