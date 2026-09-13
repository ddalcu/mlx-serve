import SwiftUI

struct ModelSettingsRequest: Identifiable {
    let path: String
    let title: String
    var id: String { path }
}

/// Per-model context / KV quant / MTP (issue #269). Writes the server's
/// `model-settings.json`; a resident model is reloaded (unload + load) so the
/// change applies with the server never leaving `.running`.
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
                Picker("KV cache", selection: Binding(
                    get: { override.kvQuant?.rawValue ?? "" },
                    set: { override.kvQuant = KvQuantChoice(rawValue: $0) })) {
                    Text("Default").tag("")
                    ForEach(KvQuantChoice.allCases, id: \.rawValue) { Text($0.label).tag($0.rawValue) }
                }
                Picker("MTP", selection: Binding(
                    get: { override.mtp.map { $0 ? 1 : 0 } ?? -1 },
                    set: { override.mtp = $0 < 0 ? nil : $0 == 1 })) {
                    Text("Default").tag(-1)
                    Text("On").tag(1)
                    Text("Off").tag(0)
                }
                if let live, live.loaded {
                    LabeledContent("Live") {
                        Text("\(ContextSizeDisplay.formatTokens(live.contextLength)) context, KV \(live.kvQuant.isEmpty ? "default" : live.kvQuant)")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            Text("Applied when the model loads; a resident model is reloaded now.")
                .font(.caption2).foregroundStyle(.secondary)
                .padding(.horizontal, 16)
            if let error {
                Text(error).font(.caption).foregroundStyle(.red).padding(.horizontal, 16)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save") { Task { await save() } }
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
        if server.status == .running, let live, live.loaded {
            do {
                try await server.unloadModel(id: live.name)
                _ = try await server.loadModel(id: request.path, setDefault: request.path == appState.selectedModelPath)
            } catch {
                self.error = "Saved, but the reload failed: \(error.localizedDescription)"
                return
            }
        }
        dismiss()
    }
}
