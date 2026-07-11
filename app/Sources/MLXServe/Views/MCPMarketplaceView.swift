import SwiftUI
import AppKit

/// Sheet shown from the MCP toggle pill in ChatView. Two responsibilities:
/// 1. Toggle curated MCP servers on/off and capture their required secrets/args.
/// 2. Provide an escape hatch to edit ~/.mlx-serve/mcp.json directly for custom or advanced servers.
struct MCPMarketplaceView: View {
    @EnvironmentObject var mcpManager: MCPManager
    @Environment(\.dismiss) private var dismiss

    /// Form state, keyed by catalog entry ID. We hydrate from MCPConfig on appear and write back on Save.
    @State private var enabledByID: [String: Bool] = [:]
    @State private var valuesByID: [String: [String: String]] = [:]
    @State private var saveError: String?
    /// Set true after `hydrateFromConfig` runs. Guards `applyAndSpawn` so we don't trigger spawns when
    /// the toggles' initial values get assigned during the first render.
    @State private var hydrated: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    introSection
                    // Only servers this build can actually run: the MAS build
                    // offers just the ones pre-baked into the bundled guest image.
                    ForEach(MCPCatalog.visibleEntries()) { entry in
                        MCPCatalogRow(
                            entry: entry,
                            isEnabled: bindingForEnabled(entry.id),
                            values: bindingForValues(entry.id),
                            status: status(for: entry.id),
                            onToggle: { applyAndSpawn(entryID: entry.id) }
                        )
                    }
                    customServersSection
                }
                .padding(20)
            }
            Divider()
            footer
        }
        .frame(minWidth: 580, minHeight: 600)
        .onAppear(perform: hydrateFromConfig)
        // Re-read mcp.json whenever the user comes back to the app — covers the "edit mcp.json in
        // an external editor, save, switch back" flow so newly-added entries appear without a
        // manual refresh.
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            hydrateFromConfig()
        }
    }

    /// Compact transport tag shown next to a custom server's id. HTTP entries get an explicit pill;
    /// stdio is the default and gets nothing (saves visual noise).
    @ViewBuilder
    private func transportBadge(for entry: MCPServerEntry) -> some View {
        switch entry.transport {
        case .http:
            Text("HTTP")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(Color.blue.opacity(0.18))
                .foregroundStyle(.blue)
                .clipShape(Capsule())
        case .malformed:
            Text("malformed")
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 5).padding(.vertical, 1)
                .background(Color.red.opacity(0.18))
                .foregroundStyle(.red)
                .clipShape(Capsule())
        case .stdio:
            EmptyView()
        }
    }

    private func customIcon(for entry: MCPServerEntry) -> String {
        switch entry.transport {
        case .http: return "globe"
        case .stdio: return "puzzlepiece"
        case .malformed: return "exclamationmark.triangle"
        }
    }

    private func commandSummary(for entry: MCPServerEntry) -> String {
        switch entry.transport {
        case .stdio:
            let cmd = entry.command ?? ""
            let args = (entry.args ?? []).joined(separator: " ")
            return "\(cmd) \(args)"
        case .http:
            return entry.url ?? ""
        case .malformed:
            return "(missing command or url)"
        }
    }

    // MARK: - Sections

    private var header: some View {
        HStack {
            Image(systemName: "puzzlepiece.extension.fill")
                .font(.title2)
                .foregroundStyle(.purple)
            VStack(alignment: .leading, spacing: 2) {
                Text("MCP Marketplace").font(.title3.weight(.semibold))
                Text("Connect external tools via the Model Context Protocol")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var introSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Curated servers")
                .font(.subheadline.weight(.semibold))
            Text("Toggle a server on, fill in any required fields, then click Save. Servers spawn lazily the next time you send a message with MCP mode on.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 4)
    }

    private var customServersSection: some View {
        // Preserve the order users hand-edited in mcp.json — no alphabetic sort.
        // Keyed on the VISIBLE catalog: an entry whose curated row is hidden in
        // this build (e.g. playwright config carried over from a DMG install)
        // surfaces here with its spawn error instead of disappearing.
        let custom = mcpManager.config.mcpServers.filter { id, _ in
            MCPCatalog.visibleEntry(for: id) == nil
        }
        if custom.isEmpty {
            return AnyView(EmptyView())
        }
        return AnyView(
            VStack(alignment: .leading, spacing: 8) {
                Divider().padding(.vertical, 8)
                Text("Custom servers (from mcp.json)")
                    .font(.subheadline.weight(.semibold))
                // OrderedDictionary doesn't conform to RandomAccessCollection directly; .elements does.
                ForEach(custom.elements, id: \.key) { id, entry in
                    HStack(spacing: 8) {
                        Image(systemName: customIcon(for: entry))
                            .foregroundStyle(.secondary)
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(id).font(.body.weight(.medium))
                                transportBadge(for: entry)
                            }
                            Text(commandSummary(for: entry))
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Spacer()
                        if let session = mcpManager.sessions[id] {
                            HStack(spacing: 4) {
                                Circle().fill(.green).frame(width: 8, height: 8)
                                Text("\(session.tools.count) tool\(session.tools.count == 1 ? "" : "s")")
                                    .font(.caption2.weight(.medium))
                                    .foregroundStyle(.green)
                            }
                        } else if let err = mcpManager.startErrors[id] {
                            HStack(spacing: 4) {
                                Circle().fill(.red).frame(width: 8, height: 8)
                                Text("error").font(.caption2.weight(.medium)).foregroundStyle(.red)
                            }
                            .help(err)
                        }
                        Toggle("", isOn: Binding(
                            get: { entry.isEnabled },
                            set: { newValue in
                                var cfg = mcpManager.config
                                cfg.mcpServers[id]?.disabled = !newValue
                                try? mcpManager.saveConfig(cfg)
                                Task { await mcpManager.startEnabled() }
                            }
                        ))
                        .labelsHidden()
                    }
                    .padding(10)
                    .background(Color.secondary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
        )
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Button {
                openMCPJsonInEditor()
            } label: {
                Label("Open mcp.json", systemImage: "doc.text")
            }
            .help("Edit ~/.mlx-serve/mcp.json to add custom servers or fine-tune existing ones")

            if let err = saveError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Save") { saveAndDismiss() }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    // MARK: - Live status

    /// Snapshot of the per-server runtime state for rendering the indicator dot.
    enum RowStatus {
        case off                     // toggle off — gray
        case starting                // toggle on, spawn in flight — yellow
        case connected(toolCount: Int) // session live — green + count
        case failed(detail: String)  // last spawn errored — red + tooltip
    }

    private func status(for id: String) -> RowStatus {
        if let session = mcpManager.sessions[id] {
            return .connected(toolCount: session.tools.count)
        }
        if let err = mcpManager.startErrors[id] {
            return .failed(detail: err)
        }
        if (enabledByID[id] ?? false) && mcpManager.isStarting {
            return .starting
        }
        return .off
    }

    /// Called from the row when the user flips its toggle (or edits a required input). Persists
    /// the change to mcp.json and immediately tries to spawn so the indicator becomes meaningful.
    /// Toggling off tears down the session.
    private func applyAndSpawn(entryID: String) {
        // Skip the noisy initial cascade where `hydrateFromConfig` writes the toggle bindings on first
        // render — we don't want to auto-spawn every enabled server just because the user opened the sheet.
        guard hydrated else { return }
        // Build a partial config update for just this entry from the current form state.
        guard let entry = MCPCatalog.visibleEntry(for: entryID) else { return }
        let isOn = enabledByID[entryID] ?? false
        var newConfig = mcpManager.config
        if isOn {
            // Don't spawn if a required field is still empty — wait for the user to fill it in.
            let values = valuesByID[entryID] ?? [:]
            let missing = entry.inputs.contains { input in
                input.required && (values[input.id]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            }
            if missing {
                // Still persist the toggle so the form stays consistent, but skip spawn.
                newConfig.mcpServers[entryID] = entry.materialize(values: values)
                newConfig.mcpServers[entryID]?.disabled = true
                try? mcpManager.saveConfig(newConfig)
                return
            }
            newConfig.mcpServers[entryID] = entry.materialize(values: values)
        } else if var existing = newConfig.mcpServers[entryID] {
            existing.disabled = true
            newConfig.mcpServers[entryID] = existing
        }
        try? mcpManager.saveConfig(newConfig)
        // Kick off start/stop in the background. startEnabled is idempotent and tears down disabled servers.
        // We don't have access to the active chat session here, so we leave `defaultCwd` as whatever
        // ChatView last set (or nil if MCP was never used in chat yet). The marketplace is most often
        // opened from chat, so this is usually fine.
        Task { await mcpManager.startEnabled() }
    }

    // MARK: - State helpers

    private func bindingForEnabled(_ id: String) -> Binding<Bool> {
        Binding(
            get: { enabledByID[id] ?? false },
            set: { enabledByID[id] = $0 }
        )
    }

    private func bindingForValues(_ id: String) -> Binding<[String: String]> {
        Binding(
            get: { valuesByID[id] ?? [:] },
            set: { valuesByID[id] = $0 }
        )
    }

    private func hydrateFromConfig() {
        mcpManager.reloadConfig()
        for entry in MCPCatalog.visibleEntries() {
            if let server = mcpManager.config.mcpServers[entry.id] {
                enabledByID[entry.id] = server.isEnabled
                valuesByID[entry.id] = entry.extractValues(from: server)
            } else {
                enabledByID[entry.id] = false
                valuesByID[entry.id] = [:]
            }
        }
        // Defer flipping `hydrated` until the next runloop tick so the state-write cascade above has
        // settled before we start treating toggle changes as user actions.
        DispatchQueue.main.async { hydrated = true }
    }

    private func saveAndDismiss() {
        var newConfig = mcpManager.config
        // Visible entries only: Save must never touch config for rows this build
        // doesn't render (their enabledByID is never hydrated, so iterating the
        // full catalog would silently disable a carried-over hidden entry).
        for entry in MCPCatalog.visibleEntries() {
            let isOn = enabledByID[entry.id] ?? false
            let values = valuesByID[entry.id] ?? [:]
            if isOn {
                let missing = entry.inputs.filter { input in
                    input.required && (values[input.id]?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
                }
                if !missing.isEmpty {
                    saveError = "\(entry.name): missing \(missing.map(\.label).joined(separator: ", "))"
                    return
                }
                newConfig.mcpServers[entry.id] = entry.materialize(values: values)
            } else {
                // Toggle off: keep the entry but mark disabled (preserves user-entered tokens for next time).
                if var existing = newConfig.mcpServers[entry.id] {
                    existing.disabled = true
                    newConfig.mcpServers[entry.id] = existing
                }
            }
        }
        do {
            try mcpManager.saveConfig(newConfig)
            saveError = nil
            dismiss()
        } catch {
            saveError = "Failed to save mcp.json: \(error.localizedDescription)"
        }
    }

    private func openMCPJsonInEditor() {
        let path = MCPConfigStore.path
        if !FileManager.default.fileExists(atPath: path) {
            let scaffold = """
            {
              "mcpServers": {

              }
            }
            """
            try? scaffold.write(toFile: path, atomically: true, encoding: .utf8)
        }
        NSWorkspace.shared.open(URL(fileURLWithPath: path))
    }
}

// MARK: - Row

private struct MCPCatalogRow: View {
    let entry: MCPCatalogEntry
    @Binding var isEnabled: Bool
    @Binding var values: [String: String]
    let status: MCPMarketplaceView.RowStatus
    let onToggle: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: entry.icon)
                    .font(.title3)
                    .foregroundStyle(.purple)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.name).font(.body.weight(.semibold))
                    Text(entry.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                statusIndicator
                Toggle("", isOn: $isEnabled)
                    .labelsHidden()
                    .onChange(of: isEnabled) { _, _ in onToggle() }
            }
            if case .failed(let detail) = status {
                Label(detail, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(4)
            }
            if isEnabled {
                if !entry.inputs.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(entry.inputs) { input in
                            inputField(input)
                        }
                    }
                    .padding(.leading, 38)
                }
                if let notes = entry.notes {
                    Text(notes)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.leading, 38)
                }
            }
        }
        .padding(12)
        .background(Color.secondary.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    /// Compact status pill: colored dot + optional tool count or progress spinner.
    @ViewBuilder
    private var statusIndicator: some View {
        switch status {
        case .off:
            EmptyView()
        case .starting:
            HStack(spacing: 4) {
                ProgressView().controlSize(.small)
                Text("starting")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .help("Spawning the MCP server and listing its tools…")
        case .connected(let count):
            HStack(spacing: 4) {
                Circle().fill(.green).frame(width: 8, height: 8)
                Text("\(count) tool\(count == 1 ? "" : "s")")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.green)
            }
            .help("Connected — \(count) tool\(count == 1 ? "" : "s") available")
        case .failed(let detail):
            HStack(spacing: 4) {
                Circle().fill(.red).frame(width: 8, height: 8)
                Text("error")
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.red)
            }
            .help(detail)
        }
    }

    @ViewBuilder
    private func inputField(_ input: MCPCatalogInput) -> some View {
        let binding = Binding(
            get: { values[input.id] ?? "" },
            set: { values[input.id] = $0 }
        )
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 4) {
                Text(input.label)
                    .font(.caption.weight(.medium))
                if input.required {
                    Text("*").foregroundStyle(.red).font(.caption)
                }
            }
            if input.isSecret {
                SecureField(input.placeholder ?? "", text: binding)
                    .textFieldStyle(.roundedBorder)
            } else {
                TextField(input.placeholder ?? "", text: binding)
                    .textFieldStyle(.roundedBorder)
            }
            if let help = input.helpText {
                Text(help).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
