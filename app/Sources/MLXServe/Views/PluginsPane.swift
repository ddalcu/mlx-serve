import SwiftUI
import AppKit

/// Models > Plugins: every outside-plugin control in one place. The master switch and a switch per installed
/// plugin, the catalog of what the CommunityServe/plugins repo offers (Install / Update / Remove, compiled here after
/// you confirm the commit), and instructions to paste to a coding agent. The server reads the switches at startup.
struct PluginsPane: View {
    @StateObject private var manager = PluginManager()
    @State private var switches = OutsidePlugins.loadSwitches()
    @State private var saveError: String?
    @State private var copiedName: String?

    private var binaryPath: String { PluginManager.binaryPath() }

    var body: some View {
        if BuildFeatures.current.isMAS {
            // The Mac App Store sandbox forbids loading outside code.
            Text("Outside plugins are not available in the Mac App Store build. Get the direct download to use them.")
                .font(.app(.body)).foregroundStyle(.secondary)
                .padding(24).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        } else {
            content
        }
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                masterSwitch
                if !manager.receipts.isEmpty { installedSection }
                availableSection
                PluginInstallPanel(manager: manager)
                agentSection
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .task { await manager.loadCatalog() }
    }

    // MARK: Master switch

    private var masterSwitch: some View {
        card(title: "Outside plugins", subtitle: "Model, media and quant-format plugins from the community. A plugin is native code that runs with your rights. Restart the server after any change.",
             systemImage: "puzzlepiece.extension", tint: .purple) {
            HStack {
                Text("Load outside plugins at server start").font(.app(.body))
                Spacer()
                Toggle("", isOn: Binding(get: { switches.enabled }, set: { switches.enabled = $0; save() }))
                    .labelsHidden().toggleStyle(.switch)
            }
            .padding(10)
            if let saveError {
                Text(saveError).font(.app(.caption)).foregroundStyle(.red).padding([.horizontal, .bottom], 10)
            }
        }
    }

    // MARK: Installed

    private var installedSection: some View {
        card(title: "Installed", subtitle: "In ~/.mlx-serve/plugins. Each is checked against its recorded hash when the server loads it.",
             systemImage: "checkmark.seal", tint: .green) {
            ForEach(manager.receipts) { r in
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(r.name).font(.app(.body).weight(.medium))
                        Spacer()
                        Button("Remove", role: .destructive) { manager.uninstall(r.name) }
                            .disabled(manager.isBusy)
                        Toggle("", isOn: Binding(get: { switches.isOn(r.name) }, set: { switches.set(r.name, on: $0); save() }))
                            .labelsHidden().toggleStyle(.switch)
                            .disabled(!switches.enabled)
                    }
                    Text(OutsidePlugins.summary(r)).font(.app(.caption2)).foregroundStyle(.secondary)
                }
                .padding(10)
            }
        }
    }

    // MARK: Catalog

    private var availableSection: some View {
        card(title: "Available", subtitle: "From github.com/CommunityServe/plugins. Install downloads the source, shows you the commit, then compiles it on this Mac with a pinned Zig. Needs the Xcode Command Line Tools.",
             systemImage: "square.and.arrow.down", tint: .blue) {
            HStack(spacing: 8) {
                if manager.loadingCatalog { ProgressView().controlSize(.small) }
                if let error = manager.catalogError {
                    Text(error).font(.app(.caption)).foregroundStyle(.red)
                } else if manager.catalog.isEmpty && !manager.loadingCatalog {
                    Text("No plugins listed.").font(.app(.caption)).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Refresh") { Task { await manager.loadCatalog() } }
                    .disabled(manager.loadingCatalog || manager.isBusy)
            }
            .padding(10)
            ForEach(manager.catalog) { entry in
                PluginCatalogRow(manager: manager, entry: entry, binaryPath: binaryPath, copied: $copiedName)
            }
        }
    }

    // MARK: Agent

    private var agentSection: some View {
        let text = PluginAgentInstructions.generic(binaryPath: binaryPath, runningVersion: manager.runningVersion)
        return card(title: "Install with an AI agent", subtitle: "Paste this into a coding agent (pi, Claude Code, Codex...) to have it install a plugin safely. Each plugin above also has its own copy button.",
                    systemImage: "sparkles", tint: .orange) {
            VStack(alignment: .leading, spacing: 8) {
                ScrollView {
                    Text(text)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
                Button(copiedName == "generic" ? "Copied" : "Copy instructions") { copy(text, as: "generic") }
            }
            .padding(10)
        }
    }

    private func copy(_ text: String, as name: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copiedName = name
    }

    private func save() {
        do {
            try OutsidePlugins.save(switches)
            saveError = nil
        } catch {
            saveError = error.localizedDescription
        }
    }

    private func card<Content: View>(title: String, subtitle: String, systemImage: String, tint: Color,
                                     @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: systemImage).foregroundStyle(tint).frame(width: 18)
                VStack(alignment: .leading, spacing: 1) {
                    Text(L10n.text(title)).font(.app(.subheadline).weight(.semibold))
                    Text(L10n.text(subtitle)).font(.app(.caption2)).foregroundStyle(.secondary)
                }
                Spacer()
            }
            VStack(alignment: .leading, spacing: 0) { content() }
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

private struct PluginCatalogRow: View {
    @ObservedObject var manager: PluginManager
    let entry: PluginCatalog.Entry
    let binaryPath: String
    @Binding var copied: String?

    private var status: PluginCatalog.Status { PluginCatalog.status(of: entry, receipts: manager.receipts) }
    private var compatible: Bool { PluginCatalog.runs(on: manager.runningVersion, requirement: entry.requiresMlxServe) }

    private var detail: String {
        var text = "\(entry.kind) \(entry.version)"
        if !entry.requiresMlxServe.isEmpty { text += " · needs mlx-serve \(entry.requiresMlxServe)" }
        if !compatible { text += " · this mlx-serve (\(manager.runningVersion ?? "?")) is too old" }
        return text
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(entry.name).font(.app(.body).weight(.medium))
                Spacer()
                Button {
                    let text = PluginAgentInstructions.text(for: entry, binaryPath: binaryPath, runningVersion: manager.runningVersion)
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    copied = entry.name
                } label: {
                    Label(copied == entry.name ? "Copied" : "Agent instructions", systemImage: "doc.on.doc")
                }
                .help("Copy instructions for a coding agent to install this plugin")
                switch status {
                case .notInstalled:
                    Button("Install") { manager.prepare(entry) }
                        .disabled(manager.isBusy || !compatible)
                case .updateAvailable(let installed):
                    Button(L10n.format("Update from %@", installed.isEmpty ? "–" : installed)) { manager.prepare(entry) }
                        .disabled(manager.isBusy || !compatible)
                case .installed:
                    Text("Installed").font(.app(.caption)).foregroundStyle(.secondary)
                }
            }
            if !entry.summary.isEmpty {
                Text(entry.summary).font(.app(.caption)).foregroundStyle(.secondary)
            }
            Text(detail).font(.app(.caption2)).foregroundStyle(.secondary)
            ForEach(entry.models, id: \.self) { m in
                Text("• " + m).font(.app(.caption2)).foregroundStyle(.tertiary)
            }
        }
        .padding(10)
    }
}

/// The install in progress: the confirmation of the fetched commit, then the build log, then the outcome.
private struct PluginInstallPanel: View {
    @ObservedObject var manager: PluginManager

    var body: some View {
        switch manager.phase {
        case .idle:
            EmptyView()
        case .fetching(let name):
            progress(L10n.format("Fetching the source for %@…", name))
        case .awaitingConfirmation(let name, let commit, let version):
            VStack(alignment: .leading, spacing: 8) {
                Text(L10n.format("Compile and install %@ %@?", name, version)).font(.app(.body).bold())
                Text(L10n.format("Source: %@ at commit %@. A plugin is native code that runs with your rights: install it only if you trust this code. It will be compiled on this Mac and installed into ~/.mlx-serve/plugins.", PluginCatalog.repoURL, commit))
                    .font(.app(.caption))
                    .textSelection(.enabled)
                ScrollView {
                    Text(manager.manifestText(for: name))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 140)
                HStack {
                    Button("Compile and install") { manager.confirmInstall() }
                        .keyboardShortcut(.defaultAction)
                    Button("Cancel") { manager.cancel() }
                }
            }
            logView
        case .building(let name):
            progress(L10n.format("Compiling %@ (the first build also downloads Zig)…", name))
            logView
        case .done(let name):
            VStack(alignment: .leading, spacing: 6) {
                Text(L10n.format("%@ installed. Turn on Outside plugins above, then restart the server.", name))
                    .font(.app(.caption)).foregroundStyle(.green)
                Button("Dismiss") { manager.dismiss() }
            }
        case .failed(let name, let message):
            VStack(alignment: .leading, spacing: 6) {
                Text(L10n.format("%@: %@", name, message)).font(.app(.caption)).foregroundStyle(.red)
                Button("Dismiss") { manager.dismiss() }
            }
            logView
        }
    }

    private func progress(_ text: String) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(text).font(.app(.caption))
            Spacer()
            Button("Cancel") { manager.cancel() }
        }
    }

    @ViewBuilder private var logView: some View {
        if !manager.log.isEmpty {
            ScrollView {
                Text(manager.log)
                    .font(.system(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 180)
        }
    }
}
