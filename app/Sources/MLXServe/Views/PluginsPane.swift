import SwiftUI
import AppKit

/// Models > Plugins: the plugins as a list (installed, then what the CommunityServe/plugins catalog offers) beside
/// the selected plugin's page: what it does, whether this Mac meets its needs, its packs (on disk or a download),
/// Install / Update / Remove and its switch, the install log, and instructions for a coding agent. The server reads
/// the switches at startup. Pure rules: `PluginListing`, `PluginCatalog`.
struct PluginsPane: View {
    @StateObject private var manager = PluginManager()
    @State private var switches = OutsidePlugins.loadSwitches()
    @State private var saveError: String?
    @State private var selection: PluginListing.Selection?

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
        let lists = PluginListing.items(receipts: manager.receipts, catalog: manager.catalog)
        let current = PluginListing.resolve(selection, lists)
        return VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                PluginList(lists: lists, selection: current, switches: switches, manager: manager) { selection = $0 }
                    .frame(width: 270)
                Divider()
                PluginDetail(selection: current, lists: lists, switches: switches, manager: manager,
                             setSwitch: { name, on in switches.set(name, on: on); save() })
            }
        }
        .task { await manager.loadCatalog() }
    }

    private var header: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Outside plugins").font(.app(.sectionTitle, weight: .semibold))
                Text("Community code, compiled on this Mac. Changes apply when the server restarts.")
                    .font(.app(.explainer)).foregroundStyle(.secondary)
                if let saveError { Text(saveError).font(.app(.annotation)).foregroundStyle(.red) }
            }
            Spacer()
            if manager.loadingCatalog { ProgressView().controlSize(.small) }
            Button { Task { await manager.loadCatalog() } } label: {
                Label { Text("Refresh").font(.app(.value)) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .disabled(manager.loadingCatalog || manager.isBusy)
            Toggle(isOn: Binding(get: { switches.enabled }, set: { switches.enabled = $0; save() })) {
                Text("Load at server start").font(.app(.value))
            }
            .toggleStyle(.switch)
        }
        .padding(.horizontal, 18).padding(.vertical, 12)
    }

    private func save() {
        do {
            try OutsidePlugins.save(switches)
            saveError = nil
        } catch {
            saveError = error.localizedDescription
        }
    }
}

// MARK: - List

private struct PluginList: View {
    let lists: PluginListing.Lists
    let selection: PluginListing.Selection
    let switches: OutsidePlugins.Switches
    @ObservedObject var manager: PluginManager
    let select: (PluginListing.Selection) -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 2) {
                if !lists.installed.isEmpty {
                    heading("Installed")
                    ForEach(lists.installed) { row($0) }
                }
                heading("Available")
                if lists.available.isEmpty {
                    Text(manager.catalogError ?? (manager.loadingCatalog ? L10n.text("Loading…") : L10n.text("Nothing more to install.")))
                        .font(.app(.annotation)).foregroundStyle(manager.catalogError == nil ? Color.secondary : Color.red)
                        .padding(.horizontal, 10).padding(.vertical, 4)
                }
                ForEach(lists.available) { row($0) }
                Divider().padding(.vertical, 8)
                PluginListRow(selected: selection == .agent, action: { select(.agent) }) {
                    Image(systemName: "sparkles").foregroundStyle(.orange).frame(width: 28, height: 28)
                    Text("Install with an AI agent").font(.app(.rowTitle))
                    Spacer(minLength: 0)
                }
            }
            .padding(8)
        }
    }

    private func heading(_ title: String) -> some View {
        Text(L10n.text(title)).font(.app(.annotation, weight: .semibold)).foregroundStyle(.secondary)
            .padding(.horizontal, 10).padding(.top, 8).padding(.bottom, 2)
    }

    private func row(_ item: PluginListing.Item) -> some View {
        let state = PluginListing.state(item, switches: switches, running: manager.runningVersion)
        return PluginListRow(selected: selection == .plugin(item.name), action: { select(.plugin(item.name)) }) {
            PluginMonogram(name: item.name, kind: item.kind, size: 28)
            VStack(alignment: .leading, spacing: 1) {
                Text(item.name).font(.app(.rowTitle, weight: .medium)).lineLimit(1)
                Text(PluginLabels.kind(item.kind)).font(.app(.annotation)).foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            if manager.phase.pluginName == item.name, manager.isBusy {
                ProgressView().controlSize(.small)
            } else {
                PluginStateBadge(state: state, compact: true)
            }
        }
    }
}

/// One row: the button covers the whole highlight, filled like every sidebar row (`SidebarRowStyle`). The label is
/// stored built, not as a closure: SwiftUI cannot see a stored closure change, so a row kept its install spinner.
private struct PluginListRow<Label: View>: View {
    let selected: Bool
    let action: () -> Void
    @ViewBuilder let label: Label
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) { label }
                .padding(.horizontal, 8).padding(.vertical, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .background(SidebarRowStyle.fill(selected: selected, hovering: hovering), in: RoundedRectangle(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

// MARK: - Detail

private struct PluginDetail: View {
    let selection: PluginListing.Selection
    let lists: PluginListing.Lists
    let switches: OutsidePlugins.Switches
    @ObservedObject var manager: PluginManager
    let setSwitch: (String, Bool) -> Void

    var body: some View {
        ScrollView {
            Group {
                switch selection {
                case .agent:
                    PluginAgentPage(manager: manager)
                case .plugin(let name):
                    if let item = (lists.installed + lists.available).first(where: { $0.name == name }) {
                        PluginPage(item: item, switches: switches, manager: manager, setSwitch: setSwitch)
                            .id(item.name)
                    }
                }
            }
            .padding(22)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct PluginPage: View {
    let item: PluginListing.Item
    let switches: OutsidePlugins.Switches
    @ObservedObject var manager: PluginManager
    let setSwitch: (String, Bool) -> Void
    @EnvironmentObject var downloads: DownloadManager
    @State private var onDisk: Set<String> = []
    @State private var copied = false

    private var state: PluginListing.State { PluginListing.state(item, switches: switches, running: manager.runningVersion) }
    private var packs: [PluginListing.Pack] { item.entry.map(PluginListing.packs) ?? [] }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            title
            if let summary = item.entry?.summary, !summary.isEmpty {
                Text(summary).font(.app(.explainer)).fixedSize(horizontal: false, vertical: true)
            }
            if manager.phase.pluginName == item.name { PluginInstallPanel(manager: manager) }
            needs
            if !packs.isEmpty { packList }
            if let receipt = item.receipt { details(receipt) }
            agentFooter
        }
        .task(id: item.name) { refreshOnDisk() }
    }

    private var title: some View {
        HStack(alignment: .center, spacing: 14) {
            PluginMonogram(name: item.name, kind: item.kind, size: 46)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.name).font(.app(.sectionTitle, weight: .semibold))
                HStack(spacing: 8) {
                    Text(PluginLabels.kind(item.kind)).font(.app(.annotation)).foregroundStyle(.secondary)
                    if let version = item.receipt?.version ?? item.entry?.version, !version.isEmpty {
                        Text(version).font(.app(.annotation, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    PluginStateBadge(state: state, compact: false)
                }
            }
            Spacer()
            actions
        }
    }

    @ViewBuilder private var actions: some View {
        switch state {
        case .available, .needsNewer:
            Button { if let e = item.entry { manager.prepare(e) } } label: { Text("Install").font(.app(.value, weight: .medium)) }
                .buttonStyle(.borderedProminent)
                .disabled(manager.isBusy || state != .available)
        case .updateAvailable(let version):
            Button { if let e = item.entry { manager.prepare(e) } } label: {
                Text(L10n.format("Update to %@", version)).font(.app(.value, weight: .medium))
            }
            .buttonStyle(.borderedProminent)
            .disabled(manager.isBusy)
            removeButton
        case .on, .off:
            Toggle(isOn: Binding(get: { switches.isOn(item.name) }, set: { setSwitch(item.name, $0) })) {
                Text("Enabled").font(.app(.value))
            }
            .toggleStyle(.switch)
            .disabled(!switches.enabled)
            .help(switches.enabled ? "" : L10n.text("Turn on Load at server start first"))
            removeButton
        }
    }

    private var removeButton: some View {
        Button(role: .destructive) { manager.uninstall(item.name) } label: { Text("Remove").font(.app(.value)) }
            .disabled(manager.isBusy)
    }

    // MARK: What it needs

    private var needs: some View {
        PluginSection(title: "Needs") {
            if let requirement = item.entry?.requiresMlxServe, !requirement.isEmpty {
                let met = PluginCatalog.runs(on: manager.runningVersion, requirement: requirement)
                PluginCheckRow(met: met, title: L10n.format("mlx-serve %@", requirement),
                               detail: L10n.format("This Mac runs %@", manager.runningVersion ?? "?"))
            }
            if item.receipt == nil {
                PluginCheckRow(met: nil, title: L10n.text("Xcode Command Line Tools"),
                               detail: L10n.text("Install compiles the source here with a pinned Zig"))
            } else {
                PluginCheckRow(met: switches.enabled, title: L10n.text("Plugins load at server start"),
                               detail: switches.enabled ? L10n.text("On") : L10n.text("Off: turn on Load at server start above"))
            }
        }
    }

    // MARK: Packs

    private var packList: some View {
        PluginSection(title: "Packs it serves") {
            ForEach(packs, id: \.text) { pack in
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(pack.text).font(.app(.rowTitle))
                        if let repo = pack.repo {
                            Text(repo).font(.app(.annotation, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                    }
                    Spacer()
                    if let repo = pack.repo { packStatus(repo) }
                }
                .padding(.vertical, 4)
            }
        }
    }

    @ViewBuilder private func packStatus(_ repo: String) -> some View {
        if onDisk.contains(repo) {
            Label { Text("On this Mac").font(.app(.value)) } icon: { Image(systemName: "checkmark.circle.fill").foregroundStyle(.green) }
        } else if let d = downloads.downloads[repo], d.status == .downloading {
            HStack(spacing: 8) {
                ProgressView(value: d.progress).frame(width: 90)
                Text("\(Int(d.progress * 100))%").font(.app(.value).monospacedDigit()).foregroundStyle(.secondary)
            }
        } else {
            Button { downloads.start(repoId: repo) { refreshOnDisk() } } label: {
                Label { Text("Download").font(.app(.value)) } icon: { Image(systemName: "arrow.down.circle") }
            }
        }
    }

    private func refreshOnDisk() {
        onDisk = Set(packs.compactMap(\.repo).filter { ServerManager.resolveModelDir(repo: $0) != nil })
    }

    // MARK: Details

    private func details(_ receipt: OutsidePlugins.Receipt) -> some View {
        PluginSection(title: "Installed file") {
            PluginFactRow(label: "Signed", value: receipt.signature.isEmpty ? L10n.text("unknown") : receipt.signature)
            PluginFactRow(label: "SHA-256", value: receipt.sha256, mono: true)
            if !receipt.source.isEmpty { PluginFactRow(label: "Source", value: receipt.source, mono: true) }
            Text("Checked against this hash every time the server loads it.").font(.app(.annotation)).foregroundStyle(.secondary)
        }
    }

    private var agentFooter: some View {
        HStack(spacing: 10) {
            Text("Rather have a coding agent install it?").font(.app(.annotation)).foregroundStyle(.secondary)
            Button {
                guard let e = item.entry else { return }
                let text = PluginAgentInstructions.text(for: e, binaryPath: PluginManager.binaryPath(), runningVersion: manager.runningVersion)
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                copied = true
            } label: {
                Label { Text(copied ? "Copied" : "Copy agent instructions").font(.app(.value)) } icon: { Image(systemName: "doc.on.doc") }
            }
            .disabled(item.entry == nil)
        }
    }
}

/// The catalog-wide prompt for a coding agent, on its own page rather than under the list.
private struct PluginAgentPage: View {
    @ObservedObject var manager: PluginManager
    @State private var copied = false

    var body: some View {
        let text = PluginAgentInstructions.generic(binaryPath: PluginManager.binaryPath(), runningVersion: manager.runningVersion)
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "sparkles").font(.app(.sectionTitle)).foregroundStyle(.orange)
                Text("Install with an AI agent").font(.app(.sectionTitle, weight: .semibold))
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    copied = true
                } label: {
                    Label { Text(copied ? "Copied" : "Copy instructions").font(.app(.value, weight: .medium)) } icon: { Image(systemName: "doc.on.doc") }
                }
                .buttonStyle(.borderedProminent)
            }
            Text("Paste this into a coding agent (pi, Claude Code, Codex...) to have it install a plugin safely. Each plugin's page also has its own copy button.")
                .font(.app(.explainer)).foregroundStyle(.secondary)
            Text(text)
                .font(.app(.caption, design: .monospaced))
                .textSelection(.enabled)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

// MARK: - Pieces

private enum PluginLabels {
    static func kind(_ kind: String) -> String {
        switch kind {
        case "model": return L10n.text("Model")
        case "media": return L10n.text("Media")
        case "quant": return L10n.text("Quant format")
        default: return kind
        }
    }
}

/// A plugin has no artwork: its initial on its kind's colour.
private struct PluginMonogram: View {
    let name: String
    let kind: String
    let size: CGFloat

    private var tint: Color {
        switch kind {
        case "model": return .purple
        case "media": return .teal
        case "quant": return .orange
        default: return .gray
        }
    }

    var body: some View {
        Text(String(name.prefix(1)).uppercased())
            .font(.app(size > 32 ? .sectionTitle : .rowTitle, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(tint.gradient, in: RoundedRectangle(cornerRadius: size * 0.26))
    }
}

private struct PluginStateBadge: View {
    let state: PluginListing.State
    let compact: Bool

    private var look: (text: String, color: Color) {
        switch state {
        case .on: return (L10n.text("On"), .green)
        case .off: return (L10n.text("Off"), .secondary)
        case .updateAvailable(let v): return (compact ? L10n.text("Update") : L10n.format("Update to %@", v), .orange)
        case .available: return (L10n.text("Not installed"), .blue)
        case .needsNewer(let r): return (compact ? L10n.text("Needs update") : L10n.format("Needs mlx-serve %@", r), .red)
        }
    }

    var body: some View {
        let look = self.look
        HStack(spacing: 5) {
            Circle().fill(look.color).frame(width: 7, height: 7)
            Text(look.text).font(.app(.annotation, weight: .medium)).foregroundStyle(look.color == .secondary ? Color.secondary : look.color)
        }
        .padding(.horizontal, compact ? 0 : 8).padding(.vertical, compact ? 0 : 3)
        .background(compact ? Color.clear : look.color.opacity(0.12), in: Capsule())
    }
}

private struct PluginSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.text(title)).font(.app(.annotation, weight: .semibold)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) { content }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}

/// A requirement: met (green), not met (red), or one this Mac cannot check up front (nil).
private struct PluginCheckRow: View {
    let met: Bool?
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: met == nil ? "info.circle" : (met! ? "checkmark.circle.fill" : "xmark.circle.fill"))
                .foregroundStyle(met == nil ? Color.secondary : (met! ? Color.green : Color.red))
            Text(title).font(.app(.rowTitle))
            Text(detail).font(.app(.annotation)).foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
    }
}

private struct PluginFactRow: View {
    let label: String
    let value: String
    var mono = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(L10n.text(label)).font(.app(.annotation)).foregroundStyle(.secondary).frame(width: 70, alignment: .leading)
            Text(value).font(mono ? .app(.annotation, design: .monospaced) : .app(.value)).textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }
}

private extension PluginManager.Phase {
    /// The plugin an install step is about, if any.
    var pluginName: String? {
        switch self {
        case .idle: return nil
        case .fetching(let n), .building(let n), .done(let n): return n
        case .awaitingConfirmation(let n, _, _), .failed(let n, _): return n
        }
    }
}

/// The install in progress: the confirmation of the fetched commit, then the build log, then the outcome.
private struct PluginInstallPanel: View {
    @ObservedObject var manager: PluginManager

    var body: some View {
        VStack(alignment: .leading, spacing: 10) { phaseView }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder private var phaseView: some View {
        switch manager.phase {
        case .idle:
            EmptyView()
        case .fetching(let name):
            progress(L10n.format("Fetching the source for %@…", name))
        case .awaitingConfirmation(let name, let commit, let version):
            Text(L10n.format("Compile and install %@ %@?", name, version)).font(.app(.rowTitle, weight: .semibold))
            Text(L10n.format("Source: %@ at commit %@. A plugin is native code that runs with your rights: install it only if you trust this code. It will be compiled on this Mac and installed into ~/.mlx-serve/plugins.", PluginCatalog.repoURL, commit))
                .font(.app(.annotation))
                .textSelection(.enabled)
            ScrollView {
                Text(manager.manifestText(for: name))
                    .font(.app(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 140)
            HStack {
                Button { manager.confirmInstall() } label: { Text("Compile and install").font(.app(.value, weight: .medium)) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                Button { manager.cancel() } label: { Text("Cancel").font(.app(.value)) }
            }
            logView
        case .building(let name):
            progress(L10n.format("Compiling %@ (the first build also downloads Zig)…", name))
            logView
        case .done(let name):
            Text(L10n.format("%@ installed. Switch it on, then restart the server.", name))
                .font(.app(.annotation)).foregroundStyle(.green)
            Button { manager.dismiss() } label: { Text("Dismiss").font(.app(.value)) }
        case .failed(let name, let message):
            Text(L10n.format("%@: %@", name, message)).font(.app(.annotation)).foregroundStyle(.red).textSelection(.enabled)
            Button { manager.dismiss() } label: { Text("Dismiss").font(.app(.value)) }
            logView
        }
    }

    private func progress(_ text: String) -> some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text(text).font(.app(.annotation))
            Spacer()
            Button { manager.cancel() } label: { Text("Cancel").font(.app(.value)) }
        }
    }

    @ViewBuilder private var logView: some View {
        if !manager.log.isEmpty {
            ScrollView {
                Text(manager.log)
                    .font(.app(.caption2, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 180)
        }
    }
}
