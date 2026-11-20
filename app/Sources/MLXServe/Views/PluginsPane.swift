import SwiftUI
import AppKit

/// Models > Plugins: the plugins as a list (installed, then what the CommunityServe/plugins catalog offers) beside
/// the selected plugin's page: what it does, one Install that builds it and switches it on, its packs (on disk or a
/// download), what this Mac needs, and a restart when the running server has not applied a change. The server reads
/// the switches at startup. Pure rules: `PluginListing`, `PluginCatalog`.
struct PluginsPane: View {
    var restartServer: () -> Void = {}
    @StateObject private var manager = PluginManager()
    @EnvironmentObject private var server: ServerManager
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

    private var context: PluginContext {
        PluginContext(switches: switches, serverUp: server.status == .running,
                      install: { entries in
                          manager.install(entries) { names in
                              switches = switches.turningOn(names, installed: manager.receipts.map(\.name))
                              save()
                          }
                      },
                      setSwitch: { name, on in
                          if on { switches = switches.turningOn([name], installed: manager.receipts.map(\.name)) } else { switches.set(name, on: false) }
                          save()
                      },
                      setOption: { plugin, name, value in
                          switches.setOption(plugin, name, value)
                          save()
                      },
                      restartServer: restartServer)
    }

    private var content: some View {
        let lists = PluginListing.items(receipts: manager.receipts, catalog: manager.catalog)
        let current = PluginListing.resolve(selection, lists)
        return VStack(spacing: 0) {
            header(lists)
            Divider()
            HStack(spacing: 0) {
                PluginList(lists: lists, selection: current, context: context, manager: manager) { selection = $0 }
                    .frame(width: 300)
                Divider()
                PluginDetail(selection: current, lists: lists, context: context, manager: manager)
            }
        }
        .task { await manager.loadCatalog(port: server.port) }
        .onChange(of: server.status) { _, status in
            guard status == .running else { return }
            Task { await manager.refreshServerStatus(port: server.port) }
        }
    }

    private func header(_ lists: PluginListing.Lists) -> some View {
        HStack(spacing: 14) {
            AsyncImage(url: PluginCatalog.logoURL) { phase in
                if let image = phase.image {
                    image.resizable().interpolation(.high).frame(width: 40, height: 40)
                        .clipShape(RoundedRectangle(cornerRadius: 9))
                } else {
                    RoundedRectangle(cornerRadius: 9).fill(.quaternary).frame(width: 40, height: 40)
                }
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("Outside plugins").font(.app(.sectionTitle, weight: .semibold))
                Text("Community add-ons that teach mlx-serve new models. Each one is built from source on this Mac.")
                    .font(.app(.explainer)).foregroundStyle(.secondary)
                if let saveError { Text(saveError).font(.app(.annotation)).foregroundStyle(.red) }
            }
            Spacer()
            if manager.loadingCatalog { ProgressView().controlSize(.small) }
            let stale = PluginListing.rebuildable(lists.installed, server: manager.serverStatus)
            if !stale.isEmpty {
                Button { context.install(stale) } label: { Text("Rebuild all").font(.app(.value, weight: .medium)) }
                    .buttonStyle(.borderedProminent)
                    .disabled(manager.isBusy || manager.phase != .idle)
                    .help(L10n.text("Rebuild every plugin this mlx-serve refused as built for another version"))
            }
            Button { Task { await manager.loadCatalog(port: server.port) } } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help(L10n.text("Refresh"))
            .disabled(manager.loadingCatalog || manager.isBusy)
            if !lists.installed.isEmpty {
                Toggle(isOn: Binding(get: { switches.enabled }, set: { switches.enabled = $0; save() })) {
                    Text("Load plugins").font(.app(.value))
                }
                .toggleStyle(.switch)
                .help(L10n.text("Off: the server starts without any outside plugin"))
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
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

/// What every page needs from the pane: the switches, whether a server is up to apply them, and the actions.
private struct PluginContext {
    let switches: OutsidePlugins.Switches
    let serverUp: Bool
    let install: ([PluginCatalog.Entry]) -> Void
    let setSwitch: (String, Bool) -> Void
    let setOption: (_ plugin: String, _ name: String, _ value: String?) -> Void
    let restartServer: () -> Void

    @MainActor func state(_ item: PluginListing.Item, _ manager: PluginManager) -> PluginListing.State {
        PluginListing.state(item, switches: switches, running: manager.runningVersion, server: manager.serverStatus, serverUp: serverUp)
    }
}

// MARK: - List

private struct PluginList: View {
    let lists: PluginListing.Lists
    let selection: PluginListing.Selection
    let context: PluginContext
    @ObservedObject var manager: PluginManager
    let select: (PluginListing.Selection) -> Void

    var body: some View {
        VStack(spacing: 0) {
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
                }
                .padding(10)
            }
            Divider()
            PluginListRow(selected: selection == .agent, action: { select(.agent) }) {
                Image(systemName: "sparkles").foregroundStyle(.secondary).frame(width: 20)
                Text("Use a coding agent instead").font(.app(.value)).foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .padding(10)
        }
    }

    private func heading(_ title: String) -> some View {
        Text(L10n.text(title).uppercased()).font(.app(.annotation, weight: .semibold)).foregroundStyle(.tertiary)
            .padding(.horizontal, 10).padding(.top, 10).padding(.bottom, 4)
    }

    private func row(_ item: PluginListing.Item) -> some View {
        let state = context.state(item, manager)
        let busy = manager.isBusy && manager.pending.contains(item.name)
        return PluginListRow(selected: selection == .plugin(item.name), action: { select(.plugin(item.name)) }) {
            PluginIcon(item: item, size: 32)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title).font(.app(.rowTitle, weight: .medium)).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                if busy {
                    Text("Installing…").font(.app(.annotation)).foregroundStyle(.secondary)
                } else {
                    PluginStateBadge(state: state, kind: item.kind, compact: true)
                }
            }
            Spacer(minLength: 0)
            if busy { ProgressView().controlSize(.small) }
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
                .padding(.horizontal, 8).padding(.vertical, 7)
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
                .background(SidebarRowStyle.fill(selected: selected, hovering: hovering), in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

// MARK: - Detail

private struct PluginDetail: View {
    let selection: PluginListing.Selection
    let lists: PluginListing.Lists
    let context: PluginContext
    @ObservedObject var manager: PluginManager

    var body: some View {
        ScrollView {
            Group {
                switch selection {
                case .agent:
                    PluginAgentPage(manager: manager)
                case .plugin(let name):
                    if let item = (lists.installed + lists.available).first(where: { $0.name == name }) {
                        PluginPage(item: item, context: context, manager: manager)
                            .id(item.name)
                    }
                }
            }
            .padding(28)
            .frame(maxWidth: 780, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct PluginPage: View {
    let item: PluginListing.Item
    let context: PluginContext
    @ObservedObject var manager: PluginManager
    @EnvironmentObject var downloads: DownloadManager
    @State private var onDisk: Set<String> = []
    @State private var copied = false

    private var state: PluginListing.State { context.state(item, manager) }
    private var packs: [PluginListing.Pack] { item.entry.map(PluginListing.packs) ?? [] }
    private var installing: Bool { manager.phase != .idle && manager.pending.contains(item.name) }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            hero
            if installing {
                PluginInstallPanel(manager: manager, context: context, title: item.title)
            } else if state == .restartNeeded {
                restartCard
            } else if state == .available {
                trustNote
            }
            if state == .needsRebuild, let why = manager.serverStatus[item.name]?.why {
                PluginCallout(tint: .orange, icon: "exclamationmark.triangle.fill",
                              title: L10n.text("Built for another mlx-serve"),
                              detail: L10n.format("This mlx-serve refused it: %@.", why))
            }
            if let receipt = item.receipt, !receipt.options.isEmpty {
                PluginOptions(options: receipt.options, values: context.switches.options[item.name] ?? [:]) {
                    context.setOption(item.name, $0, $1)
                }
            }
            if !packs.isEmpty { packList }
            requirements
            if let receipt = item.receipt { details(receipt) }
            agentFooter
        }
        .task(id: item.name) { refreshOnDisk() }
    }

    // MARK: Hero

    private var hero: some View {
        HStack(alignment: .top, spacing: 16) {
            PluginIcon(item: item, size: 60)
            VStack(alignment: .leading, spacing: 6) {
                Text(item.title).font(.app(.sectionTitle, weight: .semibold)).textSelection(.enabled)
                HStack(spacing: 8) {
                    if item.title != item.name {
                        Text(item.name).font(.app(.annotation, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                        Text("·").foregroundStyle(.tertiary)
                    }
                    Text(PluginLabels.kind(item.kind)).font(.app(.annotation)).foregroundStyle(.secondary)
                    if let version = item.receipt?.version ?? item.entry?.version, !version.isEmpty {
                        Text("·").foregroundStyle(.tertiary)
                        Text(version).font(.app(.annotation, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    PluginStateBadge(state: state, kind: item.kind, compact: false)
                }
                if let summary = item.entry?.summary, !summary.isEmpty {
                    Text(summary).font(.app(.explainer)).fixedSize(horizontal: false, vertical: true).padding(.top, 4)
                }
                if let entry = item.entry { links(entry) }
            }
            Spacer(minLength: 12)
            actions
        }
    }

    /// Where the plugin comes from: the project it packages, and its folder in the plugins repo.
    private func links(_ entry: PluginCatalog.Entry) -> some View {
        HStack(spacing: 16) {
            if let home = PluginCatalog.homepageURL(entry) {
                Link(destination: home) {
                    Label { Text((home.host ?? "") + home.path).font(.app(.annotation)) } icon: { Image(systemName: "house") }
                }
                .help(L10n.text("The original project"))
            }
            Link(destination: PluginCatalog.folderURL(entry.name)) {
                Label {
                    Text("CommunityServe/plugins").font(.app(.annotation))
                } icon: {
                    AsyncImage(url: PluginCatalog.logoURL) { phase in
                        if let image = phase.image {
                            image.resizable().interpolation(.high).frame(width: 14, height: 14)
                                .clipShape(RoundedRectangle(cornerRadius: 3))
                        } else {
                            Image(systemName: "shippingbox")
                        }
                    }
                }
            }
            .help(L10n.text("This plugin's source in the CommunityServe catalog"))
        }
        .padding(.top, 2)
    }

    @ViewBuilder private var actions: some View {
        let entry = item.entry
        switch state {
        case .available, .needsNewer:
            Button { if let entry { context.install([entry]) } } label: {
                Text("Install").font(.app(.value, weight: .semibold)).frame(minWidth: 84)
            }
            .buttonStyle(.borderedProminent).controlSize(.large)
            .disabled(manager.isBusy || installing || state != .available)
        case .updateAvailable(let version):
            VStack(alignment: .trailing, spacing: 8) {
                Button { if let entry { context.install([entry]) } } label: {
                    Text(L10n.format("Update to %@", version)).font(.app(.value, weight: .semibold))
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(manager.isBusy || installing)
                removeButton
            }
        case .needsRebuild:
            VStack(alignment: .trailing, spacing: 8) {
                Button { if let entry { context.install([entry]) } } label: { Text("Rebuild").font(.app(.value, weight: .semibold)) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
                    .disabled(manager.isBusy || installing || entry == nil)
                    .help(entry == nil ? L10n.text("Not in the catalog: rebuild it from its source with this release's kit") : "")
                removeButton
            }
        case .on, .off, .restartNeeded:
            VStack(alignment: .trailing, spacing: 8) {
                Toggle(isOn: Binding(get: { context.switches.enabled && context.switches.isOn(item.name) },
                                     set: { context.setSwitch(item.name, $0) })) {
                    Text("Enabled").font(.app(.value))
                }
                .toggleStyle(.switch)
                .disabled(manager.isBusy)
                removeButton
            }
        }
    }

    private var removeButton: some View {
        Button(role: .destructive) { manager.uninstall(item.name) } label: { Text("Remove").font(.app(.value)) }
            .buttonStyle(.borderless)
            .disabled(manager.isBusy)
    }

    private var trustNote: some View {
        Label {
            Text("Install downloads its source from CommunityServe/plugins and shows you what it will build. Once you approve, it is compiled and tested on this Mac and turned on.")
                .font(.app(.annotation)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "checkmark.shield").foregroundStyle(.secondary)
        }
    }

    private var restartCard: some View {
        PluginCallout(tint: .orange, icon: "arrow.clockwise.circle.fill",
                      title: L10n.text("Restart the server to apply"),
                      detail: L10n.text("The running server loaded its plugins when it started.")) {
            Button { context.restartServer() } label: { Text("Restart server").font(.app(.value, weight: .medium)) }
                .buttonStyle(.borderedProminent).tint(.orange)
        }
    }

    // MARK: Packs

    private var packList: some View {
        PluginSection(title: "Models it adds") {
            ForEach(Array(packs.enumerated()), id: \.element.id) { index, pack in
                if index > 0 { Divider() }
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(pack.repo.map { ($0 as NSString).lastPathComponent } ?? pack.text).font(.app(.rowTitle))
                        Text(pack.repo == nil ? "" : pack.text).font(.app(.annotation)).foregroundStyle(.secondary)
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

    // MARK: Requirements

    private var requirements: some View {
        PluginSection(title: "Requirements") {
            if let requirement = item.entry?.requiresMlxServe, !requirement.isEmpty {
                let met = PluginCatalog.runs(on: manager.runningVersion, requirement: requirement)
                PluginCheckRow(met: met, title: L10n.format("mlx-serve %@", requirement),
                               detail: manager.runningVersion.map { L10n.format("This Mac runs %@", $0) } ?? "")
            }
            if item.receipt == nil {
                PluginCheckRow(met: manager.developerTools, title: L10n.text("Command Line Tools"),
                               detail: manager.developerTools == false
                                   ? L10n.text("Install opens Apple's installer for them")
                                   : L10n.text("Used to compile the plugin"))
            }
        }
    }

    // MARK: Details

    private func details(_ receipt: OutsidePlugins.Receipt) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                PluginFactRow(label: "Signed", value: receipt.signature.isEmpty ? L10n.text("unknown") : receipt.signature)
                PluginFactRow(label: "SHA-256", value: receipt.sha256, mono: true)
                if !receipt.source.isEmpty { PluginFactRow(label: "Source", value: receipt.source, mono: true) }
                Text("Checked against this hash every time the server loads it.").font(.app(.annotation)).foregroundStyle(.secondary)
            }
            .padding(.top, 8)
        } label: {
            Text("Technical details").font(.app(.annotation, weight: .semibold)).foregroundStyle(.secondary)
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
                Label { Text(copied ? "Copied" : "Copy agent instructions").font(.app(.annotation)) } icon: { Image(systemName: "doc.on.doc") }
            }
            .buttonStyle(.link)
            .disabled(item.entry == nil)
        }
        .padding(.top, 4)
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
        case "engine": return L10n.text("Engine")
        default: return kind
        }
    }
}

/// The catalog's icon when it has one, else the monogram (also while the icon loads or if it fails).
private struct PluginIcon: View {
    let item: PluginListing.Item
    let size: CGFloat

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: size * 0.24)
        if let url = item.entry.flatMap(PluginCatalog.iconURL) {
            AsyncImage(url: url) { phase in
                if let image = phase.image {
                    image.resizable().interpolation(.high).aspectRatio(contentMode: .fill)
                        .frame(width: size, height: size).clipShape(shape)
                        .overlay(shape.strokeBorder(.separator.opacity(0.5)))
                } else {
                    PluginMonogram(name: item.name, kind: item.kind, size: size)
                }
            }
        } else {
            PluginMonogram(name: item.name, kind: item.kind, size: size)
        }
    }
}

/// A plugin with no icon: its initial on its kind's colour.
private struct PluginMonogram: View {
    let name: String
    let kind: String
    let size: CGFloat

    private var tint: Color {
        switch kind {
        case "model": return .purple
        case "media": return .teal
        case "quant": return .orange
        case "engine": return .indigo
        default: return .gray
        }
    }

    var body: some View {
        Text(String(name.prefix(1)).uppercased())
            .font(.app(size > 40 ? .sectionTitle : .rowTitle, weight: .bold))
            .foregroundStyle(.white)
            .frame(width: size, height: size)
            .background(tint.gradient, in: RoundedRectangle(cornerRadius: size * 0.24))
    }
}

/// The state as a dot and a word; in the list it is the row's second line, beside the kind.
private struct PluginStateBadge: View {
    let state: PluginListing.State
    let kind: String
    let compact: Bool

    private var look: (text: String, color: Color) {
        switch state {
        case .on: return (L10n.text("On"), .green)
        case .off: return (L10n.text("Off"), .secondary)
        case .updateAvailable(let v): return (compact ? L10n.text("Update") : L10n.format("Update to %@", v), .orange)
        case .available: return (L10n.text("Not installed"), .secondary)
        case .needsNewer(let r): return (compact ? L10n.text("Needs update") : L10n.format("Needs mlx-serve %@", r), .red)
        case .needsRebuild: return (L10n.text("Needs rebuild"), .orange)
        case .restartNeeded: return (L10n.text("Restart to apply"), .orange)
        }
    }

    var body: some View {
        let look = self.look
        if compact {
            HStack(spacing: 5) {
                Text(PluginLabels.kind(kind)).foregroundStyle(.secondary)
                Text("·").foregroundStyle(.tertiary)
                if look.color != .secondary { Circle().fill(look.color).frame(width: 6, height: 6) }
                Text(look.text).foregroundStyle(look.color == .secondary ? Color.secondary : look.color)
            }
            .font(.app(.annotation)).lineLimit(1)
        } else {
            HStack(spacing: 5) {
                Circle().fill(look.color).frame(width: 7, height: 7)
                Text(look.text).font(.app(.annotation, weight: .medium)).foregroundStyle(look.color == .secondary ? Color.secondary : look.color)
            }
            .padding(.horizontal, 8).padding(.vertical, 3)
            .background(look.color.opacity(0.12), in: Capsule())
        }
    }
}

private struct PluginSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.text(title)).font(.app(.annotation, weight: .semibold)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 8) { content }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.separator.opacity(0.6)))
        }
    }
}

/// A plugin's own settings: the server sets them when it starts, so a change on a running server asks for a restart.
private struct PluginOptions: View {
    let options: [OutsidePlugins.Option]
    let values: [String: String]
    let set: (_ name: String, _ value: String?) -> Void

    var body: some View {
        PluginSection(title: "Settings") {
            ForEach(Array(options.enumerated()), id: \.element.id) { index, option in
                if index > 0 { Divider() }
                PluginOptionRow(option: option, saved: values[option.name]) { set(option.name, $0) }
            }
            Text("The server reads these when it starts.").font(.app(.annotation)).foregroundStyle(.secondary)
        }
    }
}

/// One option: a switch, or a field whose empty value is the plugin's own default. A value its kind refuses is not
/// saved.
private struct PluginOptionRow: View {
    let option: OutsidePlugins.Option
    let saved: String?
    let set: (String?) -> Void
    @State private var text = ""

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(option.name).font(.app(.value, design: .monospaced)).textSelection(.enabled)
                if !option.help.isEmpty {
                    Text(option.help).font(.app(.annotation)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 12)
            control
        }
        .padding(.vertical, 4)
        .onAppear { text = saved ?? "" }
    }

    @ViewBuilder private var control: some View {
        if option.kind == .bool {
            Toggle(isOn: Binding(get: { (saved ?? option.defaultValue) == "1" }, set: { set($0 ? "1" : "0") })) { EmptyView() }
                .labelsHidden().toggleStyle(.switch)
        } else {
            VStack(alignment: .trailing, spacing: 2) {
                TextField(option.defaultValue, text: $text)
                    .textFieldStyle(.roundedBorder)
                    .font(.app(.value, design: option.kind == .number ? .monospaced : .default))
                    .frame(width: option.kind == .number ? 120 : 240)
                    .onChange(of: text) { _, value in
                        if value.isEmpty { set(nil) } else if option.accepts(value) { set(value) }
                    }
                if !text.isEmpty && !option.accepts(text) {
                    Group { if option.kind == .number { Text("Not a number") } else { Text("Too long") } }
                        .font(.app(.annotation)).foregroundStyle(.red)
                }
            }
        }
    }
}

/// A tinted card: what happened or what to do next, with its action on the right.
private struct PluginCallout<Actions: View>: View {
    let tint: Color
    let icon: String
    let title: String
    let detail: String
    @ViewBuilder let actions: Actions

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: icon).font(.app(.title2)).foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.app(.rowTitle, weight: .semibold))
                if !detail.isEmpty {
                    Text(detail).font(.app(.annotation)).foregroundStyle(.secondary).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            actions
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.10), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(tint.opacity(0.25)))
    }
}

extension PluginCallout where Actions == EmptyView {
    init(tint: Color, icon: String, title: String, detail: String) {
        self.init(tint: tint, icon: icon, title: title, detail: detail) { EmptyView() }
    }
}

/// A requirement: met (green), not met (orange), or not checked yet (nil).
private struct PluginCheckRow: View {
    let met: Bool?
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: met == nil ? "circle.dashed" : (met! ? "checkmark.circle.fill" : "exclamationmark.circle.fill"))
                .foregroundStyle(met == nil ? Color.secondary : (met! ? Color.green : Color.orange))
            Text(title).font(.app(.rowTitle))
            Spacer(minLength: 8)
            Text(detail).font(.app(.annotation)).foregroundStyle(.secondary)
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

/// The install in progress as three steps, then the outcome. The raw log stays behind Details.
private struct PluginInstallPanel: View {
    @ObservedObject var manager: PluginManager
    let context: PluginContext
    let title: String

    /// The page's title for its own install; Rebuild all keeps the joined names.
    private func label(_ name: String) -> String { manager.pending.count == 1 ? title : name }

    var body: some View {
        switch manager.phase {
        case .idle:
            EmptyView()
        case .fetching(let name), .building(let name), .installing(let name):
            progress(name)
        case .awaitingConfirmation(let name, let commit, let version):
            approval(name: name, commit: commit, version: version)
        case .done(let name):
            PluginCallout(tint: .green, icon: "checkmark.circle.fill",
                          title: L10n.format("%@ is installed and on", label(name)),
                          detail: context.serverUp ? L10n.text("Restart the server to start using it.")
                                                   : L10n.text("It loads when the server starts.")) {
                HStack {
                    if context.serverUp {
                        Button { context.restartServer(); manager.dismiss() } label: {
                            Text("Restart server").font(.app(.value, weight: .medium))
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    Button { manager.dismiss() } label: { Text("Done").font(.app(.value)) }
                }
            }
        case .failed(let name, let message):
            VStack(alignment: .leading, spacing: 10) {
                PluginCallout(tint: .red, icon: "xmark.octagon.fill",
                              title: L10n.format("%@ did not install", label(name)), detail: message) {
                    HStack {
                        Button { manager.retry() } label: { Text("Try again").font(.app(.value, weight: .medium)) }
                            .buttonStyle(.borderedProminent)
                        Button { manager.dismiss() } label: { Text("Dismiss").font(.app(.value)) }
                    }
                }
                logView
            }
        }
    }

    /// The one approval: what will be compiled, named by its commit, before anything runs.
    private func approval(name: String, commit: String, version: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(L10n.format("Build and install %@ %@?", label(name), version)).font(.app(.rowTitle, weight: .semibold))
            ForEach(PluginListing.installSteps(manager.phase), id: \.title) { PluginStepRow(step: $0) }
            HStack(spacing: 6) {
                Text(L10n.format("Source: CommunityServe/plugins at %@", String(commit.prefix(10))))
                    .font(.app(.annotation, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                if let url = PluginCatalog.commitURL(commit) {
                    Link(destination: url) { Text("View on GitHub").font(.app(.annotation)) }
                }
            }
            Text("It is compiled and tested on this Mac, then turned on. A plugin is native code that runs with your rights: approve it only if you trust it.")
                .font(.app(.annotation)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            HStack {
                Button { manager.confirm() } label: { Text("Build and install").font(.app(.value, weight: .semibold)) }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                Button { manager.cancel() } label: { Text("Cancel").font(.app(.value)) }
            }
            let manifest = manager.manifestText(for: name)
            if !manifest.isEmpty {
                DisclosureGroup {
                    ScrollView {
                        Text(manifest).font(.app(.caption2, design: .monospaced)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 160)
                } label: {
                    Text("plugin.json").font(.app(.annotation)).foregroundStyle(.secondary)
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.accentColor.opacity(0.2)))
    }

    private func progress(_ name: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(L10n.format("Installing %@", label(name))).font(.app(.rowTitle, weight: .semibold))
                Spacer()
                Button { manager.cancel() } label: { Text("Cancel").font(.app(.value)) }
            }
            ForEach(PluginListing.installSteps(manager.phase), id: \.title) { PluginStepRow(step: $0) }
            Text("The first install also downloads the Zig compiler, so it can take a few minutes.")
                .font(.app(.annotation)).foregroundStyle(.secondary)
            logView
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.07), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.accentColor.opacity(0.2)))
    }

    @ViewBuilder private var logView: some View {
        if !manager.log.isEmpty {
            DisclosureGroup {
                ScrollView {
                    Text(manager.log)
                        .font(.app(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 180)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(manager.log, forType: .string)
                } label: { Text("Copy log").font(.app(.annotation)) }
                .buttonStyle(.link)
            } label: {
                Text("Details").font(.app(.annotation)).foregroundStyle(.secondary)
            }
        }
    }
}

private struct PluginStepRow: View {
    let step: PluginListing.Step

    var body: some View {
        HStack(spacing: 10) {
            icon.frame(width: 18)
            Text(step.title).font(.app(.value)).foregroundStyle(step.status == .waiting ? Color.secondary : Color.primary)
        }
    }

    @ViewBuilder private var icon: some View {
        switch step.status {
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .active: ProgressView().controlSize(.small)
        case .waiting: Image(systemName: "circle").foregroundStyle(.tertiary)
        }
    }
}
