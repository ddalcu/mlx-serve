import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// Single-window form for the user-facing mlx-serve tunables. Bindings flow
/// through `appState.serverOptions`; AppState's `didSet` auto-saves to
/// UserDefaults.
struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    /// Live filter text. Pushed down the tree via `\.settingsSearchQuery`;
    /// every row decides for itself whether to stay on screen (see
    /// `SearchableRow`), and every section collapses when nothing inside it
    /// survived. The matching rule is pure and tested — `SettingsSearch`.
    @State private var searchQuery = ""

    /// Sidebar selection. `.all` (the default) renders the whole form — exactly
    /// what this screen was before the sidebar existed; a category renders just
    /// that section. Never coexists with search text: see `SettingsSelection`.
    @State private var selection: SettingsSelection = .all

    /// Rows still visible under the current filter, summed up the tree from
    /// `SettingsVisibleRowCountKey`. Drives the "no matches" placeholder.
    @State private var visibleRows = 0

    private var filtering: Bool { !SettingsSearch.tokens(searchQuery).isEmpty }

    /// Categories the form actually renders for the active engine — the sidebar
    /// must never offer a section that isn't there.
    private var categories: [SettingsCategory] {
        SettingsCategory.visible(engine: server.modelInfo?.engine,
                                 selfUpdate: BuildFeatures.current.selfUpdate)
    }

    var body: some View {
        shell
        .navigationTitle("Settings")
        // An engine switch can retire the selected category (load a GGUF model
        // while "MLX Performance" is selected) — fall back to All rather than
        // leave a blank pane.
        .onChange(of: categories) { _, visible in
            selection = SettingsSelection.reconciled(selection, visible: visible)
        }
    }

    /// This only ever renders inside the chat window's detail column
    /// (`ChatWorkspace.settings`; the Settings Window scene is gone). A
    /// `NavigationSplitView` nested inside another fights the outer sidebar
    /// for column behaviour, so the shell is a plain two-pane HStack — same
    /// category list, same form. Tasks answers the same problem the other
    /// way, by becoming real columns of the window's own split
    /// (`TaskListPane` / `TaskDetailPane`).
    private var shell: some View {
        HStack(spacing: 0) {
            categoryList
                .frame(width: 210)
            Divider()
            form
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var categoryList: some View {
        SettingsSidebar(categories: categories, selection: $selection) {
            // Picking a category clears the search: the two are alternative
            // ways to narrow, and letting them stack strands the user on an
            // empty pane with a stale query they can't see.
            searchQuery = ""
        }
    }

    private var form: some View {
        VStack(spacing: 0) {
            if server.needsRestartFor(appState.serverOptions) {
                RestartBanner()
            }
            SettingsSearchField(text: $searchQuery)
                // Typing searches across EVERYTHING, so it snaps the sidebar
                // back to All — a search that silently only looked inside the
                // selected category would hide its own best answers.
                .onChange(of: searchQuery) { _, q in
                    selection = SettingsSelection.afterQueryEdit(query: q, current: selection)
                }
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    SettingsSection(
                        category: .modelFolders,
                        subtitle: "Choose where downloads are saved, and add a folder to scan if some of your models live elsewhere. Every folder listed here is served — restart the server after changing them."
                    ) {
                        ModelFoldersSectionContent()
                    }
                    SettingsSection(
                        category: .server,
                        subtitle: "Server-launch flags. Restart the server to apply changes."
                    ) {
                        ServerSectionContent()
                    }
                    SettingsSection(
                        category: .lanSharing,
                        subtitle: "Share models with other Macs on your local network and use theirs — zero-setup discovery over Bonjour, everything off by default. Restart the server to apply."
                    ) {
                        LanSharingSectionContent()
                    }
                    // Engine-aware sections. Each panel is hidden when its
                    // controls don't apply to the active engine — flipping
                    // `--kv-quant` on a GGUF model silently no-ops, so we'd
                    // rather not show that picker at all than mislead.
                    EngineAwareSections()
                    SettingsSection(
                        category: .requestDefaults,
                        subtitle: "Apply on the next chat request — no restart needed."
                    ) {
                        RequestDefaultsSectionContent()
                    }

                    SettingsSection(
                        category: .voice,
                        subtitle: "Clone your voice once — hands-free voice mode answers in it via the local TTS model. No clip set: answers use the macOS system voice. Applies to the next spoken sentence — no restart needed."
                    ) {
                        WakePhraseSectionContent()
                        VoiceCloneSectionContent()
                    }

                    SettingsSection(
                        category: .sandbox,
                        subtitle: BuildFeatures.current.hostShell
                            ? "Run the agent's shell commands inside an isolated Linux sandbox instead of directly on this Mac. Off by default; applies to the next command — no restart needed."
                            : "Agent shell commands always run inside an isolated Linux sandbox in this build — they never touch macOS directly. The guest OS ships inside the app."
                    ) {
                        SandboxSectionContent()
                    }

                    SettingsSection(
                        category: .messaging,
                        subtitle: "Message your local model from your phone via a Telegram bot. No public URL or port-forwarding needed — the app long-polls Telegram over your normal internet connection, so it works behind home Wi-Fi."
                    ) {
                        MessagingSectionContent(bridge: appState.telegramBridge)
                    }

                    // The Mac App Store updates the app itself; a pane offering a
                    // DMG self-update would be dead UI there (and an App Review flag).
                    // `SettingsCategory.visible(selfUpdate:)` mirrors this so the
                    // sidebar never lists a section that isn't built.
                    if BuildFeatures.current.selfUpdate {
                        SettingsSection(
                            category: .updates,
                            subtitle: "New versions ship on the project's GitHub releases page. Installing downloads the notarized app, swaps it in place, and relaunches — chats, models, and settings are untouched."
                        ) {
                            UpdatesSectionContent(updates: appState.updates)
                        }
                    }

                    // Not folded into Updates: that section is gated on
                    // `selfUpdate`, so on a Mac App Store build these links
                    // would never render.
                    SettingsSection(
                        category: .about,
                        subtitle: "mlx-serve is free and open source, built by one person. Star it, follow along, or just say hello — questions and bug reports are welcome."
                    ) {
                        ForEach(CommunityLinks.all) { item in
                            SettingsRow(title: item.title, explainer: item.explainer) {
                                Link(item.actionLabel, destination: item.url)
                            }
                        }
                    }

                    if filtering && visibleRows == 0 {
                        NoSearchResults(query: searchQuery) { searchQuery = "" }
                    }

                    ResetDefaultsFooter()
                }
                .environment(\.settingsSearchQuery, searchQuery)
                .environment(\.settingsSelection, selection)
                .onPreferenceChange(SettingsVisibleRowCountKey.self) { visibleRows = $0 }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

// MARK: - Sidebar

/// Category filter for the form: "All Settings" plus one row per section that
/// the active engine actually renders. Icons are SF Symbols — free with the OS,
/// no assets.
private struct SettingsSidebar: View {
    let categories: [SettingsCategory]
    @Binding var selection: SettingsSelection
    /// Run when the user picks a row — clears the search field (a category and a
    /// query never coexist).
    let onPick: () -> Void

    var body: some View {
        List(selection: Binding(
            get: { selection },
            set: { newValue in
                selection = newValue
                onPick()
            }
        )) {
            Label("All Settings", systemImage: "square.grid.2x2")
                .tag(SettingsSelection.all)
            Section {
                ForEach(categories) { category in
                    Label(category.sidebarLabel, systemImage: category.icon)
                        .tag(SettingsSelection.category(category))
                }
            }
        }
        .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 260)
    }
}

// MARK: - Search plumbing

/// The active filter text, pushed down to every row. A section whose *header*
/// matches re-publishes a blank query to its children so the whole section
/// shows (searching "telegram" should reveal the bot token, not just the rows
/// whose own text happens to contain the word).
private struct SettingsSearchQueryKey: EnvironmentKey {
    static let defaultValue: String = ""
}

extension EnvironmentValues {
    fileprivate var settingsSearchQuery: String {
        get { self[SettingsSearchQueryKey.self] }
        set { self[SettingsSearchQueryKey.self] = newValue }
    }
}

/// The sidebar's category filter, pushed down so each `SettingsSection` — and
/// `EngineAwareSections`, which builds its own — can decide whether it renders.
private struct SettingsSelectionKey: EnvironmentKey {
    static let defaultValue: SettingsSelection = .all
}

extension EnvironmentValues {
    fileprivate var settingsSelection: SettingsSelection {
        get { self[SettingsSelectionKey.self] }
        set { self[SettingsSelectionKey.self] = newValue }
    }
}

/// Number of rows that survived the filter, summed up the view tree. Sections
/// read it to decide whether to collapse; the root reads it to decide whether
/// to show the "no matches" placeholder.
private struct SettingsVisibleRowCountKey: PreferenceKey {
    static let defaultValue: Int = 0
    static func reduce(value: inout Int, nextValue: () -> Int) { value += nextValue() }
}

/// Wraps one row so the filter can hide it, and reports it as visible when it
/// survives. `searchText` is conventionally `[label, description]` — the same
/// text the row renders, so what you read is what you can search for.
private struct SearchableRow<Content: View>: View {
    let searchText: [String]
    @ViewBuilder var content: Content

    @Environment(\.settingsSearchQuery) private var query

    var body: some View {
        if SettingsSearch.matches(query: query, in: searchText) {
            content.preference(key: SettingsVisibleRowCountKey.self, value: 1)
        }
    }
}

/// Filter field pinned above the scrolling form.
private struct SettingsSearchField: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Filter settings", text: $text)
                .textFieldStyle(.plain)
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .foregroundStyle(.secondary)
                .help("Clear the filter")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 24)
        .padding(.top, 14)
    }
}

/// Shown when the filter matches nothing at all.
private struct NoSearchResults: View {
    let query: String
    let clear: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("No settings match “\(query)”")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button("Clear filter", action: clear)
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 48)
    }
}

// MARK: - Reset to Defaults footer

/// Restores the fields of whatever the sidebar has selected — that one section,
/// or (under "All Settings") the whole screen, keeping the Telegram bot token.
private struct ResetDefaultsFooter: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.settingsSelection) private var selection
    @State private var showConfirm = false

    private var label: String { SettingsReset.buttonLabel(selection) }

    private var helpText: String { SettingsReset.confirmMessage(selection) }

    /// Nothing to reset in this section → no button.
    private var hidden: Bool {
        if case .category(let c) = selection { return !SettingsReset.isResettable(c) }
        return false
    }

    var body: some View {
        if !hidden {
            SearchableRow(searchText: [label, helpText]) {
                HStack {
                    Spacer()
                    Button(role: .destructive) {
                        showConfirm = true
                    } label: {
                        Label(label, systemImage: "arrow.uturn.backward.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help(helpText)
                }
                .padding(.top, 4)
                .confirmationDialog(
                    SettingsReset.confirmTitle(selection),
                    isPresented: $showConfirm,
                    titleVisibility: .visible
                ) {
                    Button("Reset", role: .destructive) {
                        appState.serverOptions = SettingsReset.apply(selection, to: appState.serverOptions)
                    }
                    .keyboardShortcut(.defaultAction)
                    Button("Cancel", role: .cancel) { }
                } message: {
                    Text(helpText)
                }
            }
        }
    }
}

// MARK: - Restart banner

private struct RestartBanner: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "arrow.clockwise.circle.fill")
                .font(.title2)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Some changes require a server restart")
                    .font(.subheadline.weight(.semibold))
                Text("Click Restart Now to apply, or Discard to revert the unsaved server-launch fields.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Restart Now") {
                let opts = appState.serverOptions
                let model = appState.selectedModelPath
                server.stop()
                if !model.isEmpty {
                    server.start(modelPath: model, options: opts)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(appState.selectedModelPath.isEmpty)

            Button("Discard") {
                if let last = server.lastLaunchedOptions {
                    // Revert every server-launch field to the last-launched
                    // snapshot; per-request defaults are preserved. Start
                    // from `last` (which has all server fields right) and
                    // patch the per-request fields back from `current` so
                    // the user's mid-session sampler tweaks survive.
                    let current = appState.serverOptions
                    var reverted = last
                    reverted.defaultMaxTokens       = current.defaultMaxTokens
                    reverted.defaultTemperature     = current.defaultTemperature
                    reverted.defaultTopP            = current.defaultTopP
                    reverted.defaultTopK            = current.defaultTopK
                    reverted.defaultRepeatPenalty   = current.defaultRepeatPenalty
                    reverted.defaultPresencePenalty = current.defaultPresencePenalty
                    reverted.defaultReasoningBudget = current.defaultReasoningBudget
                    reverted.defaultEnableThinking  = current.defaultEnableThinking
                    reverted.perRequestEnablePLD    = current.perRequestEnablePLD
                    reverted.perRequestEnableDrafter = current.perRequestEnableDrafter
                    appState.serverOptions = reverted
                }
            }
            .buttonStyle(.bordered)
            .disabled(server.lastLaunchedOptions == nil)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color.orange.opacity(0.10))
        .overlay(Divider(), alignment: .bottom)
    }
}

// MARK: - Engine-aware section composer

/// Renders the engine-specific section set for the active model:
///   - MLX target:   Common Performance + MLX Performance + MLX Spec Decode
///   - GGUF target:  Common Performance + GGUF Performance
///   - DSV4 target:  Common Performance + DeepSeek-V4 (ds4) section
///   - No model yet: All sections shown (so users can pre-tune before
///                   loading); a banner clarifies that some controls only
///                   apply once a matching engine is loaded.
private struct EngineAwareSections: View {
    @EnvironmentObject var server: ServerManager
    @Environment(\.settingsSearchQuery) private var query
    @Environment(\.settingsSelection) private var selection

    /// Resolved engine for routing UI decisions. Nil when no model has
    /// loaded yet (server stopped or first start in progress) — that
    /// case shows all sections so users can pre-tune.
    private var engine: ServerEngine? { server.modelInfo?.engine }

    var body: some View {
        // Engine-specific sections. Show all when no model is loaded so
        // the user can pre-tune; otherwise show only the matching set.
        let showMLX = (engine == nil || engine == .mlx)
        let showLlama = (engine == nil || engine == .llama)
        let showDs4 = (engine == nil || engine == .dsv4)

        if showMLX {
            SettingsSection(
                category: .specDecode,
                subtitle: "Big throughput wins on echo-heavy work; gates auto-disable on novel content. PLD, the drafter, and MTP are MLX-only — they no-op on GGUF / DSV4."
            ) {
                SpecDecodeSectionContent()
            }
        }

        // ONE Performance section. The universal rows always apply; the MLX-only
        // ones (continuous batching, KV-quant, hot prefix cache) join them when
        // an MLX model is serving — on GGUF/DSV4 they'd silently no-op, so they
        // stay hidden rather than lie.
        SettingsSection(
            category: .performance,
            subtitle: showMLX
                ? "Continuous batching, KV-cache quantization, and the cross-request hot prefix cache. Server-launch flags — restart to apply."
                : "Tunables that apply regardless of engine. Server-launch flags — restart to apply."
        ) {
            CommonPerformanceSectionContent()
            if showMLX {
                PerformanceSectionContent()
            }
        }

        if showLlama {
            SettingsSection(
                category: .ggufPerformance,
                subtitle: "Knobs that apply when an embedded llama.cpp engine is serving a `.gguf` model. Distinct from the MLX Performance section — different kernels, different KV layout."
            ) {
                LlamaPerformanceSectionContent()
            }
        }

        if showDs4 {
            SettingsSection(
                category: .ds4,
                subtitle: "Knobs for the embedded ds4 engine serving DeepSeek-V4-Flash. Ignored by the MLX and llama.cpp engines."
            ) {
                Ds4PerformanceSectionContent()
            }
        }

        // The pre-tune banner explains why EVERY engine section is on screen —
        // meaningless once the sidebar has narrowed to one of them.
        if engine == nil, selection == .all, SettingsSearch.tokens(query).isEmpty {
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text("No model loaded yet — every section is shown so you can pre-tune. Once a model is active, sections that don't apply will hide automatically.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 24)
        }
    }
}

// MARK: - Section frame

private struct SettingsSection<Content: View>: View {
    /// Identity. The sidebar row and this header both read their text from it,
    /// so a section can never exist without a way to reach it (and its two
    /// labels can't drift). Pinned by `testEverySettingsSectionDeclaresACategory`.
    let category: SettingsCategory
    let subtitle: String
    @ViewBuilder var content: Content

    @Environment(\.settingsSearchQuery) private var query
    @Environment(\.settingsSelection) private var selection
    @State private var visibleRows = 0

    private var title: String { category.title }

    /// Both filter decisions (what query the rows see, whether to hide the
    /// chrome) live in the pure, tested `SettingsSearch.SectionFilter`.
    private var filter: SettingsSearch.SectionFilter {
        SettingsSearch.section(query: query, title: title)
    }

    @ViewBuilder
    var body: some View {
        // Sidebar filter. Unlike the search filter below — which must keep a
        // collapsed section's rows in the tree so they can publish their count
        // and bring the section back — this one drops the subtree outright. It
        // can: a category is only ever selected while the search field is EMPTY
        // (SettingsSelection's invariant), so no row count depends on it.
        if selection.shows(category) {
            sectionBody
        }
    }

    @ViewBuilder
    private var sectionBody: some View {
        // `content` keeps rendering even when collapsed (as a stack of empty
        // rows), because it is the only thing that publishes
        // `SettingsVisibleRowCountKey`. Dropping it from the tree would pin
        // `visibleRows` at 0 and the section could never come back when the
        // query changes. So we hide the *chrome* — header, padding, card —
        // never the subtree.
        let collapsed = filter.collapsed(visibleRows: visibleRows)
        VStack(alignment: .leading, spacing: collapsed ? 0 : 12) {
            if !collapsed {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.title3.weight(.semibold))
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: collapsed ? 0 : 18) {
                content
            }
            .environment(\.settingsSearchQuery, filter.childQuery)
            .padding(collapsed ? 0 : 16)
            .background(collapsed ? Color.clear : Color(NSColor.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .onPreferenceChange(SettingsVisibleRowCountKey.self) { visibleRows = $0 }
        .padding(.bottom, collapsed ? 0 : 24)
    }
}

// MARK: - One row helper

private struct SettingsRow<Control: View>: View {
    let title: String
    let explainer: String
    /// True when this field has been changed since the running server was
    /// last launched — i.e. the user has edited it but not yet hit "Restart
    /// Now". Drives the orange restart icon. False (or always-false for
    /// per-request fields) hides the icon. We deliberately don't show it on
    /// every server-launch row by default — that's noisy when nothing has
    /// actually been changed yet.
    var isDirty: Bool = false
    @ViewBuilder var control: Control

    var body: some View {
        SearchableRow(searchText: [title, explainer]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    HStack(spacing: 6) {
                        Text(title)
                            .font(.body)
                        if isDirty {
                            Image(systemName: "arrow.clockwise.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .help("Restart the server to apply this change")
                        }
                    }
                    Spacer(minLength: 12)
                    control
                        .frame(maxWidth: 280, alignment: .trailing)
                }
                Text(explainer)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// Shared dirty-bit helper. Compares a single `ServerOptions` keypath against
/// the snapshot the server was last launched with. Returns false until the
/// server has been launched at least once (no baseline to compare against).
fileprivate struct ServerLaunchDirty {
    let current: ServerOptions
    let last: ServerOptions?

    func dirty<V: Equatable>(_ keyPath: KeyPath<ServerOptions, V>) -> Bool {
        guard let last else { return false }
        return current[keyPath: keyPath] != last[keyPath: keyPath]
    }
}

// MARK: - Model folders section

/// One row showing the user-configured extra discovery root. The path is
/// rendered verbatim (raw, not standardized) so the user sees exactly what
/// they picked; discovery silently skips it when it doesn't resolve to an
/// existing directory. Picking a folder triggers an immediate refresh so the
/// menu-bar picker updates without a server restart.
private struct ModelFoldersSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager

    private static let explainer = "Accepts both flat layout (<name>/config.json) and 2-level layout (<author>/<name>/config.json)."
    private static let defaultExplainer = "Where new downloads are saved. Everything already downloaded keeps working — the old folder stays in the scan list. Restart the server to apply."

    /// Re-read after a pick so the row repaints without an @Published mirror of
    /// a value that lives in UserDefaults.
    @State private var downloadFolderTick = 0

    var body: some View {
        if BuildFeatures.current.customModelFolders { defaultFolderRow }
        customFolderRow
    }

    // MARK: - Default (download destination)

    @ViewBuilder
    private var defaultFolderRow: some View {
        let roots = ModelRoots()
        let configured = roots.configuredDownloadRoot
        let unavailable = roots.downloadRootIsUnavailable
        SearchableRow(searchText: ["Default folder", "download", Self.defaultExplainer]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Default folder")
                        .font(.body)
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(configured ?? roots.downloadRoot)
                            .font(.caption.monospaced())
                            // An unreachable folder is shown in the warning
                            // colour rather than swapped for the fallback: the
                            // path the user chose is the thing they need to see
                            // to understand where their downloads went instead.
                            .foregroundStyle(unavailable ? AnyShapeStyle(.orange) : AnyShapeStyle(configured == nil ? .secondary : .primary))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: 220, alignment: .trailing)
                        Button("Choose…") { chooseDownloadFolder() }
                            .buttonStyle(.bordered)
                        Button("Reset") {
                            SecurityScopedBookmark.clear(name: DownloadManager.downloadFolderBookmarkName)
                            ModelRoots().configuredDownloadRoot = nil
                            applyFolderChange()
                        }
                        .buttonStyle(.bordered)
                        .disabled(configured == nil)
                    }
                }
                Text(unavailable
                     ? "That folder isn't reachable right now, so downloads are going to \(ModelRoots.builtInRoot) instead."
                     : Self.defaultExplainer)
                    .font(.caption2)
                    .foregroundStyle(unavailable ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .id(downloadFolderTick)
        }
    }

    private func chooseDownloadFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "New model downloads will be saved here."
        if let existing = ModelRoots().configuredDownloadRoot {
            panel.directoryURL = URL(fileURLWithPath: (existing as NSString).expandingTildeInPath)
        }
        guard AppActivation.runModal(panel) == .OK, let url = panel.url else { return }
        // Store the bookmark while the panel's grant is still live — the same
        // rule the agent-workspace picker follows.
        SecurityScopedBookmark.store(url, name: DownloadManager.downloadFolderBookmarkName)
        SecurityScopedBookmark.startAccessOnce(name: DownloadManager.downloadFolderBookmarkName)
        ModelRoots().configuredDownloadRoot = url.path
        applyFolderChange()
    }

    /// One place for "the library's folders changed": the downloader re-reads
    /// its destination, the app rescans, and the server needs a restart to pick
    /// up the new `--model-dir` set (the banner at the top of Settings says so).
    private func applyFolderChange() {
        downloads.refreshRoots()
        appState.refreshModels()
        downloadFolderTick += 1
    }

    // MARK: - Custom (extra scan folder)

    private var customFolderRow: some View {
        let pathText: String = {
            let raw = downloads.customRoot?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return raw.isEmpty ? "(none)" : raw
        }()
        let hasPath = !(downloads.customRoot?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)

        return SearchableRow(searchText: ["Custom folder", Self.explainer]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Custom folder")
                        .font(.body)
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(pathText)
                            .font(.caption.monospaced())
                            .foregroundStyle(hasPath ? .primary : .secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: 220, alignment: .trailing)
                        Button("Choose…") { choose() }
                            .buttonStyle(.bordered)
                        Button("Clear") {
                            downloads.customRoot = nil
                            appState.refreshModels()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!hasPath)
                    }
                }
                Text(Self.explainer)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if let existing = downloads.customRoot,
           !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            panel.directoryURL = URL(fileURLWithPath: (existing as NSString).expandingTildeInPath)
        }
        if AppActivation.runModal(panel) == .OK, let url = panel.url {
            downloads.customRoot = url.path
            appState.refreshModels()
        }
    }
}

// MARK: - Server section

// MARK: - LAN sharing section

private struct LanSharingSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        if let m = meta["lanShareEnabled"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanShareEnabled)) {
                Toggle("", isOn: $appState.serverOptions.lanShareEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if appState.serverOptions.lanShareEnabled {
            if let m = meta["lanShareAll"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanShareAll)) {
                    Toggle("", isOn: $appState.serverOptions.lanShareAll)
                        .labelsHidden()
                        .toggleStyle(.switch)
                }
            }
            if !appState.serverOptions.lanShareAll {
                sharedModelList
            }
            if let m = meta["lanName"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanName)) {
                    TextField(
                        "",
                        text: $appState.serverOptions.lanName,
                        prompt: Text(Host.current().localizedName ?? "this Mac")
                    )
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 200)
                }
            }
        }
        if let m = meta["lanDiscoverEnabled"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanDiscoverEnabled)) {
                Toggle("", isOn: $appState.serverOptions.lanDiscoverEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        // The privacy disclosure — sharing means running other people's
        // prompts, and using a network model means the host reads yours.
        Text("Privacy: prompts sent to a model you share are processed on — and visible to — this Mac. Prompts you send to a network model are visible to the Mac hosting it. Traffic stays on your local network, and everything here is off unless you turn it on.")
            .font(.caption)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 4)
    }

    /// One checkbox per local model name. Names are deduped — a GGUF and an
    /// MLX build of the same repo share a name and are shared together (the
    /// server matches share entries against registry ids basename-tolerantly).
    private var sharedModelList: some View {
        let names = Array(Set(appState.localModels.map(\.name))).sorted()
        return VStack(alignment: .leading, spacing: 4) {
            if names.isEmpty {
                Text("No local models yet — download one first.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            ForEach(names, id: \.self) { name in
                Toggle(name, isOn: Binding(
                    get: { appState.serverOptions.lanSharedModels.contains(name) },
                    set: { on in
                        var list = appState.serverOptions.lanSharedModels
                        list.removeAll { $0 == name }
                        if on { list.append(name) }
                        appState.serverOptions.lanSharedModels = list.sorted()
                    }
                ))
                .toggleStyle(.checkbox)
                .font(.caption)
            }
        }
        .padding(.leading, 8)
        .padding(.vertical, 2)
    }
}

private struct ServerSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        if let m = meta["host"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.host)
            ) {
                TextField(
                    "",
                    text: Binding(
                        get: { appState.serverOptions.host },
                        set: { appState.serverOptions.host = $0.trimmingCharacters(in: .whitespaces) }
                    ),
                    prompt: Text("0.0.0.0")
                )
                .textFieldStyle(.roundedBorder)
                .font(.body.monospacedDigit())
                .multilineTextAlignment(.trailing)
                .frame(width: 160)
            }
        }
        PortRow()
        ContextSizeRow()
        if let m = meta["noVision"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.noVision)
            ) {
                Toggle("", isOn: $appState.serverOptions.noVision)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["enableMetrics"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enableMetrics)
            ) {
                Toggle("", isOn: $appState.serverOptions.enableMetrics)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["apiKey"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.apiKey)
            ) {
                SecureField(
                    "",
                    text: Binding(
                        get: { appState.serverOptions.apiKey },
                        set: { appState.serverOptions.apiKey = $0.trimmingCharacters(in: .whitespaces) }
                    ),
                    prompt: Text("none")
                )
                .textFieldStyle(.roundedBorder)
                .frame(width: 180)
            }
        }
        if let m = meta["toolAutocorrect"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.toolAutocorrect)
            ) {
                Toggle("", isOn: $appState.serverOptions.toolAutocorrect)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["logLevel"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.logLevel)
            ) {
                Picker("", selection: $appState.serverOptions.logLevel) {
                    ForEach(ServerOptions.LogLevel.allCases) { lvl in
                        Text(lvl.label).tag(lvl)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 180)
            }
        }
        if let m = meta["logToFile"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.logToFile)
            ) {
                Toggle("", isOn: $appState.serverOptions.logToFile)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["maxResidentMemGB"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.maxResidentMemGB)
            ) {
                let gb = appState.serverOptions.maxResidentMemGB
                snappingSlider(
                    presets: ServerOptions.residentMemPresets(
                        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory),
                    current: gb,
                    set: { appState.serverOptions.maxResidentMemGB = $0 },
                    label: gb == 0 ? "Auto" : "\(gb) GB"
                )
            }
        }
        if let m = meta["skipMemPreflight"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.skipMemPreflight)
            ) {
                Toggle("", isOn: $appState.serverOptions.skipMemPreflight)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

/// Port text field with commit-on-valid semantics. The field edits a local
/// string so the user can clear it or type through invalid intermediate
/// states; only values `ServerOptions.parsePort` accepts are committed to
/// storage. Submitting (or an external change like Reset to Defaults /
/// Discard) snaps the display back to the last committed value.
private struct PortRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var text: String = ""

    private var isDirty: Bool {
        guard let last = server.liveLaunchedOptions else { return false }
        return appState.serverOptions.port != last.port
    }

    var body: some View {
        if let m = ServerOptions.serverFlagFields["port"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: isDirty
            ) {
                TextField("", text: $text, prompt: Text("11234"))
                    .textFieldStyle(.roundedBorder)
                    .font(.body.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                    .onAppear { text = "\(appState.serverOptions.port)" }
                    .onChange(of: text) { _, newValue in
                        if let p = ServerOptions.parsePort(newValue) {
                            appState.serverOptions.port = p
                        }
                    }
                    .onChange(of: appState.serverOptions.port) { _, newPort in
                        if ServerOptions.parsePort(text) != newPort {
                            text = "\(newPort)"
                        }
                    }
                    .onSubmit { text = "\(appState.serverOptions.port)" }
            }
        }
    }
}

/// Snapping slider over a fixed list of common context lengths, capped at the
/// model's declared maximum. The slider position 0 is "Auto" (= use model
/// default at load time). A secondary line shows the GPU-safe ceiling for
/// this Mac and warns when the chosen value exceeds it.
private struct ContextSizeRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    // Powers of two plus 1.5× midpoints (issue #188: 32K→64K→128K jumps are
    // too coarse on a memory-limited Mac). Every value is a multiple of 1024
    // so formatTokens renders it exactly.
    private static let allPresets: [Int] = [
        0, 4_096, 6_144, 8_192, 12_288, 16_384, 24_576, 32_768,
        49_152, 65_536, 98_304, 131_072, 196_608, 262_144,
        393_216, 524_288, 786_432, 1_048_576,
    ]

    /// Drop any preset larger than the model's `max_position_embeddings` so
    /// the slider can't pick a value the model would refuse. Auto (0) always
    /// stays. We deliberately use `modelMaxTokens` (the architectural cap from
    /// config.json) — NOT `contextLength` (which is the *running* server's
    /// effective context size and would change with this very setting).
    private var presets: [Int] {
        let modelMax = server.modelInfo?.modelMaxTokens ?? 0
        guard modelMax > 0 else { return Self.allPresets }
        return Self.allPresets.filter { $0 == 0 || $0 <= modelMax }
    }

    private var currentIndex: Int {
        let value = appState.serverOptions.ctxSize
        if let i = presets.firstIndex(of: value) { return i }
        // User has a value that doesn't match a preset (legacy data) — snap
        // visually to the closest non-Auto preset without mutating storage.
        guard value > 0 else { return 0 }
        var best = 1
        for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
            best = i
        }
        return best
    }

    /// Shared with `ContextSizeDisplayTests` — the row shows three different
    /// token counts, so the formatting and the copy live in one tested place.
    private static func formatTokens(_ n: Int) -> String {
        ContextSizeDisplay.formatTokens(n)
    }

    private var isDirty: Bool {
        guard let last = server.liveLaunchedOptions else { return false }
        return appState.serverOptions.ctxSize != last.ctxSize
    }

    var body: some View {
        // The three cap pills carry their own vocabulary ("GPU-safe max"), so
        // they belong in the haystack alongside the label and help text.
        SearchableRow(searchText: [
            "Context size", ContextSizeDisplay.helpText,
            "Model max", "GPU-safe max", "In use",
        ]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    HStack(spacing: 6) {
                        Text("Context size")
                            .font(.body)
                        if isDirty {
                            Image(systemName: "arrow.clockwise.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .help("Restart the server to apply this change")
                        }
                    }
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Slider(
                            value: Binding(
                                get: { Double(currentIndex) },
                                set: { raw in
                                    let i = Int(raw.rounded())
                                    let clamped = max(0, min(i, presets.count - 1))
                                    appState.serverOptions.ctxSize = presets[clamped]
                                }
                            ),
                            in: 0...Double(max(1, presets.count - 1)),
                            step: 1
                        )
                        .frame(width: 200)
                        Text(Self.formatTokens(appState.serverOptions.ctxSize))
                            .font(.body.monospacedDigit())
                            .frame(minWidth: 56, alignment: .trailing)
                    }
                }
                Text(ContextSizeDisplay.helpText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                // Cap info: model max + GPU-safe max for this Mac. Visible only
                // when the server has reported them (after first model load).
                HStack(spacing: 12) {
                    if let modelMax = server.modelInfo?.modelMaxTokens, modelMax > 0 {
                        capPill(
                            label: "Model max",
                            value: Self.formatTokens(modelMax),
                            warn: false
                        )
                    }
                    if let safeMax = server.memoryInfo?.maxSafeContext, safeMax > 0 {
                        let chosen = appState.serverOptions.ctxSize
                        let exceeds = chosen > 0 && chosen > safeMax
                        capPill(
                            label: "GPU-safe max",
                            value: Self.formatTokens(safeMax),
                            warn: exceeds
                        )
                    }
                    // What the running server actually pinned and enforces. This is
                    // the number agent CLIs (pi / opencode / Claude Code) are handed,
                    // so "Auto" must not look like a mystery.
                    if let inUse = ContextSizeDisplay.inUseValue(
                        contextLength: server.modelInfo?.contextLength) {
                        capPill(label: "In use", value: inUse, warn: false)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func capPill(label: String, value: String, warn: Bool) -> some View {
        let labelColor: Color = warn ? .orange : .secondary
        let valueColor: Color = warn ? .orange : .primary
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(labelColor)
            Text(value)
                .font(.caption2.monospacedDigit().weight(.medium))
                .foregroundStyle(valueColor)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background((warn ? Color.orange : Color.secondary).opacity(0.10))
        .clipShape(Capsule())
    }
}

// MARK: - Spec-decode section

private struct SpecDecodeSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    /// `draftBlockSize` stays CLI-only — `recommendedBlockSize` in drafter.zig
    /// auto-picks per target (E2B=2, E4B=4, 31B=8, 26B-A4B=4); the field is
    /// kept in ServerOptions so power users who set it via CLI keep working.

    var body: some View {
        let opts = $appState.serverOptions
        // Drafter and PLD are mutually exclusive at the request level
        // (`drafter > PLD > regular` in src/server.zig). When drafter is on
        // we lock the PLD toggles down so users can't accidentally enable a
        // setting that would never apply.
        let drafterActive = !appState.serverOptions.drafterPath.isEmpty
        let pldUsable = appState.serverOptions.enablePLD && !drafterActive

        DrafterRow()
        if let m = meta["enablePLD"] {
            let suffix = drafterActive
                ? " Locked off while Drafter is on (Drafter takes priority)."
                : ""
            SettingsRow(
                title: m.title,
                explainer: m.explainer + suffix,
                isDirty: dirty.dirty(\.enablePLD)
            ) {
                Toggle("", isOn: opts.enablePLD)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .disabled(drafterActive)
            }
        }
        if let m = meta["pldDraftLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldDraftLen)
            ) {
                Stepper(value: opts.pldDraftLen, in: 1...16) {
                    Text("\(appState.serverOptions.pldDraftLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!pldUsable)
            }
        }
        if let m = meta["pldKeyLen"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.pldKeyLen)
            ) {
                Stepper(value: opts.pldKeyLen, in: 1...8) {
                    Text("\(appState.serverOptions.pldKeyLen)")
                        .font(.body.monospacedDigit())
                }
                .disabled(!pldUsable)
            }
        }

        // Native multi-token prediction — the model's OWN trained head, so it's
        // a different mechanism from PLD (which guesses by copying from the
        // prompt) and from the drafter (a separate small model). It needs no
        // extra download and no compatible pairing: a Qwen 3.5/3.6 checkpoint
        // either ships the head or it doesn't.
        SettingsSubheader("Multi-Token Prediction — Qwen 3.5 / 3.6")
        if let m = meta["enableMTP"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enableMTP)
            ) {
                Toggle("", isOn: opts.enableMTP)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["mtpDepth"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.mtpDepth)
            ) {
                // 0 is the server's "auto" sentinel (the adaptive controller);
                // 1...6 is the fixed range it accepts — 7+ hits a measured
                // occupancy cliff in the verify kernel, so it isn't offered.
                Picker("", selection: opts.mtpDepth) {
                    // Automatic is ONE entry, probe-backed internally with the
                    // per-silicon table as fallback — shipping "Automatic"
                    // beside "Probe" would ask a question nobody can answer
                    // without benchmarking. It DISPLAYS what it resolved to
                    // instead (MLX_SERVE_SPEC_COST_PROBE=0 is the A/B arm).
                    Text(server.specCost?.automaticLabel ?? "Automatic").tag(0)
                    ForEach(1...6, id: \.self) { n in
                        Text("\(n) token\(n == 1 ? "" : "s")").tag(n)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 140)
                .disabled(!appState.serverOptions.enableMTP)
            }
        }
        if let m = meta["forceMTPOnMoE"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.forceMTPOnMoE)
            ) {
                Toggle("", isOn: opts.forceMTPOnMoE)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .disabled(!appState.serverOptions.enableMTP)
            }
        }
        // DSpark is DeepSeek-V4's own draft — independent of the Qwen MTP
        // toggles above, so it is never disabled by them.
        if let m = meta["enableDSpark"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enableDSpark)
            ) {
                Toggle("", isOn: opts.enableDSpark)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

/// A labelled group INSIDE a section (e.g. the MTP block within Speculative
/// Decoding). Chrome, not content: it hides itself while a search filter is
/// active, so a group header can't be left stranded above rows the filter
/// removed. (A category selection never coexists with a query, so the sidebar
/// can't strand it either.)
private struct SettingsSubheader: View {
    let text: String
    @Environment(\.settingsSearchQuery) private var query

    init(_ text: String) { self.text = text }

    var body: some View {
        if SettingsSearch.tokens(query).isEmpty {
            Text(text)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .padding(.top, 4)
        }
    }
}

// MARK: - Performance section

private struct PerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions

        if let m = meta["maxConcurrent"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.maxConcurrent)
            ) {
                Stepper(value: opts.maxConcurrent, in: 1...8) {
                    Text("\(appState.serverOptions.maxConcurrent)")
                        .font(.body.monospacedDigit())
                }
            }
        }
        if let m = meta["anePrefill"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.anePrefill)
            ) {
                // The switch works everywhere (the server declines by name
                // where it can't run), so the per-Mac caution rides beside
                // it rather than gating it — a hidden or disabled switch on
                // a Mac that gets RAM tomorrow is the dead-control class.
                VStack(alignment: .trailing, spacing: 4) {
                    Toggle("", isOn: opts.anePrefill)
                        .labelsHidden()
                        .toggleStyle(.switch)
                    if let caution = AnePrefillAdvice.liveCaution {
                        Text(caution)
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .multilineTextAlignment(.trailing)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        if let m = meta["decodeAttnQuant"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.decodeAttnQuantChoice)
            ) {
                // Tri-state under a binary toggle: untouched (nil) RENDERS as
                // the server default (on); touching it stores an explicit
                // choice — which is exactly what the positive flag form means
                // (it opts dsv4's characterization-gated requant in).
                Toggle("", isOn: Binding(
                    get: { appState.serverOptions.decodeAttnQuantChoice ?? true },
                    set: { appState.serverOptions.decodeAttnQuantChoice = $0 }
                ))
                .labelsHidden()
                .toggleStyle(.switch)
            }
        }
        if let m = meta["kvQuant"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.kvQuant)
            ) {
                Picker("", selection: opts.kvQuant) {
                    ForEach(ServerOptions.KVQuant.allCases) { q in
                        Text(q.label).tag(q)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 220)
            }
        }
        if let m = meta["prefixCacheEntries"] {
            // Surface the RAM clamp so a 16 GB Mac user who sets, say, 8 sees
            // that the launcher will actually pass 1 (and why).
            let ram = ProcessInfo.processInfo.physicalMemory
            let set = appState.serverOptions.prefixCacheEntries
            let effective = ServerOptions.ramCappedPrefixCacheEntries(set, physicalMemoryBytes: ram)
            let capNote = effective < set
                ? "  ·  This Mac (\(MemoryInfo.format(Int64(ram)))) launches with \(effective) to keep cache memory bounded."
                : ""
            SettingsRow(
                title: m.title,
                explainer: m.explainer + capNote,
                isDirty: dirty.dirty(\.prefixCacheEntries)
            ) {
                Stepper(value: opts.prefixCacheEntries, in: 0...16) {
                    Text("\(appState.serverOptions.prefixCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
        if let m = meta["prefixCacheMem"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.prefixCacheMem)
            ) {
                TextField("", text: opts.prefixCacheMem, prompt: Text("2GB"))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
            }
        }
        if let m = meta["enablePrefixCacheDisk"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enablePrefixCacheDisk)
            ) {
                Toggle("", isOn: opts.enablePrefixCacheDisk)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
        if let m = meta["prefixCacheDisk"], appState.serverOptions.enablePrefixCacheDisk {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.prefixCacheDisk)
            ) {
                TextField("", text: opts.prefixCacheDisk, prompt: Text("10GB"))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
            }
        }
    }
}

// MARK: - Common-engine performance section

/// Knobs that apply to every backend (MLX / llama.cpp / ds4). Today this
/// is just the chat-template tokenize cache — the warm-path tokenize
/// stripper that brought a 1813-token Gemma 4 repeat from 240 ms to
/// 0.002 ms. Reorg-friendly: anything we add later that crosses engines
/// (e.g. shared HTTP timeout overrides) lands here.
private struct CommonPerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions
        if let m = meta["tokenizeCacheEntries"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.tokenizeCacheEntries)
            ) {
                Stepper(value: opts.tokenizeCacheEntries, in: 0...32) {
                    Text("\(appState.serverOptions.tokenizeCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
    }
}

// MARK: - GGUF (llama.cpp) performance section

/// Knobs specific to the embedded llama.cpp engine — surfaced only when
/// the active model loaded through that path (or pre-load, when no
/// engine has been chosen yet). MLX's `--kv-quant` and `--prefix-cache-*`
/// don't apply here; llama.cpp has its own KV scheme and its own
/// multi-session LRU.
private struct LlamaPerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions
        if let m = meta["llamaKvQuant"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.llamaKvQuant)
            ) {
                Picker("", selection: opts.llamaKvQuant) {
                    ForEach(ServerOptions.LlamaKVQuant.allCases) { q in
                        Text(q.label).tag(q)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 260)
            }
        }
        if let m = meta["llamaCacheEntries"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.llamaCacheEntries)
            ) {
                Stepper(value: opts.llamaCacheEntries, in: 1...8) {
                    Text("\(appState.serverOptions.llamaCacheEntries)")
                        .font(.body.monospacedDigit())
                }
            }
        }
    }
}

// MARK: - ds4 (DeepSeek-V4-Flash) performance section

/// Knobs specific to the embedded ds4 engine — surfaced only when the active
/// model loaded through that path (DeepSeek-V4-Flash GGUF), or pre-load when
/// no engine has been chosen yet. Today this is just SSD weight streaming:
/// the lever that lets a model larger than RAM load by streaming experts off
/// disk instead of OOMing at warmup (issue #39).
private struct Ds4PerformanceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        if let m = meta["ssdStreaming"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.ssdStreaming)
            ) {
                Toggle("", isOn: $appState.serverOptions.ssdStreaming)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }
    }
}

// MARK: - Drafter row

/// Three-state speculative-decoding toggle for the Gemma 4 assistant drafter.
private struct DrafterRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var downloads: DownloadManager

    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    /// Drafter the loaded model would pair with — nil for non-Gemma-4 or
    /// when no matching checkpoint is on disk.
    private var recommended: LocalDrafter? {
        guard let info = server.modelInfo else { return nil }
        return downloads.recommendedDrafterFor(
            modelPath: appState.selectedModelPath,
            architecture: info.architecture,
            isMoE: info.isMoE
        )
    }

    /// True when the loaded target is a Gemma 4 model (any size). Tells us
    /// whether to surface "drafter not found" (worth fixing) vs "drafter is
    /// Gemma 4 only" (architectural).
    private var targetIsGemma4: Bool {
        let arch = server.modelInfo?.architecture ?? ""
        return arch == "gemma4" || arch == "gemma4_text"
    }

    private var isMoeTarget: Bool { server.modelInfo?.isMoE ?? false }

    private var explainer: String {
        if let r = recommended {
            return "Pairs with the small assistant drafter for +27–40% on code & agents (dense Gemma 4 only). On automatically: \(r.url.lastPathComponent)."
        }
        // Server hasn't reported a model yet — either it's not started or
        // we're mid-handshake. Don't claim the architecture is wrong.
        if server.modelInfo == nil {
            if appState.selectedModelPath.isEmpty {
                return "Select a model to check drafter compatibility."
            }
            return "Start the server to check drafter compatibility."
        }
        // Server reported a model but didn't include `architecture` in its
        // /v1/models meta — that field landed in the same release that
        // unhid this row, so an older bundled binary will leave it empty.
        if (server.modelInfo?.architecture ?? "").isEmpty {
            return "Drafter status unavailable (server build pre-dates this UI). Use --drafter via CLI."
        }
        if !targetIsGemma4 {
            return "Drafter is Gemma 4 only."
        }
        if isMoeTarget {
            return "No drafter for the MoE Gemma 4 — it regresses decode there. Use PLD instead."
        }
        return "Drafter checkpoint not found. New Gemma 4 downloads bring it automatically."
    }

    /// The drafter this target would pair with, whether or not it's on disk —
    /// what the Download button fetches.
    private var pairedDrafterRepo: String? {
        DownloadManager.companionDrafterRepo(forRepoId: appState.selectedModelPath)
    }

    private var toggleEnabled: Bool { recommended != nil }

    var body: some View {
        // `explainer` is state-dependent (names the discovered checkpoint, or
        // why there isn't one), so the searchable text follows the UI.
        SearchableRow(searchText: ["Enable Assistant MTP Drafter model", explainer]) {
            rowBody
        }
    }

    @ViewBuilder
    private var rowBody: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                HStack(spacing: 6) {
                    Text("Enable Assistant MTP Drafter model")
                        .font(.body)
                    if dirty.dirty(\.drafterPath) {
                        Image(systemName: "arrow.clockwise.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .help("Restart the server to apply this change")
                    }
                }
                Spacer(minLength: 12)
                control
                    .frame(maxWidth: 280, alignment: .trailing)
            }
            Text(explainer)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Status pill — green for "ready", yellow for the MoE caution.
            if let r = recommended {
                HStack(spacing: 8) {
                    statusPill(
                        text: "✓ \(r.url.lastPathComponent)",
                        warn: false
                    )
                    if isMoeTarget && !appState.serverOptions.drafterPath.isEmpty {
                        statusPill(
                            text: "⚠ Drafter regresses ~30% on MoE — PLD is recommended",
                            warn: true
                        )
                    }
                }
                .padding(.top, 2)
            } else if server.modelInfo != nil, targetIsGemma4, let repo = pairedDrafterRepo {
                // A dense Gemma 4 is loaded but its drafter isn't on disk —
                Button(downloads.downloads[repo]?.status == .downloading
                       ? "Downloading drafter…" : "Download drafter") {
                    downloads.start(repoId: repo) { appState.refreshModels() }
                }
                .controlSize(.small)
                .disabled(downloads.downloads[repo]?.status == .downloading)
                .padding(.top, 2)
            }
        }
    }

    @ViewBuilder
    private var control: some View {
        // Off writes `drafterOptOut` as well as clearing the path: the app pairs
        // a dense Gemma 4 with its drafter on its own now, so an empty path
        // alone would read as "not paired yet" and pair itself again at the next
        // model switch. Turning it back on clears the opt-out.
        let isOn = Binding<Bool>(
            get: { !appState.serverOptions.drafterPath.isEmpty },
            set: { newValue in
                appState.serverOptions.drafterOptOut = !newValue
                if newValue {
                    if let r = recommended {
                        appState.serverOptions.drafterPath = r.url.path
                    }
                } else {
                    appState.serverOptions.drafterPath = ""
                }
            }
        )
        Toggle("", isOn: isOn)
            .labelsHidden()
            .toggleStyle(.switch)
            .disabled(!toggleEnabled)
    }

    @ViewBuilder
    private func statusPill(text: String, warn: Bool) -> some View {
        let fg: Color = warn ? .orange : .green
        Text(text)
            .font(.caption2.monospacedDigit())
            .foregroundStyle(fg)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(fg.opacity(0.10))
            .clipShape(Capsule())
    }
}

// MARK: - Per-request defaults section

private struct RequestDefaultsSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.requestDefaultFields }

    /// Snapping presets for Max Tokens. Position 0 is "Auto" (= 0 sentinel):
    /// the request omits max_tokens and the server pegs generation to the
    /// remaining context window — the right cap on a small-RAM / small-context
    /// machine, where a fixed number would over- or under-shoot. The rest are
    /// powers of 2 from 256 up to 256K plus 1.5× midpoints from 3K up
    /// (issue #188 — finer steps; every value formats exactly under the
    /// 1024-division formatter).
    private static let maxTokensPresets: [Int] = [
        0, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576,
        32768, 49152, 65536, 98304, 131072, 196608, 262144,
    ]

    /// Snapping presets for Reasoning Budget. Position 0 is the special
    /// "Unlimited" sentinel (-1); the rest are powers of 2 from 256 up to 32K.
    private static let reasoningPresets: [Int] = [
        -1, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ]

    private static func formatTokens(_ n: Int) -> String {
        if n >= 1_048_576 { return "\(n / 1_048_576)M" }
        if n >= 1024 { return "\(n / 1024)K" }
        return "\(n)"
    }

    var body: some View {
        let opts = $appState.serverOptions

        // Max Tokens — snapping slider
        if let m = meta["defaultMaxTokens"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.maxTokensPresets,
                    current: appState.serverOptions.defaultMaxTokens,
                    set: { appState.serverOptions.defaultMaxTokens = $0 },
                    label: appState.serverOptions.defaultMaxTokens <= 0
                        ? "Auto"
                        : Self.formatTokens(appState.serverOptions.defaultMaxTokens)
                )
            }
        }
        if let m = meta["defaultTemperature"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 8) {
                        Slider(value: opts.defaultTemperature, in: 0...2, step: 0.05)
                        Text(String(format: "%.2f", appState.serverOptions.defaultTemperature))
                            .font(.body.monospacedDigit())
                            .frame(minWidth: 36, alignment: .trailing)
                    }
                    recPill(server.modelInfo?.recTemperature.map { String(format: "%.2f", $0) })
                }
            }
        }
        if let m = meta["defaultTopP"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    HStack(spacing: 8) {
                        Slider(value: opts.defaultTopP, in: 0.1...1.0, step: 0.01)
                        Text(String(format: "%.2f", appState.serverOptions.defaultTopP))
                            .font(.body.monospacedDigit())
                            .frame(minWidth: 36, alignment: .trailing)
                    }
                    recPill(server.modelInfo?.recTopP.map { String(format: "%.2f", $0) })
                }
            }
        }
        if let m = meta["defaultTopK"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                VStack(alignment: .trailing, spacing: 4) {
                    Stepper(value: opts.defaultTopK, in: 0...1000) {
                        Text(appState.serverOptions.defaultTopK == 0
                             ? "Disabled"
                             : "\(appState.serverOptions.defaultTopK)")
                            .font(.body.monospacedDigit())
                    }
                    // Top-k is the one sampling field that actually falls
                    // through to the model's recommendation: when the slider
                    // reads "Disabled" (0) no `--top-k` flag is sent, so the
                    // model's own value takes effect. Say so when it's live.
                    recPill(
                        server.modelInfo?.recTopK.map { "\($0)" },
                        active: server.modelInfo?.recTopK != nil
                            && appState.serverOptions.defaultTopK == 0
                    )
                }
            }
        }
        if let m = meta["defaultRepeatPenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultRepeatPenalty, in: 1.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultRepeatPenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        if let m = meta["defaultPresencePenalty"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                HStack(spacing: 8) {
                    Slider(value: opts.defaultPresencePenalty, in: 0.0...2.0, step: 0.01)
                    Text(String(format: "%.2f", appState.serverOptions.defaultPresencePenalty))
                        .font(.body.monospacedDigit())
                        .frame(minWidth: 40, alignment: .trailing)
                }
            }
        }
        // Reasoning Budget — snapping slider; position 0 is the "Unlimited"
        // sentinel (-1).
        if let m = meta["defaultReasoningBudget"] {
            SettingsRow(title: m.title, explainer: m.explainer) {
                snappingSlider(
                    presets: Self.reasoningPresets,
                    current: appState.serverOptions.defaultReasoningBudget,
                    set: { appState.serverOptions.defaultReasoningBudget = $0 },
                    label: appState.serverOptions.defaultReasoningBudget < 0
                        ? "Unlimited"
                        : Self.formatTokens(appState.serverOptions.defaultReasoningBudget)
                )
            }
        }
    }

    /// Small "model recommends" hint pill shown under a sampling slider. The
    /// value comes from the loaded model's `generation_config.json` (surfaced
    /// over `/v1/models`); nil → nothing rendered (no model loaded, or the
    /// model ships no recommendation). `active=true` switches the styling to
    /// green + "(in effect)" for the top-k case, where a Disabled slider
    /// actually lets the model's value win.
    @ViewBuilder
    private func recPill(_ value: String?, active: Bool = false) -> some View {
        if let value {
            let color: Color = active ? .green : .secondary
            HStack(spacing: 4) {
                Text(active ? "Model default (in effect):" : "Model recommends:")
                    .font(.caption2)
                Text(value)
                    .font(.caption2.monospacedDigit().weight(.medium))
            }
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
        }
    }

}

// MARK: - Voice (wake phrase) section

/// The hands-free wake phrase ("Hey Loki" by default). App-side like the
/// clone clip — binds straight through `appState.serverOptions.wakePhrase`,
/// applied live by `VoiceModeController` (no restart). Stored as typed;
/// matching normalizes case/punctuation and the assistant renames itself
/// after the phrase's last word in the voice system prompt.
private struct WakePhraseSectionContent: View {
    @EnvironmentObject var appState: AppState

    private static let explainer = "What you say to address the assistant in hands-free voice mode. Case and punctuation don't matter, and common greetings (hey, hi, okay…) are accepted before the name. The assistant takes the last word as its name. Empty = \"Hey Loki\"."

    var body: some View {
        SearchableRow(searchText: ["Wake phrase", "Hey Loki", Self.explainer]) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Wake phrase").font(.subheadline.weight(.semibold))
                TextField("Hey Loki", text: $appState.serverOptions.wakePhrase)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 220)
                Text(Self.explainer)
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Voice (clone clip) section

/// The global voice-clone clip: pick an audio file or record a few seconds,
/// normalized via `AudioReference` (24 kHz mono WAV — what Qwen3-TTS
/// `ref_audio` expects) and copied to `~/.mlx-serve/voice-clips/` so it
/// survives relaunch. An app-side setting like the sandbox — binds straight
/// through `appState.serverOptions.voiceClonePath`, no restart banner, no CLI
/// flag. Voice mode's `ClonedVoiceSynthesizer` re-reads the path per sentence.
private struct VoiceCloneSectionContent: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var recorder = AudioRecorder()
    @State private var voiceError: String?
    /// Built lazily against the app's server so previews reuse the resident
    /// Kokoro model instead of loading it per click.
    @StateObject private var previewer = VoicePreviewer()
    @EnvironmentObject private var downloads: DownloadManager

    private static let explainer = "A few seconds of clean speech works best. Answers are synthesized locally by the Audio pane's TTS model (downloaded on first use)."
    /// Per-engine blurb. The old single string described Kokoro AND cloning and
    /// was shown under every tab, so picking "System voice" read as advice about
    /// a model you had not chosen.
    private static func engineExplainer(_ e: VoiceEngine) -> String {
        switch e {
        case .system:
            return "The built-in macOS voice. No download and no GPU, but the least natural of the three."
        case .clone:
            return "Copies the voice from the clip below using Qwen3-TTS. Slower and much heavier than Kokoro, and it needs the model downloaded from the Audio tile."
        case .kokoro:
            return "A small, very fast model (about 17x realtime, a tenth of the memory of the cloning model) with 54 built-in voices you can blend. It can't copy your voice."
        }
    }

    var body: some View {
        SearchableRow(searchText: ["Voice engine", "Kokoro", "System voice", "cloned"]) {
            engineBody
        }
        // The clip control only makes sense for the backend that can USE it —
        // Kokoro has no cloning and asking it to clone is a named 400, so the
        // control is hidden rather than left dead (the image-preset rule).
        if appState.serverOptions.voiceEngine == .clone {
            SearchableRow(searchText: ["Voice clone clip", Self.explainer, "Record", "Choose file"]) {
                clipBody
            }
        }
        if appState.serverOptions.voiceEngine == .kokoro {
            SearchableRow(searchText: ["Kokoro voice", "blend", "preview"]
                          + AudioModelPreset.kokoroVoices) {
                kokoroBody
            }
        }
    }

    @ViewBuilder
    private var engineBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Voice engine").font(.subheadline.weight(.semibold))
            Picker("", selection: $appState.serverOptions.voiceEngine) {
                ForEach(VoiceEngine.allCases, id: \.self) { e in
                    Text(e.label).tag(e)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            Text(Self.engineExplainer(appState.serverOptions.voiceEngine))
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        // Stop a preview when the engine changes — otherwise a Kokoro sample
        // keeps talking after switching to the system voice.
        .onChange(of: appState.serverOptions.voiceEngine) { _, _ in previewer.stop() }
        .onAppear { previewer.attach(server: appState.server) }
    }

    private var kokoroBundle: MediaBundle { AudioModelPreset.kokoro82M.bundle }
    private var kokoroReady: Bool { downloads.bundleReady(kokoroBundle) }

    @ViewBuilder
    private var kokoroBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Kokoro voice").font(.subheadline.weight(.semibold))
            // Selecting the engine has to be able to GET the model — the gen
            // panes have had this bar all along; Settings ▸ Voice was the one
            // place that offered a backend with no way to fetch it. Collapses to
            // nothing once the bundle is complete.
            BundleDownloadBar(bundle: kokoroBundle)
            // A bare `.frame(maxWidth:)` centres the popup inside the flexible
            // width; the row needs an explicit leading alignment plus a Spacer
            // to sit against the left edge like every other control here.
            HStack(spacing: 8) {
                Picker("", selection: $appState.serverOptions.kokoroVoice) {
                    // Grouped by language so 54 entries are navigable, and named
                    // rather than shown as raw wire ids.
                    ForEach(KokoroVoiceCatalog.grouped(), id: \.language) { group in
                        Section(group.language) {
                            ForEach(group.voices, id: \.self) { v in
                                Text(KokoroVoiceCatalog.displayName(for: v)).tag(v)
                            }
                        }
                    }
                }
                .labelsHidden()
                .frame(width: 260)

                Button {
                    previewer.preview(appState.serverOptions.kokoroVoice)
                } label: {
                    if previewer.isPreviewing(appState.serverOptions.kokoroVoice) {
                        ProgressView().controlSize(.small)
                    } else {
                        Label("Play", systemImage: "play.circle")
                    }
                }
                .help(kokoroReady ? "Hear a short sample of this voice"
                                  : "Download the model first")
                .disabled(previewer.active != nil || !kokoroReady)

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text("Voices blend: type several separated by commas (af_bella,af_sky) to make a new one.")
                .font(.caption2).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let e = previewer.error {
                Text(e).font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        // Auditioning on SELECTION is the whole point — the user asked to hear
        // the voice they just picked, not to hunt for a play button.
        // Audition on SELECTION — but only once the weights are here, or every
        // pick would fire a request that can only fail.
        .onChange(of: appState.serverOptions.kokoroVoice) { _, newValue in
            if kokoroReady { previewer.preview(newValue) }
        }
        .onDisappear { previewer.stop() }
    }

    @ViewBuilder
    private var clipBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Voice clone clip").font(.subheadline.weight(.semibold))
            HStack(spacing: 8) {
                if !appState.serverOptions.voiceClonePath.isEmpty {
                    Image(systemName: "waveform").foregroundStyle(.secondary)
                    // Prefer the display label — the stored file is always the
                    // normalized "voice-clone.wav", which says nothing.
                    Text(appState.serverOptions.voiceCloneLabel.isEmpty
                         ? (appState.serverOptions.voiceClonePath as NSString).lastPathComponent
                         : appState.serverOptions.voiceCloneLabel)
                        .font(.caption).lineLimit(1).truncationMode(.middle)
                    Button { clearVoice() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                        .help("Remove the clip — voice mode falls back to the system voice")
                } else {
                    Text("None — voice mode uses the system voice.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                Button { chooseVoiceFile() } label: { Label("Choose file…", systemImage: "folder") }
                if recorder.isRecording {
                    Button(role: .destructive) { stopRecording() } label: {
                        Label(String(format: "Stop (%.1fs)", recorder.duration), systemImage: "stop.circle")
                    }
                } else {
                    Button { startRecording() } label: { Label("Record", systemImage: "mic") }
                }
            }
            .font(.caption)
            Text(Self.explainer)
                .font(.caption2).foregroundStyle(.secondary)
            if let voiceError {
                Text(voiceError).font(.caption).foregroundStyle(.red)
            }
        }
    }

    private func chooseVoiceFile() {
        voiceError = nil
        do {
            guard let picked = try VoiceCloneMenuModel.pickAndPersistClip() else { return }
            appState.serverOptions.voiceClonePath = picked.path
            appState.serverOptions.voiceCloneLabel = picked.label
            appState.serverOptions.voiceCloneEnabled = true
        } catch {
            voiceError = error.localizedDescription
        }
    }

    private func startRecording() {
        voiceError = nil
        Task {
            guard await AudioRecorder.requestPermission() else {
                voiceError = "Microphone access denied. Enable it in System Settings ▸ Privacy ▸ Microphone."
                return
            }
            do { try recorder.start() }
            catch { voiceError = error.localizedDescription }
        }
    }

    private func stopRecording() {
        guard let data = recorder.stop() else { voiceError = "Nothing was recorded."; return }
        do {
            let normalized = try AudioReference.normalizedReferenceWav(fromRecordedPCM: data)
            appState.serverOptions.voiceClonePath = VoiceCloneClipStore.persist(normalized)
            appState.serverOptions.voiceCloneLabel = "Recorded clip"
            appState.serverOptions.voiceCloneEnabled = true
        } catch {
            voiceError = error.localizedDescription
        }
    }

    private func clearVoice() {
        appState.serverOptions.voiceClonePath = ""
        appState.serverOptions.voiceCloneLabel = ""
    }
}

// MARK: - Agent sandbox section

/// Toggle + base-image field for the agent execution sandbox. This is an
/// app-side agent-behavior setting (the tool executor reads it), NOT a
/// server-launch flag — so it binds straight through
/// `appState.serverOptions.sandbox` with no restart banner and no CLI flag.
private struct SandboxSectionContent: View {
    @EnvironmentObject var appState: AppState
    @State private var showResetConfirm = false
    @State private var resetting = false
    /// Mirrors the stored default so the row re-renders on change; writes go
    /// through `AppState.setDefaultAgentWorkspace` (retarget + VM remount),
    /// never directly through this binding.
    @AppStorage(ChatSession.defaultWorkspaceDefaultsKey) private var storedWorkspace = ""

    private var currentWorkspace: String {
        storedWorkspace.isEmpty ? ChatSession.builtinDefaultWorkingDirectory : storedWorkspace
    }

    var body: some View {
        SettingsRow(
            title: "Only use tools when I ask",
            explainer: "ON = tools are opt-in: a chat only gets them when you turn the wrench (or MCP) on yourself, and sending a message that looks like a task — \"make me a website\", \"npm install react\" — no longer stops to ask whether to enable Tools or MCP first. OFF = that suggestion still appears before such a message is sent. Either way the toolbar toggles, and any agent that decides them for its tab, are unchanged."
        ) {
            Toggle("", isOn: $appState.serverOptions.toolsOnlyWhenAsked)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Agent workspace folder",
            explainer: "The default working folder for the agent's tools (shell, readFile, writeFile, …) in every chat — and the folder shared into the sandbox VM at /workspace while the sandbox is on. Changing it moves chats still on the previous default, remounts a running sandbox, and restarts any open terminal sessions in the new folder; a chat with its own picked folder (the folder icon on the Agent pill) keeps it."
        ) {
            HStack(spacing: 8) {
                Text((currentWorkspace as NSString).abbreviatingWithTildeInPath)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(currentWorkspace)
                Button("Choose…") {
                    if let picked = WorkspacePicker.pickDirectory() {
                        appState.setDefaultAgentWorkspace(picked)
                    }
                }
            }
        }

        // No host shell in the App Store build → the sandbox can't be turned
        // off (`AgentSandbox.resolveEnabled`), so offering the toggle would be
        // a lie; the base image is likewise locked to the bundled guest.
        if BuildFeatures.current.hostShell {
            SettingsRow(
                title: "Sandbox agent commands",
                explainer: "OFF = the agent runs shell commands directly on macOS (fast, full access to your files). ON = commands run inside an isolated Linux sandbox that can only touch the current working folder, so a bad command can't harm the rest of your Mac. Costs a bit more memory while active because it spins up a lightweight virtual machine for the session."
            ) {
                Toggle("", isOn: $appState.serverOptions.sandbox.enabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
            }
        }

        SettingsRow(
            title: "Network + port mapping",
            explainer: "ON = the sandbox has outbound internet (NAT), and any server the agent starts inside it is automatically reachable on this Mac at localhost with the same port — e.g. a dev server on 8080 appears at localhost:8080 (bound to localhost only, never your LAN). OFF = the sandbox gets no network device at all: fully isolated, but installs and downloads inside it will fail. Applies to the next sandbox session."
        ) {
            Toggle("", isOn: $appState.serverOptions.sandbox.network)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Reset sandbox",
            explainer: "Deletes ALL sandbox data and returns it to factory state: the downloaded guest image and everything inside it (installed CLIs like pi/hermes, their configs and logins, any files created outside the shared workspace), the cached kernel, the sandbox ssh identity, and the activity transcript. Any running guest and live agent sessions are stopped immediately. Your workspace folder, models, and other app data on this Mac are not touched. The sandbox re-provisions itself on next use."
        ) {
            Button(role: .destructive) {
                showResetConfirm = true
            } label: {
                if resetting {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Resetting…")
                    }
                } else {
                    Label("Reset Sandbox…", systemImage: "trash")
                        .foregroundStyle(.red)
                }
            }
            .disabled(resetting)
            .confirmationDialog(
                "Reset the Agent Sandbox?",
                isPresented: $showResetConfirm,
                titleVisibility: .visible
            ) {
                Button("Delete All Sandbox Data", role: .destructive) {
                    resetting = true
                    AgentSandbox.shared.resetAllData {
                        resetting = false
                    }
                }
                .keyboardShortcut(.defaultAction)
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("""
                This permanently deletes everything the sandbox has downloaded and every change made inside it — installed agent CLIs (pi, hermes), their configs and logins, and any files outside the shared workspace. Any running guest and live sessions stop immediately.

                Files in your workspace folder on this Mac are kept. This cannot be undone.
                """)
            }
        }
    }
}

// MARK: - Messaging (Telegram bot) section

/// Settings for the Telegram bot bridge. The whole thing is two steps for the
/// user: create a bot in @BotFather, paste the token, flip the switch — then
/// message the bot once to lock it to your chat (trust-on-first-use). `@Observed`
/// on the live bridge so the status pill updates as it connects.
private struct MessagingSectionContent: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject var bridge: TelegramBridge

    private var telegram: ServerOptions.TelegramConfig { appState.serverOptions.telegram }

    var body: some View {
        // Live status pill (only meaningful once enabled).
        if telegram.enabled {
            SearchableRow(searchText: ["Status", "Telegram bot bridge connection status"]) {
                HStack(spacing: 8) {
                    Text("Status")
                        .font(.body)
                    Spacer(minLength: 12)
                    statusPill
                }
            }
        }

        SettingsRow(
            title: "Enable Telegram bot",
            explainer: "Long-polls Telegram for messages and relays them to your local model. Needs a bot token (below) and a running model."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enabled)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Bot token",
            explainer: "Paste the token @BotFather gives you after /newbot. Stored locally on this Mac and sent only to Telegram's API."
        ) {
            TextField("", text: $appState.serverOptions.telegram.botToken,
                      prompt: Text("123456:ABC-DEF…"))
                .textFieldStyle(.roundedBorder)
                .font(.body.monospaced())
                .frame(width: 260)
        }

        SettingsRow(
            title: "Tools",
            explainer: "OFF = plain chat (safe). ON = the bot can run shell commands and read/write files on this Mac, triggered from your phone. Confined to ~/.mlx-serve/telegram-workspace. Only enable if you understand the risk — anyone who can message the locked chat gets this power."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.agentMode)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "MCP tools",
            explainer: "Expose your enabled MCP servers (configured in the MCP marketplace) to the bot and to the tasks it creates. Works with or without Tools. Servers start on first use."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.useMCP)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Enable thinking",
            explainer: "Send reasoning-enabled requests for models that support it. The bot replies with the final answer only (no thinking trace)."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enableThinking)
                .labelsHidden()
                .toggleStyle(.switch)
        }

        SettingsRow(
            title: "Answer as agent",
            explainer: "Reply as one of your agents (Chat window ▸ Agents): its prompt, tools, model and workspace. \"None\" uses the settings above."
        ) {
            Picker("", selection: $appState.serverOptions.telegram.agentId) {
                Text("None").tag(UUID?.none)
                ForEach(appState.agents.allAgents) { agent in
                    Text(agent.name).tag(UUID?.some(agent.id))
                }
            }
            .labelsHidden()
            .frame(width: 200)
        }

        // Allow-list / lock control.
        SearchableRow(searchText: ["Locked to", "Reset lock", Self.lockExplainer]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Locked to")
                        .font(.body)
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(lockLabel)
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(telegram.allowedChatIds.isEmpty ? .secondary : .primary)
                        Button("Reset lock") {
                            appState.serverOptions.telegram.allowedChatIds = []
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(telegram.allowedChatIds.isEmpty)
                    }
                }
                Text(Self.lockExplainer)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        // Setup steps. The divider rides inside the searchable row so a
        // filtered view never leaves a dangling separator behind.
        SearchableRow(searchText: ["Setup", "BotFather", "newbot", "token", "lock it to your chat"]) {
            VStack(alignment: .leading, spacing: 6) {
                Divider()
                    .padding(.bottom, 6)
                Text("Setup")
                    .font(.caption.weight(.semibold))
                Text("1. In Telegram, open @BotFather and send /newbot.")
                    .font(.caption2).foregroundStyle(.secondary)
                Text("2. Copy the token it gives you and paste it above.")
                    .font(.caption2).foregroundStyle(.secondary)
                Text("3. Turn on “Enable Telegram bot”, then message your bot once to lock it to your chat.")
                    .font(.caption2).foregroundStyle(.secondary)
                Link("Open @BotFather ↗", destination: URL(string: "https://t.me/botfather")!)
                    .font(.caption2)
            }
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 2)
        }
    }

    private static let lockExplainer = "The first chat that messages the bot is adopted as the owner; everyone else is refused. Reset to hand the bot to a different chat."

    private var lockLabel: String {
        let ids = telegram.allowedChatIds
        switch ids.count {
        case 0: return "no chat yet (first to message wins)"
        case 1: return "chat \(ids[0])"
        default: return "\(ids.count) chats"
        }
    }

    @ViewBuilder
    private var statusPill: some View {
        let (text, color): (String, Color) = {
            switch bridge.status {
            case .off:               return ("Off", .secondary)
            case .connecting:        return ("Connecting…", .orange)
            case .listening(let u):  return (u.map { "Listening as @\($0)" } ?? "Listening", .green)
            case .error(let m):      return (m, .red)
            }
        }()
        Text(text)
            .font(.caption2.monospaced())
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .clipShape(Capsule())
            .lineLimit(2)
            .frame(maxWidth: 280, alignment: .trailing)
    }
}

// MARK: - Updates section

/// Auto-update controls: the daily-check toggle, a manual check button with
/// inline status, and — once a newer release is known — an install row that
/// mirrors the tray banner's one-click update.
private struct UpdatesSectionContent: View {
    @ObservedObject var updates: UpdateChecker
    /// Read once from `mlx-serve --version` (a print-and-exit that never boots
    /// the server), so the embedded-engine versions show even when it's stopped.
    @State private var engineVersions: [EngineVersion] = []

    /// Engine rows to display — drops the `mlx-serve` app row (already shown as
    /// "Installed version"). Falls back to the compile-time llama pin so the
    /// section is never empty if the probe hasn't returned yet.
    private var engineRows: [EngineVersion] {
        let rows = engineVersions.filter { $0.name != "mlx-serve" }
        return rows.isEmpty
            ? [EngineVersion(name: "llama.cpp", version: UpdateChecker.bundledLlamaTag)]
            : rows
    }

    var body: some View {
        SettingsRow(
            title: "Check for updates automatically",
            explainer: "Checks the GitHub releases page once a day and shows an update banner in the menu bar tray when a newer version ships. No data beyond the version request leaves this Mac."
        ) {
            Toggle("", isOn: Binding(
                get: { updates.autoCheckEnabled },
                set: { updates.autoCheckEnabled = $0 }))
                .toggleStyle(.switch)
                .labelsHidden()
        }

        SettingsRow(
            title: "Installed version — v\(updates.currentVersion)",
            explainer: statusText
        ) {
            Button {
                Task { await updates.check(userInitiated: true) }
            } label: {
                if case .checking = updates.phase {
                    ProgressView().controlSize(.small)
                } else {
                    Text("Check Now")
                }
            }
            .disabled(busy)
        }

        // Embedded-engine versions, read from `mlx-serve --version` — a
        // print-and-exit path that never binds a port or loads a model, so
        // Settings shows them even when the server is stopped.
        ForEach(engineRows) { row in
            SettingsRow(
                title: "\(Self.engineLabel(row.name)) — \(row.version)",
                explainer: Self.engineExplainer(row.name)
            ) {
                EmptyView()
            }
        }
        .task {
            guard engineVersions.isEmpty else { return }
            engineVersions = await EngineVersions.probe(binaryPath: ServerManager.resolveBinaryPath())
        }

        if let update = updates.available {
            SettingsRow(
                title: "MLX Core v\(update.version) is available",
                explainer: "Downloads MLXCore.dmg from the release, replaces the app, and relaunches."
            ) {
                switch updates.phase {
                case .downloading(let fraction):
                    ProgressView(value: max(0, min(1, fraction)))
                        .progressViewStyle(.linear)
                        .frame(width: 160)
                case .installing:
                    ProgressView().controlSize(.small)
                default:
                    Button("Download & Install") {
                        Task { await updates.downloadAndInstall() }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
    }

    private var busy: Bool {
        switch updates.phase {
        case .checking, .downloading, .installing: return true
        default: return false
        }
    }

    /// Friendly display name for a raw `--version` component name.
    // Label + explainer text live on EngineVersions (pure, tested —
    // EngineVersionsTests) so the display mapping and the parse contract
    // stay in one place.
    private static func engineLabel(_ name: String) -> String {
        EngineVersions.displayLabel(name)
    }

    private static func engineExplainer(_ name: String) -> String {
        EngineVersions.explainer(name)
    }

    private var statusText: String {
        switch updates.phase {
        case .upToDate:
            return "You're on the latest release."
        case .failed(let message):
            return "Update failed: \(message)"
        case .installing:
            return "Installing — the app will relaunch."
        case .downloading:
            return "Downloading the update…"
        default:
            return "Releases are published at github.com/\(UpdateChecker.repo)/releases."
        }
    }
}

// MARK: - Shared numeric control

/// A slider that snaps to a discrete preset list — the float value is the index
/// into `presets`, rounding pins to the nearest entry. Shared: a setting whose
/// useful values are a ladder (memory caps, token budgets) reads better as this
/// than as a free-text field, and a value picked off a ladder cannot be a typo
/// the launcher has to defend against.
@ViewBuilder
private func snappingSlider(
    presets: [Int],
    current: Int,
    set: @escaping (Int) -> Void,
    label: String
) -> some View {
    let safePresets = presets.isEmpty ? [0] : presets
    let currentIdx = closestPresetIndex(in: safePresets, to: current)
    HStack(spacing: 8) {
        Slider(
            value: Binding(
                get: { Double(currentIdx) },
                set: { raw in
                    let i = Int(raw.rounded())
                    let clamped = max(0, min(i, safePresets.count - 1))
                    set(safePresets[clamped])
                }
            ),
            in: 0...Double(max(1, safePresets.count - 1)),
            step: 1
        )
        .frame(width: 200)
        Text(label)
            .font(.body.monospacedDigit())
            .frame(minWidth: 70, alignment: .trailing)
    }
}

/// Index of the preset closest to `value`, so a stored value not on the snap
/// grid still positions the slider sensibly.
private func closestPresetIndex(in presets: [Int], to value: Int) -> Int {
    if let exact = presets.firstIndex(of: value) { return exact }
    var best = 0
    for i in 1..<presets.count where abs(presets[i] - value) < abs(presets[best] - value) {
        best = i
    }
    return best
}
