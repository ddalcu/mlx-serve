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

    /// What the sections below the containers edit — see `SettingsFormState`.
    @StateObject private var formState = SettingsFormState()

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
        VStack(spacing: 0) {
            SettingsSearchField(text: $searchQuery)
                // Typing searches across EVERYTHING, so it snaps the sidebar
                // back to All — a search that silently only looked inside the
                // selected category would hide its own best answers.
                .onChange(of: searchQuery) { _, q in
                    selection = SettingsSelection.afterQueryEdit(query: q, current: selection)
                }
            SettingsSidebar(categories: categories, selection: $selection) {
                // Picking a category clears the search: the two are alternative
                // ways to narrow, and letting them stack strands the user on an
                // empty pane with a stale query they can't see.
                searchQuery = ""
            }
        }
    }

    private var form: some View {
        VStack(spacing: 0) {
            if server.needsRestartFor(appState.serverOptions) {
                RestartBanner()
            }
            ScrollView {
                // Lazy while nothing is filtered — a `ScrollView` is measured from
                // its content, so an eager stack lays out every section. A query
                // stays eager: a deferred section publishes no count to collapse on.
                Group {
                    if filtering {
                        VStack(alignment: .leading, spacing: 0) {
                            sections
                        }
                    } else {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            sections
                        }
                    }
                }
                .environment(\.settingsSearchQuery, searchQuery)
                .environment(\.settingsSelection, selection)
                .environmentObject(formState)
                .onPreferenceChange(SettingsVisibleRowCountKey.self) { visibleRows = $0 }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    /// The form's sections, in sidebar order. ONE list, so the lazy and the eager
    /// container cannot drift.
    @ViewBuilder
    private var sections: some View {
        SettingsSection(
            category: .server,
            subtitle: "How the server starts and where your models live; restart the server to apply changes."
        ) {
            ModelFoldersSectionContent()
            ServerSectionContent()
        }
        SettingsSection(
            category: .memory,
            subtitle: "How much memory models and caches may use; restart the server to apply changes."
        ) {
            MemorySectionContent(showMLX: server.modelInfo?.engine.isMlxPath ?? true)
        }
        SettingsSection(
            category: .lanSharing,
            subtitle: "Use your other machines. Over your network, each one takes whole jobs; a Mac joined to this one by a Thunderbolt cable works on the same job with it."
        ) {
            ClusterMapView(worker: appState.clusterWorker)
            LanSharingSectionContent()
            ThunderboltClusterSectionContent()
        }
        SettingsSection(
            category: .providers,
            subtitle: "Add other OpenAI-compatible services, and their models appear in the model picker."
        ) {
            ProvidersSectionContent()
        }
        // Engine-aware sections. Each panel is hidden when its
        // controls don't apply to the active engine — flipping
        // `--kv-quant` on a GGUF model silently no-ops, so we'd
        // rather not show that picker at all than mislead.
        EngineAwareSections()
        SettingsSection(
            category: .requestDefaults,
            subtitle: "Defaults for every request, such as temperature, used from the next message on."
        ) {
            GlobalGenerationDefaultsView()
        }

        SettingsSection(
            category: .interface,
            subtitle: "How the app looks and how you open the Quick Launcher."
        ) {
            InterfaceSectionContent()
        }

        SettingsSection(
            category: .voice,
            subtitle: "Your wake phrase and voice for hands-free mode, and how meeting transcripts start, stop and get their notes."
        ) {
            WakePhraseSectionContent()
            VoiceCloneSectionContent()
            TranscribeSectionContent()
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
            subtitle: "Chat with your local model from your phone through a Telegram bot."
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
                subtitle: "Install new versions from GitHub while keeping your chats, models and settings."
            ) {
                UpdatesSectionContent(updates: appState.updates)
            }
        }

        // Not folded into Updates: that section is gated on
        // `selfUpdate`, so on a Mac App Store build these links
        // would never render.
        SettingsSection(
            category: .about,
            subtitle: "mlx-serve is free and open source, made by one person, and questions and bug reports are welcome."
        ) {
            ForEach(CommunityLinks.all) { item in
                SettingsRow(title: item.title, explainer: item.explainer) {
                    Link(L10n.text(item.actionLabel), destination: item.url)
                }
            }
        }

        if filtering && visibleRows == 0 {
            NoSearchResults(query: searchQuery) { searchQuery = "" }
        }

        ResetDefaultsFooter()
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
            Label("All Settings", systemImage: "square.grid.2x2").font(.app(.body))
                .tag(SettingsSelection.all)
            Section {
                ForEach(categories) { category in
                    Label(L10n.text(category.sidebarLabel), systemImage: category.icon)
                        .tag(SettingsSelection.category(category)).font(.app(.body))
                }
            }
        }
        // The sidebar rows are the app's other callouts, not the platform
        // default: without this they render at 13pt while every pane they open
        // reads from the ladder.
        .font(.app(.callout))
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
    /// The active filter text. `fileprivate` to this file's rows plus the
    /// stacks that draw their own lines under it.
    var settingsSearchQuery: String {
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
struct SearchableRow<Content: View>: View {
    let searchText: [String]
    /// Gap the row sits in, for the hairline drawn above it — the card's
    /// own by default, so the next row someone adds has its line without
    /// asking. A row inside a sub-stack of its own passes that stack's
    /// spacing; one that draws its own line (or none at all) passes nil.
    /// Inside the filter, so a row the query hides takes its line with it.
    var separatorGap: CGFloat? = SettingsCardMetrics.gap
    @ViewBuilder var content: Content

    @Environment(\.settingsSearchQuery) private var query

    var body: some View {
        if SettingsSearch.matches(query: query, in: searchText) {
            separated.preference(key: SettingsVisibleRowCountKey.self, value: 1)
        }
    }

    @ViewBuilder
    private var separated: some View {
        if let separatorGap { content.settingsRowSeparator(gap: separatorGap) }
        else { content }
    }
}

/// The hairline between two rows of a settings card — the line macOS System
/// Settings draws, so a dense card reads as a list rather than a pile. It is
/// the row's BACKGROUND, never its layout: the row keeps the height it had,
/// and the line lands in the middle of the gap above it.
private struct SettingsRowSeparator: ViewModifier {
    let gap: CGFloat

    func body(content: Content) -> some View {
        content.background(alignment: .top) {
            Divider()
                .offset(y: -gap / 2)
        }
    }
}

extension View {
    /// See `SettingsRowSeparator`.
    func settingsRowSeparator(gap: CGFloat) -> some View {
        modifier(SettingsRowSeparator(gap: gap))
    }
}

/// Filter field at the top of the sidebar, above "All Settings".
private struct SettingsSearchField: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Filter settings", text: $text)
                .textFieldStyle(.plain).font(.app(.body))
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
        .padding(.horizontal, 10)
        .padding(.top, 12)
        .padding(.bottom, 4)
    }
}

/// Shown when the filter matches nothing at all.
private struct NoSearchResults: View {
    let query: String
    let clear: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .font(.app(.title2))
                .foregroundStyle(.secondary)
            Text(L10n.format("No settings match \"%@\"", query))
                .font(.app(.subheadline))
                .foregroundStyle(.secondary)
            Button(action: clear, label: { Text(L10n.text("Clear filter"))
                .font(.app(.body)) })
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
    @State private var resetError: String?

    private var label: String { SettingsReset.buttonLabel(selection) }

    private var helpText: String { SettingsReset.confirmMessage(selection) }

    /// Nothing to reset in this section → no button.
    private var hidden: Bool {
        if case .category(let c) = selection { return !SettingsReset.isResettable(c) }
        return false
    }

    var body: some View {
        if !hidden {
            SearchableRow(searchText: [label, helpText],
                          separatorGap: nil) {
                HStack {
                    Spacer()
                    Button(role: .destructive) {
                        showConfirm = true
                    } label: {
                        Label(L10n.text(label), systemImage: "arrow.uturn.backward.circle").font(.app(.body))
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .help(helpText)
                }
                .padding(.top, 4)
                .overlay(alignment: .bottomLeading) {
                    if let resetError { Text(verbatim: resetError).font(.app(.caption)).foregroundStyle(.red) }
                }
                .confirmationDialog(
                    SettingsReset.confirmTitle(selection),
                    isPresented: $showConfirm,
                    titleVisibility: .visible
                ) {
                    Button(role: .destructive) {
                        do {
                            if selection == .all || selection == .category(.requestDefaults) {
                                try GenerationDefaultsFile.save(.init())
                            }
                            appState.serverOptions = SettingsReset.apply(selection, to: appState.serverOptions)
                            resetError = nil
                        } catch { resetError = "Could not reset generation defaults: \(error.localizedDescription)" }
                    } label: { Text("Reset")
                        .font(.app(.body)) }
                    .keyboardShortcut(.defaultAction)
                    Button(role: .cancel) { } label: { Text("Cancel")
                        .font(.app(.body)) }
                } message: {
                    Text(L10n.text(helpText))
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
                .font(.app(.title2))
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(L10n.text("Some changes require a server restart"))
                    .font(.app(.subheadline).weight(.semibold))
                Text(L10n.text("Click Restart Now to apply, or Discard to revert the unsaved server-launch fields."))
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                let opts = appState.serverOptions
                let model = appState.selectedModelPath
                server.stop()
                if !model.isEmpty {
                    server.start(modelPath: model, options: opts)
                }
            } label: { Text("Restart Now")
                .font(.app(.body)) }
            .buttonStyle(.borderedProminent)
            .disabled(appState.selectedModelPath.isEmpty)

            Button {
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
            } label: { Text("Discard")
                .font(.app(.body)) }
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
///   - GGUF / DSV4:  Common Performance
///   Engines (every engine's own flags) always renders.
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
        let showMLX = engine?.isMlxPath ?? true

        if showMLX {
            SettingsSection(
                category: .specDecode,
                subtitle: "Ways to make replies faster by guessing ahead, for MLX models only."
            ) {
                SpecDecodeSectionContent()
            }
        }

        // ONE Performance section. The universal rows always apply; the MLX-only
        // ones (continuous batching, attention requant, SSD prefix cache) join
        // them when an MLX model is serving — on GGUF/DSV4 they'd silently no-op,
        // so they stay hidden rather than lie.
        SettingsSection(
            category: .performance,
            subtitle: showMLX
                ? "Continuous batching, decode attention requant, and the SSD prefix cache. Server-launch flags — restart to apply."
                : "Tunables that apply regardless of engine. Server-launch flags — restart to apply."
        ) {
            CommonPerformanceSectionContent()
            if showMLX {
                PerformanceSectionContent()
            }
        }

        // Unconditional: the media offloads apply to generation models, which
        // are not the text engine the gates above key on.
        SettingsSection(
            category: .neuralEngine,
            subtitle: "Use the Apple Neural Engine alongside the GPU, at the cost of extra memory and disk; restart the server to apply changes."
        ) {
            NeuralEngineSectionContent()
        }

        // Always on screen: these rows configure engines that are NOT loaded
        // yet, so gating them on the active engine hid the way to opt into
        // the next one.
        SettingsSection(
            category: .engines,
            subtitle: "Settings for the engines that run .gguf files; restart the server to apply changes."
        ) {
            EnginesSectionContent()
        }

        // The pre-tune banner explains why EVERY engine section is on screen —
        // meaningless once the sidebar has narrowed to one of them.
        if engine == nil, selection == .all, SettingsSearch.tokens(query).isEmpty {
            HStack(spacing: 8) {
                Image(systemName: "info.circle")
                    .foregroundStyle(.secondary)
                Text(L10n.text("No model loaded yet — every section is shown so you can pre-tune. Once a model is active, sections that don't apply will hide automatically."))
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 24)
        }
    }
}

// MARK: - Section frame

/// Space between two rows of a settings card, and the card's own inset. The
/// rows' hairlines are placed from these two numbers, so they move together.
private enum SettingsCardMetrics {
    static let gap: CGFloat = 18
    static let inset: CGFloat = 16
}

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
                    Text(L10n.text(title))
                        .font(.app(.title3).weight(.semibold))
                    Text(L10n.text(subtitle))
                        .font(.app(.caption))
                        .foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: collapsed ? 0 : SettingsCardMetrics.gap) {
                content
            }
            .environment(\.settingsSearchQuery, filter.childQuery)
            .padding(collapsed ? 0 : SettingsCardMetrics.inset)
            .background(collapsed ? Color.clear : Color(NSColor.controlBackgroundColor))
            // The FIRST row's hairline lands inside this inset — empty card
            // ground, painted over, so a card never opens on a line, and a
            // filtered card whose surviving row is not the first is covered
            // the same way. Only the hairline's own 2 pt band is covered,
            // though: the first row's chrome — a focused field's ring, a
            // control's shadow — reaches into this inset too, and a
            // full-inset cover would eat it.
            .overlay(alignment: .top) {
                if !collapsed {
                    Color(NSColor.controlBackgroundColor)
                        .frame(height: 2)
                        .offset(y: SettingsCardMetrics.inset - SettingsCardMetrics.gap / 2)
                }
            }
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
    /// The setting's memory/disk cost; orange and bold while it is switched on.
    var cost: String? = nil
    var costActive: Bool = false
    /// A row drawn inside a card's own sub-stack passes that stack's spacing
    /// instead; nil leaves the line off (a row that draws its own).
    var separatorGap: CGFloat? = SettingsCardMetrics.gap
    @ViewBuilder var control: Control

    var body: some View {
        SearchableRow(searchText: [title, explainer] + [cost].compactMap { $0 },
                      separatorGap: separatorGap) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    HStack(spacing: 6) {
                        Text(L10n.text(title))
                            .font(.app(.rowTitle))
                        if isDirty {
                            Image(systemName: "arrow.clockwise.circle.fill")
                                .font(.app(.caption))
                                .foregroundStyle(.orange)
                                .help("Restart the server to apply this change")
                        }
                    }
                    Spacer(minLength: 12)
                    control
                        .frame(maxWidth: 280, alignment: .trailing)
                }
            Text(L10n.text(explainer))
                    .font(.app(.explainer))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let cost {
                    Text(L10n.text(cost))
                        .font(.app(.value))
                        .fontWeight(costActive ? .semibold : .regular)
                        .foregroundStyle(costActive ? Color.orange : Color.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
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

    private static let explainer = "Another folder to scan for models, laid out as <name>/ or <author>/<name>/."
    private static let defaultExplainer = "Where new downloads go; models you already downloaded keep working."

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
                    Text(L10n.text("Default folder"))
                        .font(.app(.body))
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(configured ?? roots.downloadRoot)
                            .font(.app(.caption).monospaced())
                            // An unreachable folder is shown in the warning
                            // colour rather than swapped for the fallback: the
                            // path the user chose is the thing they need to see
                            // to understand where their downloads went instead.
                            .foregroundStyle(unavailable ? AnyShapeStyle(.orange) : AnyShapeStyle(configured == nil ? .secondary : .primary))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: 220, alignment: .trailing)
                        Button { chooseDownloadFolder() } label: { Text("Choose…")
                            .font(.app(.body)) }
                            .buttonStyle(.bordered)
                        Button {
                            SecurityScopedBookmark.clear(name: DownloadManager.downloadFolderBookmarkName)
                            ModelRoots().configuredDownloadRoot = nil
                            applyFolderChange()
                        } label: { Text("Reset")
                            .font(.app(.body)) }
                        .buttonStyle(.bordered)
                        .disabled(configured == nil)
                    }
                }
                Text(unavailable
                     ? L10n.format("That folder isn't reachable right now, so downloads are going to %@ instead.", ModelRoots.builtInRoot)
                     : L10n.text(Self.defaultExplainer))
                    .font(.app(.caption2))
                    .foregroundStyle(unavailable ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .id(downloadFolderTick)
        }
    }

    private func chooseDownloadFolder() {
        let panel = OpenPanel.make()
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
                    Text(L10n.text("Custom folder"))
                        .font(.app(.body))
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(L10n.text(pathText))
                            .font(.app(.caption).monospaced())
                            .foregroundStyle(hasPath ? .primary : .secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: 220, alignment: .trailing)
                        Button { choose() } label: { Text("Choose…")
                            .font(.app(.body)) }
                            .buttonStyle(.bordered)
                        Button {
                            downloads.customRoot = nil
                            appState.refreshModels()
                        } label: { Text("Clear")
                            .font(.app(.body)) }
                        .buttonStyle(.bordered)
                        .disabled(!hasPath)
                    }
                }
                Text(L10n.text(Self.explainer))
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func choose() {
        let panel = OpenPanel.make()
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
    @State private var peerKeys: [String: String] = [:]

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        SearchableRow(searchText: ["Network", "LAN sharing"], separatorGap: nil) {
            VStack(alignment: .leading, spacing: 4) {
                Label("Over your network", systemImage: "network").font(.app(.headline)).padding(.top, 8)
                Text("Any machine on your network, Mac or Linux, can take whole jobs: a chat, a picture, a storyboard shot. It works over Wi-Fi or Ethernet, and each machine works on its own.")
                    .font(.app(.caption)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        if let m = meta["lanShareEnabled"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanShareEnabled)) {
                Toggle("", isOn: $appState.serverOptions.lanShareEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch).font(.app(.body))
            }
        }
        if appState.serverOptions.lanShareEnabled {
            if let m = meta["lanShareAll"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanShareAll)) {
                    Toggle("", isOn: $appState.serverOptions.lanShareAll)
                        .labelsHidden()
                        .toggleStyle(.switch).font(.app(.body))
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
                        prompt: Text(Host.current().localizedName ?? L10n.text("this Mac"))
                    )
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 200).font(.app(.body))
                }
            }
        }
        if let m = meta["lanDiscoverEnabled"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.lanDiscoverEnabled)) {
                Toggle("", isOn: $appState.serverOptions.lanDiscoverEnabled)
                    .labelsHidden()
                    .toggleStyle(.switch).font(.app(.body))
            }
            .task {
                peerKeys = LanPeerKeys.load()
                await server.refreshModels()
            }
        }
        if appState.serverOptions.lanDiscoverEnabled {
            LanPeerKeyRows(keys: $peerKeys)
        }
        // The privacy disclosure — sharing means running other people's
        // prompts, and using a network model means the host reads yours.
        SearchableRow(searchText: ["Privacy", Self.privacyNote],
                      separatorGap: nil) {
            Text(L10n.text(Self.privacyNote))
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 4)
        }
    }

    private static let privacyNote = "Privacy: prompts sent to a model you share are processed on — and visible to — this Mac. Prompts you send to a network model are visible to the Mac hosting it. Traffic stays on your local network, and everything here is off unless you turn it on."

    /// One checkbox per local model name. Names are deduped — a GGUF and an
    /// MLX build of the same repo share a name and are shared together (the
    /// server matches share entries against registry ids basename-tolerantly).
    private var sharedModelList: some View {
        let names = Array(Set(appState.localModels.map(\.name))).sorted()
        return VStack(alignment: .leading, spacing: 4) {
            if names.isEmpty {
                Text(L10n.text("No local models yet — download one first."))
                    .font(.app(.caption))
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
                .font(.app(.caption))
            }
        }
        .padding(.leading, 8)
        .padding(.vertical, 2)
    }
}

/// Settings ▸ Cluster, Thunderbolt half: two Macs on one cable working on the same job.
/// One button pairs them; the controls behind it sit under Advanced.
private struct ThunderboltClusterSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.settingsSearchQuery) private var query
    @State private var ports: [RdmaPort] = []
    @State private var setupError: String?
    /// A Pair, Join or Reconnect is waiting for the cable's address.
    @State private var settingUp = false

    private func pairing(_ action: @escaping @MainActor () async -> String?) {
        settingUp = true
        setupError = nil
        Task {
            setupError = await action()
            settingUp = false
        }
    }
    /// A Mac on the network that leads a pair and waits for one to join.
    @State private var leader: String?
    @State private var leadInstead = false
    @State private var pairModel = ""
    @State private var showAdvanced = false

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }
    private var options: ServerOptions { appState.serverOptions }
    private var port: RdmaPort? {
        let dev = ClusterSetup.device(options, ports: { ports })
        return ports.first { $0.device == dev }
    }

    var body: some View {
        SearchableRow(searchText: ["Thunderbolt", "RDMA", "pair", "cluster"], separatorGap: nil) {
            VStack(alignment: .leading, spacing: 4) {
                Label("Thunderbolt pair", systemImage: "bolt.horizontal.fill").font(.app(.headline)).padding(.top, 12)
                Text("Two Macs joined by a Thunderbolt 5 cable work on the same job together: one video made faster, or a chat model too big for one Mac. Only the two Macs on the cable take part.")
                    .font(.app(.caption)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        SearchableRow(searchText: ["Pair", "Join", "Thunderbolt"], separatorGap: nil) {
            pairCard
        }
        .task { await watch() }
        SearchableRow(searchText: ["Advanced", "Thunderbolt"], separatorGap: nil) {
            Button { showAdvanced.toggle() } label: {
                Label(showAdvanced ? L10n.text("Hide advanced Thunderbolt settings") : L10n.text("Advanced Thunderbolt settings"),
                      systemImage: showAdvanced ? "chevron.down" : "chevron.right")
                    .font(.app(.caption))
            }
            .buttonStyle(.link)
        }
        if showAdvanced || !query.isEmpty { advancedRows }
    }

    private var pair: ThunderboltPair {
        let up = ports.first(where: \.active)
        let addressed = up.map { ClusterSetup.ipv4(on: $0.interface) == ClusterSetup.linkAddress(options.clusterRole) } ?? false
        return .of(ports: ports, role: options.clusterRole, hasAddress: addressed, cluster: server.cluster, leader: leader)
    }

    /// The models a pair can run, video first.
    /// Chat models too big for this Mac alone first (what a pair is for), then the rest.
    private var pairable: [LocalModel] {
        let models = appState.localModels.filter { ClusterSetup.servableTypes.contains($0.modelType) }
        let here = Int64(SystemMetrics.availableBytes())
        func needsPair(_ m: LocalModel) -> Bool {
            m.modelType != "minimax_h3" && PairPlan.fit(bytes: ResidentSize.bytes(of: m), freeHere: here, freeThere: nil) != .alone
        }
        return models.sorted { a, b in
            needsPair(a) != needsPair(b) ? needsPair(a) : a.name < b.name
        }
    }

    /// The cable comes and goes, and a Mac on the network may start leading: look again every few seconds.
    private func watch() async {
        while !Task.isCancelled {
            ports = await Task.detached { ClusterSetup.ports() }.value
            await server.refreshPairState()
            if case .ready = pair {
                leader = server.lanPeers.filter(\.value.waitsForPartner).keys.sorted().first
            } else {
                leader = nil
            }
            try? await Task.sleep(for: .seconds(5))
        }
    }

    @ViewBuilder private var pairCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            switch pair {
            case .off:
                status(L10n.text("Thunderbolt pairing is turned off on this Mac."), "bolt.slash", .secondary)
                Text("To turn it on, on both Macs: shut down, then hold the power button until Options appears and open it. Choose Utilities ▸ Terminal, type rdma_ctl enable, press Return, and restart.")
                    .font(.app(.caption2)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            case .noCable:
                status(L10n.text("Connect the other Mac to this one with a Thunderbolt 5 cable."), "cable.connector", .secondary)
            case .ready(let leader):
                if let leader, !leadInstead {
                    let model = server.lanPeers[leader]?.leads ?? ""
                    let haveIt = PairLookup.has(model, in: appState.localModels)
                    status(L10n.format("%@ is waiting to pair with this Mac.", leader), "bolt.horizontal.circle", .accentColor)
                    if !haveIt {
                        Text(L10n.format("It runs %@, which this Mac doesn't have. Download the same model here to join.", model))
                            .font(.app(.caption)).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                    }
                    HStack(spacing: 12) {
                        Button { pairing { await ThunderboltPairing.join(appState: appState, ports: ports) } } label: {
                            Text(L10n.format("Join %@", leader)).font(.app(.body))
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!haveIt)
                        Button { leadInstead = true } label: { Text("Lead from this Mac instead").font(.app(.caption)) }
                            .buttonStyle(.link)
                    }
                } else {
                    status(L10n.text("A Mac is connected to this one by Thunderbolt."), "bolt.horizontal.circle", .accentColor)
                    leadControls
                }
                Text("The first time, macOS asks for your password to set up the cable. It stays set up after restarts.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            case .reconnect(let role):
                status(L10n.text("Paired, but the cable needs setting up again after a restart."), "arrow.triangle.2.circlepath", .orange)
                Button { pairing { await ThunderboltPairing.reconnect(role, appState: appState, ports: ports) } } label: {
                    Text("Reconnect").font(.app(.body))
                }
                .buttonStyle(.borderedProminent)
            case .leading(let partner):
                if let partner, let c = server.cluster, c.workerModel == "missing" || c.workerModel == "different" {
                    status(c.workerModel == "missing"
                           ? L10n.format("%@ doesn't have %@. Download the same model there, then join again.", partner, c.model)
                           : L10n.format("%@ has another copy of %@ (a different quantization or revision). Download this one there, then join again.", partner, c.model),
                           "exclamationmark.triangle", .orange)
                } else if let partner {
                    status(L10n.format("Paired with %@ for %@.", partner, server.cluster?.model ?? ""), "bolt.horizontal.fill", .green)
                    if let c = server.cluster, c.mode == "rows" {
                        VideoClusterAdviceRow(cluster: c)
                    } else if let c = server.cluster, c.planned, c.layers > 0 {
                        ClusterSplitBar(cluster: c)
                    }
                } else if let other = server.lanPeers.filter(\.value.waitsForPartner).keys.sorted().first {
                    // Both Macs clicked Pair: one of them has to join the other.
                    status(L10n.format("%@ is also waiting for a Mac to join it.", other), "exclamationmark.triangle", .orange)
                    Button { pairing { await ThunderboltPairing.join(appState: appState, ports: ports) } } label: {
                        Text(L10n.format("Join %@", other)).font(.app(.body))
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    status(L10n.text("Waiting for the other Mac. On it, open Settings ▸ Cluster and click Join."), "hourglass", .secondary)
                }
                unpairButton
            case .helping:
                HelperStatus(worker: appState.clusterWorker)
                unpairButton
            }
            if settingUp {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text("Setting up the cable…").font(.app(.caption)).foregroundStyle(.secondary)
                }
            } else if let setupError {
                Text(verbatim: setupError).font(.app(.caption)).foregroundStyle(.red)
            }
        }
        .disabled(settingUp)
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }

    /// Which model the pair runs, and the button that makes this Mac lead it.
    @ViewBuilder private var leadControls: some View {
        let models = pairable
        if models.isEmpty {
            Text("A pair runs MiniMax-H3 (video), or Qwen3.8-Flash-Next or DeepSeek-V4.1 (chat). Download one on both Macs to use it.")
                .font(.app(.caption)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        } else {
            HStack(spacing: 8) {
                Text("Use the pair for").font(.app(.caption))
                Picker("", selection: $pairModel) {
                    ForEach(models, id: \.path) { m in
                        Text(verbatim: pairLabel(m)).tag(m.path)
                    }
                }
                .labelsHidden().frame(maxWidth: 320)
            }
            .onAppear { if !models.contains(where: { $0.path == pairModel }) { pairModel = models[0].path } }
            if let m = models.first(where: { $0.path == pairModel }) { pairEstimate(m) }
            HStack(spacing: 12) {
                Button { pairing { await ThunderboltPairing.lead(model: pairModel, appState: appState, ports: ports) } } label: {
                    Text("Pair").font(.app(.body))
                }
                .buttonStyle(.borderedProminent)
                .disabled(pairModel.isEmpty)
                Button { pairing { await ThunderboltPairing.join(appState: appState, ports: ports) } } label: {
                    Text("Join the Mac that leads instead").font(.app(.caption))
                }
                .buttonStyle(.link)
            }
            Text("This Mac leads and restarts its server; then click Join on the other Mac. Both need the same model.")
                .font(.app(.caption2)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func pairLabel(_ m: LocalModel) -> String {
        let use: String = m.modelType == "minimax_h3" ? L10n.text("video") : L10n.text("chat")
        return "\(m.name) · \(use) · \(m.sizeFormatted)"
    }

    /// Whether the model needs the pair at all, and about how much memory each Mac gives it.
    @ViewBuilder private func pairEstimate(_ m: LocalModel) -> some View {
        let video = m.modelType == "minimax_h3"
        let here = Int64(SystemMetrics.availableBytes())
        let other = PairLookup.otherMac(server.lanPeers, workerName: server.cluster?.workerName ?? "")
        let there = other.map { $0.peer.freeBytes }
        let bytes = ResidentSize.bytes(of: m)
        let share = PairPlan.share(bytes: bytes, video: video, mode: options.clusterMode, freeHere: here, freeThere: there ?? here)
        let otherName = other?.name ?? L10n.text("the other Mac")
        VStack(alignment: .leading, spacing: 2) {
            if !video {
                switch PairPlan.fit(bytes: bytes, freeHere: here, freeThere: there) {
                case .alone:
                    Text("Fits on this Mac alone. A pair runs it a little slower; pair it only to free memory here.")
                case .needsBoth:
                    Text("Too big for this Mac alone: the pair runs it across both Macs.")
                case .tooBig:
                    Text("Too big even for both Macs together.").foregroundStyle(.orange)
                }
            }
            Text(L10n.format("About %@ on this Mac and %@ on %@.",
                             MemoryInfo.format(share.here), MemoryInfo.format(share.there), otherName))
        }
        .font(.app(.caption2)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
    }

    private var unpairButton: some View {
        Button { ThunderboltPairing.unpair(appState: appState); leadInstead = false } label: { Text("Unpair").font(.app(.body)) }
    }

    private func status(_ text: String, _ symbol: String, _ color: Color) -> some View {
        Label { Text(verbatim: text).font(.app(.body)).fixedSize(horizontal: false, vertical: true) } icon: {
            Image(systemName: symbol).foregroundStyle(color)
        }
    }

    @ViewBuilder private var advancedRows: some View {
        if let m = meta["clusterRole"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.clusterRole)) {
                Picker("", selection: $appState.serverOptions.clusterRole) {
                    Text("Off").tag(ClusterRole.off)
                    Text("Host").tag(ClusterRole.host)
                    Text("Worker").tag(ClusterRole.worker)
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 220)
            }
            .task { ports = await Task.detached { ClusterSetup.ports() }.value }
        }
        if options.clusterRole != .off {
            linkRow
            if let m = meta["clusterDevice"], ports.count > 1 {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.clusterDevice)) {
                    Picker("", selection: $appState.serverOptions.clusterDevice) {
                        Text("Automatic").tag("")
                        ForEach(ports, id: \.device) { p in
                            Text(verbatim: "\(p.interface)\(p.active ? "" : " (no cable)")").tag(p.device)
                        }
                    }
                    .labelsHidden().frame(width: 200)
                }
            }
        }
        if options.clusterRole == .host { hostRows }
        if options.clusterRole == .worker { workerRows }
    }

    /// The cable and its address: RDMA carries nothing until the port has one.
    @ViewBuilder private var linkRow: some View {
        SettingsRow(title: L10n.text("Link"), explainer: L10n.text("A Thunderbolt 5 cable straight between the two Macs, with RDMA turned on.")) {
            if let p = port, p.active {
                if let ip = ClusterSetup.ipv4(on: p.interface) {
                    Label(L10n.format("%@ connected · %@", p.interface, ip), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.app(.caption))
                } else {
                    Button(L10n.format("Set up %@…", p.interface)) {
                        setupError = ThunderboltAddressJob.apply(options.clusterRole)
                        ports = ClusterSetup.ports()
                    }
                    .help(L10n.text("Gives the port its cluster address; the first time macOS asks for your password, and it stays set up after restarts."))
                }
            } else if ports.isEmpty {
                Text("RDMA is off: run `rdma_ctl enable` in Recovery").font(.app(.caption)).foregroundStyle(.orange)
            } else {
                Text("No cable on a Thunderbolt 5 port").font(.app(.caption)).foregroundStyle(.orange)
            }
        }
        if let setupError {
            Text(verbatim: setupError).font(.app(.caption)).foregroundStyle(.red)
        }
    }

    @ViewBuilder private var hostRows: some View {
        if let m = meta["clusterModelPath"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.clusterModelPath)) {
                Picker("", selection: $appState.serverOptions.clusterModelPath) {
                    Text("Choose…").tag("")
                    ForEach(appState.localModels.filter { ClusterSetup.servableTypes.contains($0.modelType) }, id: \.path) { m in
                        Text(verbatim: m.name).tag(m.path)
                    }
                }
                .labelsHidden().frame(width: 260)
            }
        }
        if videoModel {
            SettingsRow(title: L10n.text("How it splits"),
                        explainer: L10n.text("Both Macs keep the whole video pack loaded, so the worker needs the same pack and the room for it. Every step's frames are split between the two over the cable, by each Mac's measured speed.")) {
                Text("Video steps").font(.app(.caption)).foregroundStyle(.secondary)
            }
        } else if let m = meta["clusterMode"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.clusterMode)) {
                Picker("", selection: $appState.serverOptions.clusterMode) {
                    Text("By layers").tag(ClusterMode.pipeline)
                    Text("By experts").tag(ClusterMode.tensor)
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 220)
            }
        }
        SettingsRow(title: L10n.text("Status"), explainer: L10n.format("The worker joins at %@.", ClusterSetup.coordinator(options, ports: { ports }))) {
            Text(verbatim: clusterStatus).font(.app(.caption)).foregroundStyle(.secondary)
        }
    }

    private var videoModel: Bool {
        appState.localModels.first { $0.path == options.clusterModelPath }?.modelType == "minimax_h3"
    }

    private var clusterStatus: String {
        guard let c = server.cluster else {
            return options.clusterModelPath.isEmpty ? L10n.text("Choose a model") : L10n.text("Starts with the server")
        }
        if !c.joined { return L10n.text("Waiting for the worker") }
        let worker = c.workerName.isEmpty ? L10n.text("The worker") : ClusterVideoAdvice.shortName(c.workerName)
        if videoModel {
            if !c.planned { return L10n.format("%@ joined; it loads %@ with the first video", worker, c.model) }
            return c.workerReady ? L10n.format("%@ holds %@ for video steps", worker, c.model) : L10n.format("%@ is loading %@", worker, c.model)
        }
        if !c.planned { return L10n.format("Worker joined; it loads its part with %@", c.model) }
        return c.workerReady ? L10n.format("Serving %@ on both Macs", c.model) : L10n.text("The worker is loading its part")
    }

    @ViewBuilder private var workerRows: some View {
        if let m = meta["clusterCoordinator"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.clusterCoordinator)) {
                TextField("", text: $appState.serverOptions.clusterCoordinator,
                          prompt: Text(verbatim: ClusterSetup.coordinator(options, ports: { ports })))
                    .textFieldStyle(.roundedBorder).multilineTextAlignment(.trailing)
                    .frame(width: 200).font(.app(.body))
            }
        }
        WorkerStatusRow(worker: appState.clusterWorker)
    }
}

private struct WorkerStatusRow: View {
    @ObservedObject var worker: ClusterWorker

    var body: some View {
        SettingsRow(title: L10n.text("Status"), explainer: L10n.text("This Mac runs its part of the host's model; it needs the same model in its library.")) {
            switch worker.state {
            case .stopped: Text("Stopped").font(.app(.caption)).foregroundStyle(.secondary)
            case .waiting: Text("Waiting for the host").font(.app(.caption)).foregroundStyle(.secondary)
            case .loading(let what): Text(verbatim: L10n.format("Loading %@", what)).font(.app(.caption))
            case .ready(let what): Label(L10n.format("Serving %@", what), systemImage: "checkmark.circle.fill")
                .font(.app(.caption)).foregroundStyle(.green)
            case .failed(let why): Text(verbatim: why).font(.app(.caption)).foregroundStyle(.red)
            }
        }
    }
}

/// The Thunderbolt pair card's line while this Mac helps the one that leads.
private struct HelperStatus: View {
    @ObservedObject var worker: ClusterWorker

    var body: some View {
        switch worker.state {
        case .stopped, .waiting:
            Label(L10n.text("Joined. Waiting for the Mac that leads to start its job."), systemImage: "hourglass")
                .font(.app(.body))
        case .loading(let what):
            Label(L10n.format("Loading %@ to help the Mac that leads", what), systemImage: "arrow.down.circle")
                .font(.app(.body))
        case .ready(let what):
            Label(L10n.format("Helping the Mac that leads with %@", what), systemImage: "bolt.horizontal.fill")
                .font(.app(.body)).foregroundStyle(.green)
        case .failed(let why):
            Label(why, systemImage: "exclamationmark.triangle").font(.app(.body)).foregroundStyle(.red)
        }
    }
}

/// The top of Settings ▸ Cluster: this Mac, the Mac paired with it by Thunderbolt, and the
/// machines on the network, each drawn with the icon macOS uses for it.
private struct ClusterMapView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @ObservedObject var worker: ClusterWorker

    var body: some View {
        SearchableRow(searchText: ["Cluster", "machines", "Thunderbolt", "network"], separatorGap: nil) {
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .iconMiddle, spacing: 20) {
                    MachineTile(icon: MachineKind.thisMac.icon, name: thisMac, caption: L10n.text("This Mac"))
                    if let pair {
                        ThunderboltLink(model: pair.model)
                        MachineTile(icon: pair.kind.icon, name: pair.name, caption: L10n.text("Paired"))
                    }
                }
                if let together {
                    Text(verbatim: together)
                        .font(.app(.caption)).foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Divider()
                Text("On your network").font(.app(.caption).weight(.semibold)).foregroundStyle(.secondary)
                let peers = networkMachines
                if peers.isEmpty {
                    Text(appState.serverOptions.lanDiscoverEnabled
                         ? L10n.text("No other machines found yet. They show up here when they share models.")
                         : L10n.text("Turn on finding network models below to see your other machines."))
                        .font(.app(.caption)).foregroundStyle(.secondary)
                } else {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(alignment: .top, spacing: 20) {
                            ForEach(peers, id: \.self) { name in
                                let kind = server.lanPeers[name]?.kind ?? .unknown
                                MachineTile(icon: kind.icon, name: name, caption: kind.label)
                            }
                        }
                    }
                }
            }
            .padding(.vertical, 6)
        }
    }

    private var thisMac: String {
        let n = appState.serverOptions.lanName
        return n.isEmpty ? (Host.current().localizedName ?? L10n.text("This Mac")) : n
    }

    /// The Mac on the other end of the cable once paired, and the model the two run.
    private var pair: (name: String, kind: MachineKind, model: String)? {
        switch appState.serverOptions.clusterRole {
        case .host:
            guard let c = server.cluster, c.joined else { return nil }
            let model = (c.model as NSString).lastPathComponent
            // The worker names itself when the pair's model first loads; until then, the Mac
            // the network reaches over the cable.
            if let other = PairLookup.otherMac(server.lanPeers, workerName: c.workerName) {
                return (other.name, other.peer.kind, model)
            }
            let name = c.workerName.isEmpty ? L10n.text("The other Mac") : ClusterVideoAdvice.shortName(c.workerName)
            return (name, .mac, model)
        case .worker:
            guard case .ready = worker.state else { return nil }
            // The leader reports its pair's model and that a Mac joined it.
            guard let (name, peer) = server.lanPeers.first(where: { !$0.value.leads.isEmpty && $0.value.joined })
                .map({ ($0.key, $0.value) }) else { return (L10n.text("The Mac that leads"), .mac, "") }
            return (name, peer.kind, peer.leads)
        case .off:
            return nil
        }
    }

    /// What the pair does for one video.
    private var together: String? {
        guard let c = server.cluster, c.joined, c.mode == "rows" else { return nil }
        return ClusterVideoAdvice.of(c).text(worker: c.workerName)
    }

    /// Every machine on the network but the paired one.
    private var networkMachines: [String] {
        var names = Set(server.lanPeers.keys)
        names.formUnion(server.allModels.compactMap(\.lanPeer))
        if let pair { names = names.filter { !StoryboardCluster.sameMachine($0, pair.name) } }
        return names.sorted()
    }
}

/// One machine on the Cluster map: its macOS icon, name and kind.
private struct MachineTile: View {
    let icon: NSImage
    let name: String
    let caption: String

    var body: some View {
        VStack(spacing: 2) {
            Image(nsImage: icon).resizable().interpolation(.high).frame(width: 64, height: 64)
                .alignmentGuide(.iconMiddle) { $0[VerticalAlignment.center] }
            Text(verbatim: name).font(.app(.body)).lineLimit(1)
            Text(verbatim: caption).font(.app(.caption2)).foregroundStyle(.secondary)
        }
        .frame(width: 112)
    }
}

/// The cable between the two paired Macs, with the model they run together.
private struct ThunderboltLink: View {
    let model: String

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 6) {
                Rectangle().fill(Color.accentColor.opacity(0.6)).frame(width: 36, height: 2)
                Image(systemName: "bolt.fill").font(.system(size: 18, weight: .semibold)).foregroundStyle(Color.accentColor)
                Rectangle().fill(Color.accentColor.opacity(0.6)).frame(width: 36, height: 2)
            }
            .alignmentGuide(.iconMiddle) { $0[VerticalAlignment.center] }
            Text("Thunderbolt").font(.app(.caption2)).foregroundStyle(.secondary)
            if !model.isEmpty {
                Text(verbatim: model).font(.app(.caption)).lineLimit(1).truncationMode(.middle).frame(maxWidth: 180)
            }
        }
    }
}

extension VerticalAlignment {
    /// The middle of the machine icons on the Cluster map, where the Thunderbolt line runs.
    private enum IconMiddle: AlignmentID {
        static func defaultValue(in d: ViewDimensions) -> CGFloat { d[VerticalAlignment.center] }
    }
    fileprivate static let iconMiddle = VerticalAlignment(IconMiddle.self)
}

/// Video steps: how fast the worker is against this Mac, what that means, and what the last
/// video did, with its row split as two bars.
private struct VideoClusterAdviceRow: View {
    let cluster: ClusterInfo

    var body: some View {
        let advice = ClusterVideoAdvice.of(cluster)
        VStack(alignment: .leading, spacing: 4) {
            Label {
                Text(verbatim: advice.text(worker: cluster.workerName)).font(.app(.caption))
            } icon: {
                Image(systemName: { if case .tooSlow = advice { return "tortoise" } else { return "bolt.horizontal" } }())
                    .foregroundStyle({ if case .tooSlow = advice { return Color.orange } else { return Color.accentColor } }())
            }
            if cluster.linkGbps > 0 {
                Text(verbatim: L10n.format("Link: %@ GB/s each way · %@ free on the worker",
                                           String(format: "%.1f", cluster.linkGbps),
                                           ByteCountFormatter.string(fromByteCount: Int64(cluster.workerFreeMib) << 20, countStyle: .memory)))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
            if cluster.lastSplit {
                let here = Double(cluster.splitRows[0]) / Double(max(1, cluster.splitRows[0] + cluster.splitRows[1]))
                GeometryReader { g in
                    HStack(spacing: 2) {
                        RoundedRectangle(cornerRadius: 3).fill(Color.accentColor).frame(width: max(4, g.size.width * here - 1))
                        RoundedRectangle(cornerRadius: 3).fill(Color.purple).frame(width: max(4, g.size.width * (1 - here) - 1))
                    }
                }
                .frame(height: 10)
                Text(verbatim: L10n.format("Last video: %lld%% of every step here, %lld%% on the worker",
                                           Int64((here * 100).rounded()), Int64(((1 - here) * 100).rounded())))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            } else if cluster.lastJob.hasPrefix("alone: ") {
                Text(verbatim: L10n.format("Last video ran here alone: %@", String(cluster.lastJob.dropFirst(7))))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

/// Which layers each Mac holds, as two bars in proportion.
private struct ClusterSplitBar: View {
    let cluster: ClusterInfo

    var body: some View {
        let here = cluster.mode == "tensor" ? 1.0 : Double(cluster.here) / Double(max(1, cluster.layers))
        VStack(alignment: .leading, spacing: 4) {
            GeometryReader { g in
                HStack(spacing: 2) {
                    RoundedRectangle(cornerRadius: 3).fill(Color.accentColor).frame(width: max(4, g.size.width * here - 1))
                    RoundedRectangle(cornerRadius: 3).fill(Color.purple).frame(width: max(4, g.size.width * (1 - here) - 1))
                }
            }
            .frame(height: 10)
            Text(verbatim: cluster.mode == "tensor"
                 ? L10n.text("Every layer on both Macs, each with half of every expert")
                 : L10n.format("Layers 1–%lld here, %lld–%lld on the worker", Int64(cluster.here), Int64(cluster.here + 1), Int64(cluster.layers)))
                .font(.app(.caption2)).foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

/// A key field per LAN peer that runs with --api-key. Saving writes
/// lan-keys.json, which the server picks up on its own.
private struct LanPeerKeyRows: View {
    @EnvironmentObject var server: ServerManager
    @Binding var keys: [String: String]
    @State private var saveError: String?

    var body: some View {
        let names = LanPeerKeys.rows(needingKey: server.lanPeersNeedingKey, saved: keys)
        if !names.isEmpty {
            SearchableRow(searchText: ["API key", "LAN peer"] + names) {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(names, id: \.self) { name in
                        HStack(spacing: 8) {
                            Text(verbatim: name).font(.app(.body))
                            if server.lanPeersNeedingKey.contains(name) {
                                Text(L10n.text("Needs key")).font(.app(.caption)).foregroundStyle(.orange)
                            }
                            Spacer()
                            SecureField("api key", text: Binding(
                                get: { keys[name] ?? "" },
                                set: { keys[name] = $0 }
                            ), prompt: Text(L10n.text("API key")))
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 200)
                            .onSubmit(save)
                        }
                    }
                    HStack {
                        Spacer()
                        Button { save() } label: { Text("Save").font(.app(.body)) }
                    }
                    if let saveError {
                        Text(verbatim: saveError).font(.app(.caption)).foregroundStyle(.red)
                    }
                }
            }
        }
    }

    private func save() {
        do {
            try LanPeerKeys.save(keys)
            saveError = nil
        } catch {
            saveError = error.localizedDescription
            return
        }
        Task {
            // The server retries the peer within a second of the file changing.
            try? await Task.sleep(for: .seconds(3))
            await server.refreshModels()
        }
    }
}

// MARK: - Providers section

/// Rows of `~/.mlx-serve/providers.json`. Every edit saves the file and asks
/// the running server to reload; the health dot is the server's own probe
/// (`GET /v1/providers`), never a guess made here.
private struct ProvidersSectionContent: View {
    @EnvironmentObject var server: ServerManager
    @EnvironmentObject var formState: SettingsFormState
    @State private var status: [ProviderStatus] = []
    @State private var saveError: String?

    var body: some View {
        // One searchable row: a section whose content publishes no row count
        // never collapses under the filter.
        SearchableRow(searchText: ["Providers", "OpenAI-compatible chat endpoints", "cloud API", "providers.json", "API key"]
                      + formState.providerEntries.map(\.name)) {
            providersBody
        }
    }

    private var providersBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            if formState.providerEntries.isEmpty {
                Text(L10n.text("No providers yet."))
                    .font(.app(.caption))
                    .foregroundStyle(.tertiary)
            }
            ForEach($formState.providerEntries) { $entry in
                ProviderRow(entry: $entry,
                            serverPort: server.port,
                            status: status.first { $0.name == entry.name },
                            duplicate: ProvidersFile.duplicateNames(formState.providerEntries).contains(entry.name),
                            onDelete: { [id = entry.id] in
                                formState.removeProvider(id: id)
                                save()
                            },
                            onCommit: save)
            }
            HStack {
                Button { formState.providerEntries.append(ProviderEntry()) } label: { Label("Add Provider", systemImage: "plus") }
                Spacer()
                // Fields also save on Enter, but an edit followed by a click
                // elsewhere never submits — this is the button that always writes.
                Button { save() } label: { Text("Save")
                    .font(.app(.body)) }
                .keyboardShortcut("s", modifiers: .command)
                .help("Write providers.json and ask the server to re-probe now")
            }
            if let saveError {
                Text(L10n.text(saveError)).font(.app(.caption)).foregroundStyle(.red)
            }
            Text(L10n.text("Keys are stored in plain text in ~/.mlx-serve/providers.json. Prefer an environment variable name for a shared machine. Provider models are never shared over the LAN."))
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .task { await refreshStatus() }
        .onChange(of: server.status) { _, s in
            if s == .running { Task { await refreshStatus() } }
        }
    }

    private func save() {
        do {
            try ProvidersFile.save(formState.providerEntries)
            saveError = nil
        } catch {
            saveError = "Could not save providers.json: \(error.localizedDescription)"
            return
        }
        Task {
            await server.reloadProviders()
            // The probe runs on the server's own thread right after reload.
            try? await Task.sleep(for: .seconds(2))
            await refreshStatus()
        }
    }

    private func refreshStatus() async {
        guard server.status == .running else { return }
        status = await server.providerStatus()
    }
}

private struct ProviderRow: View {
    @Binding var entry: ProviderEntry
    @EnvironmentObject var formState: SettingsFormState
    let serverPort: UInt16
    let status: ProviderStatus?
    let duplicate: Bool
    let onDelete: () -> Void
    let onCommit: () -> Void
    @State private var picking = false

    /// The model-id field's own text, held on the store: the entry keeps the
    /// parsed ids, so a rebuilt row would otherwise show the parsed list back.
    private var modelsText: Binding<String> {
        Binding(
            get: { formState.providerModelText[entry.id] ?? entry.models.joined(separator: ", ") },
            set: { formState.providerModelText[entry.id] = $0 }
        )
    }

    private var problem: String? {
        if duplicate { return "Another provider already uses this name" }
        return entry.problem(serverPort: serverPort)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                healthDot
                TextField("name", text: $entry.name, prompt: Text("name"))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .onSubmit(onCommit)
                TextField("url", text: $entry.url, prompt: Text("https://api.openai.com/v1"))
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(onCommit)
                Toggle("", isOn: $entry.enabled)
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .onChange(of: entry.enabled) { _, _ in onCommit() }.font(.app(.body))
                Button(role: .destructive, action: onDelete) { Image(systemName: "trash") }
                    .buttonStyle(.borderless)
                    .help("Remove this provider")
            }
            HStack(spacing: 8) {
                SecureField("api key", text: $entry.apiKey, prompt: Text(L10n.text("API key")))
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(onCommit)
                TextField("env", text: $entry.apiKeyEnv, prompt: Text(L10n.text("or env var, e.g. OPENAI_API_KEY")))
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(onCommit)
            }
            HStack(spacing: 8) {
                TextField("models", text: modelsText, prompt: Text(L10n.text("Models, comma-separated — only these are exposed; empty = all the provider lists")))
                    .textFieldStyle(.roundedBorder)
                    .font(.app(.caption))
                    .onChange(of: formState.providerModelText[entry.id]) { _, t in
                        entry.models = ProviderEntry.parseModelList(t ?? "")
                    }
                    .onSubmit(onCommit)
                Button { picking = true } label: { Text(L10n.text("Pick…"))
                    .font(.app(.body)) }
                    .disabled(entry.problem() != nil)
                    .help(L10n.text("Fetch the provider's model list and tick the ones to expose"))
            }
            .sheet(isPresented: $picking) {
                ProviderModelPickerSheet(entry: entry) { chosen in
                    formState.providerModelText[entry.id] = chosen.joined(separator: ", ")
                    entry.models = chosen
                    onCommit()
                }
            }
            if let problem {
                Text(L10n.text(problem)).font(.app(.caption2)).foregroundStyle(.orange)
            } else if let status {
                Text(L10n.text(statusLine(status))).font(.app(.caption2)).foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.08)))
    }

    private var healthDot: some View {
        Circle()
            .fill(status.map { $0.up ? Color.green : ($0.probed ? Color.red : Color.gray) } ?? Color.gray)
            .frame(width: 8, height: 8)
            .help(status.map(statusLine) ?? "Not reported by the server yet")
    }

    private func statusLine(_ s: ProviderStatus) -> String {
        if !s.probed { return "Checking…" }
        return s.up ? "Up — \(s.models) model\(s.models == 1 ? "" : "s")" : "Unreachable"
    }
}

/// Fetches `<url>/models` with the row's key and lets the user tick the ids
/// to expose. Done writes the list back as the comma-separated field.
private struct ProviderModelPickerSheet: View {
    let entry: ProviderEntry
    let onDone: ([String]) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var ids: [String] = []
    @State private var chosen: Set<String> = []
    @State private var filter = ""
    @State private var error: String?
    @State private var loading = true

    private var shown: [String] {
        filter.isEmpty ? ids : ids.filter { $0.localizedCaseInsensitiveContains(filter) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L10n.format("Models at %@", entry.name)).font(.app(.headline))
            TextField("Filter", text: $filter).textFieldStyle(.roundedBorder)
            if loading {
                ProgressView().frame(maxWidth: .infinity)
            } else if let error {
                Text(error).foregroundStyle(.red).font(.app(.caption))
            } else {
                List(shown, id: \.self) { id in
                    Toggle(id, isOn: Binding(
                        get: { chosen.contains(id) },
                        set: { on in if on { chosen.insert(id) } else { chosen.remove(id) } }
                    ))
                }
                .listStyle(.plain)
            }
            HStack {
                Text("\(chosen.count) of \(ids.count) selected").font(.app(.caption)).foregroundStyle(.secondary)
                Button { chosen = [] } label: { Text("Clear")
                    .font(.app(.body)) }.disabled(chosen.isEmpty)
                Spacer()
                Button { dismiss() } label: { Text("Cancel")
                    .font(.app(.body)) }.keyboardShortcut(.cancelAction)
                Button {
                    onDone(ids.filter { chosen.contains($0) })
                    dismiss()
                } label: { Text("Done")
                    .font(.app(.body)) }
                .keyboardShortcut(.defaultAction)
                .disabled(loading)
            }
        }
        .padding()
        .frame(width: 460, height: 520)
        .task { await load() }
    }

    private func load() async {
        chosen = Set(entry.models)
        var key = entry.apiKey
        if !entry.apiKeyEnv.isEmpty {
            // `LoginShellEnv.values(of:)` spawns the user's login shell — blocking,
            // and documented as off-main only. This runs from `.task`, which
            // inherits the main actor, so it has to hop off before asking.
            let name = entry.apiKeyEnv
            let shell = await Task.detached(priority: .userInitiated) { LoginShellEnv.values(of: [name]) }.value
            if let v = shell[name], !v.isEmpty { key = v }
        }
        var lastError = "No model list at \(entry.url)"
        for url in ProviderEntry.modelsURLs(for: entry.url) {
            var req = URLRequest(url: url, timeoutInterval: 15)
            if !key.isEmpty { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
            do {
                let (data, resp) = try await URLSession.shared.data(for: req)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
                if code == 200, let list = ProviderEntry.modelIds(fromModelsBody: data) {
                    ids = list.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
                    loading = false
                    return
                }
                lastError = "HTTP \(code) from \(url.absoluteString)"
            } catch {
                lastError = error.localizedDescription
            }
        }
        self.error = lastError
        loading = false
    }
}

private struct ServerSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    /// Chat models only — the `isChatPickable` filter every other model picker uses.
    private var pickable: [LocalModel] {
        appState.localModels.filter(\.isChatPickable)
    }

    /// The tray picker's labels: a `.menu` Picker keys its checkmark by title, so
    /// duplicate labels get an engine suffix.
    private func startupModelLabel(_ model: LocalModel, dupNames: Set<String>) -> String {
        let label = model.displayLabel
        guard dupNames.contains(label) else { return label }
        return "\(label) · \(model.engine.shortLabel)"
    }

    /// Under "Last model used" the dropdown shows the rule's answer, read from the
    /// same `StartupModelChoice.resolved` as the launch gate. The setter only runs
    /// in `.pinned` mode; the control is disabled otherwise.
    private var startupModelDisplay: Binding<String> {
        Binding(
            get: {
                guard appState.startupModelMode == .lastUsed else {
                    return appState.startupModelPinnedPath
                }
                return StartupModelChoice.resolved(mode: .lastUsed,
                                                   pinnedPath: nil,
                                                   lastUsed: StartupModelChoice.lastUsed(),
                                                   installedPaths: pickable.map(\.path)) ?? ""
            },
            set: { appState.startupModelPinnedPath = $0 }
        )
    }

    var body: some View {
        SettingsRow(
            title: "Preload the model when the server starts",
            explainer: "Load your model when the server starts instead of at your first message."
        ) {
            Toggle("", isOn: $appState.loadModelAtStart)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }
        SettingsRow(
            title: "Which model",
            explainer: "Load the model you used last, or always the same one."
        ) {
            Picker("", selection: $appState.startupModelMode) {
                ForEach(StartupModelChoice.Mode.allCases) { mode in
                    Text(L10n.text(mode.label)).tag(mode)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .disabled(!appState.loadModelAtStart).font(.app(.body))
        }
        SettingsRow(
            title: "Model",
            explainer: "If this model is removed, the server starts with no model loaded."
        ) {
            VStack(alignment: .trailing, spacing: 4) {
                Picker("", selection: startupModelDisplay) {
                    let dupNames = LocalModel.duplicateNames(in: pickable)
                    ForEach(pickable) { model in
                        Text(L10n.text(startupModelLabel(model, dupNames: dupNames))).font(.app(.body)).tag(model.path)
                    }
                    // A selection that matches no row renders blank, so empty and
                    // uninstalled selections each get a row of their own.
                    if startupModelDisplay.wrappedValue.isEmpty {
                        Text(L10n.text("None — starts with no model")).font(.app(.body)).tag("")
                    }
                    if !appState.startupModelPinnedPath.isEmpty,
                       !pickable.contains(where: { $0.path == appState.startupModelPinnedPath }) {
                        Text("\((appState.startupModelPinnedPath as NSString).lastPathComponent) — no longer installed").font(.app(.body))
                            .tag(appState.startupModelPinnedPath)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 180)
                .disabled(!appState.loadModelAtStart || appState.startupModelMode == .lastUsed)

                if appState.startupModelMode == .lastUsed {
                    Text(L10n.text("Resolved each time the app starts, so it keeps up as you switch models."))
                        .font(.app(.caption2))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.trailing)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
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
                .font(.app(.body).monospacedDigit())
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
                    .toggleStyle(.switch).font(.app(.body))
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
                    .toggleStyle(.switch).font(.app(.body))
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
                .frame(width: 180).font(.app(.body))
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
                    .toggleStyle(.switch).font(.app(.body))
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
                        Text(L10n.text(lvl.label)).font(.app(.body)).tag(lvl)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 180).font(.app(.body))
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
                    .toggleStyle(.switch).font(.app(.body))
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
    @EnvironmentObject var formState: SettingsFormState

    /// The field's text lives on the store so what is being typed survives a
    /// query edit; nil means the field has not been shown yet.
    private var text: Binding<String> {
        Binding(get: { formState.portText ?? "" },
                set: { formState.portText = $0 })
    }

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
                TextField("", text: text, prompt: Text("11234"))
                    .textFieldStyle(.roundedBorder)
                    .font(.app(.body).monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 90)
                    .onAppear {
                        if formState.portText == nil { formState.portText = "\(appState.serverOptions.port)" }
                    }
                    .onChange(of: formState.portText) { _, newValue in
                        if let p = ServerOptions.parsePort(newValue ?? "") {
                            appState.serverOptions.port = p
                        }
                    }
                    .onChange(of: appState.serverOptions.port) { _, newPort in
                        if ServerOptions.parsePort(formState.portText ?? "") != newPort {
                            formState.portText = "\(newPort)"
                        }
                    }
                    .onSubmit { formState.portText = "\(appState.serverOptions.port)" }
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
    private static let allPresets = ContextSizeDisplay.presets

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
                        Text(L10n.text("Context size"))
                            .font(.app(.body))
                        if isDirty {
                            Image(systemName: "arrow.clockwise.circle.fill")
                                .font(.app(.caption))
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
                            .font(.app(.body).monospacedDigit())
                            .frame(minWidth: 56, alignment: .trailing)
                    }
                }
                Text(L10n.text(ContextSizeDisplay.helpText))
                    .font(.app(.caption2))
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
            Text(L10n.text(label))
                .font(.app(.caption2))
                .foregroundStyle(labelColor)
            Text(value)
                .font(.app(.caption2).monospacedDigit().weight(.medium))
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

    var body: some View {
        let opts = $appState.serverOptions
        // Drafter and PLD are mutually exclusive at the request level
        // (`drafter > PLD > regular` in src/server.zig). When drafter is on
        // we lock the PLD toggles down so users can't accidentally enable a
        // setting that would never apply.
        let drafterActive = server.chatModelInfo?.drafterLoaded ?? false
        let pldUsable = appState.serverOptions.enablePLD && !drafterActive

        Text(L10n.text("Drafters are chosen per model: Model Settings, Speculation."))
            .font(.app(.caption2))
            .foregroundStyle(.secondary)
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
                    .disabled(drafterActive).font(.app(.body))
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
                        .font(.app(.body).monospacedDigit())
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
                        .font(.app(.body).monospacedDigit())
                }
                .disabled(!pldUsable)
            }
        }

        // Native multi-token prediction — the model's OWN trained head, so it's
        // a different mechanism from PLD (which guesses by copying from the
        // prompt) and from the drafter (a separate small model). It needs no
        // extra download and no compatible pairing: a Qwen 3.5/3.8 checkpoint
        // either ships the head or it doesn't.
        SettingsSubheader("Multi-Token Prediction — Qwen 3.5 / 3.8")
        if let m = meta["enableMTP"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.enableMTP)
            ) {
                Toggle("", isOn: opts.enableMTP)
                    .labelsHidden()
                    .toggleStyle(.switch).font(.app(.body))
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
                    if let specCost = server.specCost {
                        Text(L10n.format("Automatic (measured: %lld tokens)", Int64(specCost.mtpDepthCap))).font(.app(.body)).tag(0)
                    } else {
                        Text(L10n.text("Automatic")).font(.app(.body)).tag(0)
                    }
                    ForEach(1...6, id: \.self) { n in
                        Text("\(n) token\(n == 1 ? "" : "s")").font(.app(.body)).tag(n)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 140)
                .disabled(!appState.serverOptions.enableMTP).font(.app(.body))
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
                    .toggleStyle(.switch).font(.app(.body))
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
            Text(L10n.text(text))
                .font(.app(.caption).weight(.semibold))
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
                VStack(alignment: .trailing, spacing: 4) {
                    Stepper(value: opts.maxConcurrent, in: 1...8) {
                        Text("\(appState.serverOptions.maxConcurrent)")
                            .font(.app(.body).monospacedDigit())
                    }
                    if let b = server.batching {
                        Text(L10n.text(b.label))
                            .font(.app(.caption))
                            .foregroundStyle(b.supported ? .secondary : Color.orange)
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
                .toggleStyle(.switch).font(.app(.body))
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
                    .toggleStyle(.switch).font(.app(.body))
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

// MARK: - Memory section

/// Every engine memory knob in one place. The resident-model limits, the OS
/// reserve and the preflight apply to every engine; the KV and prefix-cache rows
/// and the n-gram table are MLX-only and hide when a GGUF/DSV4 model is serving.
private struct MemorySectionContent: View {
    let showMLX: Bool
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions

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
        if let m = meta["maxResidentModels"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.maxResidentModels)
            ) {
                Stepper(value: opts.maxResidentModels, in: 1...8) {
                    Text("\(appState.serverOptions.maxResidentModels)")
                        .font(.app(.body).monospacedDigit())
                }
            }
        }
        if let m = meta["idleEvictSecs"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.idleEvictSecs)
            ) {
                let secs = appState.serverOptions.idleEvictSecs
                snappingSlider(
                    presets: ServerOptions.idleEvictPresets,
                    current: secs,
                    set: { appState.serverOptions.idleEvictSecs = $0 },
                    label: ServerOptions.idleEvictLabel(secs)
                )
            }
        }
        if let m = meta["skipMemPreflight"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.skipMemPreflight)
            ) {
                Toggle("", isOn: opts.skipMemPreflight)
                    .labelsHidden()
                    .toggleStyle(.switch).font(.app(.body))
            }
        }
        if let m = meta["osReserveGiB"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.osReserveGiB)
            ) {
                let auto = ServerOptions.autoOsReserveGiB(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
                Picker("", selection: opts.osReserveGiB) {
                    Text("Auto (\(auto.formatted(.number.precision(.fractionLength(0...1)))) GB)")
                        .font(.app(.body)).tag(Int?.none)
                    ForEach([0, 2, 3, 4, 6, 8], id: \.self) { gib in
                        Text(gib == 0 ? "Off" : "\(gib) GB").font(.app(.body)).tag(Int?.some(gib))
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(minWidth: 220).font(.app(.body))
            }
        }
        if showMLX {
            if let m = meta["kvQuant"] {
                SettingsRow(
                    title: m.title,
                    explainer: m.explainer,
                    isDirty: dirty.dirty(\.kvQuant)
                ) {
                    Picker("", selection: opts.kvQuant) {
                        ForEach(ServerOptions.KVQuant.allCases) { q in
                            Text(L10n.text(q.label)).font(.app(.body)).tag(q)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(minWidth: 220).font(.app(.body))
                }
            }
            if let m = meta["hotPrefixCacheEnabled"] {
                SettingsRow(
                    title: m.title,
                    explainer: m.explainer,
                    isDirty: dirty.dirty(\.hotPrefixCacheEnabled)
                ) {
                    Toggle("", isOn: opts.hotPrefixCacheEnabled)
                        .labelsHidden()
                        .toggleStyle(.switch).font(.app(.body))
                }
            }
            if let m = meta["prefixCacheEntries"], appState.serverOptions.hotPrefixCacheEnabled {
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
                    Stepper(value: opts.prefixCacheEntries, in: 1...16) {
                        Text("\(appState.serverOptions.prefixCacheEntries)")
                            .font(.app(.body).monospacedDigit())
                    }
                }
            }
            if let m = meta["prefixCacheMem"], appState.serverOptions.hotPrefixCacheEnabled {
                SettingsRow(
                    title: m.title,
                    explainer: m.explainer,
                    isDirty: dirty.dirty(\.prefixCacheMem)
                ) {
                    TextField("", text: opts.prefixCacheMem, prompt: Text("Auto"))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 110)
                }
            }
            if let m = meta["pleGpu"] {
                SettingsRow(
                    title: m.title,
                    explainer: m.explainer,
                    isDirty: dirty.dirty(\.pleGpu),
                    cost: m.cost,
                    costActive: appState.serverOptions.pleGpu
                ) {
                    Toggle("", isOn: opts.pleGpu)
                        .labelsHidden()
                        .toggleStyle(.switch).font(.app(.body))
                }
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
private struct NeuralEngineSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }

    var body: some View {
        let opts = $appState.serverOptions

        if let m = meta["anePrefill"] {
            SettingsRow(
                title: m.title,
                explainer: m.explainer,
                isDirty: dirty.dirty(\.anePrefill),
                cost: m.cost,
                costActive: appState.serverOptions.anePrefill
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
                        Text(L10n.text(caution))
                            .font(.app(.caption2))
                            .foregroundStyle(.orange)
                            .multilineTextAlignment(.trailing)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
        if let m = meta["w8a8"], AnePrefillAdvice.liveIsM5Family { // inert without NAX: hidden off the M5 family
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.w8a8)) {
                Toggle("", isOn: opts.w8a8).labelsHidden().toggleStyle(.switch).font(.app(.body))
            }
        }
        if let m = meta["aneImage"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.aneImage),
                        cost: m.cost, costActive: appState.serverOptions.aneImage) {
                Toggle("", isOn: opts.aneImage).labelsHidden().toggleStyle(.switch).font(.app(.body))
            }
        }
        if let m = meta["aneVideo"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.aneVideo),
                        cost: m.cost, costActive: appState.serverOptions.aneVideo) {
                Toggle("", isOn: opts.aneVideo).labelsHidden().toggleStyle(.switch).font(.app(.body))
            }
        }
        if let m = meta["aneAudio"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.aneAudio),
                        cost: m.cost, costActive: appState.serverOptions.aneAudio) {
                Toggle("", isOn: opts.aneAudio).labelsHidden().toggleStyle(.switch).font(.app(.body))
            }
        }
        if let m = meta["aneSlots"] {
            SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.aneSlots)) {
                Stepper(value: opts.aneSlots, in: 1...4) {
                    Text("\(appState.serverOptions.aneSlots)")
                        .font(.app(.body).monospacedDigit())
                }
            }
        }
    }
}

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
                        .font(.app(.body).monospacedDigit())
                }
            }
        }
    }
}

// MARK: - GGUF (llama.cpp) performance section

/// Every embedded engine's launch flags, grouped by engine.
private struct EnginesSectionContent: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.settingsSearchQuery) private var query

    private var meta: [String: ServerOptionField] { ServerOptions.serverFlagFields }
    private var dirty: ServerLaunchDirty {
        ServerLaunchDirty(current: appState.serverOptions, last: server.liveLaunchedOptions)
    }
    /// A search narrows to rows, so the engine boxes step aside while one runs.
    private var grouped: Bool { SettingsSearch.tokens(query).isEmpty }

    var body: some View {
        let opts = $appState.serverOptions
        EngineGroup(name: "mlx-serve-gguf", blurb: "GGUF files on MLX itself. Experimental.", boxed: grouped) {
            if let m = meta["mlxGguf"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.mlxGguf)) {
                    Toggle("", isOn: opts.mlxGguf)
                        .labelsHidden()
                        .toggleStyle(.switch).font(.app(.body))
                }
            }
        }
        EngineGroup(name: "llama.cpp", blurb: "Serves every other .gguf file. Its own kernels and KV layout, so the MLX rows do not apply.", boxed: grouped) {
            if let m = meta["llamaKvQuant"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.llamaKvQuant)) {
                    Picker("", selection: opts.llamaKvQuant) {
                        ForEach(ServerOptions.LlamaKVQuant.allCases) { q in
                            Text(L10n.text(q.label)).font(.app(.body)).tag(q)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(minWidth: 260).font(.app(.body))
                }
            }
            if let m = meta["llamaCacheEntries"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.llamaCacheEntries)) {
                    Stepper(value: opts.llamaCacheEntries, in: 1...8) {
                        Text("\(appState.serverOptions.llamaCacheEntries)")
                            .font(.app(.body).monospacedDigit())
                    }
                }
            }
            if let m = meta["llamaMtpDrafts"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.llamaMtpDrafts)) {
                    Stepper(value: opts.llamaMtpDrafts, in: 0...8) {
                        Text(appState.serverOptions.llamaMtpDrafts == 0 ? L10n.text("Off") : "\(appState.serverOptions.llamaMtpDrafts)")
                            .font(.app(.body).monospacedDigit())
                    }
                }
            }
            if let m = meta["llamaUbatch"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.llamaUbatch)) {
                    Picker("", selection: opts.llamaUbatch) {
                        Text(L10n.text("Default (512)")).font(.app(.body)).tag(0)
                        ForEach([1024, 2048, 4096], id: \.self) { n in
                            Text(verbatim: "\(n)").font(.app(.body)).tag(n)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(minWidth: 140).font(.app(.body))
                }
            }
        }
        EngineGroup(name: "ds4", blurb: "Serves DeepSeek-V4-Flash GGUF files.", boxed: grouped) {
            if let m = meta["ssdStreaming"] {
                SettingsRow(title: m.title, explainer: m.explainer, isDirty: dirty.dirty(\.ssdStreaming)) {
                    Toggle("", isOn: opts.ssdStreaming)
                        .labelsHidden()
                        .toggleStyle(.switch).font(.app(.body))
                }
            }
        }
    }
}

/// One engine's rows in a box under its name. Unboxed, the rows sit bare in the
/// section, so a search that filters every row out leaves no empty box behind.
private struct EngineGroup<Content: View>: View {
    let name: String
    let blurb: String
    let boxed: Bool
    @ViewBuilder var content: Content

    var body: some View {
        if boxed {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.app(.headline))
                    Text(L10n.text(blurb)).font(.app(.caption2)).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 18) { content }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
        } else {
            content
        }
    }
}


// MARK: - Per-request defaults section

/// Appearance mode, accent color, text size, compact mode and the Quick
/// Launcher shortcut — client-side display prefs, none of them a launch
/// flag, so they're `@AppStorage`-backed instead of riding `ServerOptions`.
private struct InterfaceSectionContent: View {
    @EnvironmentObject var appState: AppState
    @AppStorage(InterfacePrefKey.language) private var languageRaw = ""
    @AppStorage(InterfacePrefKey.appearanceMode) private var appearanceModeRaw = AppAppearanceMode.system.rawValue
    @AppStorage(InterfacePrefKey.accentColor) private var accentColorRaw = AppAccentColor.system.rawValue
    @AppStorage(InterfacePrefKey.textSize) private var textSizeRaw = ChatTextSize.medium.rawValue
    @AppStorage(InterfacePrefKey.chatColumn) private var chatColumnRaw = ChatColumnWidth.wide.rawValue
    @AppStorage(InterfacePrefKey.compactMode) private var compactMode = false
    @AppStorage(InterfacePrefKey.showActivityPhrases) private var showActivityPhrases = true
    @AppStorage(InterfacePrefKey.terminalTheme) private var terminalThemeId = TerminalTheme.defaultId
    @AppStorage(InterfacePrefKey.terminalBackground) private var terminalBackgroundHex = ""
    @AppStorage(InterfacePrefKey.audioPlayerStyle) private var audioPlayerStyle = AudioPlayerStyle.defaultStyle

    var body: some View {
        SettingsRow(title: "Language",
                    explainer: "\"System Default\" follows the Mac. The app relaunches to apply the change.") {
            // A catalog is resolved once per process, so the change lands on
            // the next start — the app starts that one itself.
            Picker("", selection: languageChoice) {
                ForEach(AppLanguageChoice.all) { choice in
                    Text(verbatim: choice.label).tag(choice)
                }
            }
            .labelsHidden()
            .frame(width: 160).font(.app(.body))
        }
        SettingsRow(title: "Appearance", explainer: "Follow the system, or always use light or dark.") {
            Picker("", selection: $appearanceModeRaw) {
                ForEach(AppAppearanceMode.allCases) { mode in
                    Text(L10n.text(mode.label)).tag(mode.rawValue)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 180).font(.app(.body))
        }
        SettingsRow(title: "Accent Color", explainer: "Color for buttons, links and your message bubbles.") {
            Picker("", selection: $accentColorRaw) {
                ForEach(AppAccentColor.allCases) { accent in
                    Text(L10n.text(accent.label)).font(.app(.body)).tag(accent.rawValue)
                }
            }
            .labelsHidden()
            .frame(width: 140).font(.app(.body))
        }
        SettingsRow(title: "Text Size", explainer: "Size of the text in chats.") {
            Picker("", selection: $textSizeRaw) {
                ForEach(ChatTextSize.allCases) { size in
                    Text(L10n.text(size.label)).font(.app(.body)).tag(size.rawValue)
                }
            }
            .labelsHidden()
            .frame(width: 140).font(.app(.body))
        }
        SettingsRow(title: "Chat Column",
                    explainer: "How wide conversations are, also on ⌘⌥1–3.") {
            Picker("", selection: $chatColumnRaw) {
                ForEach(ChatColumnWidth.allCases) { width in
                    Text(L10n.text(width.label)).tag(width.rawValue)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 220).font(.app(.body))
        }
        SettingsRow(title: "Compact Mode", explainer: "Less space between messages, also on ⌘⌥C.") {
            Toggle("", isOn: $compactMode)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }
        SettingsRow(title: "Show Activity Phrases",
                    explainer: "Show rotating activity text while generating. GPU, memory and elapsed time remain visible.") {
            Toggle("Show Activity Phrases", isOn: $showActivityPhrases)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }
        SettingsRow(title: "Audio Player",
                    explainer: "How the Voice, Music and Sound tabs show the player and your history.") {
            Picker("", selection: $audioPlayerStyle) {
                ForEach(AudioPlayerStyle.allCases) { style in
                    Text(verbatim: style.label).font(.app(.body)).tag(style)
                }
            }
            .labelsHidden()
            .frame(width: 140).font(.app(.body))
        }
        SettingsRow(title: "Terminal Theme",
                    explainer: "Colors for new terminals; right-click a terminal to change just that one.") {
            Picker("", selection: $terminalThemeId) {
                ForEach(TerminalTheme.all) { theme in
                    Text(theme.name).font(.app(.body)).tag(theme.id)
                }
            }
            .labelsHidden()
            .frame(width: 160).font(.app(.body))
        }
        SettingsRow(title: "Terminal Background",
                    explainer: "Background color behind the terminal theme.") {
            HStack(spacing: 8) {
                ColorPicker("", selection: terminalBackground, supportsOpacity: false)
                    .labelsHidden()
                if !terminalBackgroundHex.isEmpty {
                    Button { terminalBackgroundHex = "" } label: { Text("Reset")
                        .font(.app(.body)) }
                        .controlSize(.small)
                }
            }
        }
        SettingsRow(title: "Quick Launcher Shortcut",
                    explainer: "The key combo that opens the Quick Launcher from any app.") {
            HotKeyRecorderControl(onChange: { appState.quickLauncher.updateHotKey() })
        }
    }
}

extension InterfaceSectionContent {
    /// The stored preference ↔ the row the picker shows. A preference naming
    /// a catalog this build no longer ships reads as System Default, and
    /// picking it again ("") means "follow the Mac" — install() then writes
    /// no override, which is also what an untouched preference looks like.
    fileprivate var languageChoice: Binding<AppLanguageChoice> {
        Binding(
            get: { AppLanguageChoice(raw: languageRaw, localizations: Bundle.main.localizations) },
            set: { choice in
                languageRaw = choice.code ?? ""
                AppRelaunch.relaunch(at: Bundle.main.bundlePath)
            })
    }

    /// The color well ↔ the stored "#RRGGBB" (empty = the theme's ground).
    fileprivate var terminalBackground: Binding<Color> {
        Binding(
            get: {
                let rgb = TerminalTheme.RGB(hex: terminalBackgroundHex)
                    ?? (TerminalTheme.theme(terminalThemeId) ?? TerminalTheme.theme(TerminalTheme.defaultId)!).background
                return Color(.sRGB, red: Double(rgb.r) / 255, green: Double(rgb.g) / 255, blue: Double(rgb.b) / 255)
            },
            set: { color in
                guard let c = NSColor(color).usingColorSpace(.sRGB) else { return }
                terminalBackgroundHex = TerminalTheme.RGB(UInt8((c.redComponent * 255).rounded()),
                                                          UInt8((c.greenComponent * 255).rounded()),
                                                          UInt8((c.blueComponent * 255).rounded())).hex
            })
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

    private static let explainer = "What you say to call the assistant in hands-free mode."

    var body: some View {
        SearchableRow(searchText: ["Wake phrase", "Hey Loki", Self.explainer]) {
            VStack(alignment: .leading, spacing: 6) {
                Text(L10n.text("Wake phrase")).font(.app(.rowTitle).weight(.semibold))
                TextField("Hey Loki", text: $appState.serverOptions.wakePhrase)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 220)
                Text(L10n.text(Self.explainer))
                    .font(.app(.caption2)).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Transcribe section

/// The tray Transcript's settings: which model writes the meeting notes, when a
/// recording stops by itself and when one starts by itself. App-side like the
/// voice fields, so never a restart.
private struct TranscribeSectionContent: View {
    @EnvironmentObject var appState: AppState

    private var pickable: [LocalModel] { appState.localModels.filter(\.isChatPickable) }

    /// A chosen model that is gone shows as the fallback it gets: the chat model in use.
    private var notesModel: Binding<String> {
        Binding(get: {
            let v = appState.serverOptions.transcriptNotesModel
            return v == ServerOptions.notesOff || pickable.contains(where: { $0.path == v }) ? v : ""
        }, set: { appState.serverOptions.transcriptNotesModel = $0 })
    }

    var body: some View {
        SettingsRow(title: "Meeting notes", explainer: "The model that writes a title, summary, decisions and action items above each transcript.") {
            Picker("", selection: notesModel) {
                Text(L10n.text("Chat model in use")).font(.app(.body)).tag("")
                Text(L10n.text("Off")).font(.app(.body)).tag(ServerOptions.notesOff)
                ForEach(pickable) { model in
                    Text(model.displayLabel).font(.app(.body)).tag(model.path)
                }
            }
            .labelsHidden()
            .frame(width: 220).font(.app(.body))
        }
        SettingsRow(title: "Stop after silence", explainer: "Ends a recording when nothing has been heard for this long, then transcribes it.") {
            Picker("", selection: $appState.serverOptions.transcriptAutoStopMinutes) {
                Text(L10n.text("Never")).font(.app(.body)).tag(0)
                ForEach([5, 10, 15, 30, 60], id: \.self) { minutes in
                    Text(L10n.format("%d minutes", minutes)).font(.app(.body)).tag(minutes)
                }
            }
            .labelsHidden()
            .frame(width: 140).font(.app(.body))
        }
        SettingsRow(title: "Start when the camera turns on", explainer: "Starts a transcript when an app such as Zoom, Teams or FaceTime turns on your camera.") {
            Toggle("", isOn: $appState.serverOptions.transcriptAutoStartCamera)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
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
    /// The recorder is the store's, so a filter edit cannot stop a capture
    /// mid-recording; the store builds it on the first record.
    @EnvironmentObject var formState: SettingsFormState
    @State private var voiceError: String?
    /// Built lazily against the app's server so previews reuse the resident
    /// Kokoro model instead of loading it per click.
    @StateObject private var previewer = VoicePreviewer()
    @EnvironmentObject private var downloads: DownloadManager

    private static let explainer = "A few seconds of clear speech that answers are spoken in."
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
        if appState.serverOptions.voiceEngine == .system {
            SearchableRow(searchText: ["System voice", "Apple voice"]) {
                systemVoiceBody
            }
        }
    }

    /// The app-wide Apple voice; an agent with its own voice still overrides it.
    @ViewBuilder
    private var systemVoiceBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(L10n.text("System voice")).font(.app(.headline).weight(.semibold))
            HStack(spacing: 8) {
                Picker("", selection: Binding(get: { appState.voice.selectedVoiceId },
                                              set: { appState.voice.selectVoice($0) })) {
                    ForEach(appState.voice.availableVoices) { v in
                        Text(v.displayName).font(.app(.body)).tag(Optional(v.id))
                    }
                }
                .labelsHidden()
                .frame(width: 260)
                Spacer(minLength: 0)
            }
        }
    }

    @ViewBuilder
    private var engineBody: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(L10n.text("Voice engine")).font(.app(.rowTitle).weight(.semibold))
            Picker("", selection: $appState.serverOptions.voiceEngine) {
                ForEach(VoiceEngine.allCases, id: \.self) { e in
                    Text(L10n.text(e.label)).tag(e)
                }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            Text(L10n.text(Self.engineExplainer(appState.serverOptions.voiceEngine)))
                .font(.app(.caption2)).foregroundStyle(.secondary)
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
            Text(L10n.text("Kokoro voice")).font(.app(.headline).weight(.semibold))
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
                        Section(L10n.text(group.language)) {
                            ForEach(group.voices, id: \.self) { v in
                                Text(KokoroVoiceCatalog.displayName(for: v)).font(.app(.body)).tag(v)
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
                        Label("Play", systemImage: "play.circle").font(.app(.body))
                    }
                }
                .help(kokoroReady ? "Hear a short sample of this voice"
                                  : "Download the model first")
                .disabled(previewer.active != nil || !kokoroReady)

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(L10n.text("Voices blend: type several separated by commas (af_bella,af_sky) to make a new one."))
                .font(.app(.caption2)).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let e = previewer.error {
                Text(L10n.text(e)).font(.app(.caption)).foregroundStyle(.orange)
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
            Text(L10n.text("Voice clone clip")).font(.app(.headline).weight(.semibold))
            HStack(spacing: 8) {
                if !appState.serverOptions.voiceClonePath.isEmpty {
                    Image(systemName: "waveform").foregroundStyle(.secondary)
                    // Prefer the display label — the stored file is always the
                    // normalized "voice-clone.wav", which says nothing.
                    // "Recorded clip" is copy, not a file name: the row resolves
                    // it, so the stored label stays language-neutral.
                    let label = appState.serverOptions.voiceCloneLabel.isEmpty
                        ? (appState.serverOptions.voiceClonePath as NSString).lastPathComponent
                        : appState.serverOptions.voiceCloneLabel
                    Text(L10n.text(label))
                        .font(.app(.caption)).lineLimit(1).truncationMode(.middle)
                    Button { clearVoice() } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.borderless).foregroundStyle(.secondary)
                        .help("Remove the clip — voice mode falls back to the system voice")
                } else {
                    Text(L10n.text("None — voice mode uses the system voice."))
                        .font(.app(.caption)).foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 8) {
                Button { chooseVoiceFile() } label: { Label("Choose file…", systemImage: "folder") }
                if let recorder = formState.recorder {
                    VoiceRecordControl(recorder: recorder, onRecord: startRecording, onStop: stopRecording)
                } else {
                    Button { startRecording() } label: { Label("Record", systemImage: "mic") }
                }
            }
            .font(.app(.caption))
            Text(L10n.text(Self.explainer))
                .font(.app(.caption2)).foregroundStyle(.secondary)
            if let voiceError {
                Text(L10n.text(voiceError)).font(.app(.caption)).foregroundStyle(.red)
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
                voiceError = L10n.text("Microphone access denied. Enable it in System Settings ▸ Privacy ▸ Microphone.")
                return
            }
            do { try formState.audioRecorder().start() }
            catch { voiceError = error.localizedDescription }
        }
    }

    private func stopRecording() {
        guard let data = formState.recorder?.stop() else { voiceError = L10n.text("Nothing was recorded."); return }
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

/// Record / stop for the clip row. The store owns the recorder, so this is
/// what still observes it — the elapsed label has to tick while recording.
private struct VoiceRecordControl: View {
    @ObservedObject var recorder: AudioRecorder
    let onRecord: () -> Void
    let onStop: () -> Void

    var body: some View {
        if recorder.isRecording {
            Button(role: .destructive, action: onStop) {
                Label(L10n.format("Stop (%.1fs)", recorder.duration), systemImage: "stop.circle")
            }
        } else {
            Button(action: onRecord) { Label("Record", systemImage: "mic").font(.app(.callout)) }
        }
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
            explainer: "Chats use tools only when you turn them on yourself."
        ) {
            Toggle("", isOn: $appState.serverOptions.toolsOnlyWhenAsked)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "Agent workspace folder",
            explainer: "The folder the agent works in by default, also shared with the sandbox."
        ) {
            HStack(spacing: 8) {
                Text((currentWorkspace as NSString).abbreviatingWithTildeInPath)
                    .font(.app(.caption).monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .help(currentWorkspace)
                Button {
                    if let picked = WorkspacePicker.pickDirectory() {
                        appState.setDefaultAgentWorkspace(picked)
                    }
                } label: { Text("Choose…")
                    .font(.app(.body)) }
            }
        }

        // No host shell in the App Store build → the sandbox can't be turned
        // off (`AgentSandbox.resolveEnabled`), so offering the toggle would be
        // a lie; the base image is likewise locked to the bundled guest.
        if BuildFeatures.current.hostShell {
            SettingsRow(
                title: "Sandbox agent commands",
                explainer: "Run the agent's commands in an isolated Linux sandbox that can only reach the work folder."
            ) {
                Toggle("", isOn: $appState.serverOptions.sandbox.enabled)
                    .labelsHidden()
                    .toggleStyle(.switch).font(.app(.body))
            }
        }

        SettingsRow(
            title: "Network + port mapping",
            explainer: "Give the sandbox internet access and open its servers on localhost."
        ) {
            Toggle("", isOn: $appState.serverOptions.sandbox.network)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "Reset sandbox",
            explainer: "Erase everything in the sandbox, including installed tools and logins, but not your files or models."
        ) {
            Button(role: .destructive) {
                showResetConfirm = true
            } label: {
                if resetting {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Resetting…").font(.app(.body))
                    }
                } else {
                    Label("Reset Sandbox…", systemImage: "trash").font(.app(.body))
                        .foregroundStyle(.red)
                }
            }
            .disabled(resetting)
            .confirmationDialog(
                "Reset the Agent Sandbox?",
                isPresented: $showResetConfirm,
                titleVisibility: .visible
            ) {
                Button(role: .destructive) {
                    resetting = true
                    AgentSandbox.shared.resetAllData {
                        resetting = false
                    }
                } label: { Text("Delete All Sandbox Data")
                    .font(.app(.body)) }
                .keyboardShortcut(.defaultAction)
                Button(role: .cancel) {} label: { Text("Cancel")
                    .font(.app(.body)) }
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
                        .font(.app(.body))
                    Spacer(minLength: 12)
                    statusPill
                }
            }
        }

        SettingsRow(
            title: "Enable Telegram bot",
            explainer: "Pass Telegram messages to your local model, which needs a bot token and a running model."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enabled)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "Bot token",
            explainer: "The token @BotFather gives you, stored only on this Mac."
        ) {
            TextField("", text: $appState.serverOptions.telegram.botToken,
                      prompt: Text("123456:ABC-DEF…"))
                .textFieldStyle(.roundedBorder)
                .font(.app(.body).monospaced())
                .frame(width: 260)
        }

        SettingsRow(
            title: "Tools",
            explainer: "Let the bot run commands and edit files on this Mac, a power anyone who can message it gets too."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.agentMode)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "MCP tools",
            explainer: "Let the bot use your enabled MCP servers."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.useMCP)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "Enable thinking",
            explainer: "Let models think before answering; the bot sends only the answer."
        ) {
            Toggle("", isOn: $appState.serverOptions.telegram.enableThinking)
                .labelsHidden()
                .toggleStyle(.switch).font(.app(.body))
        }

        SettingsRow(
            title: "Answer as agent",
            explainer: "Reply as one of your agents instead of with the settings above."
        ) {
            Picker("", selection: $appState.serverOptions.telegram.agentId) {
                Text("None").font(.app(.body)).tag(UUID?.none)
                ForEach(appState.agents.allAgents) { agent in
                    Text(agent.name).font(.app(.body)).tag(UUID?.some(agent.id))
                }
            }
            .labelsHidden()
            .frame(width: 200).font(.app(.body))
        }

        // Allow-list / lock control.
        SearchableRow(searchText: ["Locked to", "Reset lock", Self.lockExplainer]) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text("Locked to")
                        .font(.app(.body))
                    Spacer(minLength: 12)
                    HStack(spacing: 8) {
                        Text(L10n.text(lockLabel))
                            .font(.app(.caption).monospacedDigit())
                            .foregroundStyle(telegram.allowedChatIds.isEmpty ? .secondary : .primary)
                        Button {
                            appState.serverOptions.telegram.allowedChatIds = []
                        } label: { Text("Reset lock")
                            .font(.app(.body)) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(telegram.allowedChatIds.isEmpty)
                    }
                }
                Text(L10n.text(Self.lockExplainer))
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }

        // Setup steps. The divider rides inside the searchable row so a
        // filtered view never leaves a dangling separator behind.
        SearchableRow(searchText: ["Setup", "BotFather", "newbot", "token", "lock it to your chat"],
                      separatorGap: nil) {
            VStack(alignment: .leading, spacing: 6) {
                Divider()
                    .padding(.bottom, 6)
                Text("Setup")
                    .font(.app(.caption).weight(.semibold))
                Text("1. In Telegram, open @BotFather and send /newbot.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
                Text("2. Copy the token it gives you and paste it above.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
                Text("3. Turn on “Enable Telegram bot”, then message your bot once to lock it to your chat.")
                    .font(.app(.caption2)).foregroundStyle(.secondary)
                Link("Open @BotFather ↗", destination: URL(string: "https://t.me/botfather")!)
                    .font(.app(.caption2))
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
            .font(.app(.caption2).monospaced())
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
    @EnvironmentObject var formState: SettingsFormState

    /// Engine rows to display — drops the `mlx-serve` app row (already shown as
    /// "Installed version"). Falls back to the compile-time llama pin so the
    /// section is never empty if the probe hasn't returned yet.
    private var engineRows: [EngineVersion] {
        let rows = formState.engineVersions.filter { $0.name != "mlx-serve" }
        return rows.isEmpty
            ? [EngineVersion(name: "llama.cpp", version: UpdateChecker.bundledLlamaTag)]
            : rows
    }

    var body: some View {
        SettingsRow(
            title: "Check for updates automatically",
            explainer: "Check GitHub for a new version once a day."
        ) {
            Toggle("", isOn: Binding(
                get: { updates.autoCheckEnabled },
                set: { updates.autoCheckEnabled = $0 }))
                .toggleStyle(.switch)
                .labelsHidden().font(.app(.body))
        }

        SettingsRow(
            title: L10n.format("Installed version — v%@", updates.currentVersion),
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
            .font(.app(.callout))
            .disabled(busy)
        }

        // Embedded-engine versions, read from `mlx-serve --version` — a
        // print-and-exit path that never binds a port or loads a model, so
        // Settings shows them even when the server is stopped.
        ForEach(engineRows) { row in
            SettingsRow(
                title: L10n.format("%@ — %@", Self.engineLabel(row.name), row.version),
                explainer: Self.engineExplainer(row.name)
            ) {
                EmptyView()
            }
        }
        .task {
            guard formState.engineVersions.isEmpty else { return }
            formState.engineVersions = await EngineVersions.probe(binaryPath: ServerManager.resolveBinaryPath())
        }

        if let update = updates.available {
            SettingsRow(
                title: L10n.format("MLX-Serve v%@ is available", update.version),
                explainer: L10n.text("Downloads MLX-Serve.dmg from the release, replaces the app, and relaunches.")
            ) {
                switch updates.phase {
                case .downloading(let fraction):
                    ProgressView(value: max(0, min(1, fraction)))
                        .progressViewStyle(.linear)
                        .frame(width: 160)
                case .installing:
                    ProgressView().controlSize(.small)
                default:
                    Button {
                        Task { await updates.downloadAndInstall() }
                    } label: { Text("Download & Install")
                        .font(.app(.body)) }
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
        Text(L10n.text(label))
            .font(.app(.body).monospacedDigit())
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
