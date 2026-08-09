import SwiftUI
import AppKit

/// The Agents window: saved personas on the left, the selected one's editor on
/// the right. Configuration only — you TALK to an agent in the Chat window (or
/// the tray, or a task); this is where you decide who they are.
enum AgentsWindowFocus {
    /// The id to select, or nil to leave the selection exactly as it is.
    static func selection(pending: UUID?, current: UUID?, first: UUID?) -> UUID? {
        if let pending { return pending }
        return current == nil ? first : nil
    }
}

/// The editing state the two agent columns share.
@MainActor
final class AgentsWorkspaceModel: ObservableObject {
    @Published var selectedId: UUID?
    /// The row being edited, held apart from the store so typing doesn't
    /// rewrite the JSON on every keystroke; committed on change of
    /// focus/selection and on Save.
    @Published var draft: Agent?
    @Published var isWriting = false
    @Published var alert: AlertItem?

    /// ONE alert presentation path per surface — an alert modifier on an
    /// ancestor shadows a descendant's, so a second one would silently never
    /// present (the sandbox ✕-confirm class).
    struct AlertItem: Identifiable {
        enum Kind { case message(String), confirmDelete(Agent) }
        let id = UUID()
        let title: String
        let kind: Kind
    }

    func message(_ title: String, _ text: String) {
        alert = AlertItem(title: title, kind: .message(text))
    }

    /// Write the draft back to the store — Save, selection changes and
    /// `adopt` all commit through here. Wake-phrase collisions are refused at
    /// save time: a colliding phrase makes both agents unreachable by voice
    /// and there is nothing to see until you try talking.
    func commitDraft(to store: AgentStore, defaultAgentId: UUID?) {
        guard var d = draft, !d.isBuiltIn else { return }
        if let phrase = d.wakePhrase,
           WakeWord.collides(phrase, with: store.takenWakePhrases(excluding: d.id)) {
            d.wakePhrase = nil
            draft = d
            message("That wake phrase is taken",
                    "Another agent (or the app's own phrase) already answers to that name, so both would be unreachable. Pick a different one.")
        }
        store.update(d)
        // A live tab talking to this agent picks the change up on its next
        // turn; its voice applies from the next sentence.
        if defaultAgentId == d.id { ActiveAgentVoice.set(d.resolvedVoice) }
    }

    /// Select `agent` for editing, committing the outgoing draft FIRST.
    /// Create/duplicate set `selectedId` AND `draft` in one move, so the
    /// detail pane's `onChange(of: selectedId)` commit reads the NEW agent —
    /// without this, pending edits to whoever was selected are silently lost.
    func adopt(_ agent: Agent, committingTo store: AgentStore, defaultAgentId: UUID?) {
        commitDraft(to: store, defaultAgentId: defaultAgentId)
        selectedId = agent.id
        draft = agent
    }
}

struct AgentsWindow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var store: AgentStore

    @StateObject private var model = AgentsWorkspaceModel()

    var body: some View {
        NavigationSplitView {
            AgentListPane(model: model)
                .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        } detail: {
            AgentDetailPane(model: model)
        }
        .onAppear { AppActivation.focus() }
    }
}

// MARK: - The two columns

/// The agent list. A column of the chat window's split when Agents is up, and
/// the leading column of the standalone Agents window.
struct AgentListPane: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var store: AgentStore
    @ObservedObject var model: AgentsWorkspaceModel

    var body: some View {
        list
            // Title and control in the toolbar, the same shape the Tasks column
            // takes — both are static, so neither is the runtime-variable
            // content NSToolbar cannot re-measure.
            .toolbar {
                // Same shape as the Tasks column (`PaneTitleBar`), except the
                // control is a MENU — the + offers the agent types.
                if #available(macOS 26.0, *) {
                    ToolbarItem(placement: .navigation) { paneTitleOnly }
                        .sharedBackgroundVisibility(.hidden)
                } else {
                    ToolbarItem(placement: .navigation) { paneTitleOnly }
                }
            }
    }

    private var paneTitleOnly: some View {
        Text("Agents")
            .font(.headline)
            .foregroundStyle(.primary)
            .padding(.leading, 4)
    }

    /// `+` offers the TYPES, not a blank row.
    @ViewBuilder
    var newAgentMenuItems: some View {
        Section("Start from a type") {
            ForEach(Agent.starters) { starter in
                Button {
                    newAgent(basedOn: starter)
                } label: {
                    Label(starter.name, systemImage: starter.symbol)
                }
            }
        }
        Divider()
        Button {
            newAgent(basedOn: nil)
        } label: {
            Label("Blank agent", systemImage: "square.dashed")
        }
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 2) {
                sectionLabel("Your agents")
                createRow
                ForEach(store.sortedAgents) { agent in
                    agentRow(agent)
                }
                sectionLabel("Starters")
                    .padding(.top, 10)
                ForEach(Agent.starters) { agent in
                    agentRow(agent)
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 12)
        }
    }

    private func sectionLabel(_ title: String) -> some View {
        Text(title.uppercased())
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .kerning(0.5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.top, 8)
            .padding(.bottom, 4)
    }

    /// The one way to make an agent, and it offers the TYPES — a dashed row
    /// rather than a filled one, because it creates rather than navigates.
    private var createRow: some View {
        Menu {
            newAgentMenuItems
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "plus")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 18)
                Text("Create New Agent")
                    .font(.subheadline)
                Spacer(minLength: 0)
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .frame(height: 34)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                    .foregroundStyle(.quaternary))
            .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .help("New agent — start from a type, or blank")
    }

    private func agentRow(_ agent: Agent) -> some View {
        AgentListRow(agent: agent,
                     decision: appState.agentModelDecision(for: agent),
                     selected: model.selectedId == agent.id,
                     select: { model.selectedId = agent.id },
                     startChat: {
                         appState.showConversation()
                         appState.startChat(withAgent: agent.id)
                     })
    }

    /// A new agent, optionally seeded from a starter.
    private func newAgent(basedOn starter: Agent?) {
        var agent = Agent(name: starter.map { "\($0.name) copy" } ?? "New Agent",
                          brief: starter?.brief ?? "",
                          systemPrompt: starter?.systemPrompt ?? "")
        if let starter {
            agent.symbol = starter.symbol
            agent.capabilities = starter.capabilities
        }
        store.add(agent)
        model.adopt(agent, committingTo: store, defaultAgentId: appState.defaultAgentId)
    }
}

/// The selected agent's editor.
struct AgentDetailPane: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var store: AgentStore
    @ObservedObject var model: AgentsWorkspaceModel

    var body: some View {
        Group {
            if let draft = model.draft {
                AgentEditor(agent: bindingToDraft(draft),
                            isWriting: $model.isWriting,
                            onWrite: { writePrompt() },
                            onSave: { commit() },
                            onStartChat: {
                                commit()
                                appState.showConversation()
                                appState.startChat(withAgent: draft.id)
                            },
                            onDuplicate: { duplicate(draft) },
                            onDelete: { model.alert = .init(title: "Delete “\(draft.name)”?",
                                                            kind: .confirmDelete(draft)) },
                            onNotify: { model.message($0, $0) })
            } else {
                ContentUnavailableView("No agent selected",
                                       systemImage: "person.crop.circle.badge.questionmark",
                                       description: Text("Pick an agent, or create one from a type."))
            }
        }
        .onChange(of: model.selectedId) { _, newValue in
            commit()
            model.draft = store.agent(id: newValue)
        }
        .onAppear { applyFocus() }
        // A deep link that arrives while this is already open has to move the
        // selection — `onAppear` alone would leave the user staring at whoever
        // they were editing.
        .onChange(of: appState.pendingAgentSelection) { _, _ in applyFocus() }
        .alert(item: $model.alert) { item in
            switch item.kind {
            case .message(let text):
                return Alert(title: Text("Agents"), message: Text(text),
                             dismissButton: .default(Text("OK")))
            case .confirmDelete(let agent):
                return Alert(title: Text(item.title),
                             message: Text("This can't be undone."),
                             primaryButton: .destructive(Text("Delete")) {
                                 store.delete(id: agent.id)
                                 model.selectedId = store.allAgents.first?.id
                                 model.draft = store.agent(id: model.selectedId)
                             },
                             secondaryButton: .cancel())
            }
        }
    }

    private func bindingToDraft(_ current: Agent) -> Binding<Agent> {
        Binding(get: { model.draft ?? current }, set: { model.draft = $0 })
    }

    /// Land on whoever was asked for, then consume the request.
    private func applyFocus() {
        guard let id = AgentsWindowFocus.selection(pending: appState.pendingAgentSelection,
                                                   current: model.selectedId,
                                                   first: store.allAgents.first?.id) else { return }
        appState.pendingAgentSelection = nil
        // A deep link to the agent ALREADY showing can't rely on
        // `onChange(of: selectedId)` — the id doesn't change, so the draft has
        // to be reloaded here or the click does nothing at all. Commit first,
        // exactly like a selection change would: the reload must not discard
        // pending edits.
        if model.selectedId == id {
            model.commitDraft(to: store, defaultAgentId: appState.defaultAgentId)
            model.draft = store.agent(id: id)
        } else {
            model.selectedId = id
        }
    }

    private func duplicate(_ agent: Agent) {
        model.adopt(store.duplicate(agent), committingTo: store,
                    defaultAgentId: appState.defaultAgentId)
    }

    /// Write the draft back to the store (the model owns the rule — see
    /// `AgentsWorkspaceModel.commitDraft`).
    private func commit() {
        model.commitDraft(to: store, defaultAgentId: appState.defaultAgentId)
    }

    /// Ask the current model to turn the brief into a system prompt. A failure
    /// falls back to the user's own words rather than losing what they typed.
    private func writePrompt() {
        guard var d = model.draft, !d.isBuiltIn else { return }
        let brief = d.brief.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !brief.isEmpty else {
            model.message("Describe the agent first",
                          "Write a line or two about the assistant you want, then let the model turn it into a prompt.")
            return
        }
        model.isWriting = true
        Task {
            defer { model.isWriting = false }
            let result: AgentWriter.Draft
            do {
                result = try await AgentComposer.draftAgent(brief: brief, appState: appState)
            } catch {
                result = AgentWriter.fallbackDraft(brief: brief)
                model.message("Wrote it from your description",
                              "\(error.localizedDescription)\n\nYour description was saved as the prompt — edit it directly, or try again once a model is running.")
            }
            d.systemPrompt = result.systemPrompt
            if d.name.isEmpty || d.name == "New Agent" { d.name = result.name }
            if d.symbol == "sparkles" { d.symbol = AgentSymbol.pick(for: "\(brief) \(result.name)") }
            model.draft = d
            commit()
        }
    }
}

// MARK: - Sidebar row

private struct AgentListRow: View {
    let agent: Agent
    let decision: AgentModelSwitch.Decision
    let selected: Bool
    let select: () -> Void
    let startChat: () -> Void

    @State private var hovering = false

    private var selectable: Bool { AgentModelSwitch.isSelectable(decision) }

    var body: some View {
        // A real Button for the row, and the start-chat control as a sibling
        // laid OVER its trailing edge — never a tap gesture wrapped around
        // both, which on macOS swallows the child's clicks silently.
        Button(action: select) {
            HStack(spacing: 8) {
                Image(systemName: agent.symbol)
                    .font(.system(size: 12))
                    .frame(width: 18)
                    .foregroundStyle(selectable ? Color.accentColor : .secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text(agent.name)
                        .font(.subheadline)
                        .foregroundStyle(selected ? Color.accentColor : .primary)
                        .lineLimit(1)
                    if let sub = subtitle {
                        Text(sub).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                    }
                }
                Spacer(minLength: 0)
                // Reserved at all times, so the name doesn't reflow under the
                // pointer as the trailing control appears.
                Color.clear.frame(width: 20, height: 1)
            }
            .padding(.horizontal, 8)
            .frame(minHeight: 34)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(SidebarRowStyle.fill(selected: selected, hovering: hovering)))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(selectable ? 1 : 0.55)
        .overlay(alignment: .trailing) { trailingControl }
        .onHover { hovering = $0 }
    }

    /// A built-in says so; anything else offers the one thing you do WITH an
    /// agent rather than to it. Only on the row you are pointing at or on —
    /// a control on every row is a column of buttons, not a list of agents.
    @ViewBuilder
    private var trailingControl: some View {
        if hovering || selected {
            Button(action: startChat) {
                Image(systemName: "bubble.left.and.bubble.right")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(!selectable)
            .help("Start a chat with \(agent.name)")
            .padding(.trailing, 6)
        } else if agent.isBuiltIn {
            Image(systemName: "lock")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.trailing, 10)
                .help("Built-in — duplicate it to make changes")
        }
    }

    private var subtitle: String? {
        switch decision {
        case .needsDownload: return "Model not downloaded"
        case .unavailable: return "Model unreachable"
        case .noChange, .load, .lan:
            let brief = agent.brief.trimmingCharacters(in: .whitespacesAndNewlines)
            return brief.isEmpty ? nil : brief
        }
    }
}

// MARK: - Editor

private struct AgentEditor: View {
    @Binding var agent: Agent
    @Binding var isWriting: Bool
    let onWrite: () -> Void
    let onSave: () -> Void
    /// Start a conversation as this agent. The only moment a session's agent
    /// can be decided (there is no `setAgent`), so it needs a home — it used to
    /// be a menu in the sidebar, which is the wrong place now that agents have
    /// a pane of their own.
    let onStartChat: () -> Void
    let onDuplicate: () -> Void
    let onDelete: () -> Void
    let onNotify: (String) -> Void

    @EnvironmentObject var appState: AppState
    @State private var showAdvancedTools = false
    /// Collapsed state of Capabilities / Model / Workspace / Sampling. Per-agent
    /// like `showAdvancedTools` — the editor is REUSED across selections, so it
    /// re-syncs on the id change rather than leaking one agent's disclosure into
    /// the next.
    @State private var showMoreOptions = false
    @StateObject private var previewer = VoicePreviewer()
    /// Uploaded clips, re-stated when the editor appears and after an upload —
    /// not per render (the body re-evaluates far too often to stat a folder).
    @State private var clips: [VoiceClipLibrary.Clip] = []
    @State private var clipError: String?
    /// Is the Qwen3-TTS checkpoint (the model the clone path actually loads —
    /// `AudioGenSettings.resolvedModel`) on disk? Without it every cloned sentence
    /// silently falls back to the system voice, so the clip rows must not be
    /// offered as if they worked. Stat'd per appearance, never per render.
    @State private var ttsDownloaded = false
    /// System-voice auditioning. Built lazily by `SystemSpeechSynthesizer` itself,
    /// so holding one here doesn't create an audio graph until Preview is pressed
    /// (the launch-eager-audio TCC rule).
    @State private var systemPreview = SystemSpeechSynthesizer()

    private var readOnly: Bool { agent.isBuiltIn }

    var body: some View {
        // A ScrollView of hand-built cards, not a `Form(.formStyle(.grouped))`.
        ScrollView {
            editorColumn
                .frame(maxWidth: AgentEditorMetrics.contentMaxWidth, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(AgentEditorMetrics.contentPadding)
        }
        .toolbar { agentActions }
        .onAppear {
            previewer.attach(server: appState.server)
            showAdvancedTools = agent.capabilities.advancedTools != nil
            clips = VoiceClipLibrary.clips()
            ttsDownloaded = VoiceCloneMenuModel.ttsModelDownloaded()
        }
        // The editor is REUSED as the selection changes (same class as the chat
        // detail view), so per-agent view state has to re-sync on the id change.
        .onChange(of: agent.id) { _, _ in
            showAdvancedTools = agent.capabilities.advancedTools != nil
            showMoreOptions = false
        }
        .onDisappear { onSave() }
    }

    /// The column itself. Each section is its own typed property rather than a
    /// nesting of expressions: one deeply-nested view expression here took the
    /// release build from ~48s to over ten minutes with no warning, which is
    /// SwiftUI's type-checker grinding rather than anything being wrong.
    private var editorColumn: some View {
        VStack(alignment: .leading, spacing: AgentEditorMetrics.sectionSpacing) {
            identityHeader
            descriptionField
            readOnlyNotice
            promptSection
            identitySection
            // Everything below is collapsed by default. An agent is a prompt, a
            // name and a voice; capabilities, a pinned model, a workspace and
            // sampling are real but rarely-touched, and putting five sections of
            // them between "what should this be?" and the Delete button made the
            // editor read as a settings panel. The row names whatever is set
            // behind it (`AgentAdvancedSummary`) so a collapsed non-default is
            // still discoverable.
            moreOptionsSection
            advancedSections
            startChatButton
        }
    }

    @ViewBuilder
    private var advancedSections: some View {
        if showMoreOptions {
            capabilitiesSection
            modelSection
            workspaceSection
            samplingSection
        }
    }

    // MARK: Name & description

    /// Who this agent IS: the symbol on its own card, the name beside it.
    private var identityHeader: some View {
        HStack(alignment: .bottom, spacing: AgentEditorMetrics.labelSpacing) {
            symbolPicker
                .padding(AgentEditorMetrics.avatarPadding)
                .agentSurface(radius: AgentEditorMetrics.wellRadius)
            AgentLabeledField("Name") { nameField }
        }
    }

    private var nameField: some View {
        TextField("Name", text: $agent.name)
            .textFieldStyle(.plain)
            .font(.title3.weight(.medium))
            .disabled(readOnly)
    }

    private var descriptionField: some View {
        AgentLabeledField("Description") { briefField }
    }

    /// A built-in refuses every edit, so it says so where you would start
    /// typing — and offers the one move that makes it yours.
    @ViewBuilder
    private var readOnlyNotice: some View {
        if readOnly {
            AgentCard {
                HStack(spacing: 8) {
                    Image(systemName: "lock.fill").foregroundStyle(.secondary)
                    Text("This is one of the built-in agents. Duplicate it to make it yours.")
                        .font(.callout)
                    Spacer(minLength: 8)
                    Button("Duplicate", action: onDuplicate)
                }
            }
        }
    }

    /// Same `prompt:` rule as the wake phrase: an example handed to the title
    /// argument becomes a permanent LABEL beside the field instead of showing
    /// through an empty one.
    private var briefField: some View {
        TextField("", text: $agent.brief,
                  prompt: Text("e.g. a blunt Swift code reviewer that never comments on style"),
                  axis: .vertical)
            .textFieldStyle(.plain)
            .lineLimit(1...3)
            .disabled(readOnly)
    }

    /// Duplicate and Delete, as icons. Delete is HIDDEN on a built-in rather
    /// than shown disabled: it can only fail there, and a dead control is worse
    /// than an absent one.
    @ToolbarContentBuilder
    private var agentActions: some ToolbarContent {
        ToolbarItem {
            Button(action: onDuplicate) {
                Image(systemName: "plus.square.on.square")
            }
            .help("Duplicate this agent")
        }
        if !readOnly {
            ToolbarItem {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                }
                .help("Delete this agent")
            }
        }
    }

    /// The one thing you DO with an agent, in the flow of the column at the end
    /// of it — sized, not stretched: a full-width button reads as a bar. It was
    /// one (`safeAreaInset`), which spent a permanent strip of the pane on a
    /// control you press once. Duplicate and Delete stay toolbar icons: they act
    /// on the agent rather than with it, and a destructive control beside the
    /// primary action is how the wrong one gets clicked.
    private var startChatButton: some View {
        Button(action: onStartChat) {
            Label("Start Chat with this Agent", systemImage: "bubble.left.and.bubble.right")
                .frame(maxWidth: AgentEditorMetrics.primaryMaxWidth)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
    }

    /// The symbol IS its own picker — click the glyph.
    private var symbolPicker: some View {
        Menu {
            ForEach(AgentSymbol.pickerChoices, id: \.self) { symbol in
                Button { agent.symbol = symbol } label: {
                    Label(symbol, systemImage: symbol)
                }
            }
        } label: {
            symbolBadge
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .disabled(readOnly)
        .help("Change the symbol")
    }

    private var symbolBadge: some View {
        Image(systemName: agent.symbol)
            .font(.system(size: 19, weight: .medium))
            .foregroundStyle(Color.accentColor)
            .frame(width: AgentEditorMetrics.avatarSize, height: AgentEditorMetrics.avatarSize)
            .background(Circle().fill(Color.accentColor.opacity(0.16)))
            .overlay(alignment: .bottomTrailing) { symbolEditHint }
            .contentShape(Circle())
    }

    @ViewBuilder
    private var symbolEditHint: some View {
        if !readOnly {
            Image(systemName: "pencil.circle.fill")
                .font(.system(size: 14))
                .symbolRenderingMode(.palette)
                .foregroundStyle(Color.white, Color.accentColor)
        }
    }

    /// Voice belongs to identity, not to a section of its own: how an agent
    /// SOUNDS is the same kind of fact as what it's called and what wakes it,
    /// and all three are what you set when making one.
    private var identitySection: some View {
        AgentSection("Identity") {
            AgentCard {
                wakePhraseRow
                Divider()
                voiceRows
            }
        }
    }

    private var wakePhraseRow: some View {
        AgentEditorRow("Wake phrase",
                       caption: "Say this to hand the conversation to \(agent.name). Blank uses the app's own phrase.") {
            // `prompt:`, not the title argument — a TextField's title is a
            // LABEL, so passing the app phrase there parked it beside the
            // field permanently instead of showing through an empty one.
            TextField("", text: Binding(get: { agent.wakePhrase ?? "" },
                                        set: { agent.wakePhrase = $0.isEmpty ? nil : $0 }),
                      prompt: Text(appPhraseDisplay))
                .textFieldStyle(.plain)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 220)
                .disabled(readOnly)
        }
    }

    /// The app's own wake phrase, tidied for display ("Hey, Jarvis!" → "Hey
    /// Jarvis") — normalized first, so the placeholder shows the gate that will
    /// actually be listened for rather than the user's raw typing.
    private var appPhraseDisplay: String {
        WakeWord.display(WakeWord.normalizePhrase(appState.serverOptions.wakePhrase)
                         ?? WakeWord.defaultPhrase)
    }

    // MARK: More options

    /// The disclosure row for the collapsed sections.
    private var moreOptionsSection: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) { showMoreOptions.toggle() }
        } label: {
            HStack(spacing: 8) {
                Image(systemName: showMoreOptions ? "chevron.down" : "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("More options").font(.headline)
                Spacer(minLength: 8)
                // What's set behind the row while it's shut, so a collapsed
                // non-default isn't a setting nobody can find again.
                if !showMoreOptions, let summary = AgentAdvancedSummary.text(for: agent) {
                    Text(summary)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
            .padding(.horizontal, AgentEditorMetrics.cardPadding)
            .padding(.vertical, AgentEditorMetrics.cardPadding - 4)
            .frame(maxWidth: .infinity, alignment: .leading)
            .agentSurface()
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Capabilities, model, workspace and sampling. Most agents need none of these.")
    }

    // MARK: Prompt

    /// The prompt, then the things you can do to it.
    private var promptSection: some View {
        AgentSection("Prompt") {
            AgentCard {
                promptEditor
                promptActions
            }
        }
    }

    private var promptEditor: some View {
        TextEditor(text: $agent.systemPrompt)
            .font(.body)
            .scrollContentBackground(.hidden)
            .frame(minHeight: AgentEditorMetrics.promptMinHeight)
            .padding(AgentEditorMetrics.wellPadding)
            .agentWell()
            .disabled(readOnly)
    }

    private var promptActions: some View {
        HStack(spacing: 12) {
            AgentPillButton(title: isWriting ? "Writing…" : "Write it for me",
                            systemImage: "sparkles",
                            isBusy: isWriting,
                            action: onWrite)
                .disabled(readOnly || isWriting)
                .help("Ask the current model to turn your description into a system prompt. You can edit whatever it writes.")
            // The description itself lives in the header, under the name, where
            // it reads as what the agent IS. It was edited here too — one value
            // with two fields on one screen, which is confusing even when (as
            // here, sharing a binding) they cannot disagree.
            Text("Written from the description under the name.")
                .font(.subheadline).foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 8)
            Text("\(agent.systemPrompt.count)/\(AgentWriter.maxPromptCharacters)")
                .font(.subheadline).foregroundStyle(.tertiary)
                .monospacedDigit()
        }
    }

    // MARK: Capabilities

    private var capabilitiesSection: some View {
        AgentSection("Capabilities") {
            AgentCard(spacing: 12) {
                capabilityToggles
                Divider()
                advancedToolsDisclosure
            }
        }
    }

    /// Every control here sits on the trailing edge of an `AgentEditorRow`, the
    /// same grammar Identity speaks — a stack of leading-aligned controls in a
    /// card beside a card of trailing-aligned ones reads as two designs.
    /// `.switch` explicitly: a macOS Toggle outside a Form is a CHECKBOX, and
    /// the Form had been promoting these silently.
    @ViewBuilder
    private var capabilityToggles: some View {
        AgentEditorRow("Tools") {
            Toggle("", isOn: Binding(get: { agent.capabilities.tools },
                                     set: { agent.capabilities.tools = $0 }))
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(readOnly || showAdvancedTools)
                .help("The tool-calling loop: shell, files, search, tasks, media generation.")
        }
        AgentEditorRow("MCP") {
            Toggle("", isOn: Binding(get: { agent.capabilities.mcp },
                                     set: { agent.capabilities.mcp = $0 }))
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(readOnly)
                .help("Add the tools from every enabled Model Context Protocol server.")
        }
        AgentEditorRow("Web") {
            Toggle("", isOn: Binding(get: { agent.capabilities.web },
                                     set: { agent.capabilities.web = $0 }))
                .labelsHidden()
                .toggleStyle(.switch)
                .disabled(readOnly || showAdvancedTools)
                .help("Browse pages and search the web (browse + webSearch).")
        }
        AgentEditorRow("Thinking") {
            Picker("", selection: tristate($agent.enableThinking)) {
                Text("App default").tag(TriChoice.appDefault)
                Text("On").tag(TriChoice.on)
                Text("Off").tag(TriChoice.off)
            }
            .labelsHidden()
            .frame(width: Self.triPickerWidth)
            .disabled(readOnly)
        }
        AgentEditorRow("Approve tools") {
            Picker("", selection: tristate($agent.autoApproveTools)) {
                Text("App default").tag(TriChoice.appDefault)
                Text("Automatically").tag(TriChoice.on)
                Text("Ask every time").tag(TriChoice.off)
            }
            .labelsHidden()
            .frame(width: Self.triPickerWidth)
            .disabled(readOnly)
        }
    }

    /// ONE width for the tri-state pop-ups: `.fixedSize()` sized each to its
    /// own widest menu item, so Thinking and Approve tools rendered two
    /// different-width controls stacked on the same trailing edge. Wide
    /// enough for "Ask every time".
    private static let triPickerWidth: CGFloat = 150

    private var advancedToolsDisclosure: some View {
        DisclosureGroup(isExpanded: $showAdvancedTools) {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Pick exactly which tools this agent may call. Turning this on freezes the coarse switches above.")
                        .font(.caption2).foregroundStyle(.secondary)
                    // The chat's Tools menu's own groups (`AgentToolGroup`) —
                    // one grouping for both surfaces, and SessionToolDisableTests
                    // pins that the groups cover exactly the toggleable set. A
                    // flat 19-row centered column was a list nobody could scan.
                    LazyVGrid(columns: [GridItem(.flexible(), alignment: .topLeading),
                                        GridItem(.flexible(), alignment: .topLeading)],
                              alignment: .leading, spacing: 16) {
                        ForEach(AgentToolGroup.allCases, id: \.self) { group in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(group.title.uppercased())
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .kerning(0.5)
                                ForEach(group.tools, id: \.self) { tool in
                                    Toggle(isOn: toolBinding(tool)) {
                                        Label(tool.displayName, systemImage: tool.icon)
                                    }
                                    .disabled(readOnly || !showAdvancedTools)
                                }
                            }
                        }
                    }
                    if showAdvancedTools {
                        Button("Back to the simple switches") {
                            agent.capabilities.closeAdvanced()
                            showAdvancedTools = false
                        }
                        .disabled(readOnly)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
            } label: {
                // macOS only hit-tests the chevron on a DisclosureGroup's plain
                // string label, so the word "Advanced" was dead. The label holds
                // no buttons of its own, so a tap gesture here is safe (a parent
                // gesture around embedded Buttons is what swallows child clicks).
                Text("Advanced")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                    .onTapGesture { showAdvancedTools.toggle() }
            }
        .onChange(of: showAdvancedTools) { _, expanded in
            // Seed from the coarse resolution so the two views can never
            // disagree at the moment Advanced opens.
            if expanded { agent.capabilities.openAdvanced() }
        }
    }

    private func toolBinding(_ tool: AgentToolKind) -> Binding<Bool> {
        Binding(
            get: { agent.capabilities.resolvedTools().contains(tool) },
            set: { on in
                agent.capabilities.openAdvanced()
                var set = agent.capabilities.advancedTools ?? []
                if on { set.insert(tool) } else { set.remove(tool) }
                agent.capabilities.advancedTools = set
            })
    }

    // MARK: Model

    private var modelSection: some View {
        AgentSection("Model") {
            AgentCard(spacing: 12) {
                modelPicker
                modelState
            }
        }
    }

    private var modelPicker: some View {
        AgentEditorRow("Model") {
            Picker("", selection: Binding(get: { agent.modelPath ?? "" },
                                          set: { agent.modelPath = $0.isEmpty ? nil : $0 })) {
                Text("Current").tag("")
                ForEach(appState.localModels.filter(\.isChatPickable), id: \.path) { model in
                    Text(model.name).tag(model.path)
                }
                ForEach(appState.server.lanModels(capability: "chat"), id: \.name) { info in
                    Text("\(info.name) (network)").tag(info.name)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 260)
            .disabled(readOnly)
        }
    }

    @ViewBuilder
    private var modelState: some View {
        switch appState.agentModelDecision(for: agent) {
        case .needsDownload(let path):
            HStack {
                Label("Not downloaded", systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange).font(.subheadline)
                Spacer(minLength: 8)
                Button("Open Model Browser") {
                    appState.showModels()
                }
                .controlSize(.small)
                .help("This agent can't answer until \((path as NSString).lastPathComponent) is on disk. Nothing is downloaded automatically.")
            }
        case .unavailable(let reason):
            Label(reason, systemImage: "wifi.slash").foregroundStyle(.orange).font(.subheadline)
        case .noChange, .load, .lan:
            Text("Selecting this agent loads its model; “Current” leaves whatever is running alone.")
                .font(.subheadline).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Workspace

    private var workspaceSection: some View {
        AgentSection("Workspace") {
            AgentCard(spacing: 12) {
                AgentEditorRow("Folder",
                               caption: "Where this agent's file and shell tools run. Several agents may share a folder.") {
                    Text(agent.workingDirectory ?? "App default")
                        .font(.subheadline.monospaced())
                        .lineLimit(1).truncationMode(.head)
                        .foregroundStyle(agent.workingDirectory == nil ? .secondary : .primary)
                }
                workspaceActions
            }
        }
    }

    private var workspaceActions: some View {
        HStack(spacing: 8) {
            Button("Choose…") {
                guard let picked = WorkspacePicker.pickDirectory() else { return }
                agent.workingDirectory = picked
                // Per-agent folders need per-agent bookmarks under the
                // App Sandbox — the global default's slot can't stand in.
                SecurityScopedBookmark.store(URL(fileURLWithPath: picked),
                                             name: SecurityScopedBookmark.agentWorkspaceName(agent.id))
                onSave()
            }
            .disabled(readOnly)
            if agent.workingDirectory != nil {
                Button("Reset") {
                    SecurityScopedBookmark.clear(name: SecurityScopedBookmark.agentWorkspaceName(agent.id))
                    agent.workingDirectory = nil
                    onSave()
                }
                .disabled(readOnly)
            }
            Spacer(minLength: 0)
        }
    }

    // MARK: Voice (rows, rendered inside Identity)

    @ViewBuilder
    private var voiceRows: some View {
        AgentEditorRow("Voice", caption: voiceCaption) {
            AgentVoiceMenu(voice: $agent.voice,
                           systemVoices: appState.voice.availableVoices,
                           clips: clips,
                           globalClipPath: appState.serverOptions.voiceClonePath,
                           globalClipLabel: appState.serverOptions.voiceCloneLabel,
                           cloneAvailable: ttsDownloaded,
                           onAddClip: { addVoiceClip() },
                           onRevealClips: { VoiceClipLibrary.revealInFinder() })
                .disabled(readOnly)
        }
        // An agent already pointing at a clip when the model is gone would
        // just quietly speak in the system voice — say so instead.
        if case .clone = agent.voice, !ttsDownloaded,
           let reason = VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: false) {
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .font(.subheadline).foregroundStyle(.orange)
        }
        voiceActions
    }

    /// Nil rather than a sentence when the agent has its own voice: the row's
    /// trailing control already names what will speak.
    private var voiceCaption: String? {
        agent.voice == nil ? "Speaks with the app's voice (Settings ▸ Voice)." : nil
    }

    private var voiceActions: some View {
        HStack(spacing: 8) {
            Button("Preview") { previewVoice() }
                .disabled(previewer.active != nil || agent.voice == nil)
            Button("Add Voice…") { addVoiceClip() }
                .disabled(readOnly)
                .help("Add a recording of a voice to clone. It's normalized and kept in ~/.mlx-serve/voice-clips so any agent can use it later.")
            if let error = previewer.error ?? clipError {
                Text(error).font(.subheadline).foregroundStyle(.orange)
            }
            Spacer(minLength: 0)
        }
    }

    /// Audition the selected voice. An uploaded clip PLAYS THE FILE — that's the
    /// question you're asking of a reference recording ("is this the right
    /// take?"), and it needs no model downloaded; Kokoro synthesizes a sample
    /// sentence, and the system voice speaks one.
    private func previewVoice() {
        switch agent.voice {
        case .kokoro(let v) where !v.trimmingCharacters(in: .whitespaces).isEmpty:
            previewer.preview(v)
        case .clone(let path) where !path.isEmpty:
            previewer.playClip(path: path)
        case .system(let id):
            previewer.stop()
            systemPreview.voiceIdentifier = id.isEmpty ? nil : id
            systemPreview.enqueue("Hi, this is how I'll sound.")
        case .kokoro, .clone, .none:
            onNotify("Pick a voice first.")
        }
    }

    private func addVoiceClip() {
        do {
            guard let clip = try VoiceClipLibrary.pickAndInstall() else { return }
            clipError = nil
            clips = VoiceClipLibrary.clips()
            agent.voice = .clone(clip.path)
            onSave()
            // Play it back straight away — the point of uploading is hearing that
            // the right file landed.
            previewer.playClip(path: clip.path)
        } catch {
            clipError = error.localizedDescription
        }
    }

    // MARK: Sampling

    // Every row shows its control: on App default it renders the app's saved
    // value grayed out, so the row says what the agent will actually run with
    // instead of hiding it. The checkbox sits to the RIGHT of the control.

    private static let topKPresets: [Int] = [0, 5, 10, 20, 40, 64, 100, 200, 500, 1000]
    /// Same grid as Settings ▸ Max Tokens; 0 = Auto (omit, server pegs to the
    /// remaining context).
    private static let maxTokensPresets: [Int] = [
        0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    ]

    /// Merge note: the seven controls are main's (#135 — an agent can pin its
    /// own sampling); the card and the row grammar are this branch's. They were
    /// written against a grouped `Form`, so each row moved from
    /// `LabeledContent` to `AgentEditorRow` — otherwise this card would be the
    /// one place in the editor where a label sits in a different column from
    /// every other label.
    private var samplingSection: some View {
        AgentSection("Sampling") {
            AgentCard(spacing: 12) {
                sliderRow("Temperature", value: $agent.temperature,
                          seed: appState.serverOptions.defaultTemperature, in: 0...1.5, step: 0.05,
                          ends: ("Focused", "Creative"))
                Divider()
                sliderRow("Top-p", value: $agent.topP,
                          seed: appState.serverOptions.defaultTopP, in: 0.1...1.0, step: 0.01,
                          ends: ("Focused", "Varied"))
                Divider()
                // 0 omits the field, so the model's own generation_config wins.
                presetRow("Top-k", value: $agent.topK, seed: appState.serverOptions.defaultTopK,
                          presets: Self.topKPresets, ends: ("Auto", "Wide"),
                          label: { $0 == 0 ? "Auto" : "\($0)" })
                Divider()
                sliderRow("Repeat penalty", value: $agent.repeatPenalty,
                          seed: appState.serverOptions.defaultRepeatPenalty, in: 1.0...2.0, step: 0.01,
                          ends: ("Off", "Strong"))
                Divider()
                sliderRow("Presence penalty", value: $agent.presencePenalty,
                          seed: appState.serverOptions.defaultPresencePenalty, in: 0.0...2.0, step: 0.01,
                          ends: ("Off", "Strong"))
                Divider()
                presetRow("Max tokens", value: $agent.maxTokens, seed: appState.maxTokens,
                          presets: Self.maxTokensPresets, ends: ("Auto", "Long"),
                          label: { $0 <= 0 ? "Auto" : Self.formatTokens($0) })
                Divider()
                reasoningEffortRow
            }
        }
    }

    private static func formatTokens(_ n: Int) -> String {
        n >= 1024 ? "\(n / 1024)K" : "\(n)"
    }

    /// The right-hand "Default" checkbox every sampling row carries: checked =
    /// nil (follow Settings, control grayed), unchecking seeds from the app's
    /// saved value so the control picks up exactly where the gray preview was.
    /// One word on purpose — "App default" wrapped onto two lines beside the
    /// sliders.
    private func appDefaultToggle<T>(value: Binding<T?>, seed: T) -> some View {
        Toggle("Default", isOn: Binding(
            get: { value.wrappedValue == nil },
            set: { value.wrappedValue = $0 ? nil : seed }))
            .toggleStyle(.checkbox)
            .disabled(readOnly)
    }

    private func sliderRow(_ title: String, value: Binding<Double?>, seed: Double,
                           in range: ClosedRange<Double>, step: Double,
                           ends: (String, String)) -> some View {
        let isDefault = value.wrappedValue == nil
        return AgentEditorRow(title, alignment: .center) {
            HStack(spacing: 10) {
                VStack(spacing: 1) {
                    Slider(value: Binding(get: { value.wrappedValue ?? seed },
                                          set: { value.wrappedValue = $0 }),
                           in: range, step: step)
                        .disabled(readOnly || isDefault)
                    endLabels(ends, dimmed: isDefault)
                }
                .frame(width: 160)
                Text(String(format: "%.2f", value.wrappedValue ?? seed))
                    .font(.caption.monospaced())
                    .foregroundStyle(isDefault ? .secondary : .primary)
                    .frame(width: Self.valueColumnWidth, alignment: .trailing)
                appDefaultToggle(value: value, seed: seed)
            }
        }
    }

    /// The value readout is a FIXED column: its text changes length while you
    /// drag ("Auto" → "5" → "1000"), and letting it resize reflows the HStack
    /// and moves the slider under the pointer mid-drag.
    private static let valueColumnWidth: CGFloat = 44

    /// Snapping preset slider (the Settings idiom) for values whose useful
    /// range spans orders of magnitude — a linear slider wastes its whole
    /// travel on the far end.
    private func presetRow(_ title: String, value: Binding<Int?>, seed: Int,
                           presets: [Int], ends: (String, String),
                           label: @escaping (Int) -> String) -> some View {
        let isDefault = value.wrappedValue == nil
        let effective = value.wrappedValue ?? seed
        return AgentEditorRow(title, alignment: .center) {
            HStack(spacing: 10) {
                VStack(spacing: 1) {
                    Slider(value: Binding(
                        get: { Double(Self.closestIndex(in: presets, to: effective)) },
                        set: { value.wrappedValue = presets[max(0, min(Int($0.rounded()), presets.count - 1))] }),
                           in: 0...Double(presets.count - 1), step: 1)
                        .disabled(readOnly || isDefault)
                    endLabels(ends, dimmed: isDefault)
                }
                .frame(width: 160)
                Text(label(effective))
                    .font(.caption.monospaced())
                    .foregroundStyle(isDefault ? .secondary : .primary)
                    .frame(width: Self.valueColumnWidth, alignment: .trailing)
                appDefaultToggle(value: value, seed: seed)
            }
        }
    }

    /// The one-word meaning of each end of a slider, tucked under the track.
    private func endLabels(_ ends: (String, String), dimmed: Bool) -> some View {
        HStack {
            Text(ends.0)
            Spacer()
            Text(ends.1)
        }
        .font(.caption2)
        .foregroundStyle(dimmed ? .tertiary : .secondary)
    }

    /// A stored value off the grid still positions the slider sensibly.
    private static func closestIndex(in presets: [Int], to value: Int) -> Int {
        presets.enumerated().min { abs($0.element - value) < abs($1.element - value) }?.offset ?? 0
    }

    /// Effort levels rather than a token field — the levels are the server's
    /// own `reasoning_effort` budgets (`AgentReasoningEffort`), stored as
    /// tokens so the wire stays `reasoning_budget`.
    private var reasoningEffortRow: some View {
        let isDefault = agent.reasoningBudget == nil
        let seedTokens = AgentReasoningEffort
            .nearest(to: appState.serverOptions.defaultReasoningBudget).budgetTokens
        return AgentEditorRow("Reasoning budget", alignment: .center) {
            HStack(spacing: 10) {
                Picker("", selection: Binding(
                    get: { AgentReasoningEffort.nearest(to: agent.reasoningBudget
                                                        ?? appState.serverOptions.defaultReasoningBudget) },
                    set: { agent.reasoningBudget = $0.budgetTokens })) {
                    ForEach(AgentReasoningEffort.allCases, id: \.self) { level in
                        Text(level.label).tag(level)
                    }
                }
                // A MENU picker, like the Model row above it. As a segmented
                // control this row rendered torn apart — a phantom ~110pt of
                // claimed height pushed a dead band into the card and the
                // Default checkbox out of the row, bare box, label gone (live
                // 2026-08-09 screenshot) — and neither `.fixedSize()`, an
                // explicit frame, nor sibling order changed it, while the same
                // segmented style renders fine in every other pane. Whatever
                // macOS 26 dislikes is specific to this card's subtree; the
                // menu style is the shape the card already proves out.
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .disabled(readOnly || isDefault)
                appDefaultToggle(value: $agent.reasoningBudget, seed: seedTokens)
            }
        }
    }

    // MARK: Tri-state helper

    private enum TriChoice: Hashable { case appDefault, on, off }

    private func tristate(_ binding: Binding<Bool?>) -> Binding<TriChoice> {
        Binding(
            get: {
                switch binding.wrappedValue {
                case .none: return .appDefault
                case .some(true): return .on
                case .some(false): return .off
                }
            },
            set: { choice in
                switch choice {
                case .appDefault: binding.wrappedValue = nil
                case .on: binding.wrappedValue = true
                case .off: binding.wrappedValue = false
                }
            })
    }
}

// MARK: - Voice picker (writes the AGENT, not the app settings)

/// Kokoro, your voices, system — bound to an AGENT's own voice, with an explicit
/// "App voice" entry for the nil case and the uploaded-clip library in between.
/// A `Menu` rather than a `Picker` because it also carries ACTIONS (add a clip,
/// open the folder), which a Picker can't hold.
private struct AgentVoiceMenu: View {
    @Binding var voice: AgentVoice?
    let systemVoices: [VoiceOption]
    let clips: [VoiceClipLibrary.Clip]
    let globalClipPath: String
    let globalClipLabel: String
    /// False when the Qwen3-TTS checkpoint isn't downloaded: clips are listed but
    /// not selectable, with the reason spelled out, rather than picking one and
    /// hearing the system voice (the audio-preset honesty rule).
    let cloneAvailable: Bool
    let onAddClip: () -> Void
    let onRevealClips: () -> Void

    /// The menu alone — the "Voice" label belongs to the `AgentEditorRow` this
    /// sits in, which also owns the caption under it. A `LabeledContent` here
    /// would draw a second label beside the row's own.
    var body: some View {
            Menu {
                choice("App voice", isOn: voice == nil) { voice = nil }
                Divider()
                Menu("Kokoro") {
                    ForEach(KokoroVoiceCatalog.grouped(), id: \.language) { group in
                        Menu(group.language) {
                            ForEach(group.voices, id: \.self) { v in
                                choice(KokoroVoiceCatalog.displayName(for: v),
                                       isOn: voice == .kokoro(v)) { voice = .kokoro(v) }
                            }
                        }
                    }
                }
                Menu("Your voices") {
                    if !globalClipPath.isEmpty {
                        choice(globalClipLabel.isEmpty ? "Settings clip" : "\(globalClipLabel) (Settings)",
                               isOn: voice == .clone(globalClipPath)) { voice = .clone(globalClipPath) }
                            .disabled(!cloneAvailable)
                    }
                    ForEach(clips) { clip in
                        choice(clip.name, isOn: voice == .clone(clip.path)) { voice = .clone(clip.path) }
                            .disabled(!cloneAvailable)
                    }
                    if clips.isEmpty && globalClipPath.isEmpty {
                        Text("No clips yet")
                    }
                    if let reason = VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: cloneAvailable) {
                        Text(reason)
                    }
                    Divider()
                    Button("Add Voice…", action: onAddClip)
                    Button("Open Voices Folder", action: onRevealClips)
                }
                Menu("System") {
                    ForEach(systemVoices) { v in
                        choice(v.displayName, isOn: voice == .system(v.id)) { voice = .system(v.id) }
                    }
                    if systemVoices.isEmpty { Text("No voices installed") }
                }
            } label: {
                Text(label)
            }
            .fixedSize()
    }

    /// One tickable row — a real Button so the whole row is the hit target.
    @ViewBuilder
    private func choice(_ title: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            if isOn { Label(title, systemImage: "checkmark") } else { Text(title) }
        }
    }

    /// What will actually speak, named — never a bare "Clone" for a file the user
    /// gave a name to.
    private var label: String {
        switch voice {
        case .none:
            return "App voice"
        case .kokoro(let v):
            return v.trimmingCharacters(in: .whitespaces).isEmpty
                ? "App voice" : KokoroVoiceCatalog.displayName(for: v)
        case .clone(let path):
            if path == globalClipPath {
                return globalClipLabel.isEmpty ? "Settings clip" : globalClipLabel
            }
            return VoiceClipLibrary.displayName(forPath: path)
        case .system(let id):
            return systemVoices.first { $0.id == id }?.displayName ?? "System voice"
        }
    }
}
