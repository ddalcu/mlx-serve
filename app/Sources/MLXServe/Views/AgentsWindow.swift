import SwiftUI
import AppKit

/// The Agents window: saved personas on the left, the selected one's editor on
/// the right. Configuration only — you TALK to an agent in the Chat window (or
/// the tray, or a task); this is where you decide who they are.
///
/// Everything an agent owns is an override of an app default, so every control
/// here has a "use the app's setting" state. The one thing deliberately absent is
/// the Agent Sandbox: that stays a single global flag.
/// Which agent the window should be showing.
///
/// Pure because the interesting part is a three-way precedence, and the window
/// is a single reused `Window`: a deep link ("Set by Chef · Edit Agent…" on a
/// locked composer disc) has to retarget one that is already open on somebody
/// else, while a plain re-publish must not yank the user's selection back to the
/// top of the list mid-edit. Tested in `AgentsWindowFocusTests`.
enum AgentsWindowFocus {
    /// The id to select, or nil to leave the selection exactly as it is.
    static func selection(pending: UUID?, current: UUID?, first: UUID?) -> UUID? {
        if let pending { return pending }
        return current == nil ? first : nil
    }
}

struct AgentsWindow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var store: AgentStore

    @State private var selectedId: UUID?
    /// The row being edited, held locally so typing doesn't rewrite the JSON on
    /// every keystroke; committed on change of focus/selection and on Save.
    @State private var draft: Agent?
    @State private var alertItem: AlertItem?
    @State private var isWriting = false

    /// ONE alert presentation path for this window — an alert modifier on an
    /// ancestor shadows a descendant's, so a second one would silently never
    /// present (the sandbox ✕-confirm class).
    private struct AlertItem: Identifiable {
        enum Kind { case message(String), confirmDelete(Agent) }
        let id = UUID()
        let title: String
        let kind: Kind
    }

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            if let draft {
                AgentEditor(agent: bindingToDraft(draft),
                            isWriting: $isWriting,
                            onWrite: { writePrompt() },
                            onSave: { commit() },
                            onDuplicate: { duplicate(draft) },
                            onDelete: { alertItem = AlertItem(title: "Delete “\(draft.name)”?",
                                                              kind: .confirmDelete(draft)) },
                            onNotify: { alertItem = AlertItem(title: $0, kind: .message($0)) })
                    .environmentObject(appState)
            } else {
                ContentUnavailableView("No agent selected",
                                       systemImage: "person.crop.circle.badge.questionmark",
                                       description: Text("Pick an agent, or create one from a description."))
            }
        }
        .onChange(of: selectedId) { _, newValue in
            commit()
            draft = store.agent(id: newValue)
        }
        .onAppear {
            AppActivation.focus()
            applyFocus()
        }
        // The window is a single reused instance, so a deep link that arrives
        // while it is already open has to move the selection — `onAppear` alone
        // would leave the user staring at whoever they were editing.
        .onChange(of: appState.pendingAgentSelection) { _, _ in applyFocus() }
        .alert(item: $alertItem) { item in
            switch item.kind {
            case .message(let text):
                return Alert(title: Text("Agents"), message: Text(text),
                             dismissButton: .default(Text("OK")))
            case .confirmDelete(let agent):
                return Alert(title: Text(item.title),
                             message: Text("This can't be undone."),
                             primaryButton: .destructive(Text("Delete")) {
                                 store.delete(id: agent.id)
                                 selectedId = store.allAgents.first?.id
                                 draft = store.agent(id: selectedId)
                             },
                             secondaryButton: .cancel())
            }
        }
    }

    // MARK: Sidebar

    private var sidebar: some View {
        List(selection: $selectedId) {
            Section("Your agents") {
                if store.sortedAgents.isEmpty {
                    Text("None yet — tap + and describe the assistant you want.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .listRowSeparator(.hidden)
                }
                ForEach(store.sortedAgents) { agent in
                    AgentRow(agent: agent, decision: appState.agentModelDecision(for: agent))
                        .tag(agent.id)
                }
            }
            Section("Starters") {
                ForEach(Agent.starters) { agent in
                    AgentRow(agent: agent, decision: appState.agentModelDecision(for: agent))
                        .tag(agent.id)
                }
            }
        }
        .navigationSplitViewColumnWidth(min: 200, ideal: 240)
        .toolbar {
            ToolbarItem {
                Button { newAgent() } label: { Image(systemName: "plus") }
                    .help("New agent")
            }
        }
    }

    // MARK: Actions

    private func bindingToDraft(_ current: Agent) -> Binding<Agent> {
        Binding(get: { draft ?? current }, set: { draft = $0 })
    }

    /// Land on whoever was asked for, then consume the request.
    private func applyFocus() {
        guard let id = AgentsWindowFocus.selection(pending: appState.pendingAgentSelection,
                                                   current: selectedId,
                                                   first: store.allAgents.first?.id) else { return }
        appState.pendingAgentSelection = nil
        // A deep link to the agent ALREADY showing can't rely on
        // `onChange(of: selectedId)` — the id doesn't change, so the draft has
        // to be reloaded here or the click does nothing at all.
        if selectedId == id {
            draft = store.agent(id: id)
        } else {
            selectedId = id
        }
    }

    private func newAgent() {
        commit()
        let agent = Agent(name: "New Agent", brief: "", systemPrompt: "")
        store.add(agent)
        selectedId = agent.id
        draft = agent
    }

    private func duplicate(_ agent: Agent) {
        let copy = store.duplicate(agent)
        selectedId = copy.id
        draft = copy
    }

    /// Write the draft back to the store. Wake-phrase collisions are refused
    /// HERE, at save time: a colliding phrase makes both agents unreachable by
    /// voice and there is nothing to see until you try talking.
    private func commit() {
        guard var d = draft, !d.isBuiltIn else { return }
        if let phrase = d.wakePhrase,
           WakeWord.collides(phrase, with: store.takenWakePhrases(excluding: d.id)) {
            d.wakePhrase = nil
            draft = d
            alertItem = AlertItem(
                title: "That wake phrase is taken",
                kind: .message("Another agent (or the app's own phrase) already answers to that name, so both would be unreachable. Pick a different one."))
        }
        store.update(d)
        // A live tab talking to this agent picks the change up on its next turn;
        // its voice applies from the next sentence.
        if appState.defaultAgentId == d.id { ActiveAgentVoice.set(d.resolvedVoice) }
    }

    /// Ask the current model to turn the brief into a system prompt. A failure
    /// falls back to the user's own words rather than losing what they typed.
    private func writePrompt() {
        guard var d = draft, !d.isBuiltIn else { return }
        let brief = d.brief.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !brief.isEmpty else {
            alertItem = AlertItem(title: "Describe the agent first",
                                  kind: .message("Write a line or two about the assistant you want, then let the model turn it into a prompt."))
            return
        }
        isWriting = true
        Task {
            defer { isWriting = false }
            let result: AgentWriter.Draft
            do {
                result = try await AgentComposer.draftAgent(brief: brief, appState: appState)
            } catch {
                result = AgentWriter.fallbackDraft(brief: brief)
                alertItem = AlertItem(title: "Wrote it from your description",
                                      kind: .message("\(error.localizedDescription)\n\nYour description was saved as the prompt — edit it directly, or try again once a model is running."))
            }
            d.systemPrompt = result.systemPrompt
            if d.name.isEmpty || d.name == "New Agent" { d.name = result.name }
            if d.symbol == "sparkles" { d.symbol = AgentSymbol.pick(for: "\(brief) \(result.name)") }
            draft = d
            commit()
        }
    }
}

// MARK: - Sidebar row

private struct AgentRow: View {
    let agent: Agent
    let decision: AgentModelSwitch.Decision

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: agent.symbol)
                .font(.system(size: 12))
                .frame(width: 18)
                .foregroundStyle(AgentModelSwitch.isSelectable(decision) ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(agent.name).font(.subheadline).lineLimit(1)
                if let sub = subtitle {
                    Text(sub).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
            }
            Spacer(minLength: 0)
            if agent.isBuiltIn {
                Image(systemName: "lock").font(.caption2).foregroundStyle(.tertiary)
                    .help("Built-in — duplicate it to make changes")
            }
        }
        .opacity(AgentModelSwitch.isSelectable(decision) ? 1 : 0.55)
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
        Form {
            if readOnly {
                Section {
                    HStack(spacing: 8) {
                        Image(systemName: "lock.fill").foregroundStyle(.secondary)
                        Text("This is one of the built-in agents. Duplicate it to make it yours.")
                            .font(.callout)
                        Spacer()
                        Button("Duplicate", action: onDuplicate)
                    }
                }
            }
            identitySection
            promptSection
            // Everything below is collapsed by default. An agent is a prompt, a
            // name and a voice; capabilities, a pinned model, a workspace and
            // sampling are real but rarely-touched, and putting five sections of
            // them between "what should this be?" and the Delete button made the
            // editor read as a settings panel. The row names whatever is set
            // behind it (`AgentAdvancedSummary`) so a collapsed non-default is
            // still discoverable.
            moreOptionsSection
            if showMoreOptions {
                capabilitiesSection
                modelSection
                workspaceSection
                samplingSection
            }
            if !readOnly {
                Section {
                    HStack {
                        Button("Duplicate", action: onDuplicate)
                        Spacer()
                        Button("Delete", role: .destructive, action: onDelete)
                    }
                }
            }
        }
        .formStyle(.grouped)
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

    // MARK: Identity

    private var identitySection: some View {
        Section("Identity") {
            TextField("Name", text: $agent.name)
                .disabled(readOnly)
            Picker("Symbol", selection: $agent.symbol) {
                ForEach(AgentSymbol.pickerChoices, id: \.self) { symbol in
                    Label(symbol, systemImage: symbol).labelStyle(.iconOnly).tag(symbol)
                }
            }
            .pickerStyle(.menu)
            .disabled(readOnly)
            LabeledContent("Wake phrase") {
                VStack(alignment: .leading, spacing: 2) {
                    // `prompt:`, not the title argument — a TextField's title is a
                    // LABEL, so passing the app phrase there parked it beside the
                    // field permanently instead of showing through an empty one.
                    TextField("", text: Binding(get: { agent.wakePhrase ?? "" },
                                                set: { agent.wakePhrase = $0.isEmpty ? nil : $0 }),
                              prompt: Text(appPhraseDisplay))
                        .disabled(readOnly)
                    Text("Say this to hand the conversation to \(agent.name). Blank uses the app's own phrase.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
            // Voice belongs to identity, not to a section of its own: how an
            // agent SOUNDS is the same kind of fact as what it's called and what
            // wakes it, and all three are what you set when making one.
            voiceRows
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
    ///
    /// A Button rather than a `DisclosureGroup`: the four things it reveals are
    /// `Section`s, and nesting sections inside a disclosure loses their headers
    /// and their grouped-form styling. Toggling a flag that gates them keeps
    /// each section exactly as it was.
    private var moreOptionsSection: some View {
        Section {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { showMoreOptions.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: showMoreOptions ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text("More options")
                    Spacer()
                    // What's set behind the row while it's shut, so a collapsed
                    // non-default isn't a setting nobody can find again.
                    if !showMoreOptions, let summary = AgentAdvancedSummary.text(for: agent) {
                        Text(summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Capabilities, model, workspace and sampling. Most agents need none of these.")
        }
    }

    // MARK: Prompt

    private var promptSection: some View {
        Section("Prompt") {
            VStack(alignment: .leading, spacing: 6) {
                Text("What should this agent be?").font(.caption).foregroundStyle(.secondary)
                // Same `prompt:` rule as the wake phrase: an example handed to the
                // title argument becomes a permanent LABEL beside the field.
                TextField("", text: $agent.brief,
                          prompt: Text("e.g. a blunt Swift code reviewer that never comments on style"),
                          axis: .vertical)
                    .lineLimit(2...4)
                    .disabled(readOnly)
                HStack {
                    Button {
                        onWrite()
                    } label: {
                        if isWriting {
                            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Writing…") }
                        } else {
                            Text("Write it for me")
                        }
                    }
                    .disabled(readOnly || isWriting)
                    .help("Ask the current model to turn your description into a system prompt. You can edit whatever it writes.")
                    Spacer()
                    Text("\(agent.systemPrompt.count)/\(AgentWriter.maxPromptCharacters)")
                        .font(.caption2).foregroundStyle(.tertiary)
                }
            }
            TextEditor(text: $agent.systemPrompt)
                .font(.body)
                .frame(minHeight: 120)
                .disabled(readOnly)
        }
    }

    // MARK: Capabilities

    private var capabilitiesSection: some View {
        Section("Capabilities") {
            Toggle("Tools", isOn: Binding(get: { agent.capabilities.tools },
                                          set: { agent.capabilities.tools = $0 }))
                .disabled(readOnly || showAdvancedTools)
                .help("The tool-calling loop: shell, files, search, tasks, media generation.")
            Toggle("MCP", isOn: Binding(get: { agent.capabilities.mcp },
                                        set: { agent.capabilities.mcp = $0 }))
                .disabled(readOnly)
                .help("Add the tools from every enabled Model Context Protocol server.")
            Toggle("Web", isOn: Binding(get: { agent.capabilities.web },
                                        set: { agent.capabilities.web = $0 }))
                .disabled(readOnly || showAdvancedTools)
                .help("Browse pages and search the web (browse + webSearch).")
            Picker("Thinking", selection: tristate($agent.enableThinking)) {
                Text("App default").tag(TriChoice.appDefault)
                Text("On").tag(TriChoice.on)
                Text("Off").tag(TriChoice.off)
            }
            .disabled(readOnly)
            Picker("Approve tools", selection: tristate($agent.autoApproveTools)) {
                Text("App default").tag(TriChoice.appDefault)
                Text("Automatically").tag(TriChoice.on)
                Text("Ask every time").tag(TriChoice.off)
            }
            .disabled(readOnly)

            DisclosureGroup(isExpanded: $showAdvancedTools) {
                Text("Pick exactly which tools this agent may call. Turning this on freezes the coarse switches above.")
                    .font(.caption2).foregroundStyle(.secondary)
                ForEach(AgentToolKind.allCases.filter { $0 != .searchDocuments }, id: \.self) { tool in
                    Toggle(isOn: toolBinding(tool)) {
                        Label(tool.displayName, systemImage: tool.icon)
                    }
                    .disabled(readOnly || !showAdvancedTools)
                }
                if showAdvancedTools {
                    Button("Back to the simple switches") {
                        agent.capabilities.closeAdvanced()
                        showAdvancedTools = false
                    }
                    .disabled(readOnly)
                }
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
        Section("Model") {
            Picker("Model", selection: Binding(get: { agent.modelPath ?? "" },
                                               set: { agent.modelPath = $0.isEmpty ? nil : $0 })) {
                Text("Current").tag("")
                ForEach(appState.localModels.filter(\.isChatPickable), id: \.path) { model in
                    Text(model.name).tag(model.path)
                }
                ForEach(appState.server.lanModels(capability: "chat"), id: \.name) { info in
                    Text("\(info.name) (network)").tag(info.name)
                }
            }
            .disabled(readOnly)
            let decision = appState.agentModelDecision(for: agent)
            switch decision {
            case .needsDownload(let path):
                HStack {
                    Label("Not downloaded", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange).font(.caption)
                    Spacer()
                    Button("Open Model Browser") {
                        appState.pendingModelBrowserOpenTick += 1
                    }
                    .controlSize(.small)
                    .help("This agent can't answer until \((path as NSString).lastPathComponent) is on disk. Nothing is downloaded automatically.")
                }
            case .unavailable(let reason):
                Label(reason, systemImage: "wifi.slash").foregroundStyle(.orange).font(.caption)
            case .noChange, .load, .lan:
                Text("Selecting this agent loads its model; “Current” leaves whatever is running alone.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: Workspace

    private var workspaceSection: some View {
        Section("Workspace") {
            LabeledContent("Folder") {
                HStack(spacing: 8) {
                    Text(agent.workingDirectory ?? "App default")
                        .font(.caption.monospaced())
                        .lineLimit(1).truncationMode(.head)
                        .foregroundStyle(agent.workingDirectory == nil ? .secondary : .primary)
                    Spacer()
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
                }
            }
            Text("Where this agent's file and shell tools run. Several agents may share a folder.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: Voice (rows, rendered inside Identity)

    @ViewBuilder
    private var voiceRows: some View {
        AgentVoiceMenu(voice: $agent.voice,
                       systemVoices: appState.voice.availableVoices,
                       clips: clips,
                       globalClipPath: appState.serverOptions.voiceClonePath,
                       globalClipLabel: appState.serverOptions.voiceCloneLabel,
                       cloneAvailable: ttsDownloaded,
                       onAddClip: { addVoiceClip() },
                       onRevealClips: { VoiceClipLibrary.revealInFinder() })
            .disabled(readOnly)
        // An agent already pointing at a clip when the model is gone would
        // just quietly speak in the system voice — say so instead.
        if case .clone = agent.voice, !ttsDownloaded,
           let reason = VoiceCloneMenuModel.cloneUnavailableReason(ttsModelDownloaded: false) {
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .font(.caption).foregroundStyle(.orange)
        }
        HStack(spacing: 10) {
            Button("Preview") { previewVoice() }
                .disabled(previewer.active != nil || agent.voice == nil)
            Button("Add Voice…") { addVoiceClip() }
                .disabled(readOnly)
                .help("Add a recording of a voice to clone. It's normalized and kept in ~/.mlx-serve/voice-clips so any agent can use it later.")
            if let error = previewer.error ?? clipError {
                Text(error).font(.caption2).foregroundStyle(.orange)
            }
            Spacer()
        }
        if agent.voice == nil {
            Text("Speaks with the app's voice (Settings ▸ Voice).")
                .font(.caption2).foregroundStyle(.secondary)
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

    private var samplingSection: some View {
        Section("Sampling") {
            LabeledContent("Temperature") {
                HStack(spacing: 10) {
                    Toggle("App default", isOn: Binding(
                        get: { agent.temperature == nil },
                        set: { agent.temperature = $0 ? nil : appState.serverOptions.defaultTemperature }))
                        .toggleStyle(.checkbox)
                        .disabled(readOnly)
                    if let value = agent.temperature {
                        Slider(value: Binding(get: { value }, set: { agent.temperature = $0 }),
                               in: 0...1.5, step: 0.05)
                            .frame(width: 160)
                            .disabled(readOnly)
                        Text(String(format: "%.2f", value)).font(.caption.monospaced())
                    }
                }
            }
            LabeledContent("Max tokens") {
                HStack(spacing: 10) {
                    Toggle("App default", isOn: Binding(
                        get: { agent.maxTokens == nil },
                        set: { agent.maxTokens = $0 ? nil : appState.maxTokens }))
                        .toggleStyle(.checkbox)
                        .disabled(readOnly)
                    if let value = agent.maxTokens {
                        TextField("", value: Binding(get: { value }, set: { agent.maxTokens = $0 }),
                                  format: .number)
                            .frame(width: 90)
                            .disabled(readOnly)
                    }
                }
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
///
/// There were two of these; the voice-mode sheet's copy (`VoiceSelectorMenu`)
/// is gone. The speaking voice belongs to WHO is answering, so it is set here
/// for an agent and in Settings ▸ Voice for the app itself — a third picker
/// inside the orb could only disagree with the two that own it, and the
/// synthesizer re-reads the value per utterance anyway.
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

    var body: some View {
        LabeledContent("Voice") {
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
