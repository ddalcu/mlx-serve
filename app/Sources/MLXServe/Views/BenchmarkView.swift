import SwiftUI

/// The Benchmarks window: climb the context ladder, keep your own history,
/// compare against what everyone else measured.
///
/// Three panes, because they answer three different questions — "how fast is
/// this?", "did my change help?", and "how does my Mac compare?".
///
/// A run measures the server EXACTLY as configured. There are no knobs here:
/// kv-quant, MTP, PLD and context are changed in Settings, and the effective
/// values are read from `/props` and recorded on every row so the board can
/// show how a number was achieved.
///
/// Picking a model loads it, so the settings card reads what a run will measure.
struct BenchmarkView: View {

    @EnvironmentObject private var appState: AppState
    @EnvironmentObject private var server: ServerManager

    @StateObject private var runner = BenchmarkRunner()

    @State private var pane: Pane = .run
    @State private var lastResults: [BenchmarkResult] = []
    @State private var history: [BenchmarkResult] = []
    @State private var community: [BenchmarkResult] = []
    @State private var communityLoading = false
    @State private var communityError: String?
    @State private var submitState: SubmitState = .idle
    @State private var runError: String?
    @State private var isRunning = false
    @State private var ranAtLeastOnce = false
    @State private var liveSettings: [String: String] = [:]
    @State private var historySelection: String?
    @State private var communitySelection: String?
    @State private var sheetSource: SheetSource?
    /// Free text recorded on every row of a run. Kept across launches so a
    /// name typed once stays.
    @AppStorage("benchmarkNote") private var note = ""
    /// The model to benchmark, by path. Empty = follow the tray's selection.
    @State private var pickedModelPath = ""
    @State private var loadingModel = false
    @State private var modelSettings: ModelSettingsRequest?
    @State private var confirmingClear = false

    /// Community filters. Machine defaults to THIS Mac once the board has a
    /// row for it — the point of the board is "what will I get".
    @State private var machineFilter = ""
    @State private var modelFilter = ""
    @State private var machineFilterSeeded = false
    @State private var communitySort: [BenchmarkFamilySort] = [BenchmarkFamilySort(.date, order: .reverse)]

    private let client = BenchmarkCommunityClient()
    private let api = APIClient()
    private let ladder = BenchmarkSuite.ladder

    /// Named `Pane`, not `Section` — a nested `Section` inside a View shadows
    /// SwiftUI's own and turns any later `Section { }` here into a baffling
    /// type error.
    enum Pane: String, CaseIterable, Identifiable {
        case run = "Run"
        case history = "History"
        case community = "Community"
        var id: String { rawValue }
    }

    enum SubmitState: Equatable {
        case idle, sending, sent, failed(String)
    }

    /// One sheet presentation per window; the item decides what it shows.
    struct SheetSource: Identifiable {
        let id: String
        let source: BenchmarkSessionSheet.Source
    }

    /// Read once. `benchmarkHardware()` iterates the IORegistry, and a view
    /// body re-evaluates constantly — this must never be called from one.
    private let hardware = SystemMetrics.benchmarkHardware()

    var body: some View {
        Group {
            switch pane {
            case .run: runPane
            case .history: historyPane
            case .community: communityPane
            }
        }
        .frame(minWidth: 940, minHeight: 600)
        .background(.background)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Picker("View", selection: $pane) {
                    ForEach(Pane.allCases) { pane in
                        Text(L10n.text(pane.rawValue)).tag(pane)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
        }
        .onAppear {
            history = BenchmarkStore.loadLocal()
            if pickedModelPath.isEmpty { pickedModelPath = appState.selectedModelPath }
        }
        // Every visit re-fetches: the board is small and a row shared a
        // minute ago must show up without hunting for Refresh.
        .task(id: pane) {
            if pane == .community { await loadCommunity() }
        }
        // Keyed on status AND model: a stop/start comes back with the same
        // model name, and a task keyed on the name alone never re-fired, so
        // the card sat on "Reading settings…" against a running server.
        .task(id: settingsRefreshKey) { await refreshSettings() }
        // Load on pick so the settings card shows what a run will measure.
        .onChange(of: pickedModelPath) { old, new in
            guard !old.isEmpty, !new.isEmpty, !isRunning else { return }
            Task { await loadPick() }
        }
        .sheet(item: $modelSettings, onDismiss: { Task { await refreshSettings() } }) {
            ModelSettingsSheet(request: $0)
                .environmentObject(appState)
                .environmentObject(server)
        }
        .sheet(item: $sheetSource) { item in
            BenchmarkSessionSheet(source: item.source)
                .environmentObject(appState)
                .environmentObject(server)
        }
    }

    // MARK: - Run

    private var runPane: some View {
        ScrollView {
            HStack(alignment: .top, spacing: 18) {
                VStack(spacing: 18) {
                    setupCard
                    settingsCard
                    runControl
                }
                .frame(width: 390)

                VStack(spacing: 18) {
                    if !visibleResults.isEmpty {
                        resultCard
                    } else if ranAtLeastOnce && !isRunning {
                        noUsableRunsCard
                    } else {
                        resultPlaceholder
                    }
                }
                .frame(maxWidth: .infinity, alignment: .top)
            }
            .padding(20)
        }
    }

    /// While running, the rungs completed so far; afterwards the whole session.
    private var visibleResults: [BenchmarkResult] {
        isRunning ? runner.completedRungs : lastResults
    }

    private var contextLength: Int? { server.residentChatModel?.contextLength }

    private var pickableModels: [LocalModel] { appState.localModels.filter { $0.isChatPickable } }

    private var pickedModel: LocalModel? {
        pickableModels.first { $0.path == pickedModelPath }
            ?? pickableModels.first { $0.path == appState.selectedModelPath }
            ?? pickableModels.first
    }

    /// The picked model is the one answering right now, so its advertised
    /// context and settings describe what a run would measure.
    private var pickedIsResident: Bool {
        server.status == .running && server.residentChatModel != nil
            && pickedModel?.path == appState.selectedModelPath
    }

    private var contextTooSmall: Bool {
        if case .contextTooSmall = preflight { return true }
        return false
    }

    private var settingsRefreshKey: String {
        "\(server.status.label)|\(server.residentChatModel?.name ?? "")|\(contextLength ?? 0)"
    }

    private var preflight: LadderPreflight {
        LadderPreflight.decide(contextLength: pickedIsResident ? contextLength : nil, ladder: ladder)
    }

    private var resultPlaceholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "chart.xyaxis.line")
                .font(.app(.largeTitle, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No result yet")
                .font(.app(.callout).weight(.medium))
                .foregroundStyle(.secondary)
            Text("Run the ladder to see decode and prefill speed at each context size.")
                .font(.app(.caption))
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 240)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 64)
        .background(.quaternary.opacity(0.14), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(.quaternary, style: StrokeStyle(lineWidth: 1, dash: [5, 4]))
        }
    }

    private var setupCard: some View {
        BenchCard("Setup", icon: "gearshape") {
            VStack(spacing: 0) {
                BenchRow("Model", detail: pickedIsResident || loadingModel ? nil : "Loaded when the run starts.") {
                    if pickableModels.isEmpty {
                        Text("No chat model on this Mac").foregroundStyle(.secondary)
                    } else {
                        HStack(spacing: 6) {
                            if loadingModel { ProgressView().controlSize(.small) }
                            Picker("Model", selection: $pickedModelPath) {
                                ForEach(pickableModels) { model in
                                    Text(model.name).tag(model.path)
                                }
                            }
                            .labelsHidden()
                            .frame(maxWidth: 220)
                            .disabled(loadingModel || isRunning)
                        }
                    }
                }
                Divider().padding(.vertical, 9)
                BenchRow("Context") {
                    Text(pickedIsResident ? (contextLength.map { ContextSizeDisplay.formatTokens($0) } ?? "—") : "—")
                }
                Divider().padding(.vertical, 9)
                BenchRow("This Mac") {
                    Text(hardware.displayName)
                }
                Divider().padding(.vertical, 9)
                BenchRow("Note", detail: "Anything you want on the row: your name, a nickname, what else was running.") {
                    TextField("Optional", text: $note)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 170)
                        .onChange(of: note) { _, new in
                            if new.count > BenchmarkResult.maxNoteLength { note = String(new.prefix(BenchmarkResult.maxNoteLength)) }
                        }
                }
            }
        }
    }

    private var settingsCard: some View {
        BenchCard("Server settings", icon: "slider.horizontal.3",
                  footnote: "A run measures the server exactly as configured. These are recorded with every result.") {
            VStack(alignment: .leading, spacing: 10) {
                if liveSettings.isEmpty {
                    // Settings are per MODEL: a running server with nothing
                    // resident publishes none, so "reading" would never end.
                    Text(loadingModel ? "Loading model…"
                         : server.status != .running ? "Server not running"
                         : server.residentChatModel == nil ? "No model loaded — settings are read from the loaded model."
                         : "Reading settings…")
                        .font(.app(.callout))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    BenchmarkSettingsChips(settings: liveSettings)
                }
                HStack {
                    Button("Change in Settings…") { appState.showSettings() }
                        .controlSize(.small)
                    Button("Model Settings…") {
                        guard let pick = pickedModel else { return }
                        modelSettings = ModelSettingsRequest(path: pick.path, title: ModelDisplayName.pretty(pick.displayLabel))
                    }
                    .controlSize(.small)
                    .disabled(pickedModel == nil || loadingModel || isRunning)
                    Button {
                        Task { await refreshSettings() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .controlSize(.small)
                    .help("Re-read the server's settings")
                }
            }
        }
    }

    private var runControl: some View {
        VStack(spacing: 12) {
            if case .contextTooSmall(let have, let need) = preflight {
                VStack(alignment: .leading, spacing: 8) {
                    Label("This model is serving \(ContextSizeDisplay.formatTokens(have)) of context; the ladder needs \(ContextSizeDisplay.formatTokens(need)). Raise it in Settings ▸ Context.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.app(.callout))
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Change in Settings…") { appState.showSettings() }
                        .controlSize(.small)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            if isRunning {
                VStack(spacing: 8) {
                    ProgressView(value: runner.progress.fraction)
                        .progressViewStyle(.linear)
                    HStack {
                        Text(runner.phase.localizedText)
                            .font(.app(.callout))
                            .foregroundStyle(.secondary)
                        Spacer()
                        if runner.discardedRuns > 0 {
                            Label("\(runner.discardedRuns) discarded", systemImage: "exclamationmark.triangle.fill")
                                .font(.app(.caption))
                                .foregroundStyle(.orange)
                        }
                        // Stops after the request in flight: a 16k prefill
                        // cannot be taken back from the server mid-forward.
                        Button(role: .destructive) {
                            runner.cancel()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .tint(.red)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(runner.phase == .stopping)
                    }
                }
            }

            if let runError {
                Label(runError, systemImage: "exclamationmark.octagon.fill")
                    .font(.app(.callout))
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Button {
                Task { await runBenchmark() }
            } label: {
                Label(isRunning ? "Running…" : "Run Benchmark", systemImage: "play.fill")
                    .padding(.horizontal, 8)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(isRunning || loadingModel || pickedModel == nil || contextTooSmall)

            if pickedModel == nil {
                Text("Download a chat model from the menu bar to run a benchmark.")
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            } else if !pickedIsResident {
                Text(server.status == .running ? "The model is loaded when the run starts."
                     : "The server is started with this model when the run starts.")
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var resultCard: some View {
        let rungs = visibleResults
        return BenchCard("Result", icon: "chart.xyaxis.line") {
            VStack(spacing: 16) {
                BenchmarkLadderChart(points: BenchmarkLadderChart.points(
                    decode: rungs.map { ($0.effectiveTargetTokens, $0.decodeTps) },
                    ceiling: rungs.map { ($0.effectiveTargetTokens, $0.ceilingDecodeTps ?? 0) }))
                BenchmarkPrefillChart(points: BenchmarkPrefillChart.points(
                    rungs.map { ($0.effectiveTargetTokens, $0.prefillTps) }))
                BenchmarkRungTable(rows: BenchmarkRungTable.rows(rungs))

                if !isRunning {
                    Divider()
                    HStack(alignment: .top, spacing: 12) {
                        switch submitState {
                        case .idle:
                            Button {
                                Task { await submit() }
                            } label: {
                                Label("Share to Community", systemImage: "square.and.arrow.up")
                            }
                            .controlSize(.regular)
                        case .sending:
                            ProgressView().controlSize(.small)
                        case .sent:
                            Label("Shared", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        case .failed(let message):
                            // Retry sends only the rows that did not land.
                            VStack(alignment: .leading, spacing: 4) {
                                Label(message, systemImage: "exclamationmark.octagon.fill")
                                    .font(.app(.caption))
                                    .foregroundStyle(.red)
                                Button("Try Again") { Task { await submit() } }
                                    .controlSize(.small)
                            }
                        }
                        Spacer(minLength: 0)
                        Text("Sends these numbers plus the server settings, your chip, GPU cores, memory and macOS version. No account, nothing identifying.")
                            .font(.app(.caption2))
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.trailing)
                            .frame(maxWidth: 260)
                    }
                }
            }
        }
    }

    private var noUsableRunsCard: some View {
        BenchCard("Result", icon: "chart.xyaxis.line") {
            VStack(alignment: .leading, spacing: 6) {
                Label("No usable runs", systemImage: "exclamationmark.triangle.fill")
                    .font(.app(.callout).weight(.medium))
                    .foregroundStyle(.orange)
                Text("Every coding run was served from the KV cache, so nothing was actually measured. Restarting the server clears it.")
                    .font(.app(.caption))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - History

    private var historySessions: [BenchmarkSession] { BenchmarkSession.group(history) }

    private var historyPane: some View {
        Group {
            if history.isEmpty {
                ContentUnavailableView {
                    Label("No Runs Yet", systemImage: "clock")
                } description: {
                    Text("Benchmarks you run are kept here, on this Mac. Double-click a row for the full detail.")
                } actions: {
                    Button("Run a Benchmark") { pane = .run }
                        .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 0) {
                    Table(historySessions, selection: $historySelection) {
                        TableColumn("Date") { session in
                            Text(session.date, format: .dateTime.month(.abbreviated).day().hour().minute())
                                .foregroundStyle(.secondary)
                        }
                        .width(min: 100, ideal: 120)

                        TableColumn("Model") { session in
                            Text(session.modelId).lineLimit(1).truncationMode(.middle)
                        }
                        .width(min: 140, ideal: 220)

                        TableColumn("Settings") { session in
                            BenchmarkSettingsChips(settings: session.settings)
                        }
                        .width(min: 160, ideal: 220)

                        TableColumn("Note") { session in
                            Text(session.note ?? "").lineLimit(1).foregroundStyle(.secondary)
                        }
                        .width(min: 60, ideal: 120)

                        TableColumnForEach(ladder) { rung in
                            TableColumn(rung.title) { session in
                                rateCell(session.decode(at: rung.targetTokens))
                            }
                            .width(min: 50, ideal: 60)
                        }
                    }
                    .tableStyle(.inset(alternatesRowBackgrounds: true))
                    .contextMenu(forSelectionType: String.self) { ids in
                        if let id = ids.first, let session = historySessions.first(where: { $0.id == id }) {
                            Button("Show Details") { sheetSource = SheetSource(id: id, source: .session(session)) }
                        }
                    } primaryAction: { ids in
                        if let id = ids.first, let session = historySessions.first(where: { $0.id == id }) {
                            sheetSource = SheetSource(id: id, source: .session(session))
                        }
                    }

                    footerBar {
                        Text("\(historySessions.count) session\(historySessions.count == 1 ? "" : "s") · decode tok/s per rung · double-click for detail")
                        Spacer()
                        Button(role: .destructive) {
                            confirmingClear = true
                        } label: {
                            Label("Clear", systemImage: "trash")
                        }
                        .controlSize(.small)
                        .confirmationDialog("Delete every benchmark run kept on this Mac?",
                                            isPresented: $confirmingClear, titleVisibility: .visible) {
                            Button("Delete History", role: .destructive) {
                                BenchmarkStore.saveLocal([])
                                history = []
                            }
                        } message: {
                            Text("Results already shared to the community stay on the board.")
                        }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
        }
    }

    private func rateCell(_ value: Double?) -> some View {
        Text(BenchmarkFormat.rate(value ?? 0, decimals: 1))
            .font(.app(.callout))
            .monospacedDigit()
            .fontWeight(value == nil ? .regular : .medium)
            .foregroundStyle(value == nil ? AnyShapeStyle(.tertiary) : AnyShapeStyle(.primary))
    }

    // MARK: - Community

    private var communityPane: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Picker("Machine", selection: $machineFilter) {
                    Text("All machines").tag("")
                    ForEach(communityMachines, id: \.self) { Text($0).tag($0) }
                }
                .frame(maxWidth: 260)
                Picker("Model", selection: $modelFilter) {
                    Text("All models").tag("")
                    ForEach(communityModels, id: \.self) { Text($0).tag($0) }
                }
                .frame(maxWidth: 320)
                Spacer()
                Button {
                    Task { await loadCommunity() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .controlSize(.small)
                .disabled(communityLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)

            Divider()

            if communityLoading {
                ProgressView("Loading results…")
                    .controlSize(.small)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let communityError {
                ContentUnavailableView {
                    Label("Couldn't Load Results", systemImage: "wifi.exclamationmark")
                } description: {
                    Text(communityError)
                } actions: {
                    Button("Try Again") { Task { await loadCommunity() } }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if visibleFamilies.isEmpty {
                ContentUnavailableView {
                    Label("Nothing Here Yet", systemImage: "person.2")
                } description: {
                    Text(machineFilter.isEmpty && modelFilter.isEmpty
                         ? "Be the first to share a result."
                         : "Nothing matches those filters yet.")
                } actions: {
                    if !machineFilter.isEmpty || !modelFilter.isEmpty {
                        Button("Show Everything") { machineFilter = ""; modelFilter = "" }
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Table(visibleFamilies, selection: $communitySelection, sortOrder: $communitySort) {
                    TableColumn("Latest", sortUsing: BenchmarkFamilySort(.date)) { family in
                        Text(family.latestDate, format: .dateTime.month(.abbreviated).day().hour().minute())
                            .foregroundStyle(.secondary)
                    }
                    .width(min: 100, ideal: 120)

                    TableColumn("Machine", sortUsing: BenchmarkFamilySort(.machine)) { family in
                        Text(family.hardware.displayName).lineLimit(1)
                    }
                    .width(min: 150, ideal: 200)

                    TableColumn("Model", sortUsing: BenchmarkFamilySort(.model)) { family in
                        Text(family.modelId).lineLimit(1).truncationMode(.middle)
                    }
                    .width(min: 140, ideal: 200)

                    TableColumn("Settings") { family in
                        BenchmarkSettingsChips(settings: family.settings)
                    }
                    .width(min: 160, ideal: 220)

                    TableColumnForEach(ladder) { rung in
                        TableColumn(rung.title, sortUsing: BenchmarkFamilySort(.rung(rung.targetTokens))) { family in
                            rateCell(family.decode(at: rung.targetTokens))
                        }
                        .width(min: 50, ideal: 60)
                    }

                    // n is never hidden: a cell built from one submission is a
                    // data point, not a benchmark.
                    TableColumn("n", sortUsing: BenchmarkFamilySort(.sessions)) { family in
                        Text("\(family.sessionCount)")
                            .font(.app(.callout))
                            .monospacedDigit()
                            .foregroundStyle(family.sessionCount == 1 ? .orange : .secondary)
                    }
                    .width(min: 28, ideal: 36)
                }
                .tableStyle(.inset(alternatesRowBackgrounds: true))
                .contextMenu(forSelectionType: String.self) { ids in
                    if let id = ids.first, let family = visibleFamilies.first(where: { $0.id == id }) {
                        Button("Show Details") { sheetSource = SheetSource(id: id, source: .family(family)) }
                    }
                } primaryAction: { ids in
                    if let id = ids.first, let family = visibleFamilies.first(where: { $0.id == id }) {
                        sheetSource = SheetSource(id: id, source: .family(family))
                    }
                }

                footerBar {
                    Text("Median decode tok/s per rung, per machine + model + settings. n = sessions behind each row. Double-click for detail.")
                    Spacer()
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var communityMachines: [String] {
        Array(Set(community.map { $0.hardware.displayName })).sorted()
    }

    private var communityModels: [String] {
        Array(Set(community.map(\.modelId))).sorted()
    }

    private var visibleFamilies: [BenchmarkStore.CellFamily] {
        let rows = community.filter {
            (machineFilter.isEmpty || $0.hardware.displayName == machineFilter)
                && (modelFilter.isEmpty || $0.modelId == modelFilter)
        }
        return BenchmarkStore.aggregateSessions(rows).sorted(using: communitySort)
    }

    // MARK: - Chrome

    private func footerBar<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        HStack(spacing: 8) {
            content()
        }
        .font(.app(.caption))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(.bar)
        .overlay(alignment: .top) { Divider() }
    }

    // MARK: - Actions

    private func refreshSettings() async {
        guard server.status == .running, let model = server.residentChatModel?.name else {
            liveSettings = [:]
            return
        }
        let props = (try? await api.fetchPropsRaw(port: server.port, model: model)) ?? [:]
        liveSettings = BenchmarkSettings.flatten(props: props)
    }

    /// Make the picked model the resident chat model: start the server with
    /// it, hot-switch to it, or hot-load it into a headless server. Returns
    /// the model the server now answers with.
    private func loadPickedModel() async -> ModelInfo? {
        guard let pick = pickedModel else { return nil }
        if server.status != .running || appState.selectedModelPath != pick.path {
            guard await appState.useModelAndAwaitReady(atPath: pick.path) else { return nil }
        }
        // Ready means /health answered, not that the model list caught up; and a
        // `--model` launch skips `ensureDefaultChatModel`, so load it directly.
        await server.refreshModels()
        if server.residentChatModel == nil {
            _ = try? await server.loadModel(id: pick.path)
        }
        return server.residentChatModel
    }

    private func loadPick() async {
        loadingModel = true
        runError = nil
        defer { loadingModel = false }
        if await loadPickedModel() == nil {
            runError = L10n.text("The model could not be loaded. Check the server log.")
        }
        await refreshSettings()
    }

    private func runBenchmark() async {
        guard pickedModel != nil else { return }
        isRunning = true
        runError = nil
        submitState = .idle
        ranAtLeastOnce = true
        lastResults = []
        defer { isRunning = false }

        guard let resident = await loadPickedModel() else {
            runError = L10n.text("The model could not be loaded. Check the server log.")
            return
        }
        if case .contextTooSmall(let have, let need) = LadderPreflight.decide(contextLength: resident.contextLength, ladder: ladder) {
            runError = L10n.format("This model is serving %@ of context; the ladder needs %@. Raise it in Settings ▸ Context.",
                                   ContextSizeDisplay.formatTokens(have), ContextSizeDisplay.formatTokens(need))
            return
        }
        let model = resident.name
        await refreshSettings()

        let results = await runner.run(ladder: ladder, modelId: model, port: server.port,
                                       note: BenchmarkResult.cleanNote(note), hardware: hardware)
        if case .failed(let message) = runner.phase { runError = message }
        if case .cancelled = runner.phase, !results.isEmpty {
            runError = L10n.format("Stopped after %lld rung%@; the drift check was skipped.",
                                   Int64(results.count), results.count == 1 ? "" : "s")
        }
        lastResults = results
        if !results.isEmpty { history = BenchmarkStore.appendLocal(results) }
        await refreshSettings()
    }

    private func submit() async {
        submitState = .sending
        let outcome = await client.submit(BenchmarkStore.unsent(lastResults))
        BenchmarkStore.markShared(outcome.sentIds)
        if let error = outcome.error {
            submitState = .failed(error.localizedDescription)
        } else {
            submitState = .sent
            community = []   // stale until the next visit re-fetches
        }
    }

    private func loadCommunity() async {
        communityLoading = true
        communityError = nil
        defer { communityLoading = false }
        do {
            community = try await client.fetch()
            // First load only: land on this Mac when the board knows it.
            if !machineFilterSeeded {
                machineFilterSeeded = true
                if communityMachines.contains(hardware.displayName) { machineFilter = hardware.displayName }
            }
        } catch {
            communityError = error.localizedDescription
        }
    }
}
