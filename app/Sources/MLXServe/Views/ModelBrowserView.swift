import SwiftUI

/// Model Browser: five sections (`ModelBrowserSection`), switched by the bar
/// across the top of the content area.
struct ModelBrowserPane: View {
    /// Which section is showing. The pane's own bar drives it: the sidebar is
    /// the conversation list (plus the Models row that gets you here), so the
    /// sub-items live across the top of the content area.
    @Binding var section: ModelBrowserSection

    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    @State private var localFilter = ""

    /// Downloading *or* failed — both belong in the queue and both earn a badge.
    private var activeDownloads: [(repoId: String, state: DownloadManager.DownloadState)] {
        downloads.downloads
            .filter { $0.value.status == .downloading || $0.value.status == .failed }
            .sorted { $0.key < $1.key }
            .map { (repoId: $0.key, state: $0.value) }
    }

    /// Counts beside the section names, from the ONE shared builder so they
    /// cannot drift from what the panes list.
    private var badges: ModelBrowserBadgeCounts {
        func readyCount<P: MediaModelPreset>(_ presets: [P]) -> Int {
            presets.filter { downloads.bundleReady($0.bundle) }.count
        }
        let media = readyCount(ImageModelPreset.all)
            + readyCount(AudioModelPreset.allIncludingVoiceOnly)
            + readyCount(VideoModelPreset.all)
            + readyCount(MusicModelPreset.all)
        return .live(localModelCount: appState.localModels.count,
                     activeDownloadCount: activeDownloads.count,
                     mediaReadyCount: media)
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelBrowserSectionBar(section: $section, badges: badges,
                                   isDownloading: !activeDownloads.isEmpty)
            Divider()
            detail
        }
        // The floor the browser's table needs, asserted from INSIDE the chat
        // window's detail column: below it the action cell clips off the right
        // edge (`ModelBrowserMetrics`). On the pane rather than the window, so
        // it only applies while the browser is up — a chat transcript is happy
        // much narrower.
        .frame(minWidth: ModelBrowserMetrics.minDetailWidth)
        .task {
            if searchService.models.isEmpty {
                await searchService.search()
            }
        }
        .onChange(of: section) { _, _ in appState.refreshModels() }
        // Live-refresh on-disk sizes while a disk-state pane is showing and a
        // download is in flight, so completion + growing size show up without
        // the user navigating away and back. The task id flips when the section
        // changes or the active-download set changes, which cancels +
        // re-evaluates the guard — so it self-terminates once everything
        // finishes.
        .task(id: "\(section.rawValue)-\(activeDownloads.count)") {
            guard ModelBrowserSection.shouldLivePoll(section: section,
                                                     hasActiveDownloads: !activeDownloads.isEmpty) else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                appState.refreshModels()
            }
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch section {
        case .recommended: RecommendedPane()
        case .discover:     DiscoverPane()
        case .myModels:     MyModelsPane(filter: $localFilter)
        case .downloads:    DownloadsPane(items: activeDownloads)
        case .media:        MediaPane()
        }
    }
}

// MARK: - Section bar

/// The browser's five sections, across the top of the content area.
private struct ModelBrowserSectionBar: View {
    @Binding var section: ModelBrowserSection
    let badges: ModelBrowserBadgeCounts
    let isDownloading: Bool

    var body: some View {
        HStack(spacing: 6) {
            ForEach(ModelBrowserSection.allCases) { item in
                chip(item)
            }
            Spacer(minLength: 8)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func chip(_ item: ModelBrowserSection) -> some View {
        let isSelected = section == item
        return Button {
            section = item
        } label: {
            HStack(spacing: 5) {
                Image(systemName: item.systemImage)
                    .font(.system(size: 11, weight: .medium))
                Text(item.title).font(.callout).lineLimit(1)
                if item == .downloads, isDownloading {
                    ProgressView()
                        .controlSize(.small)
                        .scaleEffect(0.55)
                        .frame(width: 10)
                }
                if let badge = badges.badge(for: item) {
                    Text(badge)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(isSelected ? .primary : .secondary)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                }
            }
            .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(isSelected ? Color.accentColor.opacity(0.14) : Color.clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Recommended

/// Every curated Gemma 4 / Qwen 3.5-3.6 checkpoint, grouped by family and
/// explained in plain English. This is the friendly front door for someone
/// who has never picked a local model before — Discover's HuggingFace search
/// (with its 1M+ repos, quant/pull-count columns, and RAM-fitness dots)
/// assumes you already know roughly what you're looking for.
private struct RecommendedPane: View {
    /// Everything below the one recommendation. Collapsed by default: picking a
    /// model used to mean choosing a vendor taxonomy before you could chat, and
    /// the answer for someone who has downloaded nothing is one model, not
    /// fourteen across four sections.
    private var memory: SystemMemoryInfo { SystemMemoryInfo.current() }
    private var starter: RecommendedModelPick {
        RecommendedModelPick.starterPick(physicalMemoryBytes: memory.totalBytes)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                MemorySummaryCard(memory: memory, recommended: starter)
                RecommendedModelTable(memory: memory, recommendedId: starter.id)
            }
            .padding(16)
        }
        .navigationTitle("Recommended")
    }
}

/// The "what does this Mac have, and how much can a model use?" summary at the
/// top of the Recommended pane: total RAM, the usable-for-models budget (the
/// Metal working-set ceiling), a capacity bar, and the one recommended pick.
private struct MemorySummaryCard: View {
    let memory: SystemMemoryInfo
    let recommended: RecommendedModelPick
    @EnvironmentObject var server: ServerManager

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "memorychip")
                    .font(.title2)
                    .foregroundStyle(.tint)
                    .frame(width: 26)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Your Mac")
                        .font(.headline)
                    Text("\(memory.totalLabel) of memory · about \(memory.usableLabel) usable for models")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }

            // The same GPU + Available RAM meter the menu bar shows.
            MemoryMeter.live(server: server.memoryInfo)

            HStack(spacing: 6) {
                Image(systemName: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.tint)
                Text("Recommended for your Mac:")
                    .foregroundStyle(.secondary)
                Text(recommended.name)
                    .fontWeight(.semibold)
                Spacer(minLength: 0)
            }
            .font(.callout)
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 12).fill(.quaternary.opacity(0.22)))
    }
}

/// Column widths shared by the table header and every row so they stay
/// aligned. The Model column flexes; the rest are fixed.
private enum RecTableMetrics {
    static let capability: CGFloat = 92
    static let size: CGFloat = 62
    static let memory: CGFloat = 140
    static let action: CGFloat = 116
    static let spacing: CGFloat = 10
    static let hPad: CGFloat = 12
}

/// The Recommended pane's model table: a column header, then every curated pick
/// grouped by family. Each row shows its capability, download size, and — the
/// point of this pane — exactly how its memory requirement fits THIS Mac. Every
/// model is visible (no "requires more RAM" disclosure): the fit badge is what
/// makes the comparison scannable at a glance.
private struct RecommendedModelTable: View {
    let memory: SystemMemoryInfo
    let recommendedId: String

    private struct Family: Identifiable {
        let id: String
        let title: String
        let systemImage: String
        let tint: Color
        let picks: [RecommendedModelPick]
    }

    private var families: [Family] {
        [
            Family(id: "gemma", title: "Gemma 4", systemImage: "g.circle", tint: .blue,
                   picks: RecommendedModelPick.gemmaCatalog),
            Family(id: "qwen", title: "Qwen", systemImage: "q.circle", tint: .teal,
                   picks: RecommendedModelPick.qwenCatalog),
            Family(id: "laguna", title: "Laguna", systemImage: "chevron.left.forwardslash.chevron.right", tint: .purple,
                   picks: RecommendedModelPick.poolsideCatalog),
            Family(id: "largest", title: "Largest models", systemImage: "memorychip", tint: .red,
                   picks: RecommendedModelPick.largestCatalog),
        ]
    }

    var body: some View {
        VStack(spacing: 0) {
            RecommendedTableHeader()
            Divider()
            ForEach(families) { family in
                familyHeader(family)
                ForEach(family.picks) { pick in
                    Divider().padding(.leading, RecTableMetrics.hPad)
                    RecommendedModelTableRow(
                        pick: pick,
                        memory: memory,
                        isRecommended: pick.id == recommendedId
                    )
                }
            }
        }
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.15)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.quaternary.opacity(0.4)))
    }

    private func familyHeader(_ family: Family) -> some View {
        HStack(spacing: 7) {
            Image(systemName: family.systemImage)
                .font(.caption)
                .foregroundStyle(family.tint)
            Text(family.title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, RecTableMetrics.hPad)
        .padding(.vertical, 7)
        .background(.quaternary.opacity(0.12))
    }
}

/// The column titles, aligned to the row columns via `RecTableMetrics`.
private struct RecommendedTableHeader: View {
    var body: some View {
        HStack(spacing: RecTableMetrics.spacing) {
            Text("Model")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("Capability")
                .frame(width: RecTableMetrics.capability, alignment: .leading)
            Text("Size")
                .frame(width: RecTableMetrics.size, alignment: .trailing)
            Text("Memory needed")
                .frame(width: RecTableMetrics.memory, alignment: .leading)
            Color.clear
                .frame(width: RecTableMetrics.action, height: 1)
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(.secondary)
        .textCase(.uppercase)
        .padding(.horizontal, RecTableMetrics.hPad)
        .padding(.vertical, 8)
    }
}

/// Two thin comparative bars — intelligence (blue) over speed (green) — the
/// compact, table-cell form of the retired `CapabilityBars`. No number is
/// drawn: the scores are a hand-maintained comparison between these picks (see
/// `RecommendedModels.swift`'s header), and printing "62" would claim a
/// precision they don't have. The tooltip names the two bars and flags an
/// estimated intelligence score.
private struct MiniCapability: View {
    let pick: RecommendedModelPick

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            bar(pick.intelligenceBar, .blue)
            bar(pick.speedBar, .green)
        }
        .help("Top bar: intelligence\(pick.intelligenceIsEstimated ? " (our estimate)" : ""). Bottom bar: speed. Both relative to the models here.")
    }

    private func bar(_ fill: Double, _ tint: Color) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(tint.opacity(0.8))
                    .frame(width: max(2, geo.size.width * fill))
            }
        }
        .frame(height: 4)
    }
}

/// One table row for a chat-model recommendation: name + tagline, compact
/// capability bars, download size, the memory-fit cell (the pane's whole
/// point), and the Download/Use action. The full plain-English blurb moves to
/// the row's hover tooltip so the table stays scannable.
private struct RecommendedModelTableRow: View {
    let pick: RecommendedModelPick
    let memory: SystemMemoryInfo
    let isRecommended: Bool

    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var confirmDelete = false
    @State private var card: ModelCardRequest?

    /// How this pick's memory requirement fits the Mac's usable budget.
    private var fit: MemoryFit { memory.fit(neededGB: pick.approxRAMNeededGB) }

    /// For a GGUF pick, the specific quant file on disk (the repo ships many);
    /// nil until the folder resolves. Drives ready/use so a *different* quant of
    /// the same repo doesn't read as this pick being downloaded.
    private var ggufFilePath: String? {
        guard let f = pick.ggufFilename, let dir = downloads.existingModelDir(for: pick.repoId) else { return nil }
        return (dir as NSString).appendingPathComponent(f)
    }
    private var isReady: Bool {
        if pick.ggufFilename != nil {
            guard let p = ggufFilePath else { return false }
            return FileManager.default.fileExists(atPath: p)
        }
        return downloads.isReady(pick.repoId)
    }
    private var state: DownloadManager.DownloadState? { downloads.downloads[pick.repoId] }

    /// The on-disk model this row's repo resolves to, once downloaded —
    /// mirrors `ModelBrowserRow.usableModel`.
    private var usableModel: LocalModel? {
        // A GGUF quant's LocalModel.path is the FILE, not the repo dir — resolve
        // against the specific quant so "Use" loads exactly this pick.
        let path = pick.ggufFilename != nil ? ggufFilePath : downloads.existingModelDir(for: pick.repoId)
        return ModelBrowserUse.pickableModel(atPath: path, in: appState.localModels)
    }

    /// The other half of the catalogue — see `ModelBrowserUse.mediaModel`.
    private var usableMedia: (model: LocalModel, modality: MediaModality)? {
        let path = pick.ggufFilename != nil ? ggufFilePath : downloads.existingModelDir(for: pick.repoId)
        return ModelBrowserUse.mediaModel(atPath: path, in: appState.localModels)
    }

    var body: some View {
        HStack(spacing: RecTableMetrics.spacing) {
            // Model — name + tagline; the full blurb is on hover.
            Button { card = ModelCardRequest(repoId: pick.repoId, title: pick.name) } label: {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(pick.name)
                            .font(.callout.weight(.medium))
                        if isRecommended {
                            Text("Recommended")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Capsule().fill(Color.accentColor.opacity(0.18)))
                                .foregroundStyle(.tint)
                        }
                    }
                    Text(pick.tagline)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .help(pick.blurb)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)

            // Capability — intelligence over speed.
            MiniCapability(pick: pick)
                .frame(width: RecTableMetrics.capability, alignment: .leading)

            // Download size (on disk) and the quant it buys.
            VStack(alignment: .trailing, spacing: 2) {
                Text(SystemMemoryInfo.preciseGB(pick.sizeGB))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                if let quant = pick.quantLabel {
                    Text(quant)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(width: RecTableMetrics.size, alignment: .trailing)

            // Memory needed + fit against this Mac's usable budget.
            memoryCell
                .frame(width: RecTableMetrics.memory, alignment: .leading)

            // Download / Use / progress.
            actionControl
                .frame(width: RecTableMetrics.action, alignment: .trailing)
        }
        .padding(.horizontal, RecTableMetrics.hPad)
        .padding(.vertical, 9)
        .background(isRecommended ? Color.accentColor.opacity(0.07) : Color.clear)
        .sheet(item: $card) { ModelDetailSheet(request: $0) }
    }

    /// The memory column: how much RAM the model needs, and a colored badge for
    /// how that fits what this Mac can give a model.
    private var memoryCell: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(SystemMemoryInfo.preciseGB(pick.approxRAMNeededGB))
                .font(.caption.monospacedDigit().weight(.medium))
            HStack(spacing: 3) {
                Image(systemName: fitIcon)
                    .font(.system(size: 9))
                Text(fit.label)
                    .font(.caption2)
            }
            .foregroundStyle(fitColor)
        }
        .help("Needs about \(SystemMemoryInfo.preciseGB(pick.approxRAMNeededGB)) — your Mac can use about \(memory.usableLabel) for a model.")
    }

    private var fitIcon: String {
        switch fit {
        case .comfortable: return "checkmark.circle.fill"
        case .tight:       return "exclamationmark.circle.fill"
        case .exceeds:     return "xmark.circle.fill"
        }
    }

    private var fitColor: Color {
        switch fit {
        case .comfortable: return .green
        case .tight:       return .orange
        case .exceeds:     return .red
        }
    }

    @ViewBuilder
    private var actionControl: some View {
        if isReady {
            HStack(spacing: 6) {
                if let usable = usableModel {
                    let use = ModelUseState.resolve(
                        selected: appState.selectedModelPath == usable.path,
                        serverStatus: server.status
                    )
                    if use == .idle {
                        UseModelButton(path: usable.path, name: usable.name)
                    } else {
                        ModelUseBadge(state: use)
                    }
                } else if let media = usableMedia {
                    UseMediaModelButton(modality: media.modality, name: media.model.name)
                } else {
                    Text("✓ On disk")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                }
                Button {
                    confirmDelete = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red.opacity(0.7))
                }
                .buttonStyle(.plain)
                .help("Delete model")
                .alert("Delete Model", isPresented: $confirmDelete) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        downloads.deleteModel(repoId: pick.repoId)
                        appState.refreshModels()
                    }
                    .keyboardShortcut(.defaultAction)
                } message: {
                    Text("Delete \(pick.name)? This will remove all downloaded files.")
                }
            }
        } else if let state, state.status == .downloading {
            HStack(spacing: 6) {
                VStack(alignment: .trailing, spacing: 1) {
                    ProgressView(value: state.progress)
                        .frame(width: 70)
                    Text("\(state.percentFormatted) \(state.speedFormatted)")
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Button {
                    downloads.cancel(pick.repoId)
                    appState.refreshModels()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Cancel download")
            }
        } else if let state, state.status == .failed {
            VStack(alignment: .trailing, spacing: 2) {
                Button(downloads.hasPartialDownload(pick.repoId) ? "Resume" : "Retry") { startDownload() }
                    .controlSize(.small)
                if let error = state.error {
                    Text(error)
                        .font(.system(size: 9))
                        .foregroundStyle(.red)
                        .lineLimit(1)
                }
            }
        } else {
            Button(downloads.hasPartialDownload(pick.repoId) ? "Resume" : "Download") { startDownload() }
                .controlSize(.small)
        }
    }

    private func startDownload() {
        if let f = pick.ggufFilename {
            // GGUF/ds4 pick: fetch the specific quant (the download path also
            // auto-pulls the ds4 MTP draft head).
            downloads.startGguf(
                repoId: pick.repoId,
                quant: GgufQuant(filename: f, label: DownloadManager.quantLabel(forFilename: f))
            ) { appState.refreshModels() }
        } else {
            downloads.start(repoId: pick.repoId) { appState.refreshModels() }
        }
    }
}

// MARK: - Discover

/// HuggingFace search. On-disk models stay in the list, marked `✓ On disk` with
/// a Use action — never filtered out.
private struct DiscoverPane: View {
    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager

    /// Measured pane width → column tier. 0 until the first layout pass,
    /// which `ModelBrowserMetrics.tier` treats as roomy.
    @State private var paneWidth: CGFloat = 0

    private var tier: ModelBrowserMetrics.Tier {
        ModelBrowserMetrics.tier(forDetailWidth: paneWidth)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search models...", text: $searchService.searchQuery)
                        .textFieldStyle(.plain)
                        .onSubmit { Task { await searchService.search() } }
                }
                .padding(8)
                .background(.quaternary.opacity(0.5))
                .cornerRadius(8)

                // Weight-format filter: MLX (safetensors), GGUF (llama.cpp /
                // ds4), or Both. Re-runs the search on change.
                Picker("Format", selection: $searchService.format) {
                    ForEach(ModelFormat.allCases) { f in
                        Text(f.label).tag(f)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .onChange(of: searchService.format) { _, _ in
                    Task { await searchService.search() }
                }

                Button("Search") {
                    Task { await searchService.search() }
                }
                .controlSize(.regular)
            }
            .padding(12)

            Divider()

            ColumnHeaderRow(searchService: searchService, tier: tier)
                .padding(.horizontal, ModelBrowserMetrics.rowPaddingH)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.3))

            Divider()

            let onDiskCount = searchService.models.filter { downloads.isReady($0.id) }.count

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(searchService.models) { model in
                        ModelBrowserRow(
                            model: model,
                            fitness: searchService.ramFitness(for: model),
                            tier: tier
                        )
                        Divider().padding(.horizontal, 12)
                    }

                    if searchService.isLoading {
                        ProgressView()
                            .padding(20)
                    } else if let error = searchService.error {
                        Text(error)
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(20)
                    } else if searchService.models.isEmpty {
                        Text("No models found")
                            .foregroundStyle(.secondary)
                            .padding(40)
                    }

                    if searchService.hasMore && !searchService.models.isEmpty && !searchService.isLoading {
                        Button("Load More") {
                            Task { await searchService.loadMore() }
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.regular)
                        .padding(16)
                    }
                }
            }

            Divider()

            HStack {
                Text("Showing \(searchService.models.count) models")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if onDiskCount > 0 {
                    Text("· \(onDiskCount) on disk")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
                Spacer()
                Text("System RAM: \(MemoryInfo.format(Int64(searchService.systemRAM)))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        // Track the pane's width for the column-tier decision — same
        // pattern as ChatView's toolbar pill density.
        .background(
            GeometryReader { geo in
                Color.clear
                    .onAppear { paneWidth = geo.size.width }
                    .onChange(of: geo.size.width) { _, w in paneWidth = w }
            }
        )
        .navigationTitle("Discover")
    }
}

// MARK: - My Models

/// Everything on this Mac the tray picker can offer, grouped by where it came
/// from. The old "Downloaded" tab listed only `source == .mlxServe`, so it was a
/// strict subset of what you could actually load.
private struct MyModelsPane: View {
    @Binding var filter: String
    @EnvironmentObject var appState: AppState

    private var groups: [LocalModelGroup] {
        ModelBrowserUse.groupedBySource(appState.localModels, filter: filter)
    }

    private var total: Int { groups.reduce(0) { $0 + $1.models.count } }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Filter your models...", text: $filter)
                    .textFieldStyle(.plain)
                if !filter.isEmpty {
                    Button { filter = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(.quaternary.opacity(0.5))
            .cornerRadius(8)
            .padding(12)

            Divider()

            HStack(spacing: 8) {
                Text("Model")
                    .frame(maxWidth: .infinity, alignment: .leading)
                Text("Size on Disk")
                    .frame(width: 90, alignment: .trailing)
                Text("")
                    .frame(width: 120)
            }
            .font(.callout.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.quaternary.opacity(0.3))

            Divider()

            ScrollView {
                LazyVStack(spacing: 0, pinnedViews: [.sectionHeaders]) {
                    ForEach(groups) { group in
                        Section {
                            ForEach(group.models) { model in
                                LocalModelRow(model: model)
                                Divider().padding(.horizontal, 12)
                            }
                        } header: {
                            Text(group.title)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 4)
                                .background(.bar)
                        }
                    }

                    if groups.isEmpty {
                        Text(filter.isEmpty ? "No models on this Mac yet" : "No models match “\(filter)”")
                            .foregroundStyle(.secondary)
                            .padding(40)
                    }
                }
            }

            Divider()

            HStack {
                Text("\(total) model\(total == 1 ? "" : "s") on disk")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .navigationTitle("My Models")
    }
}

// MARK: - Downloads

/// The transfer queue. Promoted out of the old Downloaded tab so an in-flight
/// download is reachable (and badged) from anywhere in the browser.
private struct DownloadsPane: View {
    let items: [(repoId: String, state: DownloadManager.DownloadState)]

    var body: some View {
        VStack(spacing: 0) {
            if items.isEmpty {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No downloads in progress")
                        .foregroundStyle(.secondary)
                    Text("Start one from Recommended or Discover.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(items, id: \.repoId) { item in
                            ActiveDownloadRow(repoId: item.repoId, state: item.state)
                            Divider().padding(.horizontal, 12)
                        }
                    }
                }
            }
        }
        .navigationTitle("Downloads")
    }
}

// MARK: - Media pane

/// Every media-generation model (image/audio/video/music), grouped by
/// modality. Its own sidebar destination — there's one catalog per modality
/// and none of them are large enough to need Discover-style search, so one
/// scrollable page beats splitting each modality into its own sidebar row.
private struct MediaPane: View {
    private var physicalMemory: UInt64 { ProcessInfo.processInfo.physicalMemory }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                ModelGroupSection(
                    title: "Image",
                    subtitle: "Text-to-image generation and image editing.",
                    systemImage: "photo",
                    tint: .pink
                ) {
                    ForEach(ImageModelPreset.all) { MediaModelRow(preset: $0, modality: .image, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Audio",
                    subtitle: "Text-to-speech, with optional voice cloning.",
                    systemImage: "waveform",
                    tint: .green
                ) {
                    // The browser lists the FULL catalog — Kokoro is voice-mode
                    // only (out of `.all`, which the media panes offer) but is
                    // still a model the user can fetch from here.
                    ForEach(AudioModelPreset.allIncludingVoiceOnly) { MediaModelRow(preset: $0, modality: .voice, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Video",
                    subtitle: "Text/image-to-video, with optional audio.",
                    systemImage: "film",
                    tint: .indigo
                ) {
                    ForEach(VideoModelPreset.all) { MediaModelRow(preset: $0, modality: .video, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Music",
                    subtitle: "Text-to-music, with optional lyrics.",
                    systemImage: "music.note",
                    tint: .orange
                ) {
                    ForEach(MusicModelPreset.all) { MediaModelRow(preset: $0, modality: .music, physicalMemoryBytes: physicalMemory) }
                }
            }
            .padding(16)
        }
        .navigationTitle("Media")
    }
}

/// One family/modality group's header (icon, name, what it's for) over its
/// list of rows — shared by the Recommended pane (grouped by model family)
/// and the Media pane (grouped by modality).
private struct ModelGroupSection<Content: View>: View {
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: Color
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .foregroundStyle(tint)
                    .frame(width: 18)
                VStack(alignment: .leading, spacing: 1) {
                    Text(title).font(.subheadline.weight(.semibold))
                    Text(subtitle).font(.caption2).foregroundStyle(.secondary)
                }
                Spacer()
            }

            VStack(spacing: 0) {
                content
            }
            .background(.quaternary.opacity(0.15), in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

/// One row for any media preset (image/audio/video/music) — generic over
/// `MediaModelPreset` so the four modalities share this instead of four
/// near-duplicate views. Download/progress/retry mirrors `BundleDownloadBar`.
///
/// The terminal state used to be on-disk + Delete and nothing else, on the
/// reasoning that each gen pane keeps its own sticky model selection. But that
/// left "throw it away" as the only verb the browser offered for a model it
/// had just finished downloading (#228). `Use` opens the owning pane; the pane
/// still keeps its own selection, so this is navigation, not a second loader.
/// The modality is passed in rather than derived — the Media pane already
/// groups by it, so the call site knows it exactly.
private struct MediaModelRow<Preset: MediaModelPreset>: View {
    let preset: Preset
    let modality: MediaModality
    let physicalMemoryBytes: UInt64
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var confirmDelete = false
    @State private var card: ModelCardRequest?

    private var bundle: MediaBundle { preset.bundle }
    private var isReady: Bool { downloads.bundleReady(bundle) }
    private var active: (repo: String, index: Int, count: Int, state: DownloadManager.DownloadState)? {
        downloads.activeBundleComponent(bundle)
    }

    /// Soft signal only — shows a warning, never blocks downloading or using
    /// it (same "warn, don't gate" policy as the Recommended pane).
    private var meetsRequirements: Bool {
        preset.meetsSystemRequirements(physicalMemoryBytes: physicalMemoryBytes)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Button { card = ModelCardRequest(repoId: bundle.primaryRepo, title: preset.name) } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(preset.name)
                    .font(.callout.weight(.medium))

                Text(preset.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if bundle.components.count > 1 {
                    Text("Includes \(bundle.components.count) models (e.g. a text encoder)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                if !meetsRequirements {
                    Label(
                        "Needs about \(preset.approxRAMGB) GB of RAM — your Mac has \(MemoryInfo.format(Int64(physicalMemoryBytes)))",
                        systemImage: "exclamationmark.triangle"
                    )
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.orange)
                }

                if let active, active.state.status == .failed, let error = active.state.error {
                    Text(error)
                        .font(.system(size: 9))
                        .foregroundStyle(.red)
                        .lineLimit(1)
                }
            }
            .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .trailing, spacing: 4) {
                Text(bundle.approxSizeLabel)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                actionControl
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .sheet(item: $card) { ModelDetailSheet(request: $0) }
    }

    @ViewBuilder
    private var actionControl: some View {
        if isReady {
            HStack(spacing: 6) {
                UseMediaModelButton(modality: modality, name: preset.name)
                Button {
                    confirmDelete = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red.opacity(0.7))
                }
                .buttonStyle(.plain)
                .help("Delete model")
                .alert("Delete Model", isPresented: $confirmDelete) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        downloads.deleteModel(repoId: bundle.primaryRepo)
                        appState.refreshModels()
                    }
                    .keyboardShortcut(.defaultAction)
                } message: {
                    Text("Delete \(preset.name)? This will remove the downloaded files.")
                }
            }
        } else if let active, active.state.status == .downloading {
            HStack(spacing: 6) {
                VStack(alignment: .trailing, spacing: 1) {
                    ProgressView(value: active.state.progress)
                        .frame(width: 70)
                    Text("\(active.state.percentFormatted) \(active.state.speedFormatted)")
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Button {
                    downloads.cancelBundle(bundle)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Cancel download")
            }
        } else if active?.state.status == .failed {
            Button("Retry") { startDownload() }
                .controlSize(.small)
        } else {
            Button("Download") { startDownload() }
                .controlSize(.small)
        }
    }

    private func startDownload() {
        downloads.startBundle(bundle) { appState.refreshModels() }
    }
}

// MARK: - Use button

/// The "Use" control for any Model Browser row (Discover / My Models /
/// Recommended): selects the model, makes the server actually load it
/// (starting it if stopped, hot-switching/restarting if already running),
/// and once it's ready opens the Chat window — a click ends in a
/// ready-to-chat server, not just a selection the user then has to start
/// themselves. Shared so the three rows can't drift onto three slightly
/// different "Use" behaviors.
private struct UseModelButton: View {
    let path: String
    let name: String
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) private var openWindow
    @State private var isLoading = false

    var body: some View {
        Button {
            Task {
                isLoading = true
                let ready = await appState.useModelAndAwaitReady(atPath: path)
                isLoading = false
                if ready {
                    AppActivation.openWindow(id: "chat", using: openWindow)
                }
            }
        } label: {
            if isLoading {
                ProgressView()
                    .controlSize(.small)
                    .frame(width: 30)
            } else {
                Text("Use")
            }
        }
        .controlSize(.small)
        .disabled(isLoading)
        .help("Load \(name) as the server's model, then open chat")
    }
}

/// "Use" for a media checkpoint: open the pane that owns it.
///
/// `AppState.showCreate` is documented as the one way into a create page.
/// Music needs a second step because it has no `GenExperiment` case of its own
/// — it is a tab inside the Audio pane — and that tab is `@AppStorage`, so
/// writing it here is what makes the pane come up on Music instead of Voice.
private struct UseMediaModelButton: View {
    let modality: MediaModality
    let name: String
    @EnvironmentObject var appState: AppState
    @Environment(\.openWindow) private var openWindow
    @AppStorage("audioGenTab") private var audioTab: AudioGenView.Tab = .voice

    var body: some View {
        Button("Use") {
            if let tab = modality.audioTab { audioTab = tab }
            appState.showCreate(modality.experiment)
            AppActivation.openWindow(id: "chat", using: openWindow)
        }
        .controlSize(.small)
        .help("Open \(name) in \(modality.paneName)")
    }
}

// MARK: - In-use badge

/// Replaces the "Use" button on the model the server is pointed at, so clicking
/// Use produces immediate, visible feedback instead of just greying the button
/// out. Distinguishes "loaded and serving" from "selected, still loading" and
/// "selected, server stopped" — see `ModelUseState`.
private struct ModelUseBadge: View {
    let state: ModelUseState

    private var tint: Color {
        switch state {
        case .inUse:    return .green
        case .loading:  return .orange
        case .selected: return .secondary
        case .idle:     return .clear
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            switch state {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.5)
                    .frame(width: 8, height: 8)
            case .inUse:
                Circle()
                    .fill(tint)
                    .frame(width: 6, height: 6)
            default:
                EmptyView()
            }
            Text(state.label)
                .font(.caption.weight(.medium))
                .foregroundStyle(tint)
                .lineLimit(1)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(tint.opacity(0.12), in: Capsule())
        .help(state.help)
    }
}

// MARK: - Column Headers

/// Column widths and visibility come from `ModelBrowserMetrics` — the ONE
/// source of truth shared with `ModelBrowserRow`, so header/row alignment
/// can't drift and narrow panes drop the same columns in both.
private struct ColumnHeaderRow: View {
    @ObservedObject var searchService: HFSearchService
    let tier: ModelBrowserMetrics.Tier

    var body: some View {
        HStack(spacing: ModelBrowserMetrics.columnSpacing) {
            Text("Model")
                .frame(maxWidth: .infinity, alignment: .leading)
            SortableHeader("Quant", field: nil, searchService: searchService)
                .frame(width: ModelBrowserMetrics.quantWidth, alignment: .leading)
            SortableHeader("Size", field: nil, searchService: searchService)
                .frame(width: ModelBrowserMetrics.sizeWidth, alignment: .trailing)
            // HuggingFace pull count. Called "Pulls", NOT "Downloads": the
            // sidebar has a Downloads destination meaning "transferring right
            // now", and having both words in one window is what made users read
            // the old "Downloaded" toggle as a filter on this column. 64 wide
            // fits "Pulls" + the sort chevron.
            if tier.showsPulls {
                SortableHeader("Pulls", field: .downloads, searchService: searchService)
                    .frame(width: ModelBrowserMetrics.pullsWidth, alignment: .trailing)
                    .help("How many times this repo has been pulled from HuggingFace")
            }
            if tier.showsLikes {
                SortableHeader("Likes", field: .likes, searchService: searchService)
                    .frame(width: ModelBrowserMetrics.likesWidth, alignment: .trailing)
            }
            SortableHeader("RAM Est.", field: .estimatedSize, searchService: searchService)
                .frame(width: ModelBrowserMetrics.ramWidth, alignment: .trailing)
            if tier.showsUpdated {
                SortableHeader("Updated", field: .lastModified, searchService: searchService)
                    .frame(width: ModelBrowserMetrics.updatedWidth, alignment: .trailing)
            }
            Text("")
                .frame(width: ModelBrowserMetrics.actionWidth)
        }
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
    }
}

private struct SortableHeader: View {
    let title: String
    let field: HFSortField?
    @ObservedObject var searchService: HFSearchService

    init(_ title: String, field: HFSortField?, searchService: HFSearchService) {
        self.title = title
        self.field = field
        self.searchService = searchService
    }

    private var isActive: Bool {
        guard let field else { return false }
        return searchService.sortField == field
    }

    var body: some View {
        if let field {
            Button {
                searchService.sort(by: field)
            } label: {
                HStack(spacing: 2) {
                    Text(title)
                    if isActive {
                        Image(systemName: searchService.sortDescending ? "chevron.down" : "chevron.up")
                            .font(.system(size: 8))
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(isActive ? .primary : .secondary)
        } else {
            Text(title)
        }
    }
}

// MARK: - Model Row

private struct ModelBrowserRow: View {
    let model: HFModel
    let fitness: RAMFitness
    let tier: ModelBrowserMetrics.Tier
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    private var isReady: Bool { downloads.isReady(model.id) }
    private var state: DownloadManager.DownloadState? { downloads.downloads[model.id] }
    private var disabled: Bool { !model.isCompatible }
    @State private var card: ModelCardRequest?

    var body: some View {
        HStack(spacing: ModelBrowserMetrics.columnSpacing) {
            // Model name — takes all remaining space; click opens the card.
            Button { card = ModelCardRequest(repoId: model.id, title: model.modelName) } label: {
            VStack(alignment: .leading, spacing: 1) {
                Text(model.modelName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(model.author)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                    if let reason = model.incompatibleReason {
                        Text(reason)
                            .font(.system(size: 10))
                            .foregroundStyle(.red.opacity(0.8))
                            .lineLimit(1)
                    }
                }
            }
            .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)

            // Quantization badge
            Group {
                if let quant = model.quantization {
                    Text(quant)
                        .font(.system(size: 10).weight(.medium))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary)
                        .cornerRadius(4)
                } else {
                    Text("\u{2014}")
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(width: ModelBrowserMetrics.quantWidth, alignment: .leading)

            // Size (parsed from model name)
            Text(model.modelSize)
                .font(.callout.monospacedDigit())
                .frame(width: ModelBrowserMetrics.sizeWidth, alignment: .trailing)

            // HF pull count
            if tier.showsPulls {
                Text(formatCount(model.downloads ?? 0))
                    .font(.callout.monospacedDigit())
                    .frame(width: ModelBrowserMetrics.pullsWidth, alignment: .trailing)
            }

            // Likes
            if tier.showsLikes {
                Text(formatCount(model.likes ?? 0))
                    .font(.callout.monospacedDigit())
                    .frame(width: ModelBrowserMetrics.likesWidth, alignment: .trailing)
            }

            // RAM estimate with color indicator — 120 so GGUF range strings
            // like "21.2–55.4 GB" stay on one line. `.lineLimit(1)` is the
            // belt-and-suspenders guard against any future format that
            // exceeds the budget — we'd rather truncate than wrap.
            HStack(spacing: 4) {
                Circle()
                    .fill(fitnessColor)
                    .frame(width: 8, height: 8)
                Text(model.ramEstimate)
                    .font(.callout.monospacedDigit())
                    .lineLimit(1)
            }
            .frame(width: ModelBrowserMetrics.ramWidth, alignment: .trailing)

            // Last updated
            if tier.showsUpdated {
                Text(formatRelativeDate(model.lastModifiedDate))
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(width: ModelBrowserMetrics.updatedWidth, alignment: .trailing)
            }

            actionCell
                .frame(width: ModelBrowserMetrics.actionWidth, alignment: .center)
        }
        .padding(.horizontal, ModelBrowserMetrics.rowPaddingH)
        .padding(.vertical, 6)
        .opacity(disabled ? 0.4 : 1.0)
        .sheet(item: $card) { ModelDetailSheet(request: $0) }
    }

    private var fitnessColor: Color {
        switch fitness {
        case .fits: return .green
        case .tight: return .yellow
        case .wontFit: return .red
        case .unknown: return .gray
        }
    }

    @State private var confirmDelete = false

    /// Resolved by the pure state machine so the branch ladder is unit-tested
    /// (`ModelBrowserSectionTests`). The key change: a ready model resolves to
    /// `.onDisk` and stays in the list, where it used to be filtered out of the
    /// search results entirely — vanishing at the exact moment it finished.
    private var action: ModelRowAction {
        ModelRowAction.resolve(
            isCompatible: model.isCompatible,
            isReady: isReady,
            status: state?.status,
            hasPartial: downloads.hasPartialDownload(model.id),
            progress: state?.progress ?? 0
        )
    }

    /// The on-disk model this row's repo resolves to, when it's loadable as the
    /// server's chat model. nil for drafters, encoders, and media checkpoints —
    /// they're on disk and deletable, but "Use" would load something that can't
    /// serve a completion.
    private var usableModel: LocalModel? {
        ModelBrowserUse.pickableModel(
            atPath: downloads.existingModelDir(for: model.id),
            in: appState.localModels
        )
    }

    /// The other half of the catalogue — see `ModelBrowserUse.mediaModel`.
    private var usableMedia: (model: LocalModel, modality: MediaModality)? {
        ModelBrowserUse.mediaModel(
            atPath: downloads.existingModelDir(for: model.id),
            in: appState.localModels
        )
    }

    /// A verified community media pack downloads as its FAMILY bundle — the
    /// same allowlists + ready markers the catalog packs use — never the flat
    /// chat-default pull, which would miss subdirs and grab repo junk. nil for
    /// everything that isn't a served media repo.
    private var mediaBundle: MediaBundle? {
        guard model.mediaStructureVerified == true,
              let arch = model.mediaFamilyModelType else { return nil }
        return CustomMediaModels.bundle(arch: arch, repoId: model.id)
    }

    /// Start (or resume) this row's download. The media arm also asks the
    /// running server to rescan afterward — boot-time discovery can't see a
    /// model downloaded mid-session, and the media panes' "On This Mac" rows
    /// read `/v1/models`.
    private func startDownload() {
        if let b = mediaBundle {
            downloads.startBundle(b) {
                appState.refreshModels()
                server.rescanModels()
            }
        } else {
            downloads.start(repoId: model.id) { appState.refreshModels() }
        }
    }

    @ViewBuilder
    private var actionCell: some View {
        // Same shelf shape as GGUF, in directories instead of files: some MLX
        // repos publish every quant as a subfolder (`LiquidAI/LFM2.5-2.6B-MLX`).
        // Checked first — such a repo carries no `gguf` tag, so the two can't
        // both match.
        if model.isMlxVariantRepo, action != .unsupported {
            if case .downloading(let progress) = action {
                downloadingCell(progress: progress)
            } else {
                MlxVariantMenu(repoId: model.id, variants: model.mlxVariants, state: state)
            }
        }
        // A GGUF repo is a FOLDER OF QUANTS, not a model, so it never reaches a
        // terminal "on disk" state: owning Q4_K_M says nothing about whether you
        // also want Q8_0. Its cell stays a menu that marks what you have and
        // keeps offering what you don't — the old `.onDisk` collapse to
        // "✓ On disk" + trash left no way back to the quant picker.
        else if model.isGgufRepo, action != .unsupported {
            if case .downloading(let progress) = action {
                downloadingCell(progress: progress)
            } else {
                GgufQuantMenu(repoId: model.id, state: state)
            }
        } else {
            switch action {
            case .unsupported:
                Image(systemName: "nosign")
                    .foregroundStyle(.secondary)
                    .font(.caption)

            case .onDisk:
                HStack(spacing: 6) {
                    if let usable = usableModel {
                        let use = ModelUseState.resolve(
                            selected: appState.selectedModelPath == usable.path,
                            serverStatus: server.status
                        )
                        if use == .idle {
                            UseModelButton(path: usable.path, name: usable.name)
                        } else {
                            ModelUseBadge(state: use)
                        }
                    } else if let media = usableMedia {
                        UseMediaModelButton(modality: media.modality, name: media.model.name)
                    } else {
                        Text("✓ On disk")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.green)
                    }
                    deleteButton
                }

            case .downloading(let progress):
                downloadingCell(progress: progress)

            case .failed(let resumable):
                Button(resumable ? "Resume" : "Retry") {
                    startDownload()
                }
                .font(.callout)
                .controlSize(.small)

            case .notDownloaded(let resumable):
                Button(resumable ? "Resume" : "Download") {
                    startDownload()
                }
                .font(.callout)
                .controlSize(.small)
            }
        }
    }

    private func downloadingCell(progress: Double) -> some View {
        HStack(spacing: 4) {
            VStack(spacing: 1) {
                ProgressView(value: progress)
                    .frame(width: 50)
                Text(state?.percentFormatted ?? "")
                    .font(.system(size: 9).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Button {
                // A bundle download's task is keyed by the BUNDLE id — a
                // per-repo cancel would only wipe state, not stop the loop.
                if let b = mediaBundle { downloads.cancelBundle(b) } else { downloads.cancel(model.id) }
                appState.refreshModels()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Cancel download")
        }
    }

    private var deleteButton: some View {
        Button {
            confirmDelete = true
        } label: {
            Image(systemName: "trash")
                .foregroundStyle(.red.opacity(0.7))
        }
        .buttonStyle(.plain)
        .font(.callout)
        .help("Delete model")
        .alert("Delete Model", isPresented: $confirmDelete) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                downloads.deleteModel(repoId: model.id)
                appState.refreshModels()
            }
            .keyboardShortcut(.defaultAction)
        } message: {
            Text("Delete \(model.modelName)? This will remove all downloaded files.")
        }
    }
}

/// The action cell for a GGUF repo: one menu covering every quant, in every
/// state.
private struct GgufQuantMenu: View {
    let repoId: String
    let state: DownloadManager.DownloadState?
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var remote: [String] = []
    @State private var loaded = false
    @State private var pendingDelete: GgufQuant?

    private var menu: GgufQuantMenuModel.Menu {
        GgufQuantMenuModel.build(
            remote: remote,
            // Recursive paths so a sharded quant's nested shards are seen and an
            // incomplete split reads as available (resume), not on-disk.
            onDisk: downloads.downloadedGgufPaths(repoId: repoId)
        )
    }

    /// Where a quant lives on disk — the path the server loads and the tray
    /// picker selects, so "Use" here and picking it in the tray are the same act.
    /// `quant.filename` is the repo-relative PRIMARY shard, so this resolves to
    /// the `-00001` shard for a sharded quant (libllama auto-loads the rest).
    private func path(of quant: GgufQuant) -> String? {
        guard let dir = downloads.existingModelDir(for: repoId) else { return nil }
        return (dir as NSString).appendingPathComponent(quant.filename)
    }

    var body: some View {
        let m = menu
        Menu {
            if !m.onDisk.isEmpty {
                Section("On this Mac") {
                    ForEach(m.onDisk) { quant in
                        Button {
                            guard let p = path(of: quant) else { return }
                            Task { _ = await appState.useModelAndAwaitReady(atPath: p) }
                        } label: {
                            let selected = path(of: quant) == appState.selectedModelPath
                            Label(
                                selected ? "\(quant.label) — in use" : "\(quant.label) — use",
                                systemImage: selected ? "checkmark.circle.fill" : "checkmark"
                            )
                        }
                    }
                }
            }

            Section(m.onDisk.isEmpty ? "Choose a quant" : "Download another") {
                if !loaded {
                    Text("Loading quants…")
                } else if m.available.isEmpty {
                    Text(m.onDisk.isEmpty ? "No GGUF files found" : "Every quant is downloaded")
                } else {
                    ForEach(m.available) { quant in
                        Button(quant.label) {
                            // Pass the whole quant — a sharded one pulls every
                            // shard into `<model>/<quant>/`.
                            downloads.startGguf(repoId: repoId, quant: quant) {
                                appState.refreshModels()
                            }
                        }
                    }
                }
            }

            if !m.onDisk.isEmpty {
                // Deletes remove ONE quant. Its siblings are separate models the
                // user didn't ask to delete.
                Menu("Delete") {
                    ForEach(m.onDisk) { quant in
                        Button(quant.label, role: .destructive) { pendingDelete = quant }
                    }
                }
            }
        } label: {
            Text(GgufQuantMenuModel.buttonLabel(
                onDisk: m.onDisk,
                failed: state?.status == .failed,
                hasPartial: downloads.hasPartialDownload(repoId)
            ))
        }
        .font(.callout)
        .controlSize(.small)
        .fixedSize()
        .task {
            guard !loaded else { return }
            remote = await downloads.listGgufFiles(repoId: repoId)
            loaded = true
        }
        .alert("Delete Quant", isPresented: .init(
            get: { pendingDelete != nil },
            set: { if !$0 { pendingDelete = nil } }
        ), presenting: pendingDelete) { quant in
            Button("Cancel", role: .cancel) { pendingDelete = nil }
            Button("Delete", role: .destructive) {
                if let p = path(of: quant) { downloads.removeGgufQuant(at: p) }
                pendingDelete = nil
                appState.refreshModels()
            }
            .keyboardShortcut(.defaultAction)
        } message: { quant in
            Text("Delete the \(quant.label) quant? Other quants of this model stay on disk.")
        }
    }
}

/// The action cell for a multi-variant MLX repo — the directory-shaped twin of
/// `GgufQuantMenu`.
private struct MlxVariantMenu: View {
    let repoId: String
    let variants: [MlxVariant]
    let state: DownloadManager.DownloadState?
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var pendingDelete: MlxVariant?

    /// Each variant is its own ordinary model dir — see `MlxVariantScan.localRepoId`.
    private func localId(_ v: MlxVariant) -> String {
        MlxVariantScan.localRepoId(repoId: repoId, folder: v.folder)
    }
    private func path(of v: MlxVariant) -> String? { downloads.existingModelDir(for: localId(v)) }

    private var menu: MlxVariantMenuModel.Menu {
        MlxVariantMenuModel.build(
            remote: variants,
            onDisk: Set(variants.filter { downloads.isReady(localId($0)) }.map(\.folder))
        )
    }

    private func title(_ v: MlxVariant) -> String {
        v.sizeLabel.isEmpty ? v.label : "\(v.label) · \(v.sizeLabel)"
    }

    var body: some View {
        let m = menu
        Menu {
            if !m.onDisk.isEmpty {
                Section("On this Mac") {
                    ForEach(m.onDisk) { v in
                        Button {
                            guard let p = path(of: v) else { return }
                            Task { _ = await appState.useModelAndAwaitReady(atPath: p) }
                        } label: {
                            let selected = path(of: v) == appState.selectedModelPath
                            Label(
                                selected ? "\(v.label) — in use" : "\(v.label) — use",
                                systemImage: selected ? "checkmark.circle.fill" : "checkmark"
                            )
                        }
                    }
                }
            }

            Section(m.onDisk.isEmpty ? "Choose a quantization" : "Download another") {
                if m.available.isEmpty {
                    Text("Every quantization is downloaded")
                } else {
                    ForEach(m.available) { v in
                        Button(title(v)) {
                            downloads.startMlxVariant(repoId: repoId, variant: v) {
                                appState.refreshModels()
                            }
                        }
                    }
                }
            }

            if !m.onDisk.isEmpty {
                // Deletes remove ONE variant. Its siblings are separate models
                // the user didn't ask to delete.
                Menu("Delete") {
                    ForEach(m.onDisk) { v in
                        Button(v.label, role: .destructive) { pendingDelete = v }
                    }
                }
            }
        } label: {
            Text(MlxVariantMenuModel.buttonLabel(
                onDisk: m.onDisk,
                failed: state?.status == .failed,
                hasPartial: variants.contains { downloads.hasPartialDownload(localId($0)) }
            ))
        }
        .font(.callout)
        .controlSize(.small)
        .fixedSize()
        .alert("Delete Quantization", isPresented: .init(
            get: { pendingDelete != nil },
            set: { if !$0 { pendingDelete = nil } }
        ), presenting: pendingDelete) { v in
            Button("Cancel", role: .cancel) { pendingDelete = nil }
            Button("Delete", role: .destructive) {
                downloads.deleteModel(repoId: localId(v))
                pendingDelete = nil
                appState.refreshModels()
            }
            .keyboardShortcut(.defaultAction)
        } message: { v in
            Text("Delete the \(v.label) build? Other quantizations of this model stay on disk.")
        }
    }
}

// MARK: - Local Model Row

private struct LocalModelRow: View {
    let model: LocalModel
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var confirmDelete = false
    /// Per-row, per-session: clicking the lock arms the trash for THIS row
    /// only, and a refresh re-locks it. The friction is worth one click; the
    /// old dead badge was not worth anything.
    @State private var unlocked = false
    @State private var card: ModelCardRequest?

    /// nil for a bare folder that maps to no Hugging Face repo.
    private var cardRequest: ModelCardRequest? {
        ModelCard.repoId(localName: model.name).map {
            ModelCardRequest(repoId: $0, title: ModelDisplayName.pretty(model.displayLabel))
        }
    }

    private var useState: ModelUseState {
        ModelUseState.resolve(
            selected: appState.selectedModelPath == model.path,
            serverStatus: server.status
        )
    }

    /// Open Finder with the model selected. `path` is the directory for a
    /// safetensors checkpoint and the FILE for one GGUF quant, and
    /// `activateFileViewerSelecting` selects either — which is the behaviour
    /// you want: a quant row reveals its own file, not its repo folder.
    private func revealInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
    }

    private func performDelete() {
        downloads.deleteModel(model, unlocked: unlocked)
        appState.refreshModels()
    }

    var body: some View {
        HStack(spacing: 8) {
            Button { card = cardRequest } label: {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    // The READABLE name. `displayLabel` (the repo id, plus a
                    // quant suffix so two quants of one GGUF repo are two
                    // distinguishable rows) survives as the subtext below —
                    Text(ModelDisplayName.pretty(model.displayLabel))
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    // Drafter checkpoints are real, supported models — they
                    // just aren't loadable as a target on their own. Show a
                    // distinct badge instead of the red "unsupported" warning
                    // that the generic check would otherwise render.
                    // A folder that cannot load says so on the row itself. The
                    // alternative — hiding it — is how two junk folders sat in
                    // this library unnoticed while the server registered both.
                    if let defect = model.defect {
                        Text(defect.label)
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.orange)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.orange.opacity(0.15), in: Capsule())
                            .help(defect.explanation)
                    }
                    if model.kind == .drafter {
                        Text("Drafter")
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.purple)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.purple.opacity(0.15), in: Capsule())
                            .help("Speculative-decoding drafter — pairs with a Gemma 4 base model in Settings, not loadable on its own.")
                    }
                }
                // The id itself, under the readable name.
                Text(model.displayLabel)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                // Metadata caption: params · quant · architecture · engine, so
                // the row actually tells the user what the model is — previously
                // it was just a name and a delete button.
                HStack(spacing: 6) {
                    // For a broken folder the architecture summary is noise —
                    // what it IS matters less than why it cannot load.
                    Text(model.defect?.explanation ?? model.metadataSummary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    // Capability icons mirror the search rows.
                    if model.hasVision {
                        Image(systemName: "eye")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                            .help("Vision (image input)")
                    }
                    if model.hasToolCalling {
                        Image(systemName: "wrench")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                            .help("Tool calling")
                    }
                    // Only flag genuinely unsupported architectures. Drafters
                    // declare `gemma4_assistant` (not in supportedModelTypes)
                    // intentionally — the badge above already explains them.
                    if model.kind != .drafter, !model.isSupportedArchitecture {
                        Text("Unsupported")
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.red.opacity(0.8))
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.red.opacity(0.12), in: Capsule())
                    }
                }
            }
            .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(cardRequest == nil)
            .frame(maxWidth: .infinity, alignment: .leading)

            Text(model.sizeFormatted)
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 90, alignment: .trailing)

            // Use + Delete. "Use" is the missing terminal action the browser
            // never had — before this, the only thing you could do with a model
            // you'd downloaded was throw it away. Once picked, the button is
            // replaced by an In-use badge rather than merely greyed out, so the
            // click produces visible feedback.
            HStack(spacing: 6) {
                if model.isChatPickable {
                    if useState == .idle {
                        UseModelButton(path: model.path, name: model.name)
                    } else {
                        ModelUseBadge(state: useState)
                    }
                } else if let modality = MediaModality(modelType: model.modelType) {
                    // A media checkpoint is a real, loadable, servable model —
                    // it just is not a CHAT model, and until now that meant the
                    // only verb the browser offered for it was Delete. This
                    // does NOT go through `useModelAndAwaitReady`: that starts
                    // the server on the path as its primary chat model, which
                    // for a diffusion checkpoint means the text loader. It
                    // opens the pane that owns the model instead, and lets the
                    // pane load it the way it always has.
                    UseMediaModelButton(modality: modality, name: model.name)
                }
                if ModelRowActions.showsLock(model, unlocked: unlocked) {
                    // Locked, not read-only. This slot used to hold an `Image`
                    // of an external-drive/cloud glyph nobody could read, which
                    // did nothing when clicked. It is a Button now, and clicking
                    // it is how you get the trash. (The old symbol name is
                    // deliberately not spelled here — a source scan in
                    // `ModelRowActionsTests` asserts it is gone from this file.)
                    Button { unlocked = true } label: {
                        Image(systemName: "lock")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .help(ModelRowActions.lockHelp(model))
                } else {
                    Button {
                        confirmDelete = true
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(.red.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .help(model.defect != nil
                          ? "Delete this broken folder"
                          : (model.quantFile != nil ? "Delete this quant" : "Delete model"))
                }

                // Reveal in Finder — rightmost, on EVERY row. Six other panes
                // already had this control; the one pane that is entirely about
                // files on disk did not, so a model you did not recognise could
                // not be located from the app that listed it.
                Button(action: revealInFinder) {
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .font(.callout)
                .help(ModelRowActions.revealHelp(model))
            }
            .frame(width: 150, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        // The alert lives on the ROW so both the trash and the context menu
        // raise the same confirmation — two delete paths with two dialogs is
        // two chances to word the consequence differently.
        .alert(model.quantFile != nil ? "Delete Quant" : "Delete Model", isPresented: $confirmDelete) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) { performDelete() }
                .keyboardShortcut(.defaultAction)
        } message: {
            Text(ModelRowActions.deleteMessage(model))
        }
        .sheet(item: $card) { ModelDetailSheet(request: $0) }
        .contextMenu {
            if model.isChatPickable, useState == .idle {
                Button("Use This Model") {
                    appState.selectedModelPath = model.path
                }
            }
            if cardRequest != nil {
                Button("Model Details\u{2026}") { card = cardRequest }
            }
            Button("Show in Finder", action: revealInFinder)
            Button("Copy Path") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(model.path, forType: .string)
            }
            Button("Copy Model ID") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(model.displayLabel, forType: .string)
            }
            Divider()
            if ModelRowActions.showsTrash(model, unlocked: unlocked) {
                Button("Delete\u{2026}", role: .destructive) { confirmDelete = true }
            } else {
                // Same two-step as the lock button: the menu never deletes
                // another app's model on one click.
                Button("Unlock to Delete") { unlocked = true }
            }
        }
    }
}

// MARK: - Active Download Row

private struct ActiveDownloadRow: View {
    let repoId: String
    let state: DownloadManager.DownloadState
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var modelName: String {
        repoId.components(separatedBy: "/").last ?? repoId
    }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text(modelName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)

                if state.status == .downloading, !state.statusText.isEmpty {
                    Text("[\(state.fileIndex)/\(state.fileCount)] \(state.statusText)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                } else if state.status == .failed, let error = state.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if state.status == .downloading {
                HStack(spacing: 4) {
                    VStack(alignment: .trailing, spacing: 1) {
                        ProgressView(value: state.progress)
                            .frame(width: 80)
                        Text("\(state.percentFormatted) \(state.speedFormatted)")
                            .font(.system(size: 9).monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Button {
                        downloads.cancel(repoId)
                        appState.refreshModels()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Cancel download")
                }
                .frame(width: 116, alignment: .trailing)
            } else if state.status == .failed {
                Button(downloads.hasPartialDownload(repoId) ? "Resume" : "Retry") {
                    downloads.start(repoId: repoId) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
                .frame(width: 90, alignment: .trailing)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }
}
