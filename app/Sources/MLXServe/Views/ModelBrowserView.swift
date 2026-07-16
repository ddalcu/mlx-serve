import SwiftUI

/// Model Browser: a sidebar over six destinations (`ModelBrowserSection`).
///
/// It used to be one pane with a `Toggle("Downloaded")` push-button that swapped
/// the data source underneath you, and it deleted on-disk models from the search
/// results — so the model you just finished downloading vanished at 100%. Now
/// Discover marks what you own instead of hiding it, Downloads is a first-class
/// queue rather than rows stapled to the top of a list, and My Models shows
/// everything the tray picker offers rather than only what we fetched ourselves.
struct ModelBrowserView: View {
    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    @State private var selection: ModelBrowserSection? = .recommended
    @State private var localFilter = ""

    private var section: ModelBrowserSection { selection ?? .recommended }

    /// Downloading *or* failed — both belong in the queue and both earn a badge.
    private var activeDownloads: [(repoId: String, state: DownloadManager.DownloadState)] {
        downloads.downloads
            .filter { $0.value.status == .downloading || $0.value.status == .failed }
            .sorted { $0.key < $1.key }
            .map { (repoId: $0.key, state: $0.value) }
    }

    /// Media bundles fully on disk, across all four modality catalogs.
    private var mediaReadyCount: Int {
        func readyCount<P: MediaModelPreset>(_ presets: [P]) -> Int {
            presets.filter { downloads.bundleReady($0.bundle) }.count
        }
        return readyCount(ImageModelPreset.all) + readyCount(AudioModelPreset.all)
            + readyCount(VideoModelPreset.all) + readyCount(MusicModelPreset.all)
    }

    private var badges: ModelBrowserBadgeCounts {
        ModelBrowserBadgeCounts(
            myModels: appState.localModels.count,
            activeDownloads: activeDownloads.count,
            draftersReady: GemmaVariant.allCases.filter { downloads.isReady($0.drafterRepoId) }.count,
            mediaReady: mediaReadyCount
        )
    }

    var body: some View {
        NavigationSplitView {
            List(ModelBrowserSection.allCases, selection: $selection) { item in
                NavigationLink(value: item) {
                    Label {
                        HStack(spacing: 6) {
                            Text(item.title)
                            Spacer(minLength: 4)
                            if item == .downloads, !activeDownloads.isEmpty {
                                ProgressView()
                                    .controlSize(.small)
                                    .scaleEffect(0.6)
                            }
                            if let badge = badges.badge(for: item) {
                                Text(badge)
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 1)
                                    .background(.quaternary, in: Capsule())
                            }
                        }
                    } icon: {
                        Image(systemName: item.systemImage)
                    }
                }
            }
            .navigationSplitViewColumnWidth(
                min: ModelBrowserMetrics.sidebarMinWidth,
                ideal: ModelBrowserMetrics.sidebarIdealWidth,
                max: ModelBrowserMetrics.sidebarMaxWidth
            )
        } detail: {
            detail
                .frame(minWidth: ModelBrowserMetrics.minDetailWidth)
        }
        .task {
            if searchService.models.isEmpty {
                await searchService.search()
            }
        }
        .onChange(of: selection) { _, _ in appState.refreshModels() }
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
        case .drafters:     DraftersPane()
        case .media:        MediaPane()
        }
    }
}

// MARK: - Recommended

/// Every curated Gemma 4 / Qwen 3.5-3.6 checkpoint, grouped by family and
/// explained in plain English. This is the friendly front door for someone
/// who has never picked a local model before — Discover's HuggingFace search
/// (with its 1M+ repos, quant/pull-count columns, and RAM-fitness dots)
/// assumes you already know roughly what you're looking for.
private struct RecommendedPane: View {
    private var physicalMemory: UInt64 { ProcessInfo.processInfo.physicalMemory }
    private var ramLabel: String { MemoryInfo.format(Int64(physicalMemory)) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Your Mac has \(ramLabel) of memory")
                        .font(.title3.weight(.semibold))
                    Text("Here are the recommended models for you to download.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                ModelGroupSection(
                    title: "Gemma 4",
                    subtitle: "Google's Gemma 4 family.",
                    systemImage: "g.circle",
                    tint: .blue
                ) {
                    RecommendedFamilyRows(picks: RecommendedModelPick.gemmaCatalog, physicalMemoryBytes: physicalMemory)
                }

                ModelGroupSection(
                    title: "Qwen",
                    subtitle: "Alibaba's Qwen 3.5/3.6 family — the larger checkpoints ship a native speed boost.",
                    systemImage: "q.circle",
                    tint: .teal
                ) {
                    RecommendedFamilyRows(picks: RecommendedModelPick.qwenCatalog, physicalMemoryBytes: physicalMemory)
                }

                ModelGroupSection(
                    title: "Largest models (96 GB+ RAM)",
                    subtitle: "The biggest models this app runs — DeepSeek-V4-Flash (ds4) and Tencent's 295B Hunyuan 3 — for Macs with a lot of memory.",
                    systemImage: "memorychip",
                    tint: .red
                ) {
                    RecommendedFamilyRows(picks: RecommendedModelPick.largestCatalog, physicalMemoryBytes: physicalMemory)
                }
            }
            .padding(16)
        }
        .navigationTitle("Recommended")
    }
}

/// A family's rows, split into what this Mac's RAM covers and what it
/// doesn't. The first group renders inline; the second sits behind a
/// collapsed "Requires more RAM" disclosure — nothing is ever dropped from
/// the list, it's just deferred until the user asks to see it, rather than
/// cluttering the default view (or, as an earlier iteration did, rendering
/// inline at reduced opacity).
private struct RecommendedFamilyRows: View {
    let picks: [RecommendedModelPick]
    let physicalMemoryBytes: UInt64
    @State private var showsRequiresMoreRAM = false

    var body: some View {
        let split = picks.partitionedByRequirements(physicalMemoryBytes: physicalMemoryBytes)

        ForEach(split.fits) { pick in
            RecommendedModelListRow(pick: pick, physicalMemoryBytes: physicalMemoryBytes)
            Divider().padding(.horizontal, 12)
        }

        if !split.requiresMoreRAM.isEmpty {
            Button {
                withAnimation { showsRequiresMoreRAM.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: showsRequiresMoreRAM ? "chevron.down" : "chevron.right")
                        .font(.caption2.weight(.semibold))
                    Text("Requires more RAM (\(split.requiresMoreRAM.count))")
                        .font(.caption.weight(.medium))
                    Spacer()
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            Divider().padding(.horizontal, 12)

            if showsRequiresMoreRAM {
                ForEach(split.requiresMoreRAM) { pick in
                    RecommendedModelListRow(pick: pick, physicalMemoryBytes: physicalMemoryBytes)
                    Divider().padding(.horizontal, 12)
                }
            }
        }
    }
}

/// One list row for a chat-model recommendation: name, tagline, a plain-
/// English description underneath, capability chips, and the Download/Use
/// action — the list-style analogue of the Media pane's `MediaModelRow`, with
/// the richer copy this pane needs.
private struct RecommendedModelListRow: View {
    let pick: RecommendedModelPick
    let physicalMemoryBytes: UInt64

    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var confirmDelete = false

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

    /// Soft signal only — sorts the pick behind the family's "Requires more
    /// RAM" disclosure and explains why there. Never blocks downloading or
    /// using it (same "warn, don't gate" policy as Discover's RAM-fitness dot
    /// and ImageGenView's oversized-model alert).
    private var meetsRequirements: Bool {
        pick.meetsSystemRequirements(physicalMemoryBytes: physicalMemoryBytes)
    }

    /// The on-disk model this row's repo resolves to, once downloaded —
    /// mirrors `ModelBrowserRow.usableModel`.
    private var usableModel: LocalModel? {
        // A GGUF quant's LocalModel.path is the FILE, not the repo dir — resolve
        // against the specific quant so "Use" loads exactly this pick.
        let path = pick.ggufFilename != nil ? ggufFilePath : downloads.existingModelDir(for: pick.repoId)
        return ModelBrowserUse.pickableModel(atPath: path, in: appState.localModels)
    }

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(pick.name)
                        .font(.callout.weight(.medium))
                    Text(pick.tagline)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text(pick.blurb)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 6) {
                    ForEach(pick.highlights, id: \.self) { chip in
                        Text(chip)
                            .font(.system(size: 9).weight(.medium))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.quaternary, in: Capsule())
                    }
                }

                if !meetsRequirements {
                    Label(
                        "Needs about \(String(format: "%.0f", pick.approxRAMNeededGB)) GB of RAM — your Mac has \(MemoryInfo.format(Int64(physicalMemoryBytes)))",
                        systemImage: "exclamationmark.triangle"
                    )
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.orange)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .trailing, spacing: 4) {
                Text(pick.sizeLabel)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                actionControl
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
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
                } message: {
                    Text("Delete \(pick.name)? This will remove all downloaded files.")
                }
            }
        } else if let state, state.status == .downloading {
            HStack(spacing: 6) {
                VStack(alignment: .trailing, spacing: 1) {
                    ProgressView(value: state.fileProgress)
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
                            Text(ModelBrowserUse.groupTitle(group.source))
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
                    Text("Start one from Discover or Drafters.")
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

// MARK: - Drafters pane

/// The curated drafter catalog, previously a collapsed disclosure pinned above
/// the search results. As its own destination it's discoverable without
/// competing for vertical space with the model list.
private struct DraftersPane: View {
    @EnvironmentObject var downloads: DownloadManager

    private var rows: [DrafterCatalogRow] {
        GemmaVariant.allCases.map { v in
            DrafterCatalogRow(
                variant: v,
                repoId: v.drafterRepoId,
                pairsWith: "for \(v.label)",
                sizeEstimate: DrafterCatalogRow.sizeEstimate(for: v)
            )
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundStyle(.purple)
                    Text("What's a drafter?")
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                }
                Text("A small, fast helper model that runs alongside your main chat model. It guesses several upcoming words at once, and the main model quickly double-checks those guesses — when they're right (which is often), you get the exact same answer, just noticeably faster. It never changes what the model says, only how quickly it says it. This helps most with coding and multi-step agent work (using tools, editing files): +27–40% faster in our tests. A drafter only works alongside the specific Gemma 4 model size it's built for — pick the one matching your chat model below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(Color.purple.opacity(0.06))

            Divider()

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(rows) { row in
                        DrafterCatalogRowView(row: row)
                        Divider().padding(.horizontal, 12)
                    }
                }
            }
        }
        .navigationTitle("Drafters")
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
                    ForEach(ImageModelPreset.all) { MediaModelRow(preset: $0, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Audio",
                    subtitle: "Text-to-speech, with optional voice cloning.",
                    systemImage: "waveform",
                    tint: .green
                ) {
                    ForEach(AudioModelPreset.all) { MediaModelRow(preset: $0, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Video",
                    subtitle: "Text/image-to-video, with optional audio.",
                    systemImage: "film",
                    tint: .indigo
                ) {
                    ForEach(VideoModelPreset.all) { MediaModelRow(preset: $0, physicalMemoryBytes: physicalMemory) }
                }
                ModelGroupSection(
                    title: "Music",
                    subtitle: "Text-to-music, with optional lyrics.",
                    systemImage: "music.note",
                    tint: .orange
                ) {
                    ForEach(MusicModelPreset.all) { MediaModelRow(preset: $0, physicalMemoryBytes: physicalMemory) }
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
/// near-duplicate views. Download/progress/retry mirrors `BundleDownloadBar`;
/// unlike a chat model there's no "Use" (each gen pane keeps its own sticky
/// model selection), so the terminal state is just on-disk + Delete.
private struct MediaModelRow<Preset: MediaModelPreset>: View {
    let preset: Preset
    let physicalMemoryBytes: UInt64
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var confirmDelete = false

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
    }

    @ViewBuilder
    private var actionControl: some View {
        if isReady {
            HStack(spacing: 6) {
                Text("✓ On disk")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
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
                } message: {
                    Text("Delete \(preset.name)? This will remove the downloaded files.")
                }
            }
        } else if let active, active.state.status == .downloading {
            HStack(spacing: 6) {
                VStack(alignment: .trailing, spacing: 1) {
                    ProgressView(value: active.state.fileProgress)
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

    var body: some View {
        HStack(spacing: ModelBrowserMetrics.columnSpacing) {
            // Model name — takes all remaining space
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
            progress: state?.fileProgress ?? 0
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

    @ViewBuilder
    private var actionCell: some View {
        // A GGUF repo is a FOLDER OF QUANTS, not a model, so it never reaches a
        // terminal "on disk" state: owning Q4_K_M says nothing about whether you
        // also want Q8_0. Its cell stays a menu that marks what you have and
        // keeps offering what you don't — the old `.onDisk` collapse to
        // "✓ On disk" + trash left no way back to the quant picker.
        if model.isGgufRepo, action != .unsupported {
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
                    downloads.start(repoId: model.id) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)

            case .notDownloaded(let resumable):
                Button(resumable ? "Resume" : "Download") {
                    downloads.start(repoId: model.id) { appState.refreshModels() }
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
                downloads.cancel(model.id)
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
        } message: {
            Text("Delete \(model.modelName)? This will remove all downloaded files.")
        }
    }
}

/// The action cell for a GGUF repo: one menu covering every quant, in every
/// state.
///
/// A GGUF repo ships a folder of quants and the user picks which to run, so this
/// control never "completes". Quants on disk carry a ✓ and load on click; the
/// rest download on click; each can be deleted individually. The repo's file
/// list is fetched lazily from the HF tree API the first time the menu opens —
/// what's on DISK, though, is read every render, so a finished download shows up
/// without a refetch.
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
        } message: { quant in
            Text("Delete the \(quant.label) quant? Other quants of this model stay on disk.")
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

    private var useState: ModelUseState {
        ModelUseState.resolve(
            selected: appState.selectedModelPath == model.path,
            serverStatus: server.status
        )
    }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    // `displayLabel`, so two quants of one GGUF repo are two
                    // distinguishable rows rather than two identical ones.
                    Text(model.displayLabel)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    // Drafter checkpoints are real, supported models — they
                    // just aren't loadable as a target on their own. Show a
                    // distinct badge instead of the red "unsupported" warning
                    // that the generic check would otherwise render.
                    if model.kind == .drafter {
                        Text("Drafter")
                            .font(.system(size: 10).weight(.medium))
                            .foregroundStyle(.purple)
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(Color.purple.opacity(0.15), in: Capsule())
                            .help("Speculative-decoding drafter — pairs with a Gemma 4 base model in Settings, not loadable on its own.")
                    }
                }
                // Metadata caption: params · quant · architecture · engine, so
                // the row actually tells the user what the model is — previously
                // it was just a name and a delete button.
                HStack(spacing: 6) {
                    Text(model.metadataSummary)
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
                }
                if let reason = model.externalReadOnlyReason {
                    // Read-only: this model lives outside ~/.mlx-serve (LM Studio,
                    // the HF hub cache, or a user-added custom folder). The app
                    // loads it but never deletes into another tool's / the user's
                    // tree, so we surface a badge instead of a trash.
                    Image(systemName: "externaldrive.badge.icloud")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .help(reason)
                } else {
                    Button {
                        confirmDelete = true
                    } label: {
                        Image(systemName: "trash")
                            .foregroundStyle(.red.opacity(0.7))
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .help(model.quantFile != nil ? "Delete this quant" : "Delete model")
                    .alert(model.quantFile != nil ? "Delete Quant" : "Delete Model", isPresented: $confirmDelete) {
                        Button("Cancel", role: .cancel) {}
                        Button("Delete", role: .destructive) {
                            downloads.deleteModel(model)
                            appState.refreshModels()
                        }
                    } message: {
                        // A GGUF row is ONE quant of a repo — deleting it must not
                        // promise (or perform) the removal of its siblings.
                        Text(model.quantFile != nil
                             ? "Delete \(model.displayLabel)? Other quants of this model stay on disk."
                             : "Delete \(model.name)? This will remove all downloaded files.")
                    }
                }
            }
            .frame(width: 120, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
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
                        ProgressView(value: state.fileProgress)
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

// MARK: - Curated Drafters catalog

private struct DrafterCatalogRow: Identifiable {
    let variant: GemmaVariant
    let repoId: String
    let pairsWith: String
    let sizeEstimate: String
    var id: String { repoId }

    /// bf16 sizes (the uniform suffix used by `drafterRepoId`).
    static func sizeEstimate(for v: GemmaVariant) -> String {
        switch v {
        case .E2B:        return "~80 MB"
        case .E4B:        return "~120 MB"
        case .gemma12B:   return "~850 MB"
        case .gemma31B:   return "~150 MB"
        case .moe26B:     return "~120 MB"
        }
    }
}

private struct DrafterCatalogRowView: View {
    let row: DrafterCatalogRow
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var confirmDelete = false

    private var isReady: Bool { downloads.isReady(row.repoId) }
    private var state: DownloadManager.DownloadState? { downloads.downloads[row.repoId] }

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text(row.variant.drafterDirName)
                    .font(.callout.weight(.medium))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    Text(row.pairsWith)
                        .font(.caption)
                        .foregroundStyle(.purple)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(row.sizeEstimate)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            actionControl
                .frame(width: 110, alignment: .trailing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private var actionControl: some View {
        if isReady {
            HStack(spacing: 6) {
                Text("✓ Available")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.green)
                Button {
                    confirmDelete = true
                } label: {
                    Image(systemName: "trash")
                        .foregroundStyle(.red.opacity(0.7))
                }
                .buttonStyle(.plain)
                .font(.callout)
                .help("Delete drafter")
                .alert("Delete Drafter", isPresented: $confirmDelete) {
                    Button("Cancel", role: .cancel) {}
                    Button("Delete", role: .destructive) {
                        downloads.deleteModel(repoId: row.repoId)
                        appState.refreshModels()
                    }
                } message: {
                    Text("Delete \(row.variant.drafterDirName)?")
                }
            }
        } else if let s = state, s.status == .downloading {
            VStack(spacing: 1) {
                ProgressView(value: s.fileProgress)
                    .frame(width: 80)
                Text(s.percentFormatted)
                    .font(.system(size: 9).monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        } else {
            Button(downloads.hasPartialDownload(row.repoId) ? "Resume" : "Download") {
                Task {
                    await downloads.download(repoId: row.repoId)
                    appState.refreshModels()
                }
            }
            .font(.callout)
            .controlSize(.small)
        }
    }
}
