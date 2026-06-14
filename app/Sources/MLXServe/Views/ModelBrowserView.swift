import SwiftUI

struct ModelBrowserView: View {
    @EnvironmentObject var searchService: HFSearchService
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var showDownloadedOnly = false
    @State private var localFilter = ""

    private var filteredLocalModels: [LocalModel] {
        // The "Downloaded" tab represents what mlx-serve itself fetched.
        // Externally-discovered models (LM Studio) live in the dropdown but not here.
        let mlxServeOnly = appState.localModels.filter { $0.source == .mlxServe }
        if localFilter.isEmpty { return mlxServeOnly }
        return mlxServeOnly.filter { $0.name.localizedCaseInsensitiveContains(localFilter) }
    }

    private var activeDownloads: [(repoId: String, state: DownloadManager.DownloadState)] {
        downloads.downloads
            .filter { $0.value.status == .downloading || $0.value.status == .failed }
            .sorted { $0.key < $1.key }
            .map { (repoId: $0.key, state: $0.value) }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Search bar
            HStack(spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    if showDownloadedOnly {
                        TextField("Filter local models...", text: $localFilter)
                            .textFieldStyle(.plain)
                    } else {
                        TextField("Search models...", text: $searchService.searchQuery)
                            .textFieldStyle(.plain)
                            .onSubmit { Task { await searchService.search() } }
                    }
                }
                .padding(8)
                .background(.quaternary.opacity(0.5))
                .cornerRadius(8)

                if !showDownloadedOnly {
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

                Toggle(isOn: $showDownloadedOnly) {
                    Label("Downloaded", systemImage: "internaldrive")
                }
                .toggleStyle(.button)
                .controlSize(.regular)
            }
            .padding(12)

            Divider()

            if showDownloadedOnly {
                // Local models header
                HStack(spacing: 8) {
                    Text("Model")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Text("Size on Disk")
                        .frame(width: 90, alignment: .trailing)
                    Text("")
                        .frame(width: 64)
                }
                .font(.callout.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.quaternary.opacity(0.3))

                Divider()

                // Local model list + active downloads
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(activeDownloads, id: \.repoId) { item in
                            ActiveDownloadRow(repoId: item.repoId, state: item.state)
                            Divider().padding(.horizontal, 12)
                        }

                        ForEach(filteredLocalModels) { model in
                            LocalModelRow(model: model)
                            Divider().padding(.horizontal, 12)
                        }

                        if activeDownloads.isEmpty && filteredLocalModels.isEmpty {
                            Text("No downloaded models")
                                .foregroundStyle(.secondary)
                                .padding(40)
                        }
                    }
                }

                Divider()

                HStack {
                    Text("\(filteredLocalModels.count) downloaded\(activeDownloads.isEmpty ? "" : ", \(activeDownloads.count) in progress")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
            } else {
                // Column headers
                ColumnHeaderRow(searchService: searchService)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.quaternary.opacity(0.3))

                Divider()

                // HuggingFace model list, with the curated Drafters section
                // pinned at the top so users land on it before scrolling
                // through generic search results.
                //
                // Already-downloaded models are filtered out here — they live
                // in the Downloaded tab, and surfacing them again as
                // "Download" rows is noise. Their counts still match in the
                // footer total.
                let visibleModels = searchService.models.filter { !downloads.isReady($0.id) }
                let hiddenCount = searchService.models.count - visibleModels.count

                ScrollView {
                    LazyVStack(spacing: 0) {
                        DraftersSection()
                        ForEach(visibleModels) { model in
                            ModelBrowserRow(
                                model: model,
                                fitness: searchService.ramFitness(for: model)
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
                        } else if visibleModels.isEmpty && hiddenCount == 0 {
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
                    Text("Showing \(visibleModels.count) models")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if hiddenCount > 0 {
                        Text("· \(hiddenCount) already downloaded")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    Spacer()
                    Text("System RAM: \(MemoryInfo.format(Int64(searchService.systemRAM)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
            }
        }
        .task {
            if searchService.models.isEmpty {
                await searchService.search()
            }
        }
        .onChange(of: showDownloadedOnly) { _, isLocal in
            if isLocal { appState.refreshModels() }
        }
        // Live-refresh on-disk sizes while the Downloaded tab is open and a
        // download is in flight, so completion + growing size show up without
        // the user toggling the button. The task id flips when the tab closes
        // or the active-download set changes, which cancels + re-evaluates the
        // guard — so it self-terminates once everything finishes.
        .task(id: "\(showDownloadedOnly)-\(activeDownloads.count)") {
            guard Self.shouldLivePoll(downloadedTab: showDownloadedOnly,
                                      hasActiveDownloads: !activeDownloads.isEmpty) else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                appState.refreshModels()
            }
        }
    }

    /// Whether the Downloaded tab should live-refresh on-disk sizes: only when
    /// the tab is showing AND a download is in flight. Pulled out so the polling
    /// trigger is unit-testable without driving SwiftUI.
    static func shouldLivePoll(downloadedTab: Bool, hasActiveDownloads: Bool) -> Bool {
        downloadedTab && hasActiveDownloads
    }
}

// MARK: - Column Headers

private struct ColumnHeaderRow: View {
    @ObservedObject var searchService: HFSearchService

    var body: some View {
        HStack(spacing: 8) {
            Text("Model")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text("Cap.")
                .frame(width: 44, alignment: .center)
            SortableHeader("Quant", field: nil, searchService: searchService)
                .frame(width: 54, alignment: .leading)
            SortableHeader("Size", field: nil, searchService: searchService)
                .frame(width: 54, alignment: .trailing)
            // Downloads column: 88 wide so "Downloads" + the sort chevron
            // fit on one line. Was 72, which forced "Down-loads" wrapping
            // (visible in the screenshot bug report).
            SortableHeader("Downloads", field: .downloads, searchService: searchService)
                .frame(width: 88, alignment: .trailing)
            SortableHeader("Likes", field: .likes, searchService: searchService)
                .frame(width: 50, alignment: .trailing)
            // RAM Est. column: 120 wide so GGUF range strings produced by
            // `MemoryInfo.formatRange` ("11.1–30.9 GB", "21.2–55.4 GB",
            // up to "999.9–999.9 GB") render on a single line. Single-value
            // strings ("767 MB", "10.2 GB") were comfortable at 80; the
            // wider budget is what GGUF's min–max range needs. Keep the
            // ModelBrowserRow's RAM cell at the same width or alignment
            // drifts across rows.
            SortableHeader("RAM Est.", field: .estimatedSize, searchService: searchService)
                .frame(width: 120, alignment: .trailing)
            SortableHeader("Updated", field: .lastModified, searchService: searchService)
                .frame(width: 64, alignment: .trailing)
            // Action column: 92 wide so the Download button + menu chevron
            // render without truncation (was 64). The cell content is
            // either a "Download ▾" menu, a trash icon, a Resume/Retry
            // button, or a progress bar — 92 fits the widest.
            Text("")
                .frame(width: 92)
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
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var isReady: Bool { downloads.isReady(model.id) }
    private var state: DownloadManager.DownloadState? { downloads.downloads[model.id] }
    private var disabled: Bool { !model.isCompatible }

    /// For Gemma 4 dense/MoE base rows, the variant whose drafter pairs with
    /// this checkpoint — drives the inline "+drafter" / "✓ paired" chip. nil
    /// for non-Gemma-4 rows (most everything) and for GGUF repos (the
    /// drafter is an MLX-only kernel). The rule lives in
    /// `DownloadManager.drafterPairingVariant` so it's unit-testable.
    private var pairableVariant: GemmaVariant? {
        DownloadManager.drafterPairingVariant(
            repoId: model.id,
            isDrafter: model.isDrafter,
            isGgufRepo: model.isGgufRepo
        )
    }

    var body: some View {
        HStack(spacing: 8) {
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
                    if let v = pairableVariant {
                        DrafterPairChip(variant: v)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Capabilities icons
            HStack(spacing: 3) {
                if model.hasVision {
                    Image(systemName: "eye")
                        .help("Vision (image input)")
                }
                if model.hasToolCalling {
                    Image(systemName: "wrench")
                        .help("Tool calling")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(width: 44, alignment: .center)

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
            .frame(width: 54, alignment: .leading)

            // Size (parsed from model name)
            Text(model.modelSize)
                .font(.callout.monospacedDigit())
                .frame(width: 54, alignment: .trailing)

            // Downloads — width matched to ColumnHeaderRow (88) so the
            // "Downloads" header doesn't have to wrap.
            Text(formatCount(model.downloads ?? 0))
                .font(.callout.monospacedDigit())
                .frame(width: 88, alignment: .trailing)

            // Likes
            Text(formatCount(model.likes ?? 0))
                .font(.callout.monospacedDigit())
                .frame(width: 50, alignment: .trailing)

            // RAM estimate with color indicator — width matches
            // ColumnHeaderRow (120) so GGUF range strings like
            // "21.2–55.4 GB" stay on one line. `.lineLimit(1)` is the
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
            .frame(width: 120, alignment: .trailing)

            // Last updated
            Text(formatRelativeDate(model.lastModifiedDate))
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .trailing)

            // Download button — width matched to ColumnHeaderRow (92) so
            // the "Download ▾" GGUF menu, "Resume"/"Retry" buttons, and
            // the trash icon all render without truncation.
            downloadButton
                .frame(width: 92, alignment: .center)
        }
        .padding(.horizontal, 12)
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

    @ViewBuilder
    private var downloadButton: some View {
        if disabled {
            Image(systemName: "nosign")
                .foregroundStyle(.secondary)
                .font(.caption)
        } else if isReady {
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
        } else if let state, state.status == .downloading {
            HStack(spacing: 4) {
                VStack(spacing: 1) {
                    ProgressView(value: state.fileProgress)
                        .frame(width: 50)
                    Text(state.percentFormatted)
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
        } else if let state, state.status == .completed {
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
        } else if let state, state.status == .failed {
            if model.isGgufRepo {
                GgufDownloadMenu(repoId: model.id, label: "Retry")
            } else {
                Button(downloads.hasPartialDownload(model.id) ? "Resume" : "Retry") {
                    downloads.start(repoId: model.id) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
            }
        } else {
            if model.isGgufRepo {
                // GGUF repos ship many quants — pick one from a menu.
                GgufDownloadMenu(repoId: model.id, label: "Download")
            } else {
                Button(downloads.hasPartialDownload(model.id) ? "Resume" : "Download") {
                    downloads.start(repoId: model.id) { appState.refreshModels() }
                }
                .font(.callout)
                .controlSize(.small)
            }
        }
    }
}

/// Download button for a GGUF repo: a menu of the repo's quant files. The list
/// is fetched lazily from the HF tree API the first time the menu is shown.
private struct GgufDownloadMenu: View {
    let repoId: String
    let label: String
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var quants: [String] = []
    @State private var loaded = false

    var body: some View {
        Menu {
            if !loaded {
                Text("Loading quants…")
            } else if quants.isEmpty {
                Text("No GGUF files found")
            } else {
                ForEach(quants, id: \.self) { file in
                    Button(DownloadManager.quantLabel(forFilename: file)) {
                        downloads.startGguf(repoId: repoId, ggufFilename: file) {
                            appState.refreshModels()
                        }
                    }
                }
            }
        } label: {
            Text(label)
        }
        .font(.callout)
        .controlSize(.small)
        .fixedSize()
        .task {
            guard !loaded else { return }
            quants = await downloads.listGgufFiles(repoId: repoId)
            loaded = true
        }
    }
}

// MARK: - Local Model Row

private struct LocalModelRow: View {
    let model: LocalModel
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @State private var confirmDelete = false

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(model.name)
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

            Button {
                confirmDelete = true
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(.red.opacity(0.7))
            }
            .buttonStyle(.plain)
            .font(.callout)
            .help("Delete model")
            .frame(width: 64, alignment: .center)
            .alert("Delete Model", isPresented: $confirmDelete) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    downloads.deleteModel(model)
                    appState.refreshModels()
                }
            } message: {
                Text("Delete \(model.name)? This will remove all downloaded files.")
            }
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

// MARK: - Drafter Pair Chip

/// Small inline indicator on Gemma 4 base-model rows that surfaces the
/// matching drafter. Two states:
///   - **On disk**: green "✓ Drafter" chip — silently confirms the pairing
///     is ready, hides the download CTA.
///   - **Not on disk**: clickable "Pair with drafter (+30-40%)" chip that
///     kicks off `DownloadManager.download(repoId:)` for the matching
///     `*-it-assistant-bf16` repo.
private struct DrafterPairChip: View {
    let variant: GemmaVariant
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var repoId: String { variant.drafterRepoId }
    private var isReady: Bool { downloads.isReady(repoId) }
    private var inFlight: Bool { downloads.downloads[repoId]?.status == .downloading }

    var body: some View {
        if isReady {
            Text("✓ Drafter")
                .font(.system(size: 10).weight(.medium))
                .foregroundStyle(.green)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.green.opacity(0.10))
                .clipShape(Capsule())
                .help("\(variant.drafterDirName) is downloaded and ready to pair.")
        } else if inFlight {
            Text("Drafter…")
                .font(.system(size: 10).weight(.medium))
                .foregroundStyle(.purple)
                .padding(.horizontal, 5)
                .padding(.vertical, 1)
                .background(Color.purple.opacity(0.10))
                .clipShape(Capsule())
        } else {
            Button {
                Task {
                    await downloads.download(repoId: repoId)
                    appState.refreshModels()
                }
            } label: {
                Text("Pair with drafter +30-40%")
                    .font(.system(size: 10).weight(.medium))
                    .foregroundStyle(.purple)
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(Color.purple.opacity(0.10))
                    .clipShape(Capsule())
            }
            .buttonStyle(.plain)
            .help("Download \(variant.drafterDirName) for +30-40% on code & agents (Gemma 4 only).")
        }
    }
}

// MARK: - Curated Drafters Section

/// Top-of-browser block listing the four published Gemma 4 assistant drafter
/// checkpoints. Surfaced here so users find them without manually searching
/// "assistant-bf16". Each row reuses the underlying `DownloadManager` state
/// machine — same Resume / Download / Delete affordances as the main list.
private struct DraftersSection: View {
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    /// Collapsed by default — pairing a drafter is a one-time setup step, so
    /// we don't want this taking up vertical space above the search results
    /// after the user's already done it once.
    @State private var expanded = false

    private var rows: [DrafterCatalogRow] {
        GemmaVariant.allCases.map { v in
            DrafterCatalogRow(
                variant: v,
                repoId: v.drafterRepoId,
                pairsWith: "for \(v.label)",
                sizeEstimate: Self.sizeEstimate(for: v)
            )
        }
    }

    private static func sizeEstimate(for v: GemmaVariant) -> String {
        // bf16 sizes (the uniform suffix used by drafterRepoId).
        switch v {
        case .E2B:        return "~80 MB"
        case .E4B:        return "~120 MB"
        case .gemma12B:   return "~850 MB"
        case .gemma31B:   return "~150 MB"
        case .moe26B:     return "~120 MB"
        }
    }

    /// Count of drafters already on disk — drives the "X of 4 ready" hint
    /// in the collapsed header so users see at a glance whether they need
    /// to expand it.
    private var readyCount: Int {
        rows.filter { downloads.isReady($0.repoId) }.count
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) { expanded.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 10)
                    Image(systemName: "sparkles")
                        .foregroundStyle(.purple)
                    Text("Drafters")
                        .font(.callout.weight(.semibold))
                    Text("Pair with a Gemma 4 base model for +27–40% on code & agents")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer()
                    Text("\(readyCount) of \(rows.count) downloaded")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
            .buttonStyle(.plain)

            if expanded {
                ForEach(rows) { row in
                    DrafterCatalogRowView(row: row)
                    Divider().padding(.horizontal, 12)
                }
            }
        }
        .background(Color.purple.opacity(0.04))
    }
}

private struct DrafterCatalogRow: Identifiable {
    let variant: GemmaVariant
    let repoId: String
    let pairsWith: String
    let sizeEstimate: String
    var id: String { repoId }
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
