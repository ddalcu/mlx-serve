import Foundation

/// Weight format filter for the Model Browser's HuggingFace search. MLX and GGUF
/// are queried via distinct HF `filter` tags; `both` runs each and merges.
enum ModelFormat: String, CaseIterable, Identifiable, Sendable {
    case mlx
    case gguf
    case both

    var id: String { rawValue }

    var label: String {
        switch self {
        case .mlx: return "MLX"
        case .gguf: return "GGUF"
        case .both: return "Both"
        }
    }

    /// HF API `filter` tags to query. `both` queries each tag and merges results.
    var filterTags: [String] {
        switch self {
        case .mlx: return ["mlx"]
        case .gguf: return ["gguf"]
        case .both: return ["mlx", "gguf"]
        }
    }
}

@MainActor
class HFSearchService: ObservableObject {
    @Published var models: [HFModel] = []
    @Published var isLoading = false
    @Published var error: String?
    @Published var searchQuery = ""
    // Default to MLX (the native engine's format), not Both — mixing GGUF
    // into the default results buries the recommended MLX builds. GGUF/Both
    // stay one click away in the browser's segmented picker.
    @Published var format: ModelFormat = .mlx
    @Published var sortField: HFSortField = .downloads
    @Published var sortDescending = true
    @Published var hasMore = true

    private let pageSize = 50
    private var currentSkip = 0
    let systemRAM = ProcessInfo.processInfo.physicalMemory

    /// All fetched models (unsorted). Sorting is applied client-side on `models`.
    private var fetchedModels: [HFModel] = []

    var sortedModels: [HFModel] {
        fetchedModels.sorted { a, b in
            let result: Bool
            switch sortField {
            case .downloads:
                result = (a.downloads ?? 0) > (b.downloads ?? 0)
            case .likes:
                result = (a.likes ?? 0) > (b.likes ?? 0)
            case .lastModified:
                let da = a.lastModifiedDate ?? .distantPast
                let db = b.lastModifiedDate ?? .distantPast
                result = da > db
            case .estimatedSize:
                result = a.estimatedSizeBytes > b.estimatedSizeBytes
            }
            return sortDescending ? result : !result
        }
    }

    func search() async {
        currentSkip = 0
        fetchedModels = []
        models = []
        hasMore = true
        error = nil
        await fetchPage()
    }

    func loadMore() async {
        guard hasMore, !isLoading else { return }
        await fetchPage()
    }

    func sort(by field: HFSortField) {
        if sortField == field {
            sortDescending.toggle()
        } else {
            sortField = field
            sortDescending = true
        }
        models = sortedModels
    }

    func ramFitness(for model: HFModel) -> RAMFitness {
        let size = model.ramEstimateBytes
        guard size > 0 else { return .unknown }
        let ram = Double(systemRAM)
        let ratio = Double(size) / ram
        if ratio < 0.60 { return .fits }
        if ratio < 0.85 { return .tight }
        return .wontFit
    }

    /// True when the host has at least 96 GB of physical RAM — gate for surfacing
    /// the DeepSeek-V4-Flash ds4 download entry (model card claims 96 GB+ minimum
    /// for the IQ2XXS checkpoint).
    static func isSystemRAM96Plus() -> Bool {
        return ProcessInfo.processInfo.physicalMemory >= (96 * (UInt64(1) << 30))
    }

    private func fetchPage() async {
        isLoading = true
        defer { isLoading = false }

        // Run one query per active filter tag (MLX / GGUF, or both), all at the
        // same skip so they paginate in lockstep, then merge + dedup (a repo can
        // carry both tags). hasMore stays true while any sub-query fills a page.
        var rawAll: [HFModel] = []
        var anyFull = false
        for tag in format.filterTags {
            guard let raw = await fetchFilterPage(tag: tag) else { continue }
            if raw.count >= pageSize { anyFull = true }
            rawAll.append(contentsOf: raw)
        }

        // Filter out MLX-format DeepSeek-V4 checkpoints — the Zig MLX path
        // rejects them; users grab the GGUF + ds4 entry from the built-in catalog.
        let filtered = rawAll.filter { !$0.id.lowercased().contains("deepseek-v4-flash") }

        var seen = Set(fetchedModels.map { $0.id })
        for m in filtered where !seen.contains(m.id) {
            seen.insert(m.id)
            fetchedModels.append(m)
        }
        models = sortedModels
        currentSkip += pageSize
        hasMore = anyFull

        // Fetch tree-API sizes for any row that came back without a size.
        // Covers two distinct shapes inside `fetchFallbackSizes` →
        // `parseFallbackSize`:
        //   - MLX safetensors repos that didn't ship `safetensors.parameters`
        //     in the search response (older HF metadata): sum the `.safetensors`
        //     shards from the tree.
        //   - GGUF repos: HF never reports a single size for these (each quant
        //     is a separate file). We collect the non-mmproj `.gguf` sizes
        //     and surface a min/max range so the RAM Est column shows
        //     "1.7–8.5 GB" instead of "Unknown".
        // Earlier this filter excluded `isGgufRepo` because the per-quant
        // fetch was deferred to download time — that left the row useless
        // for browsing. Don't add that gate back without first updating
        // `parseFallbackSize` / `HFModel.ramEstimate`.
        let missing = filtered.filter { Self.needsFallbackFetch($0) }
        if !missing.isEmpty {
            Task { await fetchFallbackSizes(for: missing) }
        }
    }

    /// Fetch one page for a single HF `filter` tag at the current skip. Returns
    /// the raw decoded list (pre-dedup/DSV4-filter) or nil on error.
    private func fetchFilterPage(tag: String) async -> [HFModel]? {
        guard let url = Self.searchURL(query: searchQuery, filter: tag, skip: currentSkip, limit: pageSize) else {
            error = "Invalid search URL"
            return nil
        }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                error = "HuggingFace API error (HTTP \(code))"
                return nil
            }
            return try JSONDecoder().decode([HFModel].self, from: data)
        } catch {
            self.error = error.localizedDescription
            return nil
        }
    }

    /// Build the HF API query URL for the given page parameters. Exposed for tests.
    nonisolated static func searchURL(query: String, filter: String, skip: Int, limit: Int) -> URL? {
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "filter", value: filter),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "\(limit)"),
            URLQueryItem(name: "skip", value: "\(skip)"),
        ]
        let q = query.trimmingCharacters(in: .whitespaces)
        if !q.isEmpty { items.append(URLQueryItem(name: "search", value: q)) }
        for field in ["safetensors", "lastModified", "likes", "downloads", "tags", "pipeline_tag"] {
            items.append(URLQueryItem(name: "expand[]", value: field))
        }
        components.queryItems = items
        return components.url
    }

    /// Fetch file sizes from the tree API for models missing metadata.
    /// Branches inside `parseFallbackSize` on what the repo actually ships:
    /// safetensors → sum across shards; single GGUF → that file; multi-quant
    /// GGUF → min/max range so `HFModel.ramEstimate` can surface "1.7–8.5 GB"
    /// instead of an opaque "Unknown".
    private func fetchFallbackSizes(for models: [HFModel]) async {
        await withTaskGroup(of: (String, FallbackSize)?.self) { group in
            for model in models {
                group.addTask {
                    guard let url = URL(string: "https://huggingface.co/api/models/\(model.id)/tree/main") else { return nil }
                    guard let (data, response) = try? await URLSession.shared.data(from: url),
                          let http = response as? HTTPURLResponse, http.statusCode == 200,
                          let raw = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return nil }
                    let entries = Self.treeEntries(from: raw)
                    guard let fb = Self.parseFallbackSize(files: entries) else { return nil }
                    return (model.id, fb)
                }
            }
            for await result in group {
                guard let (id, fb) = result else { continue }
                if let idx = fetchedModels.firstIndex(where: { $0.id == id }) {
                    switch fb {
                    case .safetensorsSum(let bytes), .ggufSingle(let bytes):
                        fetchedModels[idx].fallbackSizeBytes = bytes
                    case .ggufRange(let lo, let hi):
                        fetchedModels[idx].ggufMinSizeBytes = lo
                        fetchedModels[idx].ggufMaxSizeBytes = hi
                        // Also seed fallbackSizeBytes with the max so sorting
                        // and any code reading `estimatedSizeBytes` still has
                        // a single number to work with (conservative-high).
                        fetchedModels[idx].fallbackSizeBytes = hi
                    }
                }
            }
        }
        self.models = sortedModels
    }

    /// Whether a model row needs a tree-API fallback fetch to populate its
    /// size. True when the search response didn't carry weight metadata
    /// (`estimatedSizeBytes == 0`) AND the row is one mlx-serve can actually
    /// load (`isCompatible`). Crucially: GGUF rows pass — they NEVER carry
    /// safetensors metadata, so `estimatedSizeBytes == 0` is the norm for
    /// them and they're exactly the case the GGUF range in
    /// `parseFallbackSize` was added to serve. Pulled out for unit testing
    /// — gating GGUF out here was the bug that re-surfaced "Unknown" in the
    /// Model Browser even after `parseFallbackSize` learned about GGUF.
    nonisolated static func needsFallbackFetch(_ model: HFModel) -> Bool {
        model.estimatedSizeBytes == 0 && model.isCompatible
    }

    // MARK: - Pure tree-parsing helpers (testable without a network)

    /// Typed view of a single HF tree-API entry — extracted so
    /// `parseFallbackSize` is a pure function over plain values rather than
    /// `[[String: Any]]` (which is awkward to construct in tests).
    struct TreeFileEntry: Equatable, Sendable {
        let path: String
        let size: Int64
    }

    /// Result of inspecting a repo's tree for a fallback size. Distinguishes
    /// the three shapes mlx-serve cares about so HFSearchService can route
    /// to the right field on `HFModel`.
    enum FallbackSize: Equatable, Sendable {
        case safetensorsSum(Int64)                    // all .safetensors shards summed
        case ggufSingle(Int64)                        // a single non-mmproj .gguf
        case ggufRange(min: Int64, max: Int64)        // smallest + largest non-mmproj .gguf
    }

    /// Convert the raw `[[String: Any]]` HF tree response into typed entries.
    /// Drops anything that isn't a `file`, has an unreadable size, or has no
    /// path. `nonisolated` so tests can drive it without spinning up a
    /// `@MainActor` instance.
    nonisolated static func treeEntries(from raw: [[String: Any]]) -> [TreeFileEntry] {
        raw.compactMap { entry -> TreeFileEntry? in
            guard (entry["type"] as? String) == "file",
                  let path = entry["path"] as? String else { return nil }
            let size: Int64
            if let i64 = entry["size"] as? Int64 { size = i64 }
            else if let i = entry["size"] as? Int { size = Int64(i) }
            else { return nil }
            return TreeFileEntry(path: path, size: size)
        }
    }

    /// Decide which fallback-size shape to record for a model given its
    /// tree-API file list. Safetensors wins when present (matches existing
    /// behavior). Otherwise consider non-mmproj top-level `.gguf` files
    /// ≥ 1 MB — single → `ggufSingle`, two or more → `ggufRange` (min, max).
    /// Returns nil when the repo has neither (the caller leaves the model
    /// with `ramEstimate == "Unknown"`).
    ///
    /// Why exclude:
    /// - mmproj sidecars (`mmproj-*.gguf`): CLIP vision/audio encoders, not
    ///   loadable as LLMs by either engine — `DownloadManager.isMmprojGguf`
    ///   filters them everywhere else; mirror that here so the row's size
    ///   doesn't pull in a 200 MB sidecar that the user can't actually pick.
    /// - subdir files (path contains `/`): split-shard GGUF layouts the
    ///   single-file download path can't reassemble.
    /// - files < 1 MB: README stubs / LFS pointer files that occasionally
    ///   show up with non-zero sizes; would skew the min downward.
    nonisolated static func parseFallbackSize(files: [TreeFileEntry]) -> FallbackSize? {
        let safetensorsSum = files
            .filter { $0.path.hasSuffix(".safetensors") }
            .reduce(Int64(0)) { $0 + $1.size }
        if safetensorsSum > 0 {
            return .safetensorsSum(safetensorsSum)
        }

        let ggufSizes: [Int64] = files
            .filter { entry in
                let lower = entry.path.lowercased()
                guard lower.hasSuffix(".gguf") else { return false }
                if entry.path.contains("/") { return false }                  // subdir / split shard
                if (entry.path as NSString).lastPathComponent
                    .lowercased().hasPrefix("mmproj") { return false }        // CLIP sidecar
                return entry.size >= 1_000_000
            }
            .map(\.size)
            .sorted()

        guard let lo = ggufSizes.first else { return nil }
        let hi = ggufSizes.last ?? lo
        return lo == hi ? .ggufSingle(lo) : .ggufRange(min: lo, max: hi)
    }
}
