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
    @Published var format: ModelFormat = .both
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

        // Fetch safetensors sizes for MLX models missing metadata. GGUF repos
        // report size per-quant (resolved at download time), so skip them here.
        let missing = filtered.filter { $0.estimatedSizeBytes == 0 && $0.isCompatible && !$0.isGgufRepo }
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

    /// Fetch safetensors file sizes from the tree API for models missing metadata.
    private func fetchFallbackSizes(for models: [HFModel]) async {
        await withTaskGroup(of: (String, Int64)?.self) { group in
            for model in models {
                group.addTask {
                    guard let url = URL(string: "https://huggingface.co/api/models/\(model.id)/tree/main") else { return nil }
                    guard let (data, response) = try? await URLSession.shared.data(from: url),
                          let http = response as? HTTPURLResponse, http.statusCode == 200,
                          let files = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return nil }
                    let total = files
                        .filter { ($0["path"] as? String)?.hasSuffix(".safetensors") == true && ($0["type"] as? String) == "file" }
                        .compactMap { $0["size"] as? Int64 ?? ($0["size"] as? Int).map({ Int64($0) }) }
                        .reduce(Int64(0), +)
                    return total > 0 ? (model.id, total) : nil
                }
            }
            for await result in group {
                guard let (id, size) = result else { continue }
                if let idx = fetchedModels.firstIndex(where: { $0.id == id }) {
                    fetchedModels[idx].fallbackSizeBytes = size
                }
            }
        }
        self.models = sortedModels
    }
}
