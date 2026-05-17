import Foundation

@MainActor
class HFSearchService: ObservableObject {
    @Published var models: [HFModel] = []
    @Published var isLoading = false
    @Published var error: String?
    @Published var searchQuery = ""
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

        var components = URLComponents(string: "https://huggingface.co/api/models")!
        var items: [URLQueryItem] = [
            URLQueryItem(name: "filter", value: "mlx"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "\(pageSize)"),
            URLQueryItem(name: "skip", value: "\(currentSkip)"),
        ]
        let query = searchQuery.trimmingCharacters(in: .whitespaces)
        if !query.isEmpty {
            items.append(URLQueryItem(name: "search", value: query))
        }
        // Request extra fields via expand[]
        for field in ["safetensors", "lastModified", "likes", "downloads", "tags", "pipeline_tag"] {
            items.append(URLQueryItem(name: "expand[]", value: field))
        }
        components.queryItems = items

        guard let url = components.url else {
            error = "Invalid search URL"
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                error = "HuggingFace API error (HTTP \(code))"
                return
            }
            let decoded = try JSONDecoder().decode([HFModel].self, from: data)
            // Filter out MLX-format DeepSeek-V4 checkpoints — the Zig MLX
            // path rejects them; users should grab the GGUF + ds4 entry from
            // the built-in catalog instead. Show every other result the API
            // returns.
            let filtered = decoded.filter { !$0.id.lowercased().contains("deepseek-v4-flash") }
            fetchedModels.append(contentsOf: filtered)
            models = sortedModels
            currentSkip += decoded.count
            hasMore = decoded.count >= pageSize

            // Fetch file sizes for models missing safetensors metadata
            let missing = decoded.filter { $0.estimatedSizeBytes == 0 && $0.isCompatible }
            if !missing.isEmpty {
                Task { await fetchFallbackSizes(for: missing) }
            }
        } catch {
            self.error = error.localizedDescription
        }
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
