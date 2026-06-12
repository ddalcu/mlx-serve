import Foundation
import PDFKit

// Mini in-memory RAG pipeline for "attach a folder, ask questions about it".
// Non-persistent by design: the index lives on the chat session for the app's
// lifetime and is rebuilt on re-attach. Three layers, smallest possible:
//   DocumentChunker   — pure text → chunk splitting + file-type filter
//   DocumentRetrieval — pure scoring (cosine over embeddings, lexical fallback)
//   DocumentIndex     — folder walk + extraction + embedding off the main
//                       thread, published progress for the UI chip, and the
//                       searchDocuments tool's entry point.

// MARK: - Embedding abstraction

/// Minimal text-embedding interface so the index is unit-testable with a
/// deterministic fake and degrades to lexical scoring when no embedder is
/// available. Batch-oriented and async so the GPU implementation can ride
/// one HTTP request per file instead of one per chunk; a nil row degrades
/// that chunk to lexical-only scoring.
protocol TextEmbedding: Sendable {
    /// Embed passages for indexing. Returns exactly one entry per input.
    func vectors(for texts: [String]) async -> [[Double]?]
    /// Embed a search query. Asymmetric retrieval models (BGE) prepend
    /// their query instruction here; the default treats queries as passages.
    func queryVector(for text: String) async -> [Double]?
}

extension TextEmbedding {
    func queryVector(for text: String) async -> [Double]? {
        (await vectors(for: [text])).first ?? nil
    }
}

/// GPU embedder backed by the local mlx-serve `/v1/embeddings` endpoint
/// (encoder-only BERT model, batched masked forward — measured ~1.4 ms per
/// 1200-char chunk on M4 Max). The ONLY embedder: when the server is down
/// or the encoder can't be provisioned, chunks carry nil vectors and
/// retrieval runs lexical-only. Stateless and thread-safe: one instance is
/// shared by every indexing worker. Transport errors degrade rows to nil
/// instead of failing the index build.
struct ServerEmbedding: TextEmbedding {
    /// Registry id of the encoder model (e.g. "bge-small-en-v1.5-8bit").
    let model: String
    /// (texts, model) → one vector per text. Injected so unit tests can
    /// fake the wire; production uses `APIClient.embeddings`.
    let transport: @Sendable ([String], String) async throws -> [[Double]]

    /// Mirrors the server's per-forward chunking (generate.EMBED_MAX_BATCH)
    /// so one mega-request doesn't monopolize the inference thread.
    static let maxBatch = 64

    func vectors(for texts: [String]) async -> [[Double]?] {
        var out: [[Double]?] = []
        out.reserveCapacity(texts.count)
        var start = 0
        while start < texts.count {
            let batch = Array(texts[start..<min(start + Self.maxBatch, texts.count)])
            if let rows = try? await transport(batch, model), rows.count == batch.count {
                out.append(contentsOf: rows.map { Optional($0) })
            } else {
                out.append(contentsOf: [[Double]?](repeating: nil, count: batch.count))
            }
            start += batch.count
        }
        return out
    }

    func queryVector(for text: String) async -> [Double]? {
        // BGE-family encoders are asymmetric: queries carry an instruction
        // prefix, passages don't. Other encoders (MiniLM …) are symmetric.
        let query = model.lowercased().contains("bge")
            ? "Represent this sentence for searching relevant passages: \(text)"
            : text
        return (await vectors(for: [query])).first ?? nil
    }

    /// Pick the encoder model from a `/v1/models` registry snapshot. Loaded
    /// entries win over unloaded stubs (no cold-load latency), but a stub is
    /// still used — the server hot-loads it on the first embed request.
    static func pickEmbeddingModel(_ models: [ModelInfo]) -> String? {
        let encoders = models.filter { $0.supportsEmbeddings }
        return (encoders.first { $0.loaded } ?? encoders.first)?.name
    }

    /// Probe-only factory: asks the local server for an embeddings-capable
    /// model and binds the transport to it. Returns nil when the server is
    /// down or has no encoder model. Never downloads — tests and diagnostics
    /// use this; the app uses `autoProvider`.
    static func probe(port: UInt16) -> @Sendable () async -> TextEmbedding? {
        return {
            let api = APIClient()
            guard let models = try? await api.fetchAllModels(port: port),
                  let id = pickEmbeddingModel(models) else { return nil }
            return ServerEmbedding(model: id, transport: { texts, model in
                try await api.embeddings(port: port, model: model, input: texts)
            })
        }
    }

    /// The encoder the app provisions automatically: 35 MB, 384-dim, strong
    /// retrieval quality for its size. Any embeddings-capable model already
    /// known to the server takes precedence over downloading this one.
    static let defaultEncoderRepo = "mlx-community/bge-small-en-v1.5-8bit"

    /// Seamless resolution, steps injected for unit tests:
    ///   1. Server unreachable → nil (no point downloading; lexical-only).
    ///   2. Server already knows an embeddings-capable model → use it.
    ///   3. Otherwise download the default encoder (one-time, resumable) and
    ///      register it with the server by absolute path — works regardless
    ///      of which directory the server's --model-dir scan covers.
    /// Any failure → nil → lexical-only retrieval. Never throws.
    static func resolve(
        fetchModels: @Sendable () async throws -> [ModelInfo],
        ensureEncoderOnDisk: @Sendable () async throws -> String,
        loadByPath: @Sendable (String) async throws -> String,
        transport: @escaping @Sendable ([String], String) async throws -> [[Double]]
    ) async -> TextEmbedding? {
        guard let models = try? await fetchModels() else { return nil }
        if let id = pickEmbeddingModel(models) {
            return ServerEmbedding(model: id, transport: transport)
        }
        guard let path = try? await ensureEncoderOnDisk(),
              let id = try? await loadByPath(path) else { return nil }
        return ServerEmbedding(model: id, transport: transport)
    }

    /// Production provider for `DocumentIndex`: full auto-provisioning
    /// against the local server (probe → download if absent → load by path).
    static func autoProvider(port: UInt16) -> @Sendable () async -> TextEmbedding? {
        return {
            let api = APIClient()
            return await resolve(
                fetchModels: { try await api.fetchAllModels(port: port) },
                ensureEncoderOnDisk: { try await EncoderDownloader.ensureOnDisk(repoId: defaultEncoderRepo) },
                loadByPath: { try await api.loadModel(port: port, id: $0).name },
                transport: { texts, model in
                    try await api.embeddings(port: port, model: model, input: texts)
                }
            )
        }
    }
}

/// One-time provisioning of the default embedding encoder. Rides the
/// existing `DownloadManager` (HF tree listing, resume, skip-existing,
/// disk-space checks); a shared in-flight task deduplicates concurrent
/// folder-attaches racing to download the same repo.
@MainActor
enum EncoderDownloader {
    private static var inflight: Task<String, Error>?

    static func ensureOnDisk(repoId: String) async throws -> String {
        if let task = inflight { return try await task.value }
        let task = Task<String, Error> {
            let dm = DownloadManager()
            if dm.isReady(repoId), let dir = dm.existingModelDir(for: repoId) { return dir }
            await dm.download(repoId: repoId)
            // download() reports failure via published state, not throws —
            // gate on actual on-disk readiness.
            guard dm.isReady(repoId), let dir = dm.existingModelDir(for: repoId) else {
                throw URLError(.cannotLoadFromNetwork)
            }
            return dir
        }
        inflight = task
        defer { inflight = nil }
        return try await task.value
    }
}

// MARK: - Chunking

enum DocumentChunker {

    /// Text-bearing formats worth indexing. Everything else in the folder is
    /// silently skipped (images, archives, binaries, extension-less files).
    static let indexableExtensions: Set<String> = [
        "txt", "text", "md", "markdown", "log",
        "json", "yaml", "yml", "toml", "csv", "tsv",
        "xml", "html", "htm", "rtf", "pdf",
    ]

    static func isIndexable(_ url: URL) -> Bool {
        guard !url.lastPathComponent.hasPrefix(".") else { return false }
        return indexableExtensions.contains(url.pathExtension.lowercased())
    }

    /// Split `text` into chunks of at most `maxChars`, preferring paragraph
    /// boundaries, with the tail of each chunk carried into the next so an
    /// answer spanning a boundary is still retrievable from one chunk.
    static func chunk(_ text: String, maxChars: Int = 1200, overlapChars: Int = 200) -> [String] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        guard trimmed.count > maxChars else { return [trimmed] }

        // Paragraph units. Structured files (YAML/JSON/logs) often have NO
        // blank lines — the whole file is one "paragraph" — so an oversized
        // paragraph is packed whole-LINE by whole-line, never cut mid-line.
        // Only a single monster line (minified JSON) falls back to fixed
        // character windows.
        var units: [String] = []
        for para in trimmed.components(separatedBy: "\n\n") {
            let p = para.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !p.isEmpty else { continue }
            if p.count <= maxChars {
                units.append(p)
                continue
            }
            var lineUnit = ""
            for line in p.components(separatedBy: "\n") {
                if line.count > maxChars {
                    if !lineUnit.isEmpty { units.append(lineUnit); lineUnit = "" }
                    units.append(contentsOf: hardSplit(line, maxChars: maxChars))
                } else if lineUnit.isEmpty {
                    lineUnit = line
                } else if lineUnit.count + 1 + line.count <= maxChars {
                    lineUnit += "\n" + line
                } else {
                    units.append(lineUnit)
                    lineUnit = line
                }
            }
            if !lineUnit.isEmpty { units.append(lineUnit) }
        }

        var chunks: [String] = []
        var current = ""
        for unit in units {
            if current.isEmpty {
                current = unit
            } else if current.count + 2 + unit.count <= maxChars {
                current += "\n\n" + unit
            } else {
                chunks.append(current)
                let tail = overlapTail(of: current, overlapChars)
                current = tail.isEmpty ? unit : tail + "\n\n" + unit
                if current.count > maxChars { current = unit } // overlap doesn't fit — drop it
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }

    /// Fixed-window fallback for a single line longer than `maxChars`.
    private static func hardSplit(_ line: String, maxChars: Int) -> [String] {
        let chars = Array(line)
        var pieces: [String] = []
        var start = 0
        while start < chars.count {
            let end = min(start + maxChars, chars.count)
            let piece = String(chars[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { pieces.append(piece) }
            start = end
        }
        return pieces
    }

    /// Last ~`n` chars of `s`, cut forward to a line boundary when one exists
    /// (structured files), else a whitespace boundary — the overlap never
    /// starts mid-line or mid-word.
    private static func overlapTail(of s: String, _ n: Int) -> String {
        guard n > 0 else { return "" }
        guard s.count > n else { return s }
        let tail = String(s.suffix(n))
        if let nl = tail.firstIndex(of: "\n") {
            return String(tail[tail.index(after: nl)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let ws = tail.firstIndex(of: " ") {
            return String(tail[tail.index(after: ws)...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return tail
    }
}

// MARK: - Retrieval (pure)

enum DocumentRetrieval {

    struct Chunk {
        let text: String
        let source: String      // path relative to the attached folder
        let vector: [Double]?   // nil when the embedding model is unavailable
        /// Stemmed token set, precomputed at index build (workers, parallel)
        /// so per-query lexical scoring is set lookups, not re-tokenization.
        let stems: Set<String>

        init(text: String, source: String, vector: [Double]?) {
            self.text = text
            self.source = source
            self.vector = vector
            self.stems = Set(DocumentRetrieval.tokens(text).map(DocumentRetrieval.stem))
        }
    }

    struct SearchResult {
        let text: String
        let source: String
        let score: Double
    }

    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        guard na > 0, nb > 0 else { return 0 }
        return dot / ((na * nb).squareRoot())
    }

    /// Exact-term scoring with uniform weights — kept for callers without a
    /// corpus to derive IDF from. `search` uses the IDF-weighted variant.
    static func lexicalScore(query: String, text: String) -> Double {
        let qStems = Set(tokens(query).map(stem))
        guard !qStems.isEmpty else { return 0 }
        let tStems = Set(tokens(text).map(stem))
        let hits = qStems.filter { tStems.contains($0) }.count
        return Double(hits) / Double(qStems.count)
    }

    static func tokens(_ s: String) -> [String] {
        s.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
    }

    /// Crude stem: token truncated to 6 chars — unifies inflections
    /// ("frustrated"/"frustration" → "frustr") without a real stemmer.
    static func stem(_ t: String) -> String { String(t.prefix(6)) }

    /// Hybrid weights: cosine carries meaning, lexical carries exact terms.
    /// Lexical earns 0.4 because it's IDF-weighted (clean signal): on key-value
    /// corpora (YAML/JSON) the rare literal term IS the answer, and sentence
    /// embeddings of metadata soup are barely better than noise.
    static let vectorWeight = 0.6
    static let lexicalWeight = 0.4
    /// Cap excerpts per source file so one document can't monopolize the
    /// top-K — "which customer…" questions need evidence across files.
    static let maxPerSource = 2

    static func search(chunks: [Chunk], query: String, queryVector: [Double]?, topK: Int) -> [SearchResult] {
        guard topK > 0, !chunks.isEmpty else { return [] }

        // IDF over the attached corpus, restricted to the query's stems:
        // a term in every chunk ("frustration:" is a key in every CXone file)
        // weighs ~nothing; the rare value token ("HIGH") dominates. Without
        // this, ubiquitous terms drowned the discriminating one and the
        // metadata chunks that answer the question never surfaced.
        let queryStems = Set(tokens(query).map(stem))
        let n = Double(chunks.count)
        var idf: [String: Double] = [:]
        for s in queryStems {
            let df = chunks.reduce(0) { $0 + ($1.stems.contains(s) ? 1 : 0) }
            if df > 0 { idf[s] = log((n + 1) / Double(df)) }
        }
        let totalWeight = idf.values.reduce(0, +)

        let scored = chunks.map { c -> SearchResult in
            let lexical: Double = totalWeight > 0
                ? idf.reduce(0) { $0 + (c.stems.contains($1.key) ? $1.value : 0) } / totalWeight
                : 0
            let score: Double
            if let qv = queryVector, let cv = c.vector, qv.count == cv.count {
                score = vectorWeight * cosine(qv, cv) + lexicalWeight * lexical
            } else {
                score = lexical
            }
            return SearchResult(text: c.text, source: c.source, score: score)
        }
        var results: [SearchResult] = []
        var perSource: [String: Int] = [:]
        for r in scored.sorted(by: { $0.score > $1.score }) {
            guard results.count < topK else { break }
            guard perSource[r.source, default: 0] < maxPerSource else { continue }
            perSource[r.source, default: 0] += 1
            results.append(r)
        }
        return results
    }

    /// Render results as the searchDocuments tool output the model reads.
    static func formatResults(_ results: [SearchResult], query: String) -> String {
        guard !results.isEmpty else {
            return "No excerpts matched '\(query)'. Try a different phrasing, or the attached documents may not cover this topic."
        }
        let body = results.enumerated().map { i, r in
            "[\(i + 1)] \(r.source) (relevance \(String(format: "%.2f", r.score)))\n\(r.text)"
        }.joined(separator: "\n\n---\n\n")
        return "Top \(results.count) excerpts for '\(query)':\n\n\(body)"
    }
}

// MARK: - Index

/// In-memory document index for one attached folder. Owned per chat session
/// (AppState.documentIndexes), never persisted. Indexing runs detached from
/// the main thread; `state` is published for the attachment chip.
@MainActor
final class DocumentIndex: ObservableObject, Identifiable {

    enum State: Equatable {
        /// Resolving the embedder — may include the one-time encoder-model
        /// download (~35 MB) on a machine that doesn't have it yet.
        case preparing
        case indexing(filesDone: Int, filesTotal: Int)
        case ready(files: Int, chunks: Int)
        case failed(String)
    }

    nonisolated let id = UUID()
    nonisolated let folderURL: URL
    nonisolated var folderName: String { folderURL.lastPathComponent }

    @Published private(set) var state: State = .indexing(filesDone: 0, filesTotal: 0)
    private(set) var chunks: [DocumentRetrieval.Chunk] = []

    private var indexTask: Task<Void, Never>?
    /// Resolves the embedder once per startIndexing. The production provider
    /// (`ServerEmbedding.autoProvider`) probes the local server, downloads
    /// the default encoder if needed, and registers it by path; nil →
    /// lexical-only retrieval.
    private let embedderProvider: @Sendable () async -> TextEmbedding?
    /// The embedder the index was BUILT with — queries must go through the
    /// same model or cosine compares vectors from different spaces (the
    /// dimension guard would silently drop to lexical-only).
    private var activeQueryEmbedder: TextEmbedding?

    /// Skip pathological inputs: huge single files and unbounded folders.
    nonisolated static let maxFileBytes = 8 * 1024 * 1024
    nonisolated static let maxFiles = 2000
    nonisolated static let defaultTopK = 5

    init(folderURL: URL,
         embedderProvider: @escaping @Sendable () async -> TextEmbedding?) {
        self.folderURL = folderURL
        self.embedderProvider = embedderProvider
    }

    /// Worker fan-out: overlaps file IO + extraction + chunking with the
    /// embedder's HTTP round-trips (the embedder itself is stateless and
    /// shared). Cap well below the server's appetite — embed requests
    /// serialize on its inference thread anyway.
    nonisolated static var indexingWorkers: Int {
        max(1, min(16, ProcessInfo.processInfo.activeProcessorCount - 2))
    }

    func startIndexing() {
        indexTask?.cancel()
        state = .preparing
        filesDone = 0
        let folder = folderURL
        let provider = embedderProvider
        indexTask = Task.detached(priority: .userInitiated) { [weak self] in
            let files = Self.collectFiles(in: folder)
            guard !files.isEmpty else {
                await self?.finish(.failed("No readable documents found (txt, md, pdf, json, yaml, csv …)"))
                return
            }
            // May download the encoder model on first ever use (~35 MB).
            let embedder = await provider()
            if Task.isCancelled { return }
            await self?.beginIndexing(total: files.count)
            let workers = Self.indexingWorkers
            // Worker w takes files w, w+W, w+2W … — interleaving balances
            // uneven file sizes without a work queue. Results keyed by file
            // position so the final chunk order is deterministic.
            let perFile = await withTaskGroup(of: [(Int, [DocumentRetrieval.Chunk])].self) { group in
                for w in 0..<workers {
                    group.addTask {
                        var out: [(Int, [DocumentRetrieval.Chunk])] = []
                        var i = w
                        while i < files.count, !Task.isCancelled {
                            let url = files[i]
                            var fileChunks: [DocumentRetrieval.Chunk] = []
                            if let text = Self.extractText(url) {
                                let source = Self.relativePath(of: url, in: folder)
                                let pieces = DocumentChunker.chunk(text)
                                let vectors = await embedder?.vectors(for: pieces)
                                    ?? [[Double]?](repeating: nil, count: pieces.count)
                                for (piece, vector) in zip(pieces, vectors) {
                                    fileChunks.append(.init(text: piece, source: source, vector: vector))
                                }
                            }
                            out.append((i, fileChunks))
                            await self?.bumpProgress(total: files.count)
                            i += workers
                        }
                        return out
                    }
                }
                var all = Array(repeating: [DocumentRetrieval.Chunk](), count: files.count)
                for await part in group {
                    for (i, fileChunks) in part { all[i] = fileChunks }
                }
                return all
            }
            if Task.isCancelled { return }
            let chunks = perFile.flatMap { $0 }
            await self?.finish(.ready(files: files.count, chunks: chunks.count),
                               chunks: chunks, queryEmbedder: embedder)
        }
    }

    func cancel() {
        indexTask?.cancel()
        indexTask = nil
    }

    /// The searchDocuments tool body: embed the query, rank chunks, format.
    /// Lexical fallback kicks in automatically when embeddings are missing.
    /// The query rides the same embedder that built the index (GPU or CPU).
    func search(query: String, topK: Int = DocumentIndex.defaultTopK) async -> String {
        switch state {
        case .preparing:
            return "The document index is still being prepared (resolving the embedding model). Wait a moment and search again."
        case .indexing(let done, let total):
            return "The document index is still being built (\(done)/\(total) files). Wait a moment and search again."
        case .failed(let msg):
            return "Error: document indexing failed — \(msg)"
        case .ready:
            let qv = await activeQueryEmbedder?.queryVector(for: query)
            let results = DocumentRetrieval.search(chunks: chunks, query: query,
                                                   queryVector: qv, topK: topK)
            return DocumentRetrieval.formatResults(results, query: query)
        }
    }

    /// Test hook: install pre-built chunks and mark the index ready without
    /// touching the filesystem.
    func finishForTesting(chunks: [DocumentRetrieval.Chunk], files: Int) {
        self.chunks = chunks
        self.state = .ready(files: files, chunks: chunks.count)
    }

    // MARK: - Private

    /// Completed-file counter for the progress chip; bumped by every worker
    /// (serialized through the main actor).
    private var filesDone = 0

    private func beginIndexing(total: Int) {
        filesDone = 0
        state = .indexing(filesDone: 0, filesTotal: total)
    }

    private func bumpProgress(total: Int) {
        filesDone += 1
        if case .indexing = state {
            state = .indexing(filesDone: filesDone, filesTotal: total)
        }
    }

    private func finish(_ newState: State, chunks: [DocumentRetrieval.Chunk] = [],
                        queryEmbedder: TextEmbedding? = nil) {
        self.chunks = chunks
        self.activeQueryEmbedder = queryEmbedder
        self.state = newState
    }

    nonisolated private static func collectFiles(in folder: URL) -> [URL] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: folder,
                                             includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                                             options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            return []
        }
        var files: [URL] = []
        for case let url as URL in enumerator {
            guard files.count < maxFiles else { break }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true,
                  DocumentChunker.isIndexable(url) else { continue }
            if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize,
               size > maxFileBytes { continue }
            files.append(url)
        }
        return files.sorted { $0.path < $1.path }
    }

    nonisolated private static func relativePath(of url: URL, in folder: URL) -> String {
        let folderPath = folder.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        return path.hasPrefix(folderPath) ? String(path.dropFirst(folderPath.count)) : url.lastPathComponent
    }

    /// PDF via PDFKit (text layer only — scanned PDFs without OCR come back
    /// nil and are skipped); everything else read as UTF-8 with a Latin-1
    /// fallback for legacy exports.
    nonisolated static func extractText(_ url: URL) -> String? {
        if url.pathExtension.lowercased() == "pdf" {
            guard let doc = PDFDocument(url: url), let text = doc.string,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return text
        }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1)
    }
}
