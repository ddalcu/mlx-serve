import Foundation
import AppKit

@MainActor
class DownloadManager: ObservableObject {
    @Published var downloads: [String: DownloadState] = [:]

    /// In-flight `download`/`downloadGguf` tasks keyed by repoId, so the
    /// Cancel button can interrupt them. Removed in the wrapper's `defer`.
    private var activeTasks: [String: Task<Void, Never>] = [:]

    struct DownloadState {
        var progress: Double = 0
        var status: Status = .idle
        var statusText: String = ""
        var error: String?
        var currentFile: String = ""
        var fileIndex: Int = 0
        var fileCount: Int = 0
        var bytesPerSecond: Double = 0
        var fileProgress: Double = 0

        enum Status: Equatable {
            case idle, downloading, completed, failed
        }

        var speedFormatted: String {
            if bytesPerSecond > 1_000_000 {
                return String(format: "%.1f MB/s", bytesPerSecond / 1_000_000)
            } else if bytesPerSecond > 1_000 {
                return String(format: "%.0f KB/s", bytesPerSecond / 1_000)
            }
            return ""
        }

        var percentFormatted: String {
            String(format: "%.0f%%", fileProgress * 100)
        }
    }

    let modelsDir: String = {
        let path = NSString(string: "~/.mlx-serve/models").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }()

    // MARK: - Path resolution
    //
    // New downloads land under `<modelsDir>/<author>/<name>/` (same shape as
    // LM Studio). Pre-existing flat dirs (`<modelsDir>/<name>/`) keep working
    // through the discoverer's fallback scan and `existingModelDir(for:)` —
    // no automatic migration; users can move dirs manually if they want.

    /// True iff a filename is a GGUF mlx-serve can serve. As of the embedded
    /// llama.cpp engine that's ANY `.gguf` EXCEPT mmproj sidecars (CLIP
    /// vision / audio encoders shipped alongside vision-enabled LLMs —
    /// `mmproj-*.gguf` files have `general.architecture=clip` and llama.cpp
    /// refuses to load them as language models). DeepSeek-V4-Flash routes
    /// to the ds4 engine, everything else to llama.cpp (server-side, by
    /// `ggufModelType`).
    nonisolated static func isSupportedGguf(_ filename: String) -> Bool {
        let lower = filename.lowercased()
        guard lower.hasSuffix(".gguf") else { return false }
        return !isMmprojGguf(filename)
    }

    /// True iff a basename is a multimodal-projection sidecar — the
    /// `mmproj-*.gguf` convention used by llama.cpp tooling, ollama, and
    /// LM Studio for the side-loaded CLIP vision / audio encoders. Mirrors
    /// the Zig `model_discovery.isMmprojGgufBasename` so client and server
    /// agree on which artifacts are LLMs.
    nonisolated static func isMmprojGguf(_ filename: String) -> Bool {
        let lower = filename.lowercased()
        return lower.hasSuffix(".gguf") && lower.hasPrefix("mmproj")
    }

    /// Classify a GGUF filename into the `modelType` the server reports / routes
    /// on: `deepseek_v4` for DeepSeek-V4-Flash (ds4 engine), `gguf` for any other
    /// `.gguf` (llama.cpp engine), or nil when it isn't a GGUF. Mirrors the Zig
    /// `model_discovery.isDs4GgufBasename` split so client and server agree.
    nonisolated static func ggufModelType(forBasename filename: String) -> String? {
        guard filename.lowercased().hasSuffix(".gguf") else { return nil }
        return filename.lowercased().hasPrefix("deepseek-v4-flash") ? "deepseek_v4" : "gguf"
    }

    /// Short, human-friendly label for a GGUF file in the quant picker: surfaces a
    /// quant token like `Q4_K_M` / `IQ2_XXS` / `F16` when present, else the
    /// extension-stripped basename. Pure + testable.
    nonisolated static func quantLabel(forFilename filename: String) -> String {
        let base = (filename as NSString).lastPathComponent
        if let r = base.range(of: "(IQ|Q|BF|F)[0-9][A-Za-z0-9_]*", options: [.regularExpression, .caseInsensitive]) {
            return String(base[r])
        }
        return (base as NSString).deletingPathExtension
    }

    /// Where a fresh download of `repoId` should be written. New 2-level layout.
    /// `repoId` should be `author/name`; bare names land at the legacy top level.
    nonisolated static func newLayoutDir(rootDir: String, repoId: String) -> String {
        let parts = repoId.split(separator: "/").map(String.init)
        guard parts.count >= 2 else {
            return (rootDir as NSString).appendingPathComponent(parts.last ?? repoId)
        }
        let author = parts[parts.count - 2]
        let name = parts[parts.count - 1]
        return ((rootDir as NSString).appendingPathComponent(author) as NSString)
            .appendingPathComponent(name)
    }

    /// Filter a HuggingFace `/tree/main?recursive=true` listing down to the
    /// files a model download actually needs: top-level config / tokenizer /
    /// weight files, PLUS the `mtp/` multi-token-prediction sidecar — the only
    /// nested artifact the server auto-loads (`mtp/weights.safetensors`).
    /// Without it, an MTP model silently loses its speculative-decoding speedup
    /// because a non-recursive listing returns `mtp` as a bare directory entry
    /// that the `type == "file"` filter drops. Other subdirectories (e.g.
    /// `original/` or alternate-precision shadow copies) are skipped so we don't
    /// pull tens of GB of unused weights. Returns (path, size) pairs.
    nonisolated static func selectNeededFiles(from entries: [[String: Any]], selection: FileSelection = .chatDefault) -> [(String, Int64)] {
        let neededExtensions: Set<String> = ["json", "safetensors", "jinja", "model", "txt"]
        return entries.compactMap { file -> (String, Int64)? in
            guard let path = file["path"] as? String,
                  let ftype = file["type"] as? String, ftype == "file" else { return nil }
            // Depth gate. Chat default: top-level files + the `mtp/` sidecar.
            // Media (recursive): keep nested weight subdirs (FLUX's
            // transformer/vae/text_encoder, TTS's speech_tokenizer).
            if !selection.recursive {
                guard !path.contains("/") || path.hasPrefix("mtp/") else { return nil }
            }
            let ext = (path as NSString).pathExtension.lowercased()
            guard neededExtensions.contains(ext) || (path as NSString).lastPathComponent == "chat_template.jinja" else { return nil }
            // Per-bundle junk filter.
            if selection.excludeSubstrings.contains(where: { path.contains($0) }) { return nil }
            // Safetensors allowlist (LTX): keep only the engine's 3 files, skip
            // the LoRAs / upscalers / alternate transformers (~50 GB unused).
            if ext == "safetensors", let keep = selection.keepSafetensors,
               !keep.contains((path as NSString).lastPathComponent) { return nil }
            let size = file["size"] as? Int64 ?? (file["size"] as? Int).map { Int64($0) } ?? 0
            return (path, size)
        }
    }

    /// Path of an existing model on disk. Prefers the new 2-level layout; falls
    /// back to the legacy flat layout. Returns nil when neither has a `config.json`.
    nonisolated static func existingModelDir(rootDir: String, repoId: String) -> String? {
        let fm = FileManager.default
        let new = newLayoutDir(rootDir: rootDir, repoId: repoId)
        if fm.fileExists(atPath: (new as NSString).appendingPathComponent("config.json")) {
            return new
        }
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let legacy = (rootDir as NSString).appendingPathComponent(name)
        if fm.fileExists(atPath: (legacy as NSString).appendingPathComponent("config.json")) {
            return legacy
        }
        return nil
    }

    func newLayoutDir(for repoId: String) -> String {
        Self.newLayoutDir(rootDir: modelsDir, repoId: repoId)
    }

    func existingModelDir(for repoId: String) -> String? {
        Self.existingModelDir(rootDir: modelsDir, repoId: repoId)
    }

    /// Shared NSFW content-filter classifier (Apache-2.0). The server applies it
    /// to ALL image generation (Krea license §4.2); auto-downloaded once into
    /// `~/.mlx-serve/models` and shared across every image model. Original
    /// public repo — no conversion/hosting; the Zig engine reads it directly.
    static let nsfwClassifierRepo = "Falconsai/nsfw_image_detection"

    func nsfwClassifierReady() -> Bool {
        guard let dir = existingModelDir(for: Self.nsfwClassifierRepo) else { return false }
        return FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("model.safetensors"))
    }

    /// Best-effort: provision the NSFW classifier in the background if missing.
    /// Idempotent + quiet (tracked under its own repoId, so it doesn't disturb a
    /// model's bundle progress); the server fails OPEN until it's present. Safe to
    /// call on every Image-tab appearance.
    func ensureNsfwClassifier() {
        if nsfwClassifierReady() { return }
        if activeTasks[Self.nsfwClassifierRepo] != nil { return } // already downloading
        start(repoId: Self.nsfwClassifierRepo) {}
    }

    /// Repos the app auto-provisions for its own internal use (a "vit"
    /// architecture the model picker already flags red as "Unsupported",
    /// since it isn't a chat model) — never something the user chose to
    /// download, so `discoverLocalModels` drops them before anything renders.
    /// Matched by `LocalModel.name`, which for the standard nested layout
    /// (`<root>/<org>/<repo>`) is exactly the `org/repo` string these repoIds
    /// already are.
    nonisolated static let internalHelperRepos: Set<String> = [nsfwClassifierRepo]

    /// User-configurable extra discovery root. Persisted in UserDefaults under
    /// `customModelPath` so it survives app restarts. The raw stored value is
    /// kept verbatim (we don't erase a broken path) so the user can see and
    /// fix it in Settings; discovery, however, only uses it when it resolves
    /// to an existing directory.
    private static let customRootDefaultsKey = "customModelPath"

    @Published var customRoot: String? = {
        let raw = UserDefaults.standard.string(forKey: DownloadManager.customRootDefaultsKey) ?? ""
        return raw.isEmpty ? nil : raw
    }() {
        didSet {
            let trimmed = customRoot?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let t = trimmed, !t.isEmpty {
                UserDefaults.standard.set(t, forKey: Self.customRootDefaultsKey)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.customRootDefaultsKey)
            }
        }
    }

    /// Canonicalize a directory path for de-duplication against the default
    /// roots. Returns nil when the path is empty or doesn't resolve to an
    /// existing directory.
    private func resolvedCustomRoot() -> String? {
        guard let raw = customRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else { return nil }
        let expanded = (raw as NSString).expandingTildeInPath
        let standardized = URL(fileURLWithPath: expanded).standardizedFileURL.path
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: standardized, isDirectory: &isDir), isDir.boolValue else { return nil }
        // Skip if it's the same folder we already scan as one of the defaults.
        let standardizedMlx = URL(fileURLWithPath: modelsDir).standardizedFileURL.path
        if standardized == standardizedMlx { return nil }
        if let lm = lmStudioRoot,
           URL(fileURLWithPath: lm).standardizedFileURL.path == standardized {
            return nil
        }
        return standardized
    }

    /// LM Studio's downloads root, resolved once at app launch.
    /// Reads `~/.lmstudio/settings.json`'s `downloadsFolder` field; falls back to
    /// `~/.lmstudio/models`. nil if LM Studio isn't installed or the folder is unreachable.
    let lmStudioRoot: String? = {
        let settingsPath = NSString(string: "~/.lmstudio/settings.json").expandingTildeInPath
        let configured: String? = {
            guard let data = FileManager.default.contents(atPath: settingsPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let folder = json["downloadsFolder"] as? String,
                  !folder.isEmpty else { return nil }
            return (folder as NSString).expandingTildeInPath
        }()
        let fallback = NSString(string: "~/.lmstudio/models").expandingTildeInPath
        let candidate = configured ?? fallback
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidate, isDirectory: &isDir), isDir.boolValue else { return nil }
        return candidate
    }()

    /// Check if a model has all required files for loading.
    /// Verifies: config.json, tokenizer files, chat template, and ALL safetensors shards.
    /// For GGUF-backed models (ds4 engine) the check is just "directory contains
    /// at least one non-trivial .gguf" — they ship a single artifact, not the
    /// MLX safetensors tree.
    func isReady(_ repoId: String) -> Bool {
        guard let modelDir = existingModelDir(for: repoId) else { return false }
        let fm = FileManager.default

        // GGUF fast-path. Check BEFORE the safetensors gate so a dir that
        // legitimately has no config.json still resolves as ready. Any non-trivial
        // .gguf counts now (ds4 for DSV4-Flash, llama.cpp for the rest).
        if let entries = try? fm.contentsOfDirectory(atPath: modelDir) {
            for entry in entries where Self.isSupportedGguf(entry) {
                let fullPath = (modelDir as NSString).appendingPathComponent(entry)
                let size = (try? fm.attributesOfItem(atPath: fullPath)[.size] as? UInt64) ?? 0
                if size >= 1_000_000 { return true }
            }
        }

        // Must have config.json
        guard fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("config.json")) else { return false }

        // Must have tokenizer (tokenizer.json or tokenizer.model)
        let hasTokenizer = fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("tokenizer.json"))
            || fm.fileExists(atPath: (modelDir as NSString).appendingPathComponent("tokenizer.model"))
        guard hasTokenizer else { return false }

        guard let entries = try? fm.contentsOfDirectory(atPath: modelDir) else { return false }
        let safetensors = entries.filter { $0.hasSuffix(".safetensors") }

        // Must have at least one safetensors file
        guard !safetensors.isEmpty else { return false }

        // If sharded (model.safetensors.index.json exists), check all shards are present
        let indexPath = (modelDir as NSString).appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexPath) {
            if let data = fm.contents(atPath: indexPath),
               let index = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = index["weight_map"] as? [String: String] {
                let requiredShards = Set(weightMap.values)
                for shard in requiredShards {
                    let shardPath = (modelDir as NSString).appendingPathComponent(shard)
                    guard fm.fileExists(atPath: shardPath) else { return false }
                    // Check it's not a zero-byte stub
                    let size = (try? fm.attributesOfItem(atPath: shardPath)[.size] as? UInt64) ?? 0
                    guard size > 0 else { return false }
                }
            }
        } else {
            // Single-file model — check the safetensors file is non-trivial
            guard let first = safetensors.first else { return false }
            let fullPath = (modelDir as NSString).appendingPathComponent(first)
            let size = (try? fm.attributesOfItem(atPath: fullPath)[.size] as? UInt64) ?? 0
            guard size > 1_000_000 else { return false }
        }

        return true
    }

    func modelPath(for repoId: String) -> String {
        existingModelDir(for: repoId) ?? newLayoutDir(for: repoId)
    }

    func download(repoId: String, selection: FileSelection = .chatDefault) async {
        let destDir = newLayoutDir(for: repoId)

        downloads[repoId] = DownloadState(status: .downloading, statusText: "Fetching file list...")

        do {
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)

            // `?recursive=true` so the listing includes nested sidecars — most
            // importantly the `mtp/` multi-token-prediction head. Without it HF
            // returns `mtp` as a bare directory entry and the file filter skips
            // it, silently dropping the sidecar (and the model's spec-decode
            // speedup). `selectNeededFiles` keeps top-level files + mtp/ only.
            let listURL = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true")!
            let (listData, _) = try await URLSession.shared.data(from: listURL)
            guard let files = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
                throw URLError(.cannotParseResponse)
            }

            let neededFiles = Self.selectNeededFiles(from: files, selection: selection)

            let totalSize = neededFiles.reduce(Int64(0)) { $0 + $1.1 }
            var downloadedSize: Int64 = 0

            // Pre-check disk space
            let destURL = URL(fileURLWithPath: destDir)
            if let values = try? destURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let available = values.volumeAvailableCapacityForImportantUsage,
               available < totalSize {
                throw NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError, userInfo: [
                    NSLocalizedDescriptionKey: "Not enough disk space. Need \(formatBytes(totalSize)) but only \(formatBytes(Int64(available))) available."
                ])
            }

            for (idx, (filePath, fileSize)) in neededFiles.enumerated() {
                let destPath = (destDir as NSString).appendingPathComponent(filePath)
                let partialPath = destPath + ".partial"

                // Create subdirectories if needed
                let parentDir = (destPath as NSString).deletingLastPathComponent
                try? FileManager.default.createDirectory(atPath: parentDir, withIntermediateDirectories: true)

                // Skip if already exists with right size
                if let attrs = try? FileManager.default.attributesOfItem(atPath: destPath),
                   let existingSize = attrs[.size] as? Int64,
                   existingSize == fileSize && fileSize > 0 {
                    downloadedSize += fileSize
                    downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
                    downloads[repoId]?.statusText = "Skipped \(filePath) (exists)"
                    downloads[repoId]?.fileIndex = idx + 1
                    downloads[repoId]?.fileCount = neededFiles.count
                    continue
                }

                let sizeStr = formatBytes(fileSize)
                downloads[repoId]?.currentFile = (filePath as NSString).lastPathComponent
                downloads[repoId]?.fileIndex = idx + 1
                downloads[repoId]?.fileCount = neededFiles.count
                downloads[repoId]?.fileProgress = 0
                downloads[repoId]?.bytesPerSecond = 0
                downloads[repoId]?.statusText = "\(filePath) (\(sizeStr))"

                let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filePath)")!
                let maxRetries = 20

                for attempt in 0..<maxRetries {
                    try Task.checkCancellation()

                    // Check for existing partial download
                    let existingBytes: Int64
                    if let attrs = try? FileManager.default.attributesOfItem(atPath: partialPath),
                       let size = attrs[.size] as? Int64, size > 0 {
                        existingBytes = size
                        downloads[repoId]?.statusText = "Resuming \(filePath) from \(formatBytes(existingBytes))..."
                        downloads[repoId]?.fileProgress = fileSize > 0 ? Double(existingBytes) / Double(fileSize) : 0
                    } else {
                        existingBytes = 0
                    }

                    // Create or open partial file
                    let fm = FileManager.default
                    if !fm.fileExists(atPath: partialPath) {
                        fm.createFile(atPath: partialPath, contents: nil)
                    }
                    guard let fileHandle = FileHandle(forWritingAtPath: partialPath) else {
                        throw URLError(.cannotCreateFile)
                    }
                    try fileHandle.seekToEnd()

                    var request = URLRequest(url: fileURL)
                    if existingBytes > 0 {
                        request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                    }

                    do {
                        try await downloadToFile(
                            request: request,
                            fileHandle: fileHandle,
                            repoId: repoId,
                            fileSize: fileSize,
                            existingBytes: existingBytes,
                            baseDownloaded: downloadedSize,
                            totalSize: totalSize
                        )

                        // Success — move partial to final destination
                        try? fm.removeItem(atPath: destPath)
                        try fm.moveItem(atPath: partialPath, toPath: destPath)
                        break
                    } catch {
                        // User-cancelled? Stop immediately. URLSession surfaces
                        // cancellation as NSURLErrorCancelled, not CancellationError,
                        // so route both here instead of into the retry path. Partial
                        // file stays on disk for a future resume.
                        if Self.isCancellation(error) { throw CancellationError() }
                        // Partial file stays on disk — next attempt resumes from it
                        if attempt < maxRetries - 1 {
                            let isStall = error is DownloadStallError
                            let delay = isStall ? 2.0 : Double(attempt + 1) * 2.0
                            let reason = isStall ? "Download stalled" : "Connection lost"
                            downloads[repoId]?.statusText = "\(reason), retrying in \(Int(delay))s... (\(attempt + 2)/\(maxRetries))"
                            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        } else {
                            throw error
                        }
                    }
                }

                downloadedSize += fileSize
                downloads[repoId]?.progress = totalSize > 0 ? Double(downloadedSize) / Double(totalSize) : 0
                downloads[repoId]?.fileProgress = 1.0
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete",
                                               fileIndex: neededFiles.count, fileCount: neededFiles.count)
        } catch {
            // User-cancelled? Skip the .failed row + alert — `start()`'s
            // wrapper will drop the entry and remove partials.
            if Task.isCancelled { return }
            let message = error.localizedDescription
            downloads[repoId] = DownloadState(status: .failed, error: message)
            if !(error is CancellationError) {
                presentFailureAlert(repoId: repoId, message: message)
            }
        }
    }

    /// List the top-level `.gguf` files in a HuggingFace repo (one per quant),
    /// sorted by name. Skips files in subdirectories (split-shard layouts) since
    /// the single-file download path can't reassemble them. Empty on error.
    func listGgufFiles(repoId: String) async -> [String] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main") else { return [] }
        guard let (data, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse, http.statusCode == 200,
              let files = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return files
            .compactMap { $0["path"] as? String }
            .filter { $0.lowercased().hasSuffix(".gguf") && !$0.contains("/") }
            .sorted()
    }

    /// Kick off `download(repoId:)` as a tracked, cancellable task. `onFinish`
    /// runs after the inner work returns (whether completion, failure, or
    /// cancellation) so the caller can refresh model lists exactly once.
    func start(repoId: String, onFinish: @escaping @MainActor () -> Void) {
        activeTasks[repoId]?.cancel()
        let task = Task { @MainActor [weak self] in
            await self?.download(repoId: repoId)
            self?.finalizeIfCancelled(repoId: repoId)
            self?.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    // MARK: - Media bundles
    //
    // A media model + its dependencies, downloaded as a unit (LTX → LTX +
    // Gemma-3-12B; FLUX/TTS → just the primary). Each component pulls ONLY the
    // files the engine reads (`FileSelection`). Tracked under the bundle id so
    // the gen pane can show aggregate progress / cancel.

    /// Download a bundle's components sequentially (skipping any already on
    /// disk). `onFinish` runs once after the last component settles. Stops the
    /// bundle if a component fails.
    func startBundle(_ bundle: MediaBundle, onFinish: @escaping @MainActor () -> Void) {
        activeTasks[bundle.id]?.cancel()
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            for comp in bundle.components {
                if Task.isCancelled { break }
                if self.componentReady(comp) { continue }
                await self.download(repoId: comp.repo, selection: comp.selection)
                if Task.isCancelled { break }
                if self.downloads[comp.repo]?.status == .failed { break }
            }
            self.activeTasks.removeValue(forKey: bundle.id)
            onFinish()
        }
        activeTasks[bundle.id] = task
    }

    /// Cancel a bundle download and every component's in-flight transfer.
    func cancelBundle(_ bundle: MediaBundle) {
        activeTasks[bundle.id]?.cancel()
        activeTasks.removeValue(forKey: bundle.id)
        for comp in bundle.components { cancel(comp.repo) }
    }

    /// True when every component of the bundle is present + complete on disk.
    func bundleReady(_ bundle: MediaBundle) -> Bool {
        bundle.components.allSatisfy { componentReady($0) }
    }

    func componentReady(_ comp: MediaComponent) -> Bool {
        Self.componentReady(comp, modelsRoot: modelsDir)
    }

    /// A component is ready when its model dir resolves, ALL `readyMarkers`
    /// exist (file or dir), AND at least one `.safetensors` is present — so a
    /// config-only partial download isn't mistaken for ready. `nonisolated` +
    /// static so it's unit-testable against a temp dir.
    nonisolated static func componentReady(_ comp: MediaComponent, modelsRoot: String) -> Bool {
        guard let dir = existingModelDir(rootDir: modelsRoot, repoId: comp.repo) else { return false }
        let fm = FileManager.default
        for marker in comp.readyMarkers {
            guard fm.fileExists(atPath: (dir as NSString).appendingPathComponent(marker)) else { return false }
        }
        return hasSafetensorsRecursive(dir)
    }

    nonisolated static func hasSafetensorsRecursive(_ dir: String) -> Bool {
        guard let en = FileManager.default.enumerator(atPath: dir) else { return false }
        while let f = en.nextObject() as? String {
            if (f as NSString).lastPathComponent.hasSuffix(".safetensors") { return true }
        }
        return false
    }

    /// Aggregate UI state for an in-flight (or failed) bundle download: the
    /// component currently transferring + its 1-based position. nil when the
    /// bundle is idle or fully ready.
    func activeBundleComponent(_ bundle: MediaBundle) -> (repo: String, index: Int, count: Int, state: DownloadState)? {
        for (i, comp) in bundle.components.enumerated() {
            if let st = downloads[comp.repo], st.status == .downloading || st.status == .failed {
                return (comp.repo, i + 1, bundle.components.count, st)
            }
        }
        return nil
    }

    func isBundleDownloading(_ bundle: MediaBundle) -> Bool {
        activeTasks[bundle.id] != nil
    }

    /// GGUF analogue of `start(repoId:onFinish:)`.
    func startGguf(repoId: String, ggufFilename: String, onFinish: @escaping @MainActor () -> Void) {
        activeTasks[repoId]?.cancel()
        let task = Task { @MainActor [weak self] in
            await self?.downloadGguf(repoId: repoId, ggufFilename: ggufFilename)
            self?.finalizeIfCancelled(repoId: repoId)
            self?.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    /// Cancel an in-flight download. The state row disappears from the UI and
    /// the entire download directory is removed — completed shards, config, and
    /// `.partial` files alike — so a cancel leaves ZERO footprint (no remnant
    /// that masquerades as a complete model, no undeletable config-only orphan).
    /// No-op if nothing is in flight for `repoId`. The actual wipe for a live
    /// task happens in `finalizeIfCancelled` once the task has stopped writing;
    /// the branch here covers the no-live-task case (already finished, or cancel
    /// fired before start).
    func cancel(_ repoId: String) {
        activeTasks[repoId]?.cancel()
        if activeTasks[repoId] == nil {
            downloads.removeValue(forKey: repoId)
            wipeDownloadDir(repoId)
        }
    }

    /// Post-await cleanup for the start() wrappers. When the task was cancelled
    /// mid-flight, drop the (possibly `.failed`) row and wipe the whole download
    /// dir. Runs after `download()` has fully returned, so the file handle is
    /// closed and it's safe to delete. On normal completion this is a no-op.
    private func finalizeIfCancelled(repoId: String) {
        guard Task.isCancelled else { return }
        downloads.removeValue(forKey: repoId)
        wipeDownloadDir(repoId)
    }

    /// Remove the entire download directory for `repoId`. Used only on
    /// user-cancel — distinct from the network-error resume path, which keeps
    /// `.partial` files on disk so the "Resume" button can pick up where it
    /// left off.
    private func wipeDownloadDir(_ repoId: String) {
        Self.removeModelFiles(at: newLayoutDir(for: repoId), roots: [modelsDir])
    }

    /// True if `error` represents a user/task cancellation rather than a
    /// transient failure. URLSession surfaces `session.invalidateAndCancel()`
    /// as `NSURLErrorCancelled` (NOT Swift's `CancellationError`), so the
    /// download retry loop must recognize both — otherwise a cancelled
    /// transfer falls into the generic `catch`, flashes "Connection lost,
    /// retrying…", and only unwinds when the next `Task.sleep` throws.
    nonisolated static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let ns = error as NSError
        return ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled
    }

    /// Delete a model's on-disk files given its resolved `path` (a model
    /// directory, or a single `.gguf` file living inside one). Removes the
    /// containing model directory and, when it sits in the 2-level
    /// `<author>/<name>` layout, prunes the now-empty author dir. Never deletes
    /// or climbs past a directory in `roots` (the scan roots), so emptying the
    /// last model under `~/.mlx-serve/models` can't wipe the root itself.
    /// Returns true if the model dir was removed. `nonisolated`/static so it's
    /// unit-testable without the real models root.
    @discardableResult
    nonisolated static func removeModelFiles(at path: String, roots: [String]) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDir) else { return false }
        let modelDir = isDir.boolValue ? path : (path as NSString).deletingLastPathComponent
        let normRoots = Set(roots.map { URL(fileURLWithPath: $0).standardizedFileURL.path })
        // Never remove a root directory itself.
        if normRoots.contains(URL(fileURLWithPath: modelDir).standardizedFileURL.path) { return false }
        try? fm.removeItem(atPath: modelDir)
        // Prune the parent (author) dir if it's now empty — unless it's a root.
        let authorDir = (modelDir as NSString).deletingLastPathComponent
        let authorNorm = URL(fileURLWithPath: authorDir).standardizedFileURL.path
        if !normRoots.contains(authorNorm),
           let kids = try? fm.contentsOfDirectory(atPath: authorDir),
           kids.filter({ !$0.hasPrefix(".") }).isEmpty {
            try? fm.removeItem(atPath: authorDir)
        }
        return !fm.fileExists(atPath: modelDir)
    }

    /// Download a single GGUF artifact from a HuggingFace repo. Used by the
    /// ds4-backed entries (e.g. DeepSeek-V4-Flash) which ship one big file
    /// instead of the MLX safetensors tree, and by the Model Browser's GGUF quant
    /// picker. Mirrors `download(repoId:)`'s resume/retry/disk-space shape, scoped
    /// to one file.
    func downloadGguf(repoId: String, ggufFilename: String) async {
        let destDir = newLayoutDir(for: repoId)
        downloads[repoId] = DownloadState(status: .downloading, statusText: "Fetching \(ggufFilename)...")

        do {
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)

            // HEAD to determine the GGUF's size up front for progress + disk-space check.
            let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(ggufFilename)")!
            var headReq = URLRequest(url: fileURL)
            headReq.httpMethod = "HEAD"
            let (_, headResp) = try await URLSession.shared.data(for: headReq)
            let fileSize: Int64 = {
                guard let http = headResp as? HTTPURLResponse else { return 0 }
                if let cl = http.value(forHTTPHeaderField: "Content-Length"), let n = Int64(cl) { return n }
                return http.expectedContentLength
            }()

            let destURL = URL(fileURLWithPath: destDir)
            if let values = try? destURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let available = values.volumeAvailableCapacityForImportantUsage,
               fileSize > 0, available < fileSize {
                throw NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError, userInfo: [
                    NSLocalizedDescriptionKey: "Not enough disk space. Need \(formatBytes(fileSize)) but only \(formatBytes(Int64(available))) available."
                ])
            }

            let destPath = (destDir as NSString).appendingPathComponent(ggufFilename)
            let partialPath = destPath + ".partial"

            // Skip if already exists at the expected size.
            if fileSize > 0,
               let attrs = try? FileManager.default.attributesOfItem(atPath: destPath),
               let existingSize = attrs[.size] as? Int64,
               existingSize == fileSize {
                downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete", fileIndex: 1, fileCount: 1)
                return
            }

            downloads[repoId]?.currentFile = ggufFilename
            downloads[repoId]?.fileIndex = 1
            downloads[repoId]?.fileCount = 1

            let maxRetries = 20
            for attempt in 0..<maxRetries {
                try Task.checkCancellation()

                let existingBytes: Int64
                if let attrs = try? FileManager.default.attributesOfItem(atPath: partialPath),
                   let size = attrs[.size] as? Int64, size > 0 {
                    existingBytes = size
                    downloads[repoId]?.statusText = "Resuming \(ggufFilename) from \(formatBytes(existingBytes))..."
                    downloads[repoId]?.fileProgress = fileSize > 0 ? Double(existingBytes) / Double(fileSize) : 0
                } else {
                    existingBytes = 0
                }

                let fm = FileManager.default
                if !fm.fileExists(atPath: partialPath) {
                    fm.createFile(atPath: partialPath, contents: nil)
                }
                guard let fileHandle = FileHandle(forWritingAtPath: partialPath) else {
                    throw URLError(.cannotCreateFile)
                }
                try fileHandle.seekToEnd()

                var request = URLRequest(url: fileURL)
                if existingBytes > 0 {
                    request.setValue("bytes=\(existingBytes)-", forHTTPHeaderField: "Range")
                }

                do {
                    try await downloadToFile(
                        request: request,
                        fileHandle: fileHandle,
                        repoId: repoId,
                        fileSize: fileSize,
                        existingBytes: existingBytes,
                        baseDownloaded: 0,
                        totalSize: max(fileSize, 1)
                    )
                    try? fm.removeItem(atPath: destPath)
                    try fm.moveItem(atPath: partialPath, toPath: destPath)
                    break
                } catch {
                    if Self.isCancellation(error) { throw CancellationError() }
                    if attempt < maxRetries - 1 {
                        let isStall = error is DownloadStallError
                        let delay = isStall ? 2.0 : Double(attempt + 1) * 2.0
                        let reason = isStall ? "Download stalled" : "Connection lost"
                        downloads[repoId]?.statusText = "\(reason), retrying in \(Int(delay))s... (\(attempt + 2)/\(maxRetries))"
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    } else {
                        throw error
                    }
                }
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete", fileIndex: 1, fileCount: 1)
        } catch {
            if Task.isCancelled { return }
            let message = error.localizedDescription
            downloads[repoId] = DownloadState(status: .failed, error: message)
            if !(error is CancellationError) {
                presentFailureAlert(repoId: repoId, message: message)
            }
        }
    }

    private func presentFailureAlert(repoId: String, message: String) {
        let modelName = repoId.components(separatedBy: "/").last ?? repoId
        let alert = NSAlert()
        alert.messageText = "Download Failed: \(modelName)"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        // LSUIElement app — bring focus to make sure the alert is visible.
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    /// Stream download directly to a file on disk (survives interruptions).
    /// Uses dataTask so bytes are written as they arrive — the .partial file always
    /// reflects how far we got, enabling Range-header resume on retry.
    private func downloadToFile(
        request: URLRequest,
        fileHandle: FileHandle,
        repoId: String,
        fileSize: Int64,
        existingBytes: Int64,
        baseDownloaded: Int64,
        totalSize: Int64
    ) async throws {
        let delegate = StreamingDelegate(fileHandle: fileHandle, existingBytes: existingBytes)
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 7200
        let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                delegate.onProgress = { [weak self] fileBytesTotal, speed in
                    let fileProgress = fileSize > 0 ? Double(fileBytesTotal) / Double(fileSize) : 0
                    let overallDownloaded = baseDownloaded + fileBytesTotal
                    Task { @MainActor [weak self] in
                        self?.downloads[repoId]?.fileProgress = fileProgress
                        self?.downloads[repoId]?.bytesPerSecond = speed
                        self?.downloads[repoId]?.progress = totalSize > 0 ? Double(overallDownloaded) / Double(totalSize) : 0
                    }
                }
                delegate.onComplete = { error in
                    if let error { continuation.resume(throwing: error) }
                    else { continuation.resume() }
                }
                session.dataTask(with: request).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }

    /// Check whether a model has .partial files from an interrupted download.
    func hasPartialDownload(_ repoId: String) -> Bool {
        // Look in the new layout first (where in-progress downloads live), then
        // legacy as a fallback.
        let candidates = [newLayoutDir(for: repoId), existingModelDir(for: repoId)].compactMap { $0 }
        for dir in candidates {
            if let entries = try? FileManager.default.contentsOfDirectory(atPath: dir),
               entries.contains(where: { $0.hasSuffix(".partial") }) {
                return true
            }
        }
        return false
    }

    private func makeLocalModel(at dirPath: String, displayName: String, idKey: String, source: LocalModelSource) -> LocalModel? {
        let resolved = (dirPath as NSString).resolvingSymlinksInPath
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: resolved)) ?? []

        // GGUF fast-path: surface ANY non-mmproj `.gguf` as a selectable base
        // model. Sort first so the pick is deterministic across filesystems
        // (FileManager.contentsOfDirectory order is APFS-specific). When a
        // folder ships both `gemma-4-E4B-it-Q4_K_M.gguf` and `Q8_0.gguf`
        // we deterministically pick the alphabetically-smallest basename
        // (matches the Zig server's `resolveGgufFile` tiebreaker).
        // `isSupportedGguf` already filters mmproj sidecars
        // (`mmproj-*.gguf`, the CLIP vision/audio encoders) so they can't
        // be picked here — the previous code grabbed them on Gemma 4 VL
        // / Qwen 3.6 VL repos and the server then 404'd with
        // 'unsupported model architecture: clip'.
        let ggufCandidates = entries.filter { Self.isSupportedGguf($0) }.sorted()
        if let gguf = ggufCandidates.first,
           let modelType = Self.ggufModelType(forBasename: gguf) {
            let ggufPath = (resolved as NSString).appendingPathComponent(gguf)
            let size = (try? FileManager.default.attributesOfItem(atPath: ggufPath)[.size] as? UInt64) ?? 0
            return LocalModel(
                id: "\(source.rawValue):\(idKey)",
                name: displayName,
                path: ggufPath,
                sizeFormatted: MemoryInfo.format(Int64(size)),
                modelType: modelType,
                source: source,
                kind: .base
            )
        }

        let configPath = (resolved as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath) else { return nil }

        guard entries.contains(where: { $0.hasSuffix(".safetensors") && !$0.hasSuffix(".index.json") }) else { return nil }

        let meta = Self.parseConfigMetadata(atPath: configPath)
        let modelType = meta.modelType

        let size = directorySize(resolved)
        // Drafter config dirs aren't loadable as a target — they pair with a
        // base Gemma 4 model via the `--drafter` flag. Tagging them lets the
        // Model Browser group them separately and the model picker filter
        // them out. `gemma4_unified_assistant` is the newer "unified"
        // architecture (spans dense + MoE targets) shipped with the 12B
        // drafter — same UI treatment as `gemma4_assistant`.
        let kind: ModelKind = Self.drafterModelTypes.contains(modelType) ? .drafter : .base
        return LocalModel(
            id: "\(source.rawValue):\(idKey)",
            name: displayName,
            path: resolved,
            sizeFormatted: MemoryInfo.format(Int64(size)),
            modelType: modelType,
            source: source,
            kind: kind,
            hasVision: meta.hasVision,
            quantBits: meta.quantBits,
            contextLength: meta.contextLength,
            numExperts: meta.numExperts,
            activeExperts: meta.activeExperts
        )
    }

    /// Metadata read from a model's `config.json` — the authoritative source for
    /// quant, context window, MoE expert routing, and vision (the model name only
    /// reliably carries the headline param count, which isn't a config field).
    struct ConfigMetadata: Equatable {
        var modelType = "unknown"
        var hasVision = false
        var quantBits: Int? = nil
        var contextLength: Int? = nil
        var numExperts: Int? = nil
        var activeExperts: Int? = nil
    }

    /// Parse the subset of `config.json` the Downloaded tab surfaces. `nonisolated`
    /// + static so it's unit-testable against a temp config without a real model.
    /// Tolerant of missing keys — every field is optional and defaults sensibly.
    nonisolated static func parseConfigMetadata(atPath configPath: String) -> ConfigMetadata {
        var meta = ConfigMetadata()
        guard let data = FileManager.default.contents(atPath: configPath),
              let cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return meta }
        if let mt = cfg["model_type"] as? String { meta.modelType = mt }
        // Vision: a `vision_config` block on a non-`_text` arch (the `_text`
        // guard skips text-only quantized checkpoints with a vestigial block).
        meta.hasVision = cfg["vision_config"] != nil && !meta.modelType.hasSuffix("_text")
        // Quant: MLX writes `quantization`/`quantization_config` with `bits`.
        if let q = (cfg["quantization"] ?? cfg["quantization_config"]) as? [String: Any] {
            meta.quantBits = q["bits"] as? Int
        }
        meta.contextLength = cfg["max_position_embeddings"] as? Int
        // MoE: total experts under one of several arch-specific keys; active
        // experts per token under `num_experts_per_tok`.
        meta.numExperts = (cfg["num_experts"] ?? cfg["num_local_experts"] ?? cfg["n_routed_experts"]) as? Int
        meta.activeExperts = cfg["num_experts_per_tok"] as? Int
        return meta
    }

    func discoverLocalModels() -> [LocalModel] {
        var out: [LocalModel] = []
        let fm = FileManager.default

        // ~/.mlx-serve/models — scan both layouts.
        // New: <root>/<author>/<name>/config.json (matches LM Studio).
        // Legacy: <root>/<name>/config.json — kept working for users who had
        // models predating the migration that the auto-migrator couldn't classify.
        if let entries = try? fm.contentsOfDirectory(atPath: modelsDir) {
            for entry in entries where !entry.hasPrefix(".") {
                let entryPath = (modelsDir as NSString).appendingPathComponent(entry)
                let directConfig = (entryPath as NSString).appendingPathComponent("config.json")
                if fm.fileExists(atPath: directConfig) {
                    // Legacy flat layout: entry IS the model dir.
                    if let m = makeLocalModel(at: entryPath, displayName: entry, idKey: entry, source: .mlxServe) {
                        out.append(m)
                    }
                } else if let children = try? fm.contentsOfDirectory(atPath: entryPath) {
                    // New layout: entry is an author dir, scan one level deeper.
                    for child in children where !child.hasPrefix(".") {
                        let childPath = (entryPath as NSString).appendingPathComponent(child)
                        let display = "\(entry)/\(child)"
                        if let m = makeLocalModel(at: childPath, displayName: display, idKey: display, source: .mlxServe) {
                            out.append(m)
                        }
                    }
                }
            }
        }

        // LM Studio — two levels deep: <root>/<publisher>/<repo>/
        if let root = lmStudioRoot,
           let pubs = try? FileManager.default.contentsOfDirectory(atPath: root) {
            for pub in pubs where !pub.hasPrefix(".") {
                let pubPath = (root as NSString).appendingPathComponent(pub)
                guard let repos = try? FileManager.default.contentsOfDirectory(atPath: pubPath) else { continue }
                for repo in repos where !repo.hasPrefix(".") {
                    let repoPath = (pubPath as NSString).appendingPathComponent(repo)
                    let display = "\(pub)/\(repo)"
                    if let m = makeLocalModel(at: repoPath, displayName: display, idKey: display, source: .lmStudio) {
                        out.append(m)
                    }
                }
            }
        }

        // User-configured custom root — same dual-layout scan as `~/.mlx-serve/models`.
        // resolvedCustomRoot() handles tilde expansion, existence check, and
        // dedup against the two default roots so a user pointing it at
        // `~/.mlx-serve/models` doesn't produce duplicate picker entries.
        if let root = resolvedCustomRoot(),
           let entries = try? fm.contentsOfDirectory(atPath: root) {
            for entry in entries where !entry.hasPrefix(".") {
                let entryPath = (root as NSString).appendingPathComponent(entry)
                let directConfig = (entryPath as NSString).appendingPathComponent("config.json")
                if fm.fileExists(atPath: directConfig) {
                    if let m = makeLocalModel(at: entryPath, displayName: entry, idKey: "custom:\(entry)", source: .custom) {
                        out.append(m)
                    }
                } else if let children = try? fm.contentsOfDirectory(atPath: entryPath) {
                    for child in children where !child.hasPrefix(".") {
                        let childPath = (entryPath as NSString).appendingPathComponent(child)
                        let display = "\(entry)/\(child)"
                        if let m = makeLocalModel(at: childPath, displayName: display, idKey: "custom:\(display)", source: .custom) {
                            out.append(m)
                        }
                    }
                }
            }
        }

        return out
            .filter { !Self.internalHelperRepos.contains($0.name) }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// `model_type` values that identify a Gemma 4 assistant drafter
    /// checkpoint. `gemma4_assistant` is the original (per-target) flavor;
    /// `gemma4_unified_assistant` ships with the 12B drafter and is a
    /// "unified" architecture spanning dense + MoE targets. Both are
    /// drafters as far as the UI is concerned — server-side support for the
    /// unified variant is a separate Zig change.
    nonisolated static let drafterModelTypes: Set<String> = [
        "gemma4_assistant",
        "gemma4_unified_assistant",
    ]

    /// Walk the given scan roots for published Gemma 4 assistant drafter
    /// directories that declare a drafter `model_type`. Drafters
    /// live under different authors (mlx-community for the `-bf16` quants,
    /// google for the official 12B upload), so we iterate variants and
    /// resolve each repo's `<root>/<author>/<dirname>/` path directly rather
    /// than listing a single author dir. One entry per variant — first root
    /// wins. `nonisolated` so tests can call it with a temp dir.
    nonisolated static func discoverDrafters(in roots: [String]) -> [LocalDrafter] {
        var seenVariants = Set<GemmaVariant>()
        var out: [LocalDrafter] = []
        let fm = FileManager.default

        for root in roots {
            for variant in GemmaVariant.allCases where !seenVariants.contains(variant) {
                let parts = variant.drafterRepoId.split(separator: "/")
                guard parts.count == 2 else { continue }
                let dirPath = ((root as NSString).appendingPathComponent(String(parts[0])) as NSString)
                    .appendingPathComponent(String(parts[1]))
                let configPath = (dirPath as NSString).appendingPathComponent("config.json")
                guard let cfgData = fm.contents(atPath: configPath),
                      let cfg = try? JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
                      let mt = cfg["model_type"] as? String,
                      drafterModelTypes.contains(mt) else { continue }
                out.append(LocalDrafter(url: URL(fileURLWithPath: dirPath), variant: variant))
                seenVariants.insert(variant)
            }
        }
        return out
    }

    /// Mirrors `discoverLocalModels()` — scans `~/.mlx-serve/models/` first,
    /// then LM Studio's root when present. Used by Settings to pick the right
    /// drafter for the loaded base model and by the Model Browser to badge
    /// already-downloaded drafter rows.
    func discoverDrafters() -> [LocalDrafter] {
        var roots = [modelsDir]
        if let lms = lmStudioRoot { roots.append(lms) }
        return Self.discoverDrafters(in: roots)
    }

    /// Pick the drafter that pairs with the loaded base model. Returns nil
    /// when the loaded model isn't Gemma 4, or when no matching drafter is on
    /// disk. Only the directory basename is parsed (`gemma-4-e4b-it-4bit` → E4B).
    func recommendedDrafterFor(modelPath: String, architecture: String, isMoE: Bool) -> LocalDrafter? {
        guard architecture == "gemma4" || architecture == "gemma4_text" else { return nil }
        guard let variant = gemmaVariantFor(modelPath: modelPath, isMoE: isMoE) else { return nil }
        return discoverDrafters().first { $0.variant == variant }
    }

    /// Path-only variant — used before the server has reported `architecture`
    /// (e.g. when AppState auto-syncs `drafterPath` on a model swap). Falls
    /// through to the same parser; non-Gemma paths return nil.
    func recommendedDrafterFromPath(_ modelPath: String) -> LocalDrafter? {
        guard let variant = Self.gemmaVariantFor(modelPath: modelPath, isMoE: false) else { return nil }
        return discoverDrafters().first { $0.variant == variant }
    }

    /// Same parser the recommendation uses, exposed so Model Browser can
    /// label a base-model row with its target drafter ("for E4B").
    nonisolated static func gemmaVariantFor(modelPath: String, isMoE: Bool) -> GemmaVariant? {
        let basename = (modelPath as NSString).lastPathComponent.lowercased()
        // 26B-A4B is the only Gemma 4 MoE today. Match it before the bare
        // "26b" check so the substring scan can't promote a future dense 26B
        // checkpoint into the wrong drafter.
        if isMoE || basename.contains("26b-a4b") { return .moe26B }
        if basename.contains("e4b") { return .E4B }
        if basename.contains("e2b") { return .E2B }
        if basename.contains("12b") { return .gemma12B }
        if basename.contains("31b") { return .gemma31B }
        return nil
    }

    func gemmaVariantFor(modelPath: String, isMoE: Bool) -> GemmaVariant? {
        Self.gemmaVariantFor(modelPath: modelPath, isMoE: isMoE)
    }

    func removeIncomplete(repoId: String) {
        removeFromDisk(repoId: repoId)
    }

    func deleteModel(repoId: String) {
        removeFromDisk(repoId: repoId)
    }

    /// Delete a discovered local model by its real on-disk `path`. Preferred
    /// over `deleteModel(repoId:)` for `LocalModelRow`, whose `model.id` is
    /// source-prefixed (`"mlxServe:author/name"`) and therefore can't be fed to
    /// the repoId-based path resolver — and for LM Studio / custom-root models,
    /// which live outside `modelsDir` entirely. Scopes pruning to the known
    /// scan roots so it never climbs out of a model tree.
    func deleteModel(_ model: LocalModel) {
        var roots = [modelsDir]
        if let lms = lmStudioRoot { roots.append(lms) }
        if let custom = resolvedCustomRoot() { roots.append(custom) }
        Self.removeModelFiles(at: model.path, roots: roots)
        // Clear any lingering download-state row, keyed by the clean repoId
        // (drop the `source:` prefix the LocalModel id carries).
        let cleanId = model.id.split(separator: ":", maxSplits: 1).last.map(String.init) ?? model.id
        downloads.removeValue(forKey: cleanId)
    }

    private func removeFromDisk(repoId: String) {
        let fm = FileManager.default
        // Delete both layouts if present so we don't orphan a legacy copy after
        // a partial migration. Empty author dir is also pruned.
        if let existing = existingModelDir(for: repoId) {
            try? fm.removeItem(atPath: existing)
        }
        // If the new-layout target also exists separately (e.g. interrupted
        // download), remove it too.
        let newPath = newLayoutDir(for: repoId)
        if newPath != existingModelDir(for: repoId), fm.fileExists(atPath: newPath) {
            try? fm.removeItem(atPath: newPath)
        }
        // Prune now-empty author dir.
        let parts = repoId.split(separator: "/").map(String.init)
        if parts.count >= 2 {
            let authorDir = (modelsDir as NSString).appendingPathComponent(parts[parts.count - 2])
            if let kids = try? fm.contentsOfDirectory(atPath: authorDir),
               kids.filter({ !$0.hasPrefix(".") }).isEmpty {
                try? fm.removeItem(atPath: authorDir)
            }
        }
        downloads.removeValue(forKey: repoId)
    }

    private func directorySize(_ path: String) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(atPath: path) else { return 0 }
        var total: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            let fullPath = (path as NSString).appendingPathComponent(file)
            if let attrs = try? FileManager.default.attributesOfItem(atPath: fullPath),
               let size = attrs[.size] as? UInt64 {
                total += size
            }
        }
        return total
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes > 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes > 1_000_000 { return String(format: "%.0f MB", Double(bytes) / 1e6) }
        return "\(bytes) B"
    }
}

// MARK: - Streaming Download Delegate

/// Thrown when a download stalls (speed below threshold for too long) — triggers auto-retry.
private struct DownloadStallError: Error, LocalizedError {
    var errorDescription: String? { "Download stalled — server stopped sending data" }
}

/// Writes received data directly to a file handle as it arrives.
/// If the server returns 200 instead of 206, truncates the file (Range was ignored).
private class StreamingDelegate: NSObject, URLSessionDataDelegate {
    let fileHandle: FileHandle
    var existingBytes: Int64
    var onProgress: ((Int64, Double) -> Void)?   // fileBytesTotal, speed
    var onComplete: ((Error?) -> Void)?
    private var bytesReceived: Int64 = 0
    private var statusCode: Int = 0
    private var writeError: Error?
    private let startTime = Date()
    private var lastProgressUpdate = Date.distantPast

    // Stall detection — cancels the task if speed stays below threshold
    private weak var activeTask: URLSessionDataTask?
    private(set) var stalledOut = false
    private var stallTimer: DispatchSourceTimer?
    private var stallCheckBytes: Int64 = 0
    private var slowSince: Date?
    private static let stallSpeedThreshold: Double = 10_000  // 10 KB/s
    private static let stallTimeout: TimeInterval = 30

    init(fileHandle: FileHandle, existingBytes: Int64) {
        self.fileHandle = fileHandle
        self.existingBytes = existingBytes
    }

    deinit {
        stallTimer?.cancel()
    }

    private func startStallDetection(task: URLSessionDataTask) {
        activeTask = task
        stallCheckBytes = bytesReceived
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now() + 5, repeating: 5)
        timer.setEventHandler { [weak self] in
            self?.checkForStall()
        }
        timer.resume()
        stallTimer = timer
    }

    private func checkForStall() {
        let currentBytes = bytesReceived
        let bytesSinceCheck = currentBytes - stallCheckBytes
        let recentSpeed = Double(bytesSinceCheck) / 5.0  // ~5s interval
        stallCheckBytes = currentBytes

        // Push real-time speed to UI (prevents stale speed when data stops flowing)
        onProgress?(existingBytes + currentBytes, recentSpeed)

        if recentSpeed < Self.stallSpeedThreshold {
            if slowSince == nil {
                slowSince = Date()
            } else if Date().timeIntervalSince(slowSince!) > Self.stallTimeout {
                stalledOut = true
                activeTask?.cancel()
                stallTimer?.cancel()
            }
        } else {
            slowSince = nil
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        if statusCode == 200 {
            // Server ignored Range header — sending full file; start over
            existingBytes = 0
            do {
                try fileHandle.truncate(atOffset: 0)
                try fileHandle.seek(toOffset: 0)
            } catch {
                writeError = error
                completionHandler(.cancel)
                return
            }
            startStallDetection(task: dataTask)
            completionHandler(.allow)
        } else if statusCode == 206 {
            startStallDetection(task: dataTask)
            completionHandler(.allow)
        } else {
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard writeError == nil else { return }
        do {
            try fileHandle.write(contentsOf: data)
        } catch {
            writeError = error
            dataTask.cancel()
            return
        }
        bytesReceived += Int64(data.count)
        let now = Date()
        guard now.timeIntervalSince(lastProgressUpdate) > 0.25 else { return }
        lastProgressUpdate = now
        let elapsed = now.timeIntervalSince(startTime)
        let speed = elapsed > 0 ? Double(bytesReceived) / elapsed : 0
        onProgress?(existingBytes + bytesReceived, speed)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        stallTimer?.cancel()
        try? fileHandle.close()
        let effectiveError = writeError ?? error
        if stalledOut {
            onComplete?(DownloadStallError())
        } else if effectiveError != nil {
            onComplete?(effectiveError)
        } else if statusCode != 0 && statusCode != 200 && statusCode != 206 {
            onComplete?(URLError(.badServerResponse, userInfo: [
                NSLocalizedDescriptionKey: "HTTP \(statusCode)"
            ]))
        } else {
            onComplete?(nil)
        }
    }
}
