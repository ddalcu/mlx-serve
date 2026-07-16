import Foundation
import AppKit

@MainActor
class DownloadManager: ObservableObject {
    @Published var downloads: [String: DownloadState] = [:]

    /// In-flight `download`/`downloadGguf` tasks keyed by repoId, so the
    /// Cancel button can interrupt them. Removed in the wrapper's `defer`.
    private var activeTasks: [String: Task<Void, Never>] = [:]
    /// The shard paths a repo's in-flight GGUF transfer is fetching (one entry
    /// for a single-file quant, many for a sharded one), so a cancel can be
    /// scoped to that one quant instead of the whole folder (which may already
    /// hold quants downloaded earlier). The first entry is the primary shard.
    private var activeGgufShards: [String: [String]] = [:]

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

    /// True iff a filename is a GGUF mlx-serve can serve — i.e. a language-model
    /// quant, not one of the SIDECARS a GGUF folder ships beside it. As of the
    /// embedded llama.cpp engine that's ANY `.gguf` except `isGgufSidecar`.
    /// DeepSeek-V4-Flash routes to the ds4 engine, everything else to llama.cpp
    /// (server-side, by `ggufModelType`).
    nonisolated static func isSupportedGguf(_ filename: String) -> Bool {
        let lower = filename.lowercased()
        guard lower.hasSuffix(".gguf") else { return false }
        return !isGgufSidecar(filename)
    }

    /// True iff a basename is a non-LLM `.gguf` companion file. Two kinds ship
    /// today, and NEITHER is loadable as a language model:
    ///
    /// - `mmproj-*.gguf` — the multimodal-projection sidecar (llama.cpp / ollama /
    ///   LM Studio convention for side-loaded CLIP vision & audio encoders;
    ///   `general.architecture=clip`, llama.cpp refuses to load it as an LLM).
    /// - `*tokenizer*.gguf` — audio/speech tokenizers shipped beside a TTS model
    ///   (live: `qwen3-tts-tokenizer-f16.gguf`, 341 MB, sitting next to
    ///   `qwen3-tts-0.6b-f16.gguf`).
    /// - `*-MTP-*.gguf` — the speculative-decode DRAFT HEAD (llama.cpp / ds4
    ///   convention; live: `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` in
    ///   antirez/deepseek-v4-gguf). Not loadable as a chat model — it's a
    ///   dependency of the main quant, downloaded alongside it.
    ///
    /// This has to be exhaustive because discovery lists EVERY quant in a folder
    /// as a separately selectable model — anything not filtered here becomes a
    /// tray entry the user can pick and the server can only fail to load. Mirrors
    /// the Zig `model_discovery.isGgufSidecarBasename` so client and server agree.
    nonisolated static func isGgufSidecar(_ filename: String) -> Bool {
        let lower = (filename as NSString).lastPathComponent.lowercased()
        guard lower.hasSuffix(".gguf") else { return false }
        // `-MTP-` matched as a delimited token so a chat quant whose scheme name
        // merely contains "mtp" isn't caught.
        return lower.hasPrefix("mmproj") || lower.contains("tokenizer")
            || lower.contains("-mtp-") || lower.contains("-mtp.")
    }

    /// Retained for the mmproj-specific call sites (the Swift mirror of the
    /// server's `isMmprojGgufBasename`).
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
    /// weight files, PLUS the MTP multi-token-prediction sidecar the server
    /// auto-loads. Two nested sidecar layouts are pulled: `mtp/weights.safetensors`
    /// (mlx-serve native) and `optiq/mtp.safetensors` (oMLX OptiQ). Without them
    /// an MTP model silently loses its speculative-decoding speedup because a
    /// non-recursive listing returns the dir as a bare entry that the
    /// `type == "file"` filter drops. Everything else nested — `optiq/optiq_vision.safetensors`
    /// (the server can't use a relocated vision tower, ~GB), `original/` or
    /// alternate-precision shadow copies — is skipped so we don't pull tens of
    /// GB of unused weights. This allowlist mirrors `mtp.sidecar_rel_paths`; keep
    /// them in sync. Returns (path, size) pairs.
    nonisolated static func selectNeededFiles(from entries: [[String: Any]], selection: FileSelection = .chatDefault) -> [(String, Int64)] {
        let neededExtensions: Set<String> = ["json", "safetensors", "jinja", "model", "txt"]
        return entries.compactMap { file -> (String, Int64)? in
            guard let path = file["path"] as? String,
                  let ftype = file["type"] as? String, ftype == "file" else { return nil }
            // Depth gate. Chat default: top-level files + the MTP sidecar
            // (native `mtp/` dir, or OptiQ's single `optiq/mtp.safetensors`).
            // Media (recursive): keep nested weight subdirs (FLUX's
            // transformer/vae/text_encoder, TTS's speech_tokenizer).
            if !selection.recursive {
                guard !path.contains("/") || path.hasPrefix("mtp/") || path == "optiq/mtp.safetensors" else { return nil }
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
    /// back to the legacy flat layout. Returns nil when neither holds a model.
    ///
    /// "Holds a model" is `config.json` OR at least one servable `.gguf` — a
    /// GGUF download writes exactly one file and no config, so gating purely on
    /// config.json made every GGUF folder unresolvable (which in turn made
    /// `isReady`'s GGUF fast-path below dead code, and a downloaded quant read
    /// as missing the moment the in-memory download row went away).
    nonisolated static func existingModelDir(rootDir: String, repoId: String) -> String? {
        let fm = FileManager.default
        let new = newLayoutDir(rootDir: rootDir, repoId: repoId)
        if holdsModel(new, fm: fm) { return new }
        let name = repoId.split(separator: "/").last.map(String.init) ?? repoId
        let legacy = (rootDir as NSString).appendingPathComponent(name)
        if holdsModel(legacy, fm: fm) { return legacy }
        return nil
    }

    private nonisolated static func holdsModel(_ dir: String, fm: FileManager) -> Bool {
        if fm.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json")) { return true }
        // Recursive: a large GGUF quant's only `.gguf` files are nested shards
        // (`<quant>/<quant>-00001-of-00002.gguf`), so a shallow scan misses them.
        return !ggufQuantPaths(inDir: dir).isEmpty
    }

    /// File size in bytes, resolving symlinks first. Hugging Face snapshots
    /// symlink every file into a sibling `blobs/` dir, and a bare
    /// `attributesOfItem` reports the LINK's own size (the target-path length,
    /// ~76 B) rather than the blob's — which would make every HF-cached weight
    /// look sub-1 MB and get filtered out as a stub, or a GGUF row read "0 MB".
    /// A no-op for the real files under the other roots. Returns 0 when absent
    /// or on a dangling link.
    nonisolated static func resolvedFileSize(_ path: String) -> UInt64 {
        let resolved = (path as NSString).resolvingSymlinksInPath
        return (try? FileManager.default.attributesOfItem(atPath: resolved)[.size] as? UInt64) ?? 0
    }

    /// Servable `.gguf` basenames DIRECTLY in a directory, sorted. Excludes
    /// mmproj/tokenizer sidecars and sub-1 MB stubs, and `.partial` files are a
    /// different extension so an in-flight transfer never counts as an on-disk
    /// quant. Non-recursive — for the flat single-file-per-quant layout. Sharded
    /// repos (shards nested in per-quant subfolders) need `ggufQuantPaths`.
    nonisolated static func ggufQuantFiles(inDir dir: String) -> [String] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: dir) else { return [] }
        return entries
            .filter { isSupportedGguf($0) }
            .filter { name in
                resolvedFileSize((dir as NSString).appendingPathComponent(name)) >= 1_000_000
            }
            .sorted()
    }

    /// Servable `.gguf` files under a directory, RECURSIVELY, as repo-relative
    /// paths (e.g. `Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf`), sorted. Same
    /// filters as `ggufQuantFiles` (sidecars + sub-1 MB stubs dropped) but walks
    /// subfolders so a sharded quant's shards are found. `GgufQuant.groupQuants`
    /// folds the returned paths back into quants.
    nonisolated static func ggufQuantPaths(inDir dir: String) -> [String] {
        let fm = FileManager.default
        guard let en = fm.enumerator(atPath: dir) else { return [] }
        var out: [String] = []
        while let rel = en.nextObject() as? String {
            guard isSupportedGguf(rel) else { continue }
            let full = (dir as NSString).appendingPathComponent(rel)
            if resolvedFileSize(full) >= 1_000_000 { out.append(rel) }
        }
        return out.sorted()
    }

    /// Servable `.gguf` quant paths for a MODEL dir specifically (repo-relative,
    /// sorted): top-level flat quants PLUS the shards of immediate "pure quant
    /// subfolders" (`<quant>/<quant>-NNNNN-of-MMMMM.gguf`, a subfolder whose
    /// every entry is a split shard). Unlike `ggufQuantPaths` this does NOT
    /// recurse arbitrarily — `makeLocalModels` is also called on AUTHOR dirs (to
    /// detect "this isn't a model dir, scan its children"), and full recursion
    /// there would fold shards from sibling model repos into one bogus
    /// author-named model. A real model repo directory is not a pure shard
    /// folder, so this can't mistake one for a sharded quant.
    nonisolated static func ggufQuantPathsForModelDir(_ dir: String) -> [String] {
        let fm = FileManager.default
        var out = ggufQuantFiles(inDir: dir)   // flat quants (top-level .gguf)
        if let entries = try? fm.contentsOfDirectory(atPath: dir) {
            for sub in entries where !sub.hasPrefix(".") {
                let subPath = (dir as NSString).appendingPathComponent(sub)
                guard let shards = shardSubfolderShards(subPath) else { continue }
                for shard in shards { out.append((sub as NSString).appendingPathComponent(shard)) }
            }
        }
        return out.sorted()
    }

    /// If `dir` is a pure quant subfolder — non-empty and every entry a servable
    /// `.gguf` SPLIT shard (`-NNNNN-of-MMMMM`), no config/README/nested dirs —
    /// return its shard basenames; else nil. This is what tells a quant
    /// subfolder (`Hy3-IQ1_M/`) apart from a nested model repo, so scanning an
    /// author dir never mistakes a model for a sharded quant. `.partial` files
    /// (an in-flight shard) are ignored, not disqualifying.
    private nonisolated static func shardSubfolderShards(_ dir: String) -> [String]? {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue,
              let entries = try? fm.contentsOfDirectory(atPath: dir), !entries.isEmpty else { return nil }
        var shards: [String] = []
        for e in entries where !e.hasPrefix(".") {
            if e.hasSuffix(".partial") { continue }
            let full = (dir as NSString).appendingPathComponent(e)
            var eIsDir: ObjCBool = false
            fm.fileExists(atPath: full, isDirectory: &eIsDir)
            if eIsDir.boolValue { return nil }                                   // nested dir ⇒ not a shard folder
            guard isSupportedGguf(e), GgufQuant.shardCount(forName: e) != nil else { return nil }
            if resolvedFileSize(full) >= 1_000_000 { shards.append(e) }
        }
        return shards.isEmpty ? nil : shards
    }

    /// The quant basenames of `repoId` present DIRECTLY on disk (flat layout).
    /// Empty for a safetensors repo or one we don't have.
    nonisolated static func downloadedGgufFiles(rootDir: String, repoId: String) -> [String] {
        guard let dir = existingModelDir(rootDir: rootDir, repoId: repoId) else { return [] }
        return ggufQuantFiles(inDir: dir)
    }

    func downloadedGgufFiles(repoId: String) -> [String] {
        Self.downloadedGgufFiles(rootDir: modelsDir, repoId: repoId)
    }

    /// The repo-relative `.gguf` paths of `repoId` present on disk, RECURSIVELY
    /// — the shard-aware input for the Discover row's quant menu. A flat repo
    /// returns basenames (== `downloadedGgufFiles`); a sharded repo returns
    /// nested shard paths so `GgufQuant.groupQuants` can tell a complete quant
    /// from an interrupted one.
    nonisolated static func downloadedGgufPaths(rootDir: String, repoId: String) -> [String] {
        guard let dir = existingModelDir(rootDir: rootDir, repoId: repoId) else { return [] }
        return ggufQuantPaths(inDir: dir)
    }

    func downloadedGgufPaths(repoId: String) -> [String] {
        Self.downloadedGgufPaths(rootDir: modelsDir, repoId: repoId)
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
        if let hf = huggingFaceRoot,
           URL(fileURLWithPath: hf).standardizedFileURL.path == standardized {
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

    /// The Hugging Face hub cache root, resolved once at app launch — where
    /// `huggingface_hub` (and therefore `mlx_lm.load` / `huggingface-cli`)
    /// downloads by default. Honors `HF_HUB_CACHE`, then `$HF_HOME/hub`, then
    /// `~/.cache/huggingface/hub`. nil when none exists on disk. Read-only: the
    /// app scans + loads from it but never writes/deletes into its blob layout.
    let huggingFaceRoot: String? = {
        let env = ProcessInfo.processInfo.environment
        let candidate: String = {
            if let c = env["HF_HUB_CACHE"], !c.isEmpty {
                return (c as NSString).expandingTildeInPath
            }
            if let home = env["HF_HOME"], !home.isEmpty {
                return ((home as NSString).expandingTildeInPath as NSString).appendingPathComponent("hub")
            }
            return NSString(string: "~/.cache/huggingface/hub").expandingTildeInPath
        }()
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
        // legitimately has no config.json still resolves as ready. Any COMPLETE
        // quant makes the repo ready (ds4 for DSV4-Flash, llama.cpp for the
        // rest). Grouping is shard-aware: a single-file quant is one complete
        // group; a sharded quant is ready only once every shard has landed (an
        // interrupted split download must read as not-ready → resume).
        let quantPaths = Self.ggufQuantPaths(inDir: modelDir)
        if !quantPaths.isEmpty {
            if GgufQuant.groupQuants(quantPaths).contains(where: { $0.isComplete }) { return true }
            // A dir whose only quant is an incomplete split isn't ready; fall
            // through to the safetensors gate (which also fails) → false.
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

    /// List the servable `.gguf` paths a HuggingFace repo publishes, RECURSIVELY
    /// (`?recursive=true`), as repo-relative paths sorted by name. Includes
    /// nested split shards (`<quant>/<quant>-00001-of-00002.gguf`) so
    /// `GgufQuant.groupQuants` can fold them into per-quant menu entries; the
    /// download path reassembles a sharded quant from its ordered shard list.
    /// Sidecars (mmproj/tokenizer) are dropped. Empty on error.
    func listGgufFiles(repoId: String) async -> [String] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true") else { return [] }
        guard let (data, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse, http.statusCode == 200,
              let files = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }
        return files
            .compactMap { $0["path"] as? String }
            .filter { Self.isSupportedGguf($0) }
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
    ///
    /// Cancellation is scoped to the ONE quant being fetched: a repo folder can
    /// already hold quants the user downloaded earlier, and the generic
    /// whole-folder wipe would delete them as collateral for cancelling a
    /// second download.
    func startGguf(repoId: String, quant: GgufQuant, onFinish: @escaping @MainActor () -> Void) {
        activeTasks[repoId]?.cancel()
        // Scope cancellation to this quant up front; the task augments the list
        // with the MTP dependency once it's resolved from the repo tree.
        activeGgufShards[repoId] = quant.allFiles
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            let files = await self.resolveGgufDownloadFiles(repoId: repoId, quant: quant)
            self.activeGgufShards[repoId] = files
            await self.downloadGguf(repoId: repoId, files: files)
            self.finalizeIfCancelledGguf(repoId: repoId, shards: files)
            self.activeGgufShards.removeValue(forKey: repoId)
            self.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    /// The full file list to fetch for a GGUF quant: the quant's own shard(s)
    /// PLUS, for a DeepSeek-V4 (ds4) quant, the repo's MTP draft head — the
    /// speculative-decode dependency the ds4 engine auto-loads for a faster
    /// decode. Non-ds4 quants (llama.cpp) don't use an MTP head, so nothing
    /// extra is pulled.
    private func resolveGgufDownloadFiles(repoId: String, quant: GgufQuant) async -> [String] {
        var files = quant.allFiles
        let primaryBase = (quant.filename as NSString).lastPathComponent
        guard Self.ggufModelType(forBasename: primaryBase) == "deepseek_v4" else { return files }
        if let mtp = await repoMtpSidecar(repoId: repoId), !files.contains(mtp) {
            files.append(mtp)
        }
        return files
    }

    /// Fetch the repo's full tree (NO sidecar filter — the MTP head is filtered
    /// out of the selectable-quant lists) and return the MTP draft-head path.
    private func repoMtpSidecar(repoId: String) async -> String? {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true") else { return nil }
        guard let (data, response) = try? await URLSession.shared.data(from: url),
              let http = response as? HTTPURLResponse, http.statusCode == 200,
              let entries = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return nil }
        return Self.mtpSidecarPath(in: entries.compactMap { $0["path"] as? String })
    }

    /// The MTP draft-head path in a repo's file list, if any — the ds4
    /// speculative-decode dependency (`*-MTP-*.gguf`). Pure + testable.
    nonisolated static func mtpSidecarPath(in files: [String]) -> String? {
        files.first { path in
            let base = (path as NSString).lastPathComponent.lowercased()
            return base.hasSuffix(".gguf") && (base.contains("-mtp-") || base.contains("-mtp."))
        }
    }

    /// Convenience for callers that name a single-file quant directly (the
    /// built-in ds4/GGUF catalog entries). Wraps it as a one-file shard group.
    func startGguf(repoId: String, ggufFilename: String, onFinish: @escaping @MainActor () -> Void) {
        startGguf(
            repoId: repoId,
            quant: GgufQuant(filename: ggufFilename, label: Self.quantLabel(forFilename: ggufFilename)),
            onFinish: onFinish
        )
    }

    /// Post-await cleanup for `startGguf`. Removes only the cancelled quant's
    /// shards (via `removeGgufQuant` on the primary, which for a sharded quant
    /// takes the whole subfolder), taking the repo folder down only when nothing
    /// servable is left — never a sibling quant.
    private func finalizeIfCancelledGguf(repoId: String, shards: [String]) {
        guard Task.isCancelled else { return }
        downloads.removeValue(forKey: repoId)
        guard let primary = shards.first else { return }
        let dir = existingModelDir(for: repoId) ?? newLayoutDir(for: repoId)
        removeGgufQuant(at: (dir as NSString).appendingPathComponent(primary))
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
            // A GGUF folder can hold quants from earlier downloads — those are
            // finished models, not this transfer's remnants, so the whole-folder
            // wipe must not run over them.
            if let shards = activeGgufShards[repoId], let primary = shards.first {
                let dir = existingModelDir(for: repoId) ?? newLayoutDir(for: repoId)
                removeGgufQuant(at: (dir as NSString).appendingPathComponent(primary))
                activeGgufShards.removeValue(forKey: repoId)
            } else if downloadedGgufPaths(repoId: repoId).isEmpty {
                wipeDownloadDir(repoId)
            }
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

    /// Delete ONE quant of a GGUF repo, leaving its siblings alone. A repo
    /// folder holds many independently-loadable quants, so removing the folder
    /// (what `removeModelFiles` does) would destroy quants the user never asked
    /// to delete. When the last servable quant goes the folder goes with it —
    /// an orphaned mmproj sidecar or README is dead weight — and an emptied
    /// author dir is pruned, never climbing past a scan root.
    ///
    /// `path` is the quant's PRIMARY file: a single-file quant's `.gguf`, or a
    /// sharded quant's `-00001-of-…` shard. For a sharded quant the whole quant
    /// subfolder (every shard) goes, then the repo folder if that emptied it.
    /// Returns true when `path` is gone.
    @discardableResult
    nonisolated static func removeGgufQuant(at path: String, roots: [String]) -> Bool {
        let fm = FileManager.default

        // Sharded quant: `path` is a `-NNNNN-of-MMMMM` shard whose siblings live
        // in the SAME per-quant subfolder. Remove the subfolder outright, then
        // prune the repo folder (and its author dir) if no servable quant is
        // left. Distinct from the single-file arm, whose containing dir IS the
        // repo folder holding sibling quants.
        if GgufQuant.shardCount(forName: path) != nil {
            let quantDir = (path as NSString).deletingLastPathComponent
            try? fm.removeItem(atPath: quantDir)
            let repoDir = (quantDir as NSString).deletingLastPathComponent
            if ggufQuantPaths(inDir: repoDir).isEmpty,
               !fm.fileExists(atPath: (repoDir as NSString).appendingPathComponent("config.json")) {
                removeModelFiles(at: repoDir, roots: roots)
            }
            return !fm.fileExists(atPath: path)
        }

        guard fm.fileExists(atPath: path) else { return false }
        try? fm.removeItem(atPath: path)
        try? fm.removeItem(atPath: path + ".partial")

        let dir = (path as NSString).deletingLastPathComponent
        if ggufQuantPaths(inDir: dir).isEmpty,
           !fm.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json")) {
            removeModelFiles(at: dir, roots: roots)
        }
        return !fm.fileExists(atPath: path)
    }

    /// Instance form, scoped to every root we scan (a GGUF quant can live under
    /// LM Studio's tree or a custom folder, not just `~/.mlx-serve/models`).
    @discardableResult
    func removeGgufQuant(at path: String) -> Bool {
        var roots = [modelsDir]
        if let lms = lmStudioRoot { roots.append(lms) }
        if let custom = resolvedCustomRoot() { roots.append(custom) }
        return Self.removeGgufQuant(at: path, roots: roots)
    }

    /// Download a GGUF quant's files from a HuggingFace repo. `files` is one
    /// repo-relative path for a single-file quant (the ds4-backed entries, e.g.
    /// DeepSeek-V4-Flash, and single-file GGUF picks), a sharded quant's ordered
    /// shard list (large GGUFs HF splits over ~50 GB), and/or any auto-download
    /// dependency (the ds4 MTP draft head). Mirrors `download(repoId:)`'s
    /// resume/retry/disk-space shape, looped over each file; progress is
    /// `fileIndex/fileCount` and byte progress spans them all. A nested subfolder
    /// (`<quant>/<quant>-00001-of-…`) is created as needed, mirroring HF's layout.
    func downloadGguf(repoId: String, files shards: [String]) async {
        let destDir = newLayoutDir(for: repoId)
        let primaryName = ((shards.first ?? "") as NSString).lastPathComponent
        downloads[repoId] = DownloadState(status: .downloading, statusText: "Fetching \(primaryName)...")

        do {
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)

            // HEAD every shard up front so progress + the disk-space precheck see
            // the WHOLE quant, not just the first shard.
            var sizes: [Int64] = []
            for rel in shards {
                try Task.checkCancellation()
                let fileURL = ggufShardURL(repoId: repoId, path: rel)
                var headReq = URLRequest(url: fileURL)
                headReq.httpMethod = "HEAD"
                let (_, headResp) = try await URLSession.shared.data(for: headReq)
                let sz: Int64 = {
                    guard let http = headResp as? HTTPURLResponse else { return 0 }
                    if let cl = http.value(forHTTPHeaderField: "Content-Length"), let n = Int64(cl) { return n }
                    return http.expectedContentLength
                }()
                sizes.append(sz)
            }
            let totalSize = max(sizes.reduce(Int64(0), +), 1)

            let destURL = URL(fileURLWithPath: destDir)
            if let values = try? destURL.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let available = values.volumeAvailableCapacityForImportantUsage,
               totalSize > 1, available < totalSize {
                throw NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError, userInfo: [
                    NSLocalizedDescriptionKey: "Not enough disk space. Need \(formatBytes(totalSize)) but only \(formatBytes(Int64(available))) available."
                ])
            }

            var baseDownloaded: Int64 = 0
            let fm = FileManager.default

            for (idx, rel) in shards.enumerated() {
                try Task.checkCancellation()
                let fileSize = sizes[idx]
                let shardName = (rel as NSString).lastPathComponent
                let fileURL = ggufShardURL(repoId: repoId, path: rel)
                let destPath = (destDir as NSString).appendingPathComponent(rel)
                let partialPath = destPath + ".partial"
                // Nested shard subfolder (`<quant>/`) — created on demand.
                try? fm.createDirectory(atPath: (destPath as NSString).deletingLastPathComponent, withIntermediateDirectories: true)

                downloads[repoId]?.currentFile = shardName
                downloads[repoId]?.fileIndex = idx + 1
                downloads[repoId]?.fileCount = shards.count

                // Skip a shard already present at its expected size (resume across
                // an interrupted multi-shard download).
                if fileSize > 0,
                   let attrs = try? fm.attributesOfItem(atPath: destPath),
                   let existingSize = attrs[.size] as? Int64,
                   existingSize == fileSize {
                    baseDownloaded += fileSize
                    downloads[repoId]?.progress = Double(baseDownloaded) / Double(totalSize)
                    continue
                }

                let maxRetries = 20
                for attempt in 0..<maxRetries {
                    try Task.checkCancellation()

                    let existingBytes: Int64
                    if let attrs = try? fm.attributesOfItem(atPath: partialPath),
                       let size = attrs[.size] as? Int64, size > 0 {
                        existingBytes = size
                        downloads[repoId]?.statusText = "Resuming \(shardName) from \(formatBytes(existingBytes))..."
                        downloads[repoId]?.fileProgress = fileSize > 0 ? Double(existingBytes) / Double(fileSize) : 0
                    } else {
                        existingBytes = 0
                    }

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
                            baseDownloaded: baseDownloaded,
                            totalSize: totalSize
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

                baseDownloaded += fileSize
                downloads[repoId]?.progress = Double(baseDownloaded) / Double(totalSize)
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete", fileIndex: shards.count, fileCount: shards.count)
        } catch {
            if Task.isCancelled { return }
            let message = error.localizedDescription
            downloads[repoId] = DownloadState(status: .failed, error: message)
            if !(error is CancellationError) {
                presentFailureAlert(repoId: repoId, message: message)
            }
        }
    }

    /// The HF `resolve` URL for a repo-relative shard path. Percent-encodes each
    /// path segment (leaving the `/` separators) so an unusual filename can't
    /// produce a nil `URL`.
    private func ggufShardURL(repoId: String, path: String) -> URL {
        let encoded = path
            .split(separator: "/", omittingEmptySubsequences: false)
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        return URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(encoded)")!
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

    /// Every model a directory holds.
    ///
    /// A safetensors checkpoint is exactly one model (the directory). A GGUF
    /// repo is one model PER QUANT: the folder ships `…-Q4_K_M.gguf`,
    /// `…-Q8_0.gguf`, … and each is independently loadable, so each gets its own
    /// `LocalModel` whose `path` is the FILE. Previously this returned only the
    /// alphabetically-smallest quant and silently dropped the rest, which is why
    /// the tray picker could never offer a second quant of a repo you'd
    /// downloaded two of.
    ///
    /// `nonisolated` + static so it's testable against a temp dir.
    nonisolated static func makeLocalModels(atDir dirPath: String, displayName: String, idKey: String, source: LocalModelSource) -> [LocalModel] {
        let resolved = (dirPath as NSString).resolvingSymlinksInPath
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: resolved)) ?? []

        // GGUF: one entry per servable quant (SHARD GROUP), sorted so the order
        // is stable across filesystems. `ggufQuantPaths` walks subfolders (so a
        // sharded quant's nested shards are found) and drops mmproj/tokenizer
        // sidecars; `groupQuants` folds a split quant's shards into one entry
        // whose `path` is the primary `-00001` shard — the path the server loads
        // (libllama auto-loads the rest). The old flat scan grabbed CLIP
        // sidecars on VL repos (server 404'd 'unsupported architecture: clip')
        // and couldn't see sharded quants at all.
        // Depth-limited (top-level quants + immediate pure-shard subfolders) so
        // an author-dir call finds nothing and the discovery walk recurses into
        // it instead of merging sibling repos. Only COMPLETE quants (every shard
        // present) become models — an interrupted split stays a resumable partial.
        let quantPaths = ggufQuantPathsForModelDir(resolved)
        if !quantPaths.isEmpty {
            let complete = GgufQuant.groupQuants(quantPaths).filter { $0.isComplete }
            return complete.compactMap { quant -> LocalModel? in
                let primaryBase = (quant.filename as NSString).lastPathComponent
                guard let modelType = ggufModelType(forBasename: primaryBase) else { return nil }
                let primaryPath = (resolved as NSString).appendingPathComponent(quant.filename)
                // Size = the whole quant (sum across every shard on disk).
                // resolvedFileSize follows symlinks so an HF-cached quant reports
                // its blob size, not the ~76 B link size.
                let size = quant.allFiles.reduce(UInt64(0)) { acc, rel in
                    acc + resolvedFileSize((resolved as NSString).appendingPathComponent(rel))
                }
                return LocalModel(
                    // The primary shard, not the folder — two quants of one repo
                    // must not collide on id or SwiftUI collapses them into one row.
                    id: "\(source.rawValue):\(idKey)#\(quant.filename)",
                    name: displayName,
                    path: primaryPath,
                    sizeFormatted: MemoryInfo.format(Int64(size)),
                    modelType: modelType,
                    source: source,
                    kind: .base,
                    quantFile: primaryBase
                )
            }
        }

        let configPath = (resolved as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath) else { return [] }

        guard entries.contains(where: { $0.hasSuffix(".safetensors") && !$0.hasSuffix(".index.json") }) else { return [] }

        let meta = parseConfigMetadata(atPath: configPath)
        let modelType = meta.modelType

        let size = directorySize(resolved)
        // Drafter config dirs aren't loadable as a target — they pair with a
        // base Gemma 4 model via the `--drafter` flag. Tagging them lets the
        // Model Browser group them separately and the model picker filter
        // them out. `gemma4_unified_assistant` is the newer "unified"
        // architecture (spans dense + MoE targets) shipped with the 12B
        // drafter — same UI treatment as `gemma4_assistant`.
        let kind: ModelKind = drafterModelTypes.contains(modelType) ? .drafter : .base
        return [LocalModel(
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
        )]
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
        // Whether `entry` is itself a model dir (legacy flat) or an author dir
        // (new layout) is decided by what `makeLocalModels` finds in it — NOT by
        // config.json, which a GGUF-only folder never has.
        if let entries = try? fm.contentsOfDirectory(atPath: modelsDir) {
            for entry in entries where !entry.hasPrefix(".") {
                let entryPath = (modelsDir as NSString).appendingPathComponent(entry)
                let direct = Self.makeLocalModels(atDir: entryPath, displayName: entry, idKey: entry, source: .mlxServe)
                if !direct.isEmpty {
                    // Legacy flat layout: entry IS the model dir.
                    out.append(contentsOf: direct)
                } else if let children = try? fm.contentsOfDirectory(atPath: entryPath) {
                    // New layout: entry is an author dir, scan one level deeper.
                    for child in children where !child.hasPrefix(".") {
                        let childPath = (entryPath as NSString).appendingPathComponent(child)
                        let display = "\(entry)/\(child)"
                        out.append(contentsOf: Self.makeLocalModels(atDir: childPath, displayName: display, idKey: display, source: .mlxServe))
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
                    out.append(contentsOf: Self.makeLocalModels(atDir: repoPath, displayName: display, idKey: display, source: .lmStudio))
                }
            }
        }

        // Hugging Face hub cache — `models--<org>--<repo>/snapshots/<commit>/`
        // with the active snapshot named by `refs/main`. Read-only.
        if let root = huggingFaceRoot {
            out.append(contentsOf: Self.discoverHuggingFaceModels(in: root))
        }

        // User-configured custom root — same dual-layout scan as `~/.mlx-serve/models`.
        // resolvedCustomRoot() handles tilde expansion, existence check, and
        // dedup against the two default roots so a user pointing it at
        // `~/.mlx-serve/models` doesn't produce duplicate picker entries.
        if let root = resolvedCustomRoot(),
           let entries = try? fm.contentsOfDirectory(atPath: root) {
            for entry in entries where !entry.hasPrefix(".") {
                let entryPath = (root as NSString).appendingPathComponent(entry)
                let direct = Self.makeLocalModels(atDir: entryPath, displayName: entry, idKey: "custom:\(entry)", source: .custom)
                if !direct.isEmpty {
                    out.append(contentsOf: direct)
                } else if let children = try? fm.contentsOfDirectory(atPath: entryPath) {
                    for child in children where !child.hasPrefix(".") {
                        let childPath = (entryPath as NSString).appendingPathComponent(child)
                        let display = "\(entry)/\(child)"
                        out.append(contentsOf: Self.makeLocalModels(atDir: childPath, displayName: display, idKey: "custom:\(display)", source: .custom))
                    }
                }
            }
        }

        return out
            .filter { !Self.internalHelperRepos.contains($0.name) }
            // By label, not name: sibling quants of one repo share a name, and a
            // name-only sort leaves their relative order at the mercy of the
            // filesystem.
            .sorted { $0.displayLabel.localizedCaseInsensitiveCompare($1.displayLabel) == .orderedAscending }
    }

    /// Every loadable model in a Hugging Face hub cache root. HF stores each repo
    /// as `models--<org>--<repo>/`, with the files symlinked into a sibling
    /// `blobs/` dir under `snapshots/<commit>/`; `refs/main` names the active
    /// commit. We scan the active snapshot dir through `makeLocalModels`, which
    /// drops any snapshot that isn't a loadable model (needs config.json +
    /// safetensors, or a servable GGUF) — so partial/metadata-only pulls fall
    /// away. `datasets--`/`spaces--` cache dirs share the root and are skipped.
    /// `nonisolated` + static so it's testable against a temp dir.
    nonisolated static func discoverHuggingFaceModels(in root: String) -> [LocalModel] {
        var out: [LocalModel] = []
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: root) else { return out }
        for entry in entries where entry.hasPrefix("models--") {
            let repoId = huggingFaceRepoId(fromCacheDir: entry)
            let repoDir = (root as NSString).appendingPathComponent(entry)
            guard let snapshot = huggingFaceActiveSnapshotDir(repoDir: repoDir) else { continue }
            out.append(contentsOf: makeLocalModels(atDir: snapshot, displayName: repoId,
                                                   idKey: "hf:\(repoId)", source: .huggingFace))
        }
        return out
    }

    /// `models--<org>--<repo>` → `<org>/<repo>`. HF encodes the repo id by
    /// replacing `/` with `--`; the repo NAME may itself carry single dashes, so
    /// split on `--` and rejoin everything past the org. A bare `models--<name>`
    /// (no org, e.g. `models--gpt2`) returns just the name. `nonisolated`.
    nonisolated static func huggingFaceRepoId(fromCacheDir dir: String) -> String {
        let stripped = dir.hasPrefix("models--") ? String(dir.dropFirst("models--".count)) : dir
        let parts = stripped.components(separatedBy: "--")
        guard parts.count >= 2 else { return stripped }
        return "\(parts[0])/\(parts.dropFirst().joined(separator: "--"))"
    }

    /// The snapshot directory the cache currently points `main` at. Reads
    /// `refs/main` for the commit hash and returns `snapshots/<hash>/` when it
    /// exists. With no ref (or a dangling one) it falls back to the sole snapshot
    /// dir; when several snapshots exist and no ref disambiguates, it returns nil
    /// rather than guess which revision is canonical. `nonisolated`.
    nonisolated static func huggingFaceActiveSnapshotDir(repoDir: String) -> String? {
        let fm = FileManager.default
        let snapshotsDir = (repoDir as NSString).appendingPathComponent("snapshots")
        let refPath = ((repoDir as NSString).appendingPathComponent("refs") as NSString)
            .appendingPathComponent("main")
        if let data = fm.contents(atPath: refPath),
           let hash = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
           !hash.isEmpty {
            let dir = (snapshotsDir as NSString).appendingPathComponent(hash)
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue { return dir }
        }
        // Fallback: exactly one snapshot dir → use it; otherwise we can't know
        // which revision `main` intends, so skip the repo.
        guard let snaps = try? fm.contentsOfDirectory(atPath: snapshotsDir) else { return nil }
        let dirs = snaps.filter { !$0.hasPrefix(".") }
        guard dirs.count == 1 else { return nil }
        return (snapshotsDir as NSString).appendingPathComponent(dirs[0])
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
        // Only ~/.mlx-serve/models is ours to delete. LM Studio, the Hugging Face
        // hub cache, and custom-root models are owned by another tool or the user
        // (deleting an HF snapshot orphans shared blobs and dangles refs/main; the
        // others simply aren't ours). The UI hides the trash for them; this is the
        // defensive backstop, and the roots are scoped to modelsDir so a stray call
        // can never prune into an external tree.
        guard model.isDeletable else { return }
        let roots = [modelsDir]
        if model.quantFile != nil {
            // One quant of a GGUF repo — remove that file only. Its siblings are
            // separate models the user didn't ask to delete.
            Self.removeGgufQuant(at: model.path, roots: roots)
        } else {
            Self.removeModelFiles(at: model.path, roots: roots)
        }
        // Clear any lingering download-state row, keyed by the clean repoId
        // (drop the `source:` prefix and the `#quant.gguf` suffix the
        // LocalModel id carries).
        let afterSource = model.id.split(separator: ":", maxSplits: 1).last.map(String.init) ?? model.id
        let cleanId = afterSource.split(separator: "#", maxSplits: 1).first.map(String.init) ?? afterSource
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

    nonisolated private static func directorySize(_ path: String) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(atPath: path) else { return 0 }
        var total: UInt64 = 0
        while let file = enumerator.nextObject() as? String {
            // resolvedFileSize follows symlinks: an HF snapshot's files are
            // symlinks into `blobs/`, so a bare stat would report the ~20 B link
            // size and make every HF model look like ~0 B. No-op for real files.
            total += resolvedFileSize((path as NSString).appendingPathComponent(file))
        }
        return total
    }

    /// Test seam for `directorySize` (private): pins the symlink-resolving size
    /// accounting an HF snapshot relies on.
    nonisolated static func directorySizeForTesting(_ path: String) -> UInt64 {
        directorySize(path)
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
