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
    /// The destination repoId an in-flight MLX-variant transfer is writing to
    /// (`<org>/<repo>-4bit`), keyed by the SOURCE repo the progress row belongs
    /// to. Same reason as `activeGgufShards`: a cancel must take down the one
    /// variant being fetched, never the sibling quants already on disk.
    private var activeVariantDest: [String: String] = [:]

    struct DownloadState {
        /// Fraction of the WHOLE transfer: bytes banked across every file
        /// divided by the repo's total. There is deliberately no per-file
        /// fraction to render — every bar in the app drew one, so a repo of
        /// four shards ran 0→100% four times and read as a download that kept
        /// restarting. The file being fetched is named by `currentFile` /
        /// `fileIndex`, which is the honest way to say "still going".
        var progress: Double = 0
        var status: Status = .idle
        var statusText: String = ""
        var error: String?
        var currentFile: String = ""
        var fileIndex: Int = 0
        var fileCount: Int = 0
        var bytesPerSecond: Double = 0

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
            String(format: "%.0f%%", progress * 100)
        }
    }

    /// Where new downloads land. `~/.mlx-serve/models` unless the user has
    /// chosen a folder in Settings (Developer ID builds only — see
    /// `BuildFeatures.customModelFolders`).
    ///
    /// Cached rather than computed per read: a transfer in flight must not have
    /// its destination move out from under it because the user opened Settings.
    /// `refreshRoots()` is the one place it changes.
    @Published private(set) var modelsDir: String

    /// Set only by the test initializer. When present it PINS the destination —
    /// a test driving the download loop against a temp dir must not be steered
    /// by whatever the developer happens to have configured.
    private let pinnedRoot: String?

    /// `modelsRoot` exists so the download loop can be driven against a temp
    /// dir in tests; the app takes the configured destination. Kept in sync
    /// with the Zig resolver via `ModelRoots`, which is the single source of
    /// truth for both this and what the server is told to scan.
    init(modelsRoot: String? = nil) {
        if modelsRoot == nil { Self.reactivateDownloadFolderAccess() }
        let root = modelsRoot ?? ModelRoots().downloadRoot
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        modelsDir = root
        pinnedRoot = modelsRoot
    }

    /// Security-scoped bookmark for the chosen download folder, re-armed at the
    /// point the destination is adopted. A no-op in the unsandboxed Developer
    /// ID build (which is the only build that can pick a folder today), and the
    /// convention every other picked path here follows — a bookmark stored and
    /// never started is a folder that works until relaunch.
    static let downloadFolderBookmarkName = "modelDownloadFolder"

    private static func reactivateDownloadFolderAccess() {
        guard ModelRoots().configuredDownloadRoot != nil else { return }
        _ = SecurityScopedBookmark.startAccessOnce(name: downloadFolderBookmarkName)
    }

    /// Re-read the configured download folder. Called when the setting changes;
    /// a transfer already running keeps the destination it started with, which
    /// is the only way a partial file and its `.parts` sidecar stay together.
    func refreshRoots() {
        guard pinnedRoot == nil else { return }
        Self.reactivateDownloadFolderAccess()
        let root = ModelRoots().downloadRoot
        guard root != modelsDir else { return }
        try? FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
        modelsDir = root
    }

    /// Every folder to scan, download destination first. This is what the
    /// server is handed, one `--model-dir` per entry.
    func scanRoots() -> [String] {
        ModelRoots().scanRoots(toolRoots: ToolModelRoots.detected(lmStudioRoot: lmStudioRoot))
    }

    /// The folders the app owns for READS — where a repo the user already
    /// downloaded may live: `modelsDir` (the write destination) first, then the
    /// built-in root, which keeps everything downloaded before the destination
    /// moved. Writes and cancel-cleanup stay on `modelsDir` alone. A test-pinned
    /// root stays alone so a temp-dir test can never resolve into — or delete
    /// from — the developer's real library.
    var ownedRoots: [String] {
        guard pinnedRoot == nil, modelsDir != ModelRoots.builtInRoot else { return [modelsDir] }
        return [modelsDir, ModelRoots.builtInRoot]
    }

    /// The folders READS check — everything the server scans (destination,
    /// built-in, LM Studio, custom folder), so a pack in ANY served folder
    /// never reads as "not downloaded" (`ModelRoots.readRoots`). Writes and
    /// deletes stay on `ownedRoots`. A test-pinned root stays alone —
    /// hermetic tests must never resolve into the developer's real library.
    var readRoots: [String] {
        guard pinnedRoot == nil else { return [modelsDir] }
        return ModelRoots().readRoots(toolRoots: ToolModelRoots.detected(lmStudioRoot: lmStudioRoot))
    }

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
        // `-MTP-` / `-DSpark-` matched as delimited tokens so a chat quant
        // whose scheme name merely contains the letters isn't caught. The
        // DSpark support GGUF (`DeepSeek-V4-Flash-DSpark-support.gguf`) is
        // 0731's replacement for the legacy MTP draft head — same --mtp slot
        // server-side, never a servable chat quant.
        return lower.hasPrefix("mmproj") || lower.contains("tokenizer")
            || lower.contains("-mtp-") || lower.contains("-mtp.")
            || lower.contains("-dspark-") || lower.contains("-dspark.")
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
    /// quant token like `Q4_K_M` / `IQ2_XXS` / `MXFP4` / `F16` when present, else
    /// the extension-stripped basename. Pure + testable.
    ///
    /// `MXFP`/`NVFP` lead the alternation because they are real quant families,
    /// not an `F`-then-digit accident: without them `…-MXFP4Experts-F16HC-…`
    /// skipped past its own scheme and labelled a 4-bit file "F16HC".
    nonisolated static func quantLabel(forFilename filename: String) -> String {
        let base = (filename as NSString).lastPathComponent
        if let r = base.range(of: "(MXFP|NVFP|IQ|Q|BF|F)[0-9][A-Za-z0-9_]*", options: [.regularExpression, .caseInsensitive]) {
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
            // Depth gate. Variant: exactly the named subfolder's own files
            // (`4bit/config.json`), never anything deeper. Chat default:
            // top-level files + the MTP sidecar (native `mtp/` dir, or OptiQ's
            // single `optiq/mtp.safetensors`). Media (recursive): keep nested
            // weight subdirs (FLUX's transformer/vae/text_encoder, TTS's
            // speech_tokenizer).
            if let sub = selection.subfolder {
                guard path.hasPrefix(sub + "/") else { return nil }
                guard !path.dropFirst(sub.count + 1).contains("/") else { return nil }
            } else if !selection.recursive {
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
        if holdsWeightLayout(dir) { return true }
        // Recursive: a large GGUF quant's only `.gguf` files are nested shards
        // (`<quant>/<quant>-00001-of-00002.gguf`), so a shallow scan misses them.
        return !ggufQuantPaths(inDir: dir).isEmpty
    }

    /// True when `dir` holds a safetensors model of any layout we serve. The
    /// ONE answer to "is this repo on disk?" for everything that isn't a GGUF —
    /// `ServerManager.resolveModelDir` asks it too, because a second copy of the
    /// marker list is exactly how the two sites drifted apart on klein 9B.
    ///
    /// Three shapes, each a fact about real repos rather than a preference:
    /// standard MLX (`config.json`), diffusers (`model_index.json` + weight
    /// subdirs, Mage-Flow), and CONFIGLESS weight subdirs — no root json at all,
    /// which is what `mlx-community/flux2-klein-9b-4bit` ships. Gating on the
    /// root markers made its complete 8.9 GB download read as absent forever
    /// (the Kokoro-vs-`tts()` bug: the pane offers Download, the click reverts).
    /// The server has the same problem and solves it the same way — it
    /// classifies that repo from the DiT's weight NAMES (`peekMfluxFlux2`), not
    /// from a root file.
    nonisolated static func holdsWeightLayout(_ dir: String) -> Bool {
        let fm = FileManager.default
        for marker in ["config.json", "model_index.json"] {
            if fm.fileExists(atPath: (dir as NSString).appendingPathComponent(marker)) { return true }
        }
        // A `transformer/` holding real weights. Deliberately not "has the
        // subdir": a download that got as far as creating the folder must still
        // read as incomplete, and an in-flight transfer's `.partial` is not a
        // weight. The DiT dir is the narrowest marker that no chat model has.
        let dit = (dir as NSString).appendingPathComponent("transformer")
        let shards = (try? fm.contentsOfDirectory(atPath: dit)) ?? []
        return shards.contains { $0.hasSuffix(".safetensors") }
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
        guard let dir = existingModelDir(for: repoId) else { return [] }
        return Self.ggufQuantFiles(inDir: dir)
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
        guard let dir = existingModelDir(for: repoId) else { return [] }
        return Self.ggufQuantPaths(inDir: dir)
    }

    func newLayoutDir(for repoId: String) -> String {
        Self.newLayoutDir(rootDir: modelsDir, repoId: repoId)
    }

    /// Where `repoId` lives, across every SERVED root. Reads must check them
    /// all: resolving against the destination alone is how moving it made a
    /// pre-move library read as absent, and skipping the custom scan folder is
    /// how a pack there got a Download bar over a copy already being served.
    /// Write targets keep using `newLayoutDir(for:)`.
    func existingModelDir(for repoId: String) -> String? {
        Self.existingModelDir(roots: readRoots, repoId: repoId)
    }

    /// First root holding the repo wins — the destination leads `ownedRoots`,
    /// so its copy shadows one in the built-in folder, mirroring the server's
    /// first-wins rule on repeated `--model-dir` flags.
    nonisolated static func existingModelDir(roots: [String], repoId: String) -> String? {
        for root in roots {
            if let dir = existingModelDir(rootDir: root, repoId: repoId) { return dir }
        }
        return nil
    }

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
        for owned in ownedRoots
        where URL(fileURLWithPath: owned).standardizedFileURL.path == standardized {
            return nil
        }
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
    let lmStudioRoot: String? = DownloadManager.lmStudioRootPath()

    /// The same resolution, reachable from `nonisolated` code — the launch-flag
    /// builder needs it and cannot touch this `@MainActor` instance.
    /// `home` is a parameter for the same reason `ToolModelRoots.detected` has
    /// one: a resolver that reaches the real home directory cannot be tested
    /// without depending on whether the machine running the tests happens to
    /// have LM Studio installed.
    nonisolated static func lmStudioRootPath(home: String = NSHomeDirectory()) -> String? {
        let settingsPath = (home as NSString).appendingPathComponent(".lmstudio/settings.json")
        let configured: String? = {
            guard let data = FileManager.default.contents(atPath: settingsPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let folder = json["downloadsFolder"] as? String,
                  !folder.isEmpty else { return nil }
            return (folder as NSString).expandingTildeInPath
        }()
        let fallback = (home as NSString).appendingPathComponent(".lmstudio/models")
        let candidate = configured ?? fallback
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: candidate, isDirectory: &isDir), isDir.boolValue else { return nil }
        return candidate
    }

    /// The Hugging Face hub cache root — where `huggingface_hub` (and therefore
    /// `mlx_lm.load` / `huggingface-cli`) downloads. nil when it does not exist
    /// on disk. Read-only: the app scans + loads from it but never
    /// writes/deletes into its blob layout.
    ///
    /// A `var` because the cache is MOVABLE by environment and a Finder-
    /// launched bundle cannot see the environment that moved it — the login
    /// shell is asked off-main at launch (`refreshHuggingFaceRootFromLoginShell`).
    var huggingFaceRoot: String? = DownloadManager.huggingFaceRootPath()

    /// Resolve the hub cache the way `huggingface_hub` itself does:
    /// `HF_HUB_CACHE` > `$HF_HOME/hub` > `$XDG_CACHE_HOME/huggingface/hub` >
    /// `~/.cache/huggingface/hub`. A CONFIGURED root that is not on disk is nil
    /// rather than a fall-through to the default cache — the whole point of the
    /// variable is that the models live somewhere else, so serving the default
    /// would be answering a question nobody asked.
    nonisolated static func huggingFaceRootPath(
        environment: [String: String] = LoginShellEnv.huggingFaceEnvironment(),
        home: String = NSHomeDirectory()
    ) -> String? {
        func value(_ key: String) -> String? {
            guard let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty else { return nil }
            return (raw as NSString).expandingTildeInPath
        }
        let candidate: String
        if let hub = value("HF_HUB_CACHE") {
            candidate = hub
        } else if let hfHome = value("HF_HOME") {
            candidate = (hfHome as NSString).appendingPathComponent("hub")
        } else if let xdg = value("XDG_CACHE_HOME") {
            candidate = (xdg as NSString).appendingPathComponent("huggingface/hub")
        } else {
            candidate = ((home as NSString).appendingPathComponent(".cache/huggingface/hub"))
        }
        return ModelRoots.existingDirectory(candidate)
    }

    /// Ask the login shell for the HF variables and re-resolve. Returns true
    /// when the root MOVED, so the caller can rescan. Spawns a shell — call it
    /// off the main thread.
    nonisolated static func loginShellHuggingFaceRoot() -> String? {
        LoginShellEnv.primeHuggingFace()
        return huggingFaceRootPath()
    }

    /// Adopt the login shell's answer. True when the root changed.
    func refreshHuggingFaceRootFromLoginShell() async -> Bool {
        let resolved = await Task.detached(priority: .utility) {
            DownloadManager.loginShellHuggingFaceRoot()
        }.value
        guard resolved != huggingFaceRoot else { return false }
        huggingFaceRoot = resolved
        return true
    }

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

    /// `destRepoId` splits WHERE the files land from WHICH repo they come from.
    /// Only a multi-variant MLX quant uses it: the bytes come from
    /// `LiquidAI/LFM2.5-2.6B-MLX` but the model must sit in its own 2-level dir
    /// (`LiquidAI/LFM2.5-2.6B-MLX-4bit`) for the server to discover it. Progress
    /// stays keyed on `repoId` — the row the user clicked is the repo's.
    func download(repoId: String, selection: FileSelection = .chatDefault,
                  alertOnFailure: Bool = true, destRepoId: String? = nil,
                  destDirOverride: String? = nil) async {
        // `destDirOverride`: an absolute dir that already holds the model —
        // used when a fetch ADDS to an existing pack (the Turbo adapter),
        // which may live in a non-destination owned root. Writing it to the
        // destination instead creates a fragment dir that reads as a (broken)
        // model to every resolver (live 2026-08-08).
        let destDir = destDirOverride ?? newLayoutDir(for: destRepoId ?? repoId)

        downloads[repoId] = DownloadState(status: .downloading, statusText: "Fetching file list...")

        do {
            // `?recursive=true` so the listing includes nested sidecars — most
            // importantly the `mtp/` multi-token-prediction head. Without it HF
            // returns `mtp` as a bare directory entry and the file filter skips
            // it, silently dropping the sidecar (and the model's spec-decode
            // speedup). `selectNeededFiles` keeps top-level files + mtp/ only.
            let listURL = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true")!
            let (listData, _) = try await DownloadSession.shared.data(for: Self.hfApiRequest(listURL))
            guard let files = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
                throw URLError(.cannotParseResponse)
            }

            let neededFiles = Self.selectNeededFiles(from: files, selection: selection)

            // A download that matches NOTHING must say so. `LiquidAI/LFM2.5-2.6B-MLX`
            // keeps every model in a quant subfolder and has no loadable file at
            // its root, so the whole-repo path used to create an empty
            // `models/LiquidAI/LFM2.5-2.6B-MLX/` and report "Complete" — a
            // finished download with nothing in it. The directory is created
            // AFTER this, so a refusal leaves no folder behind either.
            guard !neededFiles.isEmpty else {
                throw NSError(domain: "MLXServe.Download", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "No downloadable files in \(repoId). This repo may keep its models in subfolders — pick a quantization from the model's menu.",
                ])
            }
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)

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
                let destPath = (destDir as NSString).appendingPathComponent(selection.localPath(forRemote: filePath))
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
                downloads[repoId]?.bytesPerSecond = 0
                downloads[repoId]?.statusText = "\(filePath) (\(sizeStr))"

                let fileURL = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filePath)")!
                let maxRetries = 20

                for attempt in 0..<maxRetries {
                    try Task.checkCancellation()

                    // Bytes already banked for this file — the chunk sidecar's
                    // total after an interrupted multi-connection run, else the
                    // plain `.partial` size.
                    let existingBytes = ChunkedFileDownloader.resumableBytes(partialPath: partialPath, fileSize: fileSize)
                    if existingBytes > 0 {
                        downloads[repoId]?.statusText = "Resuming \(filePath) from \(formatBytes(existingBytes))..."
                        // Bank the resumed bytes now: the first transfer
                        // callback would count them anyway, and until it lands
                        // the bar should already reflect what is on disk.
                        downloads[repoId]?.progress = totalSize > 0
                            ? Double(downloadedSize + existingBytes) / Double(totalSize) : 0
                    }

                    do {
                        try await transferFile(
                            url: fileURL,
                            partialPath: partialPath,
                            repoId: repoId,
                            fileSize: fileSize,
                            baseDownloaded: downloadedSize,
                            totalSize: totalSize
                        )

                        try commitPartial(partialPath, to: destPath)
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
            }

            downloads[repoId] = DownloadState(progress: 1.0, status: .completed, statusText: "Complete",
                                               fileIndex: neededFiles.count, fileCount: neededFiles.count)
        } catch {
            // User-cancelled? Skip the .failed row + alert — `start()`'s
            // wrapper will drop the entry and remove partials.
            if Task.isCancelled { return }
            let message = error.localizedDescription
            downloads[repoId] = DownloadState(status: .failed, error: message)
            if alertOnFailure, !(error is CancellationError) {
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
        guard let (data, response) = try? await DownloadSession.shared.data(for: Self.hfApiRequest(url)),
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
            await self?.downloadCompanionDrafterIfNeeded(for: repoId)
            self?.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    // MARK: - Companion drafter
    //
    // The Gemma 4 assistant drafter is a DEPENDENCY of the model it pairs with,
    // not something to shop for: it only ever works alongside one Gemma 4 size,
    // and picking it yourself means knowing that. It used to have its own Model
    // Browser destination, which mostly generated the question "which of these
    // is mine?". Now it rides along with its target, the same way a ds4 GGUF
    // quant pulls its MTP head (`resolveGgufDownloadFiles`).

    /// The drafter repo that pairs with `repoId`, or nil when there isn't one.
    ///
    /// Dense Gemma 4 only. The MoE target (26B-A4B) is excluded on purpose —
    /// the drafter REGRESSES decode there (verify pays expert routing, so the
    /// server defaults it off on MoE targets), and fetching a checkpoint we
    /// then refuse to use is worse than not having it. GGUF Gemma is excluded
    /// too: it runs on llama.cpp, which has no drafter path at all.
    nonisolated static func companionDrafterRepo(forRepoId repoId: String) -> String? {
        let base = (repoId as NSString).lastPathComponent.lowercased()
        // Muse-Glimmer pairs with its DFlash assistant (one published size).
        if base.contains("muse-glimmer"), !base.contains("assistant"), !base.contains("gguf") {
            return "meta-models/Muse-Glimmer-30B-assistant"
        }
        guard base.contains("gemma-4") || base.contains("gemma4") else { return nil }
        // A drafter must not pull itself — that download is an infinite regress.
        guard !base.contains("assistant"), !base.contains("gguf") else { return nil }
        // One parser for "which Gemma size is this?" — `gemmaVariantFor` is the
        // same one the pairing and auto-sync paths use, so a new size can't be
        // taught to one of them and not the other.
        guard let variant = gemmaVariantFor(modelPath: base, isMoE: false), variant != .moe26B else { return nil }
        return variant.drafterRepoId
    }

    /// Fetch `repoId`'s drafter after it lands, unless it's already here.
    /// Failures stay silent (the Downloads pane still shows the failed row):
    /// an alert naming a repo the user never asked for reads as a bug in the
    /// download they DID ask for.
    private func downloadCompanionDrafterIfNeeded(for repoId: String) async {
        guard !Task.isCancelled,
              downloads[repoId]?.status == .completed,
              let drafter = Self.companionDrafterRepo(forRepoId: repoId),
              !isReady(drafter) else { return }
        await download(repoId: drafter, alertOnFailure: false)
    }

    // MARK: - Turbo LoRA (on demand)

    /// Fetch the H3 Turbo adapter into a pack that is already on disk.
    ///
    /// It ships inside the bundle now, so a fresh download brings it; this is
    /// for the installs that predate it, where re-downloading 69 GB to collect
    /// one 744 MB file is not an answer. Same repo the pack came from, so the
    /// destination is the pack's own directory and `download`'s size-matching
    /// skip leaves every other file alone.
    ///
    /// Idempotent while in flight: a second call (toggle, then Generate)
    /// attaches to the running task instead of starting a second transfer.
    /// `onFinish` runs on completion OR failure — the caller decides what a
    /// failure means.
    ///
    /// Failures DO alert here, unlike the companion drafter above: the user
    /// ticked a box asking for this, so silence just moves the discovery to
    /// Generate, minutes later, as the server's 400. That reasoning only holds
    /// once the mirrors actually carry the file — before then every tick
    /// alerted on a 404 nobody could fix.
    func startTurboLora(repoId: String, onFinish: @escaping @MainActor () -> Void = {}) {
        startPackFile(repoId: repoId, fileName: TurboLoraFetch.fileName, onFinish: onFinish)
    }

    /// Fetch ONE file of a pack that is already on disk, beside its weights:
    /// the Turbo adapter above, and the ACE-Step cover tokenizer
    /// (`fsq.safetensors`) for packs downloaded before cover mode — that one
    /// is TEMPORARY migration code (2026-08-22), to go once installs have
    /// re-downloaded. Same contract as `startTurboLora`.
    func startPackFile(repoId: String, fileName: String, onFinish: @escaping @MainActor () -> Void = {}) {
        if let running = activeTasks[repoId] {
            Task { @MainActor in
                _ = await running.value
                onFinish()
            }
            return
        }
        // The adapter belongs BESIDE the pack's weights, wherever those live —
        // a pack in a non-destination root must not grow a fragment dir in the
        // destination (it shadows the real pack and the server dies loading it).
        let packDir = existingModelDir(for: repoId)
        mediaBundleRepos.insert(repoId)
        turboLoraFetches.insert(repoId)
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.download(repoId: repoId,
                                selection: FileSelection(keepSafetensors: [fileName]),
                                alertOnFailure: true,
                                destDirOverride: packDir)
            self.finalizeCancelledPackFile(repoId: repoId, fileName: fileName, packDir: packDir)
            self.turboLoraFetches.remove(repoId)
            self.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    /// Repo ids whose `activeTasks` entry is a Turbo-adapter fetch, not a full
    /// pack download — `cancelTurboLora` must never cancel the latter.
    private var turboLoraFetches: Set<String> = []

    /// Stop an in-flight Turbo-adapter fetch (the toggle's off-flip). With no
    /// adapter fetch running this does NOTHING — the generic `cancel(_:)`
    /// no-task fallback wipes the repo's whole download dir, which here is a
    /// live pack.
    func cancelTurboLora(repoId: String) {
        guard turboLoraFetches.contains(repoId) else { return }
        activeTasks[repoId]?.cancel()
    }

    /// Cancel cleanup for a single-file pack fetch: drop the ONE file's
    /// partials, never the directory — the destination is the pack itself.
    private func finalizeCancelledPackFile(repoId: String, fileName: String, packDir: String?) {
        guard Task.isCancelled else { return }
        downloads.removeValue(forKey: repoId)
        let dir = packDir ?? newLayoutDir(for: repoId)
        let base = (dir as NSString).appendingPathComponent(fileName)
        try? FileManager.default.removeItem(atPath: base + ".partial")
        try? FileManager.default.removeItem(atPath: base + ".partial.parts")
    }

    // MARK: - Media bundles
    //
    // A media model + its dependencies, downloaded as a unit (LTX → LTX +
    // Gemma-3-12B; FLUX/TTS → just the primary). Each component pulls ONLY the
    // files the engine reads (`FileSelection`). Tracked under the bundle id so
    // the gen pane can show aggregate progress / cancel.

    /// Repo ids whose transfers arrived as components of a media BUNDLE (or a
    /// Turbo-adapter fetch). Surfaces about the CHAT model filter on this —
    /// the model pill's progress hairline must not render a 30 GB video pack
    /// as the chat model arriving (`ChatModelPill.chatDownload`).
    private(set) var mediaBundleRepos: Set<String> = []

    /// Download a bundle's components sequentially (skipping any already on
    /// disk). `onFinish` runs once after the last component settles. Stops the
    /// bundle if a component fails.
    func startBundle(_ bundle: MediaBundle, onFinish: @escaping @MainActor () -> Void) {
        for comp in bundle.components { mediaBundleRepos.insert(comp.repo) }
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
        Self.componentReady(comp, roots: readRoots)
    }

    /// Multi-root form: ready in ANY owned root — a pack downloaded before the
    /// destination moved must not read as absent and get offered again.
    nonisolated static func componentReady(_ comp: MediaComponent, roots: [String]) -> Bool {
        roots.contains { componentReady(comp, modelsRoot: $0) }
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
        guard let (data, response) = try? await DownloadSession.shared.data(for: Self.hfApiRequest(url)),
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

    // MARK: - Multi-variant MLX repos

    /// MLX-variant analogue of `startGguf(repoId:quant:)`: fetch ONE quant
    /// subfolder of a shelf repo into its own model dir.
    ///
    /// Progress is tracked under the SOURCE repoId (that's the row the user
    /// clicked, and the menu reads it), while the bytes land in
    /// `<org>/<repo>-<folder>`. Cancellation is scoped to that one variant —
    /// the generic whole-folder wipe would be aimed at the parent repo, which
    /// holds nothing, leaving the interrupted variant behind.
    func startMlxVariant(repoId: String, variant: MlxVariant, onFinish: @escaping @MainActor () -> Void) {
        activeTasks[repoId]?.cancel()
        let dest = MlxVariantScan.localRepoId(repoId: repoId, folder: variant.folder)
        activeVariantDest[repoId] = dest
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.download(repoId: repoId, selection: .mlxVariant(variant.folder), destRepoId: dest)
            self.finalizeIfCancelledVariant(repoId: repoId, dest: dest)
            self.activeVariantDest.removeValue(forKey: repoId)
            self.activeTasks.removeValue(forKey: repoId)
            onFinish()
        }
        activeTasks[repoId] = task
    }

    /// Post-await cleanup for `startMlxVariant` — wipes the VARIANT's dir, not
    /// the repo's, so a cancelled 8-bit never takes a finished 4-bit with it.
    private func finalizeIfCancelledVariant(repoId: String, dest: String) {
        guard Task.isCancelled else { return }
        downloads.removeValue(forKey: repoId)
        Self.removeModelFiles(at: newLayoutDir(for: dest), roots: [modelsDir])
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
        // Destination-scoped: a cancel cleans up what THIS transfer wrote
        // (transfers only ever write into `modelsDir`), never a same-named
        // quant sitting in the built-in root.
        let dir = Self.existingModelDir(rootDir: modelsDir, repoId: repoId) ?? newLayoutDir(for: repoId)
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
            if let dest = activeVariantDest[repoId] {
                // Same scoping as GGUF: the repo is a shelf, so wipe the one
                // variant's dir rather than anything under the repo's name.
                Self.removeModelFiles(at: newLayoutDir(for: dest), roots: [modelsDir])
                activeVariantDest.removeValue(forKey: repoId)
            } else if let shards = activeGgufShards[repoId], let primary = shards.first {
                // Destination-scoped, like `finalizeIfCancelledGguf`.
                let dir = Self.existingModelDir(rootDir: modelsDir, repoId: repoId) ?? newLayoutDir(for: repoId)
                removeGgufQuant(at: (dir as NSString).appendingPathComponent(primary))
                activeGgufShards.removeValue(forKey: repoId)
            } else if Self.downloadedGgufPaths(rootDir: modelsDir, repoId: repoId).isEmpty {
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
    ///
    /// A quant that never COMMITTED exists only as `<path>.partial` (plus its
    /// chunk sidecar) — which is precisely the state a Cancel leaves behind — so
    /// the partial counts as the quant here. Keying existence on the committed
    /// name alone deleted nothing on cancel: tens of GB of an 86 GB transfer
    /// stayed on disk with no UI that could reach them (the Delete submenu lists
    /// COMPLETE quants only), and the row came back offering "Resume".
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

        let partial = path + ".partial"
        guard fm.fileExists(atPath: path) || fm.fileExists(atPath: partial) else { return false }
        try? fm.removeItem(atPath: path)
        try? fm.removeItem(atPath: partial)
        // The chunk sidecar travels with the `.partial` — leaving it behind
        // would have the next download resume against a plan for bytes that
        // are no longer there.
        ChunkedResumeState.remove(forPartial: partial)

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
                var headReq = Self.hfApiRequest(fileURL)
                headReq.httpMethod = "HEAD"
                let (_, headResp) = try await DownloadSession.shared.data(for: headReq)
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

                    let existingBytes = ChunkedFileDownloader.resumableBytes(partialPath: partialPath, fileSize: fileSize)
                    if existingBytes > 0 {
                        downloads[repoId]?.statusText = "Resuming \(shardName) from \(formatBytes(existingBytes))..."
                        downloads[repoId]?.progress = totalSize > 0
                            ? Double(baseDownloaded + existingBytes) / Double(totalSize) : 0
                    }

                    do {
                        try await transferFile(
                            url: fileURL,
                            partialPath: partialPath,
                            repoId: repoId,
                            fileSize: fileSize,
                            baseDownloaded: baseDownloaded,
                            totalSize: totalSize
                        )
                        try commitPartial(partialPath, to: destPath)
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

    /// Fetch one file into its `.partial`, splitting it across
    /// `DownloadChunking.configuredConnections()` ranged connections when it's
    /// big enough to be worth it (`ChunkedFileDownloader` decides, and falls
    /// back to a single stream on its own when the origin won't range). Bytes
    /// land at their own offsets as they arrive, and per-chunk progress is
    /// banked beside the partial, so an interrupted transfer resumes exactly the
    /// hole it left.
    private func transferFile(
        url: URL,
        partialPath: String,
        repoId: String,
        fileSize: Int64,
        baseDownloaded: Int64,
        totalSize: Int64
    ) async throws {
        let downloader = ChunkedFileDownloader(
            url: url,
            partialPath: partialPath,
            fileSize: fileSize,
            session: DownloadSession.shared,
            headers: Self.hfHeaders(),
            connections: DownloadChunking.configuredConnections()
        )
        downloader.onProgress = { [weak self] fileBytesTotal, speed in
            // `fileBytesTotal` includes bytes resumed from a previous run, so
            // this is the whole transfer's position, not this session's.
            let overallDownloaded = baseDownloaded + fileBytesTotal
            Task { @MainActor [weak self] in
                self?.downloads[repoId]?.bytesPerSecond = speed
                self?.downloads[repoId]?.progress = totalSize > 0 ? Double(overallDownloaded) / Double(totalSize) : 0
            }
        }
        try await downloader.run()
    }

    /// Move a finished `.partial` onto its final name and drop the chunk
    /// sidecar — the two are written together and must be retired together, or
    /// the next download of the same file resumes against a plan for bytes that
    /// have already moved.
    private func commitPartial(_ partialPath: String, to destPath: String) throws {
        let fm = FileManager.default
        try? fm.removeItem(atPath: destPath)
        try fm.moveItem(atPath: partialPath, toPath: destPath)
        ChunkedResumeState.remove(forPartial: partialPath)
    }

    /// Check whether a model has .partial files from an interrupted download.
    func hasPartialDownload(_ repoId: String) -> Bool {
        // Look in the new layout first (where in-progress downloads live), then
        // legacy as a fallback. Destination-scoped: partials only ever live
        // where transfers write.
        let candidates = [newLayoutDir(for: repoId),
                          Self.existingModelDir(rootDir: modelsDir, repoId: repoId)].compactMap { $0 }
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
                    quantFile: primaryBase,
                    quantLabel: quant.label
                )
            }
        }

        let configPath = (resolved as NSString).appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configPath) else { return [] }

        // A defect does NOT drop the directory. Dropping it is how two junk
        // folders stayed invisible in the app while the server registered them
        // and would have died loading either one — you cannot delete what you
        // cannot see. It is listed, unpickable, and deletable instead.
        let defect = weightDefect(inDir: resolved, entries: entries)

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
            activeExperts: meta.activeExperts,
            defect: defect
        )]
    }

    /// A `.partial` beside a moving progress bar is not an interrupted download,
    /// so a dir that is the destination of a live transfer loses that defect.
    nonisolated static func clearingInFlightDefects(_ models: [LocalModel], activeDirs: Set<String>) -> [LocalModel] {
        guard !activeDirs.isEmpty else { return models }
        return models.map { m in
            guard m.defect == .interruptedDownload,
                  activeDirs.contains((m.path as NSString).standardizingPath) else { return m }
            var fixed = m
            fixed.defect = nil
            return fixed
        }
    }

    /// Smallest total weight payload that could be a real checkpoint.
    ///
    /// This is a "nothing loadable is this small" floor, NOT a guess about
    /// model size: the smallest quantized checkpoint anyone serves is orders of
    /// magnitude past a mebibyte. It exists because a file-EXISTS check let a
    /// 48 KB stub `jangtq_runtime.safetensors` present its folder as a real,
    /// selectable model. Where the checkpoint declares its own parts we do not
    /// use the floor at all — a shard index is exact.
    nonisolated static let minimumWeightBytes: UInt64 = 1024 * 1024

    /// Classify a safetensors directory: nil when it holds a loadable
    /// checkpoint, else why it does not.
    ///
    /// Order matters. An interrupted download is reported ahead of thin
    /// weights because it EXPLAINS them — the transfer stopped, so "re-download
    /// or delete" is the honest advice, where "this folder is junk" is not.
    nonisolated static func weightDefect(inDir dir: String, entries: [String]) -> ModelDefect? {
        if entries.contains(where: { $0.hasSuffix(".partial") || $0.hasSuffix(".incomplete") }) {
            return .interruptedDownload
        }

        // Exact path: the index names every shard the checkpoint needs.
        let indexPath = (dir as NSString).appendingPathComponent("model.safetensors.index.json")
        if let data = FileManager.default.contents(atPath: indexPath),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = obj["weight_map"] as? [String: String] {
            let declared = Set(map.values)
            if !declared.isEmpty {
                let missing = declared.contains {
                    !FileManager.default.fileExists(
                        atPath: (dir as NSString).appendingPathComponent($0))
                }
                return missing ? .missingShards : nil
            }
        }

        // Inexact path: no index, so all we can say is whether the bytes on
        // disk could possibly be a checkpoint. Media packs (FLUX.2 klein's
        // mflux layout) keep every weight one level down in `transformer/`,
        // `vae/`, … with nothing at the root, so the sum reads one level deep.
        let fm = FileManager.default
        func safetensorsBytes(in d: String, names: [String]) -> UInt64 {
            names.filter { $0.hasSuffix(".safetensors") }
                .reduce(UInt64(0)) { $0 + resolvedFileSize((d as NSString).appendingPathComponent($1)) }
        }
        var bytes = safetensorsBytes(in: dir, names: entries)
        for e in entries where bytes < minimumWeightBytes {
            let sub = (dir as NSString).appendingPathComponent(e)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: sub, isDirectory: &isDir), isDir.boolValue,
                  let names = try? fm.contentsOfDirectory(atPath: sub) else { continue }
            bytes += safetensorsBytes(in: sub, names: names)
        }
        return bytes >= minimumWeightBytes ? nil : .missingWeights
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
        // Vision: a NON-EMPTY `vision_config` block on a non-`_text` arch. Both
        // guards earn their place — `_text` skips text-only quantized
        // checkpoints with a vestigial block, and the emptiness check catches
        // the ones that keep the key but not the arch tag (mlx-community's
        // text-only LFM2.5 packs declare `Lfm2ForCausalLM` and ship
        // `"vision_config": {}`, which badged them vision-capable while the
        // server served them text-only). A block with no geometry is not a tower.
        let visionBlock = cfg["vision_config"] as? [String: Any]
        meta.hasVision = !(visionBlock?.isEmpty ?? true) && !meta.modelType.hasSuffix("_text")
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

        // The owned roots — download destination + `~/.mlx-serve/models` after
        // the destination moves — so the pre-move library stays in the picker.
        out.append(contentsOf: Self.mlxServeModels(inRoots: ownedRoots))

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

        // Other local-inference tools' canonical folders, auto-detected. Both
        // layouts they use are already read by `dualLayoutModels` — MTPLX
        // writes flat `Org--Name` dirs, Osaurus writes `org/repo` — so this
        // enumeration exists only because the picker walks folders separately
        // from `ModelRoots.scanRoots`, and a root added to one and not the
        // other is served but unselectable. Read-only: another tool's tree.
        for tool in ToolModelRoots.detected(lmStudioRoot: lmStudioRoot).orderedWithSource
        where tool.path != lmStudioRoot {
            out.append(contentsOf: Self.dualLayoutModels(
                atRoot: tool.path, idPrefix: "tool:", source: tool.source))
        }

        // User-configured custom root — same dual-layout scan as the owned
        // roots. resolvedCustomRoot() handles tilde expansion, existence check,
        // and dedup against the default roots so a user pointing it at
        // `~/.mlx-serve/models` doesn't produce duplicate picker entries.
        if let root = resolvedCustomRoot() {
            out.append(contentsOf: Self.dualLayoutModels(atRoot: root, idPrefix: "custom:", source: .custom))
        }

        let inFlight = Set(downloads.filter { $0.value.status == .downloading }
            .map { (newLayoutDir(for: $0.key) as NSString).standardizingPath })
        return Self.clearingInFlightDefects(out, activeDirs: inFlight)
            // By label, not name: sibling quants of one repo share a name, and a
            // name-only sort leaves their relative order at the mercy of the
            // filesystem.
            .sorted { $0.displayLabel.localizedCaseInsensitiveCompare($1.displayLabel) == .orderedAscending }
    }

    /// One root's models in the dual layout every owned folder uses.
    /// New: `<root>/<author>/<name>/` (matches LM Studio). Legacy: flat
    /// `<root>/<name>/` — kept working for users who had models predating the
    /// migration that the auto-migrator couldn't classify. Whether an entry is
    /// itself a model dir (legacy flat) or an author dir (new layout) is
    /// decided by what `makeLocalModels` finds in it — NOT by config.json,
    /// which a GGUF-only folder never has. `nonisolated` + static so it's
    /// testable against temp dirs.
    nonisolated static func dualLayoutModels(atRoot root: String, idPrefix: String, source: LocalModelSource) -> [LocalModel] {
        var out: [LocalModel] = []
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: root) else { return out }
        for entry in entries where !entry.hasPrefix(".") {
            let entryPath = (root as NSString).appendingPathComponent(entry)
            let direct = makeLocalModels(atDir: entryPath, displayName: entry, idKey: idPrefix + entry, source: source)
            if !direct.isEmpty {
                out.append(contentsOf: direct)
            } else if let children = try? FileManager.default.contentsOfDirectory(atPath: entryPath) {
                for child in children where !child.hasPrefix(".") {
                    let childPath = (entryPath as NSString).appendingPathComponent(child)
                    let display = "\(entry)/\(child)"
                    out.append(contentsOf: makeLocalModels(atDir: childPath, displayName: display, idKey: idPrefix + display, source: source))
                }
            }
        }
        return out
    }

    /// Every owned root's models, the FIRST root winning a repeated id — the
    /// same first-wins rule the server applies to repeated `--model-dir`
    /// flags, with the download destination first in both lists. This is what
    /// keeps the `~/.mlx-serve/models` library in the picker after the
    /// destination moves: the server kept scanning it while the picker read
    /// only `modelsDir` and hid it.
    nonisolated static func mlxServeModels(inRoots roots: [String]) -> [LocalModel] {
        var out: [LocalModel] = []
        var seen = Set<String>()
        for root in roots {
            for m in dualLayoutModels(atRoot: root, idPrefix: "", source: .mlxServe) where seen.insert(m.id).inserted {
                out.append(m)
            }
        }
        return out
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
        // DFlash block-drafter for Muse-Glimmer-30B (server auto-detects the
        // kind from the sidecar's config contract; same --drafter flag).
        "muse_glimmer_assistant",
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
        Self.discoverDrafters(in: readRoots)
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
    /// `unlocked` is the row's explicit second click on a model outside our
    /// own tree. It is a parameter rather than a mutation of `isDeletable`
    /// because the DEFAULT must stay refusal: every other caller keeps the
    /// old behaviour by not passing it.
    func deleteModel(_ model: LocalModel, unlocked: Bool = false) {
        // Only ~/.mlx-serve/models is ours to delete. LM Studio, the Hugging Face
        // hub cache, and custom-root models are owned by another tool or the user
        // (deleting an HF snapshot orphans shared blobs and dangles refs/main; the
        // others simply aren't ours). The UI hides the trash for them; this is the
        // defensive backstop, and the roots are scoped to modelsDir so a stray call
        // can never prune into an external tree.
        guard model.isDeletable || unlocked else { return }
        // A broken folder is deletable wherever it sits, so the ROOT LIST it is
        // bounded by has to widen with it: `roots` is what `removeModelFiles`
        // refuses to remove and stops pruning at, and a foreign root missing
        // from that set is a root this call would happily delete.
        let roots = (model.defect != nil || unlocked) ? readRoots : ownedRoots
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
        // Delete every owned copy — both layouts, both roots — so we don't
        // orphan a legacy copy after a partial migration, or a built-in-root
        // copy that would keep the model listed after "Delete". Empty author
        // dirs are also pruned.
        for root in ownedRoots {
            if let existing = Self.existingModelDir(rootDir: root, repoId: repoId) {
                try? fm.removeItem(atPath: existing)
            }
        }
        // If the new-layout target also exists separately (e.g. interrupted
        // download), remove it too.
        let newPath = newLayoutDir(for: repoId)
        if fm.fileExists(atPath: newPath) {
            try? fm.removeItem(atPath: newPath)
        }
        // Prune now-empty author dirs — in whichever owned root held the model.
        let parts = repoId.split(separator: "/").map(String.init)
        if parts.count >= 2 {
            for root in ownedRoots {
                let authorDir = (root as NSString).appendingPathComponent(parts[parts.count - 2])
                if let kids = try? fm.contentsOfDirectory(atPath: authorDir),
                   kids.filter({ !$0.hasPrefix(".") }).isEmpty {
                    try? fm.removeItem(atPath: authorDir)
                }
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

    // MARK: - Hugging Face auth

    /// The Hugging Face token to send, or nil. `HF_TOKEN` mirrors the Zig CLI
    /// (`cli.zig`'s curl prefix) and covers a terminal launch; the token file
    /// `huggingface-cli login` writes is the one that actually works in the app,
    /// because a bundle launched from Finder has NO shell environment. Buys
    /// gated repos and a higher API rate limit — not speed.
    nonisolated static func hfToken(environment: [String: String] = LoginShellEnv.huggingFaceEnvironment(),
                                    home: String = NSHomeDirectory()) -> String? {
        if let raw = environment["HF_TOKEN"]?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            return raw
        }
        var candidates: [String] = []
        if let hfHome = environment["HF_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines), !hfHome.isEmpty {
            candidates.append(((hfHome as NSString).expandingTildeInPath as NSString).appendingPathComponent("token"))
        }
        if let xdg = environment["XDG_CACHE_HOME"]?.trimmingCharacters(in: .whitespacesAndNewlines), !xdg.isEmpty {
            candidates.append(((xdg as NSString).expandingTildeInPath as NSString)
                .appendingPathComponent("huggingface/token"))
        }
        candidates.append(((home as NSString).appendingPathComponent(".cache/huggingface") as NSString)
            .appendingPathComponent("token"))
        for path in candidates {
            guard let data = FileManager.default.contents(atPath: path),
                  let raw = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty else { continue }
            return raw
        }
        return nil
    }

    nonisolated static func hfHeaders(environment: [String: String] = LoginShellEnv.huggingFaceEnvironment(),
                                      home: String = NSHomeDirectory()) -> [String: String] {
        guard let token = hfToken(environment: environment, home: home) else { return [:] }
        return ["Authorization": "Bearer \(token)"]
    }

    /// A GET against the HF API on the shared session, carrying the token when
    /// we have one. Every listing call goes through here so a gated repo fails
    /// the same way everywhere.
    nonisolated static func hfApiRequest(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        for (key, value) in hfHeaders() { request.setValue(value, forHTTPHeaderField: key) }
        return request
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes > 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes > 1_000_000 { return String(format: "%.0f MB", Double(bytes) / 1e6) }
        return "\(bytes) B"
    }
}
