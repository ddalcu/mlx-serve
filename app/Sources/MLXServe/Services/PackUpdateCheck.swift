import Foundation

/// New files in a downloaded pack: our `ddalcu/` packs and the gem repos grow
/// after release (a `drafter/` added to the 27B packs), and `download` already
/// skips every size-matching file, so the fix is one click once we can see it.
struct PackUpdate: Equatable {
    var files: Int
    var bytes: Int64
    /// Of `files`, those already on disk at another size: an update overwrites them.
    var replaces: Int = 0
}

/// The Hugging Face repo a download came from, written beside the files
/// (`.mlx-serve-source.json`) so a later update check need not guess it from the folder name.
struct PackSource: Codable, Equatable {
    var repo: String
    var subfolder: String? = nil

    static let fileName = ".mlx-serve-source.json"

    static func read(dir: String) -> PackSource? {
        guard let data = FileManager.default.contents(atPath: (dir as NSString).appendingPathComponent(fileName)) else { return nil }
        return try? JSONDecoder().decode(PackSource.self, from: data)
    }

    func write(dir: String) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        try? data.write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(Self.fileName)))
    }
}

/// Which of a repo's files a local model is.
enum UpdateSelection: Equatable {
    case chat(drafter: Bool)
    case variant(String)
    /// A GGUF quant, by its primary shard's basename.
    case gguf(String)
    case media(FileSelection)
}

enum UpdateTarget: Equatable {
    case skipped(String)
    case check(repo: String, selection: UpdateSelection)
}

enum UpdateResult: Equatable {
    case upToDate
    /// The repo is not on Hugging Face (or holds none of the model's files): a local conversion or copy.
    case localCopy
    case update(PackUpdate)
    case skipped(String)
    case failed
}

/// One model's manual check, with what applying it needs.
struct UpdateCheck: Equatable {
    var result: UpdateResult
    var repo: String = ""
    var dir: String = ""
    var selection: UpdateSelection = .chat(drafter: false)
}

enum PackUpdateCheck {
    static let interval: TimeInterval = 24 * 3600

    /// What a pack's download would fetch: the whole pack, less its `drafter/`
    /// when the socket is switched off.
    static func wanted(_ entries: [[String: Any]], withDrafter: Bool) -> [(String, Int64)] {
        DownloadManager.selectNeededFiles(from: entries, selection: withDrafter ? .chatDefault : .chatWithoutDrafter)
    }

    /// An HF tree listing as path -> size.
    static func sizes(_ entries: [[String: Any]]) -> [String: Int64] {
        Dictionary(entries.compactMap { e -> (String, Int64)? in
            guard let p = e["path"] as? String else { return nil }
            return (p, e["size"] as? Int64 ?? (e["size"] as? Int).map { Int64($0) } ?? 0)
        }, uniquingKeysWith: { a, _ in a })
    }

    /// Wanted files missing from `dir` or of another size; nil when none.
    static func pending(_ wanted: [(String, Int64)], inDir dir: String) -> PackUpdate? {
        let fm = FileManager.default
        var update = PackUpdate(files: 0, bytes: 0)
        for (path, size) in wanted {
            let local = (try? fm.attributesOfItem(atPath: (dir as NSString).appendingPathComponent(path))[.size] as? Int64) ?? nil
            guard local != size else { continue }
            update.files += 1
            update.bytes += size
            if local != nil { update.replaces += 1 }
        }
        return update.files > 0 ? update : nil
    }

    /// What to check for a local model. Another tool's folder is never written
    /// into; without a source marker the repo is guessed from the folder name.
    static func target(for m: LocalModel, marker: PackSource?, drafterOff: Bool) -> UpdateTarget {
        switch m.source {
        case .lmStudio: return .skipped("managed by LM Studio")
        case .huggingFace: return .skipped("managed by the Hugging Face cache")
        case .mtplx: return .skipped("managed by MTPLX")
        case .osaurus: return .skipped("managed by Osaurus")
        case .mlxServe, .custom: break
        }
        guard let repo = marker?.repo ?? ModelCard.repoId(localName: m.name) else { return .skipped("no Hugging Face name") }
        if let quant = m.quantFile { return .check(repo: repo, selection: .gguf(quant)) }
        if let sub = marker?.subfolder { return .check(repo: repo, selection: .variant(sub)) }
        if isMediaModelType(m.modelType) || m.modelType == "laya",
           let bundle = CustomMediaModels.bundle(arch: m.modelType, repoId: repo) {
            let comp = bundle.components.first { $0.repo == repo } ?? bundle.components[0]
            return .check(repo: repo, selection: .media(comp.selection))
        }
        return .check(repo: repo, selection: .chat(drafter: !drafterOff))
    }

    /// The listing's files a selection covers, as local paths with their sizes.
    static func wanted(_ entries: [[String: Any]], selection: UpdateSelection) -> [(String, Int64)] {
        switch selection {
        case .chat(let drafter):
            return wanted(entries, withDrafter: drafter)
        case .variant(let sub):
            let sel = FileSelection.mlxVariant(sub)
            return DownloadManager.selectNeededFiles(from: entries, selection: sel).map { (sel.localPath(forRemote: $0.0), $0.1) }
        case .media(let sel):
            return DownloadManager.selectNeededFiles(from: entries, selection: sel)
        case .gguf(let primary):
            let all = sizes(entries)
            guard let quant = GgufQuant.groupQuants(Array(all.keys))
                .first(where: { ($0.filename as NSString).lastPathComponent == primary }) else { return [] }
            return quant.allFiles.map { ($0, all[$0] ?? 0) }
        }
    }

    /// The models a manual check found an update for, in the order they are listed.
    static func available(_ checks: [String: UpdateCheck], models: [LocalModel]) -> [(model: LocalModel, check: UpdateCheck)] {
        models.compactMap { m in
            guard let c = checks[m.id], case .update = c.result else { return nil }
            return (m, c)
        }.sorted { $0.model.name < $1.model.name }
    }

    /// The alert body after a manual check: one line per update, and a warning when any replaces files.
    static func sweepMessage(_ items: [(model: LocalModel, check: UpdateCheck)]) -> String {
        var replacing = 0
        let lines = items.map { item -> String in
            guard case .update(let u) = item.check.result else { return item.model.name }
            replacing += u.replaces > 0 ? 1 : 0
            return "\(item.model.name): \(u.files) \(u.files == 1 ? "file" : "files"), \(SystemMemoryInfo.preciseGB(Double(u.bytes) / 1e9))"
        }
        let note = replacing == 0 ? "" : "\n\n\(replacing == 1 ? "One update replaces" : "\(replacing) updates replace") files you already have."
        return lines.joined(separator: "\n") + note
    }

    /// "2 updates, 4 local copies, 3 skipped" for the My Models footer.
    static func summary(_ results: [UpdateResult]) -> String {
        var counts = [0, 0, 0, 0]
        for r in results {
            switch r {
            case .update: counts[0] += 1
            case .localCopy: counts[1] += 1
            case .skipped: counts[2] += 1
            case .failed: counts[3] += 1
            case .upToDate: break
            }
        }
        let words = [("update", "updates"), ("local copy", "local copies"), ("skipped", "skipped"), ("failed", "failed")]
        let parts = zip(counts, words).filter { $0.0 > 0 }.map { "\($0.0) \($0.0 == 1 ? $0.1.0 : $0.1.1)" }
        return parts.isEmpty ? "All up to date" : parts.joined(separator: ", ")
    }

    /// A listing's HTTP status and the wanted files, judged against `dir`.
    static func classify(status: Int, wanted: [(String, Int64)], inDir dir: String) -> UpdateResult {
        if status == 401 || status == 404 { return .localCopy }
        guard status == 200 else { return .failed }
        guard !wanted.isEmpty else { return .localCopy }
        return pending(wanted, inDir: dir).map { .update($0) } ?? .upToDate
    }

    /// Our packs and the separate-repo gems are worth checking; anything else is someone else's.
    static func isChecked(repoId: String) -> Bool {
        repoId.hasPrefix("ddalcu/") || repoId == DrafterGems.qwen38DFlash2Repo || repoId == DrafterGems.museAssistantRepo
            || GemmaVariant.allCases.contains { $0.drafterRepoId == repoId }
    }

    static func isDue(lastChecked: Date?, now: Date = Date()) -> Bool {
        guard let lastChecked else { return true }
        return now.timeIntervalSince(lastChecked) >= interval
    }
}

extension DownloadManager {
    /// The pack's HF listing as path → size (nil on any failure).
    func fetchListing(repoId: String) async -> [[String: Any]]? {
        let (status, entries) = await fetchListingStatus(repoId: repoId)
        return status == 200 ? entries : nil
    }

    /// The listing with its HTTP status (0 when the request never answered).
    func fetchListingStatus(repoId: String) async -> (Int, [[String: Any]]?) {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoId)/tree/main?recursive=true"),
              let (data, response) = try? await DownloadSession.shared.data(for: Self.hfApiRequest(url)) else { return (0, nil) }
        return ((response as? HTTPURLResponse)?.statusCode ?? 0, try? JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    /// The manual "Check for Updates": every model, one at a time, no daily gate.
    func checkAllForUpdates(models: [LocalModel]) async {
        guard updateSweep == nil else { return }
        updateChecks = [:]
        let settings = ModelSettingsFile.load()
        for (i, m) in models.enumerated() {
            updateSweep = (i + 1, models.count)
            updateChecks[m.id] = await checkForUpdate(m, drafterOff: settings.override(for: m.path)?.drafter == "off")
        }
        updateSweep = nil
    }

    private func checkForUpdate(_ m: LocalModel, drafterOff: Bool) async -> UpdateCheck {
        let base = m.quantFile == nil ? m.path : (m.path as NSString).deletingLastPathComponent
        let marker = PackSource.read(dir: base) ?? PackSource.read(dir: (base as NSString).deletingLastPathComponent)
        let repo: String
        var selection: UpdateSelection
        switch PackUpdateCheck.target(for: m, marker: marker, drafterOff: drafterOff) {
        case .skipped(let why): return UpdateCheck(result: .skipped(why))
        case .check(let r, let sel): (repo, selection) = (r, sel)
        }
        let (status, entries) = await fetchListingStatus(repoId: repo)
        let wanted = PackUpdateCheck.wanted(entries ?? [], selection: selection)
        // Applying fetches `drafter/` as its own selection, which must not be empty.
        if case .chat(true) = selection, !wanted.contains(where: { $0.0.hasPrefix(DrafterGems.packFolder + "/") }) {
            selection = .chat(drafter: false)
        }
        var dir = m.path
        if case .gguf = selection, let rel = wanted.first?.0 {
            dir = m.path.hasSuffix("/" + rel) ? String(m.path.dropLast(rel.count + 1)) : base
        }
        return UpdateCheck(result: PackUpdateCheck.classify(status: status, wanted: wanted, inDir: dir),
                           repo: repo, dir: dir, selection: selection)
    }

    /// Refresh one pack's listing and pending update. `force` skips the daily gate.
    func checkPackUpdate(repoId: String, dir: String, force: Bool = false) async {
        guard Self.isCheckable(repoId: repoId, dir: dir) else { return }
        let key = "packUpdateChecked." + repoId
        let last = UserDefaults.standard.object(forKey: key) as? Date
        guard force || PackUpdateCheck.isDue(lastChecked: last) else { return }
        guard let entries = await fetchListing(repoId: repoId) else { return }
        UserDefaults.standard.set(Date(), forKey: key)
        packListings[repoId] = PackUpdateCheck.sizes(entries)
        let drafterOff = ModelSettingsFile.load().override(for: dir)?.drafter == "off"
        packUpdates[repoId] = PackUpdateCheck.pending(PackUpdateCheck.wanted(entries, withDrafter: !drafterOff), inDir: dir)
    }

    /// Daily sweep over every local chat pack and downloaded gem worth checking.
    func checkPackUpdates(models: [LocalModel]) async {
        guard !packSweepRunning else { return }
        packSweepRunning = true
        defer { packSweepRunning = false }
        for m in models where m.quantFile == nil && m.isChatPickable {
            await checkPackUpdate(repoId: m.name, dir: m.path)
        }
        for repo in Self.downloadedGemRepos() {
            await checkPackUpdate(repoId: repo, dir: Self.gemDir(repo: repo))
        }
    }

    private static func isCheckable(repoId: String, dir: String) -> Bool {
        PackUpdateCheck.isChecked(repoId: repoId) && FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("config.json"))
    }
}
