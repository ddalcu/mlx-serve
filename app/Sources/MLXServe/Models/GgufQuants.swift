import Foundation

/// One selectable quant in a GGUF repo.
///
/// A quant is a SHARD GROUP, not necessarily one file. Small quants are one
/// `.gguf` (`Qwen3.5-4B-Q4_K_M.gguf`); large ones (anything HF splits over
/// ~50 GB) ship as a subfolder of split shards
/// (`Hy3-IQ1_M/Hy3-IQ1_M-00001-of-00002.gguf`, `…-00002-of-00002.gguf`). The
/// server loads the `-00001` shard by explicit path and libllama auto-loads the
/// rest, so grouping is purely client-side.
struct GgufQuant: Identifiable, Equatable, Hashable {
    /// Repo-relative path of the PRIMARY shard — the `-00001-of-…` file for a
    /// split quant, or the whole file for a single-file quant. This is the path
    /// the server loads and the tray selects.
    let filename: String
    /// Short human label for menus (`Q4_K_M`, `IQ1_M`).
    let label: String
    /// Every repo-relative shard path, primary first. EMPTY for a single-file
    /// quant (`allFiles` then falls back to `[filename]`).
    var shards: [String] = []

    var id: String { filename }

    /// Every file this quant comprises — the shard list, or `[filename]` when
    /// it's a single file.
    var allFiles: [String] { shards.isEmpty ? [filename] : shards }

    /// True when every shard is present. A single-file quant is always complete;
    /// a split quant is complete only when the shard count matches the
    /// `-NNNNN-of-MMMMM` total encoded in the primary's name.
    var isComplete: Bool {
        guard let total = GgufQuant.shardCount(forName: filename) else { return true }
        return allFiles.count >= total
    }

    /// The `MMMMM` total from a `-NNNNN-of-MMMMM.gguf` split-shard name, or nil
    /// when the name isn't a split shard. Accepts a basename or a full path.
    static func shardCount(forName name: String) -> Int? {
        let base = (name as NSString).lastPathComponent
        guard let re = try? NSRegularExpression(pattern: "-[0-9]+-of-([0-9]+)\\.gguf$", options: [.caseInsensitive]),
              let m = re.firstMatch(in: base, range: NSRange(base.startIndex..., in: base)),
              let r = Range(m.range(at: 1), in: base) else { return nil }
        return Int(base[r])
    }

    /// Group a flat list of repo-relative `.gguf` paths into quants (shard
    /// groups). Sidecars (`mmproj-*`, `*tokenizer*`) are dropped. A split quant's
    /// shards fold into one entry (primary = the lowest-numbered shard, i.e.
    /// `-00001`); a non-split quant becomes a group of one with an empty
    /// `shards`. Result sorted by primary path so ordering is filesystem-stable.
    static func groupQuants(_ paths: [String]) -> [GgufQuant] {
        let supported = paths.filter { DownloadManager.isSupportedGguf($0) }
        var groups: [String: [String]] = [:]
        for p in supported {
            groups[shardGroupKey(p), default: []].append(p)
        }
        let quants = groups.values.compactMap { shardPaths -> GgufQuant? in
            let sorted = shardPaths.sorted()
            guard let primary = sorted.first else { return nil }
            // Sidecar primaries are already filtered above, but never let one
            // through as a quant even if a future filename slips the filter.
            guard DownloadManager.isSupportedGguf(primary) else { return nil }
            let split = shardCount(forName: primary) != nil
            return GgufQuant(
                filename: primary,
                label: DownloadManager.quantLabel(forFilename: primary),
                shards: split ? sorted : []
            )
        }
        .sorted { $0.filename < $1.filename }
        return disambiguateLabels(quants)
    }

    // MARK: - Label disambiguation
    //
    // A repo can ship several files that reduce to the SAME quant token
    // (`quantLabel`) while differing in tier, calibration, or variant —
    // antirez/deepseek-v4-gguf has four IQ2XXS files (Flash/Pro × imatrix/static).
    // When labels collide, append the distinguishing filename tokens so the
    // dropdown is unambiguous.

    /// Filename tokens that identify a model TIER — the most useful
    /// disambiguator (e.g. Flash 86 GB vs Pro 465 GB). Lowercased.
    private static let tierTokens: Set<String> = [
        "flash", "pro", "lite", "mini", "max", "air", "nano", "base", "plus", "turbo",
    ]
    /// Filename tokens that identify a meaningful VARIANT worth surfacing.
    private static let qualityTokens: Set<String> = [
        "imatrix", "fixed", "thinking", "reasoning", "distill", "distilled",
    ]

    private static func disambiguateLabels(_ quants: [GgufQuant]) -> [GgufQuant] {
        var indicesByLabel: [String: [Int]] = [:]
        for (i, q) in quants.enumerated() { indicesByLabel[q.label, default: []].append(i) }
        var out = quants
        for (_, idxs) in indicesByLabel where idxs.count > 1 {
            let tokenLists = idxs.map { tokens(of: (quants[$0].filename as NSString).lastPathComponent) }
            let common = commonTokens(tokenLists)
            for (k, i) in idxs.enumerated() {
                let extra = distinguishingTokens(tokenLists[k], common: common)
                guard !extra.isEmpty else { continue }   // the "base" file keeps the plain label
                out[i] = GgufQuant(
                    filename: quants[i].filename,
                    label: ([quants[i].label] + extra).joined(separator: " · "),
                    shards: quants[i].shards
                )
            }
        }
        return out
    }

    /// Split an extension-stripped basename into `-`/`_`/`.`/space tokens.
    static func tokens(of basename: String) -> [String] {
        (basename as NSString).deletingPathExtension
            .components(separatedBy: CharacterSet(charactersIn: "-_. "))
            .filter { !$0.isEmpty }
    }

    private static func commonTokens(_ lists: [[String]]) -> Set<String> {
        guard var acc = lists.first.map({ Set($0.map { $0.lowercased() }) }) else { return [] }
        for l in lists.dropFirst() { acc.formIntersection(l.map { $0.lowercased() }) }
        return acc
    }

    /// Up to two tokens that best distinguish this file from its collision
    /// siblings: tier first, then variant markers, else the first unique token.
    /// Original casing preserved; deduped in order.
    private static func distinguishingTokens(_ fileTokens: [String], common: Set<String>) -> [String] {
        var seen = Set<String>()
        let unique = fileTokens.filter { tok in
            let lower = tok.lowercased()
            guard !common.contains(lower) else { return false }
            return seen.insert(lower).inserted
        }
        guard !unique.isEmpty else { return [] }
        var picks: [String] = []
        if let tier = unique.first(where: { tierTokens.contains($0.lowercased()) }) { picks.append(tier) }
        for tok in unique where qualityTokens.contains(tok.lowercased()) && !picks.contains(tok) {
            picks.append(tok)
        }
        if picks.isEmpty, let first = unique.first { picks.append(first) }
        return Array(picks.prefix(2))
    }

    /// The key that folds a split quant's shards together: the path with its
    /// `-NNNNN-of-MMMMM.gguf` suffix stripped. A non-split path keys on itself,
    /// so each single-file quant is its own group.
    private static func shardGroupKey(_ path: String) -> String {
        if let r = path.range(of: "-[0-9]+-of-[0-9]+\\.gguf$", options: [.regularExpression, .caseInsensitive]) {
            return String(path[path.startIndex..<r.lowerBound])
        }
        return path
    }
}

/// The Discover row's GGUF action menu, as data.
///
/// A GGUF repo is not one model: it ships a folder of quants and the user picks
/// which to run. The row therefore never collapses into a terminal "on disk"
/// state — having `Q4_K_M` says nothing about whether you also want `Q8_0`, so
/// the menu keeps offering the ones you don't have, alongside the ones you do.
enum GgufQuantMenuModel {

    struct Menu: Equatable {
        /// Quants present on this Mac — offer Use / Delete.
        let onDisk: [GgufQuant]
        /// Quants the repo publishes that we don't have yet — offer Download.
        let available: [GgufQuant]
    }

    /// Split a repo's published `.gguf` list against what's on disk.
    ///
    /// Both lists are grouped into quants (shard groups) first, so a sharded
    /// quant folds its many shards into one menu entry. A quant is `onDisk`
    /// only when EVERY shard is present — an incomplete download stays in
    /// `available` (resume). An on-disk quant the repo no longer lists
    /// (re-quantized, renamed) stays in `onDisk`: it's still a real file the
    /// server can load, and hiding it would strand it with no way to select
    /// or delete it.
    static func build(remote: [String], onDisk: [String]) -> Menu {
        let onDiskComplete = GgufQuant.groupQuants(onDisk).filter { $0.isComplete }
        let haveComplete = Set(onDiskComplete.map(\.filename))

        let available = GgufQuant.groupQuants(remote)
            .filter { !haveComplete.contains($0.filename) }

        return Menu(onDisk: onDiskComplete, available: available)
    }

    /// The menu's button title. Owning quants outranks a failed/partial transfer
    /// of a *different* quant — the row must keep reporting what you actually
    /// have, not the state of the last thing you clicked.
    static func buttonLabel(onDisk: [GgufQuant], failed: Bool, hasPartial: Bool) -> String {
        if onDisk.count == 1 { return "✓ \(onDisk[0].label)" }
        if onDisk.count > 1 { return "✓ \(onDisk.count) on disk" }
        if failed { return "Retry" }
        return hasPartial ? "Resume" : "Download"
    }
}
