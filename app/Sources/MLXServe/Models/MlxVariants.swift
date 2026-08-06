import Foundation

/// One selectable quantization inside a multi-variant MLX repo.
///
/// Some publishers ship every quant of a model in ONE repo, as a subfolder per
/// quant, each holding a complete standalone MLX model
/// (`LiquidAI/LFM2.5-2.6B-MLX`: `4bit/`, `5bit/`, `6bit/`, `8bit/`, `bf16/`,
/// `mxfp4/`, `mxfp8/`, `nvfp4/`). That is the same "a repo is a SHELF of
/// models, not a model" shape a GGUF repo has — expressed as directories
/// instead of files — so it gets the same treatment: the user picks a quant,
/// and each one downloads, loads and deletes independently.
struct MlxVariant: Identifiable, Equatable, Hashable, Sendable {
    /// The repo-relative subfolder this variant lives in (`4bit`).
    let folder: String
    /// Short label for menus (`4-bit`, `BF16`, `MXFP4`).
    let label: String
    /// Sum of this folder's `.safetensors` — its own weights, never the repo's.
    let sizeBytes: Int64

    var id: String { folder }

    /// "1.5 GB" — shown next to the label so the pick is informed.
    var sizeLabel: String { sizeBytes > 0 ? MemoryInfo.format(sizeBytes) : "" }
}

enum MlxVariantScan {

    /// Tokenizer filenames that make a subfolder loadable on its own. Any one
    /// is enough — quantized MLX repos ship `tokenizer.json`, older SentencePiece
    /// conversions ship `tokenizer.model`.
    private static let tokenizerNames: Set<String> = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]

    /// The variants a repo's file tree publishes, smallest first. Empty for
    /// every ordinary repo.
    ///
    /// Two gates keep normal repos out, each a fact about real layouts:
    ///
    /// - a root `config.json` means the repo IS a model — its nested `mtp/` or
    ///   `original/` folders are sidecars and shadow copies, not choices;
    /// - a root `model_index.json` means diffusers, whose `transformer/` +
    ///   `text_encoder/` + `vae/` subdirs are PARTS of one model.
    ///
    /// A variant is then an IMMEDIATE subfolder holding a flat, complete model:
    /// `config.json` + weights + a tokenizer. That last requirement is what
    /// tells a quant apart from a diffusers component (which has config +
    /// weights but no tokenizer of its own) and from
    /// `mlx-community/flux2-klein-9b-4bit`, which ships no root json at all and
    /// four config-less weight subdirs.
    static func variants(files: [HFSearchService.TreeFileEntry]) -> [MlxVariant] {
        var rootNames: Set<String> = []
        var byFolder: [String: [HFSearchService.TreeFileEntry]] = [:]
        for f in files {
            let comps = f.path.split(separator: "/").map(String.init)
            if comps.count == 1 { rootNames.insert(comps[0]) }
            if comps.count == 2 { byFolder[comps[0], default: []].append(f) }
        }
        guard !rootNames.contains("config.json"), !rootNames.contains("model_index.json") else { return [] }

        var out: [MlxVariant] = []
        for (folder, entries) in byFolder {
            let names = Set(entries.map { ($0.path as NSString).lastPathComponent })
            guard names.contains("config.json"), !names.isDisjoint(with: tokenizerNames) else { continue }
            let weights = entries
                .filter { $0.path.hasSuffix(".safetensors") }
                .reduce(Int64(0)) { $0 + $1.size }
            guard weights > 0 else { continue }
            out.append(MlxVariant(folder: folder, label: label(forFolder: folder), sizeBytes: weights))
        }
        return out.sorted { ($0.sizeBytes, $0.folder) < ($1.sizeBytes, $1.folder) }
    }

    /// Menu label for a subfolder name, through the same parser the repo-id
    /// badge uses (`4bit` → `4-bit`, `mxfp4` → `MXFP4`). Unrecognized names are
    /// shown verbatim rather than guessed at.
    static func label(forFolder folder: String) -> String {
        HFModel.quantizationLabel(forId: folder) ?? folder
    }

    /// Where a variant is stored: its OWN 2-level model dir, `<org>/<repo>-<folder>`.
    ///
    /// The server's discovery walks exactly two levels (`<root>/<org>/<model>`,
    /// `model_discovery.discoverModelsInDir`), so mirroring the HF layout as
    /// `<org>/<repo>/<folder>` would leave a fully downloaded quant invisible to
    /// `list`, `/v1/models` and the tray picker. As a sibling directory each
    /// variant is an ordinary model to every layer above this one — discovery,
    /// the picker, "Use", delete, LAN sharing — with no special cases.
    static func localRepoId(repoId: String, folder: String) -> String {
        "\(repoId)-\(folder)"
    }
}

/// The Discover row's variant action menu, as data — the MLX twin of
/// `GgufQuantMenuModel`. A shelf never reaches a terminal "on disk" state:
/// owning the 4-bit says nothing about whether you also want the bf16.
enum MlxVariantMenuModel {

    struct Menu: Equatable {
        /// Variants present on this Mac — offer Use / Delete.
        let onDisk: [MlxVariant]
        /// Variants the repo publishes that we don't have yet — offer Download.
        let available: [MlxVariant]
    }

    /// Split the repo's published variants against the folders on disk. An
    /// on-disk variant the repo no longer publishes (renamed, re-quantized)
    /// stays listed: it's a real model the server can load, and dropping it
    /// would strand it with no way to select or delete it from here.
    static func build(remote: [MlxVariant], onDisk: Set<String>) -> Menu {
        let byFolder = Dictionary(remote.map { ($0.folder, $0) }, uniquingKeysWith: { first, _ in first })
        let have = onDisk
            .map { byFolder[$0] ?? MlxVariant(folder: $0, label: MlxVariantScan.label(forFolder: $0), sizeBytes: 0) }
            .sorted { ($0.sizeBytes, $0.folder) < ($1.sizeBytes, $1.folder) }
        return Menu(onDisk: have, available: remote.filter { !onDisk.contains($0.folder) })
    }

    /// The menu's button title. Owning variants outranks a failed/partial
    /// transfer of a *different* one — the row reports what you have, not the
    /// state of the last thing you clicked.
    static func buttonLabel(onDisk: [MlxVariant], failed: Bool, hasPartial: Bool) -> String {
        if onDisk.count == 1 { return "✓ \(onDisk[0].label)" }
        if onDisk.count > 1 { return "✓ \(onDisk.count) on disk" }
        if failed { return "Retry" }
        return hasPartial ? "Resume" : "Download"
    }
}
