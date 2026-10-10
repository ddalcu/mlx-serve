import Foundation

/// A speculation sidecar that fits one model's socket (Model Settings).
struct DrafterGem: Equatable, Hashable, Identifiable {
    enum Kind: String {
        case mtp, dflash2, dspark, gemmaAssistant, museAssistant, packDrafter
    }

    let kind: Kind
    /// Repo holding the files: the model's own repo for a pack subfolder, "" for MTP.
    let repo: String
    /// `drafter` when the gem rides inside the model's pack (the server finds it there).
    let subfolder: String?
    let sizeGB: Double

    var id: String { "\(kind.rawValue):\(repo)" }

    var label: String {
        switch kind {
        case .mtp: "MTP head (built in)"
        case .dflash2: "DFlash2 drafter"
        case .dspark: "DSpark drafter"
        case .gemmaAssistant: "Gemma assistant drafter"
        case .museAssistant: "Muse assistant drafter"
        case .packDrafter: "Pack drafter"
        }
    }

    var needsDownload: Bool { kind != .mtp }
    /// Loads through the server's DFlash engine (a pack's `drafter/` does too).
    var isDflash: Bool { kind != .mtp && kind != .gemmaAssistant }
    /// Terms shown where a download is offered: incoai's files are CC BY-NC-ND 4.0.
    var licenseNote: String? { repo == DrafterGems.glmDFlash2Repo ? "non-commercial license" : nil }
}

enum DrafterGems {
    static let packFolder = "drafter"
    static let qwen38DFlash2Repo = "z-lab/Qwen3.8-27B-DFlash2"
    static let glmDFlash2Repo = "incoai/GLM-5.3-Flash-DFlash2"
    static let museAssistantRepo = "meta-models/Muse-Glimmer-30B-assistant"
    /// LiquidAI's DSpark drafters, by the LFM2.5 base they draft for.
    static let lfmDSparkRepos = ["lfm2.5-2.6b": "LiquidAI/LFM2.5-2.6B-DSpark", "lfm2.5-8b-a1b": "LiquidAI/LFM2.5-8B-A1B-DSpark"]

    /// The gems that fit `repoId`, the default first (after MTP).
    /// `packFiles`: the pack's HF listing (path → size) when fetched, nil when
    /// unknown. `localDrafter`: `<model_dir>/drafter` is already on disk.
    static func gems(forRepoId repoId: String, packFiles: [String: Int64]?,
                     localDrafter: Bool, mtpAvailable: Bool) -> [DrafterGem] {
        var out: [DrafterGem] = []
        if mtpAvailable { out.append(DrafterGem(kind: .mtp, repo: "", subfolder: nil, sizeGB: 0)) }
        let base = (repoId as NSString).lastPathComponent.lowercased()
        guard !base.contains("gguf"), !base.contains("assistant"), !base.contains("dflash"), !base.contains("dspark") else { return out }

        let prefix = packFolder + "/"
        let packBytes = packFiles?.filter { $0.key.hasPrefix(prefix) }.values.reduce(0, +) ?? 0
        let packHasDrafter = localDrafter || packFiles?[prefix + "config.json"] != nil
        let packGB = Double(packBytes) / 1e9

        if base.contains("qwen3.8-27b") {
            out.append(packHasDrafter
                ? DrafterGem(kind: .dflash2, repo: repoId, subfolder: packFolder, sizeGB: packGB > 0 ? packGB : 3.85)
                : DrafterGem(kind: .dflash2, repo: qwen38DFlash2Repo, subfolder: nil, sizeGB: 3.85))
        } else if packHasDrafter {
            out.append(DrafterGem(kind: .packDrafter, repo: repoId, subfolder: packFolder, sizeGB: packGB))
        } else if base.contains("glm-5.3-flash") {
            out.append(DrafterGem(kind: .dflash2, repo: glmDFlash2Repo, subfolder: nil, sizeGB: 2.34))
        } else if base.contains("muse-glimmer") {
            out.append(DrafterGem(kind: .museAssistant, repo: museAssistantRepo, subfolder: nil, sizeGB: 5.11))
        } else if let repo = lfmDSparkRepos.first(where: { base.hasPrefix($0.key) })?.value {
            out.append(DrafterGem(kind: .dspark, repo: repo, subfolder: nil, sizeGB: 0.66))
        } else if base.contains("gemma-4") || base.contains("gemma4"),
                  let variant = DownloadManager.gemmaVariantFor(modelPath: base, isMoE: false), variant != .moe26B {
            // The MoE Gemma is left out: verify pays expert routing, the drafter regresses decode.
            out.append(DrafterGem(kind: .gemmaAssistant, repo: variant.drafterRepoId, subfolder: nil, sizeGB: variant.drafterSizeGB))
        }
        return out
    }

    /// The gem a fresh download fills its socket with. The separate-repo DFlash2
    /// drafters are offered but never auto-filled: they pay on code and edits, not everywhere.
    static func defaultGem(_ gems: [DrafterGem]) -> DrafterGem? {
        gems.first { $0.needsDownload && $0.repo != qwen38DFlash2Repo && $0.repo != glmDFlash2Repo }
    }

    /// Whether `gem` fits beside a model of `modelGB` in this Mac's usable memory.
    static func fits(_ gem: DrafterGem, modelGB: Double, memory: SystemMemoryInfo) -> Bool {
        memory.fit(neededGB: modelGB + gem.sizeGB) != .exceeds
    }
}

/// The stone a My Models row shows for the speculation method a model will run.
enum GemStone: String {
    case emerald, amethyst, sapphire, topaz, ruby

    var label: String {
        switch self {
        case .emerald: "MTP head"
        case .amethyst: "DSpark drafter"
        case .sapphire: "DFlash drafter"
        case .topaz: "DFlash2 drafter"
        case .ruby: "Gemma assistant drafter"
        }
    }

    /// The short tag the tray and My Models both print beside the stone.
    var badge: String {
        switch self {
        case .emerald: "+MTP"
        case .amethyst: "+DS"
        case .sapphire: "+DF"
        case .topaz: "+DF2"
        case .ruby: "+Drafter"
        }
    }
}

/// `skull`: MTP accepts drafts lossily (Typical / TokenV3).
struct SocketBadge: Equatable {
    var stone: GemStone?
    var skull: Bool
}

extension DrafterGems {
    /// The stone a drafter `config.json` declares, read like `dflash.zig`: the
    /// contract triple nested-first, a Markov head is DSpark, a selector or
    /// dynamic convs DFlash2; a Gemma assistant is the one non-DFlash drafter.
    static func stone(drafterConfig root: [String: Any]) -> GemStone? {
        if (root["model_type"] as? String)?.hasSuffix("_assistant") == true { return .ruby }
        let nested = root["dflash_config"] as? [String: Any] ?? [:]
        func get(_ k: String) -> Any? { nested[k] ?? root[k] }
        guard get("block_size") != nil, get("mask_token_id") != nil, get("target_layer_ids") != nil else { return nil }
        let archs = root["architectures"] as? [String] ?? []
        if root["markov_rank"] != nil || get("projector_type") as? String == "dspark" || archs.contains(where: { $0.contains("DSpark") }) {
            return .amethyst
        }
        if (get("selector_rank") as? Int ?? 0) > 0 || (get("conv_kernel_size") as? Int ?? 0) > 0 { return .topaz }
        return .sapphire
    }

    /// The method the server dispatches for this model: DFlash-family drafter >
    /// MTP > Gemma assistant.
    static func badge(_ o: ModelOverride, modelDir: String, hasMtpHead: Bool, options: ServerOptions,
                      config: (String) -> [String: Any]? = readConfig) -> SocketBadge {
        let drafterDir: String? = switch o.drafter {
        case "off": nil
        case nil, "auto": (modelDir as NSString).appendingPathComponent(packFolder)
        case let path?: path
        }
        let drafter = drafterDir.flatMap(config).flatMap(stone(drafterConfig:))
        if let drafter, drafter != .ruby { return SocketBadge(stone: drafter, skull: false) }
        guard hasMtpHead, o.mtp ?? options.enableMTP else { return SocketBadge(stone: drafter, skull: false) }
        return SocketBadge(stone: .emerald, skull: o.mtpAcceptance == .typical || o.mtpAcceptance == .tokenv3)
    }

    static func readConfig(_ dir: String) -> [String: Any]? {
        FileManager.default.contents(atPath: (dir as NSString).appendingPathComponent("config.json"))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    }
}

/// What a model's speculation socket holds, read from and written to its
/// `model-settings.json` entry (`drafter` + `mtp`).
enum DrafterSocket: Equatable {
    /// No setting: the server decides (the pack's `drafter/`, else its default MTP).
    case automatic
    case empty
    case gem(DrafterGem)
    /// A drafter path no known gem accounts for.
    case custom(String)

    /// The server binds a DFlash drafter for this socket: `automatic` loads the
    /// pack's own `drafter/` when it is on disk.
    func bindsDflash(localDrafter: Bool) -> Bool {
        switch self {
        case .automatic: localDrafter
        case .gem(let g): g.isDflash
        case .empty, .custom: false
        }
    }

    /// `pathOf`: where a separate-repo gem lives on disk, nil when absent.
    static func read(_ o: ModelOverride, gems: [DrafterGem], pathOf: (DrafterGem) -> String?) -> DrafterSocket {
        let mtp = gems.first { $0.kind == .mtp }
        switch o.drafter {
        case nil:
            if o.mtp == true, let mtp { return .gem(mtp) }
            return .automatic
        case "off":
            if o.mtp == true, let mtp { return .gem(mtp) }
            return .empty
        case "auto":
            return gems.first { $0.subfolder != nil }.map(DrafterSocket.gem) ?? .automatic
        case let path?:
            return gems.first { $0.subfolder == nil && $0.needsDownload && pathOf($0) == path }
                .map(DrafterSocket.gem) ?? .custom(path)
        }
    }

    /// `gemPath`: the on-disk dir of a separate-repo gem.
    func write(into o: inout ModelOverride, gemPath: String? = nil) {
        switch self {
        case .automatic:
            o.drafter = nil; o.mtp = nil
        case .empty:
            o.drafter = "off"; o.mtp = false
        case .custom(let path):
            o.drafter = path
        case .gem(let g) where g.kind == .mtp:
            o.drafter = "off"; o.mtp = true
        case .gem(let g) where g.subfolder != nil:
            o.drafter = "auto"; o.mtp = nil
        case .gem:
            o.drafter = gemPath; o.mtp = nil
        }
    }
}

/// One-time fold of the old global drafter (`ServerOptions.drafterPath` /
/// `drafterOptOut` + `DrafterPairing`) into per-model entries.
enum DrafterMigration {
    /// `pairs`: every local model → the companion drafter dir on disk.
    /// `selected` + `globalPath`: the model the global path was set for. Entries
    /// that already name a `drafter` are left alone.
    static func migrate(_ file: inout ModelSettingsFile, pairs: [String: String],
                        selected: String, globalPath: String, optedOut: Bool) {
        var want = pairs.mapValues { optedOut ? "off" : $0 }
        if !selected.isEmpty {
            if optedOut { want[selected] = "off" } else if !globalPath.isEmpty { want[selected] = globalPath }
        }
        for (model, value) in want {
            var o = file.override(for: model) ?? ModelOverride()
            guard o.drafter == nil else { continue }
            o.drafter = value
            file.set(o, for: model)
        }
    }
}
