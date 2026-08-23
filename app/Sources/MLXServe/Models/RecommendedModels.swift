import Foundation

private let bytesPerGiB: Double = 1_073_741_824

/// Data behind the Model Browser's "Recommended" pane: every Gemma 4 and
/// Qwen 3.5/3.6 checkpoint this app is tuned hardest for (native MTP
/// speculative decode, PLD, the assistant-drafter catalog all target these),
/// grouped by family and explained in plain English for someone who has
/// never picked a local model before. It intentionally does NOT reuse
/// `gemmaModelOptions`: that catalog is a flat download-tray list keyed for
/// CLI-style browsing, where this one needs the descriptive copy the pane
/// renders. Both point at the same underlying HuggingFace repos.
///
/// Every `sizeGB` below is the real on-disk weight size — summed from each
/// repo's safetensors dtype byte counts via the HuggingFace API, the same
/// convention `GemmaModelOption.sizeEstimate` uses (raw weights, not the
/// +20% RAM-with-overhead figure `HFModel.ramEstimate` shows elsewhere) —
/// not guessed from the model name.
///
/// # Capability scores (`intelligence` / `speed` / `contextTokens`)
///
/// The three bars the Model Browser draws. All three describe the ORIGINAL
/// published weights, not the 4-bit/2-bit build this pick downloads — a quant
/// moves quality a little and speed a lot, and pretending otherwise would make
/// every bar a claim we can't source.
///
/// **Intelligence** is the Artificial Analysis Intelligence Index
/// (<https://artificialanalysis.ai/models/open-source>), read on **2026-07-30**,
/// rescaled `round(index / 60 × 100)` so the bar is "fraction of the best model
/// available anywhere" (the index's frontier sat at 61 that day). Where a model
/// is published in both reasoning and non-reasoning modes the REASONING number
/// is used — this app runs them with thinking available. Models the site has no
/// entry for carry our own estimate and `intelligenceIsEstimated = true`; there
/// is no "unrated" state, because a missing bar reads as "bad" rather than
/// "unknown".
///
/// **Speed** is always our OWN relative estimate, never the site's: their
/// figure is measured on cloud GPUs, and that ordering does not survive the
/// move to Apple Silicon, where decode is bandwidth-bound and ACTIVE
/// parameters dominate. Same M4 Max, 26.7.12 bench (benchmarks.md; the
/// per-cell CSV lives in git history at docs/perf-csvs/all-26.7.12.csv):
/// `gemma4-26b-a4b` 118 tok/s against `gemma4-31b` 25 tok/s — a 4.7× gap a
/// cloud comparison shows as nearly level. The score is `round(100 × tok/s / 200)`
/// over PLAIN autoregressive decode, calibrated against that CSV where a row
/// exists and estimated from active params + weight bytes elsewhere.
/// Speculative decode (MTP/PLD/drafter) is deliberately excluded: it is a
/// property of how we run the model rather than of the model, it moves some
/// picks 2-3× on their own, and each pick that has one already says so in its
/// blurb. `activeParamsB` exists so these hand-edited numbers can be CHECKED
/// rather than trusted — see the ordering invariant in `RecommendedModelsTests`.
///
/// **Context** is the checkpoint's own `max_position_embeddings`, read from
/// each repo's `config.json`. It is deliberately NOT the RAM-clamped effective
/// window: the bars compare models to each other, and the clamp is a property
/// of the user's Mac.

/// Which curated section a pick belongs to. Gemma/Qwen are vendor families;
/// `largest` is a RAM tier — the biggest models this app runs (DeepSeek-V4-Flash
/// on the native MLX arch, Hunyuan 3), grouped by "needs a very large Mac"
/// rather than vendor.
enum RecommendedModelFamily: String {
    case gemma = "Gemma"
    case qwen = "Qwen"
    case poolside = "poolside"
    case largest = "Largest models"
}

/// One curated, plain-English download recommendation.
struct RecommendedModelPick: Identifiable, Hashable {
    let id: String
    let name: String
    /// Short (~3 word) framing shown right under the name.
    let tagline: String
    /// The full explanation, written for someone with zero AI experience —
    /// what it's good at and what the trade-off is versus its neighbors.
    /// Rendered as the description under the model in the list row.
    let blurb: String
    let repoId: String
    /// Approximate on-disk weight size in GB.
    let sizeGB: Double
    let family: RecommendedModelFamily
    /// 0–100. The Artificial Analysis Intelligence Index, rescaled — see the
    /// file header for the source, the as-of date and the rescale.
    let intelligence: Int
    /// True when the site had no entry for this model and `intelligence` is our
    /// own estimate. Never a reason to hide the bar; the pane says so instead.
    let intelligenceIsEstimated: Bool
    /// 0–100. Our own Apple-Silicon decode estimate — never the site's
    /// cloud-GPU speed. See the file header.
    let speed: Int
    /// The checkpoint's own context window (`max_position_embeddings`), NOT the
    /// RAM-clamped effective one.
    let contextTokens: Int
    /// Active parameters per token, in billions — dense models count their
    /// whole size, an MoE counts only what it wakes (Laguna S 2.1 is
    /// 117.6B-**A8.5B**, so 8.5). Read from each repo's config/model card, and
    /// the basis the hand-edited `speed` scores are checked against: on Apple
    /// Silicon decode is bandwidth-bound, so a model with more active
    /// parameters must never be scored FASTER than one with fewer.
    let activeParamsB: Double
    /// Overrides the generic weights×1.2 RAM estimate for picks where that
    /// formula misleads (e.g. a build whose runtime footprint or context needs
    /// push the honest recommendation gate above what the on-disk size implies).
    /// Used by DeepSeek-V4-Flash, where ×1.2 overshoots the real footprint by
    /// enough to hide the model from the Mac it was converted for.
    var ramOverrideGB: Double? = nil
    /// For a GGUF/ds4 pick: the specific `.gguf` file this recommendation
    /// downloads (the repo ships many; the curated pick names one known-good
    /// quant). nil for a safetensors pick, whose whole repo is fetched. When
    /// set, the pane downloads via the GGUF path (and auto-pulls the ds4 MTP
    /// draft head) instead of the safetensors-tree path.
    var ggufFilename: String? = nil

    var sizeLabel: String { String(format: "~%.1f GB", sizeGB) }

    /// The quant this pick downloads, read off the GGUF filename or the repo
    /// name ("…-4bit", "…-oQ2e", "…-mixed-2-3-8bit", "…-NVFP4-…"). nil when the
    /// name carries none.
    var quantLabel: String? {
        if let f = ggufFilename { return DownloadManager.quantLabel(forFilename: f) }
        let name = (repoId as NSString).lastPathComponent
        if let r = name.range(of: "mixed(-[0-9]+)+bit", options: .regularExpression) {
            let bits = name[r].dropFirst("mixed-".count).dropLast("bit".count).split(separator: "-")
            return "mixed \(bits.joined(separator: "/"))-bit"
        }
        if let r = name.range(of: "iQ-MLX-[0-9.]+bpw", options: .regularExpression) {
            let bpw = name[r].dropFirst("iQ-MLX-".count).dropLast("bpw".count)
            return "iQ-MLX \(bpw) bpw"
        }
        if let r = name.range(of: "(?<![A-Za-z0-9])[0-9]+bit", options: .regularExpression) {
            return name[r].dropLast("bit".count) + "-bit"
        }
        if let r = name.range(of: "(?<![A-Za-z0-9])(oQ[0-9]+e?|NVFP[0-9]+|MXFP[0-9]+|Q[0-9]_[A-Z0-9_]+)(?![A-Za-z0-9])",
                              options: .regularExpression) {
            return String(name[r])
        }
        return nil
    }

    // MARK: - Capability bars
    //
    // 0…1 track fills. No number is ever rendered next to them: the scores are
    // a hand-maintained comparison between these picks, and printing "62" would
    // claim a precision they don't have.

    var intelligenceBar: Double { Double(intelligence) / 100 }
    var speedBar: Double { Double(speed) / 100 }

    /// Context on a LOG scale between 32K (empty) and 1M (full). Linear would
    /// leave every pick but DeepSeek pinned at a quarter of the track, because
    /// the field's windows are powers of two two doublings apart.
    var contextBar: Double {
        let floorTokens = 32_768.0, ceilTokens = 1_048_576.0
        let t = Double(max(contextTokens, 1))
        let f = (log2(t) - log2(floorTokens)) / (log2(ceilTokens) - log2(floorTokens))
        return min(max(f, 0), 1)
    }

    /// Approximate RAM this checkpoint needs once loaded — weights plus the
    /// same ~20% KV-cache/runtime-buffer overhead `HFModel.ramEstimate` and
    /// `GemmaModelOption.sizeEstimate` budget for elsewhere in the app.
    var approxRAMNeededGB: Double { ramOverrideGB ?? sizeGB * 1.2 }

    /// Whether this Mac's physical RAM covers what the model needs. This is a
    /// SOFT signal for the UI (dim the row, explain why) — never a hard
    /// download gate. The rest of the app never blocks on this either
    /// (Discover just colors a fitness dot; ImageGenView warns and lets the
    /// user proceed anyway), so a pick that doesn't meet requirements stays
    /// fully downloadable/usable here too.
    func meetsSystemRequirements(physicalMemoryBytes: UInt64) -> Bool {
        Double(physicalMemoryBytes) >= approxRAMNeededGB * bytesPerGiB
    }
}

/// The fixed pool of picks the Recommended pane draws from, as static
/// members on the type itself so the catalogs below can use plain
/// dot-shorthand (`.gemmaE2B`).
extension RecommendedModelPick {
    static let gemmaE2B = RecommendedModelPick(
        id: "gemma-4-e2b",
        name: "Gemma 4 E2B",
        tagline: "Small and capable",
        blurb: "A well-rounded everyday assistant — good at chatting, answering questions, and simple coding help, while staying small and fast to run.",
        repoId: "mlx-community/gemma-4-e2b-it-4bit",
        sizeGB: 3.3,
        family: .gemma,
        intelligence: 15,
        intelligenceIsEstimated: false,
        speed: 90,
        contextTokens: 131_072,
        activeParamsB: 2.0
    )

    static let gemmaE4B = RecommendedModelPick(
        id: "gemma-4-e4b",
        name: "Gemma 4 E4B",
        tagline: "The sweet spot",
        blurb: "A clear step up in quality from E2B — better at longer conversations, writing, and coding — while still replying quickly. A great all-around default if you're not sure what to pick.",
        repoId: "mlx-community/gemma-4-e4b-it-4bit",
        sizeGB: 4.8,
        family: .gemma,
        intelligence: 20,
        intelligenceIsEstimated: false,
        speed: 57,
        contextTokens: 131_072,
        activeParamsB: 4.0
    )

    static let gemma12B = RecommendedModelPick(
        id: "gemma-4-12b",
        name: "Gemma 4 12B",
        tagline: "Sharper reasoning",
        blurb: "Noticeably better at following detailed, multi-step instructions and reasoning through trickier problems than the smaller Gemma models, without needing a huge amount of memory.",
        repoId: "mlx-community/gemma-4-12b-it-4bit",
        sizeGB: 6.3,
        family: .gemma,
        intelligence: 37,
        intelligenceIsEstimated: false,
        speed: 23,
        contextTokens: 262_144,
        activeParamsB: 12.0
    )

    static let gemma26bA4b = RecommendedModelPick(
        id: "gemma-4-26b-a4b",
        name: "Gemma 4 26B-A4B",
        tagline: "Big, but efficient",
        blurb: "A much larger model that uses a trick called \u{201c}mixture of experts\u{201d} — for every word it only wakes up a small part of itself, so it answers faster than you'd expect for its size while giving noticeably better answers than the smaller Gemma models.",
        repoId: "mlx-community/gemma-4-26b-a4b-it-4bit",
        sizeGB: 14.3,
        family: .gemma,
        intelligence: 33,
        intelligenceIsEstimated: false,
        speed: 59,
        contextTokens: 262_144,
        activeParamsB: 4.0
    )

    static let gemma31B = RecommendedModelPick(
        id: "gemma-4-31b",
        name: "Gemma 4 31B",
        tagline: "Gemma's biggest all-rounder",
        blurb: "Gemma's largest single model — every part of it is used on every word, with no shortcuts. Excellent general reasoning, writing, and instruction-following.",
        repoId: "mlx-community/gemma-4-31b-it-4bit",
        sizeGB: 17.2,
        family: .gemma,
        intelligence: 48,
        intelligenceIsEstimated: false,
        speed: 13,
        contextTokens: 262_144,
        activeParamsB: 31.0
    )

    static let gemma26bA4b8bit = RecommendedModelPick(
        id: "gemma-4-26b-a4b-8bit",
        name: "Gemma 4 26B-A4B (higher precision)",
        tagline: "Sharper version of the MoE model",
        blurb: "The same Gemma mixture-of-experts model as above, stored with twice the numeric precision (8-bit instead of 4-bit). That means slightly more accurate, consistent answers, at roughly double the memory.",
        repoId: "mlx-community/gemma-4-26b-a4b-it-8bit",
        sizeGB: 26.0,
        family: .gemma,
        intelligence: 33,
        intelligenceIsEstimated: false,
        speed: 33,
        contextTokens: 262_144,
        activeParamsB: 4.0
    )

    static let gemma31B8bit = RecommendedModelPick(
        id: "gemma-4-31b-8bit",
        name: "Gemma 4 31B (highest quality)",
        tagline: "The best this app offers",
        blurb: "The highest-quality model on this list: Gemma's largest model at full 8-bit precision. This is about as good as local answers get here — save it for when quality matters more than speed.",
        repoId: "mlx-community/gemma-4-31b-it-8bit",
        sizeGB: 31.5,
        family: .gemma,
        intelligence: 48,
        intelligenceIsEstimated: false,
        speed: 7,
        contextTokens: 262_144,
        activeParamsB: 31.0
    )

    /// Qwen 3.5 9B — the entry-level Qwen pick. Replaces the earlier 0.8B
    /// entry, which was too small to be a meaningful comparison against the
    /// Gemma lineup.
    static let qwen35_9b = RecommendedModelPick(
        id: "qwen35-9b",
        name: "Qwen 3.5 (9B)",
        tagline: "A capable everyday pick",
        blurb: "A well-rounded Qwen model — good at chatting, coding help, and following instructions, while staying quick to respond. A solid alternative to Gemma if you want to compare styles.",
        repoId: "mlx-community/Qwen3.5-9B-MLX-4bit",
        sizeGB: 5.5,
        family: .qwen,
        intelligence: 35,
        intelligenceIsEstimated: false,
        speed: 28,
        contextTokens: 262_144,
        activeParamsB: 9.0
    )

    /// Qwen 3.8 27B, the pick this app leads with on any Mac that can hold it.
    /// Same geometry as the 3.6 27B it replaces (dense 27B, 64 layers, 256K
    /// context) with a newer generation's weights, a vision tower and the MTP
    /// draft head inside the checkpoint rather than in a sidecar.
    ///
    /// `intelligence` is ESTIMATED: the model shipped 2026-08-14 and the index
    /// has no entry for it. Placed level with Hunyuan 3 and one point above the
    /// 3.6 27B it supersedes — a newer generation of the same size class — and
    /// below DeepSeek-V4-Flash. `speed` is our own measurement of PLAIN decode
    /// on an M4 Max (26.3 tok/s on code, 26.7 on prose, median of 3 at temp 0,
    /// `--no-mtp`): the draft head takes it to ~75, which is excluded from the
    /// score by policy and named in the blurb instead — see the file header.
    static let qwen38_27b = RecommendedModelPick(
        id: "qwen38-27b",
        name: "Qwen 3.8 27B",
        tagline: "One of the strongest models here",
        blurb: "One of the most capable models this app can run — excellent at coding and at multi-step \u{201c}agent\u{201d} tasks like using tools and following a plan, and it reads images too. It also ships with a built-in speed trick that lets it draft and double-check several words at once, so it feels noticeably faster than a plain model this size.",
        repoId: "ddalcu/Qwen3.8-27B-MLX-Serve-4bit",
        sizeGB: 18.2,
        family: .qwen,
        intelligence: 63,
        intelligenceIsEstimated: true,
        speed: 13,
        contextTokens: 262_144,
        activeParamsB: 27.0
    )

    /// Tencent Hunyuan 3 (295B-A21B MoE) — the largest open model this app
    /// runs. This is the imatrix-calibrated 2-bit build (mlx-community oQ2e):
    /// the FULL 192-expert model with attention/router/shared-expert/embeddings
    /// kept at 8-bit, ~84 GB. Unlike the older ~105 GB mixed build (which loaded
    /// on a 128 GB Mac but left almost no room for context), this fits 128 GB
    /// with a usable window, so it's recommended INLINE on 128 GB — the generic
    /// weights×1.2 formula (~100 GB) already gates it above a 96 GB Mac.
    static let hy3_295b = RecommendedModelPick(
        id: "hy3-oq2e",
        name: "Hunyuan 3 295B",
        tagline: "The biggest model here",
        blurb: "Tencent's flagship open model — 295 billion parameters, of which it wakes only 21 billion per word (mixture of experts). Top-tier reasoning, agent work, and tool use, entirely on your Mac. This is the full model stored compactly (importance-calibrated 2-bit), so on a 128 GB Mac it runs with a genuinely usable context window rather than just a few sentences. Best on Macs with 128 GB of memory or more.",
        repoId: "mlx-community/Hy3-oQ2e",
        sizeGB: 83.7,
        family: .largest,
        // Estimated: the site has no Hunyuan 3 entry. Placed just under
        // DeepSeek-V4-Flash — a comparable flagship open MoE at a similar
        // activated-parameter scale — and above every Qwen/Gemma pick here.
        intelligence: 63,
        intelligenceIsEstimated: true,
        speed: 14,
        contextTokens: 262_144,
        activeParamsB: 21.0
    )

    /// DeepSeek-V4-Flash via the embedded ds4 engine — a frontier-scale model
    /// on a very large Mac — now on our OWN native `deepseek_v4` MLX arch
    /// rather than the embedded ds4 GGUF engine, so the curated pick is our
    /// mixed 2/3/8-bit mirror (whole safetensors repo, no quant file to name).
    /// It supersedes the `antirez/deepseek-v4-gguf` IQ2XXS pick: same model,
    /// engine-native instead of converted, at the cost of the 96 GB tier —
    /// this build wants 128 GB.
    static let deepseekV4Flash = RecommendedModelPick(
        id: "deepseek-v4-flash",
        name: "DeepSeek-V4-Flash",
        tagline: "Frontier model, native MLX",
        blurb: "A frontier-scale DeepSeek model that runs natively on Apple Silicon through MLX — no GGUF conversion, no llama.cpp — for top-tier reasoning, coding, and agent work. It wakes only a fraction of itself per word (mixture of experts) and holds around a million words of context. This is our own iQ-MLX conversion at 3.3 bits per weight, about 130 GB on disk, so it wants a Mac with 128 GB of memory (raise the GPU wired limit and close other apps to fit a useful context); it also ships DeepSeek's own DSpark draft stages, which Settings can switch on for a faster reply on Macs with more than 128 GB.",
        repoId: "ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw",
        sizeGB: 129.6,
        family: .largest,
        intelligence: 67,
        intelligenceIsEstimated: false,
        // Plain autoregressive decode measures ~23 tok/s on an M4 Max against
        // Hunyuan 3's ~26, so the two score level here rather than this one
        // sitting below it (ties are what the activeParams ordering invariant
        // leaves free, and it floors this pick at Hunyuan's score anyway).
        // DSpark takes it to ~35 tok/s; spec decode is excluded from the score
        // by policy and named in the blurb instead — see the file header.
        speed: 14,
        contextTokens: 1_048_576,
        activeParamsB: 13.0,
        // Weights×1.2 would demand 141 GB and hide the model from the exact
        // machine the conversion targets: it serves in ~110 GB resident on a
        // 128 GB Mac.
        ramOverrideGB: 128.0
    )

    static let qwen36_35bA3b = RecommendedModelPick(
        id: "qwen36-35b-a3b",
        name: "Qwen 3.6 35B-A3B",
        tagline: "Qwen's largest model here",
        blurb: "Qwen's biggest model on this list — 35 billion parameters in total, but like the Gemma mixture-of-experts model above, it only activates a few billion per word, so it stays efficient. Excellent for demanding coding and reasoning work.",
        repoId: "mlx-community/Qwen3.6-35B-A3B-4bit",
        sizeGB: 19.0,
        family: .qwen,
        intelligence: 53,
        intelligenceIsEstimated: false,
        speed: 76,
        contextTokens: 262_144,
        activeParamsB: 3.0
    )

    /// poolside's Laguna XS 2.1 in poolside's own NVFP4 4-bit MLX build —
    /// the checkpoint the 26.7.12 decode-perf round was tuned and validated
    /// on (121 tok/s on an M4 Max, see benchmarks.md). ~20 GB of weights, so
    /// with the ×1.2 overhead it fits a 32 GB Mac inline.
    static let lagunaXS21 = RecommendedModelPick(
        id: "laguna-xs-2.1-nvfp4",
        name: "Laguna XS 2.1",
        tagline: "Fast coder",
        blurb: "The smaller Laguna, same specialty: writing and editing code and multi-step \u{201c}agent\u{201d} work like using tools across a project. A 33-billion-parameter mixture-of-experts model in poolside's own 4-bit NVFP4 build, and the fastest coding model in this list: about 121 tokens per second on an M4 Max. Fits a 32 GB Mac.",
        repoId: "poolside/Laguna-XS-2.1-NVFP4-mlx",
        sizeGB: 20.1,
        family: .poolside,
        // Estimated: poolside publish no Laguna entry on the site. A coding
        // specialist scores below its general-purpose size class on a general
        // index, so it sits under Qwen 3.6 35B-A3B despite the same 3B active.
        intelligence: 43,
        intelligenceIsEstimated: true,
        speed: 61,
        contextTokens: 262_144,
        activeParamsB: 3.0
    )

    /// poolside's Laguna S 2.1 — the full-size coding-specialist MoE (117.6B
    /// total, ~8.5B active per word) in poolside's own NVFP4 4-bit MLX build
    /// (~67 GB on disk, so ~80 GB RAM with the ×1.2 overhead: a 96 GB+ Mac
    /// inline, behind "Requires more RAM" below that). Replaced the compact
    /// 2-bit community build (`pipenetwork/Laguna-S-2.1-MLX-2bit`) — the
    /// 2-bit's output quality is noticeably below the NVFP4 original.
    static let lagunaS21 = RecommendedModelPick(
        id: "laguna-s-2.1-nvfp4",
        name: "Laguna S 2.1",
        tagline: "Built for code",
        blurb: "poolside's full-size coding specialist, built for writing and editing code and for multi-step \u{201c}agent\u{201d} work like using tools across a whole project. A large mixture-of-experts model — 117 billion parameters in total, but only about 8.5 billion wake up per word — in poolside's own 4-bit NVFP4 build, the quantization the model ships in. Needs a Mac with a lot of memory; the XS above covers smaller machines.",
        repoId: "poolside/Laguna-S-2.1-NVFP4-mlx",
        sizeGB: 67.0,
        family: .poolside,
        // Estimated: no site entry (see Laguna XS). Placed between Qwen 3.6
        // 35B-A3B and 27B — the full-size coder, still a specialist.
        intelligence: 55,
        intelligenceIsEstimated: true,
        speed: 28,
        contextTokens: 262_144,
        activeParamsB: 8.5
    )
}

extension RecommendedModelPick {
    /// Gemma 4 picks, ascending by size — one of the Recommended pane's two
    /// family sections.
    static let gemmaCatalog: [RecommendedModelPick] = [
        .gemmaE2B, .gemmaE4B, .gemma12B, .gemma26bA4b, .gemma31B, .gemma26bA4b8bit, .gemma31B8bit,
    ]

    /// Qwen picks, ascending by size — the Recommended pane's other family
    /// section.
    static let qwenCatalog: [RecommendedModelPick] = [
        .qwen35_9b, .qwen38_27b, .qwen36_35bA3b,
    ]

    /// poolside's Laguna family, ascending by size — the fast XS 2.1, then
    /// the full-size S 2.1, both poolside's own NVFP4 builds. Its own
    /// section: coding specialists that aren't Gemma or Qwen.
    static let poolsideCatalog: [RecommendedModelPick] = [
        .lagunaXS21, .lagunaS21,
    ]

    /// The largest models this app runs, ascending by on-disk size (the app's
    /// smallest-first convention) — Hunyuan 3 295B (the compact ~84 GB oQ2e
    /// build) then DeepSeek-V4-Flash (~87 GB ds4 GGUF). Grouped by "needs a
    /// very large Mac" rather than by vendor.
    static let largestCatalog: [RecommendedModelPick] = [
        .hy3_295b, .deepseekV4Flash,
    ]

    /// Every curated pick, across all four sections — the union the score
    /// invariants sweep and the one list a new section can't slip past.
    static let allCatalogs: [RecommendedModelPick] =
        gemmaCatalog + qwenCatalog + poolsideCatalog + largestCatalog

    // MARK: - The starter recommendation

    /// The ONE model to offer someone who has downloaded nothing yet, chosen
    /// from this Mac's physical RAM. Total by construction — every input
    /// returns a pick — and the single source of truth for all three surfaces
    /// that make this recommendation: the Model Browser's "Best for your Mac"
    /// card, the welcome window's starter card, and the chat gate. Two copies
    /// of this decision is how two of them start recommending different models.
    ///
    /// Each tier is the largest pick that still leaves the machine room to do
    /// anything else (`approxRAMNeededGB` = weights × 1.2, the app's ~20%
    /// runtime overhead), never merely the largest that fits:
    ///
    /// | Physical RAM | Pick | Disk | RAM needed |
    /// |---|---|---|---|
    /// | ≤ 8 GB  | Gemma 4 E2B  |  3.3 GB |  4.0 GB |
    /// | 8–16 GB | Gemma 4 E4B  |  4.8 GB |  5.8 GB |
    /// | 16–32 GB| Gemma 4 12B  |  6.3 GB |  7.6 GB |
    /// | 32 GB+  | Qwen 3.8 27B | 18.2 GB | 21.8 GB |
    ///
    /// Bands are upper-INCLUSIVE: a 16 GB Mac gets E4B, not 12B. A boundary
    /// machine is the one with the least headroom in its band, so it takes the
    /// smaller side.
    static func starterPick(physicalMemoryBytes: UInt64) -> RecommendedModelPick {
        let gib = Double(physicalMemoryBytes) / bytesPerGiB
        if gib <= 8 { return .gemmaE2B }
        if gib <= 16 { return .gemmaE4B }
        if gib <= 32 { return .gemma12B }
        return .qwen38_27b
    }
}

extension Array where Element == RecommendedModelPick {
    /// Split a family catalog into what this Mac's RAM covers and what it
    /// doesn't, preserving each side's relative (ascending-size) order. The
    /// Recommended pane shows the first inline and tucks the second behind a
    /// "Requires more RAM" disclosure — nothing is ever dropped, just
    /// deferred until the user asks to see it.
    func partitionedByRequirements(physicalMemoryBytes: UInt64) -> (fits: [RecommendedModelPick], requiresMoreRAM: [RecommendedModelPick]) {
        var fits: [RecommendedModelPick] = []
        var requiresMoreRAM: [RecommendedModelPick] = []
        for pick in self {
            if pick.meetsSystemRequirements(physicalMemoryBytes: physicalMemoryBytes) {
                fits.append(pick)
            } else {
                requiresMoreRAM.append(pick)
            }
        }
        return (fits, requiresMoreRAM)
    }
}
