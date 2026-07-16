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

/// Which curated section a pick belongs to. Gemma/Qwen are vendor families;
/// `largest` is a RAM tier — the biggest models this app runs (DeepSeek-V4-Flash
/// via ds4, Hunyuan 3), grouped by "needs a very large Mac" rather than vendor.
enum RecommendedModelFamily: String {
    case gemma = "Gemma"
    case qwen = "Qwen"
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
    /// Short capability chips (e.g. "Best for coding").
    let highlights: [String]
    /// Overrides the generic weights×1.2 RAM estimate for picks where that
    /// formula misleads (e.g. a build whose runtime footprint or context needs
    /// push the honest recommendation gate above what the on-disk size implies).
    /// Currently unused — every pick is well-modeled by the ×1.2 estimate.
    var ramOverrideGB: Double? = nil
    /// For a GGUF/ds4 pick: the specific `.gguf` file this recommendation
    /// downloads (the repo ships many; the curated pick names one known-good
    /// quant). nil for a safetensors pick, whose whole repo is fetched. When
    /// set, the pane downloads via the GGUF path (and auto-pulls the ds4 MTP
    /// draft head) instead of the safetensors-tree path.
    var ggufFilename: String? = nil

    var sizeLabel: String { String(format: "~%.1f GB", sizeGB) }

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
        highlights: ["Fast replies", "Everyday chat"]
    )

    static let gemmaE4B = RecommendedModelPick(
        id: "gemma-4-e4b",
        name: "Gemma 4 E4B",
        tagline: "The sweet spot",
        blurb: "A clear step up in quality from E2B — better at longer conversations, writing, and coding — while still replying quickly. A great all-around default if you're not sure what to pick.",
        repoId: "mlx-community/gemma-4-e4b-it-4bit",
        sizeGB: 4.8,
        family: .gemma,
        highlights: ["Balanced", "Coding help"]
    )

    static let gemma12B = RecommendedModelPick(
        id: "gemma-4-12b",
        name: "Gemma 4 12B",
        tagline: "Sharper reasoning",
        blurb: "Noticeably better at following detailed, multi-step instructions and reasoning through trickier problems than the smaller Gemma models, without needing a huge amount of memory.",
        repoId: "mlx-community/gemma-4-12b-it-4bit",
        sizeGB: 6.3,
        family: .gemma,
        highlights: ["Smarter", "Coding help"]
    )

    static let gemma26bA4b = RecommendedModelPick(
        id: "gemma-4-26b-a4b",
        name: "Gemma 4 26B-A4B",
        tagline: "Big, but efficient",
        blurb: "A much larger model that uses a trick called \u{201c}mixture of experts\u{201d} — for every word it only wakes up a small part of itself, so it answers faster than you'd expect for its size while giving noticeably better answers than the smaller Gemma models.",
        repoId: "mlx-community/gemma-4-26b-a4b-it-4bit",
        sizeGB: 14.3,
        family: .gemma,
        highlights: ["Strong reasoning", "Mixture of experts"]
    )

    static let gemma31B = RecommendedModelPick(
        id: "gemma-4-31b",
        name: "Gemma 4 31B",
        tagline: "Gemma's biggest all-rounder",
        blurb: "Gemma's largest single model — every part of it is used on every word, with no shortcuts. Excellent general reasoning, writing, and instruction-following.",
        repoId: "mlx-community/gemma-4-31b-it-4bit",
        sizeGB: 17.2,
        family: .gemma,
        highlights: ["Top quality", "Writing"]
    )

    static let gemma26bA4b8bit = RecommendedModelPick(
        id: "gemma-4-26b-a4b-8bit",
        name: "Gemma 4 26B-A4B (higher precision)",
        tagline: "Sharper version of the MoE model",
        blurb: "The same Gemma mixture-of-experts model as above, stored with twice the numeric precision (8-bit instead of 4-bit). That means slightly more accurate, consistent answers, at roughly double the memory.",
        repoId: "mlx-community/gemma-4-26b-a4b-it-8bit",
        sizeGB: 26.0,
        family: .gemma,
        highlights: ["Highest quality", "Mixture of experts"]
    )

    static let gemma31B8bit = RecommendedModelPick(
        id: "gemma-4-31b-8bit",
        name: "Gemma 4 31B (highest quality)",
        tagline: "The best this app offers",
        blurb: "The highest-quality model on this list: Gemma's largest model at full 8-bit precision. This is about as good as local answers get here — save it for when quality matters more than speed.",
        repoId: "mlx-community/gemma-4-31b-it-8bit",
        sizeGB: 31.5,
        family: .gemma,
        highlights: ["Highest quality", "Flagship"]
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
        highlights: ["Balanced", "Coding help"]
    )

    static let qwen36_27bMtp = RecommendedModelPick(
        id: "qwen36-27b-mtp",
        name: "Qwen 3.6 27B",
        tagline: "One of the strongest models here",
        blurb: "One of the most capable models this app can run — excellent at coding and at multi-step \u{201c}agent\u{201d} tasks like using tools and following a plan. It also ships with a built-in speed trick that lets it draft and double-check several words at once, so it feels noticeably faster than a plain model this size.",
        repoId: "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve",
        sizeGB: 15.0,
        family: .qwen,
        highlights: ["Best for coding", "Built-in speed boost", "Great at agent tasks"]
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
        highlights: ["Flagship quality", "Mixture of experts", "256K context"]
    )

    /// DeepSeek-V4-Flash via the embedded ds4 engine — a frontier-scale model
    /// on a very large Mac. The curated pick is the imatrix-calibrated IQ2XXS
    /// build (best quality-per-byte that fits a 96 GB Mac); the pane pulls the
    /// MTP draft head beside it so ds4 speculative decode is available.
    static let deepseekV4Flash = RecommendedModelPick(
        id: "deepseek-v4-flash",
        name: "DeepSeek-V4-Flash",
        tagline: "Frontier model, on your Mac",
        blurb: "A frontier-scale DeepSeek model that runs entirely on your Mac through the ds4 engine — top-tier reasoning, coding, and agent work. It wakes only a fraction of itself per word (mixture of experts) and ships a built-in draft head so it can predict several words at once for a faster decode. Needs a lot of memory — best on Macs with 96 GB or more.",
        repoId: "antirez/deepseek-v4-gguf",
        sizeGB: 86.7,
        family: .largest,
        highlights: ["Frontier quality", "Mixture of experts", "Built-in speed boost"],
        ramOverrideGB: 96.0,
        ggufFilename: "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
    )

    static let qwen36_35bA3b = RecommendedModelPick(
        id: "qwen36-35b-a3b",
        name: "Qwen 3.6 35B-A3B",
        tagline: "Qwen's largest model here",
        blurb: "Qwen's biggest model on this list — 35 billion parameters in total, but like the Gemma mixture-of-experts model above, it only activates a few billion per word, so it stays efficient. Excellent for demanding coding and reasoning work.",
        repoId: "mlx-community/Qwen3.6-35B-A3B-4bit",
        sizeGB: 19.0,
        family: .qwen,
        highlights: ["Strong reasoning", "Mixture of experts"]
    )
}

extension RecommendedModelPick {
    /// Gemma 4 picks, ascending by size — one of the Recommended pane's two
    /// family sections.
    static let gemmaCatalog: [RecommendedModelPick] = [
        .gemmaE2B, .gemmaE4B, .gemma12B, .gemma26bA4b, .gemma31B, .gemma26bA4b8bit, .gemma31B8bit,
    ]

    /// Qwen 3.5/3.6 picks, ascending by size — the Recommended pane's other
    /// family section.
    static let qwenCatalog: [RecommendedModelPick] = [
        .qwen35_9b, .qwen36_27bMtp, .qwen36_35bA3b,
    ]

    /// The largest models this app runs, ascending by on-disk size (the app's
    /// smallest-first convention) — Hunyuan 3 295B (the compact ~84 GB oQ2e
    /// build) then DeepSeek-V4-Flash (~87 GB ds4 GGUF). Grouped by "needs a
    /// very large Mac" rather than by vendor.
    static let largestCatalog: [RecommendedModelPick] = [
        .hy3_295b, .deepseekV4Flash,
    ]
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
