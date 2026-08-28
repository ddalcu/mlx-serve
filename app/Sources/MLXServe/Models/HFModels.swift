import Foundation

/// Pipeline tags that mlx-serve can run (text LLMs, vision-language models,
/// and encoder-only embedding models).
private let compatiblePipelineTags: Set<String> = [
    "text-generation",
    "image-text-to-text",
    "any-to-any",  // Gemma 4 uses this
    // Encoder-only embedding repos (BERT family) — served via /v1/embeddings
    // and used by the app's GPU folder indexing. HF tags them either way.
    "feature-extraction",
    "sentence-similarity",
]

/// Architecture tag prefixes from HuggingFace that correspond to supported model families.
/// A model is supported if any of its tags starts with one of these prefixes.
/// HF tags vary (gemma3, gemma4, gemma3n, qwen3_5, qwen3.5, etc.) so prefix matching is needed.
private let supportedArchitectureTagPrefixes: [String] = [
    "gemma",      // gemma, gemma2, gemma3, gemma3n, gemma4
    "qwen",       // qwen2, qwen3, qwen3_5, qwen3.5, qwen3.6
    "llama",
    "mistral",
    "nemotron",   // nemotron_h (Mamba2 SSM hybrid)
    "lfm",        // lfm2, lfm2_vl (Liquid state-space hybrid; VL serves images)
    "bert",       // encoder-only embedding models (/v1/embeddings)
    "hunyuan",    // Tencent Hunyuan 3 (model_type hy_v3) — repos tag "hunyuan"
    "hy_v3",      // …and/or the model_type verbatim (some tag only that)
    "laguna",     // poolside Laguna S 2.1 — repos tag "laguna" and/or "laguna-s-2.1"
    "inkling",    // Thinking Machines Inkling Small — repos tag "inkling" and/or "inkling_mm_model"
    "muse",       // meta-models Muse-Glimmer-30B — repos tag "muse_glimmer" (and *_text)
    "bailing",    // inclusionAI Ling 3.0 — repos tag "bailing_moe_v3" and/or "bailing_hybrid"
    "ling",       // …and/or the family name
    "gpt_oss",    // GPT-OSS family tags (e.g. mlx-community/gpt-oss-20b-MXFP4-Q8)
]

/// model_type values from config.json that the Zig server can load.
/// MLX-format DeepSeek-V4 is intentionally NOT in this set — DSV4 is served
/// by the embedded ds4 engine, which loads the GGUF checkpoint directly.
let supportedModelTypes: Set<String> = [
    "gemma3", "gemma3_text", "gemma4", "gemma4_text",
    "gemma4_unified", "gemma4_unified_text",
    "diffusion_gemma", // DiffusionGemma block diffusion (text-only; vision tower not wired)
    "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next",
    "qwen3_moe", "qwen3_moe_text",
    "qwen4_exp", "qwen4_exp_text", // Qwen3.8-Flash-Next (GDN + QSA + n-gram PLE MoE)
    "qwen2",
    "llama", "mistral",
    "lfm2", "lfm2_vl", // Liquid LFM2.5; the VL tag adds a SigLIP2-NaFlex tower
    "nemotron_h",
    "hy_v3", // Tencent Hunyuan 3 (295B-A21B MoE)
    "laguna", // poolside Laguna S 2.1 (117.6B-A8.5B MoE coder)
    "inkling_mm_model", // Thinking Machines Inkling Small (276B-A12B MoE)
    "muse_glimmer", "muse_glimmer_text", // meta-models Muse-Glimmer-30B (text + vision)
    "bailing_hybrid", // inclusionAI Ling 3.0 (KDA + MLA hybrid MoE)
    "bert", // encoder-only; serves /v1/embeddings (GPU document indexing)
    // GGUF engines: "gguf" = any model via the embedded llama.cpp engine;
    // "deepseek_v4" = DeepSeek-V4-Flash via the ds4 engine. Both are served, so
    // neither should be flagged "unsupported architecture" in the model browser.
    "gguf", "deepseek_v4",
    "gpt_oss",
]

/// model_type prefixes/exact values for native media-generation checkpoints
/// (image/audio/video/3D/music) served by the Zig gen engines, not the chat
/// transformer. These are real, loadable downloads — just not chat models —
/// so they must pass the architecture gate (not show "Unsupported
/// architecture" in the Downloaded tab) while still being excluded from
/// chat-model pickers (`LocalModel.isChatPickable` checks this separately).
/// Mirrors `model_discovery.isMediaModelType` (Zig).
private let mediaModelTypePrefixes: [String] = ["flux2", "krea", "mage_flow", "hunyuan3d"]
// Mirrors `gen.media_model_types` on the server. Drift here is not cosmetic:
// a media model missing from this list fails the ARCHITECTURE gate outright,
// so the browser draws it a red "Unsupported" — which is what happened to
// `minimax_music3` on the same screen that offers to download it, and to bare
// `mageflow` (MageFlow ships no root config.json, so it is classified from
// model_index.json under that spelling). `kokoro` was excluded on purpose
// because it is voice-mode-only and absent from every media picker — but
// "hidden from pickers" and "shown to the user as an unsupported
// architecture" are different decisions, and it was getting both. Exclusion
// from pickers is `isChatPickable`'s job, which reads this same list. #228.
//
// `MediaModalityParityTests` reads `media_model_types` out of src/gen.zig and
// asserts this agrees. Zig already pinned its own two copies against each
// other; nothing pinned Swift, which is why this drifted unnoticed.
private let mediaModelTypeExactValues: Set<String> = [
    "qwen3_tts", "AudioVideo", "acestep", "minimax_h3", "minimax_music3", "kokoro", "mageflow",
]

func isMediaModelType(_ modelType: String) -> Bool {
    if mediaModelTypeExactValues.contains(modelType) { return true }
    return mediaModelTypePrefixes.contains { modelType.hasPrefix($0) }
}

/// Media architectures a **Discover search row** may offer as a download.
///
/// A narrower question than `isMediaModelType`, and the two were conflated.
/// "Is this a media architecture?" decides the Unsupported badge and the
/// chat-picker exclusion — Kokoro is one, and pretending otherwise is what put
/// a red flag on it. "Should a search result offer this repo?" is a CATALOG
/// decision, and Kokoro is answered by the built-in voice-mode entry rather
/// than by arbitrary repos found on the hub. Kokoro still appears in the Media
/// pane's Audio section via `AudioModelPreset.allIncludingVoiceOnly`, so this
/// hides nothing the user needs; it only keeps the Discover door shut.
func discoverableMediaModelType(_ modelType: String) -> Bool {
    modelType != "kokoro" && isMediaModelType(modelType)
}

/// Which create pane a media checkpoint belongs to.
///
/// Mirrors Zig's `gen.modalityFromType` with ONE deliberate difference: Zig
/// collapses speech and music into `.audio` because they share an engine slot,
/// but the app has to tell them apart — they are two tabs of one pane, so a
/// Use button that only knew "audio" would drop a music model on the Voice
/// tab. `CustomMediaModels.musicFamily` is the existing discriminator; this is
/// the same split, minus the running-server requirement (a browser row is a
/// filesystem `LocalModel` with no capabilities).
enum MediaModality: CaseIterable {
    case image, voice, music, video, mesh

    init?(modelType: String) {
        if modelType.hasPrefix("flux2") || modelType.hasPrefix("krea")
            || modelType.hasPrefix("mage_flow") || modelType == "mageflow" { self = .image; return }
        if modelType.hasPrefix("hunyuan3d") { self = .mesh; return }
        switch modelType {
        case "qwen3_tts", "kokoro": self = .voice
        case "acestep", "minimax_music3": self = .music
        case "AudioVideo", "minimax_h3": self = .video
        default: return nil
        }
    }

    /// The top-level create page. Music has no `GenExperiment` case of its own
    /// — it is a tab inside Audio — so it shares `.audio` with voice.
    var experiment: GenExperiment {
        switch self {
        case .image: return .image
        case .voice, .music: return .audio
        case .video: return .video
        case .mesh: return .model3d
        }
    }

    /// Which tab of the Audio pane, or nil where the question does not apply.
    var audioTab: AudioGenView.Tab? {
        switch self {
        case .voice: return .voice
        case .music: return .music
        default: return nil
        }
    }

    /// What the Use button says it will open.
    var paneName: String {
        switch self {
        case .image: return "Image Generation"
        case .voice: return "Voice"
        case .music: return "Music"
        case .video: return "Video Generation"
        case .mesh: return "3D"
        }
    }
}

/// The `config` block the HF API returns when asked (`expand[]=config`).
/// `model_type` is the field the server dispatches on, so it's the definitive
/// arch signal a search row can carry. A diffusers repo has no `model_type`;
/// HF surfaces the pipeline's own class instead
/// (`{"diffusers":{"_class_name":"MageFlowPipeline"}}`).
struct HFConfigMeta: Codable {
    let modelType: String?
    var diffusers: DiffusersMeta? = nil
    var diffusersClassName: String? { diffusers?.className }

    struct DiffusersMeta: Codable {
        let className: String?
        enum CodingKeys: String, CodingKey { case className = "_class_name" }
    }
    enum CodingKeys: String, CodingKey { case modelType = "model_type", diffusers }
}

/// Served media families recognizable from a diffusers repo's `_class_name`.
/// Only pipelines our engines load in that repo's OWN layout belong here —
/// the class is a family claim, and the tree check still has to prove the
/// files. Krea and FLUX conversions ship a root config.json with `model_type`
/// (they come through the main door); their raw upstream layouts are not
/// served, so their classes deliberately map nowhere.
private let servedDiffusersClasses: [String: String] = [
    "MageFlowPipeline": "mage_flow", // mirrors model_discovery.peekMageFlowIndex
]

struct HFModel: Identifiable, Codable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let lastModified: String?
    let tags: [String]?
    let safetensors: HFSafetensors?
    let pipelineTag: String?
    var config: HFConfigMeta? = nil

    /// Whether the repo's file TREE proved the family bundle's ready markers
    /// (set by `HFSearchService` from the tree fetch; nil = not checked yet
    /// or unfetchable). Media repos are judged by this, never by tags alone —
    /// a raw upstream checkpoint declares the same `model_type` as a
    /// converted pack, and only the converted layout loads.
    var mediaStructureVerified: Bool? = nil

    /// Fallback file size from tree API (not from JSON — set by HFSearchService).
    /// For safetensors repos this is the sum across all `.safetensors` shards.
    /// For single-file GGUF repos this is the size of that one `.gguf`. For
    /// multi-quant GGUF repos this is the LARGEST quant's size — keeps the
    /// `estimatedSizeBytes` sort + fitness-coloring conservative-high while
    /// `ramEstimate` separately surfaces the min/max as a range string.
    var fallbackSizeBytes: Int64? = nil

    /// Smallest non-mmproj GGUF in the repo (bytes). Only populated by
    /// `HFSearchService` for GGUF repos that ship multiple quants (e.g.
    /// `unsloth/gemma-4-E4B-it-GGUF` with Q2_K through BF16). Paired with
    /// `ggufMaxSizeBytes` to drive the `ramEstimate` range string.
    var ggufMinSizeBytes: Int64? = nil
    /// Largest non-mmproj GGUF in the repo (bytes). See `ggufMinSizeBytes`.
    var ggufMaxSizeBytes: Int64? = nil

    /// The per-quant subfolders a multi-variant MLX repo publishes, smallest
    /// first — empty for every ordinary repo. Populated by `HFSearchService`
    /// from the tree fetch it already runs for the size column (such a repo
    /// carries no `safetensors` metadata: HF computes that from a ROOT index,
    /// which a subfoldered repo doesn't have, so the fetch always happens).
    var mlxVariants: [MlxVariant] = []

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes, lastModified, tags, safetensors, config
        case pipelineTag = "pipeline_tag"
    }

    /// The served media family this repo declares — `config.model_type`
    /// first (definitive), else a served diffusers `_class_name` (Mage-Flow —
    /// diffusers repos carry no model_type), else a tag spelling a media
    /// model_type verbatim (mlx-serve packs tag it, e.g. "minimax_h3").
    /// nil = not a media repo we serve; Kokoro deliberately maps nowhere
    /// (voice-mode-only catalog rule — see `discoverableMediaModelType`).
    var mediaFamilyModelType: String? {
        if let t = config?.modelType, discoverableMediaModelType(t) { return t }
        if let c = config?.diffusersClassName, let t = servedDiffusersClasses[c] { return t }
        return (tags ?? []).first(where: discoverableMediaModelType)
    }

    var isServedMediaRepo: Bool { mediaFamilyModelType != nil }

    /// Whether mlx-serve supports this model's pipeline type.
    /// Models with unknown pipeline (nil/empty) get benefit of the doubt.
    ///
    /// A served media repo is judged by its VERIFIED structure instead —
    /// its pipeline tag ("text-to-video") would fail the chat allowlist, and
    /// whitelisting media pipeline tags broadly would hand a Download button
    /// to every diffusers repo on HF. nil (tree not checked / unfetchable)
    /// stays conservatively incompatible.
    var isCompatible: Bool {
        if isServedMediaRepo { return mediaStructureVerified == true }
        guard let tag = pipelineTag, !tag.isEmpty else { return true }
        return compatiblePipelineTags.contains(tag)
    }

    /// Whether this model's architecture is known to work with mlx-serve.
    /// Checks HF tags for supported family prefixes (gemma, qwen, llama, mistral).
    /// Models with no tags get benefit of the doubt.
    ///
    /// GGUF repos get a permissive pass: many LM Studio community quant
    /// repacks tag themselves only with `gguf` / `llama-cpp` / `base_model:...`
    /// and never inherit the upstream family tag (e.g. `lmstudio-community/
    /// gemma-4-E4B-it-GGUF` carries no `gemma*` tag of its own). The embedded
    /// llama.cpp engine handles whatever architecture the .gguf declares
    /// internally, so flagging these "Unsupported architecture" was a false
    /// negative that hid legit downloads.
    var isSupportedArchitecture: Bool {
        if isGgufRepo || isServedMediaRepo { return true }
        guard let tags, !tags.isEmpty else { return true }
        return tags.contains { tag in
            supportedArchitectureTagPrefixes.contains { tag.hasPrefix($0) }
        }
    }

    /// True when this repo ships GGUF artifacts (HF tags it `gguf`). These download
    /// via the single-file GGUF path — the user picks a quant from the repo's
    /// `.gguf` list — and the server serves them through the embedded llama.cpp
    /// engine (or ds4 for DeepSeek-V4-Flash). Distinct from MLX/safetensors repos,
    /// which download the whole weight tree.
    var isGgufRepo: Bool {
        (tags ?? []).contains { $0.lowercased() == "gguf" }
    }

    /// True when this repo ships one complete MLX model PER SUBFOLDER — a shelf
    /// of quants, like a GGUF repo. The row offers a quant menu instead of a
    /// plain Download, and each pick lands in its own model dir.
    var isMlxVariantRepo: Bool { !mlxVariants.isEmpty }

    /// The FP quantization this repo uses that the server CANNOT load, if any.
    /// Since v26.6.10 (issue #24) the server loads nvfp4/mxfp4/mxfp8
    /// safetensors checkpoints natively (per-weight quant-mode resolution),
    /// so those are no longer flagged — they surface as a `quantization`
    /// badge instead. Only modes outside the server's set (`model.zig`
    /// parse: affine/nvfp4/mxfp4/mxfp8 — e.g. mxfp6) keep the flag, so the
    /// browser doesn't offer a download that errors at load. Name-based,
    /// matching the `quantization` idiom. GGUF repos are exempt: llama.cpp
    /// handles its own quant formats internally.
    var unsupportedQuantization: String? {
        if isGgufRepo { return nil }
        let lower = id.lowercased()
        for token in Self.unsupportedQuantizationTokens where lower.contains(token) {
            return token.uppercased()
        }
        return nil
    }

    private static let unsupportedQuantizationTokens: [String] = ["mxfp6"]

    /// Human-readable reason why this model isn't compatible.
    var incompatibleReason: String? {
        if isServedMediaRepo {
            return mediaStructureVerified == true ? nil
                : "Not an mlx-serve pack (missing converted files)"
        }
        if !isCompatible, let tag = pipelineTag {
            return "Not supported (\(tag))"
        }
        if !isSupportedArchitecture {
            return "Unsupported architecture"
        }
        if let quant = unsupportedQuantization {
            return "Unsupported quantization (\(quant))"
        }
        return nil
    }

    /// Whether this model supports vision (image input).
    var hasVision: Bool {
        let tag = pipelineTag ?? ""
        return tag == "image-text-to-text" || tag == "any-to-any"
    }

    /// Whether this model likely supports tool/function calling.
    /// Heuristic: instruction-tuned models from known families that implement tool call formats.
    var hasToolCalling: Bool { Self.likelyToolCalling(forName: id) }

    /// Name-based tool-calling heuristic, shared with `LocalModel` so the
    /// Downloaded tab and search rows agree. Instruction-tuned (`-it`/`-instruct`
    /// /`-chat`) checkpoints from families that implement a tool-call format.
    static func likelyToolCalling(forName name: String) -> Bool {
        let lower = name.lowercased()
        let isInstructTuned = lower.contains("-it") || lower.contains("-instruct") || lower.contains("-chat")
        guard isInstructTuned else { return false }
        let toolFamilies = ["gemma-4", "gemma-3", "qwen3", "qwen2.5", "llama-3", "mistral"]
        return toolFamilies.contains { lower.contains($0) }
    }

    /// Whether this repo is a Gemma 4 assistant drafter checkpoint. Drafters
    /// pair with a base Gemma 4 model via `--drafter <dir>`; loading one as a
    /// target on its own would fail, and they already have a dedicated home
    /// (the Drafters tab), so Discover hides them entirely.
    ///
    /// Search-result metadata carries no `model_type`, so this is name-based:
    /// any Gemma 4 repo whose name carries "assistant" as its own
    /// hyphen-delimited token. Drafters ship under more than one author
    /// (`mlx-community` for the community bf16/qat/quant builds, `google` for
    /// the official 12B upload) and in many quant/qat permutations
    /// (`-assistant-bf16`, `-qat-assistant-4bit`, `-qat-assistant-mxfp8`, …),
    /// so this checks the NAME SHAPE rather than a fixed list of exact
    /// strings — the previous regex only matched the original bf16-only,
    /// four-size-only naming and missed every qat/quant/12B variant
    /// mlx-community has since published.
    var isDrafter: Bool {
        let tokens = modelName.lowercased().split(separator: "-").map(String.init)
        return tokens.contains("gemma") && tokens.contains("4") && tokens.contains("assistant")
    }

    var author: String {
        id.split(separator: "/").first.map(String.init) ?? ""
    }

    var modelName: String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    /// Human label for the badge column, parsed from the repo id. Generalized
    /// over bit width (any N — 2/3/4/5/6/8/9/…, incl. fractional like "3.5bit")
    /// rather than a hardcoded set, covering both MLX "Nbit"/"N-bit" and GGUF
    /// "qN_"/"iqN_" naming, plus the FP weight dtypes and the server-loadable
    /// FP quant modes (NVFP4/MXFP4/MXFP8, served since v26.6.10). Returns nil
    /// when the id encodes no single quant (e.g. a multi-quant GGUF repo,
    /// where the user picks the quant at download time). Still-unloadable
    /// modes (mxfp6) deliberately don't match here — they surface via
    /// `incompatibleReason`, not as a misleading badge.
    var quantization: String? {
        // GGUF repos host multiple quants in separate files, multi-variant MLX
        // repos in separate subfolders; the repo ID alone doesn't identify one,
        // so surface "Multi" rather than "—".
        if isGgufRepo || isMlxVariantRepo { return Self.quantizationLabel(forId: id) ?? "Multi" }
        return Self.quantizationLabel(forId: id)
    }

    static func quantizationLabel(forId id: String) -> String? {
        let lower = id.lowercased()
        // FP quant modes before the bit-width regexes so a hypothetical
        // "…-nvfp4-4bit" id badges the format, not a bare width.
        for mode in ["nvfp4", "mxfp4", "mxfp8"] where lower.contains(mode) {
            return mode.uppercased()
        }
        if lower.contains("fp16") { return "FP16" }
        if lower.contains("bf16") { return "BF16" }
        if let s = firstCapture(lower, quantBitRegex) { return "\(s)-bit" }
        if let s = firstCapture(lower, ggufQuantRegex) { return "\(s)-bit" }
        return nil
    }

    /// MLX-style width: digits (optionally fractional) immediately before an
    /// optional hyphen and "bit" — "4bit", "8-bit", "3.5bit".
    private static let quantBitRegex = try! NSRegularExpression(pattern: #"(\d+(?:\.\d+)?)-?bit"#)
    /// GGUF-style width: "qN_" / "iqN_" (e.g. "Q4_K_M", "IQ3_M").
    private static let ggufQuantRegex = try! NSRegularExpression(pattern: #"i?q(\d+)_"#)

    private static func firstCapture(_ s: String, _ re: NSRegularExpression) -> String? {
        guard let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
              let r = Range(m.range(at: 1), in: s) else { return nil }
        return String(s[r])
    }

    /// Estimated on-disk / in-memory size in bytes.
    /// Prefers safetensors parameter dtype math; falls back to tree API file sizes.
    var estimatedSizeBytes: Int64 {
        if let params = safetensors?.parameters, !params.isEmpty {
            var total: Int64 = 0
            for (dtype, count) in params {
                let bytesPerParam: Double
                switch dtype.uppercased() {
                case "F64": bytesPerParam = 8
                case "F32", "U32", "I32": bytesPerParam = 4
                case "F16", "BF16", "U16", "I16": bytesPerParam = 2
                case "I8", "U8": bytesPerParam = 1
                case let d where d.contains("4"): bytesPerParam = 0.5
                default: bytesPerParam = 2
                }
                total += Int64(Double(count) * bytesPerParam)
            }
            return total
        }
        return fallbackSizeBytes ?? 0
    }

    /// Model size parsed from the name (e.g. "31B", "82M", "0.6B").
    /// More reliable than safetensors.total for quantized models, where total
    /// counts packed tensor values rather than actual parameters.
    var modelSize: String {
        Self.paramSizeLabel(forName: modelName) ?? "\u{2014}"
    }

    /// Parameter-count token parsed from a model name (e.g. "31B", "82M",
    /// "0.6B"), or nil when the name encodes none. Shared by `HFModel.modelSize`
    /// and `LocalModel` so the Downloaded tab labels match the search rows.
    static func paramSizeLabel(forName name: String) -> String? {
        // Match patterns like "31b", "E2B", "82M", "1.2B", "0.6b" preceded by - or _
        // Negative lookahead excludes "bit" (4bit, 8bit)
        let pattern = #"(?:^|[-_])[Ee]?(\d+(?:\.\d+)?[BbMm])(?![Ii])"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
              let range = Range(match.range(at: 1), in: name) else {
            return nil
        }
        return String(name[range]).uppercased()
    }

    /// RAM estimate including ~20% overhead for KV cache and runtime buffers.
    /// For multi-quant GGUF repos this returns a range string (e.g.
    /// "1.7–8.5 GB") spanning the smallest and largest quants — the user
    /// hasn't picked one yet at row-render time, so showing the range is
    /// more honest than a single number. Single-file repos (safetensors or
    /// single GGUF) get the existing single-value formatting.
    var ramEstimate: String {
        // A multi-variant MLX repo hasn't been narrowed to one quant either —
        // and its `estimatedSizeBytes` is 0 (no root safetensors metadata), so
        // without this it reads "Unknown".
        if let lo = mlxVariants.map(\.sizeBytes).min(), let hi = mlxVariants.map(\.sizeBytes).max(), lo < hi {
            return MemoryInfo.formatRange(Int64(Double(lo) * 1.2), Int64(Double(hi) * 1.2))
        }
        if let minB = ggufMinSizeBytes, let maxB = ggufMaxSizeBytes, minB < maxB {
            let lo = Int64(Double(minB) * 1.2)
            let hi = Int64(Double(maxB) * 1.2)
            return MemoryInfo.formatRange(lo, hi)
        }
        let weights = estimatedSizeBytes
        if weights == 0 { return "Unknown" }
        let withOverhead = Int64(Double(weights) * 1.2)
        return MemoryInfo.format(withOverhead)
    }

    /// Single-value RAM estimate in bytes — drives sort + fitness coloring,
    /// both of which need one number per row. For GGUF ranges we use the
    /// max (conservative-high) so a row showing "1.7–8.5 GB" on a 6 GB Mac
    /// colors as won't-fit rather than misleading-green on its smallest
    /// quant. Single-value paths are unchanged.
    var ramEstimateBytes: Int64 {
        if let maxV = mlxVariants.map(\.sizeBytes).max() {
            return Int64(Double(maxV) * 1.2)
        }
        if let maxB = ggufMaxSizeBytes {
            return Int64(Double(maxB) * 1.2)
        }
        let weights = estimatedSizeBytes
        if weights == 0 { return 0 }
        return Int64(Double(weights) * 1.2)
    }

    var lastModifiedDate: Date? {
        guard let lastModified else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: lastModified) ?? ISO8601DateFormatter().date(from: lastModified)
    }
}

struct HFSafetensors: Codable {
    let parameters: [String: Int64]?
    let total: Int64?
}

enum HFSortField: String {
    case downloads
    case likes
    case lastModified
    case estimatedSize
}

enum RAMFitness {
    case fits, tight, wontFit, unknown
}

func formatCount(_ n: Int) -> String {
    if n >= 1_000_000 { return String(format: "%.1fM", Double(n) / 1_000_000) }
    if n >= 1_000 { return String(format: "%.1fK", Double(n) / 1_000) }
    return "\(n)"
}

func formatRelativeDate(_ date: Date?) -> String {
    guard let date else { return "\u{2014}" }
    let interval = Date().timeIntervalSince(date)
    let minutes = Int(interval / 60)
    let hours = minutes / 60
    let days = hours / 24
    let months = days / 30
    let years = days / 365
    if years > 0 { return "\(years)y ago" }
    if months > 0 { return "\(months)mo ago" }
    if days > 0 { return "\(days)d ago" }
    if hours > 0 { return "\(hours)h ago" }
    if minutes > 0 { return "\(minutes)m ago" }
    return "just now"
}
