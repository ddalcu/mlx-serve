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
    "lfm",        // lfm2, lfm2-vl (Liquid state-space hybrid)
    "bert",       // encoder-only embedding models (/v1/embeddings)
    "hunyuan",    // Tencent Hunyuan 3 (model_type hy_v3) — repos tag "hunyuan"
    "hy_v3",      // …and/or the model_type verbatim (some tag only that)
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
    "qwen2",
    "llama", "mistral",
    "lfm2", "lfm2-vl",
    "nemotron_h",
    "hy_v3", // Tencent Hunyuan 3 (295B-A21B MoE)
    "bert", // encoder-only; serves /v1/embeddings (GPU document indexing)
    // GGUF engines: "gguf" = any model via the embedded llama.cpp engine;
    // "deepseek_v4" = DeepSeek-V4-Flash via the ds4 engine. Both are served, so
    // neither should be flagged "unsupported architecture" in the model browser.
    "gguf", "deepseek_v4",
]

/// model_type prefixes/exact values for native media-generation checkpoints
/// (image/audio/video/3D/music) served by the Zig gen engines, not the chat
/// transformer. These are real, loadable downloads — just not chat models —
/// so they must pass the architecture gate (not show "Unsupported
/// architecture" in the Downloaded tab) while still being excluded from
/// chat-model pickers (`LocalModel.isChatPickable` checks this separately).
/// Mirrors `model_discovery.isMediaModelType` (Zig).
private let mediaModelTypePrefixes: [String] = ["flux2", "krea", "hunyuan3d"]
private let mediaModelTypeExactValues: Set<String> = ["qwen3_tts", "AudioVideo", "acestep"]

func isMediaModelType(_ modelType: String) -> Bool {
    if mediaModelTypeExactValues.contains(modelType) { return true }
    return mediaModelTypePrefixes.contains { modelType.hasPrefix($0) }
}

struct HFModel: Identifiable, Codable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let lastModified: String?
    let tags: [String]?
    let safetensors: HFSafetensors?
    let pipelineTag: String?

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

    enum CodingKeys: String, CodingKey {
        case id, downloads, likes, lastModified, tags, safetensors
        case pipelineTag = "pipeline_tag"
    }

    /// Whether mlx-serve supports this model's pipeline type.
    /// Models with unknown pipeline (nil/empty) get benefit of the doubt.
    var isCompatible: Bool {
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
        if isGgufRepo { return true }
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
        // GGUF repos host multiple quants in separate files; the repo ID alone
        // doesn't identify one, so surface "Multi" rather than "—".
        if isGgufRepo { return Self.quantizationLabel(forId: id) ?? "Multi" }
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
