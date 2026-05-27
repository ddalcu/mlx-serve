import Foundation

/// Pipeline tags that mlx-serve can run (text LLMs and vision-language models).
private let compatiblePipelineTags: Set<String> = [
    "text-generation",
    "image-text-to-text",
    "any-to-any",  // Gemma 4 uses this
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
]

/// model_type values from config.json that the Zig server can load.
/// MLX-format DeepSeek-V4 is intentionally NOT in this set — DSV4 is served
/// by the embedded ds4 engine, which loads the GGUF checkpoint directly.
let supportedModelTypes: Set<String> = [
    "gemma3", "gemma4", "gemma4_text",
    "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next",
    "qwen2",
    "llama", "mistral",
    "lfm2", "lfm2-vl",
    "nemotron_h",
    // GGUF engines: "gguf" = any model via the embedded llama.cpp engine;
    // "deepseek_v4" = DeepSeek-V4-Flash via the ds4 engine. Both are served, so
    // neither should be flagged "unsupported architecture" in the model browser.
    "gguf", "deepseek_v4",
]

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

    /// Human-readable reason why this model isn't compatible.
    var incompatibleReason: String? {
        if !isCompatible, let tag = pipelineTag {
            return "Not supported (\(tag))"
        }
        if !isSupportedArchitecture {
            return "Unsupported architecture"
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
    var hasToolCalling: Bool {
        let lower = id.lowercased()
        let isInstructTuned = lower.contains("-it") || lower.contains("-instruct") || lower.contains("-chat")
        guard isInstructTuned else { return false }
        let toolFamilies = ["gemma-4", "gemma-3", "qwen3", "qwen2.5", "llama-3", "mistral"]
        return toolFamilies.contains { lower.contains($0) }
    }

    /// Whether this repo is a Gemma 4 assistant drafter checkpoint. Drafters
    /// pair with a base Gemma 4 model via `--drafter <dir>`; loading one as a
    /// target on its own would fail. Case-sensitive on the size designator
    /// since that's how `mlx-community` publishes them today (`E2B`, `E4B`,
    /// `26B-A4B`, `31B`).
    var isDrafter: Bool { Self.drafterRepoRegex.firstMatch(in: id, range: NSRange(id.startIndex..., in: id)) != nil }

    private static let drafterRepoRegex: NSRegularExpression = {
        let pattern = #"^mlx-community/gemma-4-(E2B|E4B|26B-A4B|31B)-it-assistant-bf16$"#
        return try! NSRegularExpression(pattern: pattern)
    }()

    var author: String {
        id.split(separator: "/").first.map(String.init) ?? ""
    }

    var modelName: String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    var quantization: String? {
        let lower = id.lowercased()
        if lower.contains("4bit") || lower.contains("4-bit") || lower.contains("q4_") { return "4-bit" }
        if lower.contains("3bit") || lower.contains("3-bit") { return "3-bit" }
        if lower.contains("6bit") || lower.contains("6-bit") { return "6-bit" }
        if lower.contains("8bit") || lower.contains("8-bit") || lower.contains("q8_") { return "8-bit" }
        if lower.contains("fp16") { return "FP16" }
        if lower.contains("bf16") { return "BF16" }
        return nil
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
        let name = modelName
        // Match patterns like "31b", "E2B", "82M", "1.2B", "0.6b" preceded by - or _
        // Negative lookahead excludes "bit" (4bit, 8bit)
        let pattern = #"(?:^|[-_])[Ee]?(\d+(?:\.\d+)?[BbMm])(?![Ii])"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: name, range: NSRange(name.startIndex..., in: name)),
              let range = Range(match.range(at: 1), in: name) else {
            return "\u{2014}"
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
