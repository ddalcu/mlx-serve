import Foundation

/// A readable name for a model, built from its repo id.
enum ModelDisplayName {

    /// Packaging detail. True of every model in the list, so it distinguishes
    /// nothing and only costs width.
    private static let noise: Set<String> = [
        "it", "instruct", "chat", "qat", "hf", "mlx", "community", "serve",
        "unquantized", "converted", "finetune", "ft",
    ]

    /// Quantization spellings that are ACRONYMS — they read as shouting when
    /// title-cased and as typos when lowercased.
    private static let acronyms: Set<String> = [
        "bf16", "fp16", "f16", "fp32", "fp8", "fp4", "int4", "int8",
        "nvfp4", "mxfp4", "mxfp8", "awq", "gptq", "gguf", "dwq", "moe", "vl",
        "tts", "stt", "vae", "lora", "api",
    ]

    /// Families whose casing is a name, not a capitalisation rule.
    private static let families: [String: String] = [
        "gemma": "Gemma", "qwen": "Qwen", "llama": "Llama", "mistral": "Mistral",
        "deepseek": "DeepSeek", "laguna": "Laguna", "kokoro": "Kokoro",
        "hunyuan": "Hunyuan", "nemotron": "Nemotron", "phi": "Phi",
        "lfm": "LFM", "flux": "FLUX", "krea": "Krea", "acestep": "ACE-Step",
        "minimax": "MiniMax", "inkling": "Inkling", "bge": "BGE",
        "mxbai": "MxBai", "embeddinggemma": "EmbeddingGemma",
    ]

    /// The readable name.
    ///
    /// - Parameter id: a repo id, a bare model name, or a LAN `id@peer`.
    static func pretty(_ id: String) -> String {
        // A LAN id names the Mac it lives on, and that is not part of the
        // model's name — it is re-attached after, so "@studio" can't be
        // title-cased into the middle of the words.
        let (base, peer) = splitPeer(id)
        guard let name = modelComponent(base), !name.isEmpty else { return id }

        var out: [String] = []
        for raw in name.split(whereSeparator: { $0 == "-" || $0 == "_" }) {
            let token = String(raw)
            // Recognition comes FIRST, splitting second: "bf16" is one word,
            // and a letters-then-digits split would render it "Bf 16".
            switch classify(token) {
            case .drop: continue
            case .use(let piece): out.append(piece)
            case .unknown:
                // Only now try a family glued to its version ("gemma4").
                let parts = splitLetterDigitRuns(token)
                for part in parts {
                    switch classify(part) {
                    case .drop: continue
                    case .use(let piece): out.append(piece)
                    case .unknown: out.append(capitalized(part))
                    }
                }
            }
        }
        guard !out.isEmpty else { return id }
        let pretty = out.joined(separator: " ")
        return peer.map { "\(pretty) @ \($0)" } ?? pretty
    }

    private enum Outcome {
        /// Carries no information — packaging detail true of every model here.
        case drop
        case use(String)
        /// Not a token this knows; the caller decides.
        case unknown
    }

    private static func classify(_ token: String) -> Outcome {
        let lower = token.lowercased()
        if lower.isEmpty || noise.contains(lower) { return .drop }
        if let family = families[lower] { return .use(family) }
        if acronyms.contains(lower) { return .use(lower.uppercased()) }
        // "4bit" → "4-bit": the reader's spelling, and it keeps the number
        // adjacent to what it measures.
        if lower.hasSuffix("bit"), let bits = Int(lower.dropLast(3)) { return .use("\(bits)-bit") }
        if let size = parameterCount(lower) { return .use(size) }
        // A bare version number: digits are already right in any case.
        if lower.allSatisfy({ $0.isNumber || $0 == "." }) { return .use(lower) }
        return .unknown
    }

    /// `12b`, `0.6b`, `82m`, `a3b` (a MoE's ACTIVE parameters) — all sizes.
    private static func parameterCount(_ token: String) -> String? {
        guard let unit = token.last, unit == "b" || unit == "m" else { return nil }
        var head = token.dropLast()
        guard !head.isEmpty else { return nil }
        // A MoE's active count is written "a3b" — part of its identity, not a
        // prefix to drop.
        let active = head.first == "a"
        if active { head = head.dropFirst() }
        guard !head.isEmpty, head.allSatisfy({ $0.isNumber || $0 == "." }) else { return nil }
        let rendered = "\(head)\(unit == "m" ? "M" : "b")"
        return active ? "A" + rendered : rendered
    }

    private static func capitalized(_ token: String) -> String {
        token.prefix(1).uppercased() + token.dropFirst().lowercased()
    }

    /// `gemma4` → `["gemma", "4"]`, so a family glued to its version still
    /// reads as two words. Letters-then-digits ONLY, and only after `classify`
    /// has declined the whole token: `bf16` is one word, and splitting it first
    /// renders "Bf 16".
    private static func splitLetterDigitRuns(_ token: String) -> [String] {
        let letters = token.prefix { $0.isLetter }
        let rest = token.dropFirst(letters.count)
        guard !letters.isEmpty, !rest.isEmpty,
              rest.allSatisfy({ $0.isNumber || $0 == "." }) else { return [token] }
        return [String(letters), String(rest)]
    }

    private static func splitPeer(_ id: String) -> (String, String?) {
        guard let at = id.lastIndex(of: "@") else { return (id, nil) }
        let peer = String(id[id.index(after: at)...])
        return peer.isEmpty ? (id, nil) : (String(id[id.startIndex..<at]), peer)
    }

    /// The model half of a repo id, or nil when there isn't one — a trailing
    /// slash has no name after it, and echoing the raw id beats title-casing
    /// the org into one.
    private static func modelComponent(_ id: String) -> String? {
        guard let slash = id.lastIndex(of: "/") else { return id }
        let tail = String(id[id.index(after: slash)...])
        return tail.isEmpty ? nil : tail
    }
}
