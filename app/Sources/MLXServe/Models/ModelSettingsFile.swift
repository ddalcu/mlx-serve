import Foundation

/// KV-cache width vocabulary of the server's `kv_quant` field.
enum KvQuantChoice: String, CaseIterable {
    case off
    case bits4 = "4"
    case bits8 = "8"

    var label: String {
        switch self {
        case .off: "Off (bf16)"
        case .bits4: "4-bit"
        case .bits8: "8-bit"
        }
    }
}

/// MTP draft-acceptance vocabulary of the server's `mtp_acceptance` field.
/// Typical and TokenV3 accept more drafts but change the output distribution.
enum MtpAcceptanceChoice: String, CaseIterable {
    case exact
    case typical
    case tokenv3

    var label: String {
        switch self {
        case .exact: "Exact (Default)"
        case .typical: "Typical (faster, lossy)"
        case .tokenv3: "TokenV3 (fastest, very lossy)"
        }
    }
}

/// One model's entry in `~/.mlx-serve/model-settings.json` — the file the
/// SERVER reads (`src/host/registry/model_settings.zig`) at every load of that model, so the
/// keys are its keys. nil = the process default. Unknown keys are kept in
/// `extra` so a future server field survives an app-side edit.
struct ModelOverride: Equatable {
    /// `alias`: a short request name for the model, read by the server per
    /// request (`resolveRequestModelId`), so it needs no reload.
    var alias: String?
    var ctxSize: Int?
    var kvQuant: KvQuantChoice?
    var mtp: Bool?
    var mtpAcceptance: MtpAcceptanceChoice?
    /// `mtp_greedy_tail`: sampled requests draft only the first MTP token from the sampler, later ones by argmax.
    var mtpGreedyTail: Bool?
    /// `drafter`: "off", "auto" or a drafter dir (the speculation socket, `DrafterSocket`).
    var drafter: String?
    /// `int8_prefill`: LOSSY int8-activation prefill (2-bit Prism packs only).
    var int8Prefill: Bool?
    /// `ane`: serve the model on the Neural Engine instead of the GPU. Read for
    /// files written before `engine`; the engine picker clears it.
    var ane: Bool?
    /// `engine`: what serves the model ("mlx", "ane", "ds4", "llama"; the row's
    /// `engines` lists the choices). nil leaves it to the server.
    var engine: String?
    /// `chat_template_kwargs`: variables handed to the model's Jinja template
    /// verbatim (`TemplateKwargs` types and names them).
    var templateKwargs: [String: Any] = [:]
    /// `config_overrides`: a JSON object the server deep-merges into the pack's `config.json` at load — this
    /// model's share of `--config-overrides`, the way to state a rope scaling the checkpoint does not carry.
    /// Held as the text the sheet edits, canonicalized by `ConfigOverrides`.
    var configOverrides: String?
    var generationDefaults = GenerationDefaults()
    var extra: [String: Any] = [:]

    init(ctxSize: Int? = nil, kvQuant: KvQuantChoice? = nil, mtp: Bool? = nil,
         mtpAcceptance: MtpAcceptanceChoice? = nil, mtpGreedyTail: Bool? = nil, int8Prefill: Bool? = nil, templateKwargs: [String: Any] = [:],
         configOverrides: String? = nil) {
        self.ctxSize = ctxSize
        self.kvQuant = kvQuant
        self.mtp = mtp
        self.mtpAcceptance = mtpAcceptance
        self.mtpGreedyTail = mtpGreedyTail
        self.int8Prefill = int8Prefill
        self.templateKwargs = templateKwargs
        self.configOverrides = configOverrides
    }

    init(json: [String: Any]) {
        var rest = json
        if let a = rest.removeValue(forKey: "alias") { alias = a as? String }
        if let c = rest.removeValue(forKey: "ctx_size") {
            if let n = c as? Int, n > 0 { ctxSize = n }
        }
        if let k = rest.removeValue(forKey: "kv_quant") {
            if let s = k as? String { kvQuant = KvQuantChoice(rawValue: s) }
            else if let n = k as? Int { kvQuant = KvQuantChoice(rawValue: n == 0 ? "off" : String(n)) }
        }
        if let m = rest.removeValue(forKey: "mtp") {
            if let b = m as? Bool { mtp = b }
        }
        if let a = rest.removeValue(forKey: "mtp_acceptance") {
            if let s = a as? String { mtpAcceptance = MtpAcceptanceChoice(rawValue: s) }
        }
        if let g = rest.removeValue(forKey: "mtp_greedy_tail") {
            if let b = g as? Bool { mtpGreedyTail = b }
        }
        if let d = rest.removeValue(forKey: "drafter") { drafter = d as? String }
        if let i = rest.removeValue(forKey: "int8_prefill") {
            if let b = i as? Bool { int8Prefill = b }
        }
        if let a = rest.removeValue(forKey: "ane") {
            if let b = a as? Bool { ane = b }
        }
        if let e = rest.removeValue(forKey: "engine") { engine = e as? String }
        if let kw = rest.removeValue(forKey: "chat_template_kwargs") as? [String: Any] { templateKwargs = kw }
        // Only an object is the server's key; another shape stays in `extra` and survives untouched.
        if let ov = rest["config_overrides"] as? [String: Any] {
            let text = ConfigOverrides.text(ov)
            if !text.isEmpty { configOverrides = text }
            rest.removeValue(forKey: "config_overrides")
        }
        if let profile = rest["generation_defaults"] as? [String: Any],
           let data = try? JSONSerialization.data(withJSONObject: profile),
           let parsed = try? JSONDecoder().decode(GenerationDefaults.self, from: data) {
            generationDefaults = parsed
            rest.removeValue(forKey: "generation_defaults")
        }
        extra = rest
    }

    var isEmpty: Bool { !hasSettings && extra.isEmpty }
    /// True when any field the sheet edits is set.
    var hasSettings: Bool {
        alias != nil || ctxSize != nil || kvQuant != nil || mtp != nil || mtpAcceptance != nil || mtpGreedyTail != nil || drafter != nil || int8Prefill != nil || ane != nil || engine != nil
            || !templateKwargs.isEmpty || configOverrides != nil || !generationDefaults.rules.isEmpty
    }
    var sortedKwargKeys: [String] { templateKwargs.keys.sorted() }

    var json: [String: Any] {
        var out = extra
        if let alias { out["alias"] = alias }
        if let ctxSize { out["ctx_size"] = ctxSize }
        if let kvQuant { out["kv_quant"] = kvQuant.rawValue }
        if let mtp { out["mtp"] = mtp }
        if let mtpAcceptance { out["mtp_acceptance"] = mtpAcceptance.rawValue }
        if let mtpGreedyTail { out["mtp_greedy_tail"] = mtpGreedyTail }
        if let drafter { out["drafter"] = drafter }
        if let int8Prefill { out["int8_prefill"] = int8Prefill }
        if let ane { out["ane"] = ane }
        if let engine { out["engine"] = engine }
        if !templateKwargs.isEmpty { out["chat_template_kwargs"] = templateKwargs }
        if let c = configOverrides, let o = ConfigOverrides.object(c) { out["config_overrides"] = o }
        if !generationDefaults.rules.isEmpty { out["generation_defaults"] = generationDefaults.json }
        return out
    }

    /// The engine this entry asks for: `engine`, else the older `ane` switch.
    var engineChoice: String? {
        get { engine ?? ane.map { $0 ? "ane" : "mlx" } }
        set {
            engine = newValue
            ane = nil
        }
    }

    /// Mirrors the server's `validAlias`: `@` routes to a peer or provider, `/`
    /// reads as an org/repo id, and "mlx-serve" names the default model.
    static func isValidAlias(_ name: String) -> Bool {
        guard !name.isEmpty, name.utf8.count <= 128, name != "mlx-serve" else { return false }
        return name.utf8.allSatisfy { $0 > 0x20 && $0 != 0x7f && !"@/\"\\".utf8.contains($0) }
    }

    /// True when a field the server reads at LOAD differs; the alias does not count.
    func changesLoad(from old: ModelOverride) -> Bool {
        var a = self, b = old
        a.alias = nil
        b.alias = nil
        a.generationDefaults = .init()
        b.generationDefaults = .init()
        return a != b
    }

    static func == (a: ModelOverride, b: ModelOverride) -> Bool {
        a.alias == b.alias && a.ctxSize == b.ctxSize && a.kvQuant == b.kvQuant && a.mtp == b.mtp && a.mtpAcceptance == b.mtpAcceptance && a.mtpGreedyTail == b.mtpGreedyTail && a.drafter == b.drafter
            && a.int8Prefill == b.int8Prefill && a.ane == b.ane && a.engine == b.engine && a.generationDefaults == b.generationDefaults
            && a.configOverrides == b.configOverrides
            && NSDictionary(dictionary: a.templateKwargs).isEqual(to: b.templateKwargs)
            && NSDictionary(dictionary: a.extra).isEqual(to: b.extra)
    }
}

/// The whole file, keyed by model path (dir, or the `.gguf` file), trailing `/` trimmed.
struct ModelSettingsFile {
    static let defaultPath = NSString(string: "~/.mlx-serve/model-settings.json").expandingTildeInPath

    private(set) var entries: [String: ModelOverride] = [:]

    init() {}

    var isEmpty: Bool { entries.isEmpty }

    static func key(_ path: String) -> String {
        var p = path
        while p.count > 1 && p.hasSuffix("/") { p.removeLast() }
        return p
    }

    func override(for path: String) -> ModelOverride? {
        entries[Self.key(path)]
    }

    /// Another model that already answers to `alias`, if any.
    func pathUsingAlias(_ alias: String, except path: String) -> String? {
        entries.first { $0.key != Self.key(path) && $0.value.alias == alias }?.key
    }

    /// Replaces the edited fields; keys the app does not know stay.
    mutating func set(_ o: ModelOverride, for path: String) {
        var merged = o
        if let old = entries[Self.key(path)] { merged.extra.merge(old.extra) { mine, _ in mine } }
        if merged.isEmpty { entries.removeValue(forKey: Self.key(path)) } else { entries[Self.key(path)] = merged }
    }

    /// Missing or malformed file = empty, same as the server.
    static func load(path: String = defaultPath) -> ModelSettingsFile {
        var file = ModelSettingsFile()
        guard let data = FileManager.default.contents(atPath: path),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return file }
        for (k, v) in root {
            guard let obj = v as? [String: Any] else { continue }
            let o = ModelOverride(json: obj)
            if !o.isEmpty { file.entries[key(k)] = o }
        }
        return file
    }

    func save(path: String = defaultPath) throws {
        let dir = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let root = entries.mapValues { $0.json }
        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }
}

/// The per-model `config_overrides` field: a JSON object in the server's `--config-overrides` grammar,
/// deep-merged into the pack's `config.json` at load. The sheet edits text, the file holds an object,
/// and `text` is canonical so two entries that mean the same compare equal.
enum ConfigOverrides {
    /// What the box's text means for the stored key.
    enum Edit: Equatable {
        case clear
        case set(String)
        /// Half-typed text: keep what is stored and block the save.
        case keep
    }

    /// The typed text as the object to store; nil when it is empty, names nothing, or is not a JSON object.
    static func object(_ text: String) -> [String: Any]? {
        guard let dict = parsed(text), !dict.isEmpty else { return nil }
        return dict
    }

    /// Whether the box may hold the text: any JSON object, an empty one included — it names nothing, which is a
    /// cleared setting rather than a mistake, so it must never block the save.
    static func isObject(_ text: String) -> Bool { parsed(text) != nil }

    /// The one decision the sheet makes with the text, so clearing it is testable.
    static func edit(_ text: String) -> Edit {
        if let dict = parsed(text) { return dict.isEmpty ? .clear : .set(ConfigOverrides.text(dict)) }
        return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .clear : .keep
    }

    /// Why the text cannot be stored; nil when it is empty or a JSON object.
    static func rejection(_ text: String) -> String? {
        edit(text) != .keep ? nil
            : "Config overrides must be one JSON object, like {\"text_config\": {\"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4}}}."
    }

    private static func parsed(_ text: String) -> [String: Any]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) else { return nil }
        return parsed as? [String: Any]
    }

    /// The stored object as one line of JSON for the box; "" when it names nothing.
    static func text(_ object: [String: Any]) -> String {
        guard !object.isEmpty, let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
              let s = String(data: data, encoding: .utf8) else { return "" }
        return s
    }
}

/// Where the overflow card's "increase context" button goes.
enum ContextIncreaseTarget: Equatable {
    case modelSettings
    case appSettings

    static func resolve(hasOverride: Bool) -> ContextIncreaseTarget {
        hasOverride ? .modelSettings : .appSettings
    }
}
