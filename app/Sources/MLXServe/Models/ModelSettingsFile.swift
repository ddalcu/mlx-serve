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
/// SERVER reads (`src/model_settings.zig`) at every load of that model, so the
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
    /// `drafter`: "off", "auto" or a drafter dir (the speculation socket, `DrafterSocket`).
    var drafter: String?
    /// `chat_template_kwargs`: variables handed to the model's Jinja template
    /// verbatim (`TemplateKwargs` types and names them).
    var templateKwargs: [String: Any] = [:]
    var extra: [String: Any] = [:]

    init(ctxSize: Int? = nil, kvQuant: KvQuantChoice? = nil, mtp: Bool? = nil,
         mtpAcceptance: MtpAcceptanceChoice? = nil, templateKwargs: [String: Any] = [:]) {
        self.ctxSize = ctxSize
        self.kvQuant = kvQuant
        self.mtp = mtp
        self.mtpAcceptance = mtpAcceptance
        self.templateKwargs = templateKwargs
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
        if let d = rest.removeValue(forKey: "drafter") { drafter = d as? String }
        if let kw = rest.removeValue(forKey: "chat_template_kwargs") as? [String: Any] { templateKwargs = kw }
        extra = rest
    }

    var isEmpty: Bool { !hasSettings && extra.isEmpty }
    /// True when any field the sheet edits is set.
    var hasSettings: Bool {
        alias != nil || ctxSize != nil || kvQuant != nil || mtp != nil || mtpAcceptance != nil || drafter != nil || !templateKwargs.isEmpty
    }
    var sortedKwargKeys: [String] { templateKwargs.keys.sorted() }

    var json: [String: Any] {
        var out = extra
        if let alias { out["alias"] = alias }
        if let ctxSize { out["ctx_size"] = ctxSize }
        if let kvQuant { out["kv_quant"] = kvQuant.rawValue }
        if let mtp { out["mtp"] = mtp }
        if let mtpAcceptance { out["mtp_acceptance"] = mtpAcceptance.rawValue }
        if let drafter { out["drafter"] = drafter }
        if !templateKwargs.isEmpty { out["chat_template_kwargs"] = templateKwargs }
        return out
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
        return a != b
    }

    static func == (a: ModelOverride, b: ModelOverride) -> Bool {
        a.alias == b.alias && a.ctxSize == b.ctxSize && a.kvQuant == b.kvQuant && a.mtp == b.mtp && a.mtpAcceptance == b.mtpAcceptance && a.drafter == b.drafter
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

/// Where the overflow card's "increase context" button goes.
enum ContextIncreaseTarget: Equatable {
    case modelSettings
    case appSettings

    static func resolve(hasOverride: Bool) -> ContextIncreaseTarget {
        hasOverride ? .modelSettings : .appSettings
    }
}
