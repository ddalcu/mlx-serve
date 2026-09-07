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

/// One model's entry in `~/.mlx-serve/model-settings.json` — the file the
/// SERVER reads (`src/model_settings.zig`) at every load of that model, so the
/// keys are its keys. nil = the process default. Unknown keys are kept in
/// `extra` so a future server field survives an app-side edit.
struct ModelOverride: Equatable {
    var ctxSize: Int?
    var kvQuant: KvQuantChoice?
    var mtp: Bool?
    var extra: [String: Any] = [:]

    init(ctxSize: Int? = nil, kvQuant: KvQuantChoice? = nil, mtp: Bool? = nil) {
        self.ctxSize = ctxSize
        self.kvQuant = kvQuant
        self.mtp = mtp
    }

    init(json: [String: Any]) {
        var rest = json
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
        extra = rest
    }

    var isEmpty: Bool { ctxSize == nil && kvQuant == nil && mtp == nil && extra.isEmpty }
    /// True when any of the three fields the sheet edits is set.
    var hasSettings: Bool { ctxSize != nil || kvQuant != nil || mtp != nil }

    var json: [String: Any] {
        var out = extra
        if let ctxSize { out["ctx_size"] = ctxSize }
        if let kvQuant { out["kv_quant"] = kvQuant.rawValue }
        if let mtp { out["mtp"] = mtp }
        return out
    }

    static func == (a: ModelOverride, b: ModelOverride) -> Bool {
        a.ctxSize == b.ctxSize && a.kvQuant == b.kvQuant && a.mtp == b.mtp
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

    /// Replaces the three edited fields; keys the app does not know stay.
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
