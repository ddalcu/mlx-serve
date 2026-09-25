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
        case .tokenv3: "TokenV3 (fastest, lossy)"
        }
    }
}

/// The server's `steering` key: absent = follow the launch flags, `null` = off
/// even with flags, object = this bank at these scales.
enum SteeringOverride: Equatable {
    case off
    case configured(name: String, ffn: Double, attn: Double)

    /// nil = malformed, by the rules the server's `Setting.fromJsonValue` applies (it
    /// ignores such an entry, so the sheet must not show it as saved).
    init?(json: Any) {
        if json is NSNull { self = .off; return }
        guard let o = json as? [String: Any], let name = o["name"] as? String,
              name.hasPrefix("/") || SteeringRegistry.isBareName(name)
        else { return nil }
        // A JSON bool bridges to NSNumber too; the server reads it as no number.
        func scale(_ key: String) -> Double?? {
            guard let v = o[key] else { return .some(nil) }
            guard let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                  n.doubleValue.isFinite, abs(n.doubleValue) <= SteeringQuickSet.limit else { return nil }
            return .some(n.doubleValue)
        }
        guard let ffn = scale("ffn"), let attn = scale("attn") else { return nil }
        // A file with neither scale steers the ffn arm at 1 (the server's default).
        self = .configured(name: name, ffn: ffn ?? (attn == nil ? 1 : 0), attn: attn ?? 0)
    }

    var json: Any {
        switch self {
        case .off: return NSNull()
        case .configured(let name, let ffn, let attn): return ["name": name, "ffn": ffn, "attn": attn]
        }
    }

    var name: String? {
        if case .configured(let name, _, _) = self { return name }
        return nil
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
    var mtpAcceptance: MtpAcceptanceChoice?
    var steering: SteeringOverride?
    var extra: [String: Any] = [:]

    init(ctxSize: Int? = nil, kvQuant: KvQuantChoice? = nil, mtp: Bool? = nil,
         mtpAcceptance: MtpAcceptanceChoice? = nil, steering: SteeringOverride? = nil) {
        self.ctxSize = ctxSize
        self.kvQuant = kvQuant
        self.mtp = mtp
        self.mtpAcceptance = mtpAcceptance
        self.steering = steering
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
        if let a = rest.removeValue(forKey: "mtp_acceptance") {
            if let s = a as? String { mtpAcceptance = MtpAcceptanceChoice(rawValue: s) }
        }
        if let s = rest.removeValue(forKey: "steering") {
            steering = SteeringOverride(json: s)
        }
        extra = rest
    }

    var isEmpty: Bool { !hasSettings && extra.isEmpty }
    /// True when any field the sheet edits is set.
    var hasSettings: Bool { ctxSize != nil || kvQuant != nil || mtp != nil || mtpAcceptance != nil || steering != nil }

    var json: [String: Any] {
        var out = extra
        if let ctxSize { out["ctx_size"] = ctxSize }
        if let kvQuant { out["kv_quant"] = kvQuant.rawValue }
        if let mtp { out["mtp"] = mtp }
        if let mtpAcceptance { out["mtp_acceptance"] = mtpAcceptance.rawValue }
        if let steering { out["steering"] = steering.json }
        return out
    }

    static func == (a: ModelOverride, b: ModelOverride) -> Bool {
        a.ctxSize == b.ctxSize && a.kvQuant == b.kvQuant && a.mtp == b.mtp && a.mtpAcceptance == b.mtpAcceptance
            && a.steering == b.steering && NSDictionary(dictionary: a.extra).isEqual(to: b.extra)
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

/// The direction banks a request can name: `~/.mlx-serve/steering/<name>.f32`.
enum SteeringRegistry {
    static let defaultDir = NSString(string: "~/.mlx-serve/steering").expandingTildeInPath

    static func names(in dir: String = defaultDir) -> [String] {
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
        return entries.filter { $0.hasSuffix(".f32") }.map { String($0.dropLast(4)) }
            .filter(isBareName).sorted()
    }

    /// The server's `isBareName`: only such a name resolves into the registry dir, so a
    /// file named any other way can be listed but never loaded by name.
    static func isBareName(_ name: String) -> Bool {
        !name.isEmpty && name.count <= 64
            && name.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "._-".contains($0)) }
    }
}
