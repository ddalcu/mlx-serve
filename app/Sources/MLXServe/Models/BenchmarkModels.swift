import Foundation

// Benchmark data model + the pure logic around it. Everything here is
// deterministic and testable; the runner (BenchmarkRunner) and the storage
// (BenchmarkStore) hold the IO.
//
// Design notes that are load-bearing rather than stylistic:
//
//  * A suite id pins a WORKLOAD forever. New workload = new id, so rows
//    submitted a year apart stay comparable and old rows never need migrating.
//  * A run measures the server exactly as configured. There are no arms: the
//    settings that shaped a number (kv-quant, MTP, PLD, context) are read from
//    the server's `/props` and recorded on every row, and rows only share a
//    median when those settings agree.
//  * A session is one climb of the context ladder: one row per rung.

// MARK: - Hardware

/// Coarse machine identity. Deliberately nothing that identifies a PERSON or a
/// specific unit — no serial, no host name, no account.
struct BenchmarkHardware: Codable, Hashable {
    var chip: String        // "Apple M4 Max"
    var gpuCores: Int       // 40 — the big within-tier differentiator
    var ramGB: Int
    var osVersion: String
    var onBattery: Bool     // a laptop on battery is a different machine

    static let unknown = BenchmarkHardware(
        chip: "Unknown", gpuCores: 0, ramGB: 0, osVersion: "", onBattery: false)

    /// "Apple M4 Max" → "M4". Empty for anything that isn't an Apple silicon
    /// chip: an honest blank filters correctly, a guessed family does not.
    static func chipFamily(_ brand: String) -> String {
        for token in brand.split(separator: " ") where token.count >= 2 && token.hasPrefix("M") {
            if token.dropFirst().allSatisfy(\.isNumber) { return String(token) }
        }
        return ""
    }

    /// "Apple M4 Max" → "Max", "Apple M4" → "". Filtered separately from the
    /// family because an M4 Max is a different machine from an M4.
    static func chipTier(_ brand: String) -> String {
        let tokens = brand.split(separator: " ").map(String.init)
        let family = chipFamily(brand)
        guard !family.isEmpty, let index = tokens.firstIndex(of: family),
              index + 1 < tokens.count else { return "" }
        let next = tokens[index + 1]
        return ["Pro", "Max", "Ultra"].contains(next) ? next : ""
    }

    var chipFamily: String { BenchmarkHardware.chipFamily(chip) }
    var chipTier: String { BenchmarkHardware.chipTier(chip) }

    /// Grid label. GPU cores are shown because they're the big within-tier
    /// differentiator — a 32-core and a 40-core M4 Max share a chip string and
    /// do not share a decode speed.
    var displayName: String {
        var parts = [chip]
        if gpuCores > 0 { parts.append("\(gpuCores) GPU") }
        parts.append("\(ramGB) GB")
        return parts.joined(separator: " · ")
    }
}

// MARK: - Result

/// One rung's measured result. Written to local history and POSTed verbatim to
/// the community database.
///
/// Every v1 field is kept (the database rules still require `armId` and
/// friends); v2 writes the constant `configured` arm and carries the rung, the
/// speculation ceiling, whether the answer used the planted constant, and the
/// server settings. All four are optional on decode so v1 rows still load.
struct BenchmarkResult: Codable, Identifiable, Hashable {
    /// Bumped when a field changes meaning. 2 = the context ladder with
    /// settings capture; 1 = the retired 2K single-prompt suite.
    static let currentSchemaVersion = 2

    static let configuredArmId = "configured"
    static let configuredArmLabel = "As configured"

    var id: String = UUID().uuidString
    var schemaVersion: Int = BenchmarkResult.currentSchemaVersion

    var sessionId: String
    var suiteId: String

    var armId: String
    var armLabel: String
    var flags: [String: String]
    var isLossy: Bool

    var modelId: String
    var quant: String?
    /// The SERVER build (`/props.settings.version`), not the app bundle.
    var engineVersion: String

    var prefillTps: Double
    var decodeTps: Double
    var ttftMs: Double
    var promptTokens: Int
    var completionTokens: Int

    var runs: Int
    var spreadPercent: Double

    var hardware: BenchmarkHardware
    var date: Date = Date()

    // v2
    /// The rung: 512 … 16384. Nobody parses suite ids.
    var targetTokens: Int?
    /// Count-to-200 decode over the same prefix; 0 when that run failed.
    var ceilingDecodeTps: Double?
    /// Did the coding answer use the constant planted mid-corpus?
    var contextUsed: Bool?
    /// Flattened `/props.settings` + `n_ctx` — see `BenchmarkSettings`.
    var settings: [String: String]?
    /// Free text the user typed in Setup: a name, a nickname, "fan on max".
    var note: String?
    /// The first rung's decode, re-measured after the whole ladder: did the
    /// machine hold its speed? Same values on every row of a session.
    var driftDecodeTps: Double?
    var driftPercent: Double?

    static let maxNoteLength = 120

    // Flattened accessors — the grids and the website read these names.
    var chip: String { hardware.chip }
    var gpuCores: Int { hardware.gpuCores }
    var ramGB: Int { hardware.ramGB }

    /// What may share a median with this row.
    var settingsSignature: String { BenchmarkSettings.signature(settings ?? [:]) }

    /// The rung. Rows without one (the pre-release 2K suite) are dropped at
    /// read time, so this only falls back for a hand-built row.
    var effectiveTargetTokens: Int { targetTokens ?? promptTokens }

    /// Only ladder rows are read back: the pre-release single-prompt suite
    /// carried no rung and is not worth a column. Settings are required too,
    /// same as the website's `isValidRow`, so both read one table.
    var isLadderRow: Bool { schemaVersion >= 2 && targetTokens != nil && settings != nil }

    init(
        id: String = UUID().uuidString,
        schemaVersion: Int = BenchmarkResult.currentSchemaVersion,
        sessionId: String,
        suiteId: String,
        armId: String = BenchmarkResult.configuredArmId,
        armLabel: String = BenchmarkResult.configuredArmLabel,
        flags: [String: String] = [:],
        isLossy: Bool? = nil,
        modelId: String,
        quant: String? = nil,
        engineVersion: String,
        prefillTps: Double,
        decodeTps: Double,
        ttftMs: Double,
        promptTokens: Int,
        completionTokens: Int,
        runs: Int,
        spreadPercent: Double,
        hardware: BenchmarkHardware,
        date: Date = Date(),
        targetTokens: Int? = nil,
        ceilingDecodeTps: Double? = nil,
        contextUsed: Bool? = nil,
        settings: [String: String]? = nil,
        note: String? = nil,
        driftDecodeTps: Double? = nil,
        driftPercent: Double? = nil
    ) {
        self.id = id
        self.schemaVersion = schemaVersion
        self.sessionId = sessionId
        self.suiteId = suiteId
        self.armId = armId
        self.armLabel = armLabel
        self.flags = flags
        self.isLossy = isLossy ?? BenchmarkSettings.isLossy(settings ?? [:])
        self.modelId = modelId
        self.quant = quant
        self.engineVersion = engineVersion
        self.prefillTps = prefillTps
        self.decodeTps = decodeTps
        self.ttftMs = ttftMs
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.runs = runs
        self.spreadPercent = spreadPercent
        self.hardware = hardware
        self.date = date
        self.targetTokens = targetTokens
        self.ceilingDecodeTps = ceilingDecodeTps
        self.contextUsed = contextUsed
        self.settings = settings
        self.note = note
        self.driftDecodeTps = driftDecodeTps
        self.driftPercent = driftPercent
    }

    /// Firebase stores no empty object, so a v2 row's `flags: {}` comes back
    /// ABSENT and the synthesized decoder refused every shared row. Every
    /// key that can legitimately be missing decodes with a default.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        schemaVersion = try c.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        sessionId = try c.decode(String.self, forKey: .sessionId)
        suiteId = try c.decode(String.self, forKey: .suiteId)
        armId = try c.decodeIfPresent(String.self, forKey: .armId) ?? BenchmarkResult.configuredArmId
        armLabel = try c.decodeIfPresent(String.self, forKey: .armLabel) ?? BenchmarkResult.configuredArmLabel
        flags = try c.decodeIfPresent([String: String].self, forKey: .flags) ?? [:]
        modelId = try c.decode(String.self, forKey: .modelId)
        quant = try c.decodeIfPresent(String.self, forKey: .quant)
        engineVersion = try c.decodeIfPresent(String.self, forKey: .engineVersion) ?? ""
        prefillTps = try c.decodeIfPresent(Double.self, forKey: .prefillTps) ?? 0
        decodeTps = try c.decode(Double.self, forKey: .decodeTps)
        ttftMs = try c.decodeIfPresent(Double.self, forKey: .ttftMs) ?? 0
        promptTokens = try c.decodeIfPresent(Int.self, forKey: .promptTokens) ?? 0
        completionTokens = try c.decodeIfPresent(Int.self, forKey: .completionTokens) ?? 0
        runs = try c.decodeIfPresent(Int.self, forKey: .runs) ?? 0
        spreadPercent = try c.decodeIfPresent(Double.self, forKey: .spreadPercent) ?? 0
        hardware = try c.decode(BenchmarkHardware.self, forKey: .hardware)
        date = try c.decodeIfPresent(Date.self, forKey: .date) ?? Date()
        targetTokens = try c.decodeIfPresent(Int.self, forKey: .targetTokens)
        ceilingDecodeTps = try c.decodeIfPresent(Double.self, forKey: .ceilingDecodeTps)
        contextUsed = try c.decodeIfPresent(Bool.self, forKey: .contextUsed)
        settings = try c.decodeIfPresent([String: String].self, forKey: .settings)
        note = try c.decodeIfPresent(String.self, forKey: .note)
        driftDecodeTps = try c.decodeIfPresent(Double.self, forKey: .driftDecodeTps)
        driftPercent = try c.decodeIfPresent(Double.self, forKey: .driftPercent)
        isLossy = try c.decodeIfPresent(Bool.self, forKey: .isLossy) ?? BenchmarkSettings.isLossy(settings ?? [:])
    }

    /// Trimmed and capped; empty becomes nil so the row carries no field.
    static func cleanNote(_ text: String) -> String? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return String(trimmed.prefix(maxNoteLength))
    }
}

extension BenchmarkResult {
    /// A row that actually measured something.
    ///
    /// A rung whose every coding run was discarded still exists as a row full
    /// of zeroes. Letting one through renders a table of dashes above a Share
    /// button, and if shared it drags a 0 tok/s sample into the community
    /// median for that cell. Empty settings (the `/props` read failed) have
    /// no comparison axis, and Firebase stores `{}` as absent, which the
    /// board rejects.
    var isPublishable: Bool { runs > 0 && decodeTps > 0 && !(settings ?? [:]).isEmpty }
}

// MARK: - Suite

/// One rung of the context ladder. The id pins the workload: the llmprobe
/// corpus at `targetTokens` of prompt, `genTokens` of coding answer, and a
/// count-to-200 ceiling run on the same prefix.
struct BenchmarkSuite: Identifiable, Hashable {
    let id: String
    let title: String
    let targetTokens: Int
    let genTokens: Int
    let runs: Int
    let warmups: Int

    static let ladder: [BenchmarkSuite] = [
        BenchmarkSuite(id: "ctx-v1-512", title: "512", targetTokens: 512, genTokens: 192, runs: 2, warmups: 1),
        BenchmarkSuite(id: "ctx-v1-1k", title: "1k", targetTokens: 1024, genTokens: 192, runs: 2, warmups: 1),
        BenchmarkSuite(id: "ctx-v1-2k", title: "2k", targetTokens: 2048, genTokens: 192, runs: 2, warmups: 1),
        BenchmarkSuite(id: "ctx-v1-4k", title: "4k", targetTokens: 4096, genTokens: 192, runs: 2, warmups: 1),
        BenchmarkSuite(id: "ctx-v1-8k", title: "8k", targetTokens: 8192, genTokens: 192, runs: 2, warmups: 1),
        BenchmarkSuite(id: "ctx-v1-16k", title: "16k", targetTokens: 16384, genTokens: 192, runs: 2, warmups: 1),
    ]

    static func byId(_ id: String) -> BenchmarkSuite? { ladder.first { $0.id == id } }

    /// "512" / "4k" / "16k" for a rung that may not be in the ladder.
    static func title(forTarget tokens: Int) -> String {
        if let suite = ladder.first(where: { $0.targetTokens == tokens }) { return suite.title }
        return tokens >= 1024 ? "\(tokens / 1024)k" : "\(tokens)"
    }
}

// MARK: - Cache contamination

enum BenchmarkPrompt {
    /// The largest share of a prompt that may come from the KV cache before the
    /// run stops being a prefill measurement.
    ///
    /// It cannot be zero. The chat template's header and the `[probe ` lead-in
    /// are byte-identical across runs, so a few tokens ALWAYS match. The
    /// server divides `prompt_per_second` by the tokens it actually computed,
    /// so a header-sized overlap costs nothing; what has to be caught is a
    /// genuine warm hit covering the whole prompt.
    static let maxCachedFraction = 0.10

    static func prefillWasReused(promptTokens: Int, cachedTokens: Int) -> Bool {
        guard promptTokens > 0 else { return true }   // measured nothing
        return Double(cachedTokens) > Double(promptTokens) * maxCachedFraction
    }
}

/// Which of a rung's two requests a sample came from, and whether it counts.
enum LadderSample {
    enum Kind { case coding, counting }

    /// A decode rate over fewer tokens than this is noise, not a ceiling: a
    /// checkpoint that answers the counting task with "1" and stops reports a
    /// one-token rate that says nothing about speculation.
    static let minCeilingTokens = 32

    /// The cache discard applies to the CODING run only: its prefill figure is
    /// the measurement. The counting run rides the same archive on purpose —
    /// a prefix hit there is what makes it a decode-only measurement — and is
    /// kept only when it generated enough to time.
    static func keep(kind: Kind, promptTokens: Int, cachedTokens: Int, completionTokens: Int) -> Bool {
        switch kind {
        case .coding: return !BenchmarkPrompt.prefillWasReused(promptTokens: promptTokens, cachedTokens: cachedTokens)
        case .counting: return completionTokens >= minCeilingTokens
        }
    }
}

// MARK: - Preflight

enum LadderPreflight: Equatable {
    case ready
    case contextTooSmall(have: Int, need: Int)
    case noModel

    /// Template slack on top of the widest rung's prompt + answer.
    static let templateSlack = 256

    static func need(_ ladder: [BenchmarkSuite]) -> Int {
        (ladder.map { $0.targetTokens + $0.genTokens }.max() ?? 0) + templateSlack
    }

    static func decide(contextLength: Int?, ladder: [BenchmarkSuite]) -> LadderPreflight {
        guard let have = contextLength, have > 0 else { return .noModel }
        let required = need(ladder)
        return have >= required ? .ready : .contextTooSmall(have: have, need: required)
    }
}

// MARK: - Settings

/// The server settings that shaped a number, as recorded on every row.
enum BenchmarkSettings {

    /// Flattened `/props`: the `settings` object plus `n_ctx`. Every value is
    /// a string so the row's wire shape is one flat map the rules can bound.
    static func flatten(props: [String: Any]) -> [String: String] {
        guard let settings = props["settings"] as? [String: Any] else { return [:] }
        var out: [String: String] = [:]
        func put(_ key: String, _ value: Any?) {
            guard let value else { return }
            // JSONSerialization hands booleans back as NSNumber too, and any
            // NSNumber casts to Bool — ask the CF type, not the cast.
            if let n = value as? NSNumber {
                if CFGetTypeID(n) == CFBooleanGetTypeID() { out[key] = n.boolValue ? "true" : "false" }
                else { out[key] = n.stringValue }
                return
            }
            if let s = value as? String { out[key] = s; return }
        }
        put("engine", settings["engine"])
        put("version", settings["version"])
        put("kv_quant", settings["kv_quant"])
        put("kv_attn_mode", settings["kv_attn_mode"])
        put("decode_attn_quant", settings["decode_attn_quant"])
        put("prefill_chunk", settings["prefill_chunk"])
        put("drafter", settings["drafter"])
        put("max_concurrent", settings["max_concurrent"])
        if let mtp = settings["mtp"] as? [String: Any] {
            put("mtp_loaded", mtp["loaded"])
            put("mtp_default_on", mtp["default_on"])
            put("mtp_depth", mtp["depth"])
            put("mtp_adaptive", mtp["adaptive"])
            put("mtp_acceptance", mtp["acceptance"])
        }
        if let pld = settings["pld"] as? [String: Any] {
            put("pld_default_on", pld["default_on"])
            put("pld_draft_len", pld["draft_len"])
            put("pld_key_len", pld["key_len"])
        }
        if let cache = settings["prefix_cache"] as? [String: Any] {
            put("prefix_cache_mem", cache["mem_bytes"])
        }
        if let gen = props["default_generation_settings"] as? [String: Any] {
            put("n_ctx", gen["n_ctx"])
        }
        return out
    }

    /// What changes a request's speed, so what may not share a median.
    /// Cache size, concurrency and the server version stay out — fragmenting
    /// by release would leave every cell at n=1 forever.
    static func signature(_ s: [String: String]) -> String {
        func flag(_ key: String) -> String {
            switch s[key] { case "true": return "1"; case "false": return "0"; default: return "?" }
        }
        return ["kv" + (s["kv_quant"] ?? "?"),
                "daq" + flag("decode_attn_quant"),
                "mtp" + flag("mtp_default_on"),
                "pld" + flag("pld_default_on"),
                s["drafter"] ?? "?"].joined(separator: "|")
    }

    /// `--kv-quant` and `--decode-attn-quant` trade output quality for speed.
    /// Speculative decoding is NOT lossy — it reproduces the same tokens.
    static func isLossy(_ s: [String: String]) -> Bool {
        if let kv = s["kv_quant"], kv != "off" { return true }
        return s["decode_attn_quant"] == "true"
    }

    /// Short chips for a table row: "KV 8-bit", "PLD", "MTP off", "ctx 48K".
    static func summaryChips(_ s: [String: String]) -> [String] {
        var chips: [String] = []
        // ds4 has no KV lever and llama.cpp no MTP; `/props` reports them off.
        let engine = s["engine"] ?? "mlx"
        if engine == "ds4" { chips.append("ds4") }
        if engine == "llama" { chips.append("llama.cpp") }
        if engine != "ds4", let kv = s["kv_quant"] {
            chips.append(kv == "off" ? "KV off" : Int(kv) != nil ? "KV \(kv)-bit" : "KV \(kv)")
        }
        if s["decode_attn_quant"] == "true" { chips.append("Attn quant") }
        if s["pld_default_on"] == "true" { chips.append("PLD") }
        if engine != "llama", let mtp = s["mtp_default_on"] { chips.append(mtp == "true" ? "MTP" : "MTP off") }
        if let drafter = s["drafter"], drafter != "none" {
            chips.append(drafter == "dflash" ? "DFlash" : "Drafter")
        }
        if let ctx = s["n_ctx"].flatMap(Int.init), ctx > 0 {
            chips.append("ctx \(ctx / 1024)K")
        }
        return chips
    }

    /// Human labels for the detail sheet's settings grid.
    static let labels: [(key: String, label: String)] = [
        ("version", "Server"), ("engine", "Engine"), ("n_ctx", "Context"),
        ("kv_quant", "KV quant"), ("kv_attn_mode", "KV attention"),
        ("decode_attn_quant", "Decode attn quant"), ("prefill_chunk", "Prefill chunk"),
        ("mtp_loaded", "MTP head"), ("mtp_default_on", "MTP on"), ("mtp_depth", "MTP depth"),
        ("mtp_adaptive", "MTP adaptive"), ("mtp_acceptance", "MTP acceptance"),
        ("drafter", "Drafter"), ("pld_default_on", "PLD"), ("pld_draft_len", "PLD draft len"),
        ("pld_key_len", "PLD key len"), ("max_concurrent", "Max concurrent"),
        ("prefix_cache_mem", "Prefix cache"),
    ]
}

// MARK: - Sessions

/// One climb of the ladder: the rows sharing a `sessionId`, sorted by rung.
struct BenchmarkSession: Identifiable, Hashable {
    let id: String
    let rungs: [BenchmarkResult]

    var first: BenchmarkResult { rungs[0] }
    var modelId: String { first.modelId }
    var hardware: BenchmarkHardware { first.hardware }
    var settings: [String: String] { first.settings ?? [:] }
    var date: Date { rungs.map(\.date).max() ?? first.date }
    var isLossy: Bool { rungs.contains { $0.isLossy } }
    var engineVersion: String { first.engineVersion }
    var note: String? { rungs.compactMap(\.note).first }
    var driftPercent: Double? { rungs.compactMap(\.driftPercent).first }
    var driftDecodeTps: Double? { rungs.compactMap(\.driftDecodeTps).first }
    /// The decode the drift is measured against: the smallest rung's.
    var driftBaselineTps: Double? { driftPercent == nil ? nil : rungs.first?.decodeTps }

    func rung(at target: Int) -> BenchmarkResult? {
        rungs.first { $0.effectiveTargetTokens == target }
    }

    func decode(at target: Int) -> Double? {
        rung(at: target).map(\.decodeTps)
    }


    /// Newest session first, rungs ascending inside each.
    static func group(_ rows: [BenchmarkResult]) -> [BenchmarkSession] {
        var bySession: [String: [BenchmarkResult]] = [:]
        for row in rows { bySession[row.sessionId, default: []].append(row) }
        return bySession.map { id, rungs in
            BenchmarkSession(id: id, rungs: rungs.sorted { $0.effectiveTargetTokens < $1.effectiveTargetTokens })
        }
        .sorted { $0.date > $1.date }
    }
}

// MARK: - Community sorting

/// One comparator type for every column of the Community table, so a
/// dynamic rung column can be a sort key like a fixed one.
struct BenchmarkFamilySort: SortComparator, Hashable {
    enum Key: Hashable {
        case machine, model, sessions, date
        case rung(Int)
    }

    var key: Key
    var order: SortOrder = .forward

    init(_ key: Key, order: SortOrder = .forward) {
        self.key = key
        self.order = order
    }

    func compare(_ a: BenchmarkStore.CellFamily, _ b: BenchmarkStore.CellFamily) -> ComparisonResult {
        let result: ComparisonResult
        switch key {
        case .machine: result = a.hardware.displayName.localizedStandardCompare(b.hardware.displayName)
        case .model: result = a.modelId.localizedStandardCompare(b.modelId)
        case .sessions: result = Self.compare(Double(a.sessionCount), Double(b.sessionCount))
        case .date: result = Self.compare(a.latestDate.timeIntervalSince1970, b.latestDate.timeIntervalSince1970)
        case .rung(let target):
            // A family with no figure at this rung sorts LAST either way: a
            // dash is not a small number.
            switch (a.decode(at: target), b.decode(at: target)) {
            case (nil, nil): return .orderedSame
            case (nil, _): return .orderedDescending
            case (_, nil): return .orderedAscending
            case (let x?, let y?): result = Self.compare(x, y)
            }
        }
        return order == .forward ? result : result.reversed
    }

    private static func compare(_ x: Double, _ y: Double) -> ComparisonResult {
        x < y ? .orderedAscending : x > y ? .orderedDescending : .orderedSame
    }
}

private extension ComparisonResult {
    var reversed: ComparisonResult {
        switch self {
        case .orderedAscending: return .orderedDescending
        case .orderedDescending: return .orderedAscending
        case .orderedSame: return .orderedSame
        }
    }
}

// MARK: - Drift

/// Did the machine hold its speed for the length of the run? (llmprobe's
/// `classifyLoadDrift`.) The smallest rung's coding decode is measured again
/// after the whole ladder, minutes of sustained load apart. A drop is thermal
/// throttling or something else arriving on the box; a rise means the warmup
/// did not warm it. Either way the figures were taken on moving ground.
enum BenchmarkDrift {
    enum Verdict: String { case steady, degraded, improved, unknown }

    /// Beyond this much movement, the run's numbers are a range, not figures.
    static let tolerancePercent = 10.0

    static func percent(first: Double?, last: Double?) -> Double? {
        guard let first, let last, first > 0, last > 0 else { return nil }
        return ((last - first) / first * 1000).rounded() / 10
    }

    static func verdict(percent: Double?) -> Verdict {
        guard let percent else { return .unknown }
        if percent <= -tolerancePercent { return .degraded }
        if percent >= tolerancePercent { return .improved }
        return .steady
    }

    /// "61.2 → 58.4 tok/s (−4.6%, steady)"
    static func summary(first: Double?, last: Double?, percent: Double?) -> String {
        guard let first, let last, let percent else { return "not measured" }
        let sign = percent > 0 ? "+" : ""
        return String(format: "%.1f → %.1f tok/s (%@%.1f%%, %@)", first, last, sign, percent,
                      verdict(percent: percent).rawValue)
    }
}

// MARK: - Stats

enum BenchmarkStats {

    /// Middle value. A single slow run (thermal blip, a background build) must
    /// not drag the published number, which a mean would let it do.
    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[mid] }
        return (sorted[mid - 1] + sorted[mid]) / 2
    }

    /// (max − min) as a percentage of the median.
    ///
    /// Relative because 2 tok/s of spread means something very different on a
    /// 3 tok/s 235B than on a 200 tok/s 2B. A wide spread is what tells the
    /// user (and later, the validator) that the run wasn't clean.
    static func spreadPercent(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mid = median(values)
        guard mid > 0, let low = values.min(), let high = values.max() else { return 0 }
        return (high - low) / mid * 100
    }
}
