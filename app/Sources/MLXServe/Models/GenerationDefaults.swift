import Foundation

struct GenerationDefaults: Codable, Equatable {
    enum Value: Codable, Equatable {
        case number(Double)
        case boolean(Bool)
        case text(String)

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let v = try? c.decode(Bool.self) { self = .boolean(v) }
            else if let v = try? c.decode(Double.self) { self = .number(v) }
            else { self = .text(try c.decode(String.self)) }
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .number(let v): try c.encode(v)
            case .boolean(let v): try c.encode(v)
            case .text(let v): try c.encode(v)
            }
        }

        var json: Any {
            switch self {
            case .number(let v): v
            case .boolean(let v): v
            case .text(let v): v
            }
        }
    }

    struct Rule: Codable, Equatable {
        var value: Value
        var ignoreClient = false
        enum CodingKeys: String, CodingKey { case value; case ignoreClient = "ignore_client" }

        init(value: Value, ignoreClient: Bool = false) {
            self.value = value
            self.ignoreClient = ignoreClient
        }

        private struct RawKey: CodingKey {
            var stringValue: String
            var intValue: Int? { nil }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }

        init(from decoder: Decoder) throws {
            let raw = try decoder.container(keyedBy: RawKey.self)
            guard raw.allKeys.allSatisfy({ CodingKeys(rawValue: $0.stringValue) != nil }) else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Unknown generation rule field"))
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            value = try c.decode(Value.self, forKey: .value)
            ignoreClient = c.contains(.ignoreClient) ? try c.decode(Bool.self, forKey: .ignoreClient) : false
        }
    }

    var rules: [String: Rule] = [:]

    init(rules: [String: Rule] = [:]) { self.rules = rules }
    init(from decoder: Decoder) throws {
        rules = try decoder.singleValueContainer().decode([String: Rule].self)
        for (key, rule) in rules {
            guard let field = GenerationField(rawValue: key) else {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Unknown generation setting: \(key)"))
            }
            let valid: Bool
            switch (field, rule.value) {
            case (.thinking, .boolean): valid = true
            case (.effort, .text(let value)): valid = ["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(value)
            case (.thinking, _), (.effort, _): valid = false
            case (_, .number(let value)):
                let integer = [.topK, .maxTokens, .budget].contains(field)
                valid = value.isFinite && field.range.contains(value) && (!integer || value.rounded() == value)
            default: valid = false
            }
            if !valid {
                throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Invalid generation setting: \(key)"))
            }
        }
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(rules)
    }
    init(json: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: json),
              let parsed = try? JSONDecoder().decode(Self.self, from: data) else { return }
        self = parsed
    }
    var json: [String: Any] {
        rules.mapValues { ["value": $0.value.json, "ignore_client": $0.ignoreClient] }
    }

    enum Mode: Int {
        case inherited, enabled, forced
    }

    func mode(_ field: GenerationField) -> Mode {
        guard let rule = rules[field.rawValue] else { return .inherited }
        return rule.ignoreClient ? .forced : .enabled
    }

    func mode(fields: [GenerationField]) -> Mode? {
        guard let first = fields.first.map(mode) else { return .inherited }
        return fields.allSatisfy { mode($0) == first } ? first : nil
    }

    mutating func setMode(_ mode: Mode, field: GenerationField, inherited: Value? = nil) {
        guard mode != .inherited else { rules[field.rawValue] = nil; return }
        var rule = rules[field.rawValue] ?? .init(value: inherited ?? field.defaultValue)
        rule.ignoreClient = mode == .forced
        rules[field.rawValue] = rule
    }

    mutating func setMode(_ mode: Mode, fields: [GenerationField], inherited: GenerationDefaults = .init()) {
        for field in fields { setMode(mode, field: field, inherited: inherited.rules[field.rawValue]?.value) }
    }

    func number(_ field: GenerationField) -> Double? {
        if case .number(let value) = rules[field.rawValue]?.value { return value }
        return nil
    }

    /// Only what the app already applied server-wide (its `--temp`/`--top-p`/`--top-k` launch
    /// flags); the other saved defaults rode the app's own chats and must not reach every client.
    static func legacy(_ options: ServerOptions) -> Self {
        var p = Self()
        p.rules["temperature"] = .init(value: .number(options.defaultTemperature))
        p.rules["top_p"] = .init(value: .number(options.defaultTopP))
        if options.defaultTopK > 0 { p.rules["top_k"] = .init(value: .number(Double(options.defaultTopK))) }
        return p
    }
}

extension GenerationDefaults {
    private static let agentNumbers: [(GenerationField, WritableKeyPath<Agent, Double?>)] = [
        (.temperature, \.temperature), (.topP, \.topP), (.repeatPenalty, \.repeatPenalty), (.presencePenalty, \.presencePenalty),
    ]
    private static let agentIntegers: [(GenerationField, WritableKeyPath<Agent, Int?>)] = [
        (.topK, \.topK), (.maxTokens, \.maxTokens), (.budget, \.reasoningBudget),
    ]

    init(agent: Agent) {
        self.init()
        for (field, key) in Self.agentNumbers {
            if let value = agent[keyPath: key] { rules[field.rawValue] = .init(value: .number(value)) }
        }
        for (field, key) in Self.agentIntegers {
            if let value = agent[keyPath: key] { rules[field.rawValue] = .init(value: .number(Double(value))) }
        }
    }

    func apply(to agent: inout Agent) {
        for (field, key) in Self.agentNumbers { agent[keyPath: key] = number(field) }
        for (field, key) in Self.agentIntegers { agent[keyPath: key] = number(field).map(Int.init) }
    }

    static func inherited(modelPath: String?) -> GenerationDefaults {
        var profile = (try? GenerationDefaultsFile.load()) ?? .init()
        if let modelPath, let model = ModelSettingsFile.load().override(for: modelPath) {
            profile.rules.merge(model.generationDefaults.rules) { _, model in model }
        }
        return profile
    }
}

enum GenerationDefaultsFile {
    static var defaultPath: String {
        NSString(string: "~/.mlx-serve/generation-settings.json").expandingTildeInPath
    }
    static let changed = Notification.Name("GenerationDefaultsChanged")

    static func load(path: String = defaultPath) throws -> GenerationDefaults {
        guard FileManager.default.fileExists(atPath: path) else { return .init() }
        return try JSONDecoder().decode(GenerationDefaults.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
    }

    static func save(_ profile: GenerationDefaults, path: String = defaultPath) throws {
        try FileManager.default.createDirectory(atPath: (path as NSString).deletingLastPathComponent,
                                               withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(profile)
        _ = try JSONDecoder().decode(GenerationDefaults.self, from: data)
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        if path == defaultPath { NotificationCenter.default.post(name: changed, object: nil) }
    }

    static func migrate(_ options: ServerOptions, path: String = defaultPath) throws {
        guard !FileManager.default.fileExists(atPath: path) else { return }
        try save(.legacy(options), path: path)
    }
}

enum GenerationField: String, CaseIterable, Identifiable {
    case temperature, topP = "top_p", topK = "top_k", minP = "min_p", repeatPenalty = "repeat_penalty"
    case presencePenalty = "presence_penalty", frequencyPenalty = "frequency_penalty"
    case maxTokens = "max_tokens", thinking = "enable_thinking", effort = "reasoning_effort"
    case budget = "reasoning_budget"

    static let agentFields: [GenerationField] = [.temperature, .topP, .topK, .repeatPenalty, .presencePenalty, .maxTokens, .budget]
    var id: String { rawValue }
    var parameterName: String { rawValue }
    var isInteger: Bool { [.topK, .maxTokens, .budget].contains(self) }
    var showsSlider: Bool { self != .thinking && self != .effort }
    var presets: [Double]? {
        switch self {
        case .topK: [0, 5, 10, 20, 40, 64, 100, 200, 500, 1000]
        case .maxTokens: [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        case .budget: [-1, 0, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        default: nil
        }
    }
    var guidance: (low: String, high: String)? {
        switch self {
        case .temperature: ("Focused", "Creative")
        case .topP: ("Focused", "Varied")
        case .topK: ("Off", "Wide")
        case .minP: ("Off", "Selective")
        case .repeatPenalty, .presencePenalty, .frequencyPenalty: ("Off", "Strong")
        case .maxTokens: ("Auto", "Long")
        case .budget: ("Unlimited", "Long")
        default: nil
        }
    }
    var sliderRange: ClosedRange<Double> { self == .repeatPenalty ? 1...2 : range }

    func numberValue(_ number: Double) -> GenerationDefaults.Value? {
        guard number.isFinite, range.contains(number) else { return nil }
        return .number(isInteger ? number.rounded() : number)
    }

    func sliderPosition(_ number: Double) -> Double {
        if let presets {
            return Double(presets.indices.min { abs(presets[$0] - number) < abs(presets[$1] - number) } ?? 0)
        }
        return min(max(number, sliderRange.lowerBound), sliderRange.upperBound)
    }

    func sliderNumber(_ position: Double) -> Double {
        guard let presets else { return position }
        return presets[max(0, min(Int(position.rounded()), presets.count - 1))]
    }
    var title: String {
        switch self {
        case .temperature: "Temperature"
        case .topP: "Top-p"
        case .topK: "Top-k"
        case .minP: "Min-p"
        case .repeatPenalty: "Repetition penalty"
        case .presencePenalty: "Presence penalty"
        case .frequencyPenalty: "Frequency penalty"
        case .maxTokens: "Max output tokens"
        case .thinking: "Thinking"
        case .effort: "Reasoning effort"
        case .budget: "Thinking budget"
        }
    }
    var defaultValue: GenerationDefaults.Value {
        switch self {
        case .temperature: .number(0.8)
        case .topP: .number(0.95)
        case .topK, .minP: .number(0)
        case .repeatPenalty: .number(1)
        case .presencePenalty, .frequencyPenalty: .number(0)
        case .maxTokens: .number(16384)
        case .budget: .number(1024)
        case .thinking: .boolean(true)
        case .effort: .text("low")
        }
    }
    var range: ClosedRange<Double> {
        switch self {
        case .topP, .minP: 0...1
        case .topK: 0...1000
        case .repeatPenalty: 0.01...10
        case .maxTokens: 0...Double(Int32.max)
        case .budget: -1...Double(Int32.max)
        default: 0...2
        }
    }
    var help: String {
        let description: String
        switch self {
        case .topK: description = "Keeps only the k most likely next tokens. Lower values narrow choices; 0 disables this filter."
        case .minP: description = "Minimum probability relative to the most likely token. 0 disables min-p."
        case .repeatPenalty: description = "1 disables repetition penalty. Nonneutral penalties can disable speculative and batched decoding."
        case .frequencyPenalty: description = "Uses this engine's existing frequency-penalty mapping. Repetition and frequency penalties share one sampler control."
        case .maxTokens: description = "0 uses remaining context. Context and memory limits still apply."
        case .budget: description = "API: reasoning_budget_tokens. -1 is unlimited; 0 closes thinking immediately. Effort words map against this budget."
        case .thinking: description = "Explicit client thinking or effort wins unless locked."
        case .effort: description = "Mapped to the model's template vocabulary. Numeric thinking budget is a separate control."
        case .temperature: description = "Lower values favor predictable replies; higher values add variety. 0 is greedy."
        case .topP: description = "Keep the smallest token pool covering this probability. 1 keeps all tokens."
        case .presencePenalty: description = "Discourage tokens already used, regardless of how often they appeared. 0 disables the penalty."
        }
        return L10n.text(description)
    }
}
