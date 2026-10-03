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

    func inheritanceState(fields: [GenerationField] = GenerationField.allCases) -> Bool? {
        let values = fields.map { rules[$0.rawValue] == nil }
        return Self.aggregate(values)
    }

    func clientLockState(fields: [GenerationField] = GenerationField.allCases) -> Bool? {
        Self.aggregate(fields.compactMap { rules[$0.rawValue]?.ignoreClient })
    }

    private static func aggregate(_ values: [Bool]) -> Bool? {
        guard let first = values.first else { return false }
        return values.allSatisfy { $0 == first } ? first : nil
    }

    mutating func setInherited(_ inherited: Bool, field: GenerationField) {
        if inherited { rules[field.rawValue] = nil }
        else if rules[field.rawValue] == nil { rules[field.rawValue] = .init(value: field.defaultValue) }
    }

    mutating func setAllInherited(_ inherited: Bool, fields: [GenerationField] = GenerationField.allCases) {
        for field in fields { setInherited(inherited, field: field) }
    }

    mutating func setAllClientLocks(_ locked: Bool, fields: [GenerationField] = GenerationField.allCases) {
        for field in fields where rules[field.rawValue] != nil { rules[field.rawValue]?.ignoreClient = locked }
    }

    func number(_ field: GenerationField) -> Double? {
        if case .number(let value) = rules[field.rawValue]?.value { return value }
        return nil
    }

    static func legacy(_ options: ServerOptions) -> Self {
        var p = Self()
        p.rules["temperature"] = .init(value: .number(options.defaultTemperature))
        p.rules["top_p"] = .init(value: .number(options.defaultTopP))
        if options.defaultTopK > 0 { p.rules["top_k"] = .init(value: .number(Double(options.defaultTopK))) }
        p.rules["max_tokens"] = .init(value: .number(Double(max(0, options.defaultMaxTokens))))
        p.rules["repeat_penalty"] = .init(value: .number(options.defaultRepeatPenalty))
        p.rules["presence_penalty"] = .init(value: .number(options.defaultPresencePenalty))
        if options.defaultReasoningBudget >= 0 {
            p.rules["reasoning_budget"] = .init(value: .number(Double(options.defaultReasoningBudget)))
        }
        if options.defaultEnableThinking { p.rules["enable_thinking"] = .init(value: .boolean(true)) }
        return p
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
    case temperature, topP = "top_p", topK = "top_k", repeatPenalty = "repeat_penalty"
    case presencePenalty = "presence_penalty", frequencyPenalty = "frequency_penalty"
    case maxTokens = "max_tokens", thinking = "enable_thinking", effort = "reasoning_effort"
    case budget = "reasoning_budget"

    static let clientFields = allCases.filter { $0 != .thinking && $0 != .effort }
    var id: String { rawValue }
    var showsSlider: Bool { [.temperature, .topP, .repeatPenalty, .presencePenalty, .frequencyPenalty].contains(self) }
    var title: String {
        switch self {
        case .temperature: "Temperature"
        case .topP: "Top-p"
        case .topK: "Top-k"
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
        case .topK: .number(0)
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
        case .topP: 0...1
        case .topK: 0...1000
        case .repeatPenalty: 0.01...10
        case .maxTokens: 0...Double(Int32.max)
        case .budget: -1...Double(Int32.max)
        default: 0...2
        }
    }
    var clientHelp: String {
        switch self {
        case .temperature, .topP, .presencePenalty: "Default inherits agent or server settings. Explicit values apply only to this chat."
        case .budget: "-1 is unlimited; 0 closes thinking immediately. Server limits still apply."
        default: help
        }
    }
    var help: String {
        switch self {
        case .topK: "0 disables top-k. Inherit uses the next configured default."
        case .repeatPenalty: "1 disables repetition penalty. Nonneutral penalties can disable speculative and batched decoding."
        case .frequencyPenalty: "Uses this engine's existing frequency-penalty mapping. Repetition and frequency penalties share one sampler control."
        case .maxTokens: "0 uses remaining context. Context and memory limits still apply."
        case .budget: "-1 is unlimited; 0 closes thinking immediately. A locked finite budget requires decode-time enforcement."
        case .thinking: "Explicit client thinking or effort wins unless locked."
        case .effort: "Mapped to the model's template vocabulary. Numeric thinking budget is a separate control."
        default: "Client values win unless Ignore client override is checked."
        }
    }
}
