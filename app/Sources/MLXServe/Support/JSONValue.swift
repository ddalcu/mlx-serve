import Foundation

/// Any string can be a key. Server ids in `mcp.json`, provider fields the app
/// does not model: the containers are dynamic, so an enum of fixed names cannot
/// express them.
struct DynamicCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

/// One JSON value, for the keys a model does not declare: `[String: Any]` cannot
/// cross a Codable container, and a key the model leaves out is a key a save
/// deletes.
enum JSONValue: Codable, Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int.self) { self = .int(v) }
        else if let v = try? c.decode(Double.self) { self = .double(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else if let v = try? c.decode([String: JSONValue].self) { self = .object(v) }
        else {
            throw DecodingError.dataCorruptedError(in: c, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }
}

extension KeyedDecodingContainer where K == DynamicCodingKey {
    /// Every key this container holds that the model does not declare.
    func unmodelled(known: Set<String>) throws -> [String: JSONValue] {
        var out: [String: JSONValue] = [:]
        for key in allKeys where !known.contains(key.stringValue) {
            out[key.stringValue] = try decode(JSONValue.self, forKey: key)
        }
        return out
    }
}

extension KeyedEncodingContainer where K == DynamicCodingKey {
    /// Writes the keys the model does not declare; `known` keeps a stray extra
    /// from emitting a second copy of a declared field.
    mutating func encodeUnmodelled(_ extra: [String: JSONValue], known: Set<String>) throws {
        for (key, value) in extra.sorted(by: { $0.key < $1.key })
        where !known.contains(key) && DynamicCodingKey(stringValue: key) != nil {
            try encode(value, forKey: DynamicCodingKey(stringValue: key)!)
        }
    }
}
