import Foundation

/// One upstream OpenAI-compatible chat provider, as stored in
/// `~/.mlx-serve/providers.json` — the file the SERVER reads (`src/providers.zig`),
/// so the keys are its keys. The app edits the file and asks the server to
/// reload; the server owns probing and routing.
struct ProviderEntry: Codable, Identifiable, Equatable {
    var id = UUID()
    var name: String = ""
    var url: String = ""
    var apiKey: String = ""
    /// Env var read by the server; wins over `apiKey` when set and non-empty.
    var apiKeyEnv: String = ""
    var enabled: Bool = true
    /// Model ids to expose. Filters the provider's own `/v1/models` list;
    /// the whole list for a provider that has none. Empty = everything.
    var models: [String] = []

    enum CodingKeys: String, CodingKey {
        case name, url, enabled, models
        case apiKey = "api_key"
        case apiKeyEnv = "api_key_env"
    }

    init() {}

    init(name: String, url: String, apiKey: String = "", apiKeyEnv: String = "", enabled: Bool = true, models: [String] = []) {
        self.name = name
        self.url = url
        self.apiKey = apiKey
        self.apiKeyEnv = apiKeyEnv
        self.enabled = enabled
        self.models = models
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        url = try c.decodeIfPresent(String.self, forKey: .url) ?? ""
        apiKey = try c.decodeIfPresent(String.self, forKey: .apiKey) ?? ""
        apiKeyEnv = try c.decodeIfPresent(String.self, forKey: .apiKeyEnv) ?? ""
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? true
        models = try c.decodeIfPresent([String].self, forKey: .models) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(name, forKey: .name)
        try c.encode(url, forKey: .url)
        if !apiKey.isEmpty { try c.encode(apiKey, forKey: .apiKey) }
        if !apiKeyEnv.isEmpty { try c.encode(apiKeyEnv, forKey: .apiKeyEnv) }
        if !enabled { try c.encode(enabled, forKey: .enabled) }
        if !models.isEmpty { try c.encode(models, forKey: .models) }
    }

    /// Why the server would skip this row, or nil. Mirrors `providers.zig`'s
    /// `parseConfigs` so the editor refuses exactly what the server refuses.
    /// `serverPort` is the port MLX Core's own server listens on: adding that
    /// server as its own provider makes every proxied request land back on it.
    func problem(serverPort: UInt16? = nil) -> String? {
        if name.isEmpty { return "Name is required" }
        if name.count > 64 { return "Name is too long" }
        if !name.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == ".") }) {
            return "Name may only use letters, digits, - _ ."
        }
        let u = url.trimmingCharacters(in: .whitespaces)
        if !(u.hasPrefix("http://") || u.hasPrefix("https://")) { return "URL must start with http:// or https://" }
        if let serverPort, Self.isLoopback(url: u, port: serverPort) {
            return "That is the server MLX Core is running — add an mlx-serve on a different port"
        }
        return nil
    }

    /// Mirrors `providers.zig`'s `isSelfUrl`: loopback host + the given port.
    static func isLoopback(url: String, port: UInt16) -> Bool {
        guard let parsed = URL(string: url), let host = parsed.host, parsed.port == Int(port) else { return false }
        return host == "localhost" || host == "::1" || host == "0.0.0.0" || host.hasPrefix("127.")
    }

    /// Ids out of an OpenAI `/v1/models` body, in listed order; nil when the
    /// body is not that shape.
    static func modelIds(fromModelsBody data: Data) -> [String]? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = obj["data"] as? [[String: Any]] else { return nil }
        return list.compactMap { ($0["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } }
    }

    /// `<url>/models` URLs to try, in order: the URL as written, then the
    /// `/v1` sibling the server also probes for a bare `host:port`.
    static func modelsURLs(for base: String) -> [URL] {
        var b = base.trimmingCharacters(in: .whitespaces)
        while b.hasSuffix("/") { b.removeLast() }
        var candidates = [b + "/models"]
        if !b.hasSuffix("/v1") { candidates.append(b + "/v1/models") }
        return candidates.compactMap(URL.init(string:))
    }

    /// Model ids typed one per line or comma-separated.
    static func parseModelList(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == "\n" || $0 == "," })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}

enum ProvidersFile {
    static let defaultPath = NSString(string: "~/.mlx-serve/providers.json").expandingTildeInPath

    /// Missing file = no providers. Accepts the bare-array form the app writes
    /// and the `{"providers":[...]}` wrapper the server also reads.
    static func load(path: String = defaultPath) -> [ProviderEntry] {
        guard let data = FileManager.default.contents(atPath: path) else { return [] }
        return decode(data)
    }

    static func decode(_ data: Data) -> [ProviderEntry] {
        let dec = JSONDecoder()
        if let list = try? dec.decode([ProviderEntry].self, from: data) { return list }
        struct Wrapper: Decodable { let providers: [ProviderEntry] }
        return (try? dec.decode(Wrapper.self, from: data))?.providers ?? []
    }

    static func encode(_ entries: [ProviderEntry]) throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return try enc.encode(entries)
    }

    static func save(_ entries: [ProviderEntry], path: String = defaultPath) throws {
        let dir = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try encode(entries).write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// Names used twice — the server keeps the first and skips the rest.
    static func duplicateNames(_ entries: [ProviderEntry]) -> Set<String> {
        var seen: Set<String> = []
        var dups: Set<String> = []
        for e in entries where !seen.insert(e.name).inserted { dups.insert(e.name) }
        return dups
    }
}

/// One row of `GET /v1/providers` — the server's live view of a provider.
struct ProviderStatus: Decodable, Equatable {
    var name: String
    var url: String
    var up: Bool
    var probed: Bool
    var models: Int

    static func decodeList(_ data: Data) -> [ProviderStatus] {
        struct Wrapper: Decodable { let providers: [ProviderStatus] }
        return (try? JSONDecoder().decode(Wrapper.self, from: data))?.providers ?? []
    }
}
