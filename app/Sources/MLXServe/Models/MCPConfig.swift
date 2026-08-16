import Foundation
import OrderedCollections

/// Claude Desktop-compatible mcp.json: {"mcpServers": { "<id>": { command, args, env, disabled? } }}.
/// Stored at ~/.mlx-serve/mcp.json. The schema matches Claude Desktop so users can paste configs across apps.
///
/// Uses `OrderedDictionary` so the order users hand-edit in mcp.json round-trips through save/load
/// (a regular Swift Dictionary would shuffle keys, and `JSONEncoder.outputFormatting.sortedKeys`
/// would alphabetize them on every save).
struct MCPConfig: Codable, Equatable {
    var mcpServers: OrderedDictionary<String, MCPServerEntry>

    init(mcpServers: OrderedDictionary<String, MCPServerEntry> = [:]) {
        self.mcpServers = mcpServers
    }

    // MARK: - Codable (preserves JSON key order via dynamic keyed containers)

    private enum TopKeys: String, CodingKey { case mcpServers }

    init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: TopKeys.self)
        if outer.contains(.mcpServers) {
            let inner = try outer.nestedContainer(keyedBy: DynamicKey.self, forKey: .mcpServers)
            // Decode the values into a regular dict first — `allKeys` order is implementation-defined
            // and Foundation's JSONDecoder shuffles them via an internal hash. We re-order in load().
            var byKey: [String: MCPServerEntry] = [:]
            for key in inner.allKeys {
                byKey[key.stringValue] = try inner.decode(MCPServerEntry.self, forKey: key)
            }
            var ordered: OrderedDictionary<String, MCPServerEntry> = [:]
            for (k, v) in byKey { ordered[k] = v }
            self.mcpServers = ordered
        } else {
            self.mcpServers = [:]
        }
    }

    func encode(to encoder: Encoder) throws {
        var outer = encoder.container(keyedBy: TopKeys.self)
        var inner = outer.nestedContainer(keyedBy: DynamicKey.self, forKey: .mcpServers)
        for (id, entry) in mcpServers {
            try inner.encode(entry, forKey: DynamicKey(stringValue: id)!)
        }
    }
}

/// Stand-in for "any string can be a key". Used to drive JSON object encode/decode without an
/// enum of fixed key names — server ids are user-defined.
private struct DynamicKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

struct MCPServerEntry: Codable, Equatable {
    /// Stdio transport — the spawned executable. Optional because HTTP-transport entries (with `url`)
    /// have no command. If both `command` and `url` are missing, the entry is malformed.
    var command: String?
    /// Stdio transport — args passed to `command`.
    var args: [String]?
    /// HTTP transport — the server URL. Connected by `MCPManager.connectHTTP` (no subprocess,
    /// the SDK's `HTTPClientTransport` with SSE streaming).
    var url: String?
    /// HTTP transport — request headers sent on EVERY request (`Authorization` tokens, API
    /// versions). This was an unknown key once: JSONDecoder dropped it silently, so the token
    /// never reached the server, and the next save DELETED it from the user's hand-edited file.
    var headers: [String: String]?
    /// Pass-through transport tag other MCP hosts write ("remote", "http", "sse"). We do not
    /// interpret it — transport derives from `url`/`command` presence — but a field we don't
    /// model is a field save() strips, so it is kept for round-trip fidelity.
    var type: String?
    var env: [String: String]?
    /// Our extension (Claude Desktop ignores unknown keys). Treat missing/false as enabled.
    var disabled: Bool?
    /// Optional working directory for the spawned subprocess. Tilde-expanded at use time.
    /// When omitted, MCPManager defaults to `~/.mlx-serve/workspace` so filesystem/shell servers
    /// land in a sane location by default rather than wherever macOS launched the .app from.
    var cwd: String?

    init(command: String? = nil, args: [String]? = nil, url: String? = nil,
         headers: [String: String]? = nil, type: String? = nil,
         env: [String: String]? = nil, disabled: Bool? = nil, cwd: String? = nil) {
        self.command = command
        self.args = args
        self.url = url
        self.headers = headers
        self.type = type
        self.env = env
        self.disabled = disabled
        self.cwd = cwd
    }

    var isEnabled: Bool { !(disabled ?? false) }

    enum Transport { case stdio, http, malformed }
    var transport: Transport {
        if (command?.isEmpty == false) { return .stdio }
        if (url?.isEmpty == false) { return .http }
        return .malformed
    }
}

enum MCPConfigStore {
    /// Default path is `~/.mlx-serve/mcp.json`. Tests override via `MCP_CONFIG_PATH` env var to avoid
    /// clobbering the user's real config — discovered the hard way after a test run littered the user's
    /// mcp.json with `bogus` and `nopkg` entries.
    static var path: String {
        if let p = ProcessInfo.processInfo.environment["MCP_CONFIG_PATH"], !p.isEmpty {
            return (p as NSString).expandingTildeInPath
        }
        let dir = NSString(string: "~/.mlx-serve").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return (dir as NSString).appendingPathComponent("mcp.json")
    }

    static func load() -> MCPConfig {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let cfg = try? JSONDecoder().decode(MCPConfig.self, from: data) else {
            return MCPConfig()
        }
        // JSONDecoder doesn't preserve JSON object key order — it parses into a hash-backed
        // store. We re-derive the user's hand-edited order by scanning the raw text and
        // reordering the OrderedDictionary to match.
        let order = extractMcpServersKeyOrder(from: data)
        return MCPConfig(mcpServers: reorder(cfg.mcpServers, by: order))
    }

    /// Reorder an OrderedDictionary so its keys appear in the given source-order list. Keys present
    /// in `dict` but missing from `order` (defensive: shouldn't happen) are appended at the end.
    private static func reorder(
        _ dict: OrderedDictionary<String, MCPServerEntry>,
        by order: [String]
    ) -> OrderedDictionary<String, MCPServerEntry> {
        var out: OrderedDictionary<String, MCPServerEntry> = [:]
        for key in order {
            if let v = dict[key] { out[key] = v }
        }
        for (k, v) in dict where out[k] == nil {
            out[k] = v
        }
        return out
    }

    /// Walk the raw mcp.json bytes and return the keys of the top-level `mcpServers` object in source
    /// order. Handles nested objects/arrays and string escapes; tolerant of extra whitespace.
    /// Returns an empty array if the structure isn't found — caller falls back to whatever order the
    /// JSONDecoder gave us.
    static func extractMcpServersKeyOrder(from data: Data) -> [String] {
        guard let text = String(data: data, encoding: .utf8) else { return [] }
        let bytes = Array(text.utf8)
        // Locate `"mcpServers"` and skip past the colon to its opening `{`.
        guard let labelStart = text.range(of: "\"mcpServers\"") else { return [] }
        var i = text.utf8.distance(from: text.utf8.startIndex, to: labelStart.upperBound.samePosition(in: text.utf8)!)
        // Advance to the next `{`.
        while i < bytes.count, bytes[i] != UInt8(ascii: "{") { i += 1 }
        guard i < bytes.count else { return [] }
        i += 1  // step past the opening brace; we're now at depth 0 *inside* mcpServers

        var keys: [String] = []
        var depth = 0
        var pendingKey: String? = nil          // last completed string at depth 0 awaiting its `:`
        var inString = false
        var escape = false
        var keyBuf: [UInt8] = []
        var sawColon = false                   // the last seen `:` belongs to pendingKey

        while i < bytes.count {
            let c = bytes[i]
            if escape {
                if inString { keyBuf.append(c) }
                escape = false
                i += 1; continue
            }
            if c == UInt8(ascii: "\\") {
                escape = true
                if inString { keyBuf.append(c) }
                i += 1; continue
            }
            if c == UInt8(ascii: "\"") {
                if inString {
                    // String just ended — at depth 0, this is a key candidate.
                    if depth == 0 && !sawColon {
                        pendingKey = String(decoding: keyBuf, as: UTF8.self)
                    }
                    keyBuf.removeAll(keepingCapacity: true)
                    inString = false
                } else {
                    inString = true
                }
                i += 1; continue
            }
            if inString { keyBuf.append(c); i += 1; continue }

            switch c {
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                if depth == 0 && pendingKey != nil && sawColon {
                    keys.append(pendingKey!)   // commit when we enter the value object/array
                    pendingKey = nil
                }
                depth += 1
            case UInt8(ascii: "}"), UInt8(ascii: "]"):
                if depth == 0 { return keys }  // end of mcpServers object
                depth -= 1
            case UInt8(ascii: ":"):
                if depth == 0 { sawColon = true }
            case UInt8(ascii: ","):
                if depth == 0 {
                    // Commit primitive values (string/number/bool/null) — value already consumed
                    // before this comma.
                    if let k = pendingKey, sawColon { keys.append(k) }
                    pendingKey = nil
                    sawColon = false
                }
            default: break
            }
            i += 1
        }
        return keys
    }

    static func save(_ config: MCPConfig) throws {
        // Foundation's JSONEncoder shuffles top-level KeyedContainer keys (it uses a hash-backed
        // store and ignores the encode call order). To actually preserve the user's hand-edited
        // mcp.json order, encode each server's value individually and stitch the outer object
        // together as a string. The per-entry JSON is still produced by JSONEncoder so escaping,
        // formatting, and value structure remain bulletproof.
        let valueEncoder = JSONEncoder()
        valueEncoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes, .sortedKeys]

        var entryJSON: [(id: String, body: String)] = []
        for (id, entry) in config.mcpServers {
            let data = try valueEncoder.encode(entry)
            let raw = String(data: data, encoding: .utf8) ?? "{}"
            // Indent the per-entry JSON by 4 spaces so it lines up under the outer object.
            let indented = raw
                .split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated()
                .map { idx, line in idx == 0 ? String(line) : "    \(line)" }
                .joined(separator: "\n")
            entryJSON.append((id, indented))
        }

        var out = "{\n  \"mcpServers\" : {"
        if entryJSON.isEmpty {
            out += "\n\n  }\n}\n"
        } else {
            for (i, e) in entryJSON.enumerated() {
                out += "\n    \"\(e.id)\" : \(e.body)"
                if i < entryJSON.count - 1 { out += "," }
            }
            out += "\n  }\n}\n"
        }

        let parent = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true)
        try out.data(using: .utf8)!.write(to: URL(fileURLWithPath: path), options: .atomic)
    }
}
