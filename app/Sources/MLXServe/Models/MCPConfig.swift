import Foundation
import OrderedCollections

/// Claude Desktop-compatible mcp.json: {"mcpServers": { "<id>": { command, args, env, disabled? } }}.
/// Stored at ~/.mlx-serve/mcp.json. The schema matches Claude Desktop so users can paste configs across apps.
/// Load also reads ~/.agents (mcp.json, .mcp.json, mcp_config.json). That directory is never written.
/// An imported server stays off until ~/.mlx-serve/mcp.json enables it.
///
/// Uses `OrderedDictionary` so the order users hand-edit in mcp.json round-trips through save/load
/// (a regular Swift Dictionary would shuffle keys, and `JSONEncoder.outputFormatting.sortedKeys`
/// would alphabetize them on every save).
struct MCPConfig: Codable, Equatable {
    var mcpServers: OrderedDictionary<String, MCPServerEntry>
    /// Keys beside `mcpServers` that this build does not model.
    var extra: [String: JSONValue] = [:]

    init(mcpServers: OrderedDictionary<String, MCPServerEntry> = [:],
         extra: [String: JSONValue] = [:]) {
        self.mcpServers = mcpServers
        self.extra = extra
    }

    // MARK: - Codable (preserves JSON key order via dynamic keyed containers)

    private enum TopKeys: String, CodingKey { case mcpServers }

    init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: TopKeys.self)
        if outer.contains(.mcpServers) {
            let inner = try outer.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .mcpServers)
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
        extra = try decoder.container(keyedBy: DynamicCodingKey.self)
            .unmodelled(known: [TopKeys.mcpServers.rawValue])
    }

    func encode(to encoder: Encoder) throws {
        var outer = encoder.container(keyedBy: TopKeys.self)
        var inner = outer.nestedContainer(keyedBy: DynamicCodingKey.self, forKey: .mcpServers)
        for (id, entry) in mcpServers {
            try inner.encode(entry, forKey: DynamicCodingKey(stringValue: id)!)
        }
        var dynamic = encoder.container(keyedBy: DynamicCodingKey.self)
        try dynamic.encodeUnmodelled(extra, known: [TopKeys.mcpServers.rawValue])
    }
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
    /// Keys inside this entry that this build does not model: another MCP host's
    /// fields, or ones a newer release added. save() rewrites the file whole.
    var extra: [String: JSONValue] = [:]

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

    private enum CodingKeys: String, CodingKey, CaseIterable {
        case command, args, url, headers, type, env, disabled, cwd
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        command = try c.decodeIfPresent(String.self, forKey: .command)
        args = try c.decodeIfPresent([String].self, forKey: .args)
        url = try c.decodeIfPresent(String.self, forKey: .url)
        headers = try c.decodeIfPresent([String: String].self, forKey: .headers)
        type = try c.decodeIfPresent(String.self, forKey: .type)
        env = try c.decodeIfPresent([String: String].self, forKey: .env)
        disabled = try c.decodeIfPresent(Bool.self, forKey: .disabled)
        cwd = try c.decodeIfPresent(String.self, forKey: .cwd)
        extra = try decoder.container(keyedBy: DynamicCodingKey.self)
            .unmodelled(known: Set(CodingKeys.allCases.map(\.stringValue)))
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(command, forKey: .command)
        try c.encodeIfPresent(args, forKey: .args)
        try c.encodeIfPresent(url, forKey: .url)
        try c.encodeIfPresent(headers, forKey: .headers)
        try c.encodeIfPresent(type, forKey: .type)
        try c.encodeIfPresent(env, forKey: .env)
        try c.encodeIfPresent(disabled, forKey: .disabled)
        try c.encodeIfPresent(cwd, forKey: .cwd)
        var dynamic = encoder.container(keyedBy: DynamicCodingKey.self)
        try dynamic.encodeUnmodelled(extra, known: Set(CodingKeys.allCases.map(\.stringValue)))
    }

    var isEnabled: Bool { !(disabled ?? false) }

    enum Transport { case stdio, http, malformed }
    var transport: Transport {
        if (command?.isEmpty == false) { return .stdio }
        if (url?.isEmpty == false) { return .http }
        return .malformed
    }
}

/// Servers read from the agents directory. `origins` names the file each id came from.
struct AgentsMCPImport {
    var servers: OrderedDictionary<String, MCPServerEntry> = [:]
    var origins: [String: String] = [:]
}

enum MCPConfigStore {
    /// Files in the agents directory that use the `mcpServers` shape, applied in
    /// this order so `mcp.json` (the commons file) wins among them. The app's own
    /// mcp.json wins over all three. Subdirectories are not scanned: `skills/`
    /// holds skill packages, not server definitions.
    static let agentsConfigFileNames = ["mcp_config.json", ".mcp.json", "mcp.json"]

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

    /// Where shared agent MCP configs live. `MCP_AGENTS_DIR` relocates or, when
    /// empty, turns the merge off. `AGENT_SYNC_HOME` is the commons relocation.
    /// A sandboxed owned file (`MCP_CONFIG_PATH`, set by tests) does not also
    /// read the developer's real directory unless `MCP_AGENTS_DIR` is set.
    static func agentsDirectory() -> String? {
        if let override = ProcessInfo.processInfo.environment["MCP_AGENTS_DIR"] {
            if override.isEmpty { return nil }
            return (override as NSString).expandingTildeInPath
        }
        if let sandbox = ProcessInfo.processInfo.environment["MCP_CONFIG_PATH"], !sandbox.isEmpty {
            return nil
        }
        if let home = ProcessInfo.processInfo.environment["AGENT_SYNC_HOME"], !home.isEmpty {
            return (home as NSString).expandingTildeInPath
        }
        return NSString(string: "~/.agents").expandingTildeInPath
    }

    static func agentsOriginLabel(_ fileName: String) -> String {
        let dir = agentsDirectory() ?? NSString(string: "~/.agents").expandingTildeInPath
        return "\((dir as NSString).abbreviatingWithTildeInPath)/\(fileName)"
    }

    /// The owned file alone. Save compares against this so an unchanged import
    /// is not copied into ~/.mlx-serve/mcp.json.
    static func loadOwned() -> MCPConfig {
        loadFile(at: path) ?? MCPConfig()
    }

    static func load() -> MCPConfig {
        let imported = agentsImport().servers
        var owned = loadOwned()
        var merged = imported
        // No owned row means this app has not opted in. The commons file's missing
        // `disabled` flag would otherwise spawn the server on the next MCP turn.
        for (id, entry) in merged where owned.mcpServers[id] == nil && entry.isEnabled {
            var shown = entry
            shown.disabled = true
            merged[id] = shown
        }
        for (id, entry) in owned.mcpServers {
            merged[id] = entry
        }
        owned.mcpServers = merged
        return owned
    }

    static func agentsImport() -> AgentsMCPImport {
        guard let dir = agentsDirectory() else { return AgentsMCPImport() }
        var imported = AgentsMCPImport()
        for name in agentsConfigFileNames {
            let file = (dir as NSString).appendingPathComponent(name)
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file, isDirectory: &isDir), !isDir.boolValue,
                  let cfg = loadFile(at: file) else { continue }
            for (id, entry) in cfg.mcpServers {
                imported.servers[id] = entry
                imported.origins[id] = name
            }
        }
        return imported
    }

    private static func loadFile(at path: String) -> MCPConfig? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let cfg = try? JSONDecoder().decode(MCPConfig.self, from: data) else {
            return nil
        }
        // JSONDecoder doesn't preserve JSON object key order — it parses into a hash-backed
        // store. We re-derive the user's hand-edited order by scanning the raw text and
        // reordering the OrderedDictionary to match.
        let order = extractMcpServersKeyOrder(from: data)
        return MCPConfig(mcpServers: reorder(cfg.mcpServers, by: order), extra: cfg.extra)
    }

    /// `disabled: false` and a missing flag are the same enabled state, so
    /// flipping a toggle back on does not snapshot the commons entry.
    static func effectivelySame(_ a: MCPServerEntry, _ b: MCPServerEntry) -> Bool {
        var x = a
        var y = b
        if x.disabled == false { x.disabled = nil }
        if y.disabled == false { y.disabled = nil }
        return x == y
    }

    /// The shown-but-not-opted-in view of a commons server. Saving it must not
    /// copy the server into ~/.mlx-serve/mcp.json. An enabled entry is an opt-in.
    static func isUnchangedImport(_ saved: MCPServerEntry, _ foreign: MCPServerEntry) -> Bool {
        guard !saved.isEnabled else { return false }
        var shown = saved
        shown.disabled = foreign.disabled
        return effectivelySame(shown, foreign)
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
        try write(ownedSnapshot(of: config))
    }

    /// Drop servers that still match the agents directory. An owned id, or an
    /// imported one the user changed (usually `disabled`), stays in the app file.
    static func ownedSnapshot(of config: MCPConfig) -> MCPConfig {
        let imported = agentsImport().servers
        let previous = loadOwned()
        var kept: OrderedDictionary<String, MCPServerEntry> = [:]
        for id in previous.mcpServers.keys {
            guard let entry = config.mcpServers[id] else { continue }
            if let foreign = imported[id], isUnchangedImport(entry, foreign) { continue }
            kept[id] = entry
        }
        for (id, entry) in config.mcpServers where previous.mcpServers[id] == nil {
            if let foreign = imported[id], isUnchangedImport(entry, foreign) { continue }
            kept[id] = entry
        }
        var out = config
        out.mcpServers = kept
        return out
    }

    private static func write(_ config: MCPConfig) throws {
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
            out += "\n\n  }"
        } else {
            for (i, e) in entryJSON.enumerated() {
                out += "\n    \"\(e.id)\" : \(e.body)"
                if i < entryJSON.count - 1 { out += "," }
            }
            out += "\n  }"
        }
        // Keys beside mcpServers that this build does not model: the file is
        // rewritten whole, so leaving them out deletes them from the config.
        for (key, value) in config.extra.sorted(by: { $0.key < $1.key })
        where key != "mcpServers" && DynamicCodingKey(stringValue: key) != nil {
            let keyJSON = String(data: try valueEncoder.encode(key), encoding: .utf8) ?? "\"\""
            let valueJSON = String(data: try valueEncoder.encode(value), encoding: .utf8) ?? "null"
            out += ",\n  \(keyJSON) : \(valueJSON)"
        }
        out += "\n}\n"

        let parent = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: parent, withIntermediateDirectories: true)
        try out.data(using: .utf8)!.write(to: URL(fileURLWithPath: path), options: .atomic)
    }
}
