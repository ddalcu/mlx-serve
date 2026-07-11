import XCTest
import Foundation
import MCP
@testable import MLXCore

final class MCPTests: XCTestCase {

    // MARK: - Tool name namespacing

    func testNamespacedNameRoundTrip() {
        let name = MCPManager.namespacedName(server: "github", tool: "search_repositories")
        XCTAssertEqual(name, "github__search_repositories")
        let parsed = MCPManager.parseNamespacedName(name)
        XCTAssertEqual(parsed?.server, "github")
        XCTAssertEqual(parsed?.tool, "search_repositories")
    }

    func testParseNamespacedNameRejectsBareName() {
        XCTAssertNil(MCPManager.parseNamespacedName("just_a_name"))
        XCTAssertNil(MCPManager.parseNamespacedName("__notool"))
        XCTAssertNil(MCPManager.parseNamespacedName("noserver__"))
    }

    func testNamespacedNamePreservesUnderscoresInToolName() {
        // Tools with multiple underscores should still parse — split on the FIRST `__`.
        let name = MCPManager.namespacedName(server: "fs", tool: "read_text_file")
        XCTAssertEqual(name, "fs__read_text_file")
        let parsed = MCPManager.parseNamespacedName(name)
        XCTAssertEqual(parsed?.tool, "read_text_file")
    }

    // MARK: - Argument conversion

    func testConvertArgumentsFromRawJSONPreservesTypes() {
        let raw = #"{"limit": 5, "verbose": true, "name": "alice"}"#
        let args = MCPManager.convertArguments(rawArguments: raw, fallback: [:])
        XCTAssertEqual(args["limit"], .int(5))
        XCTAssertEqual(args["verbose"], .bool(true))
        XCTAssertEqual(args["name"], .string("alice"))
    }

    func testConvertArgumentsFallsBackToStringMap() {
        // No raw JSON, only a string→string fallback. Numeric strings should parse as numbers,
        // truly non-JSON strings should stay as strings.
        let args = MCPManager.convertArguments(rawArguments: "", fallback: [
            "limit": "5",
            "name": "alice"
        ])
        XCTAssertEqual(args["limit"], .int(5))
        XCTAssertEqual(args["name"], .string("alice"))
    }

    /// Regression: empty args used to return nil, which made the swift-sdk omit the JSON-RPC
    /// `arguments` field. Servers with strict schemas (e.g. Azure DevOps) then rejected the call
    /// with "Required: received undefined". We now always send an object — empty {} is fine.
    func testConvertArgumentsEmptyReturnsEmptyObjectNotNil() {
        let fromBlank = MCPManager.convertArguments(rawArguments: "", fallback: [:])
        XCTAssertTrue(fromBlank.isEmpty, "Empty input should yield an empty object, not nil")

        let fromEmptyObject = MCPManager.convertArguments(rawArguments: "{}", fallback: [:])
        XCTAssertTrue(fromEmptyObject.isEmpty, "{} input should yield an empty object")
    }

    // MARK: - MCPConfig Codable round-trip (Claude Desktop format)

    /// Regression: mcp.json should round-trip in the exact order the user hand-edited.
    /// We use an OrderedDictionary internally + a text-scan key-order recovery on load + manual
    /// JSON assembly on save (Foundation's JSONEncoder/Decoder both shuffle keys via a hash store).
    func testMCPConfigPreservesKeyOrderThroughSaveAndLoad() throws {
        let tmp = NSTemporaryDirectory().appending("mcp-order-\(UUID().uuidString).json")
        setenv("MCP_CONFIG_PATH", tmp, 1)
        defer { unsetenv("MCP_CONFIG_PATH"); try? FileManager.default.removeItem(atPath: tmp) }

        // Deliberately non-alphabetical source order — exposes any sort/hash shuffle.
        let source = #"""
        {
          "mcpServers": {
            "shell":      { "command": "npx", "args": ["-y", "shell"] },
            "github":     { "command": "npx", "args": ["-y", "github"] },
            "azure-devops": { "command": "npx", "args": ["-y", "ado"] },
            "htmlhost":   { "url": "https://example.com/mcp" }
          }
        }
        """#
        try source.data(using: .utf8)!.write(to: URL(fileURLWithPath: tmp))

        // Loading must preserve source order.
        let loaded = MCPConfigStore.load()
        XCTAssertEqual(Array(loaded.mcpServers.keys), ["shell", "github", "azure-devops", "htmlhost"])

        // Saving must emit keys in OrderedDictionary order — verify by reading the raw file back.
        try MCPConfigStore.save(loaded)
        let written = try String(contentsOfFile: tmp, encoding: .utf8)
        let positions = ["shell", "github", "azure-devops", "htmlhost"].map { id -> Int in
            (written.range(of: "\"\(id)\"")?.lowerBound).map { written.distance(from: written.startIndex, to: $0) } ?? -1
        }
        XCTAssertEqual(positions, positions.sorted(),
                       "Saved file must keep keys in original order: positions=\(positions)\nfile:\n\(written)")
        XCTAssertFalse(positions.contains(-1))

        // And: load → save → load again should be a fixed point.
        let reloaded = MCPConfigStore.load()
        XCTAssertEqual(Array(reloaded.mcpServers.keys), Array(loaded.mcpServers.keys))
    }

    func testExtractMcpServersKeyOrderHandlesNestedStructures() {
        // Strings with quotes/escapes, nested objects/arrays, primitive values — scanner must skip
        // through them and only capture the top-level mcpServers keys.
        let json = #"""
        {
          "other": "ignored",
          "mcpServers": {
            "a": { "args": ["x", "y", "{not-a-key}"], "env": { "K": "v: x" } },
            "b": { "url": "https://x" },
            "c": { "args": ["with \"escaped\" quote"] }
          }
        }
        """#
        let order = MCPConfigStore.extractMcpServersKeyOrder(from: json.data(using: .utf8)!)
        XCTAssertEqual(order, ["a", "b", "c"])
    }

    func testMCPConfigClaudeDesktopFormatRoundTrip() throws {
        let json = #"""
        {
          "mcpServers": {
            "github": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-github"],
              "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_test" }
            },
            "filesystem": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/david"]
            }
          }
        }
        """#
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(MCPConfig.self, from: data)
        XCTAssertEqual(config.mcpServers.count, 2)
        XCTAssertEqual(config.mcpServers["github"]?.command, "npx")
        XCTAssertEqual(config.mcpServers["github"]?.env?["GITHUB_PERSONAL_ACCESS_TOKEN"], "ghp_test")
        XCTAssertEqual(config.mcpServers["filesystem"]?.args?.last, "/Users/david")
        XCTAssertNil(config.mcpServers["github"]?.disabled)

        // Re-encode and decode again — should be lossless for fields we model.
        // NB: bare Codable does NOT preserve key order (Foundation's JSONDecoder shuffles object keys
        // via an internal hash; order is only recovered through MCPConfigStore.load's text scan — see
        // testMCPConfigPreservesKeyOrderThroughSaveAndLoad). OrderedDictionary's `==` is order-sensitive,
        // so compare as a plain Dictionary here to assert value-losslessness without depending on hash order.
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(MCPConfig.self, from: encoded)
        XCTAssertEqual(
            Dictionary(uniqueKeysWithValues: decoded.mcpServers.map { ($0, $1) }),
            Dictionary(uniqueKeysWithValues: config.mcpServers.map { ($0, $1) }))
    }

    /// HTTP-transport entries have only a `url` field — no command/args. Must decode without dropping
    /// the entry. Regression: previously the whole mcp.json failed to decode if any entry was URL-only,
    /// because command/args were non-optional.
    func testMCPConfigDecodesHTTPTransportEntry() throws {
        let json = #"""
        {
          "mcpServers": {
            "htmlhost": { "url": "https://htmlhost.jax.workers.dev/mcp" },
            "github":   { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] }
          }
        }
        """#
        let data = json.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MCPConfig.self, from: data)
        XCTAssertEqual(cfg.mcpServers.count, 2, "Both entries should decode (HTTP + stdio)")
        let http = cfg.mcpServers["htmlhost"]
        XCTAssertEqual(http?.url, "https://htmlhost.jax.workers.dev/mcp")
        XCTAssertNil(http?.command)
        XCTAssertEqual(http?.transport, .http)
        let stdio = cfg.mcpServers["github"]
        XCTAssertEqual(stdio?.command, "npx")
        XCTAssertEqual(stdio?.transport, .stdio)
    }

    func testMCPServerEntryTransportClassification() {
        XCTAssertEqual(MCPServerEntry(command: "npx", args: ["-y", "x"]).transport, .stdio)
        XCTAssertEqual(MCPServerEntry(url: "https://example.com/mcp").transport, .http)
        XCTAssertEqual(MCPServerEntry().transport, .malformed)
        // Empty strings should not count.
        XCTAssertEqual(MCPServerEntry(command: "", url: "").transport, .malformed)
    }

    func testMCPServerEntryDisabledFlag() {
        let enabled = MCPServerEntry(command: "x", args: [], env: nil, disabled: nil)
        let off = MCPServerEntry(command: "x", args: [], env: nil, disabled: true)
        XCTAssertTrue(enabled.isEnabled)
        XCTAssertFalse(off.isEnabled)
    }

    // MARK: - Catalog completeness

    func testCatalogIncludesAllPromisedServers() {
        let ids = Set(MCPCatalog.entries.map(\.id))
        let expected: Set<String> = [
            "github", "azure-devops", "dbhub", "docker", "kubernetes",
            "playwright", "slack", "notion", "filesystem", "shell"
        ]
        XCTAssertEqual(ids, expected, "Catalog should match the marketplace promise: \(expected)")
    }

    func testCatalogEntriesAreWellFormed() {
        for entry in MCPCatalog.entries {
            XCTAssertFalse(entry.id.isEmpty, "Entry has empty id")
            XCTAssertFalse(entry.name.isEmpty, "\(entry.id) has empty name")
            XCTAssertFalse(entry.command.isEmpty, "\(entry.id) has empty command")
            XCTAssertFalse((entry.args ?? []).isEmpty, "\(entry.id) has empty args")
            // Arg placeholders must appear in args (so materialize() can find them).
            for input in entry.inputs {
                if case .arg(let placeholder) = input.kind {
                    XCTAssertTrue((entry.args ?? []).contains(placeholder),
                                  "\(entry.id): arg placeholder \(placeholder) missing from args")
                }
            }
        }
    }

    // MARK: - Catalog materialize / extract round-trip

    func testCatalogMaterializeAndExtractRoundTrip() {
        guard let github = MCPCatalog.entry(for: "github") else { return XCTFail("github missing") }
        let entry = github.materialize(values: ["github_token": "ghp_abc123"])
        XCTAssertEqual(entry.command, "npx")
        XCTAssertEqual(entry.env?["GITHUB_PERSONAL_ACCESS_TOKEN"], "ghp_abc123")

        let extracted = github.extractValues(from: entry)
        XCTAssertEqual(extracted["github_token"], "ghp_abc123")
    }

    func testCatalogMaterializeReplacesArgPlaceholder() {
        guard let dbhub = MCPCatalog.entry(for: "dbhub") else { return XCTFail("dbhub missing") }
        let dsn = "postgres://u:p@host:5432/db"
        let entry = dbhub.materialize(values: ["dsn": dsn])
        XCTAssertTrue((entry.args ?? []).contains(dsn), "DSN should be spliced into args; got \(entry.args)")
        XCTAssertFalse((entry.args ?? []).contains("<DSN>"), "Placeholder should be replaced; got \(entry.args)")

        let extracted = dbhub.extractValues(from: entry)
        XCTAssertEqual(extracted["dsn"], dsn)
    }

    /// ADO defaults to interactive (browser) auth — no --authentication flag in the base args.
    /// The PAT field is optional; filling it BOTH base64-encodes the token AND appends `--authentication pat`.
    func testAzureDevOpsDefaultsToInteractiveAuth() {
        guard let ado = MCPCatalog.entry(for: "azure-devops") else { return XCTFail("azure-devops missing") }
        // Base args should NOT pin an auth mode — that lets the server use its interactive default.
        XCTAssertFalse((ado.args ?? []).contains("--authentication"),
                       "ADO base args should not pre-set an auth mode; got \(ado.args)")

        // PAT field exists but is OPTIONAL.
        let patInput = ado.inputs.first { $0.id == "ado_pat" }
        XCTAssertNotNil(patInput)
        XCTAssertFalse(patInput?.required ?? true, "PAT should be optional (browser is the recommended default)")

        // Materialize with org only → no env, no auth flag (uses interactive browser flow).
        let interactive = ado.materialize(values: ["ado_org": "contoso"])
        XCTAssertTrue((interactive.args ?? []).contains("contoso"))
        XCTAssertFalse((interactive.args ?? []).contains("--authentication"),
                       "Empty PAT should leave args interactive; got \(interactive.args)")
        XCTAssertNil(interactive.env?["PERSONAL_ACCESS_TOKEN"])
    }

    func testAzureDevOpsPATFillsAuthFlagAndEncodesToken() {
        guard let ado = MCPCatalog.entry(for: "azure-devops") else { return XCTFail("azure-devops missing") }
        let entry = ado.materialize(values: ["ado_org": "contoso", "ado_pat": "abcdef123"])
        XCTAssertTrue((entry.args ?? []).contains("contoso"))
        XCTAssertTrue((entry.args ?? []).contains("--authentication"),
                      "Filling PAT should append --authentication; got \(entry.args)")
        XCTAssertTrue((entry.args ?? []).contains("pat"),
                      "Auth flag should be 'pat'; got \(entry.args)")

        guard let encoded = entry.env?["PERSONAL_ACCESS_TOKEN"] else {
            return XCTFail("PERSONAL_ACCESS_TOKEN should be set; got env=\(entry.env ?? [:])")
        }
        // The catalog uses leftSide "x" → base64("x:abcdef123")
        let expected = Data("x:abcdef123".utf8).base64EncodedString()
        XCTAssertEqual(encoded, expected)

        // Re-extracting the entry should NOT reveal the raw PAT (encoded inputs are write-only).
        let extracted = ado.extractValues(from: entry)
        XCTAssertNil(extracted["ado_pat"], "Encoded secrets must not round-trip back to the UI")
        XCTAssertEqual(extracted["ado_org"], "contoso")
    }

    func testCatalogEncodingHelperRoundTripsBase64Pair() {
        let encoded = MCPCatalogEntry.encode("token123", with: .base64Pair(leftSide: "user@example.com"))
        let decoded = String(data: Data(base64Encoded: encoded)!, encoding: .utf8)
        XCTAssertEqual(decoded, "user@example.com:token123")
    }

    // MARK: - Spawn pre-flight & error messages

    // The preflight moved onto `MCPSpawner`, because "does this command exist"
    // depends on WHERE the server will run — the host, or the sandbox guest.
    // HostMCPSpawner is compiled out of the App Store build.
    #if !MAS_BUILD
    func testCommandExistsTrueForKnownBinary() async {
        // /bin/sh always exists on macOS, and `command -v sh` resolves it via the login shell.
        let exists = await HostMCPSpawner().commandExists("sh")
        XCTAssertTrue(exists)
    }

    func testCommandExistsFalseForGarbageName() async {
        let exists = await HostMCPSpawner().commandExists("definitely-not-a-real-binary-xyz-9z")
        XCTAssertFalse(exists)
    }
    #endif

    func testInstallHintMentionsNodeForNpx() {
        let hint = MCPManager.installHint(for: "npx")
        XCTAssertTrue(hint.localizedCaseInsensitiveContains("node"),
                      "Hint for npx should point users at Node.js; got: \(hint)")
    }

    func testInstallHintMentionsDockerForDocker() {
        let hint = MCPManager.installHint(for: "docker")
        XCTAssertTrue(hint.localizedCaseInsensitiveContains("docker"))
    }

    func testInstallHintGenericFallbackMentionsCommand() {
        let hint = MCPManager.installHint(for: "obscure-tool")
        XCTAssertTrue(hint.contains("obscure-tool"),
                      "Generic hint should at least name the missing command; got: \(hint)")
    }

    func testCommandNotFoundErrorMessageIncludesCommandAndHint() {
        let err = MCPSpawnError.commandNotFound(command: "npx", hint: "Install Node.js")
        let msg = err.errorDescription ?? ""
        XCTAssertTrue(msg.contains("npx"), msg)
        XCTAssertTrue(msg.contains("Install Node.js"), msg)
        XCTAssertTrue(msg.contains("not found"), msg)
    }

    func testServerExitedEarlyIncludesStderrTail() {
        let err = MCPSpawnError.serverExitedEarly(status: 127, stderr: "zsh: command not found: npx")
        let msg = err.errorDescription ?? ""
        XCTAssertTrue(msg.contains("127"))
        XCTAssertTrue(msg.contains("command not found"))
    }

    // MARK: - StderrBox concurrency

    func testStderrBoxAppendAndSnapshotRoundTrip() {
        let box = StderrBox()
        box.append("hello ")
        box.append("world\n")
        XCTAssertEqual(box.snapshot(), "hello world\n")
    }

    // MARK: - Working directory defaults

    /// MCP servers spawned without a per-entry `cwd` should land in `~/.mlx-serve/workspace` so
    /// filesystem/shell servers don't anchor at wherever macOS launched the .app from.
    func testResolveWorkingDirectoryDefaultsToWorkspace() {
        let path = MCPManager.resolveWorkingDirectory(nil)
        let expected = NSString(string: "~/.mlx-serve/workspace").expandingTildeInPath
        XCTAssertEqual(path, expected)
        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: path, isDirectory: &isDir))
        XCTAssertTrue(isDir.boolValue, "Workspace path should be a directory")
    }

    func testResolveWorkingDirectoryHonorsOverride() {
        let tmp = NSTemporaryDirectory().appending("mcp-cwd-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        let path = MCPManager.resolveWorkingDirectory(tmp)
        XCTAssertEqual(path, tmp)
        XCTAssertTrue(FileManager.default.fileExists(atPath: path), "Should auto-create the override path")
    }

    func testResolveWorkingDirectoryExpandsTilde() {
        let path = MCPManager.resolveWorkingDirectory("~/.mlx-serve/workspace")
        XCTAssertFalse(path.hasPrefix("~"), "Tilde must be expanded; got \(path)")
        XCTAssertTrue(path.hasPrefix("/"), "Result must be absolute; got \(path)")
    }

    /// MCPManager.defaultCwd should be picked up by spawn-time resolution when the entry has none,
    /// so newly-spawned MCP servers follow the user's chat-session working directory.
    @MainActor func testManagerDefaultCwdFlowsIntoEntryResolution() {
        let manager = MCPManager()
        let tmp = NSTemporaryDirectory().appending("mcp-defaultcwd-\(UUID().uuidString)")
        manager.defaultCwd = tmp
        // The actual end-to-end path goes through spawnAndConnect, which is hard to drive in a unit
        // test. Instead, verify the resolver agrees with the resolution order: entry.cwd ?? default.
        let entryNoCwd = MCPServerEntry(command: "x", args: [], env: nil, disabled: false, cwd: nil)
        let entryWithCwd = MCPServerEntry(command: "x", args: [], env: nil, disabled: false, cwd: "/tmp/from-entry")

        // What spawnAndConnect computes:
        let resolvedFromDefault = MCPManager.resolveWorkingDirectory(entryNoCwd.cwd ?? manager.defaultCwd)
        XCTAssertEqual(resolvedFromDefault, tmp, "Empty entry.cwd + non-nil defaultCwd should fall through to defaultCwd")

        let resolvedFromEntry = MCPManager.resolveWorkingDirectory(entryWithCwd.cwd ?? manager.defaultCwd)
        XCTAssertEqual(resolvedFromEntry, "/tmp/from-entry", "Per-entry cwd must outrank manager defaultCwd")

        // Cleanup
        try? FileManager.default.removeItem(atPath: tmp)
    }

    /// MCPServerEntry's new `cwd` field round-trips through Codable cleanly.
    func testMCPServerEntryRoundTripsCwdField() throws {
        let entry = MCPServerEntry(command: "npx", args: ["-y", "x"], env: nil, disabled: false, cwd: "/tmp/foo")
        let data = try JSONEncoder().encode(entry)
        let decoded = try JSONDecoder().decode(MCPServerEntry.self, from: data)
        XCTAssertEqual(decoded.cwd, "/tmp/foo")
    }

    func testStderrBoxCapsAtBufferSize() {
        let box = StderrBox()
        box.append(String(repeating: "a", count: 5000))
        box.append(String(repeating: "b", count: 5000))
        let snap = box.snapshot()
        XCTAssertEqual(snap.count, 8000, "Buffer should cap to 8000 chars; got \(snap.count)")
        XCTAssertTrue(snap.hasSuffix("b"), "Tail should be the most recent bytes")
    }

    // MARK: - Tools JSON merging

    func testCombinedToolsJSONMergesAgentAndMCP() {
        let agent = #"[{"type":"function","function":{"name":"shell"}}]"#
        let mcp = #"[{"type":"function","function":{"name":"github__list_repos"}}]"#

        // Both modes on: arrays should be concatenated.
        let merged = ChatTurnEngine.combinedToolsJSON(agentMode: true, mcpToolsJSON: mcp)
        XCTAssertNotNil(merged)
        let mergedData = merged!.data(using: .utf8)!
        let arr = try? JSONSerialization.jsonObject(with: mergedData) as? [[String: Any]]
        XCTAssertEqual(arr?.count, 18, "Expected 17 agent tools (incl. createTask, killProcess/readProcessOutput/listProcesses, generate_image/audio/video) + 1 MCP tool, got \(arr?.count ?? 0)")
        XCTAssertTrue(merged!.contains("github__list_repos"))
        XCTAssertTrue(merged!.contains("shell"))

        // MCP only.
        let mcpOnly = ChatTurnEngine.combinedToolsJSON(agentMode: false, mcpToolsJSON: mcp)
        XCTAssertEqual(mcpOnly, mcp)

        // Agent only.
        let agentOnly = ChatTurnEngine.combinedToolsJSON(agentMode: true, mcpToolsJSON: nil)
        XCTAssertNotNil(agentOnly)
        XCTAssertTrue(agentOnly!.contains("shell"))
        _ = agent  // silence unused warning if the agent fixture isn't compared

        // Neither.
        XCTAssertNil(ChatTurnEngine.combinedToolsJSON(agentMode: false, mcpToolsJSON: nil))
    }

    // MARK: - Tool result rendering (swift-sdk 0.12 content shapes)

    /// Characterization for the 0.10 → 0.12 SDK bump: text flattens verbatim,
    /// non-text content is summarized, and the new resourceLink case renders
    /// a readable line instead of being dropped.
    func testRenderToolContentFlattensTextAndSummarizesTheRest() {
        let content: [Tool.Content] = [
            .text(text: "hello", annotations: nil, _meta: nil),
            .image(data: "…", mimeType: "image/png", annotations: nil, _meta: nil),
            .audio(data: "…", mimeType: "audio/wav", annotations: nil, _meta: nil),
            .resource(resource: .text("body", uri: "file:///a.txt", mimeType: "text/plain"),
                      annotations: nil, _meta: nil),
            .resourceLink(uri: "https://x.test/doc", name: "doc"),
        ]
        let out = MCPManager.renderToolContent(content)
        XCTAssertTrue(out.contains("hello"))
        XCTAssertTrue(out.contains("[image: image/png]"))
        XCTAssertTrue(out.contains("[audio: audio/wav]"))
        XCTAssertTrue(out.contains("[resource: file:///a.txt (text/plain)]"))
        XCTAssertTrue(out.contains("doc") && out.contains("https://x.test/doc"))
        XCTAssertEqual(MCPManager.renderToolContent([]), "(no content)")
    }
}
