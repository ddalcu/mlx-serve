import XCTest
@testable import MLXCore

/// Capability gating is belt-and-braces: the advertised tool list is filtered,
/// AND dispatch refuses anything outside the set (models call tools that were
/// never advertised). Both halves are pinned here, plus the sync guard that
/// keeps `AgentToolKind` and the hand-written tool JSON from drifting — an
/// ungateable tool is a silent hole in every agent's capability list.
@MainActor
final class AgentCapabilityGateTests: XCTestCase {

    private func names(in json: String) -> [String] {
        let data = json.data(using: .utf8)!
        let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
        return arr.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
    }

    // MARK: - Filtering the advertised list

    func testFilteringKeepsOnlyTheAllowedToolsAndStaysValidJSON() {
        let json = AgentPrompt.toolDefinitionsJSON(allowing: [.readFile, .shell])
        XCTAssertEqual(Set(names(in: json)), ["readFile", "shell"])
    }

    func testFilteringPreservesKeyOrderInsideEachTool() {
        // The literal is hand-ordered so a truncated writeFile still carries its
        // path. A JSONSerialization round-trip would reorder the keys — the
        // filter MUST stay line-based.
        let json = AgentPrompt.toolDefinitionsJSON(allowing: [.writeFile])
        let afterName = json[json.range(of: "\"writeFile\"")!.upperBound...]
        let path = afterName.range(of: "\"path\":{\"type\":\"string\"")!.lowerBound
        let content = afterName.range(of: "\"content\":{\"type\":\"string\"")!.lowerBound
        XCTAssertTrue(path < content, "'path' must still precede 'content' after filtering")
    }

    func testAllowingEveryToolReproducesTheFullLiteralByteForByte() {
        // Byte-identical, not merely equivalent: the tool block sits in front of
        // the whole cached system prefix, so one changed byte re-prefills every
        // agent turn for users who never make an agent.
        let all = AgentPrompt.toolDefinitionsJSON(allowing: Set(AgentToolKind.allCases))
        XCTAssertEqual(all, AgentPrompt.toolDefinitionsJSON,
                       "the unfiltered list is the filter's identity case, whitespace included")
    }

    func testAllowingNothingYieldsAnEmptyArrayThatDropsOutOfTheRequest() {
        let none = AgentPrompt.toolDefinitionsJSON(allowing: [])
        XCTAssertEqual(names(in: none), [])
        XCTAssertNil(ChatTurnEngine.combinedToolsJSON(tools: [], mcpToolsJSON: nil, docsToolJSON: nil),
                     "no tools at all ⇒ no `tools` field on the request")
    }

    func testCombinedToolsJSONFiltersTheAgentPortionOnly() {
        let mcp = #"[{"type":"function","function":{"name":"srv__thing","parameters":{"type":"object","properties":{},"required":[]}}}]"#
        let combined = ChatTurnEngine.combinedToolsJSON(tools: [.shell], mcpToolsJSON: mcp,
                                                        docsToolJSON: AgentPrompt.searchDocumentsToolJSON)!
        XCTAssertEqual(Set(names(in: combined)), ["shell", "srv__thing", "searchDocuments"])
    }

    // MARK: - Dispatch refusal

    func testDisallowedToolIsRefusedByName() {
        let refusal = AgentEngine.disallowedToolRefusal(name: "shell", allowed: [.readFile])
        XCTAssertNotNil(refusal)
        XCTAssertTrue(refusal!.contains("shell"), "the refusal must name the tool: \(refusal!)")
        XCTAssertNil(AgentEngine.disallowedToolRefusal(name: "readFile", allowed: [.readFile]))
    }

    func testRefusalSurvivesTheNameQuirksDispatchAlreadyTolerates() {
        // `canonicalToolName` strips a leaked trailing ':' and a functions. prefix —
        // the gate must run on the canonical name or it's trivially bypassed.
        XCTAssertNotNil(AgentEngine.disallowedToolRefusal(name: "functions.shell", allowed: [.readFile]))
        XCTAssertNotNil(AgentEngine.disallowedToolRefusal(name: "shell:", allowed: [.readFile]))
    }

    func testUnknownAndMcpToolsAreNotGatedHere() {
        // MCP tools are governed by the MCP flag, and a genuinely unknown name
        // must keep its existing "Unknown tool" answer.
        XCTAssertNil(AgentEngine.disallowedToolRefusal(name: "srv__thing", allowed: [.readFile]))
        XCTAssertNil(AgentEngine.disallowedToolRefusal(name: "nonsense", allowed: [.readFile]))
    }

    func testNilAllowedSetGatesNothing() {
        XCTAssertNil(AgentEngine.disallowedToolRefusal(name: "shell", allowed: nil),
                     "callers with no agent (TestServer, older surfaces) keep full access")
    }

    func testDispatchRefusesADisallowedToolBeforeExecuting() async throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("acg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        var wd: String? = dir
        let target = (dir as NSString).appendingPathComponent("should-not-exist.txt")

        let tc = APIClient.ToolCall(id: "1", name: "writeFile",
                                    arguments: ["path": "should-not-exist.txt", "content": "x"],
                                    rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd,
                                                      repetition: AgentEngine.RepetitionTracker(),
                                                      iteration: 0, agentMemory: AgentMemory(),
                                                      allowedTools: [.readFile])
        XCTAssertTrue(result.output.contains("writeFile"), "refusal names the tool: \(result.output)")
        XCTAssertFalse(FileManager.default.fileExists(atPath: target),
                       "a gated tool must be refused BEFORE it runs")
    }

    func testMetaToolsAreGatedToo() async {
        // createTask / generate_image aren't ToolHandlers — they're dispatched
        // ahead of the handler registry, so the gate has to sit ahead of THEM.
        var wd: String? = nil
        var created = false
        let tc = APIClient.ToolCall(id: "1", name: "createTask",
                                    arguments: ["goal": "do a thing"], rawArguments: "")
        let result = await AgentEngine.executeToolCall(tc, workingDirectory: &wd,
                                                      repetition: AgentEngine.RepetitionTracker(),
                                                      iteration: 0, agentMemory: AgentMemory(),
                                                      createTask: { _, _ in created = true; return "made it" },
                                                      allowedTools: [.readFile])
        XCTAssertFalse(created, "createTask must not fire for an agent that can't use it")
        XCTAssertTrue(result.output.contains("createTask"), result.output)
    }

    // MARK: - The sync guard (both directions)

    func testEveryToolKindIsBackedByADefinitionAndViceVersa() {
        let declared = Set(AgentPrompt.toolDefinitions.compactMap {
            ($0["function"] as? [String: Any])?["name"] as? String
        })
        let gateable = Set(AgentToolKind.allCases.map(\.rawValue))
        XCTAssertEqual(declared.subtracting(gateable), [],
                       "a tool in the JSON with no AgentToolKind case is UNGATEABLE — add the case")
        XCTAssertEqual(gateable.subtracting(declared), [],
                       "an AgentToolKind with no tool definition can never be advertised — remove it or add the JSON")
    }

    func testEveryToolKindHasAnIconAndDisplayName() {
        for kind in AgentToolKind.allCases {
            XCTAssertFalse(kind.icon.isEmpty, "\(kind.rawValue) needs an icon for the Advanced list")
            XCTAssertFalse(kind.displayName.isEmpty, "\(kind.rawValue) needs a display name")
        }
    }

    func testCoarseGroupsCoverEveryGateableTool() {
        // A tool in no group is unreachable from the coarse UI — the class bug
        // this guard exists for.
        let grouped = AgentCapabilities.loopTools
            .union(AgentCapabilities.webTools)
            .union([.searchDocuments])
        XCTAssertEqual(Set(AgentToolKind.allCases).subtracting(grouped), [],
                       "every tool belongs to Tools, Web, or the docs gate")
    }
}
