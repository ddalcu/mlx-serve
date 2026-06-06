import XCTest
@testable import MLXCore

/// Locks the unified tool-execution guard: built-in *and* MCP tools flow through
/// the same `AgentEngine.executeToolCall`, so the repetition block/warning logic
/// lives in one place and MCP tools can no longer loop forever.
///
/// Regression context: the 26B model looped `dbhub__search_objects` 31× in a row
/// — every call succeeded, so `StuckDetector` (which only trips on failures)
/// never fired, and MCP execution went through a *separate* branch that skipped
/// `RepetitionTracker.isBlocked` entirely. Unifying the paths plugs that gap.
@MainActor
final class AgentRepetitionTests: XCTestCase {

    /// Records calls so we can assert routing + that a blocked call never reaches
    /// the server.
    final class FakeMCPRouter: MCPToolRouting {
        private(set) var executeCount = 0
        func owns(toolName: String) -> Bool { toolName.hasPrefix("db__") }
        func executeToolCall(name: String, arguments: [String: String], rawArguments: String) async -> String {
            executeCount += 1
            return "{\"ok\":true,\"q\":\"\(arguments["q"] ?? "")\"}"
        }
    }

    // MARK: repetition key

    func testRepetitionKeyDistinguishesDifferentArgsForNonPrimaryTools() {
        // MCP tools have no designated primary arg, so they must key on the FULL
        // argument set — otherwise distinct queries collide and get over-blocked.
        let a = AgentEngine.toolRepetitionKey(name: "dbhub__execute_sql", arguments: ["sql": "SELECT 1"])
        let b = AgentEngine.toolRepetitionKey(name: "dbhub__execute_sql", arguments: ["sql": "SELECT 2"])
        XCTAssertNotEqual(a, b)
        XCTAssertEqual(a, AgentEngine.toolRepetitionKey(name: "dbhub__execute_sql", arguments: ["sql": "SELECT 1"]))
    }

    func testRepetitionKeyStableRegardlessOfDictOrder() {
        XCTAssertEqual(
            AgentEngine.toolRepetitionKey(name: "x", arguments: ["a": "1", "b": "2"]),
            AgentEngine.toolRepetitionKey(name: "x", arguments: ["b": "2", "a": "1"]))
    }

    func testRepetitionKeyPrimaryArgToolUnchanged() {
        // Existing behavior preserved: listFiles still keys on `path` only.
        XCTAssertEqual(
            AgentEngine.toolRepetitionKey(name: "listFiles", arguments: ["path": "src", "recursive": "true"]),
            "listFiles:src")
    }

    // MARK: unified guard

    func testMCPLoopGetsBlockedThroughTheSharedGuard() async {
        let mcp = FakeMCPRouter()
        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let mem = AgentMemory()
        let tc = APIClient.ToolCall(id: "1", name: "db__query", arguments: ["q": "x"], rawArguments: "{\"q\":\"x\"}")
        // The model loops on one identical call — fill the block window.
        for _ in 0..<8 { rep.track(toolCalls: [tc]) }
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: rep, iteration: 0, agentMemory: mem, mcpRouter: mcp)
        XCTAssertTrue(r.output.hasPrefix("BLOCKED"), "looping MCP call must be blocked, got: \(r.output)")
        XCTAssertEqual(mcp.executeCount, 0, "a blocked call must never reach the MCP server")
    }

    func testMCPCallRoutesThroughRouterWhenNotLooping() async {
        let mcp = FakeMCPRouter()
        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let mem = AgentMemory()
        let tc = APIClient.ToolCall(id: "1", name: "db__query", arguments: ["q": "hello"], rawArguments: "{}")
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: rep, iteration: 0, agentMemory: mem, mcpRouter: mcp)
        XCTAssertEqual(mcp.executeCount, 1)
        XCTAssertTrue(r.output.contains("ok"))
        XCTAssertEqual(r.name, "db__query")
    }

    func testDistinctMCPArgsAreNotOverBlocked() async {
        let mcp = FakeMCPRouter()
        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let mem = AgentMemory()
        // 10 DIFFERENT queries are legitimate work, not a loop — all must run.
        for i in 0..<10 {
            let tc = APIClient.ToolCall(id: "\(i)", name: "db__query", arguments: ["q": "SELECT \(i)"], rawArguments: "{}")
            rep.track(toolCalls: [tc])
            let r = await AgentEngine.executeToolCall(
                tc, workingDirectory: &wd, repetition: rep, iteration: i, agentMemory: mem, mcpRouter: mcp)
            XCTAssertFalse(r.output.hasPrefix("BLOCKED"), "distinct query \(i) must not be blocked")
        }
        XCTAssertEqual(mcp.executeCount, 10)
    }

    /// Characterization: the built-in path (no router) still behaves as before —
    /// guards the extraction of `executeBuiltinTool`.
    func testBuiltinUnknownToolStillErrors() async {
        let rep = AgentEngine.RepetitionTracker()
        var wd: String? = nil
        let mem = AgentMemory()
        let tc = APIClient.ToolCall(id: "1", name: "totally_unknown_tool", arguments: [:], rawArguments: "{}")
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: rep, iteration: 0, agentMemory: mem)
        XCTAssertTrue(r.output.contains("Unknown tool"))
    }
}
