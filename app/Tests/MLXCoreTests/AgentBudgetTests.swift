import XCTest
@testable import MLXCore

/// The CLI launcher writes config files for third-party agents (pi, opencode)
/// and env vars for Claude Code. Those configs declare how much context the
/// model has — and the agents budget their own `max_tokens` against that number,
/// NOT against anything the server says.
///
/// Regression: the pi config hardcoded `contextWindow: 32768, maxTokens: 8192`
/// while the server was advertising ~94k. Late in a long session pi's remaining
/// budget collapsed and it started sending `max_tokens=1` (observed live,
/// 2026-07-08 — `prompt=30827 tokens, max_gen=1, ctx=92387` in the server log),
/// so every tool call truncated and the session died. The launcher must derive
/// these numbers from the context the RUNNING server advertises.
final class AgentBudgetTests: XCTestCase {

    // MARK: The server's numbers

    func testARowsAgentBudgetIsReadVerbatim() {
        // The server computes the budget (src/host/launch/budget.zig); the app
        // declares exactly what the row carries.
        let row: [String: Any] = ["context": 78848, "output": 39424, "compaction_reserve": 19712,
                                  "too_small_for": ["claude": 65536]]
        XCTAssertEqual(AgentBudget.parse(row),
                       AgentBudget.Budget(context: 78848, output: 39424, compactionReserve: 19712,
                                          tooSmallFor: ["claude": 65536]))
        XCTAssertNil(AgentBudget.parse(nil), "an older server's row carries none")
        XCTAssertNil(AgentBudget.parse(["context": 0, "output": 1, "compaction_reserve": 1]))
    }

    func testParseModelInfoCarriesTheBudgetModalityAndEngines() {
        let m = APIClient.parseModelInfo([
            "id": "m", "capabilities": ["chat"],
            "agent_budget": ["context": 24576, "output": 12288, "compaction_reserve": 6144, "too_small_for": [:] as [String: Int]],
            "engines": ["mlx", "ane"],
        ])
        XCTAssertEqual(m.agentBudget?.output, 12288)
        XCTAssertEqual(m.engines, ["mlx", "ane"])
        XCTAssertNil(m.modality)
        let img = APIClient.parseModelInfo(["id": "i", "capabilities": ["image"], "modality": "image"])
        XCTAssertNil(img.agentBudget)
        XCTAssertEqual(img.modality, "image")
    }

    func testTheSmallWindowWarningNamesTheFloorTheServerSaysIsMissed() {
        let b = AgentBudget.Budget(context: 24576, output: 12288, compactionReserve: 6144,
                                   tooSmallFor: ["claude": 65536, "opencode": 32768])
        XCTAssertNotNil(AgentBudget.contextWarning(agentId: "claude", budget: b))
        XCTAssertTrue(AgentBudget.contextWarning(agentId: "opencode", budget: b)?.contains("32768") ?? false)
        XCTAssertNil(AgentBudget.contextWarning(agentId: "pi", budget: b), "pi's floor is met")
        let tiny = AgentBudget.Budget(context: 8192, output: 4096, compactionReserve: 2048,
                                      tooSmallFor: ["claude": 65536, "default": 16384])
        XCTAssertTrue(AgentBudget.contextWarning(agentId: "pi", budget: tiny)?.contains("16384") ?? false,
                      "an agent with no floor of its own takes the default")
    }

    // MARK: The configs we actually write

    func testPiConfigCarriesTheDerivedBudgetNotAHardcodedOne() throws {
        let b = AgentBudget.Budget(context: 94729, output: 47364, compactionReserve: 20000)
        let json = AgentConfigs.piModelsJSON(
            baseURL: "http://localhost:11234", model: "Qwen3.6-27B", budget: b)

        // Parse it — a broken config silently strands the user on defaults.
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(json.utf8)) as? [String: Any])
        let providers = try XCTUnwrap(obj["providers"] as? [String: Any])
        let mlx = try XCTUnwrap(providers["mlx"] as? [String: Any])
        XCTAssertEqual(mlx["baseUrl"] as? String, "http://localhost:11234/v1")
        let models = try XCTUnwrap(mlx["models"] as? [[String: Any]])
        let m = try XCTUnwrap(models.first)

        XCTAssertEqual(m["contextWindow"] as? Int, b.context)
        XCTAssertEqual(m["maxTokens"] as? Int, b.output)
        XCTAssertNotEqual(m["contextWindow"] as? Int, 32768, "still hardcoded")
        XCTAssertEqual(m["id"] as? String, "Qwen3.6-27B")
    }

    // pi loads an `AGENTS.md` from its agent config dir into every session's
    // system prompt (global context file). Ours teaches the chunked-write
    // convention: pi's `write` has no append flag and pi ALWAYS sends its
    // configured `maxTokens` (a <=0 value is a config validation error), so a
    // file bigger than the response cap can only land via bash appends — the
    // 2026-07-20 loop re-issued an impossible one-shot `write` for hours.
    func testPiSurfacesEnableReasoningEffortAndAgree() throws {
        // pi picks a reasoning level in its own UI, but only SENDS it when the
        // provider declares support. With this false the level was a local
        // label: every request arrived with no `reasoning_effort`, so the
        // server applied its own default and nothing pi showed could change it.
        // The provider block in models.json and the per-model COMPAT in the
        // extension are two copies of one contract (applyExtension does not
        // inherit provider compat), so they are pinned together.
        let json = AgentConfigs.piModelsJSON(
            baseURL: "http://127.0.0.1:11234", model: "m",
            budget: AgentBudget.Budget(context: 32768, output: 8192, compactionReserve: 8192))
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://127.0.0.1:11234")
        XCTAssertTrue(json.contains("\"supportsReasoningEffort\": true"),
                      "models.json must enable reasoning effort: \(json)")
        XCTAssertTrue(js.contains("supportsReasoningEffort: true"),
                      "extension COMPAT must enable reasoning effort: \(js)")
        XCTAssertFalse(json.contains("\"supportsReasoningEffort\": false"))
        XCTAssertFalse(js.contains("supportsReasoningEffort: false"))
    }

    func testPiSurfacesOfferEveryThinkingLevel() throws {
        // pi offers xhigh/max only when the level map names them; without them
        // `--thinking max` clamps to high and a model's max effort is unreachable.
        let json = AgentConfigs.piModelsJSON(
            baseURL: "http://127.0.0.1:11234", model: "m",
            budget: AgentBudget.Budget(context: 32768, output: 8192, compactionReserve: 8192))
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let mlx = try XCTUnwrap((obj["providers"] as? [String: Any])?["mlx"] as? [String: Any])
        let m = try XCTUnwrap((mlx["models"] as? [[String: Any]])?.first)
        let levels = try XCTUnwrap(m["thinkingLevelMap"] as? [String: String])
        XCTAssertEqual(levels, ["off": "none", "xhigh": "xhigh", "max": "max"])
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://127.0.0.1:11234")
        XCTAssertTrue(js.contains("thinkingLevelMap: \(AgentConfigs.piThinkingLevelMap)"),
                      "extension models must carry the same level map: \(js)")
    }

    func testPiAgentsMDStatesTheCapAndTheChunkingRecovery() {
        let b = AgentBudget.Budget(context: 262144, output: 65536, compactionReserve: 20000)
        let md = AgentConfigs.piAgentsMD(budget: b)
        XCTAssertTrue(md.contains("\(b.output)"),
                      "must state the real response cap, never a hardcode")
        XCTAssertTrue(md.lowercased().contains("append"),
                      "must name the chunked-append escape hatch")
        XCTAssertTrue(md.lowercased().contains("cut off"),
                      "must teach truncation recognition — the loop class misread it as a forgotten parameter")
        XCTAssertFalse(md.contains("__MLX_HOST__"),
                       "static guidance must not depend on bootstrap substitution")
    }

    func testOpencodeConfigDeclaresPerModelLimits() throws {
        let b = AgentBudget.Budget(context: 94729, output: 47364, compactionReserve: 20000)
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", model: "Qwen3.6-27B", budget: b)

        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: Data(json.utf8)) as? [String: Any])
        let provider = try XCTUnwrap(obj["provider"] as? [String: Any])
        let mlx = try XCTUnwrap(provider["mlx"] as? [String: Any])
        let models = try XCTUnwrap(mlx["models"] as? [String: Any])
        let model = try XCTUnwrap(models["Qwen3.6-27B"] as? [String: Any])
        // opencode's schema: models.<id>.limit.{context,output}
        let limit = try XCTUnwrap(model["limit"] as? [String: Any])
        XCTAssertEqual(limit["context"] as? Int, b.context)
        // opencode never sends max_tokens: limit.output is its compaction reserve.
        XCTAssertEqual(limit["output"] as? Int, b.compactionReserve)
        XCTAssertNil(obj["compaction"], "opencode 1 gets no compaction block")
        // opencode sends no reasoning_effort unless the model declares one: thinking stayed off.
        XCTAssertEqual((model["options"] as? [String: Any])?["reasoningEffort"] as? String, "medium")
        let variants = try XCTUnwrap(model["variants"] as? [String: Any])
        XCTAssertEqual((variants["none"] as? [String: Any])?["reasoningEffort"] as? String, "none")
        XCTAssertEqual((variants["high"] as? [String: Any])?["reasoningEffort"] as? String, "high")
    }

    func testPiSettingsMergeScalesCompactionAndKeepsTheRest() throws {
        let existing = #"{"theme":"dark","compaction":{"enabled":false,"reserveTokens":1}}"#
        let json = AgentConfigs.piSettingsJSON(existing: existing, budget: AgentBudget.Budget(context: 24576, output: 12288, compactionReserve: 6144))
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(obj["theme"] as? String, "dark")
        let c = try XCTUnwrap(obj["compaction"] as? [String: Any])
        XCTAssertEqual(c["enabled"] as? Bool, false, "the user's own flag survives")
        XCTAssertEqual(c["reserveTokens"] as? Int, 10240)
        XCTAssertEqual(c["keepRecentTokens"] as? Int, 6144)

        let big = AgentConfigs.piSettingsJSON(existing: "", budget: AgentBudget.Budget(context: 262144, output: 65536, compactionReserve: 20000))
        let bo = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(big.utf8)) as? [String: Any])
        let bc = try XCTUnwrap(bo["compaction"] as? [String: Any])
        XCTAssertEqual(bc["reserveTokens"] as? Int, 16384, "big windows keep pi's defaults")
        XCTAssertEqual(bc["keepRecentTokens"] as? Int, 20000)
    }

    func testOpencode2ConfigCarriesAScaledCompactionBlock() throws {
        let b = AgentBudget.Budget(context: 24576, output: 12288, compactionReserve: 6144)
        let json = AgentConfigs.opencodeJSON(
            baseURL: "http://localhost:11234", defaultModel: "m",
            entries: [AgentModelEntry(id: "m", budget: b, vision: false)], pinModel: true, compaction: true)
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let c = try XCTUnwrap(obj["compaction"] as? [String: Any])
        XCTAssertEqual(c["buffer"] as? Int, 6144)
        XCTAssertEqual((c["keep"] as? [String: Any])?["tokens"] as? Int, 6144)
        XCTAssertFalse(json.contains("'"))
    }

    func testClaudeCodeExportsCapOutputTokens() {
        let b = AgentBudget.Budget(context: 94729, output: 47364, compactionReserve: 20000)
        let script = AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "mlx-serve", budget: b)

        XCTAssertTrue(script.contains("export ANTHROPIC_BASE_URL='http://localhost:11234'"))
        // Claude Code exposes CLAUDE_CODE_MAX_OUTPUT_TOKENS (verified present in
        // the 2.1.x binary).
        XCTAssertTrue(script.contains("export CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(b.output)"),
                      "missing output cap in:\n\(script)")
        XCTAssertTrue(script.contains("ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-serve"))
    }

    /// Twin of the Zig launcher's "claude script declares the advertised
    /// context window" test. Claude Code 2.1.x assumes 200k for any model
    /// outside its own catalog and auto-compacts there;
    /// CLAUDE_CODE_MAX_CONTEXT_TOKENS is the documented override, and like
    /// every other agent's context field it is declared VERBATIM.
    func testClaudeCodeExportsDeclareAdvertisedContext() {
        let b = AgentBudget.Budget(context: 786_432, output: 65536, compactionReserve: 20000)
        let script = AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "mlx-serve", budget: b)
        XCTAssertTrue(script.contains("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=786432"),
                      "missing context window in:\n\(script)")
        XCTAssertTrue(script.hasSuffix("export CLAUDE_CODE_MAX_CONTEXT_TOKENS=786432"),
                      "context export must be the last line (no trailing newline)")

        // An unknown context is not a claim: omit the export.
        let unknown = AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "mlx-serve",
            budget: AgentBudget.Budget(context: 0, output: 8192, compactionReserve: 1024))
        XCTAssertFalse(unknown.contains("CLAUDE_CODE_MAX_CONTEXT_TOKENS"))
        XCTAssertTrue(unknown.contains("export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192"))
    }
}
