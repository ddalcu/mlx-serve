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

    // MARK: Budget derivation

    func testUnknownServerContextFallsBackToTheConservativeDefault() {
        // No server info yet (not running / pre-metadata build): never guess high.
        XCTAssertEqual(AgentBudget.forServerContext(nil).context, 32768)
        XCTAssertEqual(AgentBudget.forServerContext(nil).output, 8192)
        XCTAssertEqual(AgentBudget.forServerContext(0).context, 32768)
        XCTAssertEqual(AgentBudget.forServerContext(-1).context, 32768)
    }

    func testServerContextIsDeclaredEXACTLY_noSecondMargin() {
        // The live number a 128 GB Mac running Qwen3.6-27B pins at load.
        let b = AgentBudget.forServerContext(78848)

        // The server already reserved 15% of the memory ceiling before it
        // advertised this. Shaving a second margin here double-counted that
        // headroom AND made the CLI report a different context than Settings
        // showed (opencode said 75K where the server said 77K). Declare the
        // server's number verbatim: it IS the enforced limit, and our own
        // `clampMaxTokens` uses the same one.
        XCTAssertEqual(b.context, 78848)
        // Far above the old hardcoded 32768 — that was the original bug.
        XCTAssertGreaterThan(b.context, 60000)
        // Enough output budget for a one-shot whole-file write.
        XCTAssertGreaterThanOrEqual(b.output, 16384)
        // prompt + output must be expressible inside the window.
        XCTAssertLessThan(b.output, b.context)
    }

    func testContextNeverExceedsWhatTheServerAdvertises() {
        for advertised in [1024, 4096, 8192, 16384, 32768, 65536, 94729, 262144] {
            let b = AgentBudget.forServerContext(advertised)
            XCTAssertEqual(b.context, advertised,
                "declared context must equal the server's \(advertised)")
            XCTAssertLessThanOrEqual(b.output, b.context,
                "output \(b.output) > context \(b.context) at advertised=\(advertised)")
            XCTAssertGreaterThan(b.output, 0)
        }
    }

    // Live 2026-07-20 (pi in the sandbox, Qwen3.6-27B at 262K ctx): the flat
    // 16384 output cap — which thinking tokens share — truncated every
    // whole-file `write` of a large map.js. The server salvaged path-only
    // args, the model misread the validation error as "I forgot content",
    // and the session looped for hours; at 11% context usage pi never
    // compacted the poisoned history away. The output budget must SCALE with
    // the advertised context (context/4), capped at 65536 so a degenerate
    // runaway generation stays bounded, floored at 1024.
    func testOutputBudgetScalesWithContext() {
        XCTAssertEqual(AgentBudget.forServerContext(262144).output, 65536)
        XCTAssertEqual(AgentBudget.forServerContext(131072).output, 32768)
        XCTAssertEqual(AgentBudget.forServerContext(524288).output, 65536,
                       "runaway cap holds at huge contexts")
        XCTAssertEqual(AgentBudget.forServerContext(65536).output, 16384,
                       "mid contexts unchanged")
        XCTAssertEqual(AgentBudget.forServerContext(32768).output, 8192)
        XCTAssertEqual(AgentBudget.forServerContext(2048).output, 1024, "floor")
    }

    func testBudgetGrowsMonotonicallyWithServerContext() {
        var last = 0
        for advertised in [8192, 16384, 32768, 65536, 94729] {
            let c = AgentBudget.forServerContext(advertised).context
            XCTAssertGreaterThanOrEqual(c, last)
            last = c
        }
    }

    // MARK: The configs we actually write

    func testPiConfigCarriesTheDerivedBudgetNotAHardcodedOne() throws {
        let b = AgentBudget.forServerContext(94729)
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
            budget: AgentBudget.Budget(context: 32768, output: 8192))
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://127.0.0.1:11234")
        XCTAssertTrue(json.contains("\"supportsReasoningEffort\": true"),
                      "models.json must enable reasoning effort: \(json)")
        XCTAssertTrue(js.contains("supportsReasoningEffort: true"),
                      "extension COMPAT must enable reasoning effort: \(js)")
        XCTAssertFalse(json.contains("\"supportsReasoningEffort\": false"))
        XCTAssertFalse(js.contains("supportsReasoningEffort: false"))
    }

    func testPiAgentsMDStatesTheCapAndTheChunkingRecovery() {
        let b = AgentBudget.forServerContext(262144)
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
        let b = AgentBudget.forServerContext(94729)
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
        XCTAssertEqual(limit["output"] as? Int, b.output)
    }

    func testClaudeCodeExportsCapOutputTokens() {
        let b = AgentBudget.forServerContext(94729)
        let script = AgentConfigs.claudeCodeExports(
            baseURL: "http://localhost:11234", model: "mlx-serve", budget: b)

        XCTAssertTrue(script.contains("export ANTHROPIC_BASE_URL='http://localhost:11234'"))
        // Claude Code exposes CLAUDE_CODE_MAX_OUTPUT_TOKENS (verified present in
        // the 2.1.x binary); it has NO context-window override, so that is the
        // only budget lever we have on this CLI.
        XCTAssertTrue(script.contains("export CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(b.output)"),
                      "missing output cap in:\n\(script)")
        XCTAssertTrue(script.contains("ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-serve"))
    }
}
