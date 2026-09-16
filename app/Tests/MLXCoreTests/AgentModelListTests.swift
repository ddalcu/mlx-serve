import XCTest
@testable import MLXCore

/// In-agent model switching (/model in pi + hermes, /models in opencode):
/// the launcher no longer declares just the served model — it hands each CLI
/// the full chat-capable registry (LAN @peer entries included).
///
/// pi is special: its models.json list is static, but pi ships a first-class
/// extension API (`pi.registerProvider`, verified in 0.80.10 — the pinned
/// sandbox version). We write an extension that fetches the server's live
/// `/v1/models` at session start, so pi's picker tracks reality (LAN peers
/// come and go) instead of a launch-time snapshot. opencode and hermes have
/// no such hook — they get the snapshot baked into their configs.
final class AgentModelListTests: XCTestCase {

    private func model(_ name: String, ctx: Int = 65536, caps: [String] = ["chat"],
                       vision: Bool = false, embeddings: Bool = false,
                       lanPeer: String? = nil) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 32, hiddenSize: 4096,
                  vocabSize: 32000, contextLength: ctx, modelMaxTokens: ctx,
                  supportsVision: vision, supportsEmbeddings: embeddings,
                  capabilities: caps, lanPeer: lanPeer)
    }

    // MARK: entry selection — what the configs list

    func testChatEntriesKeepChatModelsAndDropMediaAndEmbeddings() {
        let entries = AgentModelEntry.chatEntries(from: [
            model("qwen3.6-27b", ctx: 94729),
            model("flux2-dev", caps: ["image"]),
            model("ltx-2", caps: ["video"]),
            model("bge-small", caps: ["embeddings"], embeddings: true),
            model("gemma-4-12b", ctx: 131072, caps: ["chat", "vision"], vision: true),
        ])
        XCTAssertEqual(entries.map(\.id), ["qwen3.6-27b", "gemma-4-12b"],
                       "media + embedding entries must never reach an agent CLI's chat picker")
    }

    func testChatEntriesDeriveBudgetsPerModelNotFromTheLoadedOne() {
        // The old single-model plumbing stamped the LOADED model's budget on
        // whatever the config declared. Per-model contexts differ wildly
        // (262K Qwen vs a 32K GGUF) — each entry derives its own.
        let entries = AgentModelEntry.chatEntries(from: [
            model("big", ctx: 262144),
            model("small", ctx: 32768),
        ])
        XCTAssertEqual(entries[0].budget, AgentBudget.forServerContext(262144))
        XCTAssertEqual(entries[1].budget, AgentBudget.forServerContext(32768))
        XCTAssertEqual(entries[0].budget.output, 65536)
        XCTAssertEqual(entries[1].budget.output, 16384)
    }

    func testChatEntriesIncludeLanPeersWithEmptyCapsAndFlagVision() {
        // Old-peer tolerance (ModelInfo.lanAdvertises): a pre-26.7.11 peer
        // serves chat with capabilities:[] — counts as chat, nothing else.
        let entries = AgentModelEntry.chatEntries(from: [
            model("local", ctx: 65536),
            model("remote@peer1", ctx: 78848, caps: [], lanPeer: "peer1"),
            model("img@peer1", caps: ["image"], lanPeer: "peer1"),
            model("vlm", caps: ["chat", "vision"], vision: true),
        ])
        XCTAssertEqual(entries.map(\.id), ["local", "remote@peer1", "vlm"])
        XCTAssertFalse(entries[0].vision)
        XCTAssertTrue(entries[2].vision)
    }

    func testChatEntriesDedupeAndFallBackWhenContextUnknown() {
        let entries = AgentModelEntry.chatEntries(from: [
            model("m", ctx: 65536),
            model("m", ctx: 65536),
            model("stub", ctx: 0),
        ])
        XCTAssertEqual(entries.map(\.id), ["m", "stub"])
        XCTAssertEqual(entries[1].budget, AgentBudget.fallback,
                       "a stub with no context metadata gets the conservative fallback, never 0")
    }

    // MARK: pi live-list extension

    func testPiExtensionRegistersTheMlxProviderFromLiveModels() {
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://localhost:11234")
        // pi's loader contract (verified in 0.80.10 dist): default-export a
        // factory function; jiti imports the file.
        XCTAssertTrue(js.contains("export default"), js)
        XCTAssertTrue(js.contains("pi.registerProvider(\"mlx\""),
                      "must override the models.json provider so /model ids stay mlx/<id>")
        XCTAssertTrue(js.contains("http://localhost:11234/v1/models"),
                      "the live list comes from the server, not a baked snapshot")
        // Self-contained provider config: applyExtension does NOT inherit
        // baseUrl/api from the models.json layer per-field — spell them out.
        XCTAssertTrue(js.contains("\"http://localhost:11234\" + \"/v1\"") || js.contains("http://localhost:11234/v1\""),
                      "chat baseUrl must be the /v1 root: \(js)")
        XCTAssertTrue(js.contains("openai-completions"))
    }

    func testPiExtensionMirrorsTheAgentBudgetRule() {
        // The extension computes per-model maxTokens where Swift can't reach
        // (live fetch, in-guest). The rule must be AgentBudget.forServerContext
        // verbatim: min(65536, max(1024, ctx/2)), fallback context 32768.
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://h:1")
        XCTAssertTrue(js.contains("Math.min(65536, Math.max(1024, Math.floor(ctx / 2)))"),
                      "budget rule drifted from AgentBudget: \(js)")
        XCTAssertTrue(js.contains("32768"), "fallback context missing")
    }

    func testPiExtensionCarriesCompatOnEveryModel() {
        // provider-composer's applyExtension spreads ONLY the definition —
        // provider-level compat from models.json is NOT stamped onto
        // extension-registered models. Without per-model compat pi would use
        // max_completion_tokens + the wrong thinking format.
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://h:1")
        for needle in ["maxTokensField", "max_tokens", "thinkingFormat", "qwen",
                       "supportsDeveloperRole", "supportsReasoningEffort"] {
            XCTAssertTrue(js.contains(needle), "compat field \(needle) missing: \(js)")
        }
        XCTAssertTrue(js.contains("compat: COMPAT") || js.contains("compat:"),
                      "compat must ride each model definition")
    }

    func testPiExtensionFiltersToChatAndToleratesEmptyCaps() {
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://h:1")
        XCTAssertTrue(js.contains("includes(\"chat\")"),
                      "media/embedding entries must not enter pi's picker")
        XCTAssertTrue(js.contains("length === 0"),
                      "empty capabilities = old LAN peer serving chat (lanAdvertises rule)")
    }

    func testPiExtensionAuthAndFailureModes() {
        let js = AgentConfigs.piModelsExtensionJS(baseURL: "http://__MLX_HOST__:8080",
                                                  apiKey: "sekrit-9")
        // Guest→host arrives non-loopback: the real key must ride the fetch
        // AND the registered provider.
        XCTAssertTrue(js.contains("Bearer"), js)
        XCTAssertTrue(js.contains("sekrit-9"), js)
        XCTAssertTrue(js.contains("__MLX_HOST__"),
                      "sandbox variant carries the placeholder for the bootstrap sed")
        // Server unreachable → register NOTHING (models.json fallback stands);
        // and the fetch is time-bounded so session start can't hang.
        XCTAssertTrue(js.contains("AbortController"), "unbounded fetch can hang pi startup")
        XCTAssertTrue(js.contains("return") && js.contains("catch"),
                      "fetch failure must leave the static fallback untouched")
    }

    // MARK: opencode — full list baked into the config

    func testOpencodeConfigListsEveryEntryWithItsOwnLimits() throws {
        let entries = [
            AgentModelEntry(id: "qwen3.6-27b", budget: AgentBudget.forServerContext(94729), vision: false),
            AgentModelEntry(id: "gemma-4-12b", budget: AgentBudget.forServerContext(131072), vision: true),
            AgentModelEntry(id: "remote@peer1", budget: AgentBudget.forServerContext(32768), vision: false),
        ]
        let json = AgentConfigs.opencodeJSON(baseURL: "http://localhost:11234",
                                             defaultModel: "qwen3.6-27b", entries: entries)
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let provider = try XCTUnwrap(obj["provider"] as? [String: Any])
        let mlx = try XCTUnwrap(provider["mlx"] as? [String: Any])
        let models = try XCTUnwrap(mlx["models"] as? [String: Any])
        XCTAssertEqual(Set(models.keys), ["qwen3.6-27b", "gemma-4-12b", "remote@peer1"])
        let gemma = try XCTUnwrap(models["gemma-4-12b"] as? [String: Any])
        let limit = try XCTUnwrap(gemma["limit"] as? [String: Any])
        XCTAssertEqual(limit["context"] as? Int, 131072)
        XCTAssertEqual(limit["output"] as? Int, AgentBudget.compactionReserve(131072))
        XCTAssertEqual(gemma["attachment"] as? Bool, true, "vision entry allows image attachments")
        let qwen = try XCTUnwrap(models["qwen3.6-27b"] as? [String: Any])
        XCTAssertNil(qwen["attachment"], "non-vision entries stay minimal")
        // The launch script single-quotes OPENCODE_CONFIG_CONTENT.
        XCTAssertFalse(json.contains("'"), "single quote would break the launch script")
    }

    func testOpencodeSingleModelWrapperStaysCompatible() throws {
        // The MAS instructions panel still renders the single-model shape.
        let b = AgentBudget.forServerContext(94729)
        let json = AgentConfigs.opencodeJSON(baseURL: "http://h:1", model: "m", budget: b)
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let provider = try XCTUnwrap(obj["provider"] as? [String: Any])
        let mlx = try XCTUnwrap(provider["mlx"] as? [String: Any])
        let models = try XCTUnwrap(mlx["models"] as? [String: Any])
        XCTAssertNotNil(models["m"])
    }

    // MARK: hermes — models map under the custom provider

    func testHermesConfigListsEveryEntryWithContextLength() {
        let entries = [
            AgentModelEntry(id: "qwen3.6-27b", budget: AgentBudget.forServerContext(94729), vision: false),
            AgentModelEntry(id: "remote@peer1", budget: AgentBudget.forServerContext(78848), vision: false),
        ]
        let yaml = AgentConfigs.hermesConfigYAML(
            baseURL: "http://__MLX_HOST__:11234", apiKey: "k1",
            model: "qwen3.6-27b", budget: AgentBudget.forServerContext(94729),
            entries: entries)
        // hermes /model switches among models already in the config
        // (custom_providers[].models.<id>.context_length — per-model key).
        XCTAssertTrue(yaml.contains("\"qwen3.6-27b\":"), yaml)
        XCTAssertTrue(yaml.contains("\"remote@peer1\":"), yaml)
        XCTAssertTrue(yaml.contains("context_length: 94729"), yaml)
        XCTAssertTrue(yaml.contains("context_length: 19712") == false, yaml) // outputs never leak in
        XCTAssertTrue(yaml.contains("context_length: 78848"), yaml)
        XCTAssertTrue(yaml.contains("default: \"qwen3.6-27b\""), "served model stays the default")
    }

    func testHermesConfigWithNoEntriesKeepsTheDefaultModelLine() {
        // Defensive: registry snapshot empty (server just started) — the
        // config must still carry the served model or hermes re-runs its
        // wizard. Same shape the wizard saves.
        let b = AgentBudget.Budget(context: 65536, output: 16384)
        let yaml = AgentConfigs.hermesConfigYAML(baseURL: "http://h:1", apiKey: "k",
                                                 model: "m", budget: b, entries: [])
        XCTAssertTrue(yaml.contains("default: \"m\""), yaml)
        XCTAssertTrue(yaml.contains("context_length: 65536"), yaml)
    }
}
