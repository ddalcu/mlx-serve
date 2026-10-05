import Foundation

/// How much context and output budget to declare to a third-party agent CLI.
///
/// pi and opencode do NOT read the server's `/v1/models` metadata — they budget
/// their own per-request `max_tokens` against whatever number their config file
/// declares. If we understate the context, a long session's budget collapses
/// long before the server would have complained: measured live on 2026-07-08,
/// a pi session hit `prompt=30827 tokens, max_gen=1, ctx=92387` — pi asked for
/// ONE output token while the server was offering 92k of context — because the
/// launcher had written a hardcoded `contextWindow: 32768`.
///
/// So these numbers are derived from what the running server advertises
/// (`ModelInfo.contextLength`, i.e. the server's *effective* context).
enum AgentBudget {

    struct Budget: Equatable {
        let context: Int
        let output: Int
    }

    /// Used when the server isn't running yet, or is an older build that does
    /// not report `meta.context_length`. Deliberately conservative — a CLI that
    /// under-declares merely compacts early; one that over-declares gets a hard
    /// 400 on an oversized prompt.
    static let fallback = Budget(context: 32768, output: 8192)

    /// Cap on a single response. Thinking tokens share the response budget, so
    /// "enough for a one-shot whole-file write (8–11k measured)" was NOT enough:
    /// a flat 16384 truncated every large `write` at 262K context and looped a
    /// pi session for hours (2026-07-20). The budget scales with context
    /// (context/2: thinking shares it, and one xhigh design turn on Qwen3.8
    /// spent all of a 24k window's quarter); this cap only bounds a
    /// degenerate runaway generation.
    private static let maxOutput = 65536

    /// The advertised context is declared to the CLI VERBATIM — no second margin.
    ///
    /// The server already reserved headroom before advertising: with `--ctx-size`
    /// absent it pins at 85% of the memory ceiling once, at load time, and that
    /// pinned number is what `clampMaxTokens` and the prompt-length guard enforce.
    /// Discounting it again here would double-count that reserve, and would make
    /// the CLI report a different context than the app's Settings pane shows
    /// (opencode said 75K where the server said 77K — the report that prompted
    /// this). The CLIs keep their prompt inside the window themselves; if one
    /// overshoots, the server's `400 Prompt exceeds maximum context length` is
    /// the correct, loud answer.
    static func forServerContext(_ advertised: Int?) -> Budget {
        guard let advertised, advertised > 0 else { return fallback }
        let output = min(maxOutput, max(1024, advertised / 2))
        return Budget(context: advertised, output: output)
    }

    /// Room an agent keeps free before compacting, and what it keeps after: a
    /// quarter of the window, capped where pi's and opencode2's own 20000-token
    /// defaults (sized for 200k windows) take over. Twin of Zig `compactionReserve`.
    static func compactionReserve(_ context: Int) -> Int {
        min(20000, max(1024, context / 4))
    }

    /// Below this the agent's own fixed prompt leaves every turn compacting or
    /// truncated: Claude Code sends 40-70k before the first word (tool + MCP
    /// schemas, skills catalogue), opencode ~8k, pi ~2k. Twin of Zig `contextFloor`.
    static func contextFloor(agentId: String) -> Int {
        switch agentId {
        case "claude": return 65536
        case "opencode", "opencode2": return 32768
        default: return 16384
        }
    }

    /// Alert text, or nil when the window is enough.
    static func contextWarning(agentId: String, context: Int) -> String? {
        let floor = contextFloor(agentId: agentId)
        guard context > 0, context < floor else { return nil }
        return L10n.formatUngrouped("The model advertises a %lld-token context; %@ needs %lld+ to work well. Raise Context size in Settings ▸ Server, or expect compaction and truncated turns.", context, agentId, floor)
    }
}

/// One chat-capable registry entry as declared to an agent CLI — the model
/// list behind in-agent switching (/model in pi + hermes, /models in
/// opencode). Derived from the server's /v1/models snapshot
/// (`ServerManager.allModels`), LAN `@peer` entries included.
struct AgentModelEntry: Equatable {
    let id: String
    let budget: AgentBudget.Budget
    /// Advertises image input — opencode gates attachments on this.
    let vision: Bool

    /// Chat-capable entries only — media/embedding models never enter a
    /// coding agent's picker. LAN entries go through the `lanAdvertises`
    /// tolerance (empty capabilities = old peer that serves chat). Budgets
    /// derive PER MODEL: the single-model plumbing stamped the loaded
    /// model's budget on whatever id a switch targeted.
    static func chatEntries(from models: [ModelInfo]) -> [AgentModelEntry] {
        var seen = Set<String>()
        var out: [AgentModelEntry] = []
        for m in models {
            let chat = m.lanPeer != nil
                ? m.lanAdvertises("chat")
                : (m.slotKind == .chat && !m.supportsEmbeddings)
            guard chat, !m.name.isEmpty, seen.insert(m.name).inserted else { continue }
            out.append(AgentModelEntry(
                id: m.name,
                budget: AgentBudget.forServerContext(m.contextLength),
                vision: m.supportsVision || m.capabilities.contains("vision")))
        }
        return out
    }
}

/// The config files / env scripts we write for each third-party agent CLI.
/// Pure string builders so the emitted JSON is unit-testable — a malformed
/// config silently strands the user on the CLI's own defaults.
enum AgentConfigs {

    /// Off is an explicit "none"; pi offers xhigh/max only when the map names them.
    /// Valid JSON and JS alike, so models.json and the extension share it.
    static let piThinkingLevelMap = #"{"off": "none", "xhigh": "xhigh", "max": "max"}"#

    /// pi `models.json` — written to the dedicated `~/.mlx-serve/pi/` config
    /// dir (selected via `PI_CODING_AGENT_DIR`), never the user's real
    /// `~/.pi/agent`, so their own providers are never overwritten.
    ///
    /// `apiKey` defaults to the placeholder the loopback-trusted server
    /// ignores; the SANDBOXED session passes the real `--api-key` when one is
    /// set — guest→host traffic arrives non-loopback (via the NAT gateway).
    /// `supportsReasoningEffort: true` is what lets pi's own reasoning-level
    /// picker reach the server. With it false the level was a local label pi
    /// never transmitted, so every request arrived effort-less and took the
    /// server's default. No `thinkingFormat`: pi's `qwen` format sends only
    /// `enable_thinking` and drops the level; the default sends it as
    /// `reasoning_effort`, and `thinkingLevelMap.off` makes off an explicit
    /// "none". Both ride BOTH surfaces (here and the extension's per-model
    /// definition) because applyExtension does not inherit provider compat.
    static func piModelsJSON(baseURL: String, model: String, budget: AgentBudget.Budget,
                             apiKey: String = "mlx-serve") -> String {
        """
        {
          "providers": {
            "mlx": {
              "baseUrl": "\(baseURL)/v1",
              "api": "openai-completions",
              "apiKey": "\(apiKey)",
              "compat": {
                "supportsDeveloperRole": false,
                "supportsReasoningEffort": true,
                "maxTokensField": "max_tokens"
              },
              "models": [
                {"id": "\(model)", "name": "mlx-\(model)", "input": ["text"],
                 "contextWindow": \(budget.context), "maxTokens": \(budget.output), "reasoning": true,
                 "thinkingLevelMap": \(piThinkingLevelMap)}
              ]
            }
          }
        }
        """
    }

    /// pi's global context file — `AGENTS.md` in the agent config dir is
    /// injected into every session's system prompt (pi's resource loader
    /// checks the agent dir before the workspace). It exists to break the
    /// mega-write loop (live 2026-07-20): pi ALWAYS sends its configured
    /// `maxTokens` (<=0 is a models.json validation error, there is no
    /// omit-the-field mode), its `write` tool has NO append flag, and thinking
    /// shares the response budget — so a file bigger than the cap can only
    /// land via chunked bash appends, and a truncated call re-issued
    /// unchanged fails identically forever.
    static func piAgentsMD(budget: AgentBudget.Budget) -> String {
        """
        # mlx-serve local model — session rules

        Each response (thinking + text + tool calls together) has a hard cap of
        \(budget.output) output tokens. A `write` whose content approaches that
        cap is cut off mid-call and can never succeed, however often it is
        retried.

        - Big files: never one giant `write`. Create the file with the first
          ~150 lines, then append the rest in ~150-line chunks with `bash`:
          `cat >> path <<'EOF'` … `EOF`.
        - "arguments may be truncated", or a `write` rejected for missing
          `content` right after a token-limit stop, means the call was cut
          off — do not re-issue it unchanged; split the content into smaller
          pieces instead.
        - Keep commentary before a tool call to one short sentence.
        """
    }

    /// pi live-model-list extension — dropped into the agent config dir's
    /// `extensions/` (host: `~/.mlx-serve/pi`, guest: `/root/.pi/agent`),
    /// where pi auto-discovers `.js`/`.ts` files. The factory fetches the
    /// server's `/v1/models` at session start and registers every
    /// chat-capable model on the `mlx` provider, so in-session `/model`
    /// tracks reality (LAN peers come and go) instead of a launch-time
    /// snapshot. `models.json` keeps the served model as the static
    /// fallback — an unreachable server registers NOTHING.
    ///
    /// Contracts verified against pi 0.80.10:
    /// extensions default-export a factory; `applyExtension` spreads ONLY
    /// the model definition, so `compat` must ride EVERY model (the
    /// provider-level compat in models.json is not inherited); `cost` is a
    /// required field of `ProviderModelConfig`.
    static func piModelsExtensionJS(baseURL: String, apiKey: String = "mlx-serve") -> String {
        """
        // written by mlx-serve — live model list for the `mlx` provider.
        // Regenerated at each launch; edits here are overwritten.
        const API_KEY = "\(apiKey)";
        const FALLBACK_CONTEXT = 32768;
        const COMPAT = {
          supportsDeveloperRole: false,
          supportsReasoningEffort: true,
          maxTokensField: "max_tokens",
        };

        async function fetchMlxModels() {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 4000);
          try {
            const res = await fetch("\(baseURL)/v1/models", {
              headers: { Authorization: "Bearer " + API_KEY },
              signal: controller.signal,
            });
            if (!res.ok) return [];
            const body = await res.json();
            const rows = Array.isArray(body.data) ? body.data : [];
            return rows
              .filter((row) => {
                const caps = Array.isArray(row.capabilities) ? row.capabilities : [];
                // Chat-capable only; empty caps = an old LAN peer that serves chat.
                return caps.length === 0 || caps.includes("chat");
              })
              .map((row) => {
                const meta = row.meta || {};
                const ctx = meta.context_length > 0 ? meta.context_length : FALLBACK_CONTEXT;
                // Mirrors AgentBudget.forServerContext — keep the two in sync.
                const maxTokens = Math.min(65536, Math.max(1024, Math.floor(ctx / 2)));
                const image = Array.isArray(row.input_modalities) && row.input_modalities.includes("image");
                return {
                  id: row.id,
                  name: row.id,
                  reasoning: true,
                  input: image ? ["text", "image"] : ["text"],
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                  contextWindow: ctx,
                  maxTokens: maxTokens,
                  compat: COMPAT,
                  thinkingLevelMap: \(piThinkingLevelMap),
                };
              });
          } catch {
            return []; // unreachable/slow server — the static models.json stands
          } finally {
            clearTimeout(timer);
          }
        }

        export default async function (pi) {
          const models = await fetchMlxModels();
          if (models.length === 0) return;
          pi.registerProvider("mlx", {
            name: "MLX Serve (local)",
            baseUrl: "\(baseURL)/v1",
            apiKey: API_KEY,
            api: "openai-completions",
            models,
            refreshModels: async () => {
              const fresh = await fetchMlxModels();
              return fresh.length > 0 ? fresh : models;
            },
          });
        }
        """
    }

    /// opencode provider block — shipped INLINE via `OPENCODE_CONFIG_CONTENT`
    /// (merges over the user's own config; no file writes). The launch scripts
    /// single-quote it, so the output must never contain a single quote.
    ///
    /// Unlike pi, opencode has no runtime provider-registration hook for
    /// custom providers, so the FULL chat-capable list is baked here — its
    /// in-session /models picker shows exactly these entries, each with its
    /// own limits (never the loaded model's budget stamped on everything).
    /// `pinModel` writes a top-level `"model"` — opencode 2's TUI has no
    /// `--model` flag, so the config is the only place to select one.
    /// `limit.output` is the room opencode keeps free before compacting (it
    /// never sends max_tokens), so it carries the reserve, not the response
    /// cap. `compaction` (opencode2) scales its global buffer/keep to the
    /// pinned model's window: the defaults compact a 24k window before its
    /// first reply.
    /// opencode sends `reasoning_effort` only when the model declares it;
    /// without it every turn ran thinking-off. Variants are its effort picker.
    static let opencodeReasoning = #""options": { "reasoningEffort": "medium" }, "variants": { "none": { "reasoningEffort": "none" }, "low": { "reasoningEffort": "low" }, "medium": { "reasoningEffort": "medium" }, "high": { "reasoningEffort": "high" } }"#

    static func opencodeJSON(baseURL: String, defaultModel: String,
                             entries: [AgentModelEntry], pinModel: Bool = false,
                             compaction: Bool = false) -> String {
        var list = entries
        if !list.contains(where: { $0.id == defaultModel }) {
            list.insert(AgentModelEntry(id: defaultModel, budget: AgentBudget.fallback,
                                        vision: false), at: 0)
        }
        let models = list.map { e -> String in
            let attachment = e.vision ? " \"attachment\": true," : ""
            return "\"\(e.id)\": { \"name\": \"\(e.id) (mlx-serve)\",\(attachment) "
                + "\"limit\": { \"context\": \(e.budget.context), \"output\": \(AgentBudget.compactionReserve(e.budget.context)) }, "
                + opencodeReasoning + " }"
        }.joined(separator: ",\n        ")
        let pinned = pinModel ? "\n  \"model\": \"mlx/\(defaultModel)\"," : ""
        var compactionBlock = ""
        if compaction {
            let ctx = list.first { $0.id == defaultModel }?.budget.context ?? AgentBudget.fallback.context
            let reserve = AgentBudget.compactionReserve(ctx)
            compactionBlock = "\n  \"compaction\": { \"buffer\": \(reserve), \"keep\": { \"tokens\": \(min(15000, reserve)) } },"
        }
        return """
        {
          "$schema": "https://opencode.ai/config.json",\(pinned)\(compactionBlock)
          "skills": { "paths": ["~/.mlx-serve/skills/\(AgentSkills.name)"] },
          "provider": {
            "mlx": {
              "npm": "@ai-sdk/openai-compatible",
              "name": "MLX Serve (local)",
              "options": { "baseURL": "\(baseURL)/v1" },
              "models": {
                \(models)
              }
            }
          }
        }
        """
    }

    /// Single-model convenience — the MAS instructions panel's shape (a user
    /// typing a config by hand gets the minimal one).
    static func opencodeJSON(baseURL: String, model: String, budget: AgentBudget.Budget) -> String {
        opencodeJSON(baseURL: baseURL, defaultModel: model,
                     entries: [AgentModelEntry(id: model, budget: budget, vision: false)])
    }

    static func isLoopbackBaseURL(_ url: String) -> Bool {
        guard let parsed = URL(string: url), let host = parsed.host else { return false }
        if host == "localhost" || host == "::1" { return true }
        return host.hasPrefix("127.")
    }

    /// pi `settings.json`: compaction numbers scaled to the window, everything
    /// else kept (theme, packages, the user's own `enabled`). pi compacts when
    /// context exceeds window - reserveTokens and keeps keepRecentTokens; its
    /// defaults (16384 / 20000) never compact a 24k window while max_tokens
    /// shrinks to 1. Twin of Zig `mergePiSettingsJson`.
    static func piSettingsJSON(existing: String, context: Int) -> String {
        var obj = (try? JSONSerialization.jsonObject(with: Data(existing.utf8))) as? [String: Any] ?? [:]
        var compaction = obj["compaction"] as? [String: Any] ?? [:]
        let reserve = AgentBudget.compactionReserve(context)
        compaction["reserveTokens"] = min(16384, reserve + 4096)
        compaction["keepRecentTokens"] = reserve
        obj["compaction"] = compaction
        guard let out = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: out, encoding: .utf8) else { return "{}" }
        return s.replacingOccurrences(of: "\\/", with: "/")
    }

    static func opencode2CliJSON(existing: String, baseURL: String, apiKey: String? = nil) -> String {
        let data = Data(existing.utf8)
        var obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        var plugins: [[String: Any]] = []
        if let raw = obj["plugins"] as? [Any] {
            plugins = raw.compactMap { $0 as? [String: Any] }
        }
        plugins.removeAll { p in
            let pkg = p["package"] as? String ?? ""
            return pkg == "./plugins/mlx-serve" || pkg == "mlx-serve" || pkg.hasSuffix("/mlx-serve")
        }
        let trimmed = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        var options: [String: Any] = ["metricsUrl": trimmed + "/metrics.json"]
        let token: String?
        if let k = apiKey, !k.isEmpty { token = k }
        else if !isLoopbackBaseURL(baseURL) { token = "mlx-serve" }
        else { token = nil }
        if let token { options["metricsToken"] = token }
        plugins.append(["package": "./plugins/mlx-serve", "options": options])
        obj["plugins"] = plugins
        guard let out = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: out, encoding: .utf8) else { return "{}" }
        return s.replacingOccurrences(of: "\\/", with: "/")
    }

    static func opencode2PluginSourceDir() -> URL? {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("lib/opencode2-mlx-serve")
        let candidates = [
            Bundle.main.resourceURL?.appendingPathComponent("opencode2-mlx-serve"),
            repo,
        ]
        return candidates.compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func copyOpencode2Plugin(to dest: String) {
        guard let src = opencode2PluginSourceDir() else { return }
        let fm = FileManager.default
        try? fm.createDirectory(atPath: dest, withIntermediateDirectories: true)
        guard let names = try? fm.contentsOfDirectory(atPath: src.path) else { return }
        for name in names {
            if name.hasSuffix(".test.ts") { continue }
            let keep = name.hasSuffix(".ts") || name == "tui.tsx" || name == "package.json" || name == "LICENSE"
            if !keep { continue }
            let from = (src.path as NSString).appendingPathComponent(name)
            let to = (dest as NSString).appendingPathComponent(name)
            try? fm.removeItem(atPath: to)
            try? fm.copyItem(atPath: from, toPath: to)
        }
    }

    /// oh-my-pi (omp) `models.yml` — written to the dedicated
    /// `~/.mlx-serve/omp/` config dir, never the user's real `~/.omp/agent`.
    /// The dir is selected via `PI_CODING_AGENT_DIR` — measured against omp
    /// v17: the changelog's `OMP_CODING_AGENT_DIR` rename reached only its
    /// help text, the env read is still the pi spelling (launch scripts
    /// export BOTH so a completed rename keeps working).
    ///
    /// The model list is STATIC, one entry per chat-capable model, like
    /// opencode's — deliberately NOT omp's `discovery: openai-models-list`:
    /// discovery lists every /v1/models row, so media/embedding models would
    /// enter the coding agent's picker (each at omp's 128k default context,
    /// since a media row has no context to advertise). Users who wire omp's
    /// discovery themselves still get real per-model context from the rows'
    /// top-level `max_model_len`/`context_length` twins (issue #188).
    /// `compat` keys verified against the omp schema (same vocabulary as
    /// pi's, `thinkingFormat: qwen` included).
    static func ompModelsYML(baseURL: String, defaultModel: String,
                             entries: [AgentModelEntry],
                             apiKey: String = "mlx-serve") -> String {
        var list = entries
        if !list.contains(where: { $0.id == defaultModel }) {
            list.insert(AgentModelEntry(id: defaultModel, budget: AgentBudget.fallback,
                                        vision: false), at: 0)
        }
        let models = list.map { e -> String in
            """
                  - id: "\(e.id)"
                    name: "\(e.id) (mlx-serve)"
                    reasoning: true
                    input: [\(e.vision ? "text, image" : "text")]
                    cost:
                      input: 0
                      output: 0
                      cacheRead: 0
                      cacheWrite: 0
                    contextWindow: \(e.budget.context)
                    maxTokens: \(e.budget.output)
            """
        }.joined(separator: "\n")
        return """
        # written by mlx-serve — custom `mlx` provider for oh-my-pi (omp).
        # Regenerated at each launch; edits here are overwritten.
        providers:
          mlx:
            baseUrl: \(baseURL)/v1
            api: openai-completions
            apiKey: \(apiKey)
            compat:
              supportsDeveloperRole: false
              supportsReasoningEffort: true
              maxTokensField: max_tokens
              thinkingFormat: qwen
            models:
        \(models)
        """
    }

    /// Single-model convenience — the MAS instructions panel's shape.
    static func ompModelsYML(baseURL: String, model: String,
                             budget: AgentBudget.Budget) -> String {
        ompModelsYML(baseURL: baseURL, defaultModel: model,
                     entries: [AgentModelEntry(id: model, budget: budget, vision: false)])
    }

    /// A model-derived arg, quoted only when needed so plain ids keep the exact
    /// script bytes (twin of launch.zig `appendModelArg`).
    static func shellArg(_ s: String) -> String {
        let safe = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/+:@=")
        return !s.isEmpty && s.unicodeScalars.allSatisfy(safe.contains) ? s : CLIInstaller.shellQuote(s)
    }

    /// codex launch-line overrides, merged over the user's own config.toml so
    /// nothing is written into their Codex home. Responses wire API only;
    /// keyless (no `env_key`). Twin of launch.zig `codexConfigOverrides`.
    static func codexConfigArgs(baseURL: String, model: String,
                                budget: AgentBudget.Budget) -> String {
        func opt(_ key: String, _ value: String) -> String {
            let toml = value.replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return "-c " + CLIInstaller.shellQuote("\(key)=\"\(toml)\"")
        }
        var args = [opt("model", model), opt("model_provider", "mlx")]
        if budget.context > 0 { args.append("-c model_context_window=\(budget.context)") }
        args += [opt("model_providers.mlx.name", "MLX Serve (local)"),
                 opt("model_providers.mlx.base_url", "\(baseURL)/v1"),
                 opt("model_providers.mlx.wire_api", "responses")]
        return args.joined(separator: " ")
    }

    /// Shell snippet that resolves the codex binary: PATH first, then the
    /// CLI bundled inside the desktop app (codex's rebranded app installs as
    /// ChatGPT.app or Codex.app — its own launcher checks both names in
    /// /Applications and ~/Applications, bundle id com.openai.codex — and
    /// ships the CLI at Contents/Resources/codex). Shared by the DMG launch
    /// script, the MAS instructions tab, and mirrored by `mlx-serve launch`
    /// (launch.zig), so a desktop-app-only user gets a working launch.
    static let codexBinResolver = """
        CODEX_BIN="$(command -v codex)"
        if [ -z "$CODEX_BIN" ]; then
          for app in "/Applications/ChatGPT.app" "/Applications/Codex.app" "$HOME/Applications/ChatGPT.app" "$HOME/Applications/Codex.app"; do
            if [ -x "$app/Contents/Resources/codex" ]; then CODEX_BIN="$app/Contents/Resources/codex"; break; fi
          done
        fi
        if [ -z "$CODEX_BIN" ]; then echo "codex is not installed: npm install -g @openai/codex, or install the ChatGPT app"; exit 127; fi
        """

    /// aider model metadata (litellm's registry format) — tells aider the
    /// real context window of every `openai/<id>` model so its budgeting and
    /// warnings work; without it unknown models get litellm defaults. One
    /// entry per chat-capable model, the served model force-included.
    static func aiderModelMetadataJSON(model: String, budget: AgentBudget.Budget,
                                       entries: [AgentModelEntry]) -> String {
        var list = entries
        if !list.contains(where: { $0.id == model }) {
            list.insert(AgentModelEntry(id: model, budget: budget, vision: false), at: 0)
        }
        let rows = list.map { e -> String in
            """
              "openai/\(e.id)": {
                "max_input_tokens": \(e.budget.context),
                "max_output_tokens": \(e.budget.output),
                "max_tokens": \(e.budget.output),
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": "openai",
                "mode": "chat"
              }
            """
        }.joined(separator: ",\n")
        return "{\n\(rows)\n}"
    }

    /// fx keeps custom providers only in `~/.fx/settings.json` (no config-dir
    /// override), so we own ONE key there, `providers.mlx-serve`, selected per
    /// launch with FX_PROVIDER/FX_MODEL: the user's default provider and every
    /// other setting stay as found. fx rejects unknown keys. Twin of Zig
    /// `launch.mergeFxSettingsJson`.
    static let fxProvider = "mlx-serve"

    /// The user's fx settings with our provider set; nil when the existing
    /// file is not a JSON object — never replace a file we cannot read.
    static func fxSettingsJSON(existing: String, baseURL: String,
                               entries: [AgentModelEntry]) -> String? {
        let blank = existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard var obj = blank ? [:]
                : (try? JSONSerialization.jsonObject(with: Data(existing.utf8))) as? [String: Any]
        else { return nil }
        var metadata: [String: Any] = [:]
        for e in entries {
            metadata[e.id] = ["context_window": e.budget.context,
                              "max_output_tokens": e.budget.output,
                              "supports_tool_use": true,
                              "supports_vision": e.vision]
        }
        var providers = obj["providers"] as? [String: Any] ?? [:]
        providers[fxProvider] = ["protocol": "openai-chat-completions",
                                 "base_url": "\(baseURL)/v1",
                                 "auth": ["type": "none"],
                                 "model_metadata": metadata]
        obj["providers"] = providers
        guard let data = try? JSONSerialization.data(
                withJSONObject: obj, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// grok `config.toml` under a dedicated GROK_HOME. A dummy XAI_API_KEY
    /// fails grok's key probe against xAI, so the credential is each model's
    /// `api_key`; helper calls default to xAI model ids, which would load the
    /// server's default model, so they are pinned to the launched one. The
    /// served model is force-included. Twin of Zig `launch.grokConfigToml`.
    static func grokConfigTOML(baseURL: String, model: String, budget: AgentBudget.Budget,
                               entries: [AgentModelEntry]) -> String {
        var list = entries
        if !list.contains(where: { $0.id == model }) {
            list.insert(AgentModelEntry(id: model, budget: budget, vision: false), at: 0)
        }
        let models = list.map { e in
            """

            [model."\(e.id)"]
            model = "\(e.id)"
            base_url = "\(baseURL)/v1"
            name = "\(e.id) (mlx-serve)"
            api_key = "mlx-serve"
            context_window = \(e.budget.context)
            max_completion_tokens = \(e.budget.output)
            supports_reasoning_effort = true
            reasoning_efforts = ["none", "low", "medium", "high"]
            inference_idle_timeout_secs = 1800

            """
        }.joined()
        return """
        # written by mlx-serve — dedicated GROK_HOME, regenerated at each launch.
        [models]
        default = "\(model)"
        session_summary = "\(model)"
        image_description = "\(model)"
        prompt_suggestion = "\(model)"

        """ + models
    }

    /// ZCode personal provider config — twin of Zig `launch.zcodeConfigJson`:
    /// one rule per chat model so ZCode never guesses limits from the id.
    /// The served model is force-included.
    static func zcodeProviderJSON(baseURL: String, model: String, budget: AgentBudget.Budget,
                                  entries: [AgentModelEntry]) -> String {
        var list = entries
        if !list.contains(where: { $0.id == model }) {
            list.insert(AgentModelEntry(id: model, budget: budget, vision: false), at: 0)
        }
        let rules: [[String: Any]] = list.map { e in
            ["providerId": "mlx", "modelId": e.id, "config": [
                "enabled": true,
                "properties": [
                    "contextWindow": e.budget.context, "requiresMfjsToolSchema": false,
                    "inputFormat": ["supportsText": true, "supportsImage": e.vision,
                                    "supportsVideo": false, "supportsAudio": false, "supportsPdf": false],
                    "outputFormat": ["supportsText": true], "supportsToolCall": true,
                    "supportsJsonSchemaOutput": false, "supportsNativeWebSearch": false,
                    "supportsMidConversationSystem": false,
                ],
                "optionSpecs": [
                    "reasoningLevel": ["values": ["none", "low", "medium", "high"],
                                       "map": "{\"reasoning_effort\": reasoningLevel}"],
                    "maxOutputTokens": ["max": e.budget.output,
                                        "map": "{\"max_tokens\": maxOutputTokens}"],
                ],
            ]]
        }
        let config: [String: Any] = ["schemaVersion": 1, "config": [
            "providerOrder": ["mlx"],
            "defaultModelSelection": ["providerId": "mlx", "modelId": model,
                                      "options": ["reasoningLevel": "medium"]],
            "providerConfigRules": ["providerRules": [[
                "providerId": "mlx", "providerName": "mlx-serve", "enabled": true,
                "config": ["group": "standard-personal",
                           "access": ["type": "api-key", "apiKey": "mlx-serve"],
                           "api": ["type": "openai-chat-completions", "baseUrl": "\(baseURL)/v1"],
                           "personalModelIds": list.map(\.id)],
            ]]],
            "modelConfigRules": ["manualProviderModelRules": [], "providerModelRules": rules],
        ]]
        let data = try! JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    /// ZCode reads its whole state from these, so the launch never touches
    /// the user's own ~/.zcode.
    static let zcodeExports = #"""
    export ZCODE_DATA_BASE_DIR="$HOME/.mlx-serve/zcode"
    export ZCODE_STORAGE_DIR="$HOME/.mlx-serve/zcode/storage"
    export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="$HOME/.mlx-serve/zcode/provider_config.json"
    """#

    /// hermes `.env` — the first-run wizard kill switch: hermes's
    /// `_has_any_provider_configured()` is satisfied by `OPENAI_BASE_URL`
    /// alone, and the file lives under HERMES_HOME (hermes_constants.py), so
    /// it rides the same dedicated dir as config.yaml.
    static func hermesEnvFile(baseURL: String, apiKey: String = "mlx-serve") -> String {
        """
        # written by mlx-serve — OPENAI_BASE_URL marks a provider as configured,
        # which is what keeps the first-run setup wizard out of the session.
        OPENAI_BASE_URL=\(baseURL)/v1
        OPENAI_API_KEY=\(apiKey)
        """
    }

    /// hermes `config.yaml` — mirrors EXACTLY what `hermes setup`'s
    /// custom-endpoint flow saves (verified against hermes_cli source, never
    /// its docs), plus one entry under `custom_providers[].models` per
    /// chat-capable model so in-session `/model` can switch among them
    /// (`models.<id>.context_length` is hermes's per-model context key).
    /// The served model stays `default:` and is force-included.
    static func hermesConfigYAML(baseURL: String, apiKey: String, model: String,
                                 budget: AgentBudget.Budget,
                                 entries: [AgentModelEntry]) -> String {
        var list = entries
        if !list.contains(where: { $0.id == model }) {
            list.insert(AgentModelEntry(id: model, budget: budget, vision: false), at: 0)
        }
        let models = list.map {
            "      \"\($0.id)\":\n        context_length: \($0.budget.context)"
        }.joined(separator: "\n")
        return """
        # written by mlx-serve (Agent Sandbox) — rewritten at each session start.
        # Mirrors what `hermes setup`'s custom-endpoint flow saves, so the first
        # run starts configured instead of launching the wizard. Every entry
        # under `models:` is switchable in-session via /model.
        model:
          default: "\(model)"
          provider: custom
          base_url: "\(baseURL)/v1"
          api_key: "\(apiKey)"
          api_mode: chat_completions
        custom_providers:
          - name: mlx-serve
            base_url: "\(baseURL)/v1"
            api_key: "\(apiKey)"
            model: "\(model)"
            api_mode: chat_completions
            models:
        \(models)
        """
    }

    /// Env exports for the Claude Code launch script (no trailing newline).
    /// Twin of Zig `launch.scriptFor(.claude, …)` — change both together.
    static func claudeCodeExports(baseURL: String, model: String, budget: AgentBudget.Budget) -> String {
        // A model outside Claude Code's own catalog is assumed to hold 200k and
        // auto-compacted there; CLAUDE_CODE_MAX_CONTEXT_TOKENS is the documented
        // override. Declared VERBATIM like every other agent's context field —
        // and omitted entirely when the server advertised nothing.
        let contextExport = budget.context > 0
            ? "\nexport CLAUDE_CODE_MAX_CONTEXT_TOKENS=\(budget.context)" : ""
        return """
        export ANTHROPIC_BASE_URL='\(baseURL)'
        export ANTHROPIC_API_KEY=
        export ANTHROPIC_AUTH_TOKEN=mlx-serve
        export CLAUDE_CODE_ATTRIBUTION_HEADER=0
        export ANTHROPIC_DEFAULT_OPUS_MODEL=\(model)
        export ANTHROPIC_DEFAULT_SONNET_MODEL=\(model)
        export ANTHROPIC_DEFAULT_HAIKU_MODEL=\(model)
        export CLAUDE_CODE_SUBAGENT_MODEL=\(model)
        export CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(budget.output)\(contextExport)
        """
    }
}
