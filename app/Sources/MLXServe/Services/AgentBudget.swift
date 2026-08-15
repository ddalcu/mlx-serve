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
    /// (context/4); this cap only bounds a degenerate runaway generation.
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
        let output = min(maxOutput, max(1024, advertised / 4))
        return Budget(context: advertised, output: output)
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
    /// server's default. It rides BOTH surfaces (here and the extension's
    /// per-model COMPAT) because applyExtension does not inherit provider
    /// compat — `AgentBudgetTests` pins the two together.
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
                "maxTokensField": "max_tokens",
                "thinkingFormat": "qwen"
              },
              "models": [
                {"id": "\(model)", "name": "mlx-\(model)", "input": ["text"],
                 "contextWindow": \(budget.context), "maxTokens": \(budget.output), "reasoning": true}
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
    /// Contracts verified against pi 0.80.10 (the pinned sandbox version):
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
          thinkingFormat: "qwen",
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
                const maxTokens = Math.min(65536, Math.max(1024, Math.floor(ctx / 4)));
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
    static func opencodeJSON(baseURL: String, defaultModel: String,
                             entries: [AgentModelEntry]) -> String {
        var list = entries
        if !list.contains(where: { $0.id == defaultModel }) {
            list.insert(AgentModelEntry(id: defaultModel, budget: AgentBudget.fallback,
                                        vision: false), at: 0)
        }
        let models = list.map { e -> String in
            let attachment = e.vision ? " \"attachment\": true," : ""
            return "\"\(e.id)\": { \"name\": \"\(e.id) (mlx-serve)\",\(attachment) "
                + "\"limit\": { \"context\": \(e.budget.context), \"output\": \(e.budget.output) } }"
        }.joined(separator: ",\n        ")
        return """
        {
          "$schema": "https://opencode.ai/config.json",
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
    static func claudeCodeExports(baseURL: String, model: String, budget: AgentBudget.Budget) -> String {
        """
        export ANTHROPIC_BASE_URL='\(baseURL)'
        export ANTHROPIC_API_KEY=
        export ANTHROPIC_AUTH_TOKEN=mlx-serve
        export CLAUDE_CODE_ATTRIBUTION_HEADER=0
        export ANTHROPIC_DEFAULT_OPUS_MODEL=\(model)
        export ANTHROPIC_DEFAULT_SONNET_MODEL=\(model)
        export ANTHROPIC_DEFAULT_HAIKU_MODEL=\(model)
        export CLAUDE_CODE_SUBAGENT_MODEL=\(model)
        export CLAUDE_CODE_MAX_OUTPUT_TOKENS=\(budget.output)
        """
    }
}
