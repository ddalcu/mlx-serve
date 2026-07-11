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

    /// Cap on a single response. Enough for a one-shot whole-file `write`
    /// (measured: 8–11k tokens), without inviting a mega-write on a huge context.
    private static let maxOutput = 16384

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

/// The config files / env scripts we write for each third-party agent CLI.
/// Pure string builders so the emitted JSON is unit-testable — a malformed
/// config silently strands the user on the CLI's own defaults.
enum AgentConfigs {

    /// pi `models.json` — written to the dedicated `~/.mlx-serve/pi/` config
    /// dir (selected via `PI_CODING_AGENT_DIR`), never the user's real
    /// `~/.pi/agent`, so their own providers are never overwritten.
    static func piModelsJSON(baseURL: String, model: String, budget: AgentBudget.Budget) -> String {
        """
        {
          "providers": {
            "mlx": {
              "baseUrl": "\(baseURL)/v1",
              "api": "openai-completions",
              "apiKey": "mlx-serve",
              "compat": {
                "supportsDeveloperRole": false,
                "supportsReasoningEffort": false,
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

    /// opencode provider block — shipped INLINE via `OPENCODE_CONFIG_CONTENT`
    /// (merges over the user's own config; no file writes). The launch scripts
    /// single-quote it, so the output must never contain a single quote.
    static func opencodeJSON(baseURL: String, model: String, budget: AgentBudget.Budget) -> String {
        """
        {
          "$schema": "https://opencode.ai/config.json",
          "provider": {
            "mlx": {
              "npm": "@ai-sdk/openai-compatible",
              "name": "MLX Serve (local)",
              "options": { "baseURL": "\(baseURL)/v1" },
              "models": {
                "\(model)": {
                  "name": "\(model) (mlx-serve)",
                  "limit": { "context": \(budget.context), "output": \(budget.output) }
                }
              }
            }
          }
        }
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
