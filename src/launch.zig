//! `mlx-serve launch <agent>` — configure and launch a third-party coding
//! agent against the local server, ollama-style (issue #188).
//!
//! The Swift app's `CLILauncher` + `AgentConfigs` are the DMG twin of this
//! file: same dedicated config dirs (`~/.mlx-serve/<agent>/`, NEVER a user's
//! real agent config), same env vars, same file shapes. Documented
//! duplication — change a contract on one side, change it on both
//! (`CLISetupInstructionsTests` pins the Swift side, the tests here and
//! `tests/test_launch_cmd.sh` pin this one).
//!
//! Flow: probe the server; if it's down, start the MLX Core app (`open -g -a`)
//! and wait — no app installed means instructions, not a mystery. Then read
//! `/v1/models`, derive each model's budget from its ADVERTISED context
//! (AgentBudget's formula: output = clamp(ctx/4, 1024, 65536) — never a
//! hardcoded window), write the agent's config, and exec it through a login
//! zsh so the user's PATH (nvm, Homebrew, ~/.local/bin) resolves.

const std = @import("std");
const log = @import("log.zig");
const opencode2_plugin = @import("opencode2_plugin");

pub const Budget = struct { context: u64, output: u64 };

/// Mirrors Swift `AgentBudget.fallback` — used when the server advertises no
/// context (older build, unloaded stub with no readable config).
pub const FALLBACK_BUDGET = Budget{ .context = 32768, .output = 8192 };

/// Mirrors Swift `AgentBudget.forServerContext`.
pub fn budgetForContext(ctx: u64) Budget {
    if (ctx == 0) return FALLBACK_BUDGET;
    return .{ .context = ctx, .output = @min(65536, @max(1024, ctx / 4)) };
}

/// One chat-capable /v1/models row as declared to an agent CLI.
pub const Entry = struct {
    id: []const u8,
    budget: Budget,
    vision: bool,
    loaded: bool,
};

pub const AgentKind = enum {
    claude,
    pi,
    omp,
    opencode,
    opencode2,
    codex,
    hermes,
    aider,

    pub fn fromName(name: []const u8) ?AgentKind {
        // The codex rebrand: issue #188 asks for `mlx-serve launch chatgpt`.
        if (std.mem.eql(u8, name, "chatgpt")) return .codex;
        inline for (@typeInfo(AgentKind).@"enum".field_names, 0..) |f, i| {
            if (std.mem.eql(u8, name, f)) return @fromBackingInt(@intCast(i));
        }
        return null;
    }

    pub const names = "claude, pi, omp, opencode, opencode2, codex, hermes, aider";
};

// ── Config builders (pure — unit-tested below) ──────────────────────────

/// pi `models.json` — same shape the app's `AgentConfigs.piModelsJSON`
/// writes, with every chat-capable model in the array so in-session
/// `/model` can switch (the app adds a live-list extension on top; the CLI
/// bakes the launch-time snapshot).
pub fn piModelsJson(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\{{
        \\  "providers": {{
        \\    "mlx": {{
        \\      "baseUrl": "{s}/v1",
        \\      "api": "openai-completions",
        \\      "apiKey": "mlx-serve",
        \\      "compat": {{
        \\        "supportsDeveloperRole": false,
        \\        "supportsReasoningEffort": true,
        \\        "maxTokensField": "max_tokens",
        \\        "thinkingFormat": "qwen"
        \\      }},
        \\      "models": [
    , .{base_url});
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}
            \\        {{"id": "{s}", "name": "{s} (mlx-serve)", "input": [{s}],
            \\         "contextWindow": {d}, "maxTokens": {d}, "reasoning": true}}
        , .{
            if (i == 0) "" else ",",
            e.id,
            e.id,
            if (e.vision) "\"text\", \"image\"" else "\"text\"",
            e.budget.context,
            e.budget.output,
        });
    }
    try out.appendSlice(allocator,
        \\
        \\      ]
        \\    }
        \\  }
        \\}
    );
    return out.toOwnedSlice(allocator);
}

/// oh-my-pi `models.yml` — static chat-capable list, deliberately not omp's
/// openai-models-list discovery (it would put every media model in the
/// coding picker at omp's 128k default). Same rationale as the app builder.
pub fn ompModelsYml(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\# written by mlx-serve — custom `mlx` provider for oh-my-pi (omp).
        \\# Regenerated at each launch; edits here are overwritten.
        \\providers:
        \\  mlx:
        \\    baseUrl: {s}/v1
        \\    api: openai-completions
        \\    apiKey: mlx-serve
        \\    compat:
        \\      supportsDeveloperRole: false
        \\      supportsReasoningEffort: true
        \\      maxTokensField: max_tokens
        \\      thinkingFormat: qwen
        \\    models:
        \\
    , .{base_url});
    for (entries) |e| {
        try out.print(allocator,
            \\      - id: "{s}"
            \\        name: "{s} (mlx-serve)"
            \\        reasoning: true
            \\        input: [{s}]
            \\        cost:
            \\          input: 0
            \\          output: 0
            \\          cacheRead: 0
            \\          cacheWrite: 0
            \\        contextWindow: {d}
            \\        maxTokens: {d}
            \\
        , .{ e.id, e.id, if (e.vision) "text, image" else "text", e.budget.context, e.budget.output });
    }
    return out.toOwnedSlice(allocator);
}

/// opencode config — carried inline via OPENCODE_CONFIG_CONTENT (merges over
/// the user's own config, no file writes). Single-quoted in the script, so
/// the JSON must stay single-quote-free.
pub fn opencodeJson(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\{{"$schema": "https://opencode.ai/config.json", "provider": {{"mlx": {{"npm": "@ai-sdk/openai-compatible", "name": "MLX Serve (local)", "options": {{"baseURL": "{s}/v1"}}, "models": {{
    , .{base_url});
    for (entries, 0..) |e, i| {
        try out.print(allocator, "{s}\"{s}\": {{\"name\": \"{s} (mlx-serve)\",{s} \"limit\": {{\"context\": {d}, \"output\": {d}}}}}", .{
            if (i == 0) "" else ", ",
            e.id,
            e.id,
            if (e.vision) " \"attachment\": true," else "",
            e.budget.context,
            e.budget.output,
        });
    }
    try out.appendSlice(allocator, "}}}}");
    return out.toOwnedSlice(allocator);
}

pub fn isLoopbackBaseUrl(url: []const u8) bool {
    var rest = url;
    if (std.mem.startsWith(u8, rest, "http://")) {
        rest = rest["http://".len..];
    } else if (std.mem.startsWith(u8, rest, "https://")) {
        rest = rest["https://".len..];
    }
    if (std.mem.indexOfScalar(u8, rest, '/')) |slash| rest = rest[0..slash];
    if (rest.len >= 2 and rest[0] == '[') {
        if (std.mem.indexOfScalar(u8, rest, ']')) |end| {
            return std.mem.eql(u8, rest[1..end], "::1");
        }
    }
    const host = if (std.mem.lastIndexOfScalar(u8, rest, ':')) |colon| rest[0..colon] else rest;
    if (std.mem.eql(u8, host, "localhost")) return true;
    if (std.mem.eql(u8, host, "::1")) return true;
    return std.mem.startsWith(u8, host, "127.");
}

/// What to tell the user about the monitor plugin's feed, from the status
/// `GET /metrics.json` returned. The CLI server defaults to metrics OFF and
/// answers 503; the plugin then shows `feed --metrics off` with an empty panel.
pub fn metricsFeedNote(status: u16) ?[]const u8 {
    return switch (status) {
        200 => null,
        503 => "this server was started without --metrics: the OpenCode 2 sidebar panel will read '--metrics off' (the footer turn meter still works). Restart with `mlx-serve serve --metrics` for the full panel.",
        401, 403 => "GET /metrics.json is refused (api key): the OpenCode 2 sidebar panel will read '401 unauthorized'.",
        else => "GET /metrics.json did not answer 200: the OpenCode 2 sidebar panel will read 'unreachable'.",
    };
}

fn isMlxServePlugin(v: std.json.Value) bool {
    if (v != .object) return false;
    const pkg = v.object.get("package") orelse return false;
    if (pkg != .string) return false;
    const s = pkg.string;
    if (std.mem.eql(u8, s, "./plugins/mlx-serve")) return true;
    if (std.mem.eql(u8, s, "mlx-serve")) return true;
    return std.mem.endsWith(u8, s, "/mlx-serve");
}

pub fn mergeOpencode2CliJson(
    allocator: std.mem.Allocator,
    existing: []const u8,
    base_url: []const u8,
    known_api_key: ?[]const u8,
) ![]u8 {
    const trimmed = std.mem.trim(u8, existing, " \t\r\n");
    const body = if (trimmed.len == 0) "{}" else existing;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
        try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    if (parsed.value != .object) {
        parsed.deinit();
        parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    }
    defer parsed.deinit();
    const a = parsed.arena.allocator();

    var kept = std.json.Array.init(a);
    if (parsed.value.object.get("plugins")) |pv| {
        if (pv == .array) {
            for (pv.array.items) |item| {
                if (isMlxServePlugin(item)) continue;
                try kept.append(item);
            }
        }
    }

    const metrics_url = try std.fmt.allocPrint(a, "{s}/metrics.json", .{std.mem.trimEnd(u8, base_url, "/")});
    var options: std.json.ObjectMap = .empty;
    try options.put(a, "metricsUrl", .{ .string = metrics_url });
    const token: ?[]const u8 = if (known_api_key) |k|
        (if (k.len > 0) k else null)
    else if (!isLoopbackBaseUrl(base_url))
        "mlx-serve"
    else
        null;
    if (token) |tok| {
        try options.put(a, "metricsToken", .{ .string = tok });
    }

    var entry: std.json.ObjectMap = .empty;
    try entry.put(a, "package", .{ .string = "./plugins/mlx-serve" });
    try entry.put(a, "options", .{ .object = options });
    try kept.append(.{ .object = entry });

    var obj = parsed.value.object;
    try obj.put(a, "plugins", .{ .array = kept });
    parsed.value = .{ .object = obj };
    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

/// The user's real Codex home: `${CODEX_HOME:-$HOME/.codex}`, empty = unset.
/// The profile writes there and the launched codex inherits the same value
/// (never overwritten), so the two always agree.
fn codexHome() []const u8 {
    if (std.c.getenv("CODEX_HOME")) |v| {
        const s = std.mem.span(v);
        if (s.len > 0) return s;
    }
    return std.fmt.allocPrint(std.heap.page_allocator, "{s}/.codex", .{homeDir()}) catch "/tmp/.codex";
}

/// True for the top-level tables codex itself writes back into the profile
/// file — `[projects."<cwd>"]` holds `trust_level`, which codex persists
/// wherever it last saw a trust prompt answered, profile layer included.
fn isCodexOwnedTable(header: []const u8) bool {
    const after = std.mem.trim(u8, header[1..], " \t");
    return std.mem.startsWith(u8, after, "projects.");
}

/// codex `mlx-serve.config.toml` — the `--profile mlx-serve` layer the
/// launcher adds, merged over the user's own config.toml. Responses wire API
/// only (codex-rs `WireApi` has one variant), pointing at our /v1/responses.
/// Keyless: no `env_key` and `requires_openai_auth` unset means codex skips
/// login; the loopback server ignores keys anyway.
///
/// `existing` is the file already on disk (empty when there is none): the
/// generated keys are regenerated verbatim, but codex's OWN `[projects.*]`
/// trust tables are carried over at the tail — codex persists `trust_level`
/// into this profile layer, and dropping them would make it re-ask directory
/// trust on every launch.
pub fn codexConfigToml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, budget: Budget, existing: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator,
        \\# Generated by mlx-serve. Regenerated on each launch (your [projects.*]
        \\# trust entries survive; put personal settings in config.toml).
        \\
    );
    try out.print(allocator,
        \\model = "{s}"
        \\model_provider = "mlx"
        \\model_context_window = {d}
        \\
        \\[model_providers.mlx]
        \\name = "MLX Serve (local)"
        \\base_url = "{s}/v1"
        \\wire_api = "responses"
        \\
    , .{ model, budget.context, base_url });
    var rest = existing;
    while (std.mem.indexOfScalar(u8, rest, '\n')) |nl| {
        const line = rest[0 .. nl + 1];
        rest = rest[nl + 1 ..];
        if (line[0] != '[') continue;
        if (isCodexOwnedTable(line)) {
            try out.appendSlice(allocator, "\n");
            try out.appendSlice(allocator, line);
            var body_end: usize = 0;
            while (body_end < rest.len) {
                const nl2 = std.mem.indexOfScalar(u8, rest[body_end..], '\n') orelse break;
                const next = rest[body_end .. body_end + nl2];
                if (next.len > 0 and next[0] == '[') break;
                if (std.mem.trim(u8, next, " \t\r").len == 0) {
                    if (body_end > 0) break;
                    body_end += nl2 + 1;
                    continue;
                }
                body_end += nl2 + 1;
            }
            while (body_end > 0 and std.mem.trim(u8, rest[body_end - 1 ..], " \t\r\n").len == 0) body_end -= 1;
            if (body_end > 0 and body_end < rest.len) body_end += 1;
            try out.appendSlice(allocator, rest[0..body_end]);
            rest = rest[body_end..];
        }
    }
    return out.toOwnedSlice(allocator);
}

/// hermes `config.yaml` — mirrors what `hermes setup`'s custom-endpoint flow
/// saves (see the app's AgentConfigs.hermesConfigYAML; verified against
/// hermes_cli source).
pub fn hermesConfigYaml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\# written by mlx-serve — regenerated at each launch. Mirrors what
        \\# `hermes setup`'s custom-endpoint flow saves, so the first run starts
        \\# configured instead of launching the wizard.
        \\model:
        \\  default: "{s}"
        \\  provider: custom
        \\  base_url: "{s}/v1"
        \\  api_key: "mlx-serve"
        \\  api_mode: chat_completions
        \\custom_providers:
        \\  - name: mlx-serve
        \\    base_url: "{s}/v1"
        \\    api_key: "mlx-serve"
        \\    model: "{s}"
        \\    api_mode: chat_completions
        \\    models:
        \\
    , .{ model, base_url, base_url, model });
    for (entries) |e| {
        try out.print(allocator, "      \"{s}\":\n        context_length: {d}\n", .{ e.id, e.budget.context });
    }
    return out.toOwnedSlice(allocator);
}

/// hermes `.env` — the first-run wizard kill switch: OPENAI_BASE_URL alone
/// marks a provider as configured. Lives under HERMES_HOME like config.yaml.
pub fn hermesEnvFile(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\# written by mlx-serve — OPENAI_BASE_URL marks a provider as configured,
        \\# which is what keeps the first-run setup wizard out of the session.
        \\OPENAI_BASE_URL={s}/v1
        \\OPENAI_API_KEY=mlx-serve
        \\
    , .{base_url});
}

/// aider model metadata (litellm's registry format) — the real context
/// window for every openai/<id> model.
pub fn aiderMetadataJson(allocator: std.mem.Allocator, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\n");
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}  "openai/{s}": {{
            \\    "max_input_tokens": {d},
            \\    "max_output_tokens": {d},
            \\    "max_tokens": {d},
            \\    "input_cost_per_token": 0,
            \\    "output_cost_per_token": 0,
            \\    "litellm_provider": "openai",
            \\    "mode": "chat"
            \\  }}
        , .{ if (i == 0) "" else ",\n", e.id, e.budget.context, e.budget.output, e.budget.output });
    }
    try out.appendSlice(allocator, "\n}\n");
    return out.toOwnedSlice(allocator);
}

// ── Launch script assembly ──────────────────────────────────────────────

/// Shell-quote one extra passthrough arg (single quotes, '\'' escape).
fn appendQuoted(out: *std.ArrayList(u8), allocator: std.mem.Allocator, arg: []const u8) !void {
    try out.append(allocator, '\'');
    for (arg) |c| {
        if (c == '\'') try out.appendSlice(allocator, "'\\''") else try out.append(allocator, c);
    }
    try out.append(allocator, '\'');
}

fn appendExtras(out: *std.ArrayList(u8), allocator: std.mem.Allocator, extras: []const []const u8) !void {
    for (extras) |a| {
        try out.append(allocator, ' ');
        try appendQuoted(out, allocator, a);
    }
}

/// The script body run through `/bin/zsh -l -c` (login shell = the user's
/// real PATH). Configs are written by `writeConfigs` BEFORE this runs; the
/// script only exports env and execs the agent — same split as the app's
/// prepareConfig / scriptBody.
pub fn scriptFor(allocator: std.mem.Allocator, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, opencode_config: ?[]const u8, extras: []const []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    switch (kind) {
        .claude => {
            try out.print(allocator,
                \\export ANTHROPIC_BASE_URL='{s}'
                \\export ANTHROPIC_API_KEY=
                \\export ANTHROPIC_AUTH_TOKEN=mlx-serve
                \\export CLAUDE_CODE_ATTRIBUTION_HEADER=0
                \\export ANTHROPIC_DEFAULT_OPUS_MODEL={s}
                \\export ANTHROPIC_DEFAULT_SONNET_MODEL={s}
                \\export ANTHROPIC_DEFAULT_HAIKU_MODEL={s}
                \\export CLAUDE_CODE_SUBAGENT_MODEL={s}
                \\export CLAUDE_CODE_MAX_OUTPUT_TOKENS={d}
                \\
            , .{ base_url, model, model, model, model, budget.output });
            // Claude Code assumes 200k for a model outside its catalog; declare the advertised context verbatim.
            if (budget.context > 0) {
                try out.print(allocator, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS={d}\n", .{budget.context});
            }
            try out.print(allocator, "claude --model {s}", .{model});
        },
        .pi => {
            try out.print(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi"
                \\pi --provider mlx --model {s}
            , .{model});
        },
        .omp => {
            // omp still reads pi's env spelling (measured on v17 — the OMP_
            // rename reached only its help text); export both.
            try out.print(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                \\export OMP_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                \\omp --model mlx/{s}
            , .{model});
        },
        .opencode => {
            try out.print(allocator,
                \\export OPENCODE_CONFIG_CONTENT='{s}'
                \\opencode --model mlx/{s}
            , .{ opencode_config.?, model });
        },
        .opencode2 => {
            try out.print(allocator,
                \\export OPENCODE_CONFIG_CONTENT='{s}'
                \\export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2"
                \\if ! command -v opencode2 >/dev/null 2>&1; then echo "opencode2 is not installed: npm install -g @opencode/cli"; exit 127; fi
                \\opencode2 --model mlx/{s}
            , .{ opencode_config.?, model });
        },
        .codex => {
            // No CODEX_HOME export: the user's own home (MCP servers, plugins,
            // auth) is used as-is, and the mlx-serve provider rides the
            // generated `--profile mlx-serve` layer. PATH first, then the CLI
            // the desktop app bundles (codex's rebranded app installs as
            // ChatGPT.app or Codex.app, bundle id com.openai.codex, CLI at
            // Contents/Resources/codex) — a desktop-app-only user has no codex
            // on PATH. Mirrors the Swift AgentConfigs.codexBinResolver.
            try out.appendSlice(allocator,
                \\CODEX_BIN="$(command -v codex)"
                \\if [ -z "$CODEX_BIN" ]; then
                \\  for app in "/Applications/ChatGPT.app" "/Applications/Codex.app" "$HOME/Applications/ChatGPT.app" "$HOME/Applications/Codex.app"; do
                \\    if [ -x "$app/Contents/Resources/codex" ]; then CODEX_BIN="$app/Contents/Resources/codex"; break; fi
                \\  done
                \\fi
                \\if [ -z "$CODEX_BIN" ]; then echo "codex is not installed: npm install -g @openai/codex, or install the ChatGPT app"; exit 127; fi
                \\"$CODEX_BIN" --profile mlx-serve
            );
        },
        .hermes => {
            try out.appendSlice(allocator,
                \\export HERMES_HOME="$HOME/.mlx-serve/hermes"
                \\hermes
            );
        },
        .aider => {
            try out.print(allocator,
                \\export OPENAI_API_BASE='{s}/v1'
                \\export OPENAI_API_KEY=mlx-serve
                \\aider --model openai/{s} --weak-model openai/{s} --model-metadata-file ~/.mlx-serve/aider/model-metadata.json
            , .{ base_url, model, model });
        },
    }
    try appendExtras(&out, allocator, extras);
    try out.append(allocator, '\n');
    return out.toOwnedSlice(allocator);
}

// ── Server discovery / model pick ───────────────────────────────────────

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

fn curlGet(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
    // Plain fetch, no HF token header — this talks to OUR server, never HF.
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "curl", "-fsS", "-m", "5", url },
        .stdout_limit = .limited(16 * 1024 * 1024),
    }) catch return error.FetchFailed;
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.FetchFailed,
        else => return error.FetchFailed,
    }
    return result.stdout;
}

fn serverUp(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8) bool {
    const url = std.fmt.allocPrint(allocator, "{s}/health", .{base_url}) catch return false;
    defer allocator.free(url);
    const body = curlGet(allocator, io, url) catch return false;
    allocator.free(body);
    return true;
}

/// `open -g -a "MLX Core"` — nonzero exit = the app isn't installed, which is
/// the detection: no probing of /Applications by hand.
/// HTTP status of `GET <base_url>/metrics.json`, or null when curl could not
/// reach the server at all.
fn metricsStatus(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8) ?u16 {
    const url = std.fmt.allocPrint(allocator, "{s}/metrics.json", .{base_url}) catch return null;
    defer allocator.free(url);
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "5", url },
        .stdout_limit = .limited(64),
    }) catch return null;
    defer allocator.free(result.stderr);
    defer allocator.free(result.stdout);
    return std.fmt.parseInt(u16, std.mem.trim(u8, result.stdout, " \r\n"), 10) catch null;
}

fn tryStartApp(allocator: std.mem.Allocator, io: std.Io) bool {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "open", "-g", "-a", "MLX Core" },
    }) catch return false;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

const Models = struct {
    arena: std.heap.ArenaAllocator,
    entries: []Entry,

    fn deinit(self: *Models) void {
        self.arena.deinit();
    }
};

/// Parse /v1/models into the chat-capable entries (media/embedding models
/// never enter a coding agent's picker — same rule as the app's
/// AgentModelEntry.chatEntries). Context comes from meta.context_length,
/// falling back to the top-level twin.
fn fetchChatEntries(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8) !Models {
    const url = try std.fmt.allocPrint(allocator, "{s}/v1/models", .{base_url});
    defer allocator.free(url);
    const body = try curlGet(allocator, io, url);
    defer allocator.free(body);

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return error.BadModelsJson;
    const data = switch (parsed) {
        .object => |o| o.get("data") orelse return error.BadModelsJson,
        else => return error.BadModelsJson,
    };
    if (data != .array) return error.BadModelsJson;

    var list = std.ArrayList(Entry).empty;
    for (data.array.items) |row| {
        if (row != .object) continue;
        const obj = row.object;
        const id_val = obj.get("id") orelse continue;
        if (id_val != .string or id_val.string.len == 0) continue;

        // Chat-capable only; a row with no capabilities key is an old build
        // that serves chat.
        var chat = true;
        var vision = false;
        if (obj.get("capabilities")) |caps| {
            if (caps == .array) {
                chat = caps.array.items.len == 0;
                for (caps.array.items) |c| {
                    if (c != .string) continue;
                    if (std.mem.eql(u8, c.string, "chat")) chat = true;
                    if (std.mem.eql(u8, c.string, "vision")) vision = true;
                    if (std.mem.eql(u8, c.string, "embeddings")) chat = false;
                }
            }
        }
        if (!chat) continue;

        var ctx: u64 = 0;
        if (obj.get("meta")) |meta| {
            if (meta == .object) {
                if (meta.object.get("context_length")) |v| {
                    if (v == .integer and v.integer > 0) ctx = @intCast(v.integer);
                }
            }
        }
        if (ctx == 0) {
            if (obj.get("context_length")) |v| {
                if (v == .integer and v.integer > 0) ctx = @intCast(v.integer);
            }
        }
        var loaded = false;
        if (obj.get("loaded")) |v| loaded = v == .bool and v.bool;

        try list.append(a, .{
            .id = try a.dupe(u8, id_val.string),
            .budget = budgetForContext(ctx),
            .vision = vision,
            .loaded = loaded,
        });
    }
    return .{ .arena = arena, .entries = try list.toOwnedSlice(a) };
}

// ── Config writes ───────────────────────────────────────────────────────

fn writeAgentFile(allocator: std.mem.Allocator, io: std.Io, subdir: []const u8, name: []const u8, content: []const u8) !void {
    const dir_path = try std.fmt.allocPrint(allocator, "{s}/.mlx-serve/{s}", .{ homeDir(), subdir });
    defer allocator.free(dir_path);
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = name, .data = content });
}

/// codex gets NO dedicated home: the profile lands as `mlx-serve.config.toml`
/// (the file `--profile mlx-serve` layers over the user's config.toml) inside
/// the effective CODEX_HOME, which the launch inherits unchanged. The
/// generated keys are rewritten on every launch; codex's own `[projects.*]`
/// trust tables in the existing file are carried over. The user's
/// config.toml is never written.
fn writeCodexProfile(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8, model: []const u8, budget: Budget) !void {
    const home = codexHome();
    try std.Io.Dir.cwd().createDirPath(io, home);
    var dir = try std.Io.Dir.openDirAbsolute(io, home, .{});
    defer dir.close(io);
    const existing = dir.readFileAlloc(io, "mlx-serve.config.toml", allocator, .limited(1 << 20)) catch
        try allocator.dupe(u8, "");
    defer allocator.free(existing);
    const toml = try codexConfigToml(allocator, base_url, model, budget, existing);
    defer allocator.free(toml);
    // Atomic replace: a half-written profile is a parse error for codex.
    var file = try dir.createFileAtomic(io, "mlx-serve.config.toml", .{ .replace = true });
    defer file.deinit(io);
    try file.file.writeStreamingAll(io, toml);
    try file.replace(io);
}

fn userOpencodeCliPath(allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("XDG_CONFIG_HOME")) |xdg| {
        const dir = std.mem.span(xdg);
        if (dir.len > 0) return std.fmt.allocPrint(allocator, "{s}/opencode/cli.json", .{dir});
    }
    return std.fmt.allocPrint(allocator, "{s}/.config/opencode/cli.json", .{homeDir()});
}

/// Write the agent's config files (the app's prepareConfig twin). opencode
/// carries its config inline and writes nothing.
fn writeConfigs(allocator: std.mem.Allocator, io: std.Io, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, entries: []const Entry) !void {
    switch (kind) {
        .claude, .opencode => {},
        .opencode2 => {
            const user_path = try userOpencodeCliPath(allocator);
            defer allocator.free(user_path);
            const existing = std.Io.Dir.cwd().readFileAlloc(io, user_path, allocator, .limited(1 << 20)) catch
                try allocator.dupe(u8, "{}");
            defer allocator.free(existing);
            const json = try mergeOpencode2CliJson(allocator, existing, base_url, null);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "opencode2/opencode", "cli.json", json);
            inline for (opencode2_plugin.files) |f| {
                try writeAgentFile(allocator, io, "opencode2/opencode/plugins/mlx-serve", f.name, f.bytes);
            }
        },
        .pi => {
            const json = try piModelsJson(allocator, base_url, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "pi", "models.json", json);
        },
        .omp => {
            const yml = try ompModelsYml(allocator, base_url, entries);
            defer allocator.free(yml);
            try writeAgentFile(allocator, io, "omp", "models.yml", yml);
        },
        .codex => try writeCodexProfile(allocator, io, base_url, model, budget),
        .hermes => {
            const yaml = try hermesConfigYaml(allocator, base_url, model, entries);
            defer allocator.free(yaml);
            try writeAgentFile(allocator, io, "hermes", "config.yaml", yaml);
            const env = try hermesEnvFile(allocator, base_url);
            defer allocator.free(env);
            try writeAgentFile(allocator, io, "hermes", ".env", env);
        },
        .aider => {
            const json = try aiderMetadataJson(allocator, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "aider", "model-metadata.json", json);
        },
    }
}

// ── Command entry ───────────────────────────────────────────────────────

const LaunchArgs = struct {
    kind: AgentKind,
    model: ?[]const u8 = null,
    url: ?[]const u8 = null,
    port: u16 = 11234,
    print_only: bool = false,
    no_start: bool = false,
    extras: []const []const u8 = &.{},
};

fn parseLaunchArgs(args: []const []const u8) !LaunchArgs {
    if (args.len == 0) return error.Usage;
    const kind = AgentKind.fromName(args[0]) orelse return error.UnknownAgent;
    var out = LaunchArgs{ .kind = kind };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--")) {
            out.extras = args[i + 1 ..];
            break;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.model = args[i];
        } else if (std.mem.eql(u8, arg, "--url")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.url = std.mem.trimEnd(u8, args[i], "/");
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.port = std.fmt.parseInt(u16, args[i], 10) catch return error.Usage;
        } else if (std.mem.eql(u8, arg, "--print")) {
            out.print_only = true;
        } else if (std.mem.eql(u8, arg, "--no-start")) {
            out.no_start = true;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            return error.Usage;
        } else {
            return error.Usage;
        }
    }
    return out;
}

fn printLaunchUsage() void {
    log.err(
        \\usage: mlx-serve launch <agent> [options] [-- <extra agent args>]
        \\
        \\agents: {s}
        \\
        \\options:
        \\  --model <id>   Serve this model (default: the server's default model)
        \\  --url <base>   Server base URL (default: http://127.0.0.1:<port>)
        \\  --port <n>     Server port for the default URL (default: 11234)
        \\  --print        Write the config files and print the launch script
        \\                 instead of running the agent
        \\  --no-start     Never auto-start the MLX Core app when the server is down
        \\
        \\Anything after `--` is passed to the agent, e.g.:
        \\  mlx-serve launch codex -- resume
        \\
    , .{AgentKind.names});
}

pub fn cmdLaunch(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const parsed = parseLaunchArgs(args) catch |err| {
        switch (err) {
            error.UnknownAgent => log.err("unknown agent '{s}' — supported: {s}\n", .{ args[0], AgentKind.names }),
            else => {},
        }
        printLaunchUsage();
        std.process.exit(1);
    };

    var url_buf: [64]u8 = undefined;
    const base_url = parsed.url orelse std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}", .{parsed.port}) catch unreachable;

    if (!serverUp(allocator, io, base_url)) {
        if (parsed.no_start or !tryStartApp(allocator, io)) {
            log.err("no mlx-serve server at {s}.\n", .{base_url});
            log.err("start one first:  mlx-serve serve   (or: mlx-serve run <model>)\n", .{});
            log.err("or install the MLX Core app: https://github.com/ddalcu/mlx-serve/releases\n", .{});
            std.process.exit(1);
        }
        log.info("starting the MLX Core app…\n", .{});
        var waited: usize = 0;
        while (!serverUp(allocator, io, base_url)) : (waited += 1) {
            if (waited >= 60) {
                log.err("the app started but its server never came up at {s} —\n", .{base_url});
                log.err("pick a model in the app (or check its port), then rerun.\n", .{});
                std.process.exit(1);
            }
            std.Io.sleep(io, .fromMilliseconds(1000), .real) catch {};
        }
    }

    // The server may still be scanning/loading right after boot — poll until
    // a chat-capable model shows up (a stub is fine: the first request
    // hot-loads it).
    var models: Models = undefined;
    var polls: usize = 0;
    while (true) : (polls += 1) {
        models = fetchChatEntries(allocator, io, base_url) catch |err| {
            log.err("could not read {s}/v1/models: {s}\n", .{ base_url, @errorName(err) });
            std.process.exit(1);
        };
        if (models.entries.len > 0) break;
        models.deinit();
        if (polls >= 30) {
            log.err("no chat-capable model on {s} — pull one first (mlx-serve pull <model>)\n", .{base_url});
            std.process.exit(1);
        }
        std.Io.sleep(io, .fromMilliseconds(1000), .real) catch {};
    }
    defer models.deinit();

    // Pick: --model must exist on the server; default = first loaded chat
    // row (/v1/models sorts the default first), else the first chat row.
    var pick: ?Entry = null;
    if (parsed.model) |want| {
        for (models.entries) |e| {
            if (std.mem.eql(u8, e.id, want)) pick = e;
        }
        if (pick == null) {
            log.err("model '{s}' is not on {s} — available:\n", .{ want, base_url });
            for (models.entries) |e| log.err("  {s}\n", .{e.id});
            std.process.exit(1);
        }
    } else {
        for (models.entries) |e| {
            if (e.loaded) {
                pick = e;
                break;
            }
        }
        if (pick == null) pick = models.entries[0];
    }
    const chosen = pick.?;

    writeConfigs(allocator, io, parsed.kind, base_url, chosen.id, chosen.budget, models.entries) catch |err| {
        log.err("could not write the {s} config: {s}\n", .{ @tagName(parsed.kind), @errorName(err) });
        std.process.exit(1);
    };

    const oc_config: ?[]u8 = if (parsed.kind == .opencode or parsed.kind == .opencode2)
        try opencodeJson(allocator, base_url, models.entries)
    else
        null;
    defer if (oc_config) |c| allocator.free(c);

    const script = try scriptFor(allocator, parsed.kind, base_url, chosen.id, chosen.budget, oc_config, parsed.extras);
    defer allocator.free(script);

    if (parsed.kind == .opencode2) {
        // The plugin's whole data source is /metrics.json; say so now rather
        // than leave an empty panel to explain itself.
        if (metricsStatus(allocator, io, base_url)) |status| {
            if (metricsFeedNote(status)) |note| log.warn("{s}\n", .{note});
        }
    }

    if (parsed.print_only) {
        var stdout_buf: [8192]u8 = undefined;
        var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
        stdout_w.interface.writeAll(script) catch {};
        stdout_w.interface.flush() catch {};
        return;
    }

    log.info("launching {s} with {s} ({d}K context) via {s}\n", .{
        @tagName(parsed.kind), chosen.id, chosen.budget.context / 1024, base_url,
    });
    var child = std.process.spawn(io, .{
        .argv = &.{ "/bin/zsh", "-l", "-c", script },
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch {
        log.err("could not start /bin/zsh\n", .{});
        std.process.exit(1);
    };
    const term = child.wait(io) catch std.process.exit(1);
    switch (term) {
        .exited => |code| std.process.exit(code),
        else => std.process.exit(1),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

const t = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn setEnv(name: [*:0]const u8, value: [*:0]const u8) void {
    _ = setenv(name, value, 1);
}

fn restoreEnv(name: [*:0]const u8, value: ?[*:0]const u8) void {
    if (value) |v| setEnv(name, v) else _ = unsetenv(name);
}

test "budgetForContext mirrors AgentBudget: ctx/4 clamped to [1024, 65536], 0 = fallback" {
    try t.expectEqual(FALLBACK_BUDGET, budgetForContext(0));
    try t.expectEqual(Budget{ .context = 4096, .output = 1024 }, budgetForContext(4096));
    try t.expectEqual(Budget{ .context = 2048, .output = 1024 }, budgetForContext(2048));
    try t.expectEqual(Budget{ .context = 90112, .output = 22528 }, budgetForContext(90112));
    try t.expectEqual(Budget{ .context = 1048576, .output = 65536 }, budgetForContext(1048576));
}

test "omp models.yml: static per-model entries, no discovery, pi-compat vocabulary" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
        .{ .id = "m2", .budget = .{ .context = 262144, .output = 65536 }, .vision = true, .loaded = false },
    };
    const yml = try ompModelsYml(t.allocator, "http://127.0.0.1:11234", &entries);
    defer t.allocator.free(yml);
    try t.expect(std.mem.indexOf(u8, yml, "discovery") == null);
    try t.expect(std.mem.indexOf(u8, yml, "baseUrl: http://127.0.0.1:11234/v1") != null);
    try t.expect(std.mem.indexOf(u8, yml, "contextWindow: 4096") != null);
    try t.expect(std.mem.indexOf(u8, yml, "contextWindow: 262144") != null);
    try t.expect(std.mem.indexOf(u8, yml, "input: [text, image]") != null);
    try t.expect(std.mem.indexOf(u8, yml, "thinkingFormat: qwen") != null);
}

test "codex config: responses wire API, keyless, context at the root" {
    const toml = try codexConfigToml(t.allocator, "http://127.0.0.1:11234", "m1", .{ .context = 90112, .output = 22528 }, "");
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "wire_api = \"responses\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "model_context_window = 90112") != null);
    try t.expect(std.mem.indexOf(u8, toml, "base_url = \"http://127.0.0.1:11234/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "env_key") == null);
    try t.expect(std.mem.startsWith(u8, toml, "# Generated by mlx-serve. Regenerated on each launch (your [projects.*]\n# trust entries survive; put personal settings in config.toml).\n"));
}

test "codex config: codex-owned [projects.*] tables survive the rewrite" {
    const existing =
        \\# Generated by mlx-serve. Regenerated on each launch (your [projects.*]
        \\# trust entries survive; put personal settings in config.toml).
        \\model = "old"
        \\model_provider = "mlx"
        \\model_context_window = 4096
        \\
        \\[model_providers.mlx]
        \\name = "MLX Serve (local)"
        \\base_url = "http://x:1/v1"
        \\wire_api = "responses"
        \\
        \\[projects."/work/one"]
        \\trust_level = "trusted"
        \\
        \\[projects."/work/two"]
        \\trust_level = "untrusted"
        \\
    ;
    const toml = try codexConfigToml(t.allocator, "http://x:2", "m2", .{ .context = 8192, .output = 2048 }, existing);
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "[projects.\"/work/one\"]\ntrust_level = \"trusted\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "[projects.\"/work/two\"]\ntrust_level = \"untrusted\"") != null);
    // generated keys regenerate from the new launch, not from `existing`
    try t.expect(std.mem.indexOf(u8, toml, "model = \"m2\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "model = \"old\"") == null);
    try t.expect(std.mem.indexOf(u8, toml, "base_url = \"http://x:2/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "http://x:1") == null);
    // the trust tables ride exactly once each
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, toml, "[projects.\"/work/"));
}

test "codex config: foreign tables in the profile are not kept" {
    const existing =
        \\model = "old"
        \\
        \\[model_providers.mlx]
        \\base_url = "http://x:1/v1"
        \\
        \\[mcp_servers.thing]
        \\command = "x"
        \\
    ;
    const toml = try codexConfigToml(t.allocator, "http://x:2", "m2", .{ .context = 8192, .output = 2048 }, existing);
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "mcp_servers") == null);
}

test "codex home: CODEX_HOME wins, empty falls back to ~/.codex" {
    const old = std.c.getenv("CODEX_HOME");
    const old_home = std.c.getenv("HOME");
    defer restoreEnv("CODEX_HOME", old);
    defer restoreEnv("HOME", old_home);

    setEnv("CODEX_HOME", "/custom/codex");
    try t.expectEqualStrings("/custom/codex", codexHome());

    setEnv("CODEX_HOME", "");
    setEnv("HOME", "/h");
    try t.expectEqualStrings("/h/.codex", codexHome());

    _ = unsetenv("CODEX_HOME");
    try t.expectEqualStrings("/h/.codex", codexHome());
}

test "codex profile file lands in the effective home, base config untouched" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try dir.dir.realPath(std.testing.io, &root_buf);
    var home_buf: [std.fs.max_path_bytes:0]u8 = undefined;
    const home_z = try std.fmt.bufPrint(&home_buf, "{s}/cx", .{root_buf[0..root_len]});
    home_buf[home_z.len] = 0;
    const home: [*:0]const u8 = &home_buf;

    const old = std.c.getenv("CODEX_HOME");
    defer restoreEnv("CODEX_HOME", old);
    setEnv("CODEX_HOME", home);

    try writeCodexProfile(t.allocator, std.testing.io, "http://x:1", "m1", .{ .context = 4096, .output = 1024 });
    const written = try dir.dir.readFileAlloc(std.testing.io, "cx/mlx-serve.config.toml", t.allocator, .limited(1 << 20));
    defer t.allocator.free(written);
    const expected = try codexConfigToml(t.allocator, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, "");
    defer t.allocator.free(expected);
    try t.expectEqualStrings(expected, written);
    // Overwrite at the next launch works the same way.
    try writeCodexProfile(t.allocator, std.testing.io, "http://x:2", "m2", .{ .context = 8192, .output = 2048 });
    const rewritten = try dir.dir.readFileAlloc(std.testing.io, "cx/mlx-serve.config.toml", t.allocator, .limited(1 << 20));
    defer t.allocator.free(rewritten);
    try t.expect(std.mem.indexOf(u8, rewritten, "base_url = \"http://x:2/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, rewritten, "http://x:1") == null);
    // The base config is never written.
    try t.expect(dir.dir.access(std.testing.io, "cx/config.toml", .{}) == error.FileNotFound);
}

test "codex profile write carries codex's trust tables across relaunches" {
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try dir.dir.realPath(std.testing.io, &root_buf);
    var home_buf: [std.fs.max_path_bytes:0]u8 = undefined;
    const home_z = try std.fmt.bufPrint(&home_buf, "{s}/cx", .{root_buf[0..root_len]});
    home_buf[home_z.len] = 0;
    const home: [*:0]const u8 = &home_buf;

    const old = std.c.getenv("CODEX_HOME");
    defer restoreEnv("CODEX_HOME", old);
    setEnv("CODEX_HOME", home);

    try writeCodexProfile(t.allocator, std.testing.io, "http://x:1", "m1", .{ .context = 4096, .output = 1024 });
    // codex answers a trust prompt: it persists into the profile layer.
    const before = try dir.dir.readFileAlloc(std.testing.io, "cx/mlx-serve.config.toml", t.allocator, .limited(1 << 20));
    defer t.allocator.free(before);
    const trusted = try std.fmt.allocPrint(t.allocator, "{s}\n[projects.\"/work/a\"]\ntrust_level = \"trusted\"\n", .{before});
    defer t.allocator.free(trusted);
    {
        var d = try std.Io.Dir.openDirAbsolute(std.testing.io, home[0..std.mem.len(home)], .{});
        defer d.close(std.testing.io);
        try d.writeFile(std.testing.io, .{ .sub_path = "mlx-serve.config.toml", .data = trusted });
    }
    try writeCodexProfile(t.allocator, std.testing.io, "http://x:2", "m2", .{ .context = 8192, .output = 2048 });
    const after = try dir.dir.readFileAlloc(std.testing.io, "cx/mlx-serve.config.toml", t.allocator, .limited(1 << 20));
    defer t.allocator.free(after);
    try t.expect(std.mem.indexOf(u8, after, "[projects.\"/work/a\"]\ntrust_level = \"trusted\"") != null);
    try t.expect(std.mem.indexOf(u8, after, "base_url = \"http://x:2/v1\"") != null);
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, after, "[projects.\""));
}

test "pi models.json and opencode config parse as JSON and stay single-quote-free" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = true, .loaded = true },
        .{ .id = "m2", .budget = .{ .context = 8192, .output = 2048 }, .vision = false, .loaded = false },
    };
    inline for (.{ piModelsJson, opencodeJson }) |builder| {
        const json = try builder(t.allocator, "http://127.0.0.1:11234", &entries);
        defer t.allocator.free(json);
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
        defer parsed.deinit();
        // opencode's config rides single-quoted inside the launch script.
        try t.expect(std.mem.indexOf(u8, json, "'") == null);
    }
}

test "aider metadata: litellm keys per openai/<id> entry" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
    };
    const json = try aiderMetadataJson(t.allocator, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const row = parsed.value.object.get("openai/m1").?.object;
    try t.expectEqual(@as(i64, 4096), row.get("max_input_tokens").?.integer);
    try t.expectEqual(@as(i64, 1024), row.get("max_output_tokens").?.integer);
}

test "launch args: passthrough after --, unknown agent named, url trailing slash trimmed" {
    const parsed = try parseLaunchArgs(&.{ "codex", "--url", "http://x:1/", "--print", "--", "resume", "-a" });
    try t.expectEqual(AgentKind.codex, parsed.kind);
    try t.expectEqualStrings("http://x:1", parsed.url.?);
    try t.expect(parsed.print_only);
    try t.expectEqual(@as(usize, 2), parsed.extras.len);
    try t.expectEqualStrings("resume", parsed.extras[0]);
    try t.expectError(error.UnknownAgent, parseLaunchArgs(&.{"cursor"}));
    // The rebrand alias from issue #188's own wording.
    try t.expectEqual(AgentKind.codex, (try parseLaunchArgs(&.{"chatgpt"})).kind);
}

test "script assembly: extras are shell-quoted onto the invocation line" {
    const script = try scriptFor(t.allocator, .codex, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, null, &.{ "resume", "it's" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "\"$CODEX_BIN\" --profile mlx-serve 'resume' 'it'\\''s'") != null);
    try t.expect(std.mem.indexOf(u8, script, "CODEX_HOME") == null);
}

test "codex script falls back to the desktop app's bundled CLI (ChatGPT.app rebrand)" {
    const script = try scriptFor(t.allocator, .codex, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "/Applications/ChatGPT.app") != null);
    try t.expect(std.mem.indexOf(u8, script, "/Applications/Codex.app") != null);
    try t.expect(std.mem.indexOf(u8, script, "$HOME/Applications") != null);
    try t.expect(std.mem.indexOf(u8, script, "Contents/Resources/codex") != null);
    // Never exec an empty resolution — refuse with the install hint.
    try t.expect(std.mem.indexOf(u8, script, "exit 127") != null);
    try t.expect(std.mem.indexOf(u8, script, "\n\"$CODEX_BIN\"") != null);
}

test "codex script always requests the mlx-serve profile, never a dedicated home" {
    // The invocation carries the profile request itself — NOT a CODEX_PROFILE
    // environment variable, which a user's own export (or the app's env-less
    // GUI process) could silently change or lose.
    const script = try scriptFor(t.allocator, .codex, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "\"$CODEX_BIN\" --profile mlx-serve") != null);
    try t.expect(std.mem.indexOf(u8, script, "CODEX_PROFILE") == null);
    try t.expect(std.mem.indexOf(u8, script, "CODEX_HOME") == null);
}

test "AgentKind.fromName recognizes opencode2" {
    try t.expect(AgentKind.fromName("opencode2") != null);
    try t.expectEqualStrings("opencode2", @tagName(AgentKind.fromName("opencode2").?));
}

test "opencode2 cli.json merge keeps theme and unrelated plugins, one mlx-serve entry" {
    const existing =
        \\{"theme":"nord","keybinds":{"leader":"ctrl+x"},"plugins":[{"package":"other-plugin","options":{"a":1}}]}
    ;
    const json = try mergeOpencode2CliJson(t.allocator, existing, "http://127.0.0.1:11234", null);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try t.expectEqualStrings("nord", obj.get("theme").?.string);
    try t.expectEqualStrings("ctrl+x", obj.get("keybinds").?.object.get("leader").?.string);
    const plugins = obj.get("plugins").?.array.items;
    try t.expectEqual(@as(usize, 2), plugins.len);
    var saw_other = false;
    var saw_mlx: usize = 0;
    for (plugins) |p| {
        const pkg = p.object.get("package").?.string;
        if (std.mem.eql(u8, pkg, "other-plugin")) {
            saw_other = true;
            try t.expectEqual(@as(i64, 1), p.object.get("options").?.object.get("a").?.integer);
        } else if (std.mem.eql(u8, pkg, "./plugins/mlx-serve")) {
            saw_mlx += 1;
            const opts = p.object.get("options").?.object;
            try t.expectEqualStrings("http://127.0.0.1:11234/metrics.json", opts.get("metricsUrl").?.string);
            try t.expect(opts.get("metricsToken") == null);
        }
    }
    try t.expect(saw_other);
    try t.expectEqual(@as(usize, 1), saw_mlx);
}

test "opencode2 cli.json merge replaces a prior mlx-serve plugin, does not duplicate" {
    const existing =
        \\{"plugins":[{"package":"./plugins/mlx-serve","options":{"metricsUrl":"http://old:1/metrics.json","metricsToken":"stale"}},{"package":"keep-me"}]}
    ;
    const json = try mergeOpencode2CliJson(t.allocator, existing, "http://127.0.0.1:8097", null);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const plugins = parsed.value.object.get("plugins").?.array.items;
    try t.expectEqual(@as(usize, 2), plugins.len);
    var saw_mlx: usize = 0;
    var saw_keep = false;
    for (plugins) |p| {
        const pkg = p.object.get("package").?.string;
        if (std.mem.eql(u8, pkg, "./plugins/mlx-serve")) {
            saw_mlx += 1;
            try t.expectEqualStrings("http://127.0.0.1:8097/metrics.json", p.object.get("options").?.object.get("metricsUrl").?.string);
        } else if (std.mem.eql(u8, pkg, "keep-me")) {
            saw_keep = true;
        }
    }
    try t.expectEqual(@as(usize, 1), saw_mlx);
    try t.expect(saw_keep);
}

test "opencode2 cli.json merge omits metricsToken on loopback and writes it otherwise" {
    const loop = try mergeOpencode2CliJson(t.allocator, "{}", "http://127.0.0.1:11234", null);
    defer t.allocator.free(loop);
    const loop_p = try std.json.parseFromSlice(std.json.Value, t.allocator, loop, .{});
    defer loop_p.deinit();
    const loop_opts = loop_p.value.object.get("plugins").?.array.items[0].object.get("options").?.object;
    try t.expect(loop_opts.get("metricsToken") == null);

    const remote = try mergeOpencode2CliJson(t.allocator, "{}", "http://10.0.0.2:11234", null);
    defer t.allocator.free(remote);
    const remote_p = try std.json.parseFromSlice(std.json.Value, t.allocator, remote, .{});
    defer remote_p.deinit();
    const remote_opts = remote_p.value.object.get("plugins").?.array.items[0].object.get("options").?.object;
    try t.expectEqualStrings("mlx-serve", remote_opts.get("metricsToken").?.string);
    try t.expectEqualStrings("http://10.0.0.2:11234/metrics.json", remote_opts.get("metricsUrl").?.string);

    const known = try mergeOpencode2CliJson(t.allocator, "{}", "http://127.0.0.1:11234", "secret-key");
    defer t.allocator.free(known);
    const known_p = try std.json.parseFromSlice(std.json.Value, t.allocator, known, .{});
    defer known_p.deinit();
    const known_opts = known_p.value.object.get("plugins").?.array.items[0].object.get("options").?.object;
    try t.expectEqualStrings("secret-key", known_opts.get("metricsToken").?.string);
}

test "opencode2 feed note: 200 is silent, 503 names --metrics, 401 names the key" {
    try t.expect(metricsFeedNote(200) == null);
    try t.expect(std.mem.indexOf(u8, metricsFeedNote(503).?, "--metrics") != null);
    try t.expect(std.mem.indexOf(u8, metricsFeedNote(503).?, "footer turn meter still works") != null);
    try t.expect(std.mem.indexOf(u8, metricsFeedNote(401).?, "401 unauthorized") != null);
    try t.expect(std.mem.indexOf(u8, metricsFeedNote(500).?, "unreachable") != null);
}

test "opencode2 script exports XDG_CONFIG_HOME, OPENCODE_CONFIG_CONTENT, and invokes opencode2" {
    const cfg = "{\"provider\":{}}";
    const script = try scriptFor(t.allocator, .opencode2, "http://127.0.0.1:11234", "m1", .{ .context = 4096, .output = 1024 }, cfg, &.{ "resume", "it's" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export XDG_CONFIG_HOME=\"$HOME/.mlx-serve/opencode2\"") != null);
    try t.expect(std.mem.indexOf(u8, script, "export OPENCODE_CONFIG_CONTENT='{\"provider\":{}}'") != null);
    try t.expect(std.mem.indexOf(u8, script, "opencode2 --model mlx/m1") != null);
    try t.expect(std.mem.indexOf(u8, script, "npm install -g @opencode/cli") != null);
    try t.expect(std.mem.indexOf(u8, script, "exit 127") != null);
    try t.expect(std.mem.indexOf(u8, script, "'resume' 'it'\\''s'") != null);
}

test "claude script declares the advertised context window (CLAUDE_CODE_MAX_CONTEXT_TOKENS)" {
    // Claude Code assumes 200k outside its catalog; CLAUDE_CODE_MAX_CONTEXT_TOKENS is the override.
    const script = try scriptFor(t.allocator, .claude, "http://x:1", "m1", budgetForContext(786432), null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=786432") != null);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536") != null);
    try t.expect(std.mem.indexOf(u8, script, "\nclaude --model m1") != null);

    // An unknown context is not a claim: omit the export rather than pin a
    // number the server never advertised.
    const unknown = try scriptFor(t.allocator, .claude, "http://x:1", "m1", .{ .context = 0, .output = 8192 }, null, &.{});
    defer t.allocator.free(unknown);
    try t.expect(std.mem.indexOf(u8, unknown, "CLAUDE_CODE_MAX_CONTEXT_TOKENS") == null);
    try t.expect(std.mem.indexOf(u8, unknown, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192") != null);
}
