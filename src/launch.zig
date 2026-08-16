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
    codex,
    hermes,
    aider,

    pub fn fromName(name: []const u8) ?AgentKind {
        // The codex rebrand: issue #188 asks for `mlx-serve launch chatgpt`.
        if (std.mem.eql(u8, name, "chatgpt")) return .codex;
        inline for (@typeInfo(AgentKind).@"enum".field_names, 0..) |f, i| {
            if (std.mem.eql(u8, name, f)) return @enumFromInt(i);
        }
        return null;
    }

    pub const names = "claude, pi, omp, opencode, codex, hermes, aider";
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

/// codex `config.toml` — Responses wire API only (codex-rs `WireApi` has one
/// variant), pointing at our /v1/responses. Keyless: no `env_key` and
/// `requires_openai_auth` unset means codex skips login; the loopback server
/// ignores keys anyway.
pub fn codexConfigToml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, budget: Budget) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\# written by mlx-serve — dedicated CODEX_HOME, regenerated at each launch.
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
                \\claude --model {s}
            , .{ base_url, model, model, model, model, budget.output, model });
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
        .codex => {
            // PATH first, then the CLI the desktop app bundles (codex's
            // rebranded app installs as ChatGPT.app or Codex.app, bundle id
            // com.openai.codex, CLI at Contents/Resources/codex) — a
            // desktop-app-only user has no codex on PATH. Mirrors the Swift
            // AgentConfigs.codexBinResolver.
            try out.appendSlice(allocator,
                \\export CODEX_HOME="$HOME/.mlx-serve/codex"
                \\CODEX_BIN="$(command -v codex)"
                \\if [ -z "$CODEX_BIN" ]; then
                \\  for app in "/Applications/ChatGPT.app" "/Applications/Codex.app" "$HOME/Applications/ChatGPT.app" "$HOME/Applications/Codex.app"; do
                \\    if [ -x "$app/Contents/Resources/codex" ]; then CODEX_BIN="$app/Contents/Resources/codex"; break; fi
                \\  done
                \\fi
                \\if [ -z "$CODEX_BIN" ]; then echo "codex is not installed: npm install -g @openai/codex, or install the ChatGPT app"; exit 127; fi
                \\"$CODEX_BIN"
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

/// Write the agent's config files (the app's prepareConfig twin). opencode
/// carries its config inline and writes nothing.
fn writeConfigs(allocator: std.mem.Allocator, io: std.Io, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, entries: []const Entry) !void {
    switch (kind) {
        .claude, .opencode => {},
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
        .codex => {
            const toml = try codexConfigToml(allocator, base_url, model, budget);
            defer allocator.free(toml);
            try writeAgentFile(allocator, io, "codex", "config.toml", toml);
        },
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

    const oc_config: ?[]u8 = if (parsed.kind == .opencode)
        try opencodeJson(allocator, base_url, models.entries)
    else
        null;
    defer if (oc_config) |c| allocator.free(c);

    const script = try scriptFor(allocator, parsed.kind, base_url, chosen.id, chosen.budget, oc_config, parsed.extras);
    defer allocator.free(script);

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
    const toml = try codexConfigToml(t.allocator, "http://127.0.0.1:11234", "m1", .{ .context = 90112, .output = 22528 });
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "wire_api = \"responses\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "model_context_window = 90112") != null);
    try t.expect(std.mem.indexOf(u8, toml, "base_url = \"http://127.0.0.1:11234/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "env_key") == null);
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
    try t.expect(std.mem.indexOf(u8, script, "\"$CODEX_BIN\" 'resume' 'it'\\''s'") != null);
    try t.expect(std.mem.indexOf(u8, script, "export CODEX_HOME=\"$HOME/.mlx-serve/codex\"") != null);
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
