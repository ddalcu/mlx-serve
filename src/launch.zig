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
//! (AgentBudget's formula: output = clamp(ctx/2, 1024, 65536) — never a
//! hardcoded window), write the agent's config, and exec it through a login
//! zsh so the user's PATH (nvm, Homebrew, ~/.local/bin) resolves.

const std = @import("std");
const log = @import("log.zig");
const opencode2_plugin = @import("opencode2_plugin");
const agent_skills = @import("agent_skills");

pub const Budget = struct { context: u64, output: u64 };

/// Mirrors Swift `AgentBudget.fallback` — used when the server advertises no
/// context (older build, unloaded stub with no readable config).
pub const FALLBACK_BUDGET = Budget{ .context = 32768, .output = 8192 };

/// Mirrors Swift `AgentBudget.forServerContext`.
pub fn budgetForContext(ctx: u64) Budget {
    if (ctx == 0) return FALLBACK_BUDGET;
    return .{ .context = ctx, .output = @min(65536, @max(1024, ctx / 2)) };
}

/// Room an agent keeps free before compacting, and what it keeps after: a
/// quarter of the window, capped where pi's and opencode2's own 20000-token
/// defaults (sized for 200k windows) take over.
pub fn compactionReserve(ctx: u64) u64 {
    return @min(20000, @max(1024, ctx / 4));
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
    fx,
    grok,
    zcode,

    pub fn fromName(name: []const u8) ?AgentKind {
        // The codex rebrand: issue #188 asks for `mlx-serve launch chatgpt`.
        if (std.mem.eql(u8, name, "chatgpt")) return .codex;
        inline for (@typeInfo(AgentKind).@"enum".field_names, 0..) |f, i| {
            if (std.mem.eql(u8, name, f)) return @fromBackingInt(@intCast(i));
        }
        return null;
    }

    pub const names = "claude, pi, omp, opencode, opencode2, codex, hermes, aider, fx, grok, zcode";
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
        \\        "maxTokensField": "max_tokens"
        \\      }},
        \\      "models": [
    , .{base_url});
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}
            \\        {{"id": "{s}", "name": "{s} (mlx-serve)", "input": [{s}],
            \\         "contextWindow": {d}, "maxTokens": {d}, "reasoning": true,
            \\         "thinkingLevelMap": {{"off": "none", "xhigh": "xhigh", "max": "max"}}}}
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

/// opencode sends `reasoning_effort` only when the model declares it; without
/// it every turn ran thinking-off. Variants are its in-session effort picker.
const opencode_reasoning =
    \\"options": {"reasoningEffort": "medium"}, "variants": {"none": {"reasoningEffort": "none"}, "low": {"reasoningEffort": "low"}, "medium": {"reasoningEffort": "medium"}, "high": {"reasoningEffort": "high"}}
;

/// opencode config — carried inline via OPENCODE_CONFIG_CONTENT (merges over
/// the user's own config, no file writes). Single-quoted in the script, so
/// the JSON must stay single-quote-free.
/// `pin_model` writes a top-level `"model"` — opencode 2's TUI has no
/// `--model` flag, so the config is the only place to select one.
/// `limit.output` is the room opencode keeps free before compacting (it
/// never sends max_tokens), so it carries the reserve, not the response cap.
/// `compaction` (opencode2) scales its global buffer/keep to the pinned
/// model's window: the defaults compact a 24k window before its first reply.
pub fn opencodeJson(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry, pin_model: ?[]const u8, compaction: bool) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"$schema\": \"https://opencode.ai/config.json\", ");
    if (pin_model) |m| try out.print(allocator, "\"model\": \"mlx/{s}\", ", .{m});
    if (compaction) {
        var ctx: u64 = FALLBACK_BUDGET.context;
        for (entries) |e| {
            if (pin_model == null or std.mem.eql(u8, e.id, pin_model.?)) {
                ctx = e.budget.context;
                break;
            }
        }
        const reserve = compactionReserve(ctx);
        try out.print(allocator, "\"compaction\": {{\"buffer\": {d}, \"keep\": {{\"tokens\": {d}}}}}, ", .{ reserve, @min(15000, reserve) });
    }
    try out.appendSlice(allocator, "\"skills\": {\"paths\": [\"~/.mlx-serve/" ++ skill_dir ++ "\"]}, ");
    try out.print(allocator,
        \\"provider": {{"mlx": {{"npm": "@ai-sdk/openai-compatible", "name": "MLX Serve (local)", "options": {{"baseURL": "{s}/v1"}}, "models": {{
    , .{base_url});
    for (entries, 0..) |e, i| {
        try out.print(allocator, "{s}\"{s}\": {{\"name\": \"{s} (mlx-serve)\",{s} \"limit\": {{\"context\": {d}, \"output\": {d}}}, {s}}}", .{
            if (i == 0) "" else ", ",
            e.id,
            e.id,
            if (e.vision) " \"attachment\": true," else "",
            e.budget.context,
            compactionReserve(e.budget.context),
            opencode_reasoning,
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

/// pi `settings.json`: compaction numbers scaled to the window, everything
/// else (theme, packages, the user's own `enabled`) kept. pi compacts when
/// context exceeds window - reserveTokens and keeps keepRecentTokens; its
/// defaults (16384 / 20000) never compact a 24k window while max_tokens
/// shrinks to 1.
pub fn mergePiSettingsJson(allocator: std.mem.Allocator, existing: []const u8, ctx: u64) ![]u8 {
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

    var compaction: std.json.ObjectMap = .empty;
    if (parsed.value.object.get("compaction")) |c| {
        if (c == .object) compaction = c.object;
    }
    const reserve = compactionReserve(ctx);
    try compaction.put(a, "reserveTokens", .{ .integer = @intCast(@min(16384, reserve + 4096)) });
    try compaction.put(a, "keepRecentTokens", .{ .integer = @intCast(reserve) });

    var obj = parsed.value.object;
    try obj.put(a, "compaction", .{ .object = compaction });
    parsed.value = .{ .object = obj };
    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

/// fx keeps custom providers only in `~/.fx/settings.json` (no config-dir
/// override), so the launcher owns ONE key there, `providers.mlx-serve`, and
/// selects it per launch with FX_PROVIDER/FX_MODEL: the user's default
/// provider and every other setting stay as found. fx rejects unknown keys.
pub const fx_provider = "mlx-serve";

/// The user's fx `settings.json` with our provider set. A file that is not a
/// JSON object is an error, never replaced.
pub fn mergeFxSettingsJson(allocator: std.mem.Allocator, existing: []const u8, base_url: []const u8, entries: []const Entry) ![]u8 {
    const trimmed = std.mem.trim(u8, existing, " \t\r\n");
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, if (trimmed.len == 0) "{}" else existing, .{}) catch
        return error.UnreadableFxSettings;
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnreadableFxSettings;
    const a = parsed.arena.allocator();

    var metadata: std.json.ObjectMap = .empty;
    for (entries) |e| {
        var m: std.json.ObjectMap = .empty;
        try m.put(a, "context_window", .{ .integer = @intCast(e.budget.context) });
        try m.put(a, "max_output_tokens", .{ .integer = @intCast(e.budget.output) });
        try m.put(a, "supports_tool_use", .{ .bool = true });
        try m.put(a, "supports_vision", .{ .bool = e.vision });
        try metadata.put(a, e.id, .{ .object = m });
    }
    var auth: std.json.ObjectMap = .empty;
    try auth.put(a, "type", .{ .string = "none" });
    var provider: std.json.ObjectMap = .empty;
    try provider.put(a, "protocol", .{ .string = "openai-chat-completions" });
    try provider.put(a, "base_url", .{ .string = try std.fmt.allocPrint(a, "{s}/v1", .{base_url}) });
    try provider.put(a, "auth", .{ .object = auth });
    try provider.put(a, "model_metadata", .{ .object = metadata });

    var providers: std.json.ObjectMap = .empty;
    if (parsed.value.object.get("providers")) |p| {
        if (p == .object) providers = p.object;
    }
    try providers.put(a, fx_provider, .{ .object = provider });
    try parsed.value.object.put(a, "providers", .{ .object = providers });
    return std.json.Stringify.valueAlloc(allocator, parsed.value, .{ .whitespace = .indent_2 });
}

/// One `-c key="value"` arg: codex parses the value as a TOML string, then
/// the whole arg is shell-quoted.
fn appendCodexOverride(out: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
    var arg = std.ArrayList(u8).empty;
    defer arg.deinit(allocator);
    try arg.appendSlice(allocator, key);
    try arg.appendSlice(allocator, "=\"");
    for (value) |c| {
        if (c == '"' or c == '\\') try arg.append(allocator, '\\');
        try arg.append(allocator, c);
    }
    try arg.append(allocator, '"');
    if (out.items.len != 0) try out.append(allocator, ' ');
    try out.appendSlice(allocator, "-c ");
    try appendQuoted(out, allocator, arg.items);
}

/// codex launch-line overrides, merged over the user's own config.toml so
/// nothing is written into their Codex home. Responses wire API only; keyless
/// (no `env_key`). A zero advertised context leaves codex's own default.
pub fn codexConfigOverrides(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, budget: Budget) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const url = try std.fmt.allocPrint(allocator, "{s}/v1", .{base_url});
    defer allocator.free(url);
    try appendCodexOverride(&out, allocator, "model", model);
    try appendCodexOverride(&out, allocator, "model_provider", "mlx");
    if (budget.context > 0) try out.print(allocator, " -c model_context_window={d}", .{budget.context});
    try appendCodexOverride(&out, allocator, "model_providers.mlx.name", "MLX Serve (local)");
    try appendCodexOverride(&out, allocator, "model_providers.mlx.base_url", url);
    try appendCodexOverride(&out, allocator, "model_providers.mlx.wire_api", "responses");
    return out.toOwnedSlice(allocator);
}

/// grok `config.toml` under a dedicated GROK_HOME. A dummy XAI_API_KEY fails
/// grok's key probe against xAI, so the credential is each model's `api_key`.
/// Helper calls (titles, image descriptions, suggestions) default to xAI model
/// ids, which would reach our server and load its default model: they are
/// pinned to the launched one.
pub fn grokConfigToml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\# written by mlx-serve — dedicated GROK_HOME, regenerated at each launch.
        \\[models]
        \\default = "{s}"
        \\session_summary = "{s}"
        \\image_description = "{s}"
        \\prompt_suggestion = "{s}"
        \\
    , .{ model, model, model, model });
    for (entries) |e| {
        try out.print(allocator,
            \\
            \\[model."{s}"]
            \\model = "{s}"
            \\base_url = "{s}/v1"
            \\name = "{s} (mlx-serve)"
            \\api_key = "mlx-serve"
            \\context_window = {d}
            \\max_completion_tokens = {d}
            \\supports_reasoning_effort = true
            \\reasoning_efforts = ["none", "low", "medium", "high"]
            \\inference_idle_timeout_secs = 1800
            \\
        , .{ e.id, e.id, base_url, e.id, e.budget.context, e.budget.output });
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

/// ZCode personal provider config (`ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`,
/// schema 1): one rule per chat model so ZCode never guesses limits from the id.
pub fn zcodeConfigJson(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\{{"schemaVersion":1,"config":{{
        \\"providerOrder":["mlx"],
        \\"defaultModelSelection":{{"providerId":"mlx","modelId":{f},"options":{{"reasoningLevel":"medium"}}}},
        \\"providerConfigRules":{{"providerRules":[{{"providerId":"mlx","providerName":"mlx-serve","enabled":true,"config":{{
        \\"group":"standard-personal","access":{{"type":"api-key","apiKey":"mlx-serve"}},
        \\"api":{{"type":"openai-chat-completions","baseUrl":"{s}/v1"}},"personalModelIds":[
    , .{ std.json.fmt(model, .{}), base_url });
    for (entries, 0..) |e, i| {
        try out.print(allocator, "{s}{f}", .{ if (i == 0) "" else ",", std.json.fmt(e.id, .{}) });
    }
    try out.appendSlice(allocator, "]}}]},\"modelConfigRules\":{\"manualProviderModelRules\":[],\"providerModelRules\":[");
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}{{"providerId":"mlx","modelId":{f},"config":{{"enabled":true,
            \\"properties":{{"contextWindow":{d},"requiresMfjsToolSchema":false,
            \\"inputFormat":{{"supportsText":true,"supportsImage":{},"supportsVideo":false,"supportsAudio":false,"supportsPdf":false}},
            \\"outputFormat":{{"supportsText":true}},"supportsToolCall":true,"supportsJsonSchemaOutput":false,
            \\"supportsNativeWebSearch":false,"supportsMidConversationSystem":false}},
            \\"optionSpecs":{{"reasoningLevel":{{"values":["none","low","medium","high"],"map":"{{\"reasoning_effort\": reasoningLevel}}"}},
            \\"maxOutputTokens":{{"max":{d},"map":"{{\"max_tokens\": maxOutputTokens}}"}}}}}}}}
        , .{ if (i == 0) "" else ",", std.json.fmt(e.id, .{}), e.budget.context, e.vision, e.budget.output });
    }
    try out.appendSlice(allocator, "]}}}\n");
    return out.toOwnedSlice(allocator);
}

// ── OpenCode version detection ──────────────────────────────────────────

/// The integration profile a detected OpenCode major selects. The binary
/// name no longer says the generation (Homebrew v2 ships as `opencode`),
/// so the version output is the only source of truth.
pub const OpenCodeGeneration = enum { v1, v2 };

pub const OpenCodeVersion = struct {
    generation: OpenCodeGeneration,
    /// The version token exactly as detected — the detection notice quotes
    /// the real string, a 3.0.0 is never displayed as a 2.x.
    version: []const u8,
};

/// Reads the first version token (optional `v` prefix, `digits.digits…`) out
/// of `opencode --version` output: major 1 → v1, major >= 2 → the newest
/// profile we ship. No token, major 0, or junk like `dev` is undecided.
pub fn parseOpencodeVersion(output: []const u8) ?OpenCodeVersion {
    var i: usize = 0;
    while (i < output.len) : (i += 1) {
        var start = i;
        if (output[i] == 'v' or output[i] == 'V') {
            if (i + 1 >= output.len or !std.ascii.isDigit(output[i + 1])) continue;
            start = i + 1;
        } else if (!std.ascii.isDigit(output[i])) continue;
        var j = start;
        while (j < output.len and std.ascii.isDigit(output[j])) j += 1;
        if (j >= output.len or output[j] != '.') continue;
        const major = std.fmt.parseInt(u32, output[start..j], 10) catch continue;
        if (major < 1) return null;
        var k = j;
        while (k < output.len and (std.ascii.isDigit(output[k]) or output[k] == '.')) k += 1;
        const version = std.mem.trim(u8, output[start..k], ".");
        return .{
            .generation = if (major == 1) .v1 else .v2,
            .version = version,
        };
    }
    return null;
}

/// The `launch opencode2` compatibility alias forces the v2 profile: the
/// canonical `opencode` name when it resolves to major >= 2, else the
/// legacy standalone binary — a v1 install never starts under it.
fn resolveOpencode2Bin(detected: ?OpenCodeVersion, legacy_installed: ?bool) ?[]const u8 {
    if (detected) |d| {
        if (d.generation == .v2) return "opencode";
    }
    if (legacy_installed == true) return "opencode2";
    return null;
}

/// Run one command through the same login shell the real launch execs
/// through, so PATH resolution (nvm, Homebrew) is identical for detection
/// and launch. Captured stdout comes back owned by the caller.
fn runLoginShell(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) !struct { ok: bool, out: []u8 } {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "/bin/zsh", "-l", "-c", cmd },
        .stdout_limit = .limited(64 * 1024),
    }) catch return error.LoginShellFailed;
    allocator.free(result.stderr);
    return .{
        .ok = switch (result.term) {
            .exited => |code| code == 0,
            else => false,
        },
        .out = result.stdout,
    };
}

/// Rc files print banners to stdout, so the version runs in a marked subshell
/// and only the marker's payload parses (keyed output, like Swift's
/// `detectInstalled`). Twin of OpenCodeVersion.swift's marker constants.
const version_marker = "MLXOCV=";
const version_probe_cmd = "if ! command -v opencode >/dev/null 2>&1; then printf 'MLXOCV=missing\\n'; else out=$(opencode --version 2>&1); rc=$?; printf 'MLXOCV=%s %s\\n' $rc \"$out\"; fi";

const MarkedVersion = struct { rc: u8, out: []const u8 };

/// The version subshell's exit code and output from a login-shell capture:
/// everything after the LAST `MLXOCV=<rc> ` token, so rc banners sitting
/// before it can never pose as the version. Null = it never answered.
fn extractMarkedVersion(captured: []const u8) ?MarkedVersion {
    const pos = std.mem.lastIndexOf(u8, captured, version_marker) orelse return null;
    const after = captured[pos + version_marker.len ..];
    const sp = std.mem.indexOfScalar(u8, after, ' ') orelse return null;
    const rc = std.fmt.parseInt(u8, after[0..sp], 10) catch return null;
    return .{ .rc = rc, .out = after[sp + 1 ..] };
}

const OpenCodeProbe = union(enum) {
    missing,
    version_failed: []const u8,
    unparsed: []const u8,
    ok: OpenCodeVersion,
};

/// Returned strings borrow from captured and remain valid only while it lives.
fn classifyOpenCodeProbe(captured: []const u8, shell_ok: bool) OpenCodeProbe {
    if (!shell_ok) return .{ .version_failed = captured };
    const pos = std.mem.lastIndexOf(u8, captured, version_marker) orelse return .{ .version_failed = captured };
    const payload = captured[pos + version_marker.len ..];
    if (std.mem.eql(u8, std.mem.trim(u8, payload, " \t\r\n"), "missing")) return .missing;
    const marked = extractMarkedVersion(captured) orelse return .{ .version_failed = captured };
    if (marked.rc != 0) return .{ .version_failed = marked.out };
    const parsed = parseOpencodeVersion(marked.out) orelse return .{ .unparsed = marked.out };
    return .{ .ok = parsed };
}

/// Returned version or failure output is owned by the caller and must be freed.
fn probeOpenCode(allocator: std.mem.Allocator, io: std.Io) !OpenCodeProbe {
    const run = try runLoginShell(allocator, io, version_probe_cmd);
    defer allocator.free(run.out);
    return switch (classifyOpenCodeProbe(run.out, run.ok)) {
        .missing => .missing,
        .version_failed => |out| .{ .version_failed = try allocator.dupe(u8, out) },
        .unparsed => |out| .{ .unparsed = try allocator.dupe(u8, out) },
        .ok => |parsed| .{ .ok = .{ .generation = parsed.generation, .version = try allocator.dupe(u8, parsed.version) } },
    };
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

fn needsShellQuoting(arg: []const u8) bool {
    for (arg) |c| {
        if (!(std.ascii.isAlphanumeric(c) or std.mem.indexOfScalar(u8, ".-_/+:@=", c) != null)) return true;
    }
    return arg.len == 0;
}

/// An agent argument derived from a model id: quoted only when it carries
/// shell-significant bytes, so ordinary ids keep the exact script bytes.
fn appendModelArg(out: *std.ArrayList(u8), allocator: std.mem.Allocator, arg: []const u8) !void {
    try out.append(allocator, ' ');
    if (needsShellQuoting(arg)) return appendQuoted(out, allocator, arg);
    return out.appendSlice(allocator, arg);
}

/// The script body run through `/bin/zsh -l -c` (login shell = the user's
/// real PATH). Configs are written by `writeConfigs` BEFORE this runs; the
/// script only exports env and execs the agent — same split as the app's
/// prepareConfig / scriptBody.
/// Below this the agent's own fixed prompt leaves every turn compacting or
/// truncated: Claude Code sends 40-70k before the first word (tool + MCP
/// schemas, skills catalogue), opencode ~8k, pi ~2k.
pub fn contextFloor(kind: AgentKind) u64 {
    return switch (kind) {
        .claude => 65536,
        .opencode, .opencode2 => 32768,
        else => 16384,
    };
}

pub const OpenCodeLaunch = struct { config: []const u8, bin: []const u8 };

pub fn scriptFor(allocator: std.mem.Allocator, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, opencode_launch: ?OpenCodeLaunch, extras: []const []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    if (budget.context > 0 and budget.context < contextFloor(kind)) {
        try out.print(allocator, "echo 'mlx-serve: the model advertises a {d}-token context; {s} needs {d}+ to work well (raise --ctx-size or Settings > Server > Context size).' >&2\n", .{ budget.context, @tagName(kind), contextFloor(kind) });
    }
    try out.print(allocator, "export MLX_SERVE_URL='{s}'\n", .{base_url});
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
                \\export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
                \\export API_TIMEOUT_MS=3600000
                \\export CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=1800000
                \\export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
                \\export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=1800000
                \\
            , .{ base_url, model, model, model, model, budget.output });
            // A long prefill and a long think on a local model outlast Claude Code's stream watchdogs; a fallback
            // re-sends the whole prompt as a non-stream request, which then times out and retries.
            // Claude Code assumes 200k for a model outside its catalog; declare the advertised context verbatim.
            if (budget.context > 0) {
                try out.print(allocator, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS={d}\n", .{budget.context});
            }
            try out.print(allocator, "claude --plugin-dir \"$HOME/.mlx-serve/{s}\" --model", .{claude_plugin_dir});
            try appendModelArg(&out, allocator, model);
        },
        .pi => {
            try out.appendSlice(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/pi"
                \\pi --provider mlx --model
            );
            try appendModelArg(&out, allocator, model);
        },
        .omp => {
            // omp still reads pi's env spelling (measured on v17 — the OMP_
            // rename reached only its help text); export both.
            try out.appendSlice(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                \\export OMP_CODING_AGENT_DIR="$HOME/.mlx-serve/omp"
                \\omp --model
            );
            const omp_model = try std.fmt.allocPrint(allocator, "mlx/{s}", .{model});
            defer allocator.free(omp_model);
            try appendModelArg(&out, allocator, omp_model);
        },
        .opencode => {
            try out.print(allocator,
                \\export OPENCODE_CONFIG_CONTENT='{s}'
                \\{s} --model
            , .{ opencode_launch.?.config, opencode_launch.?.bin });
            const oc_model = try std.fmt.allocPrint(allocator, "mlx/{s}", .{model});
            defer allocator.free(oc_model);
            try appendModelArg(&out, allocator, oc_model);
        },
        .opencode2 => {
            // The binary is the version-detected resolution, not a fixed
            // name: Homebrew ships v2 as `opencode`, `opencode2` is legacy.
            try out.print(allocator,
                \\export OPENCODE_CONFIG_CONTENT='{s}'
                \\export XDG_CONFIG_HOME="$HOME/.mlx-serve/opencode2"
                \\if ! command -v {s} >/dev/null 2>&1; then echo "{s} is not installed"; exit 127; fi
                \\{s} --standalone
            , .{ opencode_launch.?.config, opencode_launch.?.bin, opencode_launch.?.bin, opencode_launch.?.bin });
        },
        .codex => {
            // PATH first, then the CLI the desktop app bundles (codex's
            // rebranded app installs as ChatGPT.app or Codex.app, bundle id
            // com.openai.codex, CLI at Contents/Resources/codex) — a
            // desktop-app-only user has no codex on PATH. Mirrors the Swift
            // AgentConfigs.codexBinResolver.
            try out.appendSlice(allocator,
                \\CODEX_BIN="$(command -v codex)"
                \\if [ -z "$CODEX_BIN" ]; then
                \\  for app in "/Applications/ChatGPT.app" "/Applications/Codex.app" "$HOME/Applications/ChatGPT.app" "$HOME/Applications/Codex.app"; do
                \\    if [ -x "$app/Contents/Resources/codex" ]; then CODEX_BIN="$app/Contents/Resources/codex"; break; fi
                \\  done
                \\fi
                \\if [ -z "$CODEX_BIN" ]; then echo "codex is not installed: npm install -g @openai/codex, or install the ChatGPT app"; exit 127; fi
            );
            const overrides = try codexConfigOverrides(allocator, base_url, model, budget);
            defer allocator.free(overrides);
            try out.print(allocator, "\n\"$CODEX_BIN\" {s}", .{overrides});
        },
        .hermes => {
            try out.appendSlice(allocator,
                \\export HERMES_HOME="$HOME/.mlx-serve/hermes"
                \\hermes
            );
        },
        .zcode => {
            try out.appendSlice(allocator,
                \\export ZCODE_DATA_BASE_DIR="$HOME/.mlx-serve/zcode"
                \\export ZCODE_STORAGE_DIR="$HOME/.mlx-serve/zcode/storage"
                \\export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="$HOME/.mlx-serve/zcode/provider_config.json"
                \\if ! command -v zcode >/dev/null 2>&1; then echo "zcode is not installed: build or install ZCode (https://github.com/zai-org/ZCode)" >&2; exit 127; fi
                \\zcode
            );
        },
        .aider => {
            try out.print(allocator,
                \\export OPENAI_API_BASE='{s}/v1'
                \\export OPENAI_API_KEY=mlx-serve
                \\aider --model
            , .{base_url});
            const aider_model = try std.fmt.allocPrint(allocator, "openai/{s}", .{model});
            defer allocator.free(aider_model);
            try appendModelArg(&out, allocator, aider_model);
            try out.appendSlice(allocator, " --weak-model");
            try appendModelArg(&out, allocator, aider_model);
            try out.appendSlice(allocator, " --model-metadata-file ~/.mlx-serve/aider/model-metadata.json");
        },
        .fx => {
            try out.print(allocator,
                \\export FX_PROVIDER={s}
                \\export FX_MODEL={s}
                \\fx
            , .{ fx_provider, model });
        },
        .grok => {
            try out.appendSlice(allocator,
                \\export GROK_HOME="$HOME/.mlx-serve/grok"
                \\grok
            );
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

/// `open -g -b <bundle id>` finds the app under any bundle name; nonzero exit =
/// the app isn't installed, which is the detection.
fn tryStartApp(allocator: std.mem.Allocator, io: std.Io) bool {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "open", "-g", "-b", "com.dalcu.mlx-core" },
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

fn userOpencodeCliPath(allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("XDG_CONFIG_HOME")) |xdg| {
        const dir = std.mem.span(xdg);
        if (dir.len > 0) return std.fmt.allocPrint(allocator, "{s}/opencode/cli.json", .{dir});
    }
    return std.fmt.allocPrint(allocator, "{s}/.config/opencode/cli.json", .{homeDir()});
}

/// The shared skill folder under `~/.mlx-serve`, and the Claude Code plugin
/// that carries it (Claude has no skills dir we own; `--plugin-dir` loads it).
const skill_dir = "skills/" ++ agent_skills.name;
const claude_plugin_dir = "claude/plugin";

/// Where each agent discovers skills inside its dedicated config dir; opencode
/// reads `skills.paths` from its inline config instead, aider has no skills.
/// Codex gets none: it has no dedicated dir and no `-c` key for a skills root,
/// and a link in the user's own home would reach every codex session.
fn agentSkillLink(kind: AgentKind) ?[]const u8 {
    return switch (kind) {
        .pi => "pi/skills/" ++ agent_skills.name,
        .omp => "omp/skills/" ++ agent_skills.name,
        .hermes => "hermes/skills/" ++ agent_skills.name,
        .grok => "grok/skills/" ++ agent_skills.name,
        .claude => claude_plugin_dir ++ "/skills/" ++ agent_skills.name,
        .codex, .opencode, .opencode2, .aider, .fx, .zcode => null,
    };
}

/// Install the mlx-serve skill under `root` (`~/.mlx-serve`) wherever it is
/// missing, and link it into the agent's skills dir. Never overwrites: the
/// user's edits stick, the app's "Update System Prompt and Skills" refreshes.
pub fn installSkill(allocator: std.mem.Allocator, io: std.Io, root: []const u8, kind: AgentKind) !void {
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, root, .{});
    defer dir.close(io);
    try dir.createDirPath(io, skill_dir);
    for (agent_skills.files) |f| {
        const sub = try std.fmt.allocPrint(allocator, skill_dir ++ "/{s}", .{f.name});
        defer allocator.free(sub);
        dir.writeFile(io, .{ .sub_path = sub, .data = f.bytes, .flags = .{ .exclusive = true } }) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
    }
    const link = agentSkillLink(kind) orelse return;
    if (kind == .claude) {
        try dir.createDirPath(io, claude_plugin_dir ++ "/.claude-plugin");
        dir.writeFile(io, .{
            .sub_path = claude_plugin_dir ++ "/.claude-plugin/plugin.json",
            .data = "{\"name\": \"mlx-serve\", \"description\": \"Skills for the local mlx-serve server\"}\n",
            .flags = .{ .exclusive = true },
        }) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
    }
    try dir.createDirPath(io, std.fs.path.dirname(link).?);
    const target = try std.fmt.allocPrint(allocator, "{s}/" ++ skill_dir, .{root});
    defer allocator.free(target);
    dir.symLink(io, target, link, .{ .is_directory = true }) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
}

/// Write the agent's config files (the app's prepareConfig twin). opencode
/// carries its config inline and writes nothing.
fn writeConfigs(allocator: std.mem.Allocator, io: std.Io, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, entries: []const Entry) !void {
    const root = try std.fmt.allocPrint(allocator, "{s}/.mlx-serve", .{homeDir()});
    defer allocator.free(root);
    try installSkill(allocator, io, root, kind);
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
            const settings_path = try std.fmt.allocPrint(allocator, "{s}/.mlx-serve/pi/settings.json", .{homeDir()});
            defer allocator.free(settings_path);
            const existing = std.Io.Dir.cwd().readFileAlloc(io, settings_path, allocator, .limited(1 << 20)) catch
                try allocator.dupe(u8, "{}");
            defer allocator.free(existing);
            const settings = try mergePiSettingsJson(allocator, existing, budget.context);
            defer allocator.free(settings);
            try writeAgentFile(allocator, io, "pi", "settings.json", settings);
        },
        .omp => {
            const yml = try ompModelsYml(allocator, base_url, entries);
            defer allocator.free(yml);
            try writeAgentFile(allocator, io, "omp", "models.yml", yml);
        },
        .codex => {}, // settings ride -c overrides on the launch line
        .hermes => {
            const yaml = try hermesConfigYaml(allocator, base_url, model, entries);
            defer allocator.free(yaml);
            try writeAgentFile(allocator, io, "hermes", "config.yaml", yaml);
            const env = try hermesEnvFile(allocator, base_url);
            defer allocator.free(env);
            try writeAgentFile(allocator, io, "hermes", ".env", env);
        },
        .zcode => {
            const json = try zcodeConfigJson(allocator, base_url, model, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "zcode", "provider_config.json", json);
        },
        .aider => {
            const json = try aiderMetadataJson(allocator, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "aider", "model-metadata.json", json);
        },
        .fx => {
            const dir_path = try std.fmt.allocPrint(allocator, "{s}/.fx", .{homeDir()});
            defer allocator.free(dir_path);
            var dir = try std.Io.Dir.cwd().createDirPathOpen(io, dir_path, .{});
            defer dir.close(io);
            const existing = dir.readFileAlloc(io, "settings.json", allocator, .limited(1 << 20)) catch |err| switch (err) {
                error.FileNotFound => try allocator.dupe(u8, "{}"),
                else => return err,
            };
            defer allocator.free(existing);
            const json = try mergeFxSettingsJson(allocator, existing, base_url, entries);
            defer allocator.free(json);
            try dir.writeFile(io, .{ .sub_path = "settings.json", .data = json, .flags = .{ .permissions = .fromMode(0o600) } });
        },
        .grok => {
            const toml = try grokConfigToml(allocator, base_url, model, entries);
            defer allocator.free(toml);
            try writeAgentFile(allocator, io, "grok", "config.toml", toml);
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

    // Detect the OpenCode generation before anything is written: the v2
    // profile ships different config files (cli.json + monitor plugin), so
    // the routing decision cannot live inside the launch script.
    var oc_kind: AgentKind = parsed.kind;
    var oc_bin: []const u8 = "opencode";
    if (parsed.kind == .opencode) {
        const probe = probeOpenCode(allocator, io) catch {
            log.err("could not run the login shell to detect the OpenCode version; retry, or use `mlx-serve launch opencode2` to force the v2 integration.\n", .{});
            std.process.exit(1);
        };
        switch (probe) {
            .missing => {
                log.err("OpenCode is not installed or is not available on PATH.\n", .{});
                log.err("install OpenCode and make sure `opencode` is on PATH, then rerun.\n", .{});
                std.process.exit(1);
            },
            .version_failed, .unparsed => |out| {
                log.err("could not determine the installed OpenCode version from `opencode --version`.\n", .{});
                log.err("Output:\n{s}\n", .{out});
                log.err("mlx-serve currently supports OpenCode 1.x (the v1 integration) and 2.x or newer (the v2 integration).\n", .{});
                allocator.free(out);
                std.process.exit(1);
            },
            .ok => |v| {
                oc_kind = if (v.generation == .v1) .opencode else .opencode2;
                log.info("detected OpenCode {s}; using the {s} integration.\n", .{
                    v.version, if (v.generation == .v1) "v1" else "v2",
                });
                allocator.free(v.version);
            },
        }
    } else if (parsed.kind == .opencode2) {
        // Compatibility alias forcing the v2 profile; a v1 `opencode` must
        // never start under it, so only a major >= 2 resolution counts.
        const probe = probeOpenCode(allocator, io) catch .missing;
        var detected: ?OpenCodeVersion = null;
        switch (probe) {
            .ok => |v| {
                if (v.generation == .v2) detected = v;
                if (v.generation != .v2) allocator.free(v.version);
            },
            .version_failed, .unparsed => |out| allocator.free(out),
            .missing => {},
        }
        var legacy_ok: ?bool = null;
        if (detected == null) {
            if (runLoginShell(allocator, io, "if command -v opencode2 >/dev/null 2>&1; then printf 'MLXOCL=1\\n'; else printf 'MLXOCL=0\\n'; fi")) |legacy| {
                defer allocator.free(legacy.out);
                if (legacy.ok) {
                    if (std.mem.lastIndexOf(u8, legacy.out, "MLXOCL=")) |pos| {
                        const payload = std.mem.trim(u8, legacy.out[pos + "MLXOCL=".len ..], " \t\r\n");
                        if (std.mem.eql(u8, payload, "1")) legacy_ok = true;
                        if (std.mem.eql(u8, payload, "0")) legacy_ok = false;
                    }
                }
            } else |_| {}
            if (legacy_ok == null) {
                log.err("could not check the legacy OpenCode binary through the login shell; check your shell startup files and retry.\n", .{});
                std.process.exit(1);
            }
        }
        oc_bin = resolveOpencode2Bin(detected, legacy_ok) orelse {
            log.err("no OpenCode v2 binary found: `launch opencode2` needs an `opencode` 2.x+ or the legacy `opencode2` on PATH.\n", .{});
            log.err("install OpenCode (https://opencode.ai/docs) and rerun, or use `mlx-serve launch opencode`.\n", .{});
            std.process.exit(1);
        };
        if (detected) |v| {
            log.info("detected OpenCode {s}; using the v2 integration.\n", .{v.version});
            allocator.free(v.version);
        }
    }

    writeConfigs(allocator, io, oc_kind, base_url, chosen.id, chosen.budget, models.entries) catch |err| {
        log.err("could not write the {s} config: {s}\n", .{ @tagName(oc_kind), @errorName(err) });
        std.process.exit(1);
    };

    const oc_config: ?[]u8 = if (oc_kind == .opencode or oc_kind == .opencode2)
        try opencodeJson(allocator, base_url, models.entries, if (oc_kind == .opencode2) chosen.id else null, oc_kind == .opencode2)
    else
        null;
    defer if (oc_config) |c| allocator.free(c);

    const script = try scriptFor(allocator, oc_kind, base_url, chosen.id, chosen.budget, if (oc_config) |config| .{ .config = config, .bin = oc_bin } else null, parsed.extras);
    defer allocator.free(script);

    if (oc_kind == .opencode2) {
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

test "budgetForContext mirrors AgentBudget: ctx/2 clamped to [1024, 65536], 0 = fallback" {
    // Thinking shares the response cap: a 24k window at ctx/4 gave pi 6144,
    // which one xhigh design turn on Qwen3.8 spent entirely on thinking.
    try t.expectEqual(FALLBACK_BUDGET, budgetForContext(0));
    try t.expectEqual(Budget{ .context = 4096, .output = 2048 }, budgetForContext(4096));
    try t.expectEqual(Budget{ .context = 2048, .output = 1024 }, budgetForContext(2048));
    try t.expectEqual(Budget{ .context = 24576, .output = 12288 }, budgetForContext(24576));
    try t.expectEqual(Budget{ .context = 90112, .output = 45056 }, budgetForContext(90112));
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

test "codex overrides: responses wire API, keyless, advertised context" {
    const ov = try codexConfigOverrides(t.allocator, "http://127.0.0.1:11234", "m1", .{ .context = 90112, .output = 45056 });
    defer t.allocator.free(ov);
    try t.expectEqualStrings(
        "-c 'model=\"m1\"' -c 'model_provider=\"mlx\"' -c model_context_window=90112" ++
            " -c 'model_providers.mlx.name=\"MLX Serve (local)\"'" ++
            " -c 'model_providers.mlx.base_url=\"http://127.0.0.1:11234/v1\"'" ++
            " -c 'model_providers.mlx.wire_api=\"responses\"'",
        ov,
    );
}

test "codex overrides: a zero advertised context omits model_context_window" {
    const ov = try codexConfigOverrides(t.allocator, "http://x:1", "m1", .{ .context = 0, .output = 0 });
    defer t.allocator.free(ov);
    try t.expect(std.mem.indexOf(u8, ov, "model_context_window") == null);
    try t.expect(std.mem.indexOf(u8, ov, "-c 'model_providers.mlx.base_url=\"http://x:1/v1\"'") != null);
}

test "codex overrides escape a model id for TOML, then for the shell" {
    const ov = try codexConfigOverrides(t.allocator, "http://x:1", "o'ne\"il\\m", .{ .context = 0, .output = 0 });
    defer t.allocator.free(ov);
    try t.expect(std.mem.startsWith(u8, ov, "-c 'model=\"o'\\''ne\\\"il\\\\m\"' -c 'model_provider=\"mlx\"'"));
}

test "codex gets no skill link and no dedicated dir" {
    try t.expect(agentSkillLink(.codex) == null);
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    try installSkill(t.allocator, io, root, .codex);
    try t.expect(tmp.dir.access(io, "codex", .{}) == error.FileNotFound);
}

test "model args ride the invocation quoted only when they need it" {
    const odd = try scriptFor(t.allocator, .claude, "http://x:1", "a b'c", .{ .context = 65536, .output = 32768 }, null, &.{});
    defer t.allocator.free(odd);
    try t.expect(std.mem.indexOf(u8, odd, "--model 'a b'\\''c'") != null);
    const plain = try scriptFor(t.allocator, .omp, "http://x:1", "ok/m1", .{ .context = 65536, .output = 32768 }, null, &.{});
    defer t.allocator.free(plain);
    try t.expect(std.mem.indexOf(u8, plain, "omp --model mlx/ok/m1\n") != null);
    const oc = try scriptFor(t.allocator, .opencode, "http://x:1", "a b", .{ .context = 65536, .output = 32768 }, .{ .config = "{}", .bin = "opencode" }, &.{});
    defer t.allocator.free(oc);
    try t.expect(std.mem.indexOf(u8, oc, "\nopencode --model 'mlx/a b'\n") != null);
}

test "pi models.json and opencode config parse as JSON and stay single-quote-free" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = true, .loaded = true },
        .{ .id = "m2", .budget = .{ .context = 8192, .output = 2048 }, .vision = false, .loaded = false },
    };
    const pi_json = try piModelsJson(t.allocator, "http://127.0.0.1:11234", &entries);
    defer t.allocator.free(pi_json);
    const oc_json = try opencodeJson(t.allocator, "http://127.0.0.1:11234", &entries, "m1", true);
    defer t.allocator.free(oc_json);
    for ([_][]const u8{ pi_json, oc_json }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
        defer parsed.deinit();
        // opencode's config rides single-quoted inside the launch script.
        try t.expect(std.mem.indexOf(u8, json, "'") == null);
    }
}

test "pi models.json sends the thinking level as reasoning_effort, off as none" {
    // thinkingFormat "qwen" makes pi send only enable_thinking: low/medium never
    // reached the server and every turn thought unbounded.
    const entries = [_]Entry{.{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true }};
    const json = try piModelsJson(t.allocator, "http://127.0.0.1:11234", &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const mlx_p = parsed.value.object.get("providers").?.object.get("mlx").?.object;
    try t.expect(mlx_p.get("compat").?.object.get("thinkingFormat") == null);
    const m = mlx_p.get("models").?.array.items[0].object;
    const levels = m.get("thinkingLevelMap").?.object;
    try t.expectEqualStrings("none", levels.get("off").?.string);
    // pi offers xhigh/max only when the map names them, else clamps max to high.
    try t.expectEqualStrings("xhigh", levels.get("xhigh").?.string);
    try t.expectEqualStrings("max", levels.get("max").?.string);
}

test "compactionReserve: a quarter of the window, capped where the agents' own defaults take over" {
    // pi keeps 20000 recent tokens and opencode2 reserves a 20000 buffer by
    // default; both assume a 200k window. A 24k window compacted before its
    // first reply (opencode2) or never (pi).
    try t.expectEqual(@as(u64, 6144), compactionReserve(24576));
    try t.expectEqual(@as(u64, 2048), compactionReserve(8192));
    try t.expectEqual(@as(u64, 1024), compactionReserve(2048));
    try t.expectEqual(@as(u64, 20000), compactionReserve(262144));
}

test "pi settings.json merge scales compaction to the window and keeps the rest" {
    const existing =
        \\{"theme":"dark","defaultProvider":"mlx","compaction":{"enabled":false,"reserveTokens":1}}
    ;
    const json = try mergePiSettingsJson(t.allocator, existing, 24576);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try t.expectEqualStrings("dark", obj.get("theme").?.string);
    const c = obj.get("compaction").?.object;
    // The user's own enabled flag survives; the numbers are ours.
    try t.expectEqual(false, c.get("enabled").?.bool);
    try t.expectEqual(@as(i64, 10240), c.get("reserveTokens").?.integer);
    try t.expectEqual(@as(i64, 6144), c.get("keepRecentTokens").?.integer);

    // A big window keeps pi's own defaults (16384 / 20000); an empty file is fine.
    const big = try mergePiSettingsJson(t.allocator, "", 262144);
    defer t.allocator.free(big);
    const bp = try std.json.parseFromSlice(std.json.Value, t.allocator, big, .{});
    defer bp.deinit();
    const bc = bp.value.object.get("compaction").?.object;
    try t.expectEqual(@as(i64, 16384), bc.get("reserveTokens").?.integer);
    try t.expectEqual(@as(i64, 20000), bc.get("keepRecentTokens").?.integer);
}

test "fx settings merge owns providers.mlx-serve and keeps everything else" {
    const existing =
        \\{"provider":"gateway","models":{"gateway":"x/y"},"providers":{"other":{"protocol":"openai-chat-completions","base_url":"https://o/v1","auth":{"type":"none"}},"mlx-serve":{"base_url":"stale"}}}
    ;
    const entries = [_]Entry{
        .{ .id = "org/m1", .budget = .{ .context = 32768, .output = 16384 }, .vision = false, .loaded = true },
        .{ .id = "org/m2", .budget = .{ .context = 262144, .output = 65536 }, .vision = true, .loaded = false },
    };
    const json = try mergeFxSettingsJson(t.allocator, existing, "http://127.0.0.1:11234", &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    // The user's default provider and model stay theirs: a launch selects ours by env.
    try t.expectEqualStrings("gateway", obj.get("provider").?.string);
    try t.expectEqualStrings("x/y", obj.get("models").?.object.get("gateway").?.string);
    const providers = obj.get("providers").?.object;
    try t.expectEqualStrings("https://o/v1", providers.get("other").?.object.get("base_url").?.string);
    const ours = providers.get(fx_provider).?.object;
    try t.expectEqualStrings("openai-chat-completions", ours.get("protocol").?.string);
    try t.expectEqualStrings("http://127.0.0.1:11234/v1", ours.get("base_url").?.string);
    try t.expectEqualStrings("none", ours.get("auth").?.object.get("type").?.string);
    const m2 = ours.get("model_metadata").?.object.get("org/m2").?.object;
    try t.expectEqual(@as(i64, 262144), m2.get("context_window").?.integer);
    try t.expectEqual(@as(i64, 65536), m2.get("max_output_tokens").?.integer);
    try t.expectEqual(true, m2.get("supports_tool_use").?.bool);
    try t.expectEqual(true, m2.get("supports_vision").?.bool);
    try t.expectEqual(false, ours.get("model_metadata").?.object.get("org/m1").?.object.get("supports_vision").?.bool);
}

test "fx settings merge starts a missing file and never replaces one it cannot read" {
    const entries = [_]Entry{.{ .id = "m1", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true }};
    const fresh = try mergeFxSettingsJson(t.allocator, " \n", "http://x:1", &entries);
    defer t.allocator.free(fresh);
    try t.expect(std.mem.indexOf(u8, fresh, "\"mlx-serve\"") != null);
    try t.expectError(error.UnreadableFxSettings, mergeFxSettingsJson(t.allocator, "{\"provider\": ", "http://x:1", &entries));
    try t.expectError(error.UnreadableFxSettings, mergeFxSettingsJson(t.allocator, "[]", "http://x:1", &entries));
}

test "fx script selects our provider and model by env, never by a saved default" {
    try t.expectEqual(AgentKind.fx, AgentKind.fromName("fx").?);
    const script = try scriptFor(t.allocator, .fx, "http://x:1", "org/m1", FALLBACK_BUDGET, null, &.{"ask"});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export FX_PROVIDER=mlx-serve\nexport FX_MODEL=org/m1\nfx 'ask'\n") != null);
}

test "grok config: every chat model keyed by its id, helpers pinned to the launched model" {
    const entries = [_]Entry{
        .{ .id = "org/m1", .budget = .{ .context = 32768, .output = 16384 }, .vision = false, .loaded = true },
        .{ .id = "org/m2", .budget = .{ .context = 262144, .output = 65536 }, .vision = true, .loaded = false },
    };
    const toml = try grokConfigToml(t.allocator, "http://127.0.0.1:11234", "org/m2", &entries);
    defer t.allocator.free(toml);
    // An unknown helper model (grok-4.6) would reach the server and load its default.
    for ([_][]const u8{ "default", "session_summary", "image_description", "prompt_suggestion" }) |key| {
        const line = try std.fmt.allocPrint(t.allocator, "{s} = \"org/m2\"\n", .{key});
        defer t.allocator.free(line);
        try t.expect(std.mem.indexOf(u8, toml, line) != null);
    }
    try t.expect(std.mem.indexOf(u8, toml,
        \\[model."org/m1"]
        \\model = "org/m1"
        \\base_url = "http://127.0.0.1:11234/v1"
        \\name = "org/m1 (mlx-serve)"
        \\api_key = "mlx-serve"
        \\context_window = 32768
        \\max_completion_tokens = 16384
        \\supports_reasoning_effort = true
    ) != null);
    try t.expect(std.mem.indexOf(u8, toml, "context_window = 262144\nmax_completion_tokens = 65536\n") != null);
}

test "grok script rides a dedicated GROK_HOME" {
    try t.expectEqual(AgentKind.grok, AgentKind.fromName("grok").?);
    const script = try scriptFor(t.allocator, .grok, "http://x:1", "m1", FALLBACK_BUDGET, null, &.{"-c"});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export GROK_HOME=\"$HOME/.mlx-serve/grok\"\ngrok '-c'\n") != null);
}

test "opencode config: limit.output is the compaction reserve, opencode2 gets a scaled compaction block" {
    // opencode never sends max_tokens; `limit.output` is only the room it
    // keeps free before compacting, and opencode2 compacts at
    // context - max(min(output, 32000), buffer) with buffer defaulting to 20000.
    const entries = [_]Entry{
        .{ .id = "m1", .budget = budgetForContext(24576), .vision = false, .loaded = true },
    };
    const v1 = try opencodeJson(t.allocator, "http://127.0.0.1:11234", &entries, null, false);
    defer t.allocator.free(v1);
    const p1 = try std.json.parseFromSlice(std.json.Value, t.allocator, v1, .{});
    defer p1.deinit();
    const limit = p1.value.object.get("provider").?.object.get("mlx").?.object.get("models").?.object.get("m1").?.object.get("limit").?.object;
    try t.expectEqual(@as(i64, 6144), limit.get("output").?.integer);
    try t.expect(p1.value.object.get("compaction") == null);
    // opencode sends no reasoning_effort unless the model declares one: thinking stayed off.
    const m1 = p1.value.object.get("provider").?.object.get("mlx").?.object.get("models").?.object.get("m1").?.object;
    try t.expectEqualStrings("medium", m1.get("options").?.object.get("reasoningEffort").?.string);
    try t.expectEqualStrings("none", m1.get("variants").?.object.get("none").?.object.get("reasoningEffort").?.string);
    try t.expectEqualStrings("high", m1.get("variants").?.object.get("high").?.object.get("reasoningEffort").?.string);

    const v2 = try opencodeJson(t.allocator, "http://127.0.0.1:11234", &entries, "m1", true);
    defer t.allocator.free(v2);
    const p2 = try std.json.parseFromSlice(std.json.Value, t.allocator, v2, .{});
    defer p2.deinit();
    const c = p2.value.object.get("compaction").?.object;
    try t.expectEqual(@as(i64, 6144), c.get("buffer").?.integer);
    try t.expectEqual(@as(i64, 6144), c.get("keep").?.object.get("tokens").?.integer);
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
    try t.expect(std.mem.indexOf(u8, script, "\"$CODEX_BIN\" -c 'model=\"m1\"'") != null);
    try t.expect(std.mem.indexOf(u8, script, "-c 'model_providers.mlx.wire_api=\"responses\"' 'resume' 'it'\\''s'") != null);
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
    const script = try scriptFor(t.allocator, .opencode2, "http://127.0.0.1:11234", "m1", .{ .context = 4096, .output = 1024 }, .{ .config = cfg, .bin = "opencode2" }, &.{ "resume", "it's" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export XDG_CONFIG_HOME=\"$HOME/.mlx-serve/opencode2\"") != null);
    try t.expect(std.mem.indexOf(u8, script, "export OPENCODE_CONFIG_CONTENT='{\"provider\":{}}'") != null);
    // v2 has no root --model flag and resolves models in a SHARED background
    // service that never sees our env: --standalone, model pinned in the config.
    try t.expect(std.mem.indexOf(u8, script, "opencode2 --standalone") != null);
    try t.expect(std.mem.indexOf(u8, script, "--model") == null);
    try t.expect(std.mem.indexOf(u8, script, "opencode2 is not installed") != null);
    try t.expect(std.mem.indexOf(u8, script, "exit 127") != null);
    try t.expect(std.mem.indexOf(u8, script, "'resume' 'it'\\''s'") != null);
}

test "opencode version parser: major 1 = v1, major >= 2 = the newest profile, else undecided" {
    for ([_][]const u8{ "1.18.34", "v1.18.34", "opencode 1.18.34\n" }) |input| {
        const got = parseOpencodeVersion(input).?;
        try t.expectEqual(OpenCodeGeneration.v1, got.generation);
        try t.expectEqualStrings("1.18.34", got.version);
    }
    // Major >= 2 routes to the newest profile with its real version quoted verbatim.
    const later = [_]struct { in: []const u8, version: []const u8 }{
        .{ .in = "2.0.20", .version = "2.0.20" },
        .{ .in = "v2.0.20", .version = "2.0.20" },
        .{ .in = "opencode v2.0.20", .version = "2.0.20" },
        .{ .in = "3.0.0", .version = "3.0.0" },
        .{ .in = "2.0.20-nightly", .version = "2.0.20" },
    };
    for (later) |c| {
        const got = parseOpencodeVersion(c.in).?;
        try t.expectEqual(OpenCodeGeneration.v2, got.generation);
        try t.expectEqualStrings(c.version, got.version);
    }
    // No token, a non-numeric tag, or a pre-1 major is undecided.
    for ([_][]const u8{ "", "dev", "0.14.0", "OpenCode — canary build" }) |input|
        try t.expect(parseOpencodeVersion(input) == null);
}

test "marked version output survives login-shell rc banners" {
    // An rc file's own banner must not pose as the version (keyed output).
    const m = extractMarkedVersion("Welcome! node 18.2.0\nMLXOCV=0 opencode v2.0.20\n").?;
    try t.expectEqual(0, m.rc);
    const v = parseOpencodeVersion(m.out).?;
    try t.expectEqual(OpenCodeGeneration.v2, v.generation);
    try t.expectEqualStrings("2.0.20", v.version);
    // A nonzero rc is the --version failure itself; the tail is its message.
    try t.expectEqual(127, extractMarkedVersion("MLXOCV=127 opencode: boom\n").?.rc);
    // A multiline version output keeps its tail after the marker line.
    try t.expect(std.mem.indexOf(u8, extractMarkedVersion("b\nMLXOCV=0 open 2.0.20\nbuild x\n").?.out, "build x") != null);
    // No marker, no rc token, or a junk rc = the subshell never answered.
    try t.expect(extractMarkedVersion("opencode v2.0.20") == null);
    try t.expect(extractMarkedVersion("MLXOCV=0") == null);
    try t.expect(extractMarkedVersion("MLXOCV=x out") == null);
}

test "launch opencode2 resolves the v2 `opencode` first, then the legacy binary" {
    const v2 = OpenCodeVersion{ .generation = .v2, .version = "2.0.20" };
    const v3 = OpenCodeVersion{ .generation = .v2, .version = "3.0.0" };
    const v1 = OpenCodeVersion{ .generation = .v1, .version = "1.18.34" };
    try t.expectEqualStrings("opencode", resolveOpencode2Bin(v2, true).?);
    try t.expectEqualStrings("opencode", resolveOpencode2Bin(v3, false).?);
    try t.expectEqualStrings("opencode2", resolveOpencode2Bin(v1, true).?);
    try t.expectEqualStrings("opencode2", resolveOpencode2Bin(null, true).?);
    try t.expect(resolveOpencode2Bin(v1, false) == null);
    try t.expect(resolveOpencode2Bin(null, false) == null);
}

test "opencode launch routing: the detected generation picks the script, config, and binary" {
    const b = Budget{ .context = 65536, .output = 8192 };
    const entries = [_]Entry{.{ .id = "m1", .budget = b, .vision = false, .loaded = true }};
    // A Homebrew migration under ONE command: a v1 install gets the inline
    // config + --model arm and nothing under the v2 config dir.
    const v1_cfg = try opencodeJson(t.allocator, "http://x:1", &entries, null, false);
    defer t.allocator.free(v1_cfg);
    const before = try scriptFor(t.allocator, .opencode, "http://x:1", "m1", b, .{ .config = v1_cfg, .bin = "opencode" }, &.{});
    defer t.allocator.free(before);
    try t.expect(std.mem.indexOf(u8, before, "opencode --model mlx/m1") != null);
    try t.expect(std.mem.indexOf(u8, before, "XDG_CONFIG_HOME") == null);
    try t.expect(std.mem.indexOf(u8, v1_cfg, "\"model\"") == null);

    // After the same `opencode` name upgrades to 2.x, the SAME launch routes
    // to the v2 arm: standalone invocation under the RESOLVED binary name,
    // the dedicated XDG dir, and the model pinned in the config.
    const v2_cfg = try opencodeJson(t.allocator, "http://x:1", &entries, "m1", true);
    defer t.allocator.free(v2_cfg);
    const after = try scriptFor(t.allocator, .opencode2, "http://x:1", "m1", b, .{ .config = v2_cfg, .bin = "opencode" }, &.{});
    defer t.allocator.free(after);
    try t.expect(std.mem.indexOf(u8, after, "\nopencode --standalone") != null);
    try t.expect(std.mem.indexOf(u8, after, "command -v opencode ") != null);
    try t.expect(std.mem.indexOf(u8, after, "opencode2 --standalone") == null);
    try t.expect(std.mem.indexOf(u8, after, "XDG_CONFIG_HOME=\"$HOME/.mlx-serve/opencode2\"") != null);
    try t.expect(std.mem.indexOf(u8, v2_cfg, "\"model\": \"mlx/m1\"") != null);
    try t.expect(std.mem.indexOf(u8, v2_cfg, "\"compaction\"") != null);

    // The legacy alias resolves the standalone opencode2 binary instead.
    const legacy = try scriptFor(t.allocator, .opencode2, "http://x:1", "m1", b, .{ .config = v2_cfg, .bin = "opencode2" }, &.{});
    defer t.allocator.free(legacy);
    try t.expect(std.mem.indexOf(u8, legacy, "\nopencode2 --standalone") != null);
}

test "opencodeJson pins the default model only when asked" {
    const entries = [_]Entry{.{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = false }};
    const plain = try opencodeJson(t.allocator, "http://127.0.0.1:11234", &entries, null, false);
    defer t.allocator.free(plain);
    try t.expect(std.mem.indexOf(u8, plain, "\"model\"") == null);

    const pinned = try opencodeJson(t.allocator, "http://127.0.0.1:11234", &entries, "m1", true);
    defer t.allocator.free(pinned);
    try t.expect(std.mem.indexOf(u8, pinned, "\"model\": \"mlx/m1\"") != null);
}

test "claude script keeps a slow local turn on one streamed request" {
    const script = try scriptFor(t.allocator, .claude, "http://x:1", "m1", budgetForContext(786432), null, &.{});
    defer t.allocator.free(script);
    for ([_][]const u8{
        "export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1\n",
        "export API_TIMEOUT_MS=3600000\n",
        "export CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=1800000\n",
        "export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000\n",
        "export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=1800000\n",
    }) |line| try t.expect(std.mem.indexOf(u8, script, line) != null);
}

test "claude script declares the advertised context window (CLAUDE_CODE_MAX_CONTEXT_TOKENS)" {
    // Claude Code assumes 200k outside its catalog; CLAUDE_CODE_MAX_CONTEXT_TOKENS is the override.
    const script = try scriptFor(t.allocator, .claude, "http://x:1", "m1", budgetForContext(786432), null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=786432") != null);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536") != null);
    try t.expect(std.mem.indexOf(u8, script, " --model m1\n") != null);

    // An unknown context is not a claim: omit the export rather than pin a
    // number the server never advertised.
    const unknown = try scriptFor(t.allocator, .claude, "http://x:1", "m1", .{ .context = 0, .output = 8192 }, null, &.{});
    defer t.allocator.free(unknown);
    try t.expect(std.mem.indexOf(u8, unknown, "CLAUDE_CODE_MAX_CONTEXT_TOKENS") == null);
    try t.expect(std.mem.indexOf(u8, unknown, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192") != null);
}

test "skill install: writes a missing skill, keeps an edited one, links the agent's skills dir" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];

    try tmp.dir.createDirPath(io, "skills/mlx-serve");
    try tmp.dir.writeFile(io, .{ .sub_path = "skills/mlx-serve/SKILL.md", .data = "edited" });
    try installSkill(t.allocator, io, root, .pi);
    try installSkill(t.allocator, io, root, .claude);

    var got: [64]u8 = undefined;
    try t.expectEqualStrings("edited", try tmp.dir.readFile(io, "pi/skills/mlx-serve/SKILL.md", &got));
    try t.expectEqualStrings("edited", try tmp.dir.readFile(io, "claude/plugin/skills/mlx-serve/SKILL.md", &got));
    const media = try tmp.dir.readFileAlloc(io, "skills/mlx-serve/media.md", t.allocator, .limited(1 << 20));
    defer t.allocator.free(media);
    try t.expect(std.mem.indexOf(u8, media, "/v1/images/generations") != null);
    _ = try tmp.dir.statFile(io, "claude/plugin/.claude-plugin/plugin.json", .{});
}

test "launch scripts point every agent at the skill and export MLX_SERVE_URL" {
    const b = Budget{ .context = 65536, .output = 8192 };
    const claude = try scriptFor(t.allocator, .claude, "http://x:1", "m1", b, null, &.{});
    defer t.allocator.free(claude);
    try t.expect(std.mem.indexOf(u8, claude, "export MLX_SERVE_URL='http://x:1'\n") != null);
    try t.expect(std.mem.indexOf(u8, claude, "claude --plugin-dir \"$HOME/.mlx-serve/claude/plugin\" --model m1") != null);

    const entries = [_]Entry{.{ .id = "m1", .budget = b, .vision = false, .loaded = true }};
    const oc = try opencodeJson(t.allocator, "http://x:1", &entries, null, false);
    defer t.allocator.free(oc);
    try t.expect(std.mem.indexOf(u8, oc, "\"skills\": {\"paths\": [\"~/.mlx-serve/skills/mlx-serve\"]}") != null);
}

test "opencode probe status distinguishes missing from executable failure" {
    const cases = [_]struct { capture: []const u8, shell_ok: bool = true, expected: std.meta.Tag(OpenCodeProbe) }{
        .{ .capture = "MLXOCV=missing\n", .expected = .missing },
        .{ .capture = "banner 18.2.0\nMLXOCV=missing\n", .expected = .missing },
        .{ .capture = "MLXOCV=127 \n", .expected = .version_failed },
        .{ .capture = "MLXOCV=1 failed\n", .expected = .version_failed },
        .{ .capture = "2.0.20", .expected = .version_failed },
        .{ .capture = "MLXOCV=0", .expected = .version_failed },
        .{ .capture = "MLXOCV=x out", .expected = .version_failed },
        .{ .capture = "MLXOCV=0 dev\n", .expected = .unparsed },
        .{ .capture = "MLXOCV=0 2.0.20\n", .shell_ok = false, .expected = .version_failed },
        .{ .capture = "MLXOCV=missing\n", .shell_ok = false, .expected = .version_failed },
        .{ .capture = "MLXOCV=missing\nMLXOCV=0 2.0.20\n", .expected = .ok },
        .{ .capture = "MLXOCV=0 2.0.20\nMLXOCV=missing\n", .expected = .missing },
    };
    for (cases) |case| try t.expectEqual(case.expected, std.meta.activeTag(classifyOpenCodeProbe(case.capture, case.shell_ok)));
    for ([_][]const u8{ "1.18.34", "2.0.20", "3.0.0", "10.2.0" }) |version| {
        const captured = try std.fmt.allocPrint(t.allocator, "banner node 18.2.0\nMLXOCV=0 opencode v{s}\nbuild x\n", .{version});
        defer t.allocator.free(captured);
        const result = classifyOpenCodeProbe(captured, true).ok;
        try t.expectEqualStrings(version, result.version);
        try t.expectEqual(if (version[0] == '1' and version[1] == '.') OpenCodeGeneration.v1 else .v2, result.generation);
    }
}

test "zcode config escapes arbitrary model ids and declares server budgets and wire maps" {
    const entries = [_]Entry{
        .{ .id = "org/model\"quoted", .budget = .{ .context = 98304, .output = 49152 }, .vision = true, .loaded = false },
        .{ .id = "glm-native", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true },
    };
    const json = try zcodeConfigJson(t.allocator, "http://127.0.0.1:11234", entries[0].id, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try t.expectEqualStrings(entries[0].id, config.get("defaultModelSelection").?.object.get("modelId").?.string);
    const provider = config.get("providerConfigRules").?.object.get("providerRules").?.array.items[0].object;
    try t.expectEqualStrings("mlx-serve", provider.get("providerName").?.string);
    try t.expectEqualStrings("http://127.0.0.1:11234/v1", provider.get("config").?.object.get("api").?.object.get("baseUrl").?.string);
    const rules = config.get("modelConfigRules").?.object.get("providerModelRules").?.array.items;
    try t.expectEqual(@as(usize, 2), rules.len);
    const m = rules[0].object.get("config").?.object;
    try t.expectEqual(@as(i64, 98304), m.get("properties").?.object.get("contextWindow").?.integer);
    try t.expectEqualStrings("{\"max_tokens\": maxOutputTokens}", m.get("optionSpecs").?.object.get("maxOutputTokens").?.object.get("map").?.string);
    const script = try scriptFor(t.allocator, .zcode, "http://127.0.0.1:11234", entries[0].id, entries[0].budget, null, &.{ "--prompt", "it's a prompt" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE") != null);
    try t.expect(std.mem.indexOf(u8, script, "zcode '--prompt' 'it'\\''s a prompt'") != null);
}
