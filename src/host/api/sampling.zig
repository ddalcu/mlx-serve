//! Sampling, resolveSampling, generation settings; parseJsonFloatOpt, parseJsonTopKOpt; max_tokens defaults, seed,
//! clamps, requestEosSlice; clampMaxTokens, optSamplingRecJson; repeat penalty, kv_quant override

const std = @import("std");
const kv_quant_mod = @import("../../sdk/blocks/attention/kv_quant.zig");
const cache = @import("../../sdk/kv/cache.zig");
const generation_settings = @import("generation_settings.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const model_registry_mod = @import("../registry/registry.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const model_settings_mod = @import("../registry/model_settings.zig");
const testing = std.testing;
const http_server = @import("../http/server.zig");
const request_model = @import("request_model.zig");

const autoBudgetWindowTight = @import("../chat/tools/pipeline.zig").autoBudgetWindowTight;

const Sampling = struct { temperature: f32, top_p: f32, top_k: u32, min_p: ?f32, repeat_penalty: f32, presence_penalty: f32, frequency_penalty: f32 };

/// Sampling for one request: a forced generation-defaults rule > request body > rule >
/// CLI launch flag > model generation_config.json > hardcoded fallback. An explicit request
/// value of 0 (greedy / disabled) is a value, not an omission.
pub fn resolveSampling(gen: generation_settings.Profile, root: std.json.ObjectMap, config: *const model_mod.ModelConfig) Sampling {
    return .{
        .temperature = gen.resolve(f32, .temperature, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), http_server.serverState().server_config.default_temperature orelse config.gen_temperature orelse 1.0),
        .top_p = gen.resolve(f32, .top_p, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), http_server.serverState().server_config.default_top_p orelse config.gen_top_p orelse 1.0),
        .top_k = gen.resolve(u32, .top_k, parseJsonTopKOpt(root, "top_k"), http_server.serverState().server_config.default_top_k orelse config.gen_top_k orelse 0),
        .min_p = gen.resolveOpt(f32, .min_p, parseJsonFloatOpt(root, "min_p", 0.0, 1.0), config.gen_min_p),
        .repeat_penalty = requestRepeatPenalty(root, gen),
        .presence_penalty = gen.resolve(f32, .presence_penalty, parseJsonFloatOpt(root, "presence_penalty", 0.0, 2.0), 0.0),
        .frequency_penalty = gen.resolve(f32, .frequency_penalty, parseJsonFloatOpt(root, "frequency_penalty", 0.0, 2.0), 0.0),
    };
}

/// A zero rule means remaining context; an omitted rule uses the surface's fallback.
pub fn generationMaxTokens(gen: generation_settings.Profile, v: ?std.json.Value, fallback: u32) u32 {
    const req = resolveRequestMaxTokens(v, 0);
    const cap = gen.resolve(u32, .max_tokens, if (req > 0) req else null, fallback);
    return if (cap == 0 and gen.rules.get(.max_tokens) != null) AUTO_MAX_TOKENS_SENTINEL else cap;
}

pub var g_generation_settings: model_settings_mod.Cache = .{ .path = "" };

pub var g_generation_settings_path: [std.fs.max_path_bytes]u8 = undefined;

/// A request's generation defaults: the model's rules over the global file's, re-read when
/// either file changes.
pub fn requestProfile(io: std.Io, lm: *const LoadedModel) generation_settings.Profile {
    return .overlay(g_generation_settings.generationDefaults(io, null), request_model.g_model_aliases.generationDefaults(io, lm.path));
}

/// parseJsonFloat variant that distinguishes "omitted / wrong type" (null)
/// from an explicit value, for fields whose default comes from the
/// resolution chain above.
fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8, min: f32, max: f32) ?f32 {
    const v = root.get(key) orelse return null;
    const raw: f32 = switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    return std.math.clamp(raw, min, max);
}

/// Optional top_k body-field parse (positive integer, capped at 1000).
/// null = omitted or unusable; explicit 0 means "disable top-k".
fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = root.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
        else => null,
    };
}

/// Default when an OpenAI-style client omits `max_tokens` /
/// `max_completion_tokens` / `max_output_tokens`. OpenAI semantics: omitted
/// means "generate until EOS, bounded by the context window" — NOT a small
/// fixed cap. (The old 256 default silently broke agent clients like pi:
/// every thinking-enabled turn hit `length` mid-reasoning, so no tool call or
/// answer ever came back.) The sentinel is huge so the downstream
/// `clampMaxTokens` resolves it to the remaining context; only when the context
/// is genuinely unknown (0) do we fall back to a finite 4096.
///
/// Takes the EFFECTIVE context, not `server_config.max_context_size`: that
/// global is set only by `--ctx-size`, so under auto-context reading it would
/// return 4096 — an omitted-max_tokens client silently capped at 4096 tokens,
/// truncating any large tool call. Same class as the 256 default it replaced.
pub fn omittedMaxTokensDefault(effective_ctx: u32) u32 {
    return omittedMaxTokensDefaultWith(effective_ctx, launchMaxTokensDefault());
}

/// Pure core of the above with the launch default (`--max-tokens N`, 0 = unset) passed in.
/// The flag outranks both fallbacks.
fn omittedMaxTokensDefaultWith(effective_ctx: u32, launch_default: u32) u32 {
    if (launch_default > 0) return launch_default;
    return if (effective_ctx > 0) AUTO_MAX_TOKENS_SENTINEL else 4096;
}

/// The "no budget named, peg it to the window" sentinel, in one place; `clampMaxTokens` reduces it to `context - prompt`.
const AUTO_MAX_TOKENS_SENTINEL: u32 = std.math.maxInt(u32) / 4;

/// The one read of `ServerConfig.default_max_tokens`. 0 = flag not given.
pub fn launchMaxTokensDefault() u32 {
    return http_server.serverState().server_config.default_max_tokens;
}

/// Resolve a request's `max_tokens` (or its aliases) to an effective cap.
/// Absent OR `<= 0` means **auto**: peg generation to the remaining context
/// window via `auto_default` (the `omittedMaxTokensDefault` sentinel, which
/// `clampMaxTokens` then reduces to `context - prompt`). The app's "Auto"
/// setting sends 0 (or omits the field); both land on the context-pegged budget
/// instead of a fixed ceiling. A positive integer is an explicit cap.
pub fn resolveRequestMaxTokens(v: ?std.json.Value, auto_default: u32) u32 {
    const val = v orelse return auto_default;
    return switch (val) {
        .integer => |i| if (i > 0) @intCast(@min(i, std.math.maxInt(u32))) else auto_default,
        else => auto_default,
    };
}

/// A request's `seed`; negative is the llama.cpp "random" convention and means unseeded.
pub fn parseRequestSeed(v: ?std.json.Value) ?u64 {
    const val = v orelse return null;
    return switch (val) {
        .integer => |i| if (i < 0) null else @as(u64, @intCast(i)),
        else => null,
    };
}

/// A client-supplied JSON integer narrowed to i32 by clamping, never by `@intCast`.
pub fn clampJsonI32(i: i64) i32 {
    return @intCast(std.math.clamp(i, std.math.minInt(i32), std.math.maxInt(i32)));
}

/// Who decided the effective `max_tokens`: the client, the `--max-tokens` launch default, or nobody (auto).
const MaxTokensOrigin = enum { client, launch_default, auto };

/// The origin of the number `resolveRequestMaxTokens` returns, from the same two inputs.
pub fn maxTokensOrigin(requested: ?std.json.Value, launch_default: u32) MaxTokensOrigin {
    if (requested) |val| switch (val) {
        .integer => |i| if (i > 0) return .client,
        else => {},
    };
    return if (launch_default > 0) .launch_default else .auto;
}

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
pub const MAX_TOKENS_DESC_LEN: usize = 27;

/// Render the request log's `max_tokens=` field: auto prints as `auto` (or `auto (N)` for the
/// finite unknown-context fallback), a launch-filled budget names its source, a client's number
/// prints bare. The sentinel used to print as a client asking for a billion tokens.
pub fn describeMaxTokens(buf: []u8, value: u32, origin: MaxTokensOrigin) []const u8 {
    return switch (origin) {
        .client => std.fmt.bufPrint(buf, "{d}", .{value}) catch "?",
        .launch_default => std.fmt.bufPrint(buf, "{d} (launch default)", .{value}) catch "?",
        .auto => if (value == AUTO_MAX_TOKENS_SENTINEL)
            "auto"
        else
            std.fmt.bufPrint(buf, "auto ({d})", .{value}) catch "auto",
    };
}

/// Does the generation budget warrant the "budget squeezed" warning? The auto sentinel is
/// excluded: it is always more than 4x anything that remains.
fn maxTokensBudgetSqueezed(max_tokens: u32, remaining: u32) bool {
    if (max_tokens == AUTO_MAX_TOKENS_SENTINEL) return false;
    return remaining < max_tokens / 4;
}

/// vLLM `ignore_eos`: when true, EOS and stop tokens do not end the reply —
/// only `max_tokens` (or the request timeout) does. Stop sequences are a
/// separate field and stay active.
pub fn requestEosSlice(config: *const model_mod.ModelConfig, root: std.json.ObjectMap) []const u32 {
    if (root.get("ignore_eos")) |v| {
        if (v == .bool and v.bool) return &.{};
    }
    return config.eosTokenSlice();
}

/// Clamp max_tokens so prompt + completion doesn't exceed context length.
/// Clamp a request's `max_tokens` so `prompt + generation` stays inside
/// `effective_ctx`. `effective_ctx == 0` means "unknown" and imposes no limit.
///
/// Takes the context explicitly rather than reading `server_config.max_context_size`:
/// that global is only set by `--ctx-size`, so under AUTO context this function
/// used to return `max_tokens` untouched — the server never trimmed a client's
/// budget, never emitted the "budget squeezed" warning, and generation could run
/// past the memory-safe window. Callers pass `getEffectiveContextLength(config)`,
/// which is the same number the prompt-length guard and `/v1/models` report.
pub fn clampMaxTokens(max_tokens: u32, prompt_len: usize, effective_ctx: u32) u32 {
    if (effective_ctx == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, effective_ctx));
    if (prompt >= effective_ctx) return 1; // at least 1 token
    const remaining = effective_ctx - prompt;
    if (maxTokensBudgetSqueezed(max_tokens, remaining)) {
        log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, effective_ctx });
    } else if (max_tokens == AUTO_MAX_TOKENS_SENTINEL and autoBudgetWindowTight(remaining, effective_ctx)) {
        // The client named no budget, so this is context, not a deviation: debug.
        log.debug("  generation budget squeezed: {d} of the window remain (auto budget; prompt={d}, ctx={d})\n", .{ remaining, prompt, effective_ctx });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx={d}, prompt={d})\n", .{ max_tokens, remaining, effective_ctx, prompt });
        return remaining;
    }
    return max_tokens;
}

/// Render an optional model-author sampling recommendation (from the model's
/// generation_config.json) as a JSON scalar: the number when present, the
/// literal `null` when the model ships no value. Caller owns the slice.
/// Used for the `gen_temperature`/`gen_top_p`/`gen_top_k` meta fields the
/// Swift Settings UI reads to show "model recommends" guidance pills.
pub fn optSamplingRecJson(allocator: std.mem.Allocator, comptime T: type, v: ?T) ![]u8 {
    if (v) |val| return std.fmt.allocPrint(allocator, "{d}", .{val});
    return allocator.dupe(u8, "null");
}

/// The multiplicative `repeat_penalty` (llama.cpp's); a value <= 0 is an omission.
fn requestRepeatPenalty(root: std.json.ObjectMap, gen: generation_settings.Profile) f32 {
    const repeat = parseJsonFloat(root, "repeat_penalty", 0.0, 0.0, 10.0);
    return gen.resolve(f32, .repeat_penalty, if (repeat > 0) repeat else null, 1.0);
}

/// Parse a float from a JSON value, clamping to [min, max]. Returns default if missing/invalid.
fn parseJsonFloat(root: std.json.ObjectMap, key: []const u8, default: f32, min: f32, max: f32) f32 {
    const raw = if (root.get(key)) |v| switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => default,
    } else default;
    return std.math.clamp(raw, min, max);
}

/// Wave 1.A — parse the optional per-request `kv_quant` body field. Accepts
/// the string forms `"off"`, `"0"`, `"4"`, `"8"` and the integer forms `0`,
/// `4`, `8`. Returns null when the field is absent or unrecognized (the
/// caller falls back to the process-level `--kv-quant` default carried on
/// the scheduler). Returns `KVQuantConfig.dense` for "off"/0.
pub fn parseKvQuantOverride(root: std.json.ObjectMap) ?kv_quant_mod.KVQuantConfig {
    const v = root.get("kv_quant") orelse return null;
    return kv_quant_mod.KVQuantConfig.fromJsonValue(v);
}

// ── Vision Processing ──

// ── tests ──

test "requestEosSlice: ignore_eos swaps the EOS list for empty" {
    var config = model_mod.ModelConfig{};
    config.addEosToken(151645);
    config.addEosToken(151643);
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const parsed_off = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"ignore_eos\": true}", .{});
    defer parsed_off.deinit();
    try std.testing.expectEqual(@as(usize, 0), requestEosSlice(&config, parsed_off.value.object).len);
    const parsed_on = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"ignore_eos\": false}", .{});
    defer parsed_on.deinit();
    try std.testing.expectEqual(@as(usize, 2), requestEosSlice(&config, parsed_on.value.object).len);
    const parsed_absent = try std.json.parseFromSlice(std.json.Value, arena_state.allocator(), "{\"max_tokens\": 64}", .{});
    defer parsed_absent.deinit();
    try std.testing.expectEqual(@as(usize, 2), requestEosSlice(&config, parsed_absent.value.object).len);
}

test "parseJsonFloat returns value when present" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":0.7}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), result, 0.001);
}

test "parseJsonFloat returns default when missing" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "parseJsonFloat clamps to range" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":5.0}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result, 0.001);
}

test "parseJsonFloat handles integer value" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":1}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 0.5, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "clampMaxTokens honors the effective context (auto-pinned, not just --ctx-size)" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 0; // auto: the old code never clamped here

    // 4096-token window, 4000-token prompt -> only 96 tokens of generation left.
    try testing.expectEqual(@as(u32, 96), clampMaxTokens(8192, 4000, 4096));
    // Comfortable budget is passed through untouched.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 100, 65536));
    // Prompt at/over the window still yields at least one token.
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 4096, 4096));
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 9000, 4096));
    // ctx 0 (unknown) = no limit.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 4000, 0));
}

test "omittedMaxTokensDefault: context-bound when ctx is known, finite fallback otherwise" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;

    // OpenAI semantics: omitted max_tokens = generate until EOS bounded by the
    // context window. The old fixed 256 default broke agent clients (pi) whose
    // thinking-enabled turns hit `length` mid-reasoning on EVERY request.
    http_server.serverState().server_config.max_context_size = 32768;
    const sentinel = omittedMaxTokensDefault(32768);
    // Big enough that clampMaxTokens (the downstream bound) always wins…
    try testing.expect(sentinel > 32768);
    // …and the composition resolves to exactly the remaining context.
    try testing.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(sentinel, 1500, 32768));

    // ctx unknown (max_context_size=0): clampMaxTokens won't bound anything,
    // so the default itself must stay finite.
    http_server.serverState().server_config.max_context_size = 0;
    try testing.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(0));
}

test "resolveRequestMaxTokens: absent / 0 / negative / non-int → auto; positive → value" {
    const auto: u32 = 12345;
    // Absent → auto (the omitted-default path the app's "Auto" setting rides).
    try testing.expectEqual(auto, resolveRequestMaxTokens(null, auto));
    // 0 and negatives are "auto" too — a client must not be able to request a
    // 0-token (generate-nothing) response by selecting Auto.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = 0 }, auto));
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = -5 }, auto));
    // Wrong type (e.g. a string) → auto rather than crashing.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .string = "999" }, auto));
    // A real positive cap is honored verbatim.
    try testing.expectEqual(@as(u32, 4096), resolveRequestMaxTokens(.{ .integer = 4096 }, auto));
}

test "--max-tokens is the omitted-field default: unset is byte-identical, set wins" {
    const t = std.testing;
    // With no flag the omitted-field default is byte-for-byte what it was.
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefaultWith(32768, 0));
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefaultWith(0, 0));

    // Flag set: the launch value outranks both fallbacks.
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(786_432, 65536));
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(0, 65536));

    // The context clamp still sits on top.
    try t.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(omittedMaxTokensDefaultWith(32768, 65536), 1500, 32768));

    // A request that sends max_tokens is untouched by the flag.
    const auto = omittedMaxTokensDefaultWith(786_432, 65536);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, auto));
    try t.expectEqual(@as(u32, 200_000), resolveRequestMaxTokens(.{ .integer = 200_000 }, auto));
    try t.expectEqual(@as(u32, 65536), resolveRequestMaxTokens(.{ .integer = 0 }, auto));

    // The live wrapper reads the ServerConfig field through the one accessor.
    const slot = &@field(http_server.serverState().server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    try t.expectEqual(@as(u32, 0), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefault(32768));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(32768));
}

test "describeMaxTokens: an omitted max_tokens logs as auto, never as the sentinel" {
    const t = std.testing;
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;

    try t.expectEqualStrings("auto", describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .auto));

    // With an unknown context the auto budget is a finite 4096 that really caps generation.
    try t.expectEqualStrings("auto (4096)", describeMaxTokens(&buf, 4096, .auto));

    // `--max-tokens N` filled an omitted field: the number is real, but it is
    // the operator's, not the client's, and the line says whose it is.
    try t.expectEqualStrings("65536 (launch default)", describeMaxTokens(&buf, 65536, .launch_default));

    // A client that sent the field gets its own number back, unadorned.
    try t.expectEqualStrings("4096", describeMaxTokens(&buf, 4096, .client));

    // Origin comes from the SAME inputs `resolveRequestMaxTokens` reads, so
    // the label can't disagree with the number beside it.
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(null, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = 0 }, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = -5 }, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .string = "999" }, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 65536));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(.{ .integer = 0 }, 65536));
    // A sent field outranks the flag, so it is the CLIENT's number either way.
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 65536));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0));

    try t.expectEqualStrings(
        "1073741823 (launch default)",
        describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .launch_default),
    );
}

test "the max_tokens squeeze warning is silent for the auto budget, loud for a real over-ask" {
    const t = std.testing;

    // A real over-ask: the client named a number and the window can't pay it.
    // That is the warning's whole point (the tool-call truncation nudge).
    try t.expect(maxTokensBudgetSqueezed(32768, 4000));
    try t.expect(!maxTokensBudgetSqueezed(32768, 30000));

    // The sentinel is always > 4x whatever remains; there is nothing to be short of.
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 4000));
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 1));

    try t.expect(autoBudgetWindowTight(4000, 32768));
    try t.expect(!autoBudgetWindowTight(30000, 32768));
    // A launch default is a real number and takes the real predicate.
    try t.expect(maxTokensBudgetSqueezed(65536, 4000));
}

test "an omitted max_tokens reserves the LAUNCH default, not the whole window" {
    const t = std.testing;
    // The admission reservation reads the resolved budget, so the flag reaches it.
    const KVCache = cache.KVCache;
    const ctx: u64 = 786_432;
    const seq: u64 = 400_000; // past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;

    // No flag: the sentinel becomes the whole remaining window; only the headroom cap keeps that off the bill.
    const no_flag = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 0), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, @intCast(ctx - seq)), no_flag);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // Flag set below the headroom: the reservation is prompt + the flag.
    const flagged = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 2048), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 2048), flagged);
    try t.expectEqual(seq + 2048 + chunk, KVCache.reservedTokens(seq, flagged, chunk, ctx));
    try t.expect(KVCache.reservedTokens(seq, flagged, chunk, ctx) < KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // A flag above the headroom bills only the headroom.
    const big = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 65536), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 65536), big);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, big, chunk, ctx));
}

test "clampMaxTokens no limit when ctx_size=0" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 0;

    try testing.expectEqual(@as(u32, 1000), clampMaxTokens(1000, 500, 0));
}

test "clampMaxTokens clamps when would exceed" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, 3000, 4096));
}

test "clampMaxTokens no clamp when fits" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, 100, 4096));
}

test "clampMaxTokens at boundary" {
    const original = http_server.serverState().server_config.max_context_size;
    defer http_server.serverState().server_config.max_context_size = original;
    http_server.serverState().server_config.max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4096, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4095, 4096));
}

test "resolveSampling: forced rule > request > rule > generation_config > fallback" {
    const a = std.testing.allocator;
    const zeros = try std.json.parseFromSlice(std.json.Value, a, "{\"temperature\":0,\"top_k\":0,\"min_p\":0,\"frequency_penalty\":0}", .{});
    defer zeros.deinit();
    const config = model_mod.ModelConfig{ .gen_temperature = 0.8, .gen_top_p = 0.95, .gen_top_k = 40, .gen_min_p = 0.1 };
    var got = resolveSampling(.{}, .empty, &config);
    try std.testing.expectEqual(@as(f32, 0.8), got.temperature);
    try std.testing.expectEqual(@as(u32, 40), got.top_k);
    try std.testing.expectEqual(@as(?f32, 0.1), got.min_p);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
    got = resolveSampling(.{}, .empty, &.{});
    try std.testing.expectEqual(@as(f32, 1.0), got.temperature);
    try std.testing.expectEqual(@as(?f32, null), got.min_p);
    var gen = generation_settings.Profile{};
    gen.set(.temperature, .{ .number = 0.25 }, false);
    gen.set(.min_p, .{ .number = 0.05 }, true);
    gen.set(.repeat_penalty, .{ .number = 1.5 }, false);
    got = resolveSampling(gen, .empty, &config);
    try std.testing.expectEqual(@as(f32, 0.25), got.temperature);
    try std.testing.expectEqual(@as(f32, 1.5), got.repeat_penalty);
    // An explicit zero is a value: it beats an unforced rule, never a forced one.
    got = resolveSampling(gen, zeros.value.object, &config);
    try std.testing.expectEqual(@as(f32, 0.0), got.temperature);
    try std.testing.expectEqual(@as(u32, 0), got.top_k);
    try std.testing.expectEqual(@as(?f32, 0.05), got.min_p);
    try std.testing.expectEqual(@as(f32, 1.5), got.repeat_penalty);
    try std.testing.expectEqual(@as(f32, 0.0), got.frequency_penalty);
    gen.set(.repeat_penalty, .{ .number = 1.0 }, false);
    gen.set(.frequency_penalty, .{ .number = 0.5 }, false);
    got = resolveSampling(gen, .empty, &config);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
    try std.testing.expectEqual(@as(f32, 0.5), got.frequency_penalty);
    gen.set(.repeat_penalty, .{ .number = 1.2 }, true);
    try std.testing.expectEqual(@as(f32, 1.2), resolveSampling(gen, zeros.value.object, &config).repeat_penalty);
}

test "OpenAI frequency_penalty is its own additive penalty, never a repeat_penalty multiplier (#912)" {
    const a = std.testing.allocator;
    const body = try std.json.parseFromSlice(std.json.Value, a, "{\"frequency_penalty\":1,\"presence_penalty\":0}", .{});
    defer body.deinit();
    const got = resolveSampling(.{}, body.value.object, &.{});
    try std.testing.expectEqual(@as(f32, 1.0), got.frequency_penalty);
    try std.testing.expectEqual(@as(f32, 1.0), got.repeat_penalty);
}

test "generationMaxTokens: an auto request (0, negative, omitted) takes the rule, a forced rule takes everything" {
    var gen = generation_settings.Profile{};
    try std.testing.expectEqual(@as(u32, 7), generationMaxTokens(gen, .{ .integer = -1 }, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 7));
    gen.set(.max_tokens, .{ .number = 12 }, false);
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, .{ .integer = -1 }, 7));
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, null, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 7));
    gen.set(.max_tokens, .{ .number = 0 }, false); // 0 = auto
    try std.testing.expectEqual(AUTO_MAX_TOKENS_SENTINEL, generationMaxTokens(gen, null, 7));
    try std.testing.expectEqual(@as(u32, 64), generationMaxTokens(gen, .{ .integer = 64 }, 0));
    gen.set(.max_tokens, .{ .number = 0 }, true);
    for ([_]?std.json.Value{ null, .{ .integer = 0 }, .{ .integer = -1 }, .{ .integer = 64 } }) |value| {
        const cap = generationMaxTokens(gen, value, 0);
        try std.testing.expectEqual(AUTO_MAX_TOKENS_SENTINEL, cap);
        try std.testing.expectEqual(@as(u32, 8092), clampMaxTokens(cap, 100, 8192));
    }
    try std.testing.expectEqual(@as(u32, 0), generationMaxTokens(.{}, null, 0));
    gen.set(.max_tokens, .{ .number = 12 }, true);
    try std.testing.expectEqual(@as(u32, 12), generationMaxTokens(gen, .{ .integer = 64 }, 7));
}

test "optSamplingRecJson emits number when present, null when absent" {
    const a = std.testing.allocator;

    // Present -> bare JSON number (no quotes), so the Swift side decodes it
    // straight to Double/Int.
    const top_k = try optSamplingRecJson(a, u32, 20);
    defer a.free(top_k);
    try std.testing.expectEqualStrings("20", top_k);

    const top_p = try optSamplingRecJson(a, f32, 0.95);
    defer a.free(top_p);
    try std.testing.expectEqualStrings("0.95", top_p);

    // Absent -> JSON null literal so the model-author recommendation reads as
    // "no opinion" rather than a spurious 0.
    const none = try optSamplingRecJson(a, u32, null);
    defer a.free(none);
    try std.testing.expectEqualStrings("null", none);
}

test "a request integer from a client is clamped, never blindly @intCast" {
    // Bar: `seed: -1`, a 2^32 max_tokens and an out-of-range budget resolve without a cast.
    const t = std.testing;
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .integer = -1 }));
    try t.expectEqual(@as(?u64, 0), parseRequestSeed(.{ .integer = 0 }));
    try t.expectEqual(@as(?u64, 42), parseRequestSeed(.{ .integer = 42 }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .string = "42" }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(null));

    try t.expectEqual(@as(u32, std.math.maxInt(u32)), resolveRequestMaxTokens(.{ .integer = 1 << 32 }, 7));
    try t.expectEqual(@as(u32, 512), resolveRequestMaxTokens(.{ .integer = 512 }, 7));
    try t.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .integer = 0 }, 7));

    try t.expectEqual(@as(i32, std.math.maxInt(i32)), clampJsonI32(1 << 40));
    try t.expectEqual(@as(i32, std.math.minInt(i32)), clampJsonI32(-(1 << 40)));
    try t.expectEqual(@as(i32, -1), clampJsonI32(-1));
    try t.expectEqual(@as(i32, 1024), clampJsonI32(1024));
}
