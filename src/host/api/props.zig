//! /props, batching, settings, ANE stats, status page

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const decode_attn_quant_mod = @import("../../sdk/blocks/attention/decode_attn_quant.zig");
const mtp_acceptance_mod = @import("../../sdk/spec/mtp_acceptance.zig");
const model_mod = @import("../../sdk/config/model.zig");
const scheduler_mod = @import("../gpu_thread/scheduler.zig");
const mlx_engine = @import("../engines/mlx.zig");
const model_registry_mod = @import("../registry/registry.zig");
const metrics = @import("../status.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const build_options = @import("build_options");
const testing = std.testing;
const http_server = @import("../http/server.zig");

const Conn = @import("../http/conn.zig").Conn;
const KvAttnMode = @import("request_options.zig").KvAttnMode;
const defaultEnableMtp = @import("request_options.zig").defaultEnableMtp;
const dsv4DraftStages = @import("request_options.zig").dsv4DraftStages;
const configuredKvQuantFor = @import("../http/server.zig").configuredKvQuantFor;
const resolvedPrefixCacheMem = @import("admission.zig").resolvedPrefixCacheMem;
const PldDefaults = @import("request_options.zig").PldDefaults;
const getEffectiveContextLength = @import("admission.zig").getEffectiveContextLength;
const computeMaxSafeContext = @import("admission.zig").computeMaxSafeContext;
const sendResponse = @import("../http/respond.zig").sendResponse;

/// Pure body-builder for `GET /props`. Split out from `handleProps` so it
/// can be unit-tested without standing up a real `LoadedModel`. Caller owns
/// the returned slice.
///
/// Note: the `chat_template` field that upstream llama-server exposes here
/// is intentionally omitted. Our Swift app never read it, and stock chat
/// templates are 10–100 KB of Jinja — that turned every /props poll into
/// wasted bandwidth. Capabilities still come from `/v1/models` per entry;
/// clients that genuinely need the template can still get it from the
/// model's `config.json` / `chat_template.jinja` on disk, or by rendering
/// through `/v1/chat/completions` server-side.
fn renderPropsBody(
    allocator: std.mem.Allocator,
    config: *const model_mod.ModelConfig,
    ctx_str: []const u8,
    active_mem: usize,
    peak_mem: usize,
    available_mem: u64,
    safe_ctx: u32,
    cache_mem: usize,
    /// Registry residency of every ready model, and the KV of the hot prefix cache + live slots.
    weights_mem: u64,
    kv_cache_mem: u64,
    /// Leading-comma JSON fragments spliced before the root close (the ANE
    /// object, the qwen4 n-gram warm object). Concatenated by the handler.
    extra_json: []const u8,
) ![]u8 {
    // `available_bytes` is free SYSTEM RAM, computed with the SAME formula the
    // model-load pre-flight uses (`metrics.getAvailableMemBytes`), so the tray's
    // "Free RAM" line can never drift from the number that gates a load. Distinct
    // axis from `active_bytes` (the MLX GPU-allocator footprint).
    //
    // `cache_bytes` is MLX's reclaimable buffer pool — memory the process HOLDS
    // but is not using. It is a third axis again, and its absence is why #110
    // was invisible: the panel read 19.6 GB of `active_bytes` while the process
    // sat at 81.4 GB, and nothing we served named the other 61.
    return std.fmt.allocPrint(allocator,
        \\{{"default_generation_settings":{{"model":"{s}","n_ctx":{s}}},"total_slots":1,"model_info":{{"vocab_size":{d},"hidden_size":{d},"num_hidden_layers":{d},"num_attention_heads":{d},"num_key_value_heads":{d},"head_dim":{d},"quantization_bits":{d},"quantization_group_size":{d},"max_position_embeddings":{d}}},"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":{d},"cache_bytes":{d},"weights_bytes":{d},"kv_cache_bytes":{d}}}{s}}}
    , .{
        config.model_type,              ctx_str,
        config.vocab_size,              config.hidden_size,
        config.num_hidden_layers,       config.num_attention_heads,
        config.num_key_value_heads,     config.head_dim,
        config.quant_bits,              config.quant_group_size,
        config.max_position_embeddings, active_mem,
        peak_mem,                       available_mem,
        safe_ctx,                       cache_mem,
        weights_mem,                    kv_cache_mem,
        extra_json,
    });
}

/// The model-level half of `Scheduler.batchVerdict`: does this loaded model
/// batch decode at all? Per-slot arms (spec, grammar, logprobs) come later.
pub fn batchVerdictFor(entry: *const LoadedModel) scheduler_mod.BatchVerdict {
    // An engine beside MLX batches its own requests where it says so (llama.cpp's sequences).
    if (entry.engine) |e| return if (e.caps().batches) .ok else .embedded_engine;
    const cfg = entry.config orelse return .arch;
    return if (mlx_engine.api().batches_decode(cfg)) .ok else .arch;
}

/// The /props "batching" object: whether the loaded model rides the batched
/// decode kernel, and why not when it does not.
fn batchingPropsJson(allocator: std.mem.Allocator, why: scheduler_mod.BatchVerdict) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"batching\":{{\"supported\":{s},\"reason\":\"{s}\",\"max_group\":{d}}}", .{
        if (why == .ok) "true" else "false",
        @tagName(why),
        scheduler_mod.MAX_BATCH_GROUP,
    });
}

/// The /props "ngram_warm" object; absent when no table is warming.
pub fn ngramWarmPropsJson(allocator: std.mem.Allocator, bytes: u64, total: u64) ![]u8 {
    if (total == 0) return allocator.dupe(u8, "");
    return std.fmt.allocPrint(allocator, ",\"ngram_warm\":{{\"bytes\":{d},\"total\":{d}}}", .{ bytes, total });
}

/// Effective per-model serving settings, resolved the way a request that sends no overrides sees them.
const PropsSettings = struct {
    engine: []const u8,
    kv_quant: []const u8,
    kv_attn_mode: KvAttnMode,
    decode_attn_quant: bool,
    prefill_chunk: usize,
    mtp_loaded: bool,
    mtp_default_on: bool,
    mtp_acceptance: mtp_acceptance_mod.Mode,
    mtp_greedy_tail: bool = false,
    /// 0 = auto.
    mtp_depth: u32,
    mtp_adaptive: bool,
    native_speculation: ?struct { lane: []const u8, block_size: u32, sampled: bool } = null,
    /// 0 = no ceiling.
    max_mtp_ctx: u32,
    drafter: []const u8,
    pld: PldDefaults,
    max_concurrent: u32,
    prefix_cache_mem_bytes: u64,
    prefix_cache_disk_bytes: u64,
    prefix_cache_ram_enabled: bool = true,
    /// `--prefill-decode-share`: decode's target wall-time fraction during another slot's prefill.
    prefill_decode_share: f32 = 0,
};

const PropsEngine = @import("../engines/served.zig").Kind;

/// ds4 and llama.cpp bypass generate.zig: MLX decode levers, PLD and the MLX drafters never run there.
fn embeddedEngineSettings(st: PropsSettings, engine: PropsEngine, engine_mtp: bool) PropsSettings {
    if (engine == .mlx) return st;
    var out = st;
    out.engine = @tagName(engine);
    if (engine == .ds4) out.kv_quant = "off";
    out.decode_attn_quant = false;
    out.prefill_chunk = 0;
    out.mtp_loaded = engine_mtp;
    out.mtp_default_on = engine_mtp;
    out.mtp_adaptive = false;
    out.drafter = "none";
    out.pld = PldDefaults.off;
    out.prefill_decode_share = 0;
    return out;
}

fn propsSettingsFor(lm: *LoadedModel) PropsSettings {
    const engine: PropsEngine = if (lm.engine) |e| e.kind else .mlx;
    // The Neural Engine drafts with a DFlash drafter, not an MTP head.
    const engine_mtp = if (lm.engine) |e| e.kind != .ane and e.caps().drafts else false;
    var out = embeddedEngineSettings(mlxPropsSettings(lm), engine, engine_mtp);
    if (engine == .ane) out.kv_quant = "off";
    return out;
}

fn engineIs(lm: *const LoadedModel, kind: PropsEngine) bool {
    const e = lm.engine orelse return false;
    return e.kind == kind;
}

pub fn mlxPropsSettings(lm: *LoadedModel) PropsSettings {
    const config = lm.config.?;
    const kv = configuredKvQuantFor(config);
    const mp = mlx_engine.modelProps(lm);
    const es = mlx_engine.settings();
    return .{
        .engine = "mlx",
        .kv_quant = if (engineIs(lm, .llama)) @tagName(http_server.llama_settings.kv_quant) else if (kv.isQuant()) (if (kv.bits == 4) "4" else "8") else "off",
        .kv_attn_mode = http_server.serverState().server_config.kv_attn_mode,
        .decode_attn_quant = mp.decode_attn_quant,
        .prefill_chunk = es.prefill_chunk,
        .mtp_loaded = mp.mtp_loaded,
        .mtp_default_on = defaultEnableMtp(lm.mtp != null, dsv4DraftStages(lm)),
        .mtp_acceptance = mp.mtp_acceptance,
        .mtp_greedy_tail = mp.mtp_greedy_tail,
        .mtp_depth = lm.mtp_depth,
        .mtp_adaptive = true,
        .native_speculation = if (mp.native_lane.len > 0) .{
            .lane = mp.native_lane,
            .block_size = mp.native_block,
            .sampled = true,
        } else null,
        .max_mtp_ctx = es.max_mtp_ctx,
        .drafter = if (lm.dflash != null) "dflash" else if (lm.drafter != null) "assistant" else "none",
        .pld = .{ .enable = http_server.serverState().server_config.default_enable_pld, .draft_len = http_server.serverState().server_config.default_pld_draft_len, .key_len = http_server.serverState().server_config.default_pld_key_len },
        .max_concurrent = http_server.max_concurrent,
        .prefix_cache_mem_bytes = resolvedPrefixCacheMem(),
        .prefix_cache_disk_bytes = http_server.launch().prefix_cache_disk_bytes,
        .prefix_cache_ram_enabled = http_server.launch().prefix_cache_capacity > 0 and http_server.launch().prefix_cache_ram_enabled,
        .prefill_decode_share = es.prefill_decode_share,
    };
}

/// The /props "settings" object: leading-comma fragment spliced before the props root close.
pub fn settingsPropsJson(allocator: std.mem.Allocator, st: PropsSettings) ![]u8 {
    var param_buf: [32]u8 = undefined;
    const param: []const u8 = switch (st.mtp_acceptance) {
        .exact => "null",
        .typical => |t| try std.fmt.bufPrint(&param_buf, "{d}", .{t.delta}),
        .tokenv3 => |a| try std.fmt.bufPrint(&param_buf, "{d}", .{a}),
    };
    var mtp_buf: [512]u8 = undefined;
    const mtp_json = if (st.native_speculation) |native|
        try std.fmt.bufPrint(&mtp_buf, "{{\"loaded\":{},\"default_on\":{},\"acceptance\":null,\"acceptance_param\":null,\"greedy_tail\":null,\"depth\":null,\"adaptive\":null,\"max_ctx\":{d},\"native\":{{\"lane\":\"{s}\",\"block_size\":{d},\"sampled_acceptance\":\"{s}\"}}}}", .{
            st.mtp_loaded, st.mtp_default_on, st.max_mtp_ctx, native.lane, native.block_size, if (native.sampled and native.block_size > 0) "stochastic" else "off",
        })
    else
        try std.fmt.bufPrint(&mtp_buf, "{{\"loaded\":{},\"default_on\":{},\"acceptance\":\"{s}\",\"acceptance_param\":{s},\"greedy_tail\":{},\"depth\":{d},\"adaptive\":{},\"max_ctx\":{d}}}", .{
            st.mtp_loaded, st.mtp_default_on, mtp_acceptance_mod.name(st.mtp_acceptance), param, st.mtp_greedy_tail, st.mtp_depth, st.mtp_adaptive, st.max_mtp_ctx,
        });
    return std.fmt.allocPrint(allocator, ",\"settings\":{{\"version\":\"{s}\",\"engine\":\"{s}\",\"kv_quant\":\"{s}\",\"kv_attn_mode\":\"{s}\",\"decode_attn_quant\":{},\"prefill_chunk\":{d},\"mtp\":{s},\"drafter\":\"{s}\",\"pld\":{{\"default_on\":{},\"draft_len\":{d},\"key_len\":{d}}},\"max_concurrent\":{d},\"prefill_decode_share\":{d:.2},\"prefix_cache\":{{\"ram_enabled\":{},\"mem_bytes\":{d},\"disk_bytes\":{d}}}}}", .{
        build_options.version,     st.engine,
        st.kv_quant,               @tagName(st.kv_attn_mode),
        st.decode_attn_quant,      st.prefill_chunk,
        mtp_json,                  st.drafter,
        st.pld.enable,             st.pld.draft_len,
        st.pld.key_len,            st.max_concurrent,
        st.prefill_decode_share,   st.prefix_cache_ram_enabled,
        st.prefix_cache_mem_bytes, st.prefix_cache_disk_bytes,
    });
}

/// The /props "ane" object (A8): mode, coverage, geometry and the int8
/// bill of the resident ANE prefill engine, so "what is the Neural Engine
/// holding" is answerable without log-grepping. Pure — the handler feeds
/// it the engine's fields; returns a leading-comma fragment spliced before
/// the props root close (empty when there is no engine).
/// One ANE unit's dispatch evidence. Under dual this is the ONLY in-process
/// proof that both dies were addressed — a silently ignored affinity hint
/// looks identical from here, so the counters pair with the out-of-process
/// `macpow --dump | grep ANE0_` check, never replace it.
pub const AneUnitStat = struct { instance: i64, evals: u64, eval_failures: u64 };

pub fn anePropsJson(allocator: std.mem.Allocator, mode_name: []const u8, mlp_layers: usize, gdn_layers: usize, rows: u32, chunk_rows: u32, share: f32, int8_bytes: u64, units: []const AneUnitStat) ![]u8 {
    var evals: u64 = 0;
    var eval_failures: u64 = 0;
    for (units) |u| {
        evals += u.evals;
        eval_failures += u.eval_failures;
    }
    var rows_buf: [512]u8 = undefined;
    var used: usize = 0;
    for (units, 0..) |u, i| {
        const row = try std.fmt.bufPrint(rows_buf[used..], "{s}{{\"instance\":{d},\"evals\":{d},\"eval_failures\":{d}}}", .{ if (i == 0) "" else ",", u.instance, u.evals, u.eval_failures });
        used += row.len;
    }
    return std.fmt.allocPrint(allocator, ",\"ane\":{{\"mode\":\"{s}\",\"units\":{d},\"mlp_layers\":{d},\"gdn_layers\":{d},\"rows\":{d},\"chunk_rows\":{d},\"share\":{d:.2},\"int8_bytes\":{d},\"evals\":{d},\"eval_failures\":{d},\"unit_evals\":[{s}]}}", .{ mode_name, units.len, mlp_layers, gdn_layers, rows, chunk_rows, share, int8_bytes, evals, eval_failures, rows_buf[0..used] });
}

pub fn handleProps(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel) !void {
    const config = lm.config.?;
    const ctx_len = getEffectiveContextLength(config);
    const ctx_str = if (ctx_len > 0) blk: {
        break :blk try std.fmt.allocPrint(allocator, "{d}", .{ctx_len});
    } else try std.fmt.allocPrint(allocator, "0", .{});
    defer allocator.free(ctx_str);

    const safe_ctx = computeMaxSafeContext(config);

    // Query memory usage: mlx's allocator counters, or a GGUF engine's own figure.
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    // MLX's reclaimable buffer pool. Always read from MLX — the embedded
    // engines don't feed it, so it correctly reads ~0 on a ds4/llama session
    // rather than needing a per-engine branch.
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    if (lm.servesGguf()) {
        // A GGUF engine allocates outside MLX's counters: what it says it holds (ds4: the GGUF and its
        // session's KV/scratch; llama.cpp: the GGUF, its KV unexposed), so the app's memory bar is not 0/0.
        active_mem = @intCast(lm.engine.?.residentBytes());
        peak_mem = active_mem;
    } else {
        _ = mlx.mlx_get_active_memory(&active_mem);
        _ = mlx.mlx_get_peak_memory(&peak_mem);
    }

    // Free system RAM — same calc as the model-load pre-flight, so the app's
    // "Free RAM" line stays in lockstep with what gates a load.
    const available_mem = metrics.getAvailableMemBytes();

    // The engine's own fragments: Neural Engine prefill state, the n-gram table warm, then the cluster.
    const engine_json = mlx_engine.api().model_props_json(lm, allocator) orelse return error.OutOfMemory;
    defer allocator.free(engine_json);
    const batching_json = try batchingPropsJson(allocator, batchVerdictFor(lm));
    defer allocator.free(batching_json);
    const settings_json = try settingsPropsJson(allocator, propsSettingsFor(lm));
    defer allocator.free(settings_json);
    const cluster_json = mlx_engine.api().cluster_json(allocator) orelse return error.OutOfMemory;
    defer allocator.free(cluster_json);
    const extra_json = try std.fmt.allocPrint(allocator, "{s}{s}{s}{s},\"registry_generation\":{d}", .{ engine_json, batching_json, settings_json, cluster_json, registryGeneration() });
    defer allocator.free(extra_json);

    const kv_cache_mem: u64 = if (http_server.launch().global_scheduler) |sch|
        sch.resident_hot_cache_bytes.load(.monotonic) + sch.resident_live_kv_bytes.load(.monotonic)
    else
        0;
    const body = try renderPropsBody(allocator, config, ctx_str, active_mem, peak_mem, available_mem, safe_ctx, cache_mem, residentWeightsBytes(stream.io), kv_cache_mem, extra_json);
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// `ModelRegistry.generation`: moves with every model added and every load state change (not with
/// provider or LAN rows).
/// Moves whenever `/v1/models` would list something else: the registry or a LAN peer.
fn registryGeneration() u64 {
    const reg: u64 = if (http_server.global_registry) |r| r.generation.load(.monotonic) else 0;
    const lan: u64 = if (http_server.g_lan) |l| l.changes.load(.monotonic) else 0;
    return reg +% lan;
}

/// The registry's residency bill: the weights of every ready model, media ones included.
fn residentWeightsBytes(io: std.Io) u64 {
    const reg = http_server.global_registry orelse return 0;
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    return reg.current_resident_bytes;
}

/// Memory-only `/props` for a boot with no default chat model (headless
/// media-gen serving). Same `memory` object shape as `renderPropsBody` —
/// `MemoryInfo.parse` client-side reads only that key — with the model
/// fields omitted (there is no model config to describe).
pub fn handlePropsNoModel(allocator: std.mem.Allocator, stream: *Conn) !void {
    var active_mem: usize = 0;
    var peak_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    _ = mlx.mlx_get_peak_memory(&peak_mem);
    const available_mem = metrics.getAvailableMemBytes();
    const cluster_json = mlx_engine.api().cluster_json(allocator) orelse return error.OutOfMemory;
    defer allocator.free(cluster_json);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"total_slots":1,"memory":{{"active_bytes":{d},"peak_bytes":{d},"available_bytes":{d},"max_safe_context":0,"weights_bytes":{d}}},"batching":{{"supported":false,"reason":"no_model","max_group":{d}}}{s},"registry_generation":{d}}}
    , .{ active_mem, peak_mem, available_mem, residentWeightsBytes(stream.io), scheduler_mod.MAX_BATCH_GROUP, cluster_json, registryGeneration() });
    defer allocator.free(body);
    try sendResponse(stream, "200 OK", "application/json", body);
}

/// The built-in console at `GET /`: one self-contained page built from
/// `app-web/` (never edited by hand), served byte for byte.
///
/// Takes NO model. Everything model-shaped (the picker, capabilities, memory,
/// version, whether metrics are on) is fetched client-side from `/v1/models`,
/// `/props`, `/api/version` and `/metrics.json`, which is what lets the page
/// render on a server with nothing loaded — the default boot mode — and what
/// makes the picker follow loads/unloads without a refresh.
pub fn handleStatusPage(stream: *Conn) !void {
    try sendResponse(stream, "200 OK", "text/html; charset=utf-8", @embedFile("../../html/index.html"));
}

// ── tests ──

// --- /props payload regression ------------------------------------------------
//
// The `chat_template` field used to be emitted by /props for llama.cpp
// server-compat clients. Our own Swift app never read it, and stock chat
// templates are 10–100 KB of Jinja — that's wasted bandwidth on every poll.
// `renderPropsBody` is the pure body-builder behind `handleProps`; these
// tests pin the schema so a future "let's add it back" can't slip through
// without flipping these assertions intentionally.

test "renderPropsBody omits chat_template" {
    var config = model_mod.ModelConfig{};
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;
    config.model_type = "gemma4";

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 7777, 8888, "");
    defer testing.allocator.free(body);

    try testing.expect(std.mem.indexOf(u8, body, "\"chat_template\"") == null);
    // No ANE engine => no ane object, and the body is still valid JSON.
    try testing.expect(std.mem.indexOf(u8, body, "\"ane\"") == null);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    parsed.deinit();
}

test "anePropsJson: the /props ane object carries mode, coverage, the int8 bill and eval counts" {
    const one = [_]AneUnitStat{.{ .instance = 0, .evals = 112, .eval_failures = 0 }};
    const frag = try anePropsJson(testing.allocator, "channel", 64, 48, 8192, 8192, 0.45, 9_469_231_104, &one);
    defer testing.allocator.free(frag);
    try testing.expectEqualStrings(",\"ane\":{\"mode\":\"channel\",\"units\":1,\"mlp_layers\":64,\"gdn_layers\":48,\"rows\":8192,\"chunk_rows\":8192,\"share\":0.45,\"int8_bytes\":9469231104,\"evals\":112,\"eval_failures\":0,\"unit_evals\":[{\"instance\":0,\"evals\":112,\"eval_failures\":0}]}", frag);
    // Spliced into a props body it stays valid JSON with the object present.
    var config = model_mod.ModelConfig{};
    config.model_type = "qwen3_5_moe";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const ane = parsed.value.object.get("ane") orelse return error.MissingAne;
    try testing.expectEqualStrings("channel", ane.object.get("mode").?.string);
    try testing.expectEqual(@as(i64, 48), ane.object.get("gdn_layers").?.integer);
    // The M3 Ultra tester could not verify DISPATCH from /props (the
    // engagement lines live only in the log) — evals is the probe a bench
    // harness reads: zero with a green boot = built-but-never-dispatched.
    try testing.expectEqual(@as(i64, 112), ane.object.get("evals").?.integer);
    try testing.expectEqual(@as(i64, 0), ane.object.get("eval_failures").?.integer);

    // Dual: `evals` is the TOTAL, and `unit_evals` names each die. A
    // silently ignored affinity hint is invisible in-process — both units
    // report evals and both land on one ANE — so this row is what a tester
    // cross-checks against `macpow --dump | grep ANE0_`, which must show
    // BOTH counters moving.
    const two = [_]AneUnitStat{
        .{ .instance = 1, .evals = 112, .eval_failures = 0 },
        .{ .instance = 2, .evals = 112, .eval_failures = 3 },
    };
    const dual = try anePropsJson(testing.allocator, "channel", 64, 48, 8192, 8192, 0.45, 9_469_231_104, &two);
    defer testing.allocator.free(dual);
    const dual_body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, dual);
    defer testing.allocator.free(dual_body);
    var dual_parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, dual_body, .{});
    defer dual_parsed.deinit();
    const dane = dual_parsed.value.object.get("ane").?.object;
    try testing.expectEqual(@as(i64, 2), dane.get("units").?.integer);
    try testing.expectEqual(@as(i64, 224), dane.get("evals").?.integer);
    try testing.expectEqual(@as(i64, 3), dane.get("eval_failures").?.integer);
    const rows_json = dane.get("unit_evals").?.array;
    try testing.expectEqual(@as(usize, 2), rows_json.items.len);
    try testing.expectEqual(@as(i64, 1), rows_json.items[0].object.get("instance").?.integer);
    try testing.expectEqual(@as(i64, 2), rows_json.items[1].object.get("instance").?.integer);
    try testing.expectEqual(@as(i64, 112), rows_json.items[1].object.get("evals").?.integer);
}

test "settingsPropsJson: /props names the effective serving settings a benchmark ran under" {
    const frag = try settingsPropsJson(testing.allocator, .{
        .engine = "mlx",
        .kv_quant = "8",
        .kv_attn_mode = .auto,
        .decode_attn_quant = true,
        .prefill_chunk = 8192,
        .mtp_loaded = true,
        .mtp_default_on = true,
        .mtp_acceptance = .{ .tokenv3 = 0.95 },
        .mtp_depth = 0,
        .mtp_adaptive = true,
        .max_mtp_ctx = 32768,
        .drafter = "none",
        .pld = .{ .enable = false, .draft_len = 5, .key_len = 3 },
        .max_concurrent = 4,
        .prefix_cache_mem_bytes = 2048,
        .prefix_cache_disk_bytes = 0,
    });
    defer testing.allocator.free(frag);
    var config = model_mod.ModelConfig{};
    config.model_type = "qwen3_5";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const st = (parsed.value.object.get("settings") orelse return error.MissingSettings).object;
    try testing.expectEqualStrings(build_options.version, st.get("version").?.string);
    try testing.expectEqualStrings("8", st.get("kv_quant").?.string);
    try testing.expectEqualStrings("auto", st.get("kv_attn_mode").?.string);
    try testing.expect(st.get("decode_attn_quant").?.bool);
    const mtp = st.get("mtp").?.object;
    try testing.expect(mtp.get("default_on").?.bool);
    try testing.expectEqualStrings("tokenv3", mtp.get("acceptance").?.string);
    try testing.expectApproxEqAbs(@as(f64, 0.95), mtp.get("acceptance_param").?.float, 1e-6);
    try testing.expectEqual(@as(i64, 32768), mtp.get("max_ctx").?.integer);
    try testing.expectEqual(@as(i64, 4), st.get("max_concurrent").?.integer);

    const exact = try settingsPropsJson(testing.allocator, .{ .engine = "llama", .kv_quant = "q4", .kv_attn_mode = .dense, .decode_attn_quant = false, .prefill_chunk = 4096, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 3, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "dflash", .pld = PldDefaults.off, .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0 });
    defer testing.allocator.free(exact);
    var ep = try std.json.parseFromSlice(std.json.Value, testing.allocator, exact[",\"settings\":".len..], .{});
    defer ep.deinit();
    try testing.expect(ep.value.object.get("mtp").?.object.get("acceptance_param").? == .null);
}

test "settingsPropsJson: native lane configuration is not generic MTP or request engagement" {
    for ([_]struct { enabled: bool, sampled: bool, stages: u32 }{
        .{ .enabled = false, .sampled = true, .stages = 5 },
        .{ .enabled = true, .sampled = true, .stages = 5 },
        .{ .enabled = true, .sampled = false, .stages = 5 },
        .{ .enabled = false, .sampled = true, .stages = 0 },
    }) |case| {
        const frag = try settingsPropsJson(testing.allocator, .{
            .engine = "mlx",
            .kv_quant = "8",
            .kv_attn_mode = .auto,
            .decode_attn_quant = false,
            .prefill_chunk = 8192,
            .mtp_loaded = case.stages > 0,
            .mtp_default_on = case.enabled,
            .mtp_acceptance = .exact,
            .mtp_greedy_tail = true,
            .mtp_depth = 6,
            .mtp_adaptive = true,
            .max_mtp_ctx = 0,
            .native_speculation = .{ .lane = "dspark typical 0.3", .block_size = case.stages, .sampled = case.sampled },
            .drafter = "none",
            .pld = PldDefaults.off,
            .max_concurrent = 1,
            .prefix_cache_mem_bytes = 0,
            .prefix_cache_disk_bytes = 0,
        });
        defer testing.allocator.free(frag);
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
        defer parsed.deinit();
        const mtp = parsed.value.object.get("mtp").?.object;
        try testing.expectEqual(case.stages > 0, mtp.get("loaded").?.bool);
        try testing.expectEqual(case.enabled, mtp.get("default_on").?.bool);
        for ([_][]const u8{ "acceptance", "acceptance_param", "greedy_tail", "depth", "adaptive" }) |key|
            try testing.expect(mtp.get(key).? == .null);
        const native = mtp.get("native").?.object;
        try testing.expectEqualStrings("dspark typical 0.3", native.get("lane").?.string);
        try testing.expectEqual(@as(i64, case.stages), native.get("block_size").?.integer);
        try testing.expectEqualStrings(if (case.sampled and case.stages > 0) "stochastic" else "off", native.get("sampled_acceptance").?.string);
    }
}

test "settingsPropsJson: /props names the greedy tail" {
    for ([_]bool{ false, true }) |tail| {
        const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "8", .kv_attn_mode = .auto, .decode_attn_quant = false, .prefill_chunk = 8192, .mtp_loaded = true, .mtp_default_on = true, .mtp_acceptance = .exact, .mtp_greedy_tail = tail, .mtp_depth = 0, .mtp_adaptive = true, .max_mtp_ctx = 0, .drafter = "none", .pld = PldDefaults.off, .max_concurrent = 1, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0 });
        defer testing.allocator.free(frag);
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
        defer parsed.deinit();
        const got = parsed.value.object.get("mtp").?.object.get("greedy_tail") orelse return error.MissingGreedyTail;
        try testing.expectEqual(tail, got.bool);
    }
}

test "settingsPropsJson: /props reports the prefill decode share" {
    const frag = try settingsPropsJson(testing.allocator, .{ .engine = "mlx", .kv_quant = "off", .kv_attn_mode = .auto, .decode_attn_quant = false, .prefill_chunk = 8192, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = false, .max_mtp_ctx = 0, .drafter = "none", .pld = PldDefaults.off, .max_concurrent = 8, .prefix_cache_mem_bytes = 0, .prefix_cache_disk_bytes = 0, .prefill_decode_share = 0.5 });
    defer testing.allocator.free(frag);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
    defer parsed.deinit();
    const v = parsed.value.object.get("prefill_decode_share") orelse return error.MissingShare;
    try testing.expectApproxEqAbs(@as(f64, 0.5), v.float, 1e-6);
}

test "embeddedEngineSettings: an engine-backed model reports only the levers its engine runs" {
    const base: PropsSettings = .{ .engine = "mlx", .kv_quant = "8", .kv_attn_mode = .auto, .decode_attn_quant = true, .prefill_chunk = 8192, .mtp_loaded = false, .mtp_default_on = false, .mtp_acceptance = .exact, .mtp_depth = 0, .mtp_adaptive = true, .max_mtp_ctx = 0, .drafter = "assistant", .pld = .{ .enable = true, .draft_len = 5, .key_len = 3 }, .max_concurrent = 4, .prefix_cache_mem_bytes = 2048, .prefix_cache_disk_bytes = 0, .prefill_decode_share = 0.5 };

    const ds4 = embeddedEngineSettings(base, .ds4, true);
    try testing.expectEqualStrings("ds4", ds4.engine);
    try testing.expectEqualStrings("off", ds4.kv_quant);
    try testing.expect(!ds4.decode_attn_quant and !ds4.pld.enable and !ds4.mtp_adaptive);
    try testing.expect(ds4.mtp_loaded and ds4.mtp_default_on);
    try testing.expectEqualStrings("none", ds4.drafter);
    try testing.expectEqual(@as(usize, 0), ds4.prefill_chunk);
    try testing.expectEqual(@as(f32, 0), ds4.prefill_decode_share);

    const llama = embeddedEngineSettings(base, .llama, false);
    try testing.expectEqualStrings("llama", llama.engine);
    try testing.expectEqualStrings("8", llama.kv_quant);
    try testing.expect(!llama.decode_attn_quant and !llama.pld.enable and !llama.mtp_default_on);
    try testing.expectEqual(@as(f32, 0), llama.prefill_decode_share);

    // The Neural Engine: none of the MLX levers either.
    const ane = embeddedEngineSettings(base, .ane, false);
    try testing.expectEqualStrings("ane", ane.engine);
    try testing.expect(!ane.decode_attn_quant and !ane.pld.enable and !ane.mtp_loaded);
    try testing.expectEqual(@as(usize, 0), ane.prefill_chunk);

    try testing.expect(embeddedEngineSettings(base, .mlx, false).decode_attn_quant);
    try testing.expectEqual(@as(f32, 0.5), embeddedEngineSettings(base, .mlx, false).prefill_decode_share);
}

test "ngramWarmPropsJson: /props names how far the qwen4 ngram warm has got" {
    const none = try ngramWarmPropsJson(testing.allocator, 0, 0);
    defer testing.allocator.free(none);
    try testing.expectEqualStrings("", none);

    const frag = try ngramWarmPropsJson(testing.allocator, 17_179_869_184, 54_975_581_388);
    defer testing.allocator.free(frag);
    try testing.expectEqualStrings(",\"ngram_warm\":{\"bytes\":17179869184,\"total\":54975581388}", frag);

    var config = model_mod.ModelConfig{};
    config.model_type = "qwen4_exp";
    const body = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, frag);
    defer testing.allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const w = (parsed.value.object.get("ngram_warm") orelse return error.MissingNgramWarm).object;
    try testing.expectEqual(@as(i64, 17_179_869_184), w.get("bytes").?.integer);
    try testing.expectEqual(@as(i64, 54_975_581_388), w.get("total").?.integer);

    // Both fragments splice at the same slot, so an ANE boot on a qwen4 pack
    // must still parse.
    const one = [_]AneUnitStat{.{ .instance = 0, .evals = 1, .eval_failures = 0 }};
    const ane = try anePropsJson(testing.allocator, "channel", 64, 48, 8192, 8192, 0.45, 1, &one);
    defer testing.allocator.free(ane);
    const both = try std.fmt.allocPrint(testing.allocator, "{s}{s}", .{ ane, frag });
    defer testing.allocator.free(both);
    const body2 = try renderPropsBody(testing.allocator, &config, "4096", 1, 2, 3, 4, 5, 0, 0, both);
    defer testing.allocator.free(body2);
    var parsed2 = try std.json.parseFromSlice(std.json.Value, testing.allocator, body2, .{});
    defer parsed2.deinit();
    try testing.expect(parsed2.value.object.get("ane") != null);
    try testing.expect(parsed2.value.object.get("ngram_warm") != null);
}

test "renderPropsBody keeps fields the Swift app + integration tests rely on" {
    var config = model_mod.ModelConfig{};
    config.model_type = "gemma4";
    config.vocab_size = 32000;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    config.num_attention_heads = 32;
    config.num_key_value_heads = 8;
    config.head_dim = 128;
    config.quant_bits = 4;
    config.quant_group_size = 64;
    config.max_position_embeddings = 8192;

    const body = try renderPropsBody(testing.allocator, &config, "4096", 1234, 5678, 9_000_000_000, 16384, 4321, 7777, 8888, "");
    defer testing.allocator.free(body);

    // Hit every field a known consumer reads.
    try testing.expect(std.mem.indexOf(u8, body, "\"n_ctx\":4096") != null); // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"total_slots\":1") != null); // integration_test.sh
    try testing.expect(std.mem.indexOf(u8, body, "\"model_info\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"memory\"") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"active_bytes\":1234") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"peak_bytes\":5678") != null); // Swift fetchProps
    try testing.expect(std.mem.indexOf(u8, body, "\"available_bytes\":9000000000") != null); // Swift fetchProps (Free RAM line)
    try testing.expect(std.mem.indexOf(u8, body, "\"max_safe_context\":16384") != null); // Swift fetchProps
    // #110: the panel showed 19.6 GB (active) while the process held 81 GB.
    // The missing 61 GB was MLX's reclaimable buffer pool, which nothing we
    // expose reported — so the bug was invisible from every surface.
    try testing.expect(std.mem.indexOf(u8, body, "\"cache_bytes\":4321") != null); // Swift fetchProps
    // The tray splits active_bytes into loaded weights, the prefix KV cache and the rest.
    try testing.expect(std.mem.indexOf(u8, body, "\"weights_bytes\":7777") != null);
    try testing.expect(std.mem.indexOf(u8, body, "\"kv_cache_bytes\":8888") != null);
}
