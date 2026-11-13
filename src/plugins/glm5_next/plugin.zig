//! GLM-5.3-Flash (`glm5_next`): KDA linear attention and DeepSeek sparse attention inside DeepSeek-V4's 4-stream
//! Sinkhorn hyper-connections, a 288-expert sigmoid MoE. Its own trunk (`trunk.zig`) and forward
//! (`table_forward.zig`); batched rows advance each slot's state by its own call.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const host = @import("host.zig");
const fwd = @import("table_forward.zig");
const head = @import("head.zig");
const fixture = @import("fixture.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
const trunk = @import("trunk.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &host.unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{ .resident_bytes = &residentBytes },
    .spec = &.{ .trunk_weight = &host.trunkWeight, .head = &head.hooks },
    .meters = &host.meters,
};

const Verdict = abi.Verdict;
const Array = mlx.mlx_array;

/// KDA layers keep a recurrent state; every fourth layer caches one latent row `k` and the indexer's key and pool
/// gate `v` per token, unrotated.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isLinearLayer(@intCast(li)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .nope = true, .kv_heads = 1, .k_dim = c.dsa_kv_lora_rank, .v_dim = 2 * c.dsa_index_head_dim };
    return n;
}

/// Dense bf16 bytes per token (one latent row plus the indexer's key and gate per sparse layer) and the checkpoint.
/// The prefill bill stays the engine's `glm5PrefillMemoryNeeded` until `Bill.prefill_bytes` carries its terms.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    out.* = .{
        .kv_per_token = @as(u64, c.attnLayerCount()) * (c.dsa_kv_lora_rank + 2 * c.dsa_index_head_dim) * 2,
        .ssm_checkpoint = c.ssmCheckpointBytes(4),
    };
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return host.load(plugin.name, ctx, out);
}

/// A Sushi pack is held as its adapter leaves it (`SushiBill`); every other pack as stored (a streaming pack's bank
/// is its outside plugin's bill), plus a streaming pack's draft readout (`draftReadoutBytes`).
fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    const c = config.of(p);
    out.* = shards + draftReadoutBytes(c);
    if (c.expert_quant_json.len == 0) return .ok;
    const io = std.Io.Threaded.global_single_threaded.io();
    out.* = sushi_pack.residentBytes(io, std.heap.page_allocator, model_dir.slice(), SushiBill{ .layers = c.num_hidden_layers, .kv_lora_rank = c.dsa_kv_lora_rank }) catch |e|
        return Verdict.named(.failed, @errorName(e));
    out.* += draftReadoutBytes(c);
    return .ok;
}

/// The MTP drafts' readout a GLM streaming pack lends (`trunk.draftReadout`, at `trunk.readoutBits`), as
/// bailing_hybrid bills its own. An upper bound: the load builds it only when the outside plugin runs the head.
pub fn draftReadoutBytes(c: *const config.Config) u64 {
    if (!c.streamed_bank or !c.isGlm5()) return 0;
    return trunk.draftReadoutBytes(c.vocab_size, c.hidden_size, trunk.readoutBits(c));
}

/// A Sushi GLM tensor as the load holds it: the vision tower and the EXL3 MTP layer past the trunk not at all, the
/// KDA q/k/v projections twice (also joined into one matrix), the router, hyper-connection, indexer and KDA tables
/// beside an owned f32 copy, and `kv_b_proj` beside its two dense bf16 halves.
const SushiBill = struct {
    layers: u32,
    kv_lora_rank: u32,

    pub fn bill(f: SushiBill, t: sushi_pack.Stored) !u64 {
        if (std.mem.startsWith(u8, t.key, "model.visual.") or sushi_pack.pastTrunk(t.key, f.layers)) return 0;
        var total = t.bytes;
        if (isKdaProjection(t.key)) total += t.bytes;
        if (isF32Copied(t.key)) total += t.elements() * 4;
        if (std.mem.endsWith(u8, t.key, ".kv_b_proj.weight")) {
            if (t.shape.len != 2) return error.InvalidSafetensorsHeader;
            total += @as(u64, @intCast(t.shape[0].integer)) * f.kv_lora_rank * 2;
        }
        return total;
    }

    fn isF32Copied(key: []const u8) bool {
        for ([_][]const u8{ ".mlp.gate.weight", ".mlp.gate.e_score_correction_bias", ".hc_attn_", ".hc_ffn_", ".indexer.index_kpool_compress_", ".self_attn.A_log", ".self_attn.dt_bias" }) |p|
            if (std.mem.indexOf(u8, key, p) != null) return true;
        return false;
    }

    fn isKdaProjection(key: []const u8) bool {
        const at = std.mem.indexOf(u8, key, ".self_attn.") orelse return false;
        const rest = key[at + ".self_attn.".len ..];
        for ([_][]const u8{ "q_proj.", "k_proj.", "v_proj." }) |p| if (std.mem.startsWith(u8, rest, p)) return true;
        return false;
    }
};

test "glm5_next resident bill: a Sushi pack holds the joined KDA projections and the dense kv_b halves" {
    const header =
        \\{"model.language_model.layers.0.self_attn.q_proj.weight":{"dtype":"U32","shape":[1,1],"data_offsets":[0,100]},
        \\"model.language_model.layers.3.self_attn.kv_b_proj.weight":{"dtype":"U32","shape":[10,1],"data_offsets":[100,160]},
        \\"model.visual.patch_embed.proj.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[160,260]},
        \\"model.language_model.layers.45.enorm.weight":{"dtype":"BF16","shape":[1,1],"data_offsets":[260,360]}}
    ;
    // q 100 twice, kv_b 60 plus its dense 10 x 8 x 2 bf16 bytes.
    try std.testing.expectEqual(@as(u64, 200 + 60 + 160), try sushi_pack.headerResidentBytes(std.testing.allocator, header, "a.safetensors", null, SushiBill{ .layers = 45, .kv_lora_rank = 8 }));
}

test "glm5_next resident bill: a Sushi pack adds the f32 copies of the router and the hyper-connection tables" {
    const header =
        \\{"model.language_model.layers.3.mlp.gate.weight":{"dtype":"BF16","shape":[4,8],"data_offsets":[0,64]},
        \\"model.language_model.layers.3.hc_attn_fn":{"dtype":"BF16","shape":[16,4],"data_offsets":[64,192]},
        \\"model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate":{"dtype":"BF16","shape":[2,8],"data_offsets":[192,224]},
        \\"model.language_model.layers.3.input_layernorm.weight":{"dtype":"BF16","shape":[8],"data_offsets":[224,240]}}
    ;
    // 240 stored bytes, plus 4 bytes per element of the 32 + 64 + 16 widened ones.
    try std.testing.expectEqual(@as(u64, 240 + 4 * (32 + 64 + 16)), try sushi_pack.headerResidentBytes(std.testing.allocator, header, "a.safetensors", null, SushiBill{ .layers = 45, .kv_lora_rank = 0 }));
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "glm5_next: forward mode not declared");
    const ops = f.cache_ops orelse return Verdict.refuse(.unsupported, "glm5_next: a forward without cache_ops");
    trunk.bankPhase(m, if (f.mode == .batched_rows) @min(f.caches.len, fwd.MOE_DECODE_BATCH_MAX) else f.rows) catch |e| return Verdict.named(.failed, @errorName(e));
    if (f.mode == .batched_rows) {
        if (f.rows != 1 or f.tokens.len != f.caches.len or f.positions.len != f.caches.len) return Verdict.refuse(.unsupported, "glm5_next: batched rows take one host token per slot");
        if (f.capture_layers.len > 0 or f.media.ctx != null) return Verdict.refuse(.unsupported, "glm5_next: no captures or media on batched rows");
        batched(m, ops, f, out) catch |e| {
            host.clear(out, 0);
            return Verdict.named(.failed, @errorName(e));
        };
        return host.settleBank(m, f, out);
    }
    const v = host.forwardOne(plugin.name, fwd, m, ops, f, out);
    return if (v.isOk()) host.settleBank(m, f, out) else v;
}

/// N slots' next tokens in as few trunk forwards as the MoE row limit allows, each slot's token a row of one window:
/// logits `[N, V]` and, when asked, the final-normed hidden `[N, H]`.
fn batched(m: *host.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const n = f.caches.len;
    const groups = (n + fwd.MOE_DECODE_BATCH_MAX - 1) / fwd.MOE_DECODE_BATCH_MAX;
    const parts = try std.heap.c_allocator.alloc([2]Array, groups);
    defer std.heap.c_allocator.free(parts);
    var made: usize = 0;
    defer for (parts[0..made]) |p| for (p) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var start: usize = 0;
    while (start < n) : (start += fwd.MOE_DECODE_BATCH_MAX) {
        const end = @min(n, start + fwd.MOE_DECODE_BATCH_MAX);
        const ids = try host.hostIds(f.tokens.slice()[start..end]);
        defer _ = mlx.mlx_array_free(ids);
        const st: fwd.Step = .{
            .slots = .{ .ops = ops, .caches = f.caches.slice()[start..end], .positions = f.positions.slice()[start..end] },
            .batched = true,
        };
        const hidden = try fwd.hidden(m, &st, ids);
        defer _ = mlx.mlx_array_free(hidden);
        const logits = try m.lmHead(hidden);
        defer _ = mlx.mlx_array_free(logits);
        parts[made] = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        made += 1;
        try rows2d(m, logits, &parts[made - 1][0]);
        try rows2d(m, hidden, &parts[made - 1][1]);
        // Streamed experts: the next group's fetches rewrite the slot rows this group's last MoE layer reads.
        if ((m.experts != null or m.bank != null) and end < n) try evalParts(&parts[made - 1]);
    }
    out.logits = .{ .ctx = (try joinRows(m, parts[0..made], 0)).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = (try joinRows(m, parts[0..made], 1)).ctx };
}

fn evalParts(p: *const [2]Array) !void {
    const vec = mlx.mlx_vector_array_new_data(p, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// `[1, R, W]` as `[R, W]`.
fn rows2d(m: *host.Model, x: Array, dst: *Array) !void {
    const sh = mlx.getShape(x);
    try mlx.check(mlx.mlx_reshape(dst, x, &[_]c_int{ sh[1], sh[2] }, 2, m.s));
}

/// The groups' rows `which` (0 logits, 1 hidden) stacked in slot order, a new reference.
fn joinRows(m: *host.Model, parts: []const [2]Array, which: usize) !Array {
    var all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(all);
    if (parts.len == 1) {
        _ = mlx.mlx_array_set(&all, parts[0][which]);
        return all;
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p[which]);
    try mlx.check(mlx.mlx_concatenate_axis(&all, vec, 0, m.s));
    return all;
}

/// The tiny pack the conformance suite runs: every arm of the trunk (dense and MoE MLPs, KDA, sparse attention).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.glmFiles(a, 7) };
}

// ── tests ──

test "glm5_next resident bill: a GLM streaming pack bills its 8-bit draft readout, 643 MiB at GLM-5.3-Flash's head" {
    var c: config.Config = .{ .model_type = "glm5_next", .vocab_size = 154880, .hidden_size = 4096, .streamed_bank = true };
    try std.testing.expectEqual(@as(u64, 674_037_760), draftReadoutBytes(&c));
    c.streamed_bank = false;
    try std.testing.expectEqual(@as(u64, 0), draftReadoutBytes(&c));
}

test {
    _ = config;
    _ = fixture;
}
