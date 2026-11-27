//! Tencent Hunyuan 4-flash (`hy_v4_flash`): the module-owned native arch behind the model table.
//! Serial decode, single-flight, no prefix cache: the request's state lives on the model,
//! and position 0 resets the state.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const hy4 = @import("hy_v4_flash.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const Weights = @import("../../sdk/weights/resident.zig").Weights;
const weights_load = @import("../../sdk/weights/load.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &config.bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .release_slot = &releaseSlot,
    .activate_slot = &activateSlot,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
};

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else @as(u32, @intCast(c.layers)), cap);
    for (out[0..n], first..) |*l, li| {
        const is_indexed = if (li < c.indexed.len) c.indexed[li] else true;
        l.* = if (is_indexed)
            .{ .kind = .full, .kv_heads = @intCast(c.indexed_heads), .k_dim = @intCast(c.head_dim) }
        else
            .{ .kind = .sliding, .window = @intCast(c.sliding), .kv_heads = @intCast(c.swa_heads), .k_dim = @intCast(c.head_dim) };
    }
    return n;
}

pub const Model = struct {
    cfg: config.Config,
    weights: Weights,
    mdl: *hy4.Model,
    caps: abi.Caps,
    s: mlx.mlx_stream,
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => Verdict.refuse(.failed, "hy_v4_flash: out of memory"),
            error.MissingWeight => Verdict.refuse(.failed, "hy_v4_flash: a weight is missing"),
            else => Verdict.refuse(.failed, "hy_v4_flash: the pack does not load"),
        };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Model {
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.cfg = config.of(ctx.parsed).*;
    m.weights = (if (ctx.weights) |w| Resident.take(w) else null) orelse
        try weights_load.loadWeights(std.Io.Threaded.global_single_threaded.io(), gpa, ctx.model_dir.slice());
    errdefer m.weights.deinit();
    m.s = .{ .ctx = ctx.stream.ctx };
    if (m.s.ctx == null) m.s = mlx.gpuStream();
    m.caps = ctx.parsed.caps;

    m.mdl = try gpa.create(hy4.Model);
    errdefer gpa.destroy(m.mdl);
    m.mdl.* = try hy4.Model.init(gpa, m.cfg, &m.weights, m.s);
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Model = @ptrCast(@alignCast(mp));
    m.mdl.deinit();
    gpa.destroy(m.mdl);
    m.weights.deinit();
    gpa.destroy(m);
}

fn releaseSlot(_: *anyopaque, _: u32) callconv(.c) void {}

fn activateSlot(_: *anyopaque, _: u32) callconv(.c) Verdict {
    return .ok;
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Model = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "hy_v4_flash: forward mode not declared");
    if (f.capture_layers.len > 0) return Verdict.refuse(.unsupported, "hy_v4_flash: no layer capture");
    if (f.want_hidden) return Verdict.refuse(.unsupported, "hy_v4_flash: no hidden-state output (embeddings)");
    if (f.slots.len > 1 or f.rows == 0) return Verdict.refuse(.unsupported, "hy_v4_flash: one slot's rows at a time");

    const in = Ids.of(f) catch return Verdict.refuse(.failed, "hy_v4_flash: the token ids do not read back");
    defer in.deinit();
    const pos: usize = if (f.positions.len > 0) f.positions.slice()[0] else 0;
    if (pos == 0) {
        m.mdl.resetState();
    }
    const logits = m.mdl.forward(in.host) catch return Verdict.refuse(.failed, "hy_v4_flash: the forward failed");
    if (f.logits == .none) {
        _ = mlx.mlx_array_free(logits);
        out.logits = .{ .ctx = null };
        return .ok;
    }

    if (f.logits == .last_row and in.host.len > 1) {
        defer _ = mlx.mlx_array_free(logits);
        const sh = mlx.getShape(logits);
        var last = mlx.mlx_array_new();
        const start = [_]c_int{ @intCast(sh[0] - 1), 0 };
        const stop = [_]c_int{ @intCast(sh[0]), @intCast(sh[1]) };
        const stride = [_]c_int{ 1, 1 };
        mlx.check(mlx.mlx_slice(&last, logits, &start, 2, &stop, 2, &stride, 2, m.s)) catch {
            _ = mlx.mlx_array_free(last);
            return Verdict.refuse(.failed, "hy_v4_flash: slice last row failed");
        };
        out.logits = .{ .ctx = last.ctx };
    } else {
        out.logits = .{ .ctx = logits.ctx };
    }
    return .ok;
}

pub const Ids = struct {
    host: []const u32,
    dev: mlx.mlx_array = .{ .ctx = null },
    owned: bool = false,

    pub fn of(f: *const abi.Forward) !Ids {
        if (f.ids.ctx == null) return if (f.tokens.len == f.rows) .{ .host = f.tokens.slice() } else error.RowsMismatch;
        const dev: mlx.mlx_array = .{ .ctx = f.ids.ctx };
        var i32s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(i32s);
        try mlx.check(mlx.mlx_astype(&i32s, dev, .int32, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(i32s));
        const n = mlx.mlx_array_size(i32s);
        if (n != f.rows) return error.RowsMismatch;
        const data = mlx.mlx_array_data_int32(i32s) orelse return error.NoData;
        const host = try gpa.alloc(u32, n);
        for (host, data[0..n]) |*o, id| o.* = @intCast(id);
        return .{ .host = host, .owned = true };
    }

    pub fn deinit(in: Ids) void {
        if (in.owned) gpa.free(in.host);
    }
};

/// The tiny pack the conformance suite runs: width 64, 1 layer (sliding attention, dense MLP, no engrams), seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v: u64 = 64;
    const d: u64 = 64;
    const hc: u64 = 2;
    const q_lora: u64 = 32;
    const kv_lora: u64 = 32;
    const qk_nope: u64 = 32;
    const rope_dim: u64 = 64;
    const swa_heads: u64 = 2;
    const swa_v_dim: u64 = 32;
    const inter: u64 = 64;

    var t = sdk.tensors.Tensors.init(a, 1);
    try t.rand("model.embed_tokens.weight", .{}, &.{ v, d }, .F32, 0.5);
    try t.rand("lm_head.weight", .{}, &.{ v, d }, .F32, 0.1);
    try t.fill("model.norm.weight", .{}, &.{d}, .F32, 1);
    try t.rand("model.hc_head.hc_head_fn", .{}, &.{ hc, hc * d }, .F32, 0.02);
    try t.fill("model.hc_head.hc_head_base", .{}, &.{hc}, .F32, 0.1);
    try t.fill("model.hc_head.hc_head_scale", .{}, &.{1}, .F32, 1);

    const qd = qk_nope + rope_dim;
    try t.fill("model.layers.0.input_layernorm.weight", .{}, &.{d}, .F32, 1);
    try t.fill("model.layers.0.post_attention_layernorm.weight", .{}, &.{d}, .F32, 1);
    try t.rand("model.layers.0.self_attn.q_a_proj.weight", .{}, &.{ q_lora, d }, .F32, 0.1);
    try t.fill("model.layers.0.self_attn.q_a_layernorm.weight", .{}, &.{q_lora}, .F32, 1);
    try t.rand("model.layers.0.self_attn.q_b_proj.weight", .{}, &.{ swa_heads * qd, q_lora }, .F32, 0.1);
    try t.rand("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", .{}, &.{ kv_lora + rope_dim, d }, .F32, 0.1);
    try t.fill("model.layers.0.self_attn.kv_a_layernorm.weight", .{}, &.{kv_lora}, .F32, 1);
    try t.rand("model.layers.0.self_attn.kv_b_proj.weight", .{}, &.{ swa_heads * (qk_nope + swa_v_dim), kv_lora }, .F32, 0.1);
    try t.rand("model.layers.0.self_attn.linear_gate.weight", .{}, &.{ swa_heads * swa_v_dim, d }, .F32, 0.1);
    try t.rand("model.layers.0.self_attn.o_proj.weight", .{}, &.{ d, swa_heads * swa_v_dim }, .F32, 0.1);
    try t.fill("model.layers.0.self_attn.post_proj_norm.weight", .{}, &.{d}, .F32, 1);
    try t.fill("model.layers.0.self_attn.learnable_sink_param", .{}, &.{swa_heads}, .F32, 0.1);

    try t.rand("model.layers.0.hc_attn_layer.hc_pre.hc_fn", .{}, &.{ 2 * hc, hc * d }, .F32, 0.02);
    try t.fill("model.layers.0.hc_attn_layer.hc_pre.hc_base", .{}, &.{ 2 * hc }, .F32, 0.1);
    try t.fill("model.layers.0.hc_attn_layer.hc_pre.hc_scale", .{}, &.{2}, .F32, 1);

    try t.rand("model.layers.0.hc_mlp_layer.hc_pre.hc_fn", .{}, &.{ 2 * hc, hc * d }, .F32, 0.02);
    try t.fill("model.layers.0.hc_mlp_layer.hc_pre.hc_base", .{}, &.{ 2 * hc }, .F32, 0.1);
    try t.fill("model.layers.0.hc_mlp_layer.hc_pre.hc_scale", .{}, &.{2}, .F32, 1);

    try t.rand("model.layers.0.mlp.gate_proj.weight", .{}, &.{ inter, d }, .F32, 0.1);
    try t.rand("model.layers.0.mlp.up_proj.weight", .{}, &.{ inter, d }, .F32, 0.1);
    try t.rand("model.layers.0.mlp.down_proj.weight", .{}, &.{ d, inter }, .F32, 0.1);
    try t.fill("model.layers.0.mlp.post_fc2_norm.weight", .{}, &.{d}, .F32, 1);

    const cfg_json =
        \\{
        \\  "model_type": "hy_v4_flash",
        \\  "vocab_size": 64,
        \\  "hidden_size": 64,
        \\  "num_hidden_layers": 1,
        \\  "hc_mult": 2,
        \\  "q_lora_rank": 32,
        \\  "kv_lora_rank": 32,
        \\  "qk_nope_head_dim": 32,
        \\  "qk_rope_head_dim": 64,
        \\  "v_head_dim": 32,
        \\  "num_attention_heads": 2,
        \\  "swa_num_attention_heads": 2,
        \\  "swa_qk_nope_head_dim": 32,
        \\  "swa_v_dim": 32,
        \\  "sliding_window": 32,
        \\  "rms_norm_eps": 1e-5,
        \\  "layer_types": ["sliding_attention"],
        \\  "mlp_layer_types": ["dense"],
        \\  "engram_layer_ids": [],
        \\  "eos_token_id": [2, 120025]
        \\}
    ;

    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes = cfg_json },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7}" },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}
