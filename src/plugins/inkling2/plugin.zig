//! Thinking Machines Inkling 2 Mini (`inkling2_loop`, text): a looped core of layers run several passes (each visit
//! its own KV slot and conv state), attention without RoPE (a hidden-conditioned relative bias, short convs on k/v,
//! L2 clips, sinks, log-scaling on global layers), inputs read from earlier blocks by depth attention, looped passes
//! mixed with messages, NVFP4/MXFP8 routed experts with a softcapped SwiGLU, and a softcapped head. mlx-lm's
//! `inkling2`. Serial; images refused.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const reader = @import("../../sdk/bases/reader.zig");
const attention = @import("attention.zig");
const depth = @import("depth.zig");
const moe = @import("moe.zig");
const SdkBank = @import("../../sdk/bases/blocks/moe.zig").Bank;
pub const trunk = @import("trunk.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const gpa = std.heap.c_allocator;

pub const Model = run_mod.Model(trunk.Trunk);

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = config.layout },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &trunkWeight },
};

/// The engine's default arithmetic, plus a stored prefix's conv state: every visit's three-row f32 window.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    out.* = .{ .ssm_checkpoint = @as(u64, c.n_visits) * (c.kernel - 1) * c.convWidth() * 4 };
}

pub const Loaded = struct {
    base: Model,
    layers: []trunk.Layer = &.{},
    owned: reader.Owned = .{},
    /// `lm_head` as stored, every padded row kept (what a drafter reads).
    head_full: Linear = .{},

    fn release(m: *Loaded) void {
        m.owned.deinit();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

/// The bare `embed` and `unembed` tables a drafter reads (no embedding norm, softcap or vocab trim).
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    b.trunk_weight.trunkWeight(switch (which) {
        .embed => &m.base.weights.embed,
        .head => &m.head_full,
    }, out);
    return .ok;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    if (f.media.ctx != null or f.embeds.ctx != null) return abi.Verdict.refuse(.unsupported, "inkling2_loop: images (text only)");
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(trunk.Trunk, &m.base, f, out);
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

/// The load's reads and the arrays it derives (reshapes, casts, slices), evaluated together at the end and owned by
/// the model.
const Binder = struct {
    r: *reader.Reader,
    o: *reader.Owned,
    g: ops.Graph,
    derived: std.ArrayList(Array) = .empty,

    fn name(x: *Binder, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return x.r.name(buf, fmt, args);
    }

    /// The raw tensor `<prefix>.<fmt>`, owned.
    fn need(x: *Binder, comptime fmt: []const u8, args: anytype) !Array {
        var nb: [256]u8 = undefined;
        const a = try x.r.need(try x.name(&nb, fmt, args));
        try x.o.own(a);
        return a;
    }

    fn proj(x: *Binder, comptime fmt: []const u8, args: anytype, in_dim: u32) !Linear {
        var nb: [256]u8 = undefined;
        const l = try x.r.proj(try x.name(&nb, fmt, args), in_dim);
        try x.o.linear(l);
        return l;
    }

    /// `a` (owned by the load's graph) as a handle the model owns, evaluated with the rest.
    fn keep(x: *Binder, a: ops.Array) !Array {
        const h = x.g.escape(a);
        errdefer _ = mlx.mlx_array_free(h);
        try x.o.own(h);
        try x.derived.append(gpa, h);
        return h;
    }

    fn reshaped(x: *Binder, a: Array, sh: []const c_int) !Array {
        return x.keep(try x.g.reshape(a, sh));
    }

    fn f32Of(x: *Binder, a: Array) !Array {
        if (mlx.mlx_array_dtype(a) == .float32) return a;
        return x.keep(try x.g.astype(a, .float32));
    }

    /// `[n]` f32 ones.
    fn ones(x: *Binder, n: u32) !Array {
        return x.keep(try x.g.ones(&.{@intCast(n)}, .float32));
    }

    fn finish(x: *Binder) !void {
        if (x.derived.items.len == 0) return;
        const vec = mlx.mlx_vector_array_new_data(x.derived.items.ptr, x.derived.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    fn deinit(x: *Binder) void {
        x.derived.deinit(gpa);
        x.g.deinit();
    }
};

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try reader.Reader.open(ctx, c.block_json, "model.language_model");
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.eps, c.eps), .a = gpa, .s = r.s } };
    errdefer m.release();
    var x: Binder = .{ .r = &r, .o = &m.owned, .g = ops.Graph.init(gpa, r.s) };
    defer x.deinit();

    const ones_h = try x.ones(c.hidden);
    const ones_hd = try x.ones(c.head_dim);
    m.layers = try gpa.alloc(trunk.Layer, c.layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(&x, c, @intCast(i), ones_hd);

    var nb: [256]u8 = undefined;
    const embed = try r.table(try r.name(&nb, "embed_tokens", .{}), c.hidden);
    try m.owned.linear(embed);
    m.head_full = try r.table("lm_head", c.hidden);
    try m.owned.linear(m.head_full);
    const loop = c.core_end > c.core_start;
    m.base.weights = .{
        .embed = embed,
        .embed_norm = try x.need("embed_norm.weight", .{}),
        .layers = m.layers,
        .final_query = try x.reshaped(try x.need("final_input.query", .{}), &.{ 1, @intCast(c.hidden) }),
        .norm = try x.need("norm.weight", .{}),
        .head = try trimRows(&x, m.head_full, c.unpadded_vocab),
        .message_query = if (loop) try x.need("message_input.query", .{}) else .{},
        .visits = c.visits[0..c.n_visits],
        .core_start = c.core_start,
        .core_len = c.core_end - c.core_start,
        .passes = c.passes,
        .n_prelude_blocks = c.nPreludeBlocks(),
        .hidden = c.hidden,
        .eps = c.eps,
        .softcap = c.final_softcap,
        .ones_h = ones_h,
    };
    try x.finish();
    return m;
}

/// The head's first `rows` rows (each of a quantized head's arrays): an evaluated row slice shares its parent's
/// buffer.
fn trimRows(x: *Binder, l: Linear, rows: u32) !Linear {
    var t = l;
    const n: c_int = @intCast(rows);
    if (ops.shape(l.w)[0] == n) return l;
    t.w = try x.keep(try x.g.sliceAxis(l.w, 0, 0, n));
    if (l.s.ctx != null) t.s = try x.keep(try x.g.sliceAxis(l.s, 0, 0, n));
    if (l.b.ctx != null) t.b = try x.keep(try x.g.sliceAxis(l.b, 0, 0, n));
    return t;
}

/// A projection's output width: a quantized weight stays `[out, in]`, a dense one comes pre-transposed.
fn outWidth(l: Linear) c_int {
    const sh = ops.shape(l.w);
    return if (l.s.ctx != null) sh[sh.len - 2] else sh[sh.len - 1];
}

fn kConv(x: *Binder, c: *const config.Config, i: u32, global: bool, comptime which: []const u8) !attention.KConv {
    if (!global) {
        var nb: [256]u8 = undefined;
        const w = try reader.depthwiseConv(try x.r.need(try x.name(&nb, "layers.{d}.self_attn." ++ which ++ "_sconv.conv1d.weight", .{i})), x.r.s);
        try x.o.own(w);
        return .{ .short = w };
    }
    const proj = try x.proj("layers.{d}.self_attn." ++ which ++ "_sconv.weight_proj", .{i}, c.hidden);
    if (outWidth(proj) != @as(c_int, @intCast(c.kv_heads * c.kernel))) return error.DynamicConvWidth;
    return .{ .dynamic = .{ .static = try x.need("layers.{d}.self_attn." ++ which ++ "_sconv.static_weight", .{i}), .proj = proj, .heads = c.kv_heads } };
}

fn bankOf(x: *Binder, c: *const config.Config, i: u32, comptime which: []const u8, in_dim: u32) !SdkBank {
    var nb: [256]u8 = undefined;
    const base = try x.name(&nb, "layers.{d}.mlp.experts." ++ which, .{i});
    var bk = reader.bank(try x.r.proj(base, in_dim), c.quant);
    for ([_]Array{ bk.w, bk.s, bk.b }) |a| try x.o.own(a);
    var sb: [256]u8 = undefined;
    const gs_name = try std.fmt.bufPrint(&sb, "{s}.global_scale", .{base});
    if (x.r.has(gs_name)) {
        const gs = try x.r.need(gs_name);
        defer _ = mlx.mlx_array_free(gs);
        // mlx's global scale is the amax: the expert's factor is gs / (6 * 448).
        bk.expert_scale = try x.keep(try x.g.div(try x.g.astype(gs, .float32), try x.g.scalar(2688.0, .float32)));
    }
    return bk;
}

fn loadLayer(x: *Binder, c: *const config.Config, i: u32, ones_hd: Array) !trunk.Layer {
    const global = c.global[i];
    const D = c.hidden;
    const R: c_int = @intCast(if (global) c.rel_extent else c.window);
    const rel_proj = try x.need("layers.{d}.self_attn.rel_logits_proj.proj", .{i});
    if (ops.shape(rel_proj)[1] != R) return error.RelativeExtentMismatch;
    const attn: attention.Weights = .{
        .q = try x.proj("layers.{d}.self_attn.q_proj", .{i}, D),
        .k = try x.proj("layers.{d}.self_attn.k_proj", .{i}, D),
        .v = try x.proj("layers.{d}.self_attn.v_proj", .{i}, D),
        .r = try x.proj("layers.{d}.self_attn.r_proj", .{i}, D),
        .o = try x.proj("layers.{d}.self_attn.o_proj", .{i}, c.heads * c.head_dim),
        .gate = try x.proj("layers.{d}.self_attn.gate_proj", .{i}, D),
        .sink = try x.proj("layers.{d}.self_attn.sink_proj", .{i}, D),
        .q_clip = try x.need("layers.{d}.self_attn.q_l2_clip.scale", .{i}),
        .rel_clip = try x.need("layers.{d}.self_attn.rel_l2_clip.scale", .{i}),
        .rel_proj = rel_proj,
        .k_conv = try kConv(x, c, i, global, "k"),
        .v_conv = try kConv(x, c, i, global, "v"),
        .heads = c.heads,
        .kv_heads = c.kv_heads,
        .head_dim = c.head_dim,
        .d_rel = c.d_rel,
        .window = if (global) 0 else c.window,
        .has_tau = global,
        .log_floor = c.log_floor,
        .log_alpha = c.log_alpha,
        .qk_clip = c.qk_clip,
        .rel_clip_thr = c.rel_clip,
        .eps = c.eps,
        .ones_hd = ones_hd,
    };
    const q_count: c_int = if (global) 3 else 1;
    var L: trunk.Layer = .{
        .global = global,
        .boundary = c.boundary(i),
        .attn = attn,
        .norm_q = if (global) try x.need("layers.{d}.q_input_layernorm.weight", .{i}) else try x.need("layers.{d}.input_layernorm.weight", .{i}),
        .attn_query = if (i > 0) try x.reshaped(try x.need("layers.{d}.attention_input.query", .{i}), &.{ q_count, @intCast(D) }) else .{},
        .mlp_query = try x.reshaped(try x.need("layers.{d}.mlp_input.query", .{i}), &.{ 1, @intCast(D) }),
        .post_attn_norm = try x.need("layers.{d}.post_attention_layernorm.weight", .{i}),
        .post_attn_conv = try convWeight(x, "layers.{d}.post_attention_convolution.conv1d.weight", i),
        .post_mlp_conv = try convWeight(x, "layers.{d}.post_mlp_convolution.conv1d.weight", i),
        .moe = .{
            .router = try x.f32Of(try x.need("layers.{d}.mlp.gate.weight", .{i})),
            .bias = try x.f32Of(try x.need("layers.{d}.mlp.gate.e_score_correction_bias", .{i})),
            .gate_up = try bankOf(x, c, i, "gate_up_proj", D),
            .down = try bankOf(x, c, i, "down_proj", c.inter),
            .top_k = c.top_k,
            .inter = c.inter,
            .route_scale = c.route_scale,
            .gate_cap = c.gate_cap,
            .up_cap = c.up_cap,
        },
    };
    if (global) {
        L.norm_k = try x.need("layers.{d}.k_input_layernorm.weight", .{i});
        L.norm_v = try x.need("layers.{d}.v_input_layernorm.weight", .{i});
    }
    if (c.inCore(i)) {
        var gates: [4]depth.Gate = undefined;
        inline for (.{ "q", "k", "v", "mlp" }, 0..) |p, j| {
            var nb: [256]u8 = undefined;
            // The converter keeps these dense; a quantized down projection is another model.
            if (x.r.has(try x.name(&nb, "layers.{d}." ++ p ++ "_message_gate.down_proj.scales", .{i}))) return error.QuantizedMessageGate;
            gates[j] = .{
                .down = try x.need("layers.{d}." ++ p ++ "_message_gate.down_proj.weight", .{i}),
                .forget = try x.f32Of(try x.need("layers.{d}." ++ p ++ "_message_gate.forget_gate_proj.weight", .{i})),
                .message = try x.f32Of(try x.need("layers.{d}." ++ p ++ "_message_gate.message_gate_proj.weight", .{i})),
            };
        }
        L.gates = gates;
    }
    return L;
}

fn convWeight(x: *Binder, comptime fmt: []const u8, i: u32) !Array {
    var nb: [256]u8 = undefined;
    const w = try reader.depthwiseConv(try x.r.need(try x.name(&nb, fmt, .{i})), x.r.s);
    try x.o.own(w);
    return w;
}

/// The tiny pack the conformance suite runs: the proof fixture (16 visits over 8 layers, a looped core of 4 run 3
/// passes, sliding and global layers, 4 experts of 32), f32.
pub fn conformanceFixture(_: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile("fixtures/tiny/config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile("fixtures/tiny/model.safetensors") },
    } };
}
