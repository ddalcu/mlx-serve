//! Qwen3.5 / 3.6 / 3.8 trunks over the engine's loaded tensors: `Decoder(TableEmbed, RmsNorm, Either(Gdn,
//! GatedAttention), GatedMlp or Moe, AddNorm, LmHead)`, the engine's layered forward. The load binds what the engine's
//! loader binds (q|k|v|z|a|b joined for decode widths, the decode attention side copies, the routed banks' one-expert
//! views, the shared-expert gate as a vector) and builds the per-model GDN constants, angle rows and compiled chains.
//! Every slot's recurrent state rides `CacheOps.state`; batched rows stack it per tick. A Prism pack is the dense
//! trunk in f16 with an f32 GDN state, every projection rotated through one `rht.Registry` and the seams fusing the
//! next norm's rotation (`plugins/prism_hadamard_qwen35`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const row_exact = @import("../../sdk/bases/blocks/row_exact.zig");
const naxAvailable = @import("../../mlx/device.zig").naxAvailable;
const qmm_int8 = @import("../../sdk/blocks/qmm/qmm_int8.zig");
const rht = @import("../../sdk/quant/hadamard/rht.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const scaled = @import("../../sdk/blocks/rope/scaled.zig");
const lane_qmm = @import("../../sdk/blocks/qmm/lane_qmm.zig");
const log = @import("../../sdk/util/log.zig");
const prism_seam = @import("../prism_hadamard_qwen35/seam.zig");
const weights_mod = @import("../../sdk/weights/load.zig");
const streamed = @import("../../sdk/weights/streamed.zig");
const ld = @import("../qwen3/load.zig");
const qwen3 = @import("../qwen3/plugin.zig");
pub const config = @import("config.zig");
const mtp = @import("mtp.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const gdn = b.gdn;

pub const Mixer = bases.Either(gdn.Gdn, b.gated_attention.GatedAttention);

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = Mixer,
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

pub const Moe = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = Mixer,
    .mlp = b.moe.Moe,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

/// A Prism pack's trunk: the dense one over rotated projections, its seams fusing the next norm's rotation.
pub const Prism = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = prism_seam.RotNorm,
    .mixer = Mixer,
    .mlp = b.mlp.GatedMlp,
    .seam = prism_seam.RotAddNorm,
    .head = b.head.LmHead,
});

comptime {
    if (!Dense.batches_decode or !Moe.batches_decode or !Prism.batches_decode) @compileError("qwen3_5: a base no longer batches");
}

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &config.layout },
    .spec = &.{ .head = &head_hooks, .trunk_weight = &trunkWeight, .tile_lanes = &tileLanes },
    .weights = &.{ .layer_weight = &layerWeight },
};

pub const head_hooks: abi.HeadHooks = .{
    .load = &headLoad,
    .activate = &headSlot,
    .release = &headRelease,
    .forward = &headForward,
    .truncate = &headTruncate,
    .adopt = &headAdopt,
    .mark = &headRelease,
    .draft_select = &headSelect,
    .bill = &headBill,
    .forward_lanes = &headForwardLanes,
    .cost_fingerprint = &headFingerprint,
};

const gpa = ld.gpa;
const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const Linear = ld.Linear;

/// Every term is the engine's default arithmetic over the shell and the cache layout.
pub fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A loaded model: one of the two bases, the per-model constants, and every handle it holds.
pub const Loaded = struct {
    caps: abi.Caps,
    geo: gdn.Geo = undefined,
    angles: b.qk_rope.AngleCache = .{},
    routes: []compiled.Closure = &.{},
    body: union(enum) {
        none,
        dense: struct { base: bases.Model(Dense), layers: []Dense.Layer },
        moe: struct { base: bases.Model(Moe), layers: []Moe.Layer },
        prism: struct { base: bases.Model(Prism), layers: []Prism.Layer },
    } = .none,
    owned: ld.Owned = .{},
    /// A Prism pack's rotations: every sign vector, pooled, and the memo one forward's projections share.
    rht: ?*rht.Registry = null,
    /// The MTP head (`SpecHooks.head`), loaded on the engine's request.
    head: ?mtp.Head = null,
    /// The scaled rope's spectrum and YaRN's `[head_dim]` factor (null = none), owned.
    rope_freqs: Array = .{ .ctx = null },
    rope_mscale: Array = .{ .ctx = null },
    /// The routed experts streamed from the pack as downloaded (the `stream_experts` setting).
    experts: ?streamed.Streamed = null,

    fn release(m: *Loaded) void {
        lane_qmm.release(@intFromPtr(m));
        if (m.head) |*h| h.deinit();
        if (m.experts) |*x| x.close();
        switch (m.body) {
            .none => {},
            inline .dense, .moe, .prism => |*d| {
                gpa.free(d.layers);
                d.base.shared.deinit();
            },
        }
        for (m.routes) |*c| compiled.free(c);
        gpa.free(m.routes);
        m.angles.deinit();
        gdn.freeConsts(&m.geo);
        m.owned.deinit();
        if (m.rht) |reg| {
            reg.deinit();
            gpa.destroy(reg);
        }
    }
};

/// The LOSSY `int8_prefill` load setting (a Prism pack's rotated projections only).
fn int8Prefill(settings_json: []const u8) bool {
    const S = struct { int8_prefill: ?bool = null };
    const parsed = std.json.parseFromSlice(S, gpa, settings_json, .{ .ignore_unknown_fields = true }) catch return false;
    defer parsed.deinit();
    return parsed.value.int8_prefill orelse false;
}

pub fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = ld.failed(e);
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    const src = ctx.weights orelse return error.NoWeightSource;
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    const rotated = c.hadamard_block > 0;
    var modules: [64]u8 = undefined;
    // The Hadamard format claims a block beside the config's `modules`.
    const config_json = if (rotated) try std.fmt.bufPrint(&modules, "{{\"modules\":[{{\"block\":{d}}}]}}", .{c.hadamard_block}) else "{}";
    var r: ld.Reader = undefined;
    try r.openIn(src, c.block_json, config_json, ld.probePrefix(src, if (c.next) "model" else "language_model.model") orelse return error.MissingWeight, s);
    defer r.close();
    const act: mlx.mlx_dtype = if (rotated) .float16 else .bfloat16;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .caps = ctx.parsed.caps, .geo = .{
        .hk = @intCast(c.linear_num_key_heads),
        .hv = @intCast(c.linear_num_value_heads),
        .dk = @intCast(c.linear_key_head_dim),
        .dv = @intCast(c.linear_value_head_dim),
        .kernel = @intCast(c.linear_conv_kernel_dim),
        .state = if (rotated) .float32 else .bfloat16,
        .qk_eps = 1e-6,
        .arch = if (c.next) "qwen3_next" else "qwen3_5_moe",
        .q_scale = .{ .ctx = null },
        .k_scale = .{ .ctx = null },
        .ones = .{ .ctx = null },
    } };
    errdefer m.release();
    if (rotated) {
        const reg = try gpa.create(rht.Registry);
        reg.* = .init(gpa, c.hadamard_block);
        m.rht = reg;
        if (int8Prefill(ctx.settings_json.slice()) and mlx.metalKernelsAvailable()) {
            reg.int8 = try gpa.create(qmm_int8.Cache);
            reg.int8.?.* = .{};
        }
    }
    try gdn.consts(&m.geo, act, s);
    scaled.logScaling(if (c.next) "qwen3_next" else "qwen3_5", c.rope);
    if (try scaled.freqs(c.rope, c.ropeDims(), c.rope_theta)) |f| {
        m.rope_freqs = f;
        try m.owned.own(f);
    }
    const ms: f32 = @floatCast(rope_scaling.mscale(c.rope));
    if (ms != 1.0) {
        m.rope_mscale = try scaled.mscaleTable(ms, c.ropeDims(), c.head_dim, act, s);
        try m.owned.own(m.rope_mscale);
    }
    // The engine compiles its chains only where a GPU runs them.
    const gpu = !mlx.noGpuBackend() and mlx.streamIsGpu(s);
    if (gpu) m.geo.gate_fn = gdn.compileGate(m.geo.state, s);
    const shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps);
    if (c.moe()) {
        if (streamed.Setting.parse(ctx.settings_json.slice())) |st| m.experts = try streamed.Streamed.openPack(gpa, ctx.model_dir.slice(), s, st.slotRows(c.num_experts));
        m.body = .{ .moe = .{ .base = .{ .weights = undefined, .shared = shared, .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids }, .layers = try gpa.alloc(Moe.Layer, c.num_hidden_layers) } };
        m.routes = try gpa.alloc(compiled.Closure, c.num_hidden_layers);
        @memset(m.routes, .{ .ctx = null });
        const e = &m.body.moe;
        for (e.layers, 0..) |*L, i| {
            try layerCommon(Moe, m, &r, c, L, @intCast(i), s);
            L.mlp = try moeMlp(m, &r, c, i, s);
            if (gpu) {
                m.routes[i] = b.moe.compileRouting(&L.mlp, s);
                L.mlp.route_fn = m.routes[i];
            }
        }
        e.base.weights = try trunk(Moe, m, &r, c, e.layers);
    } else if (rotated) {
        m.body = .{ .prism = .{ .base = .{ .weights = undefined, .shared = shared, .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids }, .layers = try gpa.alloc(Prism.Layer, c.num_hidden_layers) } };
        const d = &m.body.prism;
        for (d.layers, 0..) |*L, i| {
            try layerCommon(Prism, m, &r, c, L, @intCast(i), s);
            L.mlp = try denseMlp(m, &r, c, i, s);
        }
        d.base.weights = try trunk(Prism, m, &r, c, d.layers);
        try bindRotations(m.rht.?, &d.base.weights, d.layers, r.neg_scale, s);
    } else {
        m.body = .{ .dense = .{ .base = .{ .weights = undefined, .shared = shared, .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids }, .layers = try gpa.alloc(Dense.Layer, c.num_hidden_layers) } };
        const d = &m.body.dense;
        for (d.layers, 0..) |*L, i| {
            try layerCommon(Dense, m, &r, c, L, @intCast(i), s);
            L.mlp = try denseMlp(m, &r, c, i, s);
        }
        d.base.weights = try trunk(Dense, m, &r, c, d.layers);
        d.base.row_exact = rowLane(d.layers, &d.base.weights.head);
    }
    return m;
}

fn denseMlp(m: *Loaded, r: *ld.Reader, c: *const config.Config, i: usize, s: mlx.mlx_stream) !b.mlp.GatedMlp.Weights {
    var nb: [256]u8 = undefined;
    var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), c.hidden_size);
    var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), c.hidden_size);
    const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
    const gu = if (gate.s.ctx != null) try ld.join(&m.owned, &.{ &gate, &up }, s) else b.linear.Joined{};
    for ([_]Linear{ gate, up, down }) |l| try m.owned.linear(l);
    return .{ .gate = gate, .up = up, .down = down, .gu = gu };
}

/// A Prism pack's rotations as the engine's load binds them (`registerRhtSigns`, `aliasFusedRotations`): every
/// projection reads `reg`, each sign vector pooled there (one handle per distinct vector, so the projections reading
/// one activation share its memo), its signs evaluated in one batch; a joined group keeps its join only where its parts
/// share one vector. `neg_scale`: every rotated weight's biases are -scales (`Registry.bias_is_neg_scale`).
fn bindRotations(reg: *rht.Registry, w: *Prism.Weights, layers: []Prism.Layer, neg_scale: bool, s: mlx.mlx_stream) !void {
    var lins: std.ArrayList(*Linear) = .empty;
    defer lins.deinit(gpa);
    var joins: std.ArrayList(struct { j: *b.linear.Joined, parts: [4]*const Linear, n: usize }) = .empty;
    defer joins.deinit(gpa);
    try lins.appendSlice(gpa, &.{ &w.embed.table, &w.head });
    for (layers) |*L| {
        switch (L.mixer) {
            .first => |*g| {
                try lins.appendSlice(gpa, &.{ &g.qkv, &g.qkv_parts[0], &g.qkv_parts[1], &g.qkv_parts[2], &g.z, &g.a, &g.b, &g.out, &g.in.lin });
                if (g.in.built()) try joins.append(gpa, .{ .j = &g.in, .parts = .{ &g.qkv, &g.z, &g.a, &g.b }, .n = 4 });
            },
            .second => |*a| try lins.appendSlice(gpa, &.{ &a.q.lin, &a.k.lin, &a.v.lin, &a.o.lin, &a.qkv.lin }),
        }
        const p = &L.mlp;
        try lins.appendSlice(gpa, &.{ &p.gate, &p.up, &p.down, &p.gu.lin });
        if (p.gu.built()) try joins.append(gpa, .{ .j = &p.gu, .parts = .{ &p.gate, &p.up, &p.gate, &p.gate }, .n = 2 });
    }
    // One eval for every sign vector's f32 copy (the pool's hash), as the engine's load batches them.
    var f32s: std.ArrayList(Array) = .empty;
    defer {
        for (f32s.items) |f| _ = mlx.mlx_array_free(f);
        f32s.deinit(gpa);
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (lins.items) |l| if (l.signs.ctx != null) {
        try f32s.append(gpa, try rht.signsF32(l.signs, s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, f32s.items[f32s.items.len - 1]));
    };
    try mlx.check(mlx.mlx_eval(vec));
    var k: usize = 0;
    for (lins.items) |l| if (l.signs.ctx != null) {
        try reg.register(l.w, l.signs, f32s.items[k]);
        k += 1;
    };
    // The format's handles go (a tied head shares the table's); each projection reads the pooled vector.
    var freed: std.ArrayList(?*anyopaque) = .empty;
    defer freed.deinit(gpa);
    for (lins.items) |l| {
        l.rht = reg;
        const raw = l.signs;
        if (raw.ctx == null) continue;
        l.signs = reg.get(l.w).?;
        if (std.mem.indexOfScalar(?*anyopaque, freed.items, raw.ctx) != null) continue;
        try freed.append(gpa, raw.ctx);
        _ = mlx.mlx_array_free(raw);
    }
    reg.bias_is_neg_scale = neg_scale;
    for (joins.items) |j| {
        const first = j.parts[0].signs;
        const shared = for (j.parts[0..j.n]) |p| {
            if (p.signs.ctx == null or p.signs.ctx != first.ctx) break false;
        } else true;
        if (shared) {
            j.j.lin.signs = first;
            j.j.lin.rht = reg;
        } else j.j.* = .{};
    }
}

/// `.mma` where every projection fits the row kernels (the engine's `firstRowExactMiss`; an untied table is only
/// gathered), else `.off`.
fn rowLane(layers: []const Dense.Layer, head: *const Linear) row_exact.Lane {
    if (!row_exact.fits(head)) return .off;
    for (layers) |*L| {
        const mixer = switch (L.mixer) {
            .first => |*w| fitAll(&.{ &w.qkv, &w.qkv_parts[0], &w.qkv_parts[1], &w.qkv_parts[2], &w.z, &w.a, &w.b, &w.out }),
            .second => |*w| fitAll(&.{ &w.q.lin, &w.k.lin, &w.v.lin, &w.o.lin }),
        };
        if (!mixer or !fitAll(&.{ &L.mlp.gate, &L.mlp.up, &L.mlp.down })) return .off;
    }
    return .mma;
}

fn fitAll(ls: []const *const Linear) bool {
    for (ls) |l| if (!row_exact.fits(l)) return false;
    return true;
}

/// `SpecHooks.tile_lanes`: the dense trunk's projections on the `.mma` lane re-ordered in place (`lane_qmm.tileInPlace`),
/// joined groups with their parts' row views; the head and the table keep MLX's layout (other readers).
fn tileLanes(mp: *anyopaque, lane: u32) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const d = switch (m.body) {
        .dense => |*d| d,
        else => return .ok,
    };
    if (lane != @intFromEnum(row_exact.Lane.mma) or d.base.row_exact != .mma) return .ok;
    const t: Tiler = .{ .owner = @intFromPtr(m), .s = d.base.s };
    var bytes: u64 = 0;
    for (d.layers) |*L| {
        bytes += t.layer(L) catch |e| return ld.failed(e);
    }
    if (bytes > 0) log.info("[lane] trunk projections tiled in place ({d:.1} GB, no copy)\n", .{@as(f64, @floatFromInt(bytes)) / (1 << 30)});
    return .ok;
}

const Tiler = struct {
    owner: usize,
    s: mlx.mlx_stream,

    fn layer(t: Tiler, L: *const Dense.Layer) !u64 {
        var sum: u64 = switch (L.mixer) {
            .first => |*g| (if (g.combined) 0 else try t.group(&g.in, &.{ &g.qkv, &g.qkv_parts[0], &g.qkv_parts[1], &g.qkv_parts[2], &g.z, &g.a, &g.b })) + try t.one(&g.out, &.{}),
            .second => |*a| try t.group(&a.qkv, &.{ &a.q.lin, &a.k.lin, &a.v.lin }) + try t.one(&a.o.lin, &.{}),
        };
        sum += try t.group(&L.mlp.gu, &.{ &L.mlp.gate, &L.mlp.up });
        return sum + try t.one(&L.mlp.down, &.{});
    }

    fn one(t: Tiler, l: *const Linear, widths: []const c_int) !u64 {
        if (l.w.ctx == null or l.s.ctx == null or l.mode != .affine) return 0;
        return lane_qmm.tileInPlace(t.owner, l.w, l.s, l.b, l.bits, l.group_size, widths, t.s);
    }

    /// A joined group tiles once, with its parts' row views (the parts are views of it); unjoined parts one by one.
    fn group(t: Tiler, j: *const b.linear.Joined, parts: []const *const Linear) !u64 {
        if (j.built()) return t.one(&j.lin, j.widths[0 .. std.mem.indexOfScalar(c_int, &j.widths, 0) orelse j.widths.len]);
        var sum: u64 = 0;
        for (parts) |p| sum += try t.one(p, &.{});
        return sum;
    }
};

/// A norm weight as base `D` holds it: `Prism`'s carries the registry and whether its reader needs it unrotated.
fn normOf(comptime D: type, m: *const Loaded, w: Array, want_normed: bool) @FieldType(D.Layer, "input_norm") {
    if (D != Prism) return w;
    return .{ .w = w, .reg = m.rht, .want_normed = want_normed };
}

fn norm(m: *Loaded, r: *ld.Reader, comptime leaf: []const u8, i: usize) !Array {
    var nb: [256]u8 = undefined;
    const a = try r.need(try r.name(&nb, "layers.{d}." ++ leaf, .{i}));
    try m.owned.own(a);
    return a;
}

/// The layer's norms and mixer (a GatedDeltaNet or gated full attention layer).
fn layerCommon(comptime D: type, m: *Loaded, r: *ld.Reader, c: *const config.Config, L: *D.Layer, i: u32, s: mlx.mlx_stream) !void {
    // A GDN layer's a/b projections read its input unrotated; the MLP after attention only through rotated weights.
    L.input_norm = normOf(D, m, try norm(m, r, "input_layernorm.weight", i), c.linear(i));
    L.post_norm = normOf(D, m, try norm(m, r, "post_attention_layernorm.weight", i), false);
    L.mixer = if (c.linear(i)) .{ .first = try gdnLayer(m, r, c, i, s) } else .{ .second = try attnLayer(m, r, c, i, s) };
}

fn gdnLayer(m: *Loaded, r: *ld.Reader, c: *const config.Config, i: u32, s: mlx.mlx_stream) !gdn.Gdn.Weights {
    const hidden = c.hidden_size;
    var nb: [256]u8 = undefined;
    if (c.next) {
        const qkvz = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_qkvz", .{i}), hidden);
        const ba = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_ba", .{i}), hidden);
        const o = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.out_proj", .{i}), c.linear_num_value_heads * c.linear_value_head_dim);
        for ([_]Linear{ qkvz, ba, o }) |l| try m.owned.linear(l);
        return .{
            .qkv = qkvz,
            .b = ba,
            .out = o,
            .combined = true,
            .conv_w = try norm(m, r, "linear_attn.conv1d.weight", i),
            .A_log = try norm(m, r, "linear_attn.A_log", i),
            .dt_bias = try norm(m, r, "linear_attn.dt_bias", i),
            .norm_w = try norm(m, r, "linear_attn.norm.weight", i),
            .geo = m.geo,
        };
    }
    var qkv = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_qkv", .{i}), hidden);
    var z = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_z", .{i}), hidden);
    var a = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_a", .{i}), hidden);
    var bb = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.in_proj_b", .{i}), hidden);
    const out = try r.proj(try r.name(&nb, "layers.{d}.linear_attn.out_proj", .{i}), c.linear_num_value_heads * c.linear_value_head_dim);
    const in = if (qkv.s.ctx != null) try ld.join(&m.owned, &.{ &qkv, &z, &a, &bb }, s) else b.linear.Joined{};
    for ([_]Linear{ qkv, z, a, bb, out }) |l| try m.owned.linear(l);
    return .{
        .qkv = qkv,
        .z = z,
        .a = a,
        .b = bb,
        .out = out,
        .in = in,
        .conv_w = try norm(m, r, "linear_attn.conv1d.weight", i),
        .A_log = try norm(m, r, "linear_attn.A_log", i),
        .dt_bias = try norm(m, r, "linear_attn.dt_bias", i),
        .norm_w = try norm(m, r, "linear_attn.norm.weight", i),
        .geo = m.geo,
    };
}

fn attnLayer(m: *Loaded, r: *ld.Reader, c: *const config.Config, i: u32, s: mlx.mlx_stream) !b.gated_attention.GatedAttention.Weights {
    const hidden = c.hidden_size;
    const n = c.num_hidden_layers;
    var nb: [256]u8 = undefined;
    var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
    var k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
    var v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
    const o = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), c.num_attention_heads * c.head_dim);
    // Row-exact trunks on NAX join q|k|v (one lane matmul at every width), as the engine's loader does. A rotated
    // projection takes no join and no side copy (a copy would bypass its rotation).
    const rotated = c.hadamard_block > 0;
    const qkv = if (!c.next and !c.moe() and !rotated and naxAvailable() and q.s.ctx != null) try ld.join(&m.owned, &.{ &q, &k, &v }, s) else b.linear.Joined{};
    for ([_]Linear{ q, k, v, o }) |l| try m.owned.linear(l);
    return .{
        .q = .{ .lin = q, .dq = if (rotated) .{} else try ld.sideCopy(&m.owned, q, i, n, s) },
        .k = .{ .lin = k, .dq = if (rotated) .{} else try ld.sideCopy(&m.owned, k, i, n, s) },
        .v = .{ .lin = v, .dq = if (rotated) .{} else try ld.sideCopy(&m.owned, v, i, n, s) },
        .o = .{ .lin = o, .dq = if (rotated) .{} else try ld.sideCopy(&m.owned, o, i, n, s) },
        .q_norm = try norm(m, r, "self_attn.q_norm.weight", i),
        .k_norm = try norm(m, r, "self_attn.k_norm.weight", i),
        .qkv = qkv,
        .geo = .{
            .heads = @intCast(c.num_attention_heads),
            .kv_heads = @intCast(c.num_key_value_heads),
            .head_dim = @intCast(c.head_dim),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar))),
            .rope_dims = @intCast(c.ropeDims()),
            .rope_base = c.rope_theta,
            .yarn_freqs = m.rope_freqs,
            .yarn_mscale = m.rope_mscale,
            .yarn_factor = @floatCast(rope_scaling.mscale(c.rope)),
            .rope_scale = rope_scaling.positionScale(c.rope),
            .ntk = rope_scaling.ntk(c.rope, c.max_position_embeddings),
        },
        .angles = &m.angles,
    };
}

fn moeMlp(m: *Loaded, r: *ld.Reader, c: *const config.Config, i: usize, s: mlx.mlx_stream) !b.moe.Moe.Weights {
    const hidden = c.hidden_size;
    var nb: [256]u8 = undefined;
    const router = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate", .{i}), hidden);
    try m.owned.linear(router);
    const gate = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.gate_proj", .{i}), hidden);
    const up = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.up_proj", .{i}), hidden);
    const down = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.down_proj", .{i}), c.moe_intermediate_size);
    var w: b.moe.Moe.Weights = .{
        .router = router,
        .route = .softmax,
        .gate = qwen3.bankOf(gate, c.quant),
        .up = qwen3.bankOf(up, c.quant),
        .down = qwen3.bankOf(down, c.quant),
        .top_k = @intCast(c.num_experts_per_tok),
        .experts = @intCast(c.num_experts),
    };
    if (m.experts) |*x| {
        // The banks give the quant params only: their tensors are never read.
        for ([_]Linear{ gate, up, down }) |l| for ([_]Array{ l.w, l.s, l.b }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        inline for (.{ &w.gate, &w.up, &w.down }) |bk| {
            if (bk.qp.mode != .affine) return error.StreamedPackNotAffine;
            bk.* = .{ .w = .{ .ctx = null }, .qp = bk.qp };
        }
        w.streamed = .{ .routed = x.routedSource(), .layer = x.layerOf(try r.name(&nb, "layers.{d}.mlp", .{i})) orelse return error.StreamedLayerMissing };
    } else {
        for ([_]Linear{ gate, up, down }) |l| try m.owned.linear(l);
        w.bank_views = try qwen3.bankViews(&m.owned, gate, up, down, s);
    }
    if (r.has(try r.name(&nb, "layers.{d}.mlp.shared_expert_gate.weight", .{i}))) {
        const sg = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_expert.gate_proj", .{i}), hidden);
        const su = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_expert.up_proj", .{i}), hidden);
        const sd = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_expert.down_proj", .{i}), c.shared_expert_intermediate_size);
        for ([_]Linear{ sg, su, sd }) |l| try m.owned.linear(l);
        const logit = try sharedGate(m, r, c, i, s);
        w.shared = .{ .gate = sg, .up = su, .down = sd, .gate_logit = logit.lin, .gate_vec = logit.vec };
    }
    return w;
}

/// The shared expert's one-logit gate as the engine binds it: a tiny quantized row dequantized to dense, kept
/// `[in, 1]`, and as one contiguous `[hidden]` vector the router kernel dots in.
fn sharedGate(m: *Loaded, r: *ld.Reader, c: *const config.Config, i: usize, s: mlx.mlx_stream) !struct { lin: Linear, vec: Array } {
    var nb: [256]u8 = undefined;
    var w = try r.need(try r.name(&nb, "layers.{d}.mlp.shared_expert_gate.weight", .{i}));
    try m.owned.own(w);
    var sc = try r.share(try r.name(&nb, "layers.{d}.mlp.shared_expert_gate.scales", .{i}));
    try m.owned.own(sc);
    var bi = try r.share(try r.name(&nb, "layers.{d}.mlp.shared_expert_gate.biases", .{i}));
    try m.owned.own(bi);
    try weights_mod.dequantizeSmall(&w, &sc, &bi, c.hidden_size, &m.owned.list, gpa, s);
    if (sc.ctx != null) {
        const lin = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_expert_gate", .{i}), c.hidden_size);
        try m.owned.linear(lin);
        return .{ .lin = lin, .vec = .{ .ctx = null } };
    }
    const t = try @import("../../mlx/array.zig").transposeBf16Weight(w, s);
    try m.owned.own(t);
    var vec: Array = .{ .ctx = null };
    if (mlx.mlx_array_size(t) == c.hidden_size) {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, t, &[_]c_int{@intCast(c.hidden_size)}, 1, s));
        vec = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&vec, flat, false, s));
        try m.owned.own(vec);
    }
    return .{ .lin = .{ .w = t }, .vec = vec };
}

/// Embedding, final norm and head, as the engine binds them.
fn trunk(comptime D: type, m: *Loaded, r: *ld.Reader, c: *const config.Config, layers: []const D.Layer) !D.Weights {
    var nb: [256]u8 = undefined;
    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try m.owned.own(final_norm);
    return .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = if (c.hadamard_block > 0) .float16 else .bfloat16 },
        .layers = layers,
        .final_norm = normOf(D, m, final_norm, true),
        .head = try r.head(&m.owned, embed, c.hidden_size, c.tie_word_embeddings),
    };
}

/// The embedding table and head the MTP head's readout reads, borrowed (a tied head is the table).
pub fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return switch (m.body) {
        .none => Verdict.refuse(.failed, "qwen3_5: not loaded"),
        inline .dense, .moe, .prism => |*d| ld.trunkWeight(switch (which) {
            .embed => &d.base.weights.embed.table,
            .head => &d.base.weights.head,
        }, out),
    };
}

/// A dense layer's MLP projections and a linear-attention layer's joined q|k|v and z, borrowed (`Caps.ane_prefill`).
pub fn layerWeight(mp: *anyopaque, layer: u32, which: abi.LayerWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const none = Verdict.refuse(.unsupported, "qwen3_5: the layer has no such projection");
    const l: *const Linear = switch (m.body) {
        .none => return Verdict.refuse(.failed, "qwen3_5: not loaded"),
        inline .dense, .moe, .prism => |*d| blk: {
            if (layer >= d.layers.len) return none;
            const L = &d.layers[layer];
            break :blk switch (which) {
                .mlp_gate, .mlp_up, .mlp_down => if (@TypeOf(L.mlp) != b.mlp.GatedMlp.Weights) return none else switch (which) {
                    .mlp_gate => &L.mlp.gate,
                    .mlp_up => &L.mlp.up,
                    else => &L.mlp.down,
                },
                .gdn_qkv, .gdn_z => switch (L.mixer) {
                    .first => |*g| if (g.combined) return none else if (which == .gdn_qkv) &g.qkv else &g.z,
                    .second => return none,
                },
            };
        },
    };
    if (l.w.ctx == null) return none;
    return ld.trunkWeight(l, out);
}

// ── the MTP head: its state is the head's cache, which the engine owns per slot ──

fn headLoad(mp: *anyopaque, ctx: *const abi.LoadCtx) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (m.head != null) return .ok;
    const embed = switch (m.body) {
        .none => return Verdict.refuse(.failed, "qwen3_5: not loaded"),
        inline .dense, .moe, .prism => |*d| &d.base.weights.embed.table,
    };
    m.head = mtp.load(config.of(ctx.parsed), embed, .{ .freqs = m.rope_freqs, .mscale = m.rope_mscale }, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |e| return ld.failed(e);
    return .ok;
}

fn headSlot(_: *anyopaque, _: u32) callconv(.c) Verdict {
    return .ok;
}

fn headRelease(_: *anyopaque, _: u32) callconv(.c) void {}

fn headTruncate(_: *anyopaque, _: u32, _: u64) callconv(.c) void {}

fn headAdopt(_: *anyopaque, _: u32, _: abi.Handle) callconv(.c) Verdict {
    return .ok;
}

fn headSelect(_: *anyopaque, _: abi.Array, _: *abi.Array) callconv(.c) Verdict {
    return Verdict.refuse(.unsupported, "the engine selects drafts");
}

fn headBill(_: *const abi.Parsed) callconv(.c) u64 {
    return 0;
}

fn headForward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const h = &(m.head orelse return Verdict.refuse(.failed, "qwen3_5: no MTP head loaded"));
    defer if (m.rht) |reg| reg.dropMemo();
    mtp.forward(h, f, out) catch |e| return ld.failed(e);
    return .ok;
}

fn headForwardLanes(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const h = &(m.head orelse return Verdict.refuse(.failed, "qwen3_5: no MTP head loaded"));
    defer if (m.rht) |reg| reg.dropMemo();
    mtp.forwardLanes(h, f, out) catch |e| return ld.failed(e);
    return .ok;
}

fn headFingerprint(mp: *anyopaque) callconv(.c) u32 {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return if (m.head) |*h| h.costFingerprint() else 0;
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "forward mode not declared");
    // The memo pins the last input and its rotation; nothing past this forward reads it.
    defer if (m.rht) |reg| reg.dropMemo();
    return switch (m.body) {
        .none => Verdict.refuse(.failed, "qwen3_5: not loaded"),
        .dense => |*d| run_mod.tableForward(Dense, &d.base, f, out),
        .moe => |*e| run_mod.tableForward(Moe, &e.base, f, out),
        .prism => |*d| run_mod.tableForward(Prism, &d.base, f, out),
    };
}

/// The tiny pack the conformance suite runs: four layers (GDN, GDN, GDN, attention), width 64, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const heads = 2;
    const kv_heads = 1;
    const hd = 32;
    const hk = 1;
    const hv = 2;
    const dk = 32;
    const dv = 32;
    const conv = 2 * hk * dk + hv * dv;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 35;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = shape, .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    const ones = try a.alloc(f32, h);
    @memset(ones, 1);
    try put(a, &tensors, &seed, "language_model.model.embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, "language_model.lm_head.weight", &.{ v, h }, 0.1);
    try tensors.append(a, .{ .name = "language_model.model.norm.weight", .shape = &.{h}, .data = ones });
    for (0..4) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "language_model.model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        try tensors.append(a, .{ .name = try L.n(a, i, "input_layernorm.weight"), .shape = &.{h}, .data = ones });
        try tensors.append(a, .{ .name = try L.n(a, i, "post_attention_layernorm.weight"), .shape = &.{h}, .data = ones });
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.down_proj.weight"), &.{ h, inter }, 0.1);
        if (i % 4 != 3) {
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_qkv.weight"), &.{ conv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_z.weight"), &.{ hv * dv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_a.weight"), &.{ hv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.in_proj_b.weight"), &.{ hv, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.conv1d.weight"), &.{ conv, 4, 1 }, 0.3);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.A_log"), &.{hv}, 0.5);
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.dt_bias"), &.{hv}, 0.5);
            try tensors.append(a, .{ .name = try L.n(a, i, "linear_attn.norm.weight"), .shape = &.{dv}, .data = ones[0..dv] });
            try put(a, &tensors, &seed, try L.n(a, i, "linear_attn.out_proj.weight"), &.{ h, hv * dv }, 0.1);
        } else {
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_proj.weight"), &.{ 2 * heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.v_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.o_proj.weight"), &.{ h, heads * hd }, 0.1);
            try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.q_norm.weight"), .shape = &.{hd}, .data = ones[0..hd] });
            try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.k_norm.weight"), .shape = &.{hd}, .data = ones[0..hd] });
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"qwen3_5","tie_word_embeddings":false,"text_config":{"model_type":"qwen3_5_text",
        \\ "hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
        \\ "head_dim":32,"vocab_size":64,"rms_norm_eps":1e-6,"full_attention_interval":4,"linear_num_key_heads":1,
        \\ "linear_num_value_heads":2,"linear_key_head_dim":32,"linear_value_head_dim":32,"linear_conv_kernel_dim":4,
        \\ "max_position_embeddings":4096,"hidden_act":"silu","eos_token_id":2,
        \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
