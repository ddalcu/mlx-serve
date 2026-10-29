//! The DiT forward skeleton the image/video backends share, composed at comptime from a
//! plugin's hook namespace `H`:
//!   embed (patch embed + the joint/conditioning sequence) → for each layer: block → final
//!   head. Time/text conditioning is computed by the plugin once per step and handed in as
//!   `H.Ctx`; unpatchify stays with the pipeline (layouts differ per VAE).
//! The default block is pre-norm adaLN (`adaLnBlock`): h = x + attn(mod(norm1 x))·gate_a,
//! [h += cross(h)], out = h + ffn(mod(norm2 h))·gate_f. A plugin whose block is another
//! shape (FLUX.2 double stream, mage_flow joint streams) declares its own `H.block`.
//!
//! `H` declares:
//!   Model, Input, Ctx                       the plugin's weights, forward input, per-step context
//!   fn embed(*Graph, *const Model, Input, Ctx) !Array
//!   fn layers(*const Model) usize
//!   fn final(*Graph, *const Model, Array, Ctx) !Array
//!   and either `fn block(*Graph, *const Model, usize, Array, Ctx) !Array`, or for adaLN:
//!   Layer, fn layer(*const Model, usize) *const Layer, fn mods(Ctx, usize) Mods,
//!   const mod: Modulation, fn norm1/norm2(*Graph, *const Layer, Array) !Array,
//!   fn attention/ffn(*Graph, *const Layer, Array, Ctx) !Array, optional fn cross (same shape).

const std = @import("std");
const ops = @import("../../mlx/ops.zig");
const modulate_mod = @import("../blocks/dit/modulate.zig");

const Graph = ops.Graph;
const Array = ops.Array;

/// How a block's normed activation takes its (scale, shift).
pub const Modulation = union(enum) {
    /// x·(scale + 1) + shift, the 1 made in this dtype (FLUX.2, Krea, LTX, TRELLIS.2).
    plus_one: ops.Dtype,
    /// x·scale + shift, the 1 already folded into scale (MOSS-Image, Stable Audio 3).
    folded,
    /// x·scale, no shift (Qwen-Image).
    scale_only,

    pub fn apply(m: Modulation, parent: *Graph, x: Array, scale: Array, shift: ?Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(switch (m) {
            .plus_one => |dt| try modulate_mod.modulate(&g, x, scale, shift.?, dt),
            .folded => try g.add(try g.mul(x, scale), shift.?),
            .scale_only => try g.mul(x, scale),
        });
    }
};

/// One block's modulation rows, already broadcastable to [B, L, D] and gate-activated.
pub const Mods = struct { shift_a: ?Array = null, scale_a: Array, gate_a: Array, shift_f: ?Array = null, scale_f: Array, gate_f: Array };

pub fn adaLnBlock(comptime H: type, parent: *Graph, w: *const H.Layer, x: Array, m: Mods, ctx: H.Ctx) !Array {
    var g = parent.scope();
    defer g.deinit();
    const y = try H.mod.apply(&g, try H.norm1(&g, w, x), m.scale_a, m.shift_a);
    var h = try g.add(x, try g.mul(try H.attention(&g, w, y, ctx), m.gate_a));
    if (@hasDecl(H, "cross")) h = try g.add(h, try H.cross(&g, w, h, ctx));
    const y2 = try H.mod.apply(&g, try H.norm2(&g, w, h), m.scale_f, m.shift_f);
    return parent.keep(try g.add(h, try g.mul(try H.ffn(&g, w, y2, ctx), m.gate_f)));
}

pub fn Dit(comptime H: type) type {
    return struct {
        /// The model output for one step. Each block's intermediates die with it and the
        /// previous activation is freed as the next lands; `skip` drops one block (a CFG
        /// negative branch that skips a layer).
        pub fn forward(parent: *Graph, m: *const H.Model, input: H.Input, ctx: H.Ctx, skip: ?usize) !Array {
            var g = parent.scope();
            defer g.deinit();
            var x = try H.embed(&g, m, input, ctx);
            for (0..H.layers(m)) |i| {
                if (skip != null and skip.? == i) continue;
                const next = if (@hasDecl(H, "block"))
                    try H.block(&g, m, i, x, ctx)
                else
                    try adaLnBlock(H, &g, H.layer(m, i), x, H.mods(ctx, i), ctx);
                g.drop(x);
                x = next;
            }
            return parent.keep(try H.final(&g, m, x, ctx));
        }
    };
}

// ── tests ──

const testing = std.testing;
const mlx = @import("../../mlx/ffi.zig");
const kit = @import("../blocks/dit/testkit.zig");
const Linear = @import("../blocks/dit/linear.zig").Linear;
const rope = @import("../blocks/dit/rope.zig");
const scope_mod = @import("../../mlx/scope.zig");

test {
    _ = @import("../blocks/dit/linear.zig");
    _ = @import("../blocks/dit/modulate.zig");
    _ = @import("../blocks/dit/rope.zig");
    _ = @import("../blocks/dit/timestep.zig");
    _ = @import("../blocks/mlp/gelu.zig");
    _ = @import("../blocks/norms/layer_norm_f32.zig");
    _ = @import("../blocks/norms/group_norm.zig");
    _ = @import("../blocks/vae/conv.zig");
    _ = @import("../blocks/vae/resnet.zig");
    _ = @import("../blocks/vae/snake.zig");
    _ = @import("../blocks/schedulers/flow_match.zig");
    _ = @import("../blocks/text_encoders/qwen3.zig");
}

// MOSS-Image as the proving backend: its tiny random-weight DiT through `Dit(MossHooks)`
// against a verbatim copy of moss_image.zig's `Dit` (adapters unattached, so `linL` = forward).

const A = mlx.mlx_array;
const Scope = scope_mod.Scope;
const Lin = scope_mod.Linear;
const EPS: f32 = 1e-6;
const COMPUTE: mlx.mlx_dtype = .bfloat16;

const MossCfg = struct { dim: u32, ffn_dim: u32, heads: u32 };
const MossBlock = struct { norm1: A, norm2: A, q_norm: A, k_norm: A, q: Lin, k: Lin, v: Lin, o: Lin, up_gate: Lin, down: Lin };
const MossMod = struct { rows: [8]A };
const MossRope = struct { cos: A, sin: A, n_img: c_int };

/// moss_image.zig `Dit`, verbatim but for the fields this test does not build.
const MossOracle = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    cfg: MossCfg,
    video_embedder: Lin,
    final_norm: A,
    final_linear: Lin,
    blocks: []MossBlock,

    fn situ(sc: *Scope, hidden: A, gate: A) !A {
        const g = try sc.mul(try sc.mulLike(try sc.tanh(try sc.mulLike(gate, 0.25)), 4.0), try sc.sigmoid(gate));
        const h = try sc.mulLike(try sc.tanh(try sc.div(hidden, try sc.scalarLike(hidden, 25.0))), 25.0);
        return sc.mul(g, h);
    }

    fn rotate(sc: *Scope, x: A, cos: A, sin: A) !A {
        const d = mlx.getShape(x)[3];
        const x1 = try sc.slice(x, 3, 0, @divExact(d, 2));
        const x2 = try sc.slice(x, 3, @divExact(d, 2), d);
        const lo = try sc.sub(try sc.mul(x1, cos), try sc.mul(x2, sin));
        const hi = try sc.add(try sc.mul(x2, cos), try sc.mul(x1, sin));
        return sc.concat(&.{ lo, hi }, 3);
    }

    fn heads(self: *const MossOracle, sc: *Scope, y: A, norm: ?A, cos: A, sin: A) !A {
        const sh = mlx.getShape(y);
        const h: c_int = @intCast(self.cfg.heads);
        const x4 = try sc.reshape(y, &.{ sh[0], sh[1], h, @divExact(sh[2], h) });
        const r = if (norm) |nw| try rotate(sc, try sc.rms(x4, nw, EPS), cos, sin) else x4;
        return sc.transpose(r, &.{ 0, 2, 1, 3 });
    }

    fn attend(self: *const MossOracle, sc: *Scope, q: A, k: A, v: A) !A {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.cfg.dim / self.cfg.heads)));
        return sc.merge(try sc.sdpa(q, k, v, scale));
    }

    fn block(self: *const MossOracle, b: *const MossBlock, x: A, m: *const MossMod, rp: *const MossRope) !A {
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const y = try sc.add(try sc.mul(try sc.rms(x, b.norm1, EPS), m.rows[1]), m.rows[0]);
        const q = try self.heads(&sc, try b.q.forward(&sc, y), b.q_norm, rp.cos, rp.sin);
        const k = try self.heads(&sc, try b.k.forward(&sc, y), b.k_norm, rp.cos, rp.sin);
        const v = try self.heads(&sc, try b.v.forward(&sc, y), null, rp.cos, rp.sin);
        const attn = try b.o.forward(&sc, try self.attend(&sc, q, k, v));
        const h = try sc.add(x, try sc.mul(attn, m.rows[2]));
        const y2 = try sc.add(try sc.mul(try sc.rms(h, b.norm2, EPS), m.rows[4]), m.rows[3]);
        const hg = try b.up_gate.forward(&sc, y2);
        const f: c_int = @intCast(self.cfg.ffn_dim);
        const ffn = try b.down.forward(&sc, try situ(&sc, try sc.slice(hg, 2, 0, f), try sc.slice(hg, 2, f, 2 * f)));
        return sc.out(try sc.add(h, try sc.mul(ffn, m.rows[5])));
    }

    fn forward(self: *const MossOracle, latents: A, text: A, m: *const MossMod, rp: *const MossRope, skip: ?u32) !A {
        var x = blk: {
            var sc = Scope.init(self.allocator, self.s);
            defer sc.deinit();
            const video = try self.video_embedder.forward(&sc, try sc.astype(latents, COMPUTE));
            break :blk sc.out(try sc.concat(&.{ video, text }, 1));
        };
        defer _ = mlx.mlx_array_free(x);
        for (self.blocks, 0..) |*b, i| {
            if (skip != null and skip.? == i) continue;
            const next = try self.block(b, x, m, rp);
            _ = mlx.mlx_array_free(x);
            x = next;
        }
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const img = try sc.slice(x, 1, 0, rp.n_img);
        const y = try sc.add(try sc.mul(try sc.rms(img, self.final_norm, EPS), m.rows[7]), m.rows[6]);
        return sc.out(try sc.astype(try self.final_linear.forward(&sc, y), .float32));
    }
};

/// What MOSS-Image supplies to adopt the base: its weights viewed as sdk `Linear`s, its
/// SiTU-GLU and the step context; norms, modulation, RoPE and the skeleton are sdk's.
const MossHooks = struct {
    const Layer = struct { norm1: A, norm2: A, q_norm: A, k_norm: A, q: Linear, k: Linear, v: Linear, o: Linear, up_gate: Linear, down: Linear };
    const Model = struct { cfg: MossCfg, video_embedder: Linear, final_norm: A, final_linear: Linear, blocks: []const Layer };
    const Input = struct { latents: A, text: A };
    const Ctx = struct { mod: *const MossMod, rope: *const MossRope, cfg: MossCfg };
    const mod: Modulation = .folded;

    fn embed(g: *Graph, m: *const Model, in: Input, _: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        return g.keep(try c.concat(&.{ try m.video_embedder.forward(&c, try c.astype(in.latents, COMPUTE)), in.text }, 1));
    }
    fn layers(m: *const Model) usize {
        return m.blocks.len;
    }
    fn layer(m: *const Model, i: usize) *const Layer {
        return &m.blocks[i];
    }
    fn mods(ctx: Ctx, _: usize) Mods {
        const r = ctx.mod.rows;
        return .{ .shift_a = r[0], .scale_a = r[1], .gate_a = r[2], .shift_f = r[3], .scale_f = r[4], .gate_f = r[5] };
    }
    fn norm1(g: *Graph, w: *const Layer, x: A) !A {
        return g.rmsNorm(x, w.norm1, EPS);
    }
    fn norm2(g: *Graph, w: *const Layer, x: A) !A {
        return g.rmsNorm(x, w.norm2, EPS);
    }
    /// [B, L, H·D] → [B, L, H, D] → per-head RMS + RoPE (q, k) → [B, H, L, D].
    fn heads(g: *Graph, y: A, norm: ?A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const sh = ops.shape(y);
        const h: c_int = @intCast(ctx.cfg.heads);
        const x4 = try c.reshape(y, &.{ sh[0], sh[1], h, @divExact(sh[2], h) });
        const r = if (norm) |nw| try rope.ropeSplitHalf(&c, try c.rmsNorm(x4, nw, EPS), ctx.rope.cos, ctx.rope.sin, @divExact(sh[2], 2 * h)) else x4;
        return g.keep(try c.transpose(r, &.{ 0, 2, 1, 3 }));
    }
    fn attention(g: *Graph, w: *const Layer, y: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const q = try heads(&c, try w.q.forward(&c, y), w.q_norm, ctx);
        const k = try heads(&c, try w.k.forward(&c, y), w.k_norm, ctx);
        const v = try heads(&c, try w.v.forward(&c, y), null, ctx);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ctx.cfg.dim / ctx.cfg.heads)));
        return g.keep(try w.o.forward(&c, try c.mergeHeads(try c.sdpa(q, k, v, scale, .none))));
    }
    /// SiTU-GLU: (4·tanh(g/4)·σ(g)) · (25·tanh(h/25)), in the activation's dtype.
    fn ffn(g: *Graph, w: *const Layer, y: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const f: c_int = @intCast(ctx.cfg.ffn_dim);
        const hg = try w.up_gate.forward(&c, y);
        const hidden = try c.sliceAxis(hg, 2, 0, f);
        const gate = try c.sliceAxis(hg, 2, f, 2 * f);
        const gg = try c.mul(try c.mulScalar(try c.tanh(try c.mulScalar(gate, 0.25)), 4.0), try c.sigmoid(gate));
        const hh = try c.mulScalar(try c.tanh(try c.div(hidden, try c.scalarLike(hidden, 25.0))), 25.0);
        return g.keep(try w.down.forward(&c, try c.mul(gg, hh)));
    }
    fn final(g: *Graph, m: *const Model, x: A, ctx: Ctx) !A {
        var c = g.scope();
        defer c.deinit();
        const img = try c.sliceAxis(x, 1, 0, ctx.rope.n_img);
        const r = ctx.mod.rows;
        const y = try mod.apply(&c, try c.rmsNorm(img, m.final_norm, EPS), r[7], r[6]);
        return g.keep(try c.astype(try m.final_linear.forward(&c, y), .float32));
    }
};

test "media blocks: Dit(MOSS-Image hooks) is bit-equal to moss_image.zig's Dit.forward" {
    const s = kit.cpu();
    const a = testing.allocator;
    var g = Graph.init(a, s);
    defer g.deinit();
    const cfg: MossCfg = .{ .dim = 64, .ffn_dim = 96, .heads = 4 };
    const d: c_int = 64;
    const latent_c = 16;
    const n_img = 6;
    const n_txt = 5;
    const n_layers = 3;
    var seed: u64 = 900;
    const R = struct {
        fn t(gg: *Graph, sd: *u64, sh: []const c_int, scale: f32) !A {
            sd.* += 1;
            return kit.randn(gg, sh, sd.*, COMPUTE, scale);
        }
    };

    // One weight set, seen as moss's Scope `Linear` (oracle) and as sdk `Linear` (base).
    var old_blocks: [n_layers]MossBlock = undefined;
    var new_blocks: [n_layers]MossHooks.Layer = undefined;
    for (&old_blocks, &new_blocks) |*ob, *nb| {
        ob.* = .{
            .norm1 = try R.t(&g, &seed, &.{d}, 1),
            .norm2 = try R.t(&g, &seed, &.{d}, 1),
            .q_norm = try R.t(&g, &seed, &.{16}, 1),
            .k_norm = try R.t(&g, &seed, &.{16}, 1),
            .q = .{ .w = try R.t(&g, &seed, &.{ d, d }, 0.12) },
            .k = .{ .w = try R.t(&g, &seed, &.{ d, d }, 0.12) },
            .v = .{ .w = try R.t(&g, &seed, &.{ d, d }, 0.12) },
            .o = .{ .w = try R.t(&g, &seed, &.{ d, d }, 0.12), .bias = try R.t(&g, &seed, &.{d}, 0.1) },
            .up_gate = .{ .w = try R.t(&g, &seed, &.{ 2 * 96, d }, 0.12) },
            .down = .{ .w = try R.t(&g, &seed, &.{ d, 96 }, 0.1) },
        };
        nb.* = .{ .norm1 = ob.norm1, .norm2 = ob.norm2, .q_norm = ob.q_norm, .k_norm = ob.k_norm, .q = .{ .w = ob.q.w }, .k = .{ .w = ob.k.w }, .v = .{ .w = ob.v.w }, .o = .{ .w = ob.o.w, .bias = ob.o.bias }, .up_gate = .{ .w = ob.up_gate.w }, .down = .{ .w = ob.down.w } };
    }
    // The video embedder ships a bias; the final linear is affine-quantized as in a 4-bit pack.
    const ve_w = try R.t(&g, &seed, &.{ d, latent_c }, 0.2);
    const ve_b = try R.t(&g, &seed, &.{d}, 0.1);
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, try R.t(&g, &seed, &.{ latent_c, d }, 0.1), mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var qp: [3]A = undefined;
    for (&qp, 0..) |*p, i| {
        p.* = mlx.mlx_array_new();
        _ = mlx.mlx_vector_array_get(p, qv, i);
        _ = try g.adopt(p.*);
    }
    const final_norm = try R.t(&g, &seed, &.{d}, 1);
    const oracle = MossOracle{
        .allocator = a,
        .s = s,
        .cfg = cfg,
        .video_embedder = .{ .w = ve_w, .bias = ve_b },
        .final_norm = final_norm,
        .final_linear = .{ .w = qp[0], .scales = qp[1], .biases = qp[2], .bits = 4, .group_size = 32 },
        .blocks = &old_blocks,
    };
    const model = MossHooks.Model{
        .cfg = cfg,
        .video_embedder = .{ .w = ve_w, .bias = ve_b },
        .final_norm = final_norm,
        .final_linear = .{ .w = qp[0], .scales = qp[1], .biases = qp[2], .bits = 4, .group_size = 32 },
        .blocks = &new_blocks,
    };

    var m: MossMod = undefined;
    for (&m.rows) |*r| r.* = try R.t(&g, &seed, &.{ 1, 1, d }, 0.5);
    const rp = MossRope{
        .cos = try R.t(&g, &seed, &.{ 1, n_img + n_txt, 1, 8 }, 1),
        .sin = try R.t(&g, &seed, &.{ 1, n_img + n_txt, 1, 8 }, 1),
        .n_img = n_img,
    };
    const latents = try kit.randn(&g, &.{ 1, n_img, latent_c }, 990, .float32, 1);
    const text = try R.t(&g, &seed, &.{ 1, n_txt, d }, 1);
    const ctx = MossHooks.Ctx{ .mod = &m, .rope = &rp, .cfg = cfg };

    const before = ops.liveForTest();
    for ([_]?u32{ null, 1 }) |skip| {
        const skip_at: ?usize = if (skip) |k| k else null;
        var step = g.scope();
        defer step.deinit();
        const got = try Dit(MossHooks).forward(&step, &model, .{ .latents = latents, .text = text }, ctx, skip_at);
        // The skeleton leaves only its result in the caller's graph.
        try testing.expectEqual(before + 1, ops.liveForTest());
        const want = try kit.own(&step, try oracle.forward(latents, text, &m, &rp, skip));
        try ops.expectBitEqual(got, want);
    }
    try testing.expectEqual(before, ops.liveForTest());
}
