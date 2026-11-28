//! Ming-Image's DiT (the release's `DiffusionTransformer`, Z-Image's single-stream S3-DiT shape): patch embed, two
//! modulated noise-refiner blocks over the image tokens, two unmodulated context-refiner blocks over the caption,
//! then the main blocks over `[image | caption]` and an adaLN head. Each block is sandwich-normed:
//! x + tanh(g)·norm₂(attn(norm₁(x)·(1 + s))), the same for the FFN; 3-axis RoPE as complex pairs in f32.
//! The release pads both sequences to 32 tokens with masked zero rows (`zero_masked`); a masked row neither attends
//! nor is attended to and is cropped, so the pad is dropped here and only its positions kept.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const log = @import("../../sdk/util/log.zig");
const scope_mod = @import("../../mlx/scope.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const dit_base = @import("../../sdk/bases/dit.zig");
const rope_blk = @import("../../sdk/blocks/dit/rope.zig");
const timestep_blk = @import("../../sdk/blocks/dit/timestep.zig");
const Linear = @import("../../sdk/blocks/dit/linear.zig").Linear;
const llm = @import("llm.zig");

const Weights = weights_load.Weights;
const Graph = ops.Graph;
const Array = ops.Array;
const S = mlx.mlx_stream;

const DT = llm.DT;
const ADALN_DIM: u32 = 256;
const SEQ_MULTI_OF: u32 = 32;
pub const PATCH: u32 = 2;

pub const Config = struct {
    dim: u32,
    heads: u32,
    layers: u32,
    refiners: u32,
    in_ch: u32,
    cap_dim: u32,
    eps: f32,
    theta: f64,
    axes: [3]u32,
    t_scale: f32,

    pub fn headDim(c: Config) u32 {
        return c.dim / c.heads;
    }

    pub fn parse(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Config {
        const path = try std.fmt.allocPrint(a, "{s}/transformer/config.json", .{model_dir});
        defer a.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
        defer a.free(bytes);
        var p = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        defer p.deinit();
        const o = p.value.object;
        const pad = o.get("alignment_padding_mode") orelse return error.MingImageBadConfig;
        const multi = o.get("multi_frame_output") orelse return error.MingImageBadConfig;
        // The layer-decomposition checkpoint (learned pads, one latent per layer) is another pipeline.
        if (pad != .string or !std.mem.eql(u8, pad.string, "zero_masked") or multi != .bool or multi.bool) return error.MingImageUnsupportedRecipe;
        const axes = o.get("axes_dims") orelse return error.MingImageBadConfig;
        if (axes != .array or axes.array.items.len != 3) return error.MingImageBadConfig;
        var c: Config = .{
            .dim = try int(o, "dim"),
            .heads = try int(o, "n_heads"),
            .layers = try int(o, "n_layers"),
            .refiners = try int(o, "n_refiner_layers"),
            .in_ch = try int(o, "in_channels"),
            .cap_dim = try int(o, "cap_feat_dim"),
            .eps = @floatCast(try num(o, "norm_eps")),
            .theta = try num(o, "rope_theta"),
            .axes = undefined,
            .t_scale = @floatCast(try num(o, "t_scale")),
        };
        for (&c.axes, axes.array.items) |*d, v| {
            if (v != .integer or v.integer <= 0 or @rem(v.integer, 2) != 0) return error.MingImageBadConfig;
            d.* = @intCast(v.integer);
        }
        if (c.heads == 0 or c.dim % c.heads != 0 or c.axes[0] + c.axes[1] + c.axes[2] != c.headDim() or try int(o, "n_kv_heads") != c.heads)
            return error.MingImageBadConfig;
        return c;
    }

    fn num(o: std.json.ObjectMap, key: []const u8) !f64 {
        return switch (o.get(key) orelse return error.MingImageBadConfig) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => error.MingImageBadConfig,
        };
    }

    fn int(o: std.json.ObjectMap, key: []const u8) !u32 {
        const f = try num(o, key);
        if (f < 0 or f > 1 << 20 or f != @floor(f)) return error.MingImageBadConfig;
        return @intFromFloat(f);
    }
};

const Block = struct {
    norm1: Array,
    norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    norm_q: Array,
    norm_k: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    w1: Linear,
    w2: Linear,
    w3: Linear,
    /// The four modulation rows; null in a context-refiner block.
    ada: ?Linear,

    fn load(w: *const Weights, comptime pfx: []const u8, i: usize, dim: u32, modulated: bool) !Block {
        const w1 = try llm.lin(w, pfx ++ ".{d}.feed_forward.w1", .{i}, dim);
        return .{
            .norm1 = try llm.get(w, pfx ++ ".{d}.attention_norm1.weight", .{i}),
            .norm2 = try llm.get(w, pfx ++ ".{d}.attention_norm2.weight", .{i}),
            .ffn_norm1 = try llm.get(w, pfx ++ ".{d}.ffn_norm1.weight", .{i}),
            .ffn_norm2 = try llm.get(w, pfx ++ ".{d}.ffn_norm2.weight", .{i}),
            .norm_q = try llm.get(w, pfx ++ ".{d}.attention.norm_q.weight", .{i}),
            .norm_k = try llm.get(w, pfx ++ ".{d}.attention.norm_k.weight", .{i}),
            .q = try llm.lin(w, pfx ++ ".{d}.attention.to_q", .{i}, dim),
            .k = try llm.lin(w, pfx ++ ".{d}.attention.to_k", .{i}, dim),
            .v = try llm.lin(w, pfx ++ ".{d}.attention.to_v", .{i}, dim),
            .o = try llm.lin(w, pfx ++ ".{d}.attention.to_out.0", .{i}, dim),
            .w1 = w1,
            .w2 = try llm.lin(w, pfx ++ ".{d}.feed_forward.w2", .{i}, @intCast(ops.shape(w1.w)[0])),
            .w3 = try llm.lin(w, pfx ++ ".{d}.feed_forward.w3", .{i}, dim),
            .ada = if (modulated) try llm.lin(w, pfx ++ ".{d}.adaLN_modulation.0", .{i}, @min(dim, ADALN_DIM)) else null,
        };
    }
};

/// A block's activated modulation for one step, rows broadcastable to [1, L, D].
pub const Mod = dit_base.Mods;

/// Positions on the 3 axes as one RoPE table pair `[1, L, 1, head_dim/2]` f32.
pub const Rope = struct { cos: Array, sin: Array };

pub fn ropeTable(g: *Graph, cfg: Config, ids: []const [3]u32) !Rope {
    const half: usize = cfg.headDim() / 2;
    const cs = try g.a.alloc(f32, ids.len * half);
    defer g.a.free(cs);
    const sn = try g.a.alloc(f32, ids.len * half);
    defer g.a.free(sn);
    for (ids, 0..) |id, r| {
        var col: usize = 0;
        for (cfg.axes, 0..) |d, ax| {
            for (0..d / 2) |j| {
                // freqs in f64, the angle rounded to f32 (`precompute_freqs_cis`), cos/sin of that f32.
                const freq = 1.0 / std.math.pow(f64, cfg.theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d)));
                const ang: f32 = @floatCast(@as(f64, @floatFromInt(id[ax])) * freq);
                cs[r * half + col] = @cos(ang);
                sn[r * half + col] = @sin(ang);
                col += 1;
            }
        }
    }
    const sh = [_]c_int{ 1, @intCast(ids.len), 1, @intCast(half) };
    return .{ .cos = try g.fromSlice(f32, cs, &sh), .sin = try g.fromSlice(f32, sn, &sh) };
}

/// The caption token count after the release's pad to 32 (where the image positions start).
pub fn padded(n: usize) usize {
    return std.mem.alignForward(usize, n, SEQ_MULTI_OF);
}

/// Image ids `(cap_padded + 1 + f, h, w)` frame by frame (an edit's reference is frame 1), then caption ids
/// `(1 + i, 0, 0)`: the unified order.
pub fn positions(a: std.mem.Allocator, ht: u32, wt: u32, frames: u32, cap_len: usize) ![][3]u32 {
    const per: usize = @as(usize, ht) * wt;
    const n_img = per * frames;
    const ids = try a.alloc([3]u32, n_img + cap_len);
    const t0: u32 = @intCast(padded(cap_len) + 1);
    for (0..frames) |f| for (0..ht) |h| for (0..wt) |w| {
        ids[f * per + h * wt + w] = .{ t0 + @as(u32, @intCast(f)), @intCast(h), @intCast(w) };
    };
    for (0..cap_len) |i| ids[n_img + i] = .{ @intCast(1 + i), 0, 0 };
    return ids;
}

pub const Dit = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    w: Weights,
    x_embed: Linear,
    t1: Linear,
    t2: Linear,
    cap_norm: Array,
    cap_lin: Linear,
    noise: []Block,
    context: []Block,
    blocks: []Block,
    final_ada: Linear,
    final_lin: Linear,

    pub fn load(io: std.Io, a: std.mem.Allocator, s: S, model_dir: []const u8, cfg: Config) !Dit {
        const dir = try std.fmt.allocPrint(a, "{s}/transformer", .{model_dir});
        defer a.free(dir);
        var self: Dit = undefined;
        self.allocator = a;
        self.cfg = cfg;
        self.w = try weights_load.loadWeights(io, a, dir);
        errdefer self.w.deinit();
        try llm.castFloats(&self.w, DT, s);
        const w = &self.w;
        const d = cfg.dim;
        const p2 = PATCH * PATCH * cfg.in_ch;
        self.x_embed = try llm.lin(w, "all_x_embedder.2-1", .{}, p2);
        self.t1 = try llm.lin(w, "t_embedder.mlp.0", .{}, ADALN_DIM);
        self.t2 = try llm.lin(w, "t_embedder.mlp.2", .{}, @intCast(ops.shape(self.t1.w)[0]));
        self.cap_norm = try llm.get(w, "cap_embedder.0.weight", .{});
        self.cap_lin = try llm.lin(w, "cap_embedder.1", .{}, cfg.cap_dim);
        self.final_ada = try llm.lin(w, "all_final_layer.2-1.adaLN_modulation.1", .{}, @min(d, ADALN_DIM));
        self.final_lin = try llm.lin(w, "all_final_layer.2-1.linear", .{}, d);
        self.noise = try a.alloc(Block, cfg.refiners);
        errdefer a.free(self.noise);
        self.context = try a.alloc(Block, cfg.refiners);
        errdefer a.free(self.context);
        self.blocks = try a.alloc(Block, cfg.layers);
        errdefer a.free(self.blocks);
        for (self.noise, 0..) |*b, i| b.* = try Block.load(w, "noise_refiner", i, d, true);
        for (self.context, 0..) |*b, i| b.* = try Block.load(w, "context_refiner", i, d, false);
        for (self.blocks, 0..) |*b, i| b.* = try Block.load(w, "layers", i, d, true);
        var kb: [64]u8 = undefined;
        if (w.get(try std.fmt.bufPrint(&kb, "layers.{d}.ffn_norm1.weight", .{cfg.layers})) != null) {
            log.err("[ming-image] the DiT holds more blocks than its config's {d}\n", .{cfg.layers});
            return error.MingImageBadConfig;
        }
        try scope_mod.evalAll(w);
        return self;
    }

    pub fn deinit(self: *Dit) void {
        self.allocator.free(self.blocks);
        self.allocator.free(self.context);
        self.allocator.free(self.noise);
        self.w.deinit();
    }

    /// The caption the main blocks read: `[cap_embedder(learn) | direct]`, refined once per request.
    pub fn caption(self: *const Dit, parent: *Graph, learn: Array, direct: Array, rope: Rope) !Array {
        var g = parent.scope();
        defer g.deinit();
        const cap = try self.cap_lin.forward(&g, try g.rmsNorm(learn, self.cap_norm, self.cfg.eps));
        var x = try g.concat(&.{ cap, direct }, 1);
        for (self.context) |*b| x = try self.block(&g, b, x, null, rope);
        return parent.keep(x);
    }

    /// The time embedding (adaLN input) for model time `t` (`(1 − σ)·t_scale`), bf16 [1, 256].
    pub fn timeEmbed(self: *const Dit, parent: *Graph, t: f32) !Array {
        var g = parent.scope();
        defer g.deinit();
        var feats: [256]f32 = undefined;
        timestep_blk.sinusoid(f32, &feats, t, .cos_first);
        const x = try g.astype(try g.fromSlice(f32, &feats, &.{ 1, 256 }), DT);
        return parent.keep(try self.t2.forward(&g, try g.silu(try self.t1.forward(&g, x))));
    }

    /// Every modulated block's rows for one step: noise refiners first, then the main blocks.
    pub fn mods(self: *const Dit, g: *Graph, temb: Array, out: []Mod) !void {
        const all = self.noise.len + self.blocks.len;
        std.debug.assert(out.len == all);
        for (out, 0..) |*m, i| {
            const b = if (i < self.noise.len) &self.noise[i] else &self.blocks[i - self.noise.len];
            const rows = try g.expandDims(try b.ada.?.forward(g, temb), 1); // [1, 1, 4D]
            const d: c_int = @intCast(self.cfg.dim);
            m.* = .{
                .scale_a = try g.addScalar(try g.sliceAxis(rows, 2, 0, d), 1.0),
                .gate_a = try g.tanh(try g.sliceAxis(rows, 2, d, 2 * d)),
                .scale_f = try g.addScalar(try g.sliceAxis(rows, 2, 2 * d, 3 * d), 1.0),
                .gate_f = try g.tanh(try g.sliceAxis(rows, 2, 3 * d, 4 * d)),
            };
        }
    }

    fn attention(self: *const Dit, parent: *Graph, b: *const Block, y: Array, rope: Rope) !Array {
        var g = parent.scope();
        defer g.deinit();
        const h: c_int = @intCast(self.cfg.heads);
        const hd: c_int = @intCast(self.cfg.headDim());
        const heads = struct {
            fn f(c: *Graph, x: Array, n: c_int, d: c_int, norm: ?Array, eps: f32, r: Rope) !Array {
                const x4 = try c.reshape(x, &.{ 1, ops.shape(x)[1], n, d });
                const rot = if (norm) |nw| try rope_blk.ropePairs(c, try c.rmsNorm(x4, nw, eps), r.cos, r.sin, DT) else x4;
                return c.transpose(rot, &.{ 0, 2, 1, 3 });
            }
        }.f;
        const q = try heads(&g, try b.q.forward(&g, y), h, hd, b.norm_q, 1e-5, rope);
        const k = try heads(&g, try b.k.forward(&g, y), h, hd, b.norm_k, 1e-5, rope);
        const v = try heads(&g, try b.v.forward(&g, y), h, hd, null, 0, rope);
        const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hd)));
        return parent.keep(try b.o.forward(&g, try g.mergeHeads(try g.sdpa(q, k, v, scale, .none))));
    }

    fn ffn(parent: *Graph, b: *const Block, y: Array) !Array {
        var g = parent.scope();
        defer g.deinit();
        return parent.keep(try b.w2.forward(&g, try g.mul(try g.silu(try b.w1.forward(&g, y)), try b.w3.forward(&g, y))));
    }

    /// One sandwich-normed block; `m` null = unmodulated (context refiner).
    fn block(self: *const Dit, parent: *Graph, b: *const Block, x: Array, m: ?*const Mod, rope: Rope) !Array {
        var g = parent.scope();
        defer g.deinit();
        const eps = self.cfg.eps;
        var y = try g.rmsNorm(x, b.norm1, eps);
        if (m) |mm| y = try g.mul(y, mm.scale_a);
        var a = try g.rmsNorm(try self.attention(&g, b, y, rope), b.norm2, eps);
        if (m) |mm| a = try g.mul(a, mm.gate_a);
        const h = try g.add(x, a);
        var y2 = try g.rmsNorm(h, b.ffn_norm1, eps);
        if (m) |mm| y2 = try g.mul(y2, mm.scale_f);
        var f = try g.rmsNorm(try ffn(&g, b, y2), b.ffn_norm2, eps);
        if (m) |mm| f = try g.mul(f, mm.gate_f);
        return parent.keep(try g.add(h, f));
    }

    pub const Step = struct {
        /// Patchified latents [1, N, 4·C] f32 (an edit's reference patches after the target's), the refined caption,
        /// per-block mods, the unified table.
        patches: Array,
        caption: Array,
        mods: []const Mod,
        temb: Array,
        rope: Rope,
        n_img: c_int,
    };

    /// The model output for the image tokens [1, N, 4·C] f32.
    pub fn forward(self: *const Dit, parent: *Graph, st: Step) !Array {
        return dit_base.Dit(Hooks).forward(parent, self, st, st, null);
    }

    const Hooks = struct {
        pub const Model = Dit;
        pub const Input = Step;
        pub const Ctx = Step;

        pub fn embed(parent: *Graph, m: *const Dit, in: Step, _: Step) !Array {
            var g = parent.scope();
            defer g.deinit();
            const img_rope: Rope = .{ .cos = try g.sliceAxis(in.rope.cos, 1, 0, in.n_img), .sin = try g.sliceAxis(in.rope.sin, 1, 0, in.n_img) };
            var x = try m.x_embed.forward(&g, try g.astype(in.patches, DT));
            for (m.noise, 0..) |*b, i| {
                var c = g.scope();
                defer c.deinit();
                x = try g.keep(try m.block(&c, b, x, &in.mods[i], img_rope));
            }
            return parent.keep(try g.concat(&.{ x, in.caption }, 1));
        }
        pub fn layers(m: *const Dit) usize {
            return m.blocks.len;
        }
        pub fn block(g: *Graph, m: *const Dit, i: usize, x: Array, ctx: Step) !Array {
            return m.block(g, &m.blocks[i], x, &ctx.mods[m.noise.len + i], ctx.rope);
        }
        /// LayerNorm (no affine) · (1 + adaLN(silu(t))) → linear, over the image rows only.
        pub fn final(parent: *Graph, m: *const Dit, x: Array, ctx: Step) !Array {
            var g = parent.scope();
            defer g.deinit();
            const img = try g.sliceAxis(x, 1, 0, ctx.n_img);
            const scale = try g.addScalar(try g.expandDims(try m.final_ada.forward(&g, try g.silu(ctx.temb)), 1), 1.0);
            const n = try g.layerNorm(img, .{ .ctx = null }, .{ .ctx = null }, 1e-6);
            return parent.keep(try g.astype(try m.final_lin.forward(&g, try g.mul(n, scale)), .float32));
        }
    };
};

/// NCHW latents [1, C, H, W] → patches [1, (H/2)·(W/2), 2·2·C] in the release's (ph, pw, c) order.
pub fn patchify(parent: *Graph, lat: Array) !Array {
    var g = parent.scope();
    defer g.deinit();
    const sh = ops.shape(lat);
    const c = sh[1];
    const ht = @divExact(sh[2], 2);
    const wt = @divExact(sh[3], 2);
    const t = try g.transpose(try g.reshape(lat, &.{ c, ht, 2, wt, 2 }), &.{ 1, 3, 2, 4, 0 }); // [ht, wt, 2, 2, c]
    return parent.keep(try g.reshape(t, &.{ 1, ht * wt, 4 * c }));
}

/// The inverse of `patchify`.
pub fn unpatchify(parent: *Graph, x: Array, c: c_int, ht: c_int, wt: c_int) !Array {
    var g = parent.scope();
    defer g.deinit();
    const t = try g.transpose(try g.reshape(x, &.{ ht, wt, 2, 2, c }), &.{ 4, 0, 2, 1, 3 }); // [c, ht, 2, wt, 2]
    return parent.keep(try g.reshape(t, &.{ 1, c, 2 * ht, 2 * wt }));
}
