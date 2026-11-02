//! Idol's two decoders over the finished latent: the octree decoder (cross-attention only, modulated by the octree
//! level) places the anchor points by descending an 8-level octree, sampling how many of a cell's points fall in each
//! child; the Gaussian decoder (self + cross attention) turns every anchor into 32 Gaussians. Both read the 8192
//! latent tokens as context. Linear in f16 as the reference, norms and the output heads in f32.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const scope_mod = @import("../../mlx/scope.zig");
const L = @import("layers.zig");

const A = L.A;
const Scope = L.Scope;
const Linear = L.Linear;
const Weights = L.Weights;
const NONE = L.NONE;

pub const PER_ANCHOR: usize = 32;
/// Outputs per anchor: offset 3, colour 3, scale 3, rotation 4, opacity 1, offset scale 1, each x 32.
pub const OUT: usize = PER_ANCHOR * 15;

fn onesArray(gpa: std.mem.Allocator, owned: *std.ArrayList(A)) !A {
    const ones = try gpa.alloc(f32, L.HEAD_DIM);
    defer gpa.free(ones);
    @memset(ones, 1);
    const a = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{L.HEAD_DIM}, 1, .float32);
    try owned.append(gpa, a);
    return a;
}

/// Query points as the decoders embed them: `in_proj(p) + pointPosition(p)`, then `input_layer`, f32, `[1, n, C]`.
fn embedPoints(sc: *Scope, ch: u32, in_proj: Linear, input_layer: Linear, pts: []const f32) !A {
    const n: c_int = @intCast(pts.len / 3);
    const x = try sc.fromF32(pts, &.{ 1, n, 3 });
    const emb = try L.pointPosition(sc.a, pts, ch);
    defer sc.a.free(emb);
    const pe = try sc.fromF32(emb, &.{ 1, n, @intCast(ch) });
    return input_layer.forward(sc, try sc.add(try in_proj.forward(sc, x), pe));
}

const OBlock = struct {
    cross: L.Mha,
    mlp: L.Mlp,
};

/// A decoder's geometry, from `config.json`.
pub const Cfg = struct { model_channels: u32 = 1024, cond_channels: u32 = 16, num_blocks: u32 = 4 };

pub const Octree = struct {
    gpa: std.mem.Allocator,
    act: mlx.mlx_dtype,
    ch: u32,
    in_proj: Linear,
    input_layer: Linear,
    l_fc0: Linear,
    l_fc2: Linear,
    ada: Linear,
    out_proj: Linear,
    blocks: []OBlock,
    ones: A,
    owned: std.ArrayList(A) = .empty,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, act: mlx.mlx_dtype, cfg: Cfg) !Octree {
        const ch = cfg.model_channels;
        var o: Octree = .{
            .gpa = gpa, .act = act, .ch = ch,
            .in_proj = try scope_mod.linear(w, "in_proj", .{}, 3),
            .input_layer = try scope_mod.linear(w, "input_layer", .{}, ch),
            .l_fc0 = try scope_mod.linear(w, "l_embedder.mlp.0", .{}, 256),
            .l_fc2 = try scope_mod.linear(w, "l_embedder.mlp.2", .{}, ch),
            .ada = try scope_mod.linear(w, "adaLN_modulation.1", .{}, ch),
            .out_proj = try scope_mod.linear(w, "out_proj", .{}, ch),
            .blocks = try gpa.alloc(OBlock, cfg.num_blocks),
            .ones = undefined,
        };
        errdefer o.deinit();
        for (o.blocks, 0..) |*b, i| b.* = .{
            .cross = try L.Mha.load(w, "blocks.{d}.cross_attn", .{i}, ch, cfg.cond_channels, true),
            .mlp = try L.Mlp.load(w, "blocks.{d}.mlp", .{i}, ch, ch * 4),
        };
        o.ones = try onesArray(gpa, &o.owned);
        return o;
    }

    pub fn deinit(self: *Octree) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.blocks);
    }

    /// Logits `[1, n, 8]` f32 of how each cell's points split among its children: `centres` is `n` cell centres in
    /// [0, 1], `level` the cells' octree resolution (`1 << l`), `latent` `[1, 8192, 16]` f32.
    pub fn logits(self: *const Octree, sc: *Scope, centres: []const f32, level: f32, latent: A) !A {
        var h = try embedPoints(sc, self.ch, self.in_proj, self.input_layer, centres);
        var buf: [256]f32 = undefined;
        L.sinusoid(&buf, level, 1024, true);
        const e0 = try sc.fromF32(&buf, &.{ 1, 256 });
        const lemb = try self.l_fc2.forward(sc, try sc.silu(try self.l_fc0.forward(sc, e0)));
        const mod = try L.asDtype(sc, try self.ada.forward(sc, try sc.silu(lemb)), self.act);
        const shift_msa, const scale_msa, const gate_msa, const shift_mlp, const scale_mlp, const gate_mlp = try L.chunks6(sc, mod, self.ch);
        h = try L.asDtype(sc, h, self.act);
        const ctx = try L.asDtype(sc, latent, self.act);
        for (self.blocks) |*b| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const n1 = try L.modulate(&bs, try bs.layerNorm(h, NONE, NONE, L.LN_EPS), shift_msa, scale_msa);
            const x1 = try bs.add(h, try bs.mul(try b.cross.forward(&bs, n1, ctx, self.ones), gate_msa));
            const n2 = try L.modulate(&bs, try bs.layerNorm(x1, NONE, NONE, L.LN_EPS), shift_mlp, scale_mlp);
            const y = try bs.add(x1, try bs.mul(try b.mlp.forward(&bs, n2), gate_mlp));
            try mlx.check(mlx.mlx_array_eval(y));
            h = try sc.keep(bs.out(y));
        }
        const f = try sc.layerNorm(try L.asDtype(sc, h, .float32), NONE, NONE, L.FINAL_LN_EPS);
        return self.out_proj.forward(sc, f);
    }
};

/// Splits `count` items among the 8 bins of `probs` by systematic resampling: a random offset in [0, 1/count) and
/// `count` evenly spaced points walked through the CDF. Zero-sum rows fall back to uniform.
fn systematic(probs: *const [8]f32, count: u32, rng: std.Random, out: *[8]u32) void {
    @memset(out, 0);
    if (count == 0) return;
    var p: [8]f64 = undefined;
    var sum: f64 = 0;
    for (probs, 0..) |v, i| {
        p[i] = @max(0, @as(f64, v));
        sum += p[i];
    }
    for (&p) |*v| v.* = if (sum == 0) 1.0 / 8.0 else v.* / sum;
    var cdf: [8]f64 = undefined;
    var run: f64 = 0;
    for (p, 0..) |v, i| {
        run += v;
        cdf[i] = @min(run, 1.0 - 1e-12);
    }
    const n: f64 = @floatFromInt(count);
    const off0 = rng.float(f64) / n;
    var bin: usize = 0;
    var k: u32 = 0;
    while (k < count) : (k += 1) {
        const u = @min(off0 + @as(f64, @floatFromInt(k)) / n, 1.0 - 1e-12);
        while (bin < 7 and cdf[bin] < u) bin += 1;
        out[bin] += 1;
    }
}

pub const AnchorProgress = struct { ctx: ?*anyopaque = null, report: ?*const fn (?*anyopaque, u32, u32) void = null };

/// `num_points` anchor points `[n * 3]` in [0, 1]^3, the octree descended to `level` (caller frees).
pub fn sampleAnchors(gpa: std.mem.Allocator, s: mlx.mlx_stream, oct: *const Octree, latent: A, num_points: u32, level: u32, seed: u64, prog: AnchorProgress) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var coords: std.ArrayList([3]u32) = .empty;
    defer coords.deinit(gpa);
    var counts: std.ArrayList(u32) = .empty;
    defer counts.deinit(gpa);
    try coords.append(gpa, .{ 0, 0, 0 });
    try counts.append(gpa, num_points);
    var next_c: std.ArrayList([3]u32) = .empty;
    defer next_c.deinit(gpa);
    var next_n: std.ArrayList(u32) = .empty;
    defer next_n.deinit(gpa);
    var l: u32 = 1;
    while (l <= level) : (l += 1) {
        const res_p: f32 = @floatFromInt(@as(u32, 1) << @intCast(l - 1));
        const centres = try gpa.alloc(f32, coords.items.len * 3);
        defer gpa.free(centres);
        for (coords.items, 0..) |c, i| for (0..3) |a| {
            centres[i * 3 + a] = (@as(f32, @floatFromInt(c[a])) + 0.5) / res_p;
        };
        var sc = Scope.init(gpa, s);
        defer sc.deinit();
        const lg = try oct.logits(&sc, centres, @floatFromInt(@as(u32, 1) << @intCast(l)), latent);
        try mlx.check(mlx.mlx_array_eval(lg));
        const host = mlx.mlx_array_data_float32(lg) orelse return error.NoData;
        next_c.clearRetainingCapacity();
        next_n.clearRetainingCapacity();
        for (coords.items, counts.items, 0..) |c, n, i| {
            var pr: [8]f32 = undefined;
            var mx: f32 = -std.math.inf(f32);
            for (0..8) |k| mx = @max(mx, host[i * 8 + k]);
            var sum: f32 = 0;
            for (0..8) |k| {
                pr[k] = @exp(host[i * 8 + k] - mx);
                sum += pr[k];
            }
            for (&pr) |*v| v.* /= sum;
            var got: [8]u32 = undefined;
            systematic(&pr, n, rng, &got);
            for (0..8) |k| {
                if (got[k] == 0) continue;
                try next_c.append(gpa, .{ c[0] * 2 + @as(u32, @intCast(k & 1)), c[1] * 2 + @as(u32, @intCast((k >> 1) & 1)), c[2] * 2 + @as(u32, @intCast(k >> 2)) });
                try next_n.append(gpa, got[k]);
            }
        }
        std.mem.swap(std.ArrayList([3]u32), &coords, &next_c);
        std.mem.swap(std.ArrayList(u32), &counts, &next_n);
        if (prog.report) |f| f(prog.ctx, l, level);
    }
    const out = try gpa.alloc(f32, num_points * 3);
    errdefer gpa.free(out);
    const scale: f32 = @floatFromInt(@as(u32, 1) << @intCast(level));
    var o: usize = 0;
    for (coords.items, counts.items) |c, n| {
        for (0..n) |_| {
            for (0..3) |a| {
                out[o] = (@as(f32, @floatFromInt(c[a])) + rng.float(f32)) / scale;
                o += 1;
            }
        }
    }
    if (o != out.len) return error.BadAnchorCount;
    return out;
}

const GBlock = struct {
    norm2_w: A,
    norm2_b: A,
    self: L.Mha,
    cross: L.Mha,
    mlp: L.Mlp,
};

pub const Rep = struct {
    lr_xyz: f32 = 1.0,
    lr_dc: f32 = 1.0,
    lr_scaling: f32 = 1.0,
    lr_rotation: f32 = 0.1,
    lr_opacity: f32 = 1.0,
    perturbe_size: f32 = 1.5,
    kernel: f32 = 0.0009,
    scaling_bias: f32 = 0.004,
    opacity_bias: f32 = 0.1,
};

pub const GaussianDecoder = struct {
    gpa: std.mem.Allocator,
    act: mlx.mlx_dtype,
    ch: u32,
    in_proj: Linear,
    input_layer: Linear,
    out_proj: Linear,
    blocks: []GBlock,
    ones: A,
    /// `points_offset_perturbation` `[32 * 3]` and `base_offset_scale`, on the host.
    perturbation: [PER_ANCHOR * 3]f32,
    base_offset_scale: f32,
    owned: std.ArrayList(A) = .empty,

    pub fn load(gpa: std.mem.Allocator, w: *const Weights, act: mlx.mlx_dtype, cfg: Cfg) !GaussianDecoder {
        const ch = cfg.model_channels;
        var d: GaussianDecoder = .{
            .gpa = gpa, .act = act, .ch = ch,
            .in_proj = try scope_mod.linear(w, "in_proj", .{}, 3),
            .input_layer = try scope_mod.linear(w, "input_layer", .{}, ch),
            .out_proj = try scope_mod.linear(w, "out_proj", .{}, ch),
            .blocks = try gpa.alloc(GBlock, cfg.num_blocks),
            .ones = undefined,
            .perturbation = undefined,
            .base_offset_scale = 0,
        };
        errdefer d.deinit();
        for (d.blocks, 0..) |*b, i| b.* = .{
            .norm2_w = try scope_mod.getW(w, "blocks.{d}.norm2.weight", .{i}),
            .norm2_b = try scope_mod.getW(w, "blocks.{d}.norm2.bias", .{i}),
            .self = try L.Mha.load(w, "blocks.{d}.self_attn", .{i}, ch, 0, false),
            .cross = try L.Mha.load(w, "blocks.{d}.cross_attn", .{i}, ch, cfg.cond_channels, true),
            .mlp = try L.Mlp.load(w, "blocks.{d}.mlp", .{i}, ch, ch * 4),
        };
        d.ones = try onesArray(gpa, &d.owned);
        const pp = try scope_mod.getW(w, "points_offset_perturbation", .{});
        const bs = try scope_mod.getW(w, "base_offset_scale", .{});
        try mlx.check(mlx.mlx_array_eval(pp));
        try mlx.check(mlx.mlx_array_eval(bs));
        @memcpy(&d.perturbation, (mlx.mlx_array_data_float32(pp) orelse return error.NoData)[0..d.perturbation.len]);
        d.base_offset_scale = (mlx.mlx_array_data_float32(bs) orelse return error.NoData)[0];
        return d;
    }

    pub fn deinit(self: *GaussianDecoder) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.blocks);
    }

    /// Per-anchor outputs `[1, n, 480]` f32 for `points` (`n * 3`), `latent` `[1, 8192, 16]` f32; owned by `sc`.
    pub fn forward(self: *const GaussianDecoder, sc: *Scope, points: []const f32, latent: A) !A {
        var h = try L.asDtype(sc, try embedPoints(sc, self.ch, self.in_proj, self.input_layer, points), self.act);
        const ctx = try L.asDtype(sc, latent, self.act);
        for (self.blocks) |*b| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const x1 = try bs.add(h, try b.self.forward(&bs, try bs.layerNorm(h, NONE, NONE, L.LN_EPS), null, self.ones));
            const n2 = try L.asDtype(&bs, try bs.layerNorm(try L.asDtype(&bs, x1, .float32), b.norm2_w, b.norm2_b, L.LN_EPS), self.act);
            const x2 = try bs.add(x1, try b.cross.forward(&bs, n2, ctx, self.ones));
            const y = try bs.add(x2, try b.mlp.forward(&bs, try bs.layerNorm(x2, NONE, NONE, L.LN_EPS)));
            try mlx.check(mlx.mlx_array_eval(y));
            h = try sc.keep(bs.out(y));
        }
        return self.out_proj.forward(sc, try sc.layerNorm(try L.asDtype(sc, h, .float32), NONE, NONE, L.FINAL_LN_EPS));
    }
};

/// The decoded Gaussians, activated: the object is z-up inside [-0.5, 0.5]^3.
pub const Splats = struct {
    n: usize,
    xyz: []f32,
    /// Degree-0 SH colour coefficients.
    dc: []f32,
    /// Standard deviations per axis.
    scale: []f32,
    /// Unit quaternions (w, x, y, z).
    rot: []f32,
    opacity: []f32,

    pub fn deinit(self: *Splats, gpa: std.mem.Allocator) void {
        gpa.free(self.xyz);
        gpa.free(self.dc);
        gpa.free(self.scale);
        gpa.free(self.rot);
        gpa.free(self.opacity);
    }
};

fn softplus(x: f32) f32 {
    return if (x > 20) x else std.math.log1p(@exp(x));
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `GaussianFixedlenDecoder.to_representation` + `Gaussian`: `feats` `[n_anchors * 480]` and the anchors `points`.
pub fn toSplats(gpa: std.mem.Allocator, dec: *const GaussianDecoder, rep: Rep, points: []const f32, feats: []const f32) !Splats {
    const anchors = points.len / 3;
    const n = anchors * PER_ANCHOR;
    var s: Splats = .{
        .n = n,
        .xyz = try gpa.alloc(f32, n * 3),
        .dc = try gpa.alloc(f32, n * 3),
        .scale = try gpa.alloc(f32, n * 3),
        .rot = try gpa.alloc(f32, n * 4),
        .opacity = try gpa.alloc(f32, n),
    };
    errdefer s.deinit(gpa);
    const sp_inv = rep.scaling_bias + @log(-std.math.expm1(-rep.scaling_bias));
    const op_inv = @log(rep.opacity_bias / (1 - rep.opacity_bias));
    const k2 = rep.kernel * rep.kernel;
    for (0..anchors) |a| {
        const f = feats[a * OUT ..][0..OUT];
        for (0..PER_ANCHOR) |j| {
            const g = a * PER_ANCHOR + j;
            const off_scale = softplus(f[448 + j] + dec.base_offset_scale);
            for (0..3) |k| {
                const o = std.math.tanh(f[j * 3 + k] * rep.lr_xyz + dec.perturbation[j * 3 + k]) * 0.5 * rep.perturbe_size * off_scale;
                s.xyz[g * 3 + k] = (o + points[a * 3 + k]) - 0.5;
                s.dc[g * 3 + k] = f[96 + j * 3 + k] * rep.lr_dc;
                const sc = softplus(f[192 + j * 3 + k] * rep.lr_scaling + sp_inv);
                s.scale[g * 3 + k] = @sqrt(sc * sc + k2);
            }
            var q: [4]f32 = undefined;
            var norm: f32 = 0;
            for (0..4) |k| {
                q[k] = f[288 + j * 4 + k] * rep.lr_rotation + (if (k == 0) @as(f32, 1) else 0);
                norm += q[k] * q[k];
            }
            const inv = 1.0 / @max(@sqrt(norm), 1e-12);
            for (0..4) |k| s.rot[g * 4 + k] = q[k] * inv;
            s.opacity[g] = sigmoid(f[416 + j] * rep.lr_opacity + op_inv);
        }
    }
    return s;
}

// ── tests ──

const testing = std.testing;

test "idol octree: systematic sampling splits every point, follows the probabilities and never fills an empty child" {
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    var got: [8]u32 = undefined;
    const probs = [8]f32{ 0.5, 0.25, 0.25, 0, 0, 0, 0, 0 };
    for ([_]u32{ 1, 7, 64, 1000, 262144 }) |n| {
        systematic(&probs, n, rng, &got);
        var sum: u32 = 0;
        for (got) |c| sum += c;
        try testing.expectEqual(n, sum);
        for (got[3..]) |c| try testing.expectEqual(@as(u32, 0), c);
    }
    // Stratified: with 1000 points the shares sit within one point of 500 / 250 / 250.
    systematic(&probs, 1000, rng, &got);
    try testing.expect(@abs(@as(i64, got[0]) - 500) <= 1 and @abs(@as(i64, got[1]) - 250) <= 1);
    // An all-zero row falls back to uniform rather than dropping its points.
    const zero: [8]f32 = @splat(0);
    systematic(&zero, 800, rng, &got);
    for (got) |c| try testing.expect(c >= 99 and c <= 101);
}
