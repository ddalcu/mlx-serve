//! TRELLIS.2-4B (Microsoft, MIT): single-image to textured 3D (`.mesh` modality, `trellis2`).
//!
//!   image -> DINOv3 ViT-L/16 tokens (512 and 1024 px)
//!   -> sparse-structure flow DiT (16^3 x 8 latent) -> conv3d decoder -> occupied 32^3 / 64^3 cells
//!   -> shape SLat flow DiT over those voxels (512; then a 1024 pass for the cascade)
//!   -> shape decoder (subdivision + dual-grid vertices) -> texture SLat flow DiT (conditioned on the
//!      shape latent) -> texture decoder (PBR attributes) -> mesh + GLB.
//!
//! One DiT serves all three flows: a dense 16^3 grid and a sparse voxel list are both a token
//! sequence, differing only in the per-token coordinates the 3-axis RoPE reads. Math follows
//! microsoft/TRELLIS.2 (`SparseStructureFlowModel`, `SLatFlowModel`, `FlowEulerGuidanceIntervalSampler`).
//! Parity: env-gated `TRELLIS2_*` oracles fed by tests/dump_trellis2_fixtures.py.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const scope_mod = @import("mlx_scope.zig");
const sparse = @import("trellis2_sparse.zig");

const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const Scope = scope_mod.Scope;
const Linear = scope_mod.Linear;
const Weights = model_mod.Weights;
const NONE = A{ .ctx = null };

const LN_EPS: f32 = 1e-6;
const HEAD_NORM_EPS: f32 = 1e-12;
const FINAL_LN_EPS: f32 = 1e-5;
const TIME_FREQS: usize = 256;
const SIGMA_MIN: f64 = 1e-5;

// ── RoPE ─────────────────────────────────────────────────────────────────────

/// Per-token rotation tables for 3-axis RoPE over adjacent channel pairs: a head of width D has
/// D/2 pairs, `D/2/3` frequencies per axis (x, y, z in order) and the rest identity.
/// `cos` and `sin` are `[L, 1, D]` f32 with each pair's value repeated and the sign of the first
/// element of `sin` folded in, so `rope(x) = x * cos + swap_pairs(x) * sin`.
pub const Rope = struct {
    cos: A,
    sin: A,

    /// `coords` `[L, 4]` (batch, x, y, z).
    pub fn init(sc: *Scope, coords: []const i32, head_dim: usize) !Rope {
        if (head_dim % 2 != 0) return error.BadHeadDim;
        const n = coords.len / 4;
        const pairs = head_dim / 2;
        const per_axis = pairs / 3;
        const cos_t = try sc.a.alloc(f32, n * head_dim);
        defer sc.a.free(cos_t);
        const sin_t = try sc.a.alloc(f32, n * head_dim);
        defer sc.a.free(sin_t);
        for (0..n) |l| {
            for (0..pairs) |p| {
                var phase: f32 = 0;
                if (p < 3 * per_axis) {
                    const axis = p / per_axis;
                    const j = p % per_axis;
                    const e = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(per_axis));
                    const freq = 1.0 / std.math.pow(f32, 10000.0, e);
                    phase = @as(f32, @floatFromInt(coords[l * 4 + 1 + axis])) * freq;
                }
                const c = @cos(phase);
                const s = @sin(phase);
                cos_t[l * head_dim + 2 * p] = c;
                cos_t[l * head_dim + 2 * p + 1] = c;
                sin_t[l * head_dim + 2 * p] = -s;
                sin_t[l * head_dim + 2 * p + 1] = s;
            }
        }
        const shape = [_]c_int{ @intCast(n), 1, @intCast(head_dim) };
        return .{ .cos = try sc.fromF32(cos_t, &shape), .sin = try sc.fromF32(sin_t, &shape) };
    }

    /// `x` `[1, L, H, D]` f32.
    fn apply(self: Rope, sc: *Scope, x: A) !A {
        const sh = mlx.getShape(x);
        const pair_shape = [_]c_int{ sh[0], sh[1], sh[2], @divExact(sh[3], 2), 2 };
        const flip = try sc.fromI32(&.{ 1, 0 }, &.{2});
        const swapped = try sc.reshape(try takeAxis(sc, try sc.reshape(x, &pair_shape), flip, 4), sh);
        return sc.add(try sc.mul(x, self.cos), try sc.mul(swapped, self.sin));
    }
};

/// The `[L, 4]` coordinates of a dense `res`^3 grid in the order a `[C, res, res, res]` tensor flattens (x slowest).
pub fn gridCoords(gpa: std.mem.Allocator, res: usize) ![]i32 {
    const out = try gpa.alloc(i32, res * res * res * 4);
    var i: usize = 0;
    for (0..res) |x| for (0..res) |y| for (0..res) |z| {
        out[i * 4] = 0;
        out[i * 4 + 1] = @intCast(x);
        out[i * 4 + 2] = @intCast(y);
        out[i * 4 + 3] = @intCast(z);
        i += 1;
    };
    return out;
}

fn takeAxis(sc: *Scope, x: A, ids: A, axis: c_int) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_take_axis(&o, x, ids, axis, sc.s), &o);
}

// ── Blocks ───────────────────────────────────────────────────────────────────

fn geluTanh(sc: *Scope, x: A) !A {
    const x3 = try sc.mul(try sc.mul(x, x), x);
    const inner = try sc.mulLike(try sc.add(x, try sc.mulLike(x3, 0.044715)), @sqrt(2.0 / std.math.pi));
    return sc.mul(try sc.mulLike(x, 0.5), try sc.add(try sc.tanh(inner), try sc.scalarLike(x, 1)));
}

fn layerNorm(sc: *Scope, x: A, w: A, b: A, eps: f32) !A {
    return sc.layerNorm(x, w, b, eps);
}

/// `x` cast to `like`'s dtype.
fn asLike(sc: *Scope, x: A, like: A) !A {
    const dt = mlx.mlx_array_dtype(like);
    return if (mlx.mlx_array_dtype(x) == dt) x else sc.astype(x, dt);
}

/// A projection's input width. A dense weight's columns are the width; a quantized one packs `bits` per
/// element, so only its scales' columns times the group size recover it.
fn inWidth(weight: A, scales: ?A, group_size: u32) usize {
    const cols: usize = @intCast(mlx.getShape(scales orelse weight)[1]);
    return if (scales != null) cols * group_size else cols;
}

const Block = struct {
    modulation: A,
    norm2_w: A,
    norm2_b: A,
    qkv: Linear,
    self_out: Linear,
    self_q_gamma: A,
    self_k_gamma: A,
    cross_q: Linear,
    cross_kv: Linear,
    cross_out: Linear,
    cross_q_gamma: A,
    cross_k_gamma: A,
    fc1: Linear,
    fc2: Linear,
    /// Pixal3D: the block's projection of the pixel-aligned features, added to the cross-attention output.
    proj: ?Linear,

    fn load(w: *const Weights, i: usize, channels: usize, cond_ch: usize, mlp: usize, group_size: u32) !Block {
        const c: u32 = @intCast(channels);
        var key: [96]u8 = undefined;
        const proj_w = w.get(try std.fmt.bufPrint(&key, "blocks.{d}.cross_attn.proj_linear.weight", .{i}));
        const proj_scales = w.get(try std.fmt.bufPrint(&key, "blocks.{d}.cross_attn.proj_linear.scales", .{i}));
        // Pixal3D nests the cross-attention one level down, beside its proj_linear.
        var pre_buf: [64]u8 = undefined;
        const cross = try std.fmt.bufPrint(&pre_buf, "blocks.{d}.cross_attn{s}", .{ i, if (proj_w != null) ".cross_attn_block" else "" });
        return .{
            .modulation = try scope_mod.getW(w, "blocks.{d}.modulation", .{i}),
            .norm2_w = try scope_mod.getW(w, "blocks.{d}.norm2.weight", .{i}),
            .norm2_b = try scope_mod.getW(w, "blocks.{d}.norm2.bias", .{i}),
            .qkv = try scope_mod.linear(w, "blocks.{d}.self_attn.to_qkv", .{i}, c),
            .self_out = try scope_mod.linear(w, "blocks.{d}.self_attn.to_out", .{i}, c),
            .self_q_gamma = try scope_mod.getW(w, "blocks.{d}.self_attn.q_rms_norm.gamma", .{i}),
            .self_k_gamma = try scope_mod.getW(w, "blocks.{d}.self_attn.k_rms_norm.gamma", .{i}),
            .cross_q = try scope_mod.linear(w, "{s}.to_q", .{cross}, c),
            .cross_kv = try scope_mod.linear(w, "{s}.to_kv", .{cross}, @intCast(cond_ch)),
            .cross_out = try scope_mod.linear(w, "{s}.to_out", .{cross}, c),
            .cross_q_gamma = try scope_mod.getW(w, "{s}.q_rms_norm.gamma", .{cross}),
            .cross_k_gamma = try scope_mod.getW(w, "{s}.k_rms_norm.gamma", .{cross}),
            .fc1 = try scope_mod.linear(w, "blocks.{d}.mlp.mlp.0", .{i}, c),
            .fc2 = try scope_mod.linear(w, "blocks.{d}.mlp.mlp.2", .{i}, @intCast(mlp)),
            .proj = if (proj_w) |pw| try scope_mod.linear(w, "blocks.{d}.cross_attn.proj_linear", .{i}, @intCast(inWidth(pw, proj_scales, group_size))) else null,
        };
    }
};

/// The pixel-aligned features a Pixal3D block projects (`[1, L, C']`, one row per token), or `zero`: the
/// negative branch's all-zero features, whose projection is the bias alone.
pub const Proj = union(enum) {
    none,
    zero,
    feats: A,
};

/// Intermediate activations of one forward, for the oracle's bisect.
pub const Taps = struct {
    h_in: A = NONE,
    blk0: A = NONE,
    blk_last: A = NONE,
};

pub const Dit = struct {
    gpa: std.mem.Allocator,
    act: mlx.mlx_dtype,
    channels: usize,
    heads: usize,
    head_dim: usize,
    in_ch: usize,
    out_ch: usize,
    input: Linear,
    t_fc0: Linear,
    t_fc2: Linear,
    ada: Linear,
    out: Linear,
    blocks: []Block,
    /// All-ones `[head_dim]`, the weight of the per-head L2 norm (gamma is applied after).
    ones: A,
    owned: std.ArrayList(A) = .empty,

    /// `act`: dtype of the blocks (bf16 as published; f32 for the oracle bars). The input, time and
    /// output layers always run f32, like the reference's unconverted layers.
    pub fn load(gpa: std.mem.Allocator, w: *const Weights, act: mlx.mlx_dtype, group_size: u32) !Dit {
        var blocks_n: usize = 0;
        var key: [64]u8 = undefined;
        while (w.get(try std.fmt.bufPrint(&key, "blocks.{d}.modulation", .{blocks_n})) != null) blocks_n += 1;
        if (blocks_n == 0) return error.BadDit;
        const gamma = try scope_mod.getW(w, "blocks.0.self_attn.q_rms_norm.gamma", .{});
        const heads: usize = @intCast(mlx.getShape(gamma)[0]);
        const head_dim: usize = @intCast(mlx.getShape(gamma)[1]);
        const in_w = mlx.getShape(try scope_mod.getW(w, "input_layer.weight", .{}));
        const out_w = mlx.getShape(try scope_mod.getW(w, "out_layer.weight", .{}));
        const kv_pre = if (w.get("blocks.0.cross_attn.to_kv.weight") != null) "blocks.0.cross_attn.to_kv" else "blocks.0.cross_attn.cross_attn_block.to_kv";
        const kv_in = inWidth(try scope_mod.getW(w, "{s}.weight", .{kv_pre}), w.get(try std.fmt.bufPrint(&key, "{s}.scales", .{kv_pre})), group_size);
        const fc_w = mlx.getShape(try scope_mod.getW(w, "blocks.0.mlp.mlp.0.weight", .{}));
        var d = Dit{
            .gpa = gpa,
            .act = act,
            .channels = @intCast(in_w[0]),
            .heads = heads,
            .head_dim = head_dim,
            .in_ch = @intCast(in_w[1]),
            .out_ch = @intCast(out_w[0]),
            .input = undefined,
            .t_fc0 = undefined,
            .t_fc2 = undefined,
            .ada = undefined,
            .out = undefined,
            .blocks = try gpa.alloc(Block, 0),
            .ones = undefined,
        };
        if (d.channels != heads * head_dim) return error.BadDit;
        errdefer d.deinit();
        d.input = try sparse.f32LinearOwned(gpa, &d.owned, w, "input_layer");
        d.t_fc0 = try sparse.f32LinearOwned(gpa, &d.owned, w, "t_embedder.mlp.0");
        d.t_fc2 = try sparse.f32LinearOwned(gpa, &d.owned, w, "t_embedder.mlp.2");
        d.ada = try sparse.f32LinearOwned(gpa, &d.owned, w, "adaLN_modulation.1");
        d.out = try sparse.f32LinearOwned(gpa, &d.owned, w, "out_layer");
        const ones_host = try gpa.alloc(f32, head_dim);
        defer gpa.free(ones_host);
        @memset(ones_host, 1);
        d.ones = mlx.mlx_array_new_data(ones_host.ptr, &[_]c_int{@intCast(head_dim)}, 1, .float32);
        try d.owned.append(gpa, d.ones);
        gpa.free(d.blocks);
        d.blocks = try gpa.alloc(Block, blocks_n);
        for (d.blocks, 0..) |*b, i| b.* = try Block.load(w, i, d.channels, kv_in, @intCast(fc_w[0]), group_size);
        return d;
    }

    pub fn deinit(self: *Dit) void {
        self.gpa.free(self.blocks);
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
    }

    /// `[1, 256]` sinusoid of `t` (cos then sin) through the two-layer time MLP.
    fn timeEmbedding(self: *const Dit, sc: *Scope, t: f32) !A {
        var buf: [TIME_FREQS]f32 = undefined;
        const half = TIME_FREQS / 2;
        for (0..half) |i| {
            const freq = @exp(-@as(f32, @floatCast(@log(10000.0))) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
            const arg = t * freq;
            buf[i] = @cos(arg);
            buf[half + i] = @sin(arg);
        }
        const e = try sc.fromF32(&buf, &.{ 1, TIME_FREQS });
        return self.t_fc2.forward(sc, try sc.silu(try self.t_fc0.forward(sc, e)));
    }

    /// `[1, L, H, D]` f32: the per-head L2 norm scaled by `gamma [H, D]`.
    fn headNorm(self: *const Dit, sc: *Scope, x: A, gamma: A) !A {
        var o = mlx.mlx_array_new();
        const n = try sc.res(mlx.mlx_fast_rms_norm(&o, try sc.astype(x, .float32), self.ones, HEAD_NORM_EPS, sc.s), &o);
        return sc.mul(n, try sc.astype(gamma, .float32));
    }

    /// Attention over `[1, L, H, D]` q and `[1, Lk, H, D]` k, v; returns `[1, L, C]`.
    fn attend(self: *const Dit, sc: *Scope, q: A, k: A, v: A) !A {
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const o = try sc.sdpa(try sc.transpose(q, &perm), try sc.transpose(k, &perm), try sc.transpose(v, &perm), scale);
        const sh = mlx.getShape(q);
        return sc.reshape(try sc.transpose(o, &perm), &.{ sh[0], sh[1], @intCast(self.channels) });
    }

    fn heads4(self: *const Dit, sc: *Scope, x: A, rows: c_int) !A {
        return sc.reshape(x, &.{ 1, rows, @intCast(self.heads), @intCast(self.head_dim) });
    }

    fn block(self: *const Dit, sc: *Scope, b: *const Block, x_in: A, mod: A, cond: A, proj: Proj, rope: Rope) !A {
        const c: c_int = @intCast(self.channels);
        const l: c_int = mlx.getShape(x_in)[1];
        const m = try sc.add(try asLike(sc, b.modulation, mod), mod);
        var parts: [6]A = undefined;
        for (&parts, 0..) |*p, i| p.* = try sc.slice(m, 1, @intCast(i * self.channels), @intCast((i + 1) * self.channels));
        const shift_msa, const scale_msa, const gate_msa, const shift_mlp, const scale_mlp, const gate_mlp = parts;
        const one = try sc.scalarLike(m, 1);

        var h = try modulate(sc, try layerNorm(sc, x_in, NONE, NONE, LN_EPS), shift_msa, scale_msa, one);
        const qkv = try sc.reshape(try b.qkv.forward(sc, h), &.{ 1, l, 3, @intCast(self.heads), @intCast(self.head_dim) });
        const q = try sc.astype(try rope.apply(sc, try self.headNorm(sc, try sc.reshape(try sc.slice(qkv, 2, 0, 1), &.{ 1, l, @intCast(self.heads), @intCast(self.head_dim) }), b.self_q_gamma)), self.act);
        const k = try sc.astype(try rope.apply(sc, try self.headNorm(sc, try sc.reshape(try sc.slice(qkv, 2, 1, 2), &.{ 1, l, @intCast(self.heads), @intCast(self.head_dim) }), b.self_k_gamma)), self.act);
        const v = try sc.reshape(try sc.slice(qkv, 2, 2, 3), &.{ 1, l, @intCast(self.heads), @intCast(self.head_dim) });
        var x = try sc.add(x_in, try sc.mul(try b.self_out.forward(sc, try self.attend(sc, q, k, v)), gate_msa));

        h = try layerNorm(sc, x, try asLike(sc, b.norm2_w, x), try asLike(sc, b.norm2_b, x), LN_EPS);
        const lc: c_int = mlx.getShape(cond)[1];
        const cq = try sc.astype(try self.headNorm(sc, try self.heads4(sc, try b.cross_q.forward(sc, h), l), b.cross_q_gamma), self.act);
        const ckv = try sc.reshape(try b.cross_kv.forward(sc, cond), &.{ 1, lc, 2, @intCast(self.heads), @intCast(self.head_dim) });
        const ck = try sc.astype(try self.headNorm(sc, try sc.reshape(try sc.slice(ckv, 2, 0, 1), &.{ 1, lc, @intCast(self.heads), @intCast(self.head_dim) }), b.cross_k_gamma), self.act);
        const cv = try sc.reshape(try sc.slice(ckv, 2, 1, 2), &.{ 1, lc, @intCast(self.heads), @intCast(self.head_dim) });
        var cross = try b.cross_out.forward(sc, try self.attend(sc, cq, ck, cv));
        switch (proj) {
            .none => if (b.proj != null) return error.MissingProjFeatures,
            .zero => cross = try sc.add(cross, try asLike(sc, (b.proj orelse return error.NotAProjDit).bias.?, cross)),
            .feats => |p| cross = try sc.add(cross, try (b.proj orelse return error.NotAProjDit).forward(sc, p)),
        }
        x = try sc.add(x, cross);

        h = try modulate(sc, try layerNorm(sc, x, NONE, NONE, LN_EPS), shift_mlp, scale_mlp, one);
        h = try b.fc2.forward(sc, try geluTanh(sc, try b.fc1.forward(sc, h)));
        _ = c;
        return sc.add(x, try sc.mul(h, gate_mlp));
    }

    /// One velocity prediction. `x` `[1, L, in]` f32, `cond` `[1, Lc, cond_ch]`, `t` the model
    /// timestep (1000 * flow time). Returns `[1, L, out]` f32.
    pub fn forward(self: *const Dit, sc: *Scope, x: A, t: f32, cond: A, proj_in: Proj, rope: Rope, taps: ?*Taps) !A {
        var h = try sc.astype(try self.input.forward(sc, x), self.act);
        if (taps) |tp| tp.h_in = h;
        const mod = try sc.astype(try self.ada.forward(sc, try sc.silu(try self.timeEmbedding(sc, t))), self.act);
        const cond_a = try sc.astype(cond, self.act);
        const proj: Proj = switch (proj_in) {
            .feats => |p| .{ .feats = try sc.astype(p, self.act) },
            else => proj_in,
        };
        // Each block in its own scope, evaluated and freed: at tens of thousands of tokens one block's
        // intermediates are gigabytes, and thirty of them kept alive until the step ends would not fit.
        var cur: ?A = null;
        errdefer if (cur) |c| {
            _ = mlx.mlx_array_free(c);
        };
        for (self.blocks, 0..) |*b, i| {
            var bsc = Scope.init(sc.a, sc.s);
            defer bsc.deinit();
            const y = try self.block(&bsc, b, h, mod, cond_a, proj, rope);
            try mlx.check(mlx.mlx_array_eval(y));
            if (taps) |tp| {
                if (i == 0) tp.blk0 = try sc.keep(bsc.out(y));
                if (i + 1 == self.blocks.len) tp.blk_last = try sc.keep(bsc.out(y));
            }
            if (cur) |c| _ = mlx.mlx_array_free(c);
            cur = bsc.out(y);
            h = cur.?;
        }
        h = try sc.keep(cur.?);
        cur = null;
        h = try layerNorm(sc, try sc.astype(h, .float32), NONE, NONE, FINAL_LN_EPS);
        return self.out.forward(sc, h);
    }
};

/// `x * (1 + scale) + shift`, in the blocks' dtype and the reference's order.
fn modulate(sc: *Scope, x: A, shift: A, scale: A, one: A) !A {
    return sc.add(try sc.mul(x, try sc.add(one, scale)), shift);
}

pub const OpenedDit = struct {
    w: Weights,
    dit: Dit,

    pub fn deinit(self: *OpenedDit) void {
        self.dit.deinit();
        self.w.deinit();
    }
};

/// `<dir>/<stem>.safetensors` as published in `ckpts/`. Only the published recipe is served:
/// RoPE positions, one shared modulation, QK norms on both attentions.
pub fn openDit(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, stem: []const u8, act: mlx.mlx_dtype) !OpenedDit {
    const json_path = try std.fmt.allocPrint(gpa, "{s}/{s}.json", .{ dir, stem });
    defer gpa.free(json_path);
    const bytes = try readFile(io, gpa, json_path);
    defer gpa.free(bytes);
    const Raw = struct {
        args: struct { pe_mode: []const u8 = "ape", share_mod: bool = false, qk_rms_norm: bool = false, qk_rms_norm_cross: bool = false },
        quantization: ?struct { group_size: u32 = 0 } = null,
    };
    var parsed = try std.json.parseFromSlice(Raw, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const a = parsed.value.args;
    if (!std.mem.eql(u8, a.pe_mode, "rope") or !a.share_mod or !a.qk_rms_norm or !a.qk_rms_norm_cross) return error.UnsupportedTrellis2Dit;
    const file = try std.fmt.allocPrint(gpa, "{s}.safetensors", .{stem});
    defer gpa.free(file);
    var w = try model_mod.loadWeightsFile(gpa, dir, file);
    errdefer w.deinit();
    if (act != .float32) try castBlocks(&w, act);
    const dit = try Dit.load(gpa, &w, act, if (parsed.value.quantization) |q| q.group_size else 0);
    return .{ .w = w, .dit = dit };
}

/// Pixal3D publishes its DiTs in f32; the reference converts the blocks to the model dtype at load, so they
/// sit resident at half the size. One tensor at a time keeps the peak at one f32 tensor beside the rest.
fn castBlocks(w: *Weights, dt: mlx.mlx_dtype) !void {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var it = w.map.iterator();
    while (it.next()) |e| {
        if (!std.mem.startsWith(u8, e.key_ptr.*, "blocks.") or mlx.mlx_array_dtype(e.value_ptr.*) != .float32) continue;
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_astype(&o, e.value_ptr.*, dt, s));
        try mlx.check(mlx.mlx_array_eval(o));
        _ = mlx.mlx_array_free(e.value_ptr.*);
        e.value_ptr.* = o;
    }
}

fn readFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 20));
}

// ── Sparse-structure decoder ────────────────────────────────────────────────

/// `SparseStructureDecoder` (the TRELLIS v1 `ss_dec_conv3d_16l8_fp16` checkpoint): the 16^3 x 8 latent
/// through conv3d ResBlocks and two pixel-shuffle upsamples to 64^3 occupancy logits. Channels-last.
pub const SsDecoder = struct {
    gpa: std.mem.Allocator,
    act: mlx.mlx_dtype,
    input: Conv3d,
    res: []ResBlock,
    ups: []Conv3d,
    out_norm_w: A,
    out_norm_b: A,
    out_conv: Conv3d,
    owned: std.ArrayList(A) = .empty,

    const Conv3d = struct { w: A, b: A };

    const ResBlock = struct { n1w: A, n1b: A, c1: Conv3d, n2w: A, n2b: A, c2: Conv3d };

    /// Layer order of the published checkpoint: the middle block, then per stage two ResBlocks and
    /// (between stages) the upsample conv, all numbered in one `blocks.N` sequence.
    pub fn load(gpa: std.mem.Allocator, w: *const Weights, act: mlx.mlx_dtype) !SsDecoder {
        var d = SsDecoder{ .gpa = gpa, .act = act, .input = undefined, .res = &.{}, .ups = &.{}, .out_norm_w = undefined, .out_norm_b = undefined, .out_conv = undefined };
        errdefer d.deinit();
        d.input = try d.conv(w, "input_layer");
        d.out_norm_w = try scope_mod.getW(w, "out_layer.0.weight", .{});
        d.out_norm_b = try scope_mod.getW(w, "out_layer.0.bias", .{});
        d.out_conv = try d.conv(w, "out_layer.2");
        var key: [64]u8 = undefined;
        var n_blocks: usize = 0;
        while (w.get(try std.fmt.bufPrint(&key, "blocks.{d}.norm1.weight", .{n_blocks})) != null or w.get(try std.fmt.bufPrint(&key, "blocks.{d}.conv.weight", .{n_blocks})) != null) n_blocks += 1;
        var n_up: usize = 0;
        for (0..n_blocks) |i| n_up += @intFromBool(w.get(try std.fmt.bufPrint(&key, "blocks.{d}.conv.weight", .{i})) != null);
        d.res = try gpa.alloc(ResBlock, 2 + n_blocks - n_up);
        d.ups = try gpa.alloc(Conv3d, n_up);
        var ri: usize = 0;
        var ui: usize = 0;
        for (0..2) |i| {
            d.res[ri] = try d.resBlock(w, "middle_block.{d}", i);
            ri += 1;
        }
        for (0..n_blocks) |i| {
            if (w.get(try std.fmt.bufPrint(&key, "blocks.{d}.conv.weight", .{i})) != null) {
                d.ups[ui] = try d.convAt(w, "blocks.{d}.conv", i);
                ui += 1;
            } else {
                d.res[ri] = try d.resBlock(w, "blocks.{d}", i);
                ri += 1;
            }
        }
        return d;
    }

    /// Conv weight `[O, I, 3, 3, 3]` -> channels-last `[O, 3, 3, 3, I]`, kept in `owned`.
    fn conv(self: *SsDecoder, w: *const Weights, comptime name: []const u8) !Conv3d {
        return self.convAt(w, name, 0);
    }

    fn convAt(self: *SsDecoder, w: *const Weights, comptime fmt: []const u8, i: usize) !Conv3d {
        const s = mlx.gpuStream();
        defer _ = mlx.mlx_stream_free(s);
        const src = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null) try scope_mod.getW(w, fmt ++ ".weight", .{i}) else try scope_mod.getW(w, fmt ++ ".weight", .{});
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&o, src, &[_]c_int{ 0, 2, 3, 4, 1 }, 5, s));
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_array_eval(o));
        try self.owned.append(self.gpa, o);
        const b = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null) try scope_mod.getW(w, fmt ++ ".bias", .{i}) else try scope_mod.getW(w, fmt ++ ".bias", .{});
        return .{ .w = o, .b = b };
    }

    fn resBlock(self: *SsDecoder, w: *const Weights, comptime fmt: []const u8, i: usize) !ResBlock {
        return .{
            .n1w = try scope_mod.getW(w, fmt ++ ".norm1.weight", .{i}),
            .n1b = try scope_mod.getW(w, fmt ++ ".norm1.bias", .{i}),
            .c1 = try self.convAt(w, fmt ++ ".conv1", i),
            .n2w = try scope_mod.getW(w, fmt ++ ".norm2.weight", .{i}),
            .n2b = try scope_mod.getW(w, fmt ++ ".norm2.bias", .{i}),
            .c2 = try self.convAt(w, fmt ++ ".conv2", i),
        };
    }

    pub fn deinit(self: *SsDecoder) void {
        self.gpa.free(self.res);
        self.gpa.free(self.ups);
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
    }

    fn apply(sc: *Scope, c: Conv3d, x: A) !A {
        var o = mlx.mlx_array_new();
        const y = try sc.res(mlx.mlx_conv3d(&o, x, try asLike(sc, c.w, x), 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, sc.s), &o);
        return sc.add(y, try asLike(sc, c.b, x));
    }

    fn norm(sc: *Scope, x: A, w: A, b: A) !A {
        return layerNorm(sc, x, try asLike(sc, w, x), try asLike(sc, b, x), FINAL_LN_EPS);
    }

    fn resForward(sc: *Scope, r: ResBlock, x: A) !A {
        var h = try apply(sc, r.c1, try sc.silu(try norm(sc, x, r.n1w, r.n1b)));
        h = try apply(sc, r.c2, try sc.silu(try norm(sc, h, r.n2w, r.n2b)));
        return sc.add(h, x);
    }

    /// `[B, D, H, W, 8C]` -> `[B, 2D, 2H, 2W, C]`: channel `c*8 + (a*4 + b*2 + e)` lands at offset (a, b, e).
    fn pixelShuffle(sc: *Scope, x: A) !A {
        const sh = mlx.getShape(x);
        const c = @divExact(sh[4], 8);
        const split = try sc.reshape(x, &.{ sh[0], sh[1], sh[2], sh[3], c, 2, 2, 2 });
        const moved = try sc.transpose(split, &.{ 0, 1, 5, 2, 6, 3, 7, 4 });
        return sc.reshape(moved, &.{ sh[0], sh[1] * 2, sh[2] * 2, sh[3] * 2, c });
    }

    /// `z` `[1, C, D, H, W]` f32 (the layout the flow produces). Returns `[1, 64, 64, 64]` logits (f32).
    pub fn forward(self: *const SsDecoder, sc: *Scope, z: A) !A {
        const x = try sc.transpose(z, &.{ 0, 2, 3, 4, 1 });
        var h = try sc.astype(try apply(sc, self.input, x), self.act);
        h = try resForward(sc, self.res[0], h);
        h = try resForward(sc, self.res[1], h);
        // Per stage: two ResBlocks, then the upsample conv (all but the last stage).
        var ri: usize = 2;
        for (0..self.ups.len + 1) |stage| {
            h = try resForward(sc, self.res[ri], h);
            h = try resForward(sc, self.res[ri + 1], h);
            ri += 2;
            if (stage < self.ups.len) h = try pixelShuffle(sc, try apply(sc, self.ups[stage], h));
        }
        h = try sc.astype(h, .float32);
        h = try apply(sc, self.out_conv, try sc.silu(try norm(sc, h, self.out_norm_w, self.out_norm_b)));
        const sh = mlx.getShape(h);
        return sc.reshape(h, &.{ sh[0], sh[1], sh[2], sh[3] });
    }
};

/// The cells where `logits` (`[res^3]`, x slowest) are positive, as `[n, 4]` (batch 0, x, y, z) at
/// `target` resolution: a coarser target keeps a cell when ANY of its `(res/target)^3` children is
/// positive (the reference's max-pool of the occupancy).
pub fn occupiedCells(gpa: std.mem.Allocator, logits: []const f32, res: usize, target: usize) ![]i32 {
    std.debug.assert(logits.len == res * res * res and res % target == 0);
    const r = res / target;
    const flags = try gpa.alloc(bool, target * target * target);
    defer gpa.free(flags);
    @memset(flags, false);
    for (0..res) |x| for (0..res) |y| for (0..res) |z| {
        if (logits[(x * res + y) * res + z] > 0) flags[((x / r) * target + y / r) * target + z / r] = true;
    };
    var out: std.ArrayList(i32) = .empty;
    errdefer out.deinit(gpa);
    for (0..target) |x| for (0..target) |y| for (0..target) |z| {
        if (flags[(x * target + y) * target + z]) try out.appendSlice(gpa, &.{ 0, @intCast(x), @intCast(y), @intCast(z) });
    };
    return out.toOwnedSlice(gpa);
}

// ── Sampler ──────────────────────────────────────────────────────────────────

/// `FlowEulerGuidanceIntervalSampler`: classifier-free guidance inside `interval` (flow time),
/// optionally rescaled to the conditional prediction's std.
pub const SamplerCfg = struct {
    steps: u32,
    rescale_t: f64,
    strength: f64,
    rescale: f64 = 0,
    interval: [2]f64 = .{ 0, 1 },
};

/// The published recipes (`pipeline.json`).
pub const SS_SAMPLER = SamplerCfg{ .steps = 12, .rescale_t = 5.0, .strength = 7.5, .rescale = 0.7, .interval = .{ 0.6, 1.0 } };
pub const SHAPE_SAMPLER = SamplerCfg{ .steps = 12, .rescale_t = 3.0, .strength = 7.5, .rescale = 0.5, .interval = .{ 0.6, 1.0 } };
pub const TEX_SAMPLER = SamplerCfg{ .steps = 12, .rescale_t = 3.0, .strength = 1.0, .rescale = 0.0, .interval = .{ 0.6, 0.9 } };

/// `t_seq[i]` for i in 0..=steps: `linspace(1, 0)` warped by `rescale_t`.
pub fn timeSchedule(out: []f64, rescale_t: f64) void {
    const steps = out.len - 1;
    for (out, 0..) |*t, i| {
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        t.* = rescale_t * lin / (1 + (rescale_t - 1) * lin);
    }
}

pub fn guidedAt(cfg: SamplerCfg, t: f64) bool {
    return cfg.strength != 1 and cfg.interval[0] <= t and t <= cfg.interval[1];
}

/// `x_0` implied by velocity `pred` at flow time `t`.
fn toXStart(sc: *Scope, x: A, t: f64, pred: A) !A {
    return sc.sub(try sc.mulLike(x, @floatCast(1 - SIGMA_MIN)), try sc.mulLike(pred, @floatCast(SIGMA_MIN + (1 - SIGMA_MIN) * t)));
}

/// Classifier-free guidance of two velocity predictions at flow time `t` with the std rescale.
pub fn guide(sc: *Scope, x: A, t: f64, pos: A, neg: A, cfg: SamplerCfg) !A {
    const s: f32 = @floatCast(cfg.strength);
    var pred = try sc.add(try sc.mulLike(pos, s), try sc.mulLike(neg, 1 - s));
    if (cfg.rescale > 0) {
        const x0_pos = try toXStart(sc, x, t, pos);
        const x0_cfg = try toXStart(sc, x, t, pred);
        const ratio = try sc.div(try stdAll(sc, x0_pos), try stdAll(sc, x0_cfg));
        const r: f32 = @floatCast(cfg.rescale);
        const x0 = try sc.add(try sc.mulLike(try sc.mul(x0_cfg, ratio), r), try sc.mulLike(x0_cfg, 1 - r));
        const denom: f32 = @floatCast(SIGMA_MIN + (1 - SIGMA_MIN) * t);
        pred = try sc.mulLike(try sc.sub(try sc.mulLike(x, @floatCast(1 - SIGMA_MIN)), x0), 1 / denom);
    }
    return pred;
}

fn stdAll(sc: *Scope, x: A) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_std(&o, x, false, 0, sc.s), &o);
}

pub const Flow = struct {
    dit: *const Dit,
    rope: Rope,
    cond: A,
    neg_cond: A,
    /// Channels appended to every input (the texture flow's shape latent), `[1, L, C']`.
    concat: ?A = null,
    /// Pixal3D's pixel-aligned features; the negative branch reads them as zeros.
    proj: ?A = null,

    fn velocity(self: Flow, sc: *Scope, x: A, t: f64, cfg: SamplerCfg) !A {
        const tm: f32 = @floatCast(1000 * t);
        const inp = if (self.concat) |c| try sc.concat(&.{ x, c }, 2) else x;
        const pos = try self.dit.forward(sc, inp, tm, self.cond, if (self.proj) |p| .{ .feats = p } else .none, self.rope, null);
        if (!guidedAt(cfg, t)) return pos;
        const neg = try self.dit.forward(sc, inp, tm, self.neg_cond, if (self.proj != null) .zero else .none, self.rope, null);
        return guide(sc, x, t, pos, neg, cfg);
    }

    /// Euler from `noise` `[1, L, C]` (f32) to the sample. Each step's graph is evaluated and freed
    /// before the next. `on_step(i, steps)` reports progress; a non-null error aborts.
    pub fn sample(self: Flow, gpa: std.mem.Allocator, s: S, noise: A, cfg: SamplerCfg, ctx: anytype, comptime on_step: fn (@TypeOf(ctx), u32, u32) anyerror!void) !A {
        const sched = try gpa.alloc(f64, cfg.steps + 1);
        defer gpa.free(sched);
        timeSchedule(sched, cfg.rescale_t);
        var x = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&x, noise);
        errdefer _ = mlx.mlx_array_free(x);
        for (0..cfg.steps) |i| {
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const t = sched[i];
            const v = try self.velocity(&sc, x, t, cfg);
            const next = try sc.sub(x, try sc.mulLike(v, @floatCast(t - sched[i + 1])));
            try mlx.check(mlx.mlx_array_eval(next));
            _ = mlx.mlx_array_free(x);
            x = sc.out(next);
            try on_step(ctx, @intCast(i + 1), cfg.steps);
        }
        return x;
    }
};

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first, then env-gated oracles (TRELLIS2_WEIGHTS + TRELLIS2_FIXTURES, from
// tests/dump_trellis2_fixtures.py).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

/// Row-major f32 values of `a` (any layout, any float dtype), valid for `sc`'s lifetime.
pub fn hostF32(sc: *Scope, a: A) ![]const f32 {
    var c = mlx.mlx_array_new();
    const flat = try sc.res(mlx.mlx_contiguous(&c, try sc.astype(a, .float32), false, sc.s), &c);
    try mlx.check(mlx.mlx_array_eval(flat));
    return (mlx.mlx_array_data_float32(flat) orelse return error.MlxError)[0..mlx.mlx_array_size(flat)];
}

test "trellis2 sampler: the schedule warps linspace(1, 0) by rescale_t and ends at 0" {
    var t: [13]f64 = undefined;
    timeSchedule(&t, 5.0);
    try testing.expectEqual(@as(f64, 1), t[0]);
    try testing.expectEqual(@as(f64, 0), t[12]);
    const lin = 1.0 - 3.0 / 12.0;
    try testing.expectApproxEqAbs(5.0 * lin / (1 + 4.0 * lin), t[3], 1e-12);
    for (t[1..], t[0 .. t.len - 1]) |a, b| try testing.expect(a < b);
    timeSchedule(&t, 1.0);
    try testing.expectApproxEqAbs(@as(f64, 0.5), t[6], 1e-12);
}

test "trellis2 sampler: guidance applies only inside its interval and never at strength 1" {
    try testing.expect(guidedAt(SHAPE_SAMPLER, 0.6));
    try testing.expect(guidedAt(SHAPE_SAMPLER, 1.0));
    try testing.expect(!guidedAt(SHAPE_SAMPLER, 0.59));
    try testing.expect(!guidedAt(TEX_SAMPLER, 0.7)); // strength 1: a single forward
    var one = SHAPE_SAMPLER;
    one.strength = 1;
    try testing.expect(!guidedAt(one, 0.8));
}

test "trellis2 sampler: guided velocity is s*pos + (1-s)*neg, rescaled to the conditional x0's std" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(9);
    const rng = prng.random();
    const n = 64;
    var x: [n]f32 = undefined;
    var pos: [n]f32 = undefined;
    var neg: [n]f32 = undefined;
    for (0..n) |i| {
        x[i] = rng.floatNorm(f32);
        pos[i] = rng.floatNorm(f32);
        neg[i] = rng.floatNorm(f32) * 0.5;
    }
    const cfg = SamplerCfg{ .steps = 12, .rescale_t = 3, .strength = 3.0, .rescale = 0.5, .interval = .{ 0, 1 } };
    const t = 0.8;
    const got = try hostF32(&sc, try guide(&sc, try sc.fromF32(&x, &.{ 1, n }), t, try sc.fromF32(&pos, &.{ 1, n }), try sc.fromF32(&neg, &.{ 1, n }), cfg));
    var pred: [n]f64 = undefined;
    var x0p: [n]f64 = undefined;
    var x0c: [n]f64 = undefined;
    const sig = SIGMA_MIN;
    const den = sig + (1 - sig) * t;
    for (0..n) |i| {
        pred[i] = 3.0 * pos[i] + (1 - 3.0) * neg[i];
        x0p[i] = (1 - sig) * x[i] - den * pos[i];
        x0c[i] = (1 - sig) * x[i] - den * pred[i];
    }
    const stdOf = struct {
        fn f(v: []const f64) f64 {
            var m: f64 = 0;
            for (v) |e| m += e;
            m /= @floatFromInt(v.len);
            var q: f64 = 0;
            for (v) |e| q += (e - m) * (e - m);
            return @sqrt(q / @as(f64, @floatFromInt(v.len)));
        }
    }.f;
    const ratio = stdOf(&x0p) / stdOf(&x0c);
    for (0..n) |i| {
        const x0 = 0.5 * (x0c[i] * ratio) + 0.5 * x0c[i];
        const want = ((1 - sig) * x[i] - x0) / den;
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[i], 1e-4);
    }
}

test "trellis2 rope: pair p of a head rotates by coord * freq on its axis, identity past the last axis pair" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const coords = [_]i32{ 0, 3, 5, 7, 0, 0, 0, 0 };
    const rope = try Rope.init(&sc, &coords, 128);
    var x: [2 * 128]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.1 - 0.5;
    const got = try hostF32(&sc, try rope.apply(&sc, try sc.fromF32(&x, &.{ 1, 2, 1, 128 })));
    // Token 0, pair 0 (x axis, freq 1): angle 3; pair 21 (y axis, freq 1): angle 5; pair 63 (pad): identity.
    for ([_][2]usize{ .{ 0, 3 }, .{ 21, 5 }, .{ 42, 7 }, .{ 63, 0 } }) |pa| {
        const a = x[2 * pa[0]];
        const b = x[2 * pa[0] + 1];
        const ang: f32 = @floatFromInt(pa[1]);
        try testing.expectApproxEqAbs(a * @cos(ang) - b * @sin(ang), got[2 * pa[0]], 1e-5);
        try testing.expectApproxEqAbs(a * @sin(ang) + b * @cos(ang), got[2 * pa[0] + 1], 1e-5);
    }
    // Token 1 sits at the origin: unchanged.
    for (0..128) |i| try testing.expectApproxEqAbs(x[128 + i], got[128 + i], 1e-6);
}


test "trellis2 ss decoder: occupiedCells lists positive cells x-major and pools a coarser target with any-child semantics" {
    const gpa = testing.allocator;
    var logits: [64]f32 = @splat(-1); // 4^3
    logits[(1 * 4 + 2) * 4 + 3] = 0.5; // (1,2,3)
    logits[(3 * 4 + 0) * 4 + 0] = 2; // (3,0,0)
    logits[(0 * 4 + 0) * 4 + 0] = 0; // not positive
    const full = try occupiedCells(gpa, &logits, 4, 4);
    defer gpa.free(full);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 0, 3, 0, 0 }, full);
    const pooled = try occupiedCells(gpa, &logits, 4, 2);
    defer gpa.free(pooled);
    try testing.expectEqualSlices(i32, &.{ 0, 0, 1, 1, 0, 1, 0, 0 }, pooled);
}

test "trellis2 ss decoder: the pixel shuffle sends channel c*8 + (a*4 + b*2 + e) to offset (a, b, e) of output channel c" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var x: [2 * 8]f32 = undefined; // one 1x1x1 cell, C = 2 output channels
    for (&x, 0..) |*v, i| v.* = @floatFromInt(i);
    const got = try hostF32(&sc, try SsDecoder.pixelShuffle(&sc, try sc.fromF32(&x, &.{ 1, 1, 1, 1, 16 })));
    // Output [1, 2, 2, 2, 2]: (a, b, e, c) holds input channel c*8 + a*4 + b*2 + e.
    for (0..2) |a| for (0..2) |b| for (0..2) |e| for (0..2) |c| {
        try testing.expectEqual(@as(f32, @floatFromInt(c * 8 + a * 4 + b * 2 + e)), got[((a * 2 + b) * 2 + e) * 2 + c]);
    };
}

test "trellis2 dit: a projection's input width comes from the scales once its weight is packed" {
    const zeros32 = std.mem.zeroes([512]u32);
    const zerosf = std.mem.zeroes([1024]f32);
    const dense = mlx.mlx_array_new_data(&zerosf, &[_]c_int{ 4, 256 }, 2, .float32);
    defer _ = mlx.mlx_array_free(dense);
    try testing.expectEqual(@as(usize, 256), inWidth(dense, null, 0));
    // 8-bit, group 64: 256 inputs pack into 64 words per row, 4 groups.
    const w8 = mlx.mlx_array_new_data(&zeros32, &[_]c_int{ 4, 64 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w8);
    const s8 = mlx.mlx_array_new_data(&zerosf, &[_]c_int{ 4, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(s8);
    try testing.expectEqual(@as(usize, 256), inWidth(w8, s8, 64));
    // 4-bit, group 32: the same 256 inputs in 32 words and 8 groups.
    const w4 = mlx.mlx_array_new_data(&zeros32, &[_]c_int{ 4, 32 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w4);
    const s4 = mlx.mlx_array_new_data(&zerosf, &[_]c_int{ 4, 8 }, 2, .float32);
    defer _ = mlx.mlx_array_free(s4);
    try testing.expectEqual(@as(usize, 256), inWidth(w4, s4, 32));
}

/// A tiny two-head DiT (one block, 128 channels) with deterministic weights, in the published key layout.
fn tinyDitWeights(gpa: std.mem.Allocator) !Weights {
    var w = Weights.init(gpa);
    errdefer w.deinit();
    var seed: u32 = 12345;
    const put = struct {
        fn f(wt: *Weights, rng: *u32, name: []const u8, shape: []const c_int, spread: f32, offset: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try testing.allocator.alloc(f32, n);
            defer testing.allocator.free(buf);
            for (buf) |*v| {
                rng.* = rng.* *% 1664525 +% 1013904223;
                v.* = offset + (@as(f32, @floatFromInt(rng.* >> 8)) / 16777216.0 - 0.5) * spread;
            }
            const arr = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            try wt.map.put(try wt.allocator.dupe(u8, name), arr);
        }
    }.f;
    const c = 128;
    inline for (.{
        .{ "input_layer.weight", &[_]c_int{ c, 8 } },
        .{ "t_embedder.mlp.0.weight", &[_]c_int{ c, 256 } },
        .{ "t_embedder.mlp.2.weight", &[_]c_int{ c, c } },
        .{ "adaLN_modulation.1.weight", &[_]c_int{ 6 * c, c } },
        .{ "out_layer.weight", &[_]c_int{ 8, c } },
        .{ "blocks.0.self_attn.to_qkv.weight", &[_]c_int{ 3 * c, c } },
        .{ "blocks.0.self_attn.to_out.weight", &[_]c_int{ c, c } },
        .{ "blocks.0.cross_attn.to_q.weight", &[_]c_int{ c, c } },
        .{ "blocks.0.cross_attn.to_kv.weight", &[_]c_int{ 2 * c, 64 } },
        .{ "blocks.0.cross_attn.to_out.weight", &[_]c_int{ c, c } },
        .{ "blocks.0.mlp.mlp.0.weight", &[_]c_int{ 256, c } },
        .{ "blocks.0.mlp.mlp.2.weight", &[_]c_int{ c, 256 } },
    }) |t| {
        try put(&w, &seed, t[0], t[1], 0.12, 0);
        const bias = comptime std.fmt.comptimePrint("{s}.bias", .{t[0][0 .. t[0].len - ".weight".len]});
        try put(&w, &seed, bias, &[_]c_int{t[1][0]}, 0.02, 0);
    }
    try put(&w, &seed, "blocks.0.modulation", &[_]c_int{6 * c}, 0.2, 0);
    inline for (.{ "blocks.0.norm2.weight", "blocks.0.norm2.bias" }, .{ 1.0, 0.0 }) |name, off|
        try put(&w, &seed, name, &[_]c_int{c}, 0.1, off);
    inline for (.{
        "blocks.0.self_attn.q_rms_norm.gamma",   "blocks.0.self_attn.k_rms_norm.gamma",
        "blocks.0.cross_attn.q_rms_norm.gamma",  "blocks.0.cross_attn.k_rms_norm.gamma",
    }) |name| try put(&w, &seed, name, &[_]c_int{ 2, 64 }, 0.1, 1.0);
    return w;
}

/// The converter's rule on a copy of `dense`: `blocks.*` 2-D weights whose input width divides the group go to
/// affine 8-bit (scales and biases beside them); everything else is shared as it is.
fn packTinyDit(dense: *const Weights, group: c_int) !Weights {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var out = Weights.init(dense.allocator);
    errdefer out.deinit();
    var it = dense.map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const shape = mlx.getShape(e.value_ptr.*);
        const quantize = std.mem.startsWith(u8, name, "blocks.") and std.mem.endsWith(u8, name, ".weight") and shape.len == 2 and @rem(shape[1], group) == 0;
        if (!quantize) {
            var copy = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&copy, e.value_ptr.*));
            try out.map.put(try out.allocator.dupe(u8, name), copy);
            continue;
        }
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, e.value_ptr.*, mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        const base = name[0 .. name.len - ".weight".len];
        inline for (.{ .{ 0, "weight" }, .{ 1, "scales" }, .{ 2, "biases" } }) |part| {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&a, triple, part[0]));
            const key = try std.fmt.allocPrint(out.allocator, "{s}.{s}", .{ base, part[1] });
            try out.map.put(key, a);
        }
    }
    return out;
}

test "trellis2 dit: an 8-bit pack of the block projections loads and tracks the dense pack on a forward" {
    const gpa = testing.allocator;
    var dense = try tinyDitWeights(gpa);
    defer dense.deinit();
    var packed_w = try packTinyDit(&dense, 64);
    defer packed_w.deinit();
    try testing.expect(packed_w.get("blocks.0.mlp.mlp.0.scales") != null);
    try testing.expect(packed_w.get("blocks.0.cross_attn.to_kv.scales") != null);
    try testing.expect(packed_w.get("input_layer.scales") == null);

    var d_dense = try Dit.load(gpa, &dense, .float32, 0);
    defer d_dense.deinit();
    var d_packed = try Dit.load(gpa, &packed_w, .float32, 64);
    defer d_packed.deinit();
    // Without the json's group size a packed weight's input width is unknown: refused, never misread.
    try testing.expectError(error.BadQuantGeometry, Dit.load(gpa, &packed_w, .float32, 0));

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var xs: [16 * 8]f32 = undefined;
    var cs: [8 * 64]f32 = undefined;
    for (&xs, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37);
    for (&cs, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.11);
    var coords: [16 * 4]i32 = undefined;
    for (0..16) |i| coords[i * 4 ..][0..4].* = .{ 0, @intCast(i % 4), @intCast(i / 4), 0 };
    const x = try sc.fromF32(&xs, &.{ 1, 16, 8 });
    const cond = try sc.fromF32(&cs, &.{ 1, 8, 64 });
    const rope = try Rope.init(&sc, &coords, d_dense.head_dim);
    const want = try d_dense.forward(&sc, x, 0.5, cond, .none, rope, null);
    const got = try d_packed.forward(&sc, x, 0.5, cond, .none, rope, null);
    try expectClose("tiny dit 8-bit vs dense", &sc, got, want, 0.999, 0.02);
}

test "trellis2 dit: an allocation failing anywhere in Dit.load leaves nothing leaked or freed twice" {
    var dense = try tinyDitWeights(testing.allocator);
    defer dense.deinit();
    var fail_at: usize = 0;
    while (fail_at < 400) : (fail_at += 1) {
        var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_at });
        var dit = Dit.load(failing.allocator(), &dense, .float32, 0) catch continue;
        dit.deinit();
        return;
    }
    return error.TestUnexpectedResult;
}

const OracleEnv = struct { ckpts: []const u8, fixtures: []const u8 };

fn oracleEnv() !OracleEnv {
    const w = std.c.getenv("TRELLIS2_WEIGHTS") orelse return error.SkipZigTest;
    const f = std.c.getenv("TRELLIS2_FIXTURES") orelse return error.SkipZigTest;
    return .{ .ckpts = std.fmt.allocPrint(testing.allocator, "{s}/ckpts", .{std.mem.span(w)}) catch return error.OutOfMemory, .fixtures = std.mem.span(f) };
}

pub fn hostI32(gpa: std.mem.Allocator, a: A) ![]i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const d = mlx.mlx_array_data_int32(a) orelse return error.MlxError;
    return gpa.dupe(i32, d[0..mlx.mlx_array_size(a)]);
}

pub fn expectClose(label: []const u8, sc: *Scope, got: A, want: A, min_cos: f64, tol_ratio: f64) !void {
    const g = try hostF32(sc, got);
    const w = try hostF32(sc, want);
    try testing.expectEqual(w.len, g.len);
    var dot: f64 = 0;
    var gg: f64 = 0;
    var ww: f64 = 0;
    var peak: f64 = 0;
    var worst: f64 = 0;
    for (g, w) |a, b| {
        try testing.expect(std.math.isFinite(a));
        dot += @as(f64, a) * b;
        gg += @as(f64, a) * a;
        ww += @as(f64, b) * b;
        peak = @max(peak, @abs(@as(f64, b)));
        worst = @max(worst, @abs(@as(f64, a) - b));
    }
    const cos = dot / @sqrt(gg * ww);
    const ratio = @sqrt(gg / ww);
    std.debug.print("{s}: cos {d:.7} rms ratio {d:.6} max err/peak {e:.2}\n", .{ label, cos, ratio, worst / peak });
    try testing.expect(cos > min_cos);
    try testing.expect(@abs(ratio - 1) < tol_ratio);
}

/// Runs one flow fixture (`stem` the checkpoint, `fixture` its file) at `act` and compares every tap.
fn flowOracle(stem: []const u8, fixture: []const u8, dense: bool, concat_key: ?[]const u8, act: mlx.mlx_dtype, min_cos: f64, tol_ratio: f64) !void {
    const gpa = testing.allocator;
    const env = try oracleEnv();
    defer gpa.free(env.ckpts);
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try model_mod.loadWeightsFile(gpa, env.fixtures, fixture);
    defer fx.deinit();
    var od = try openDit(io, gpa, env.ckpts, stem, act);
    defer od.deinit();

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const x_fx = fx.get("x").?;
    var coords: []i32 = undefined;
    var x: A = undefined;
    if (dense) {
        coords = try gridCoords(gpa, 16);
        const c = mlx.getShape(x_fx)[1];
        x = try sc.transpose(try sc.reshape(x_fx, &.{ 1, c, 16 * 16 * 16 }), &.{ 0, 2, 1 });
    } else {
        coords = try hostI32(gpa, fx.get("coords").?);
        const sh = mlx.getShape(x_fx);
        x = try sc.reshape(x_fx, &.{ 1, sh[0], sh[1] });
        if (concat_key) |k| {
            const cs = fx.get(k).?;
            const csh = mlx.getShape(cs);
            x = try sc.concat(&.{ x, try sc.reshape(cs, &.{ 1, csh[0], csh[1] }) }, 2);
        }
    }
    defer gpa.free(coords);
    const rope = try Rope.init(&sc, coords, od.dit.head_dim);
    var tt: f32 = undefined;
    try mlx.check(mlx.mlx_array_item_float32(&tt, fx.get("t").?));
    var taps = Taps{};
    var out = try od.dit.forward(&sc, x, tt, fx.get("cond").?, .none, rope, &taps);
    var tag: [16]u8 = undefined;
    const label = try std.fmt.bufPrint(&tag, "{s}", .{if (act == .float32) "f32" else "bf16"});
    std.debug.print("{s} {s}:\n", .{ stem, label });
    const want_out = fx.get("out").?;
    if (dense) {
        out = try sc.reshape(try sc.transpose(out, &.{ 0, 2, 1 }), mlx.getShape(want_out));
    } else {
        out = try sc.reshape(out, mlx.getShape(want_out));
    }
    inline for (.{ .{ "h_in", &taps.h_in }, .{ "blk0", &taps.blk0 }, .{ "blk_last", &taps.blk_last } }) |tp| {
        if (fx.get(tp[0])) |want| try expectClose("  " ++ tp[0], &sc, try sc.reshape(tp[1].*, mlx.getShape(want)), want, min_cos, tol_ratio);
    }
    try expectClose("  out", &sc, out, want_out, min_cos, tol_ratio);
}

test "trellis2 oracle: the sparse-structure flow DiT matches the reference (TRELLIS2_WEIGHTS, TRELLIS2_FIXTURES)" {
    try flowOracle("ss_flow_img_dit_1_3B_64_bf16", "ss_flow.safetensors", true, null, .float32, 0.9999, 1e-3);
}

test "trellis2 oracle: the shape SLat flow DiT matches the reference over a sparse voxel list" {
    try flowOracle("slat_flow_img2shape_dit_1_3B_512_bf16", "slat_shape512.safetensors", false, null, .float32, 0.9999, 1e-3);
}

test "trellis2 oracle: the texture SLat flow DiT matches the reference with the shape latent concatenated" {
    try flowOracle("slat_flow_imgshape2tex_dit_1_3B_512_bf16", "slat_tex512.safetensors", false, "concat", .float32, 0.9999, 1e-3);
}

test "trellis2 oracle: the sparse-structure decoder matches the reference logits (TRELLIS2_SS_DEC, TRELLIS2_FIXTURES)" {
    const gpa = testing.allocator;
    const dir = std.mem.span(std.c.getenv("TRELLIS2_SS_DEC") orelse return error.SkipZigTest);
    const fixtures = std.mem.span(std.c.getenv("TRELLIS2_FIXTURES") orelse return error.SkipZigTest);
    var w = try model_mod.loadWeightsFile(gpa, dir, "ss_dec_conv3d_16l8_fp16.safetensors");
    defer w.deinit();
    var dec = try SsDecoder.load(gpa, &w, .float32);
    defer dec.deinit();
    var fx = try model_mod.loadWeightsFile(gpa, fixtures, "ss_dec.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const got = try dec.forward(&sc, fx.get("z").?);
    const want = fx.get("logits").?;
    try expectClose("ss decoder logits", &sc, got, try sc.reshape(want, mlx.getShape(got)), 0.9999, 1e-3);
    // The occupancy decisions are what the pipeline uses: the same cells.
    const g = try hostF32(&sc, got);
    const wv = try hostF32(&sc, want);
    var flips: usize = 0;
    for (g, wv) |a, b| flips += @intFromBool((a > 0) != (b > 0));
    std.debug.print("ss decoder: {d} of {d} cells differ in sign\n", .{ flips, g.len });
    try testing.expectEqual(@as(usize, 0), flips);
}

test "trellis2 oracle: the blocks in bf16, as the checkpoints are published, track the fp32 reference (TRELLIS2_WEIGHTS, TRELLIS2_FIXTURES)" {
    try flowOracle("ss_flow_img_dit_1_3B_64_bf16", "ss_flow.safetensors", true, null, .bfloat16, 0.99, 0.05);
    try flowOracle("slat_flow_img2shape_dit_1_3B_512_bf16", "slat_shape512.safetensors", false, null, .bfloat16, 0.99, 0.05);
    try flowOracle("slat_flow_imgshape2tex_dit_1_3B_512_bf16", "slat_tex512.safetensors", false, "concat", .bfloat16, 0.99, 0.05);
}
