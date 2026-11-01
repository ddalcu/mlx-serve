//! Idol's denoiser (`LatentSeqMMFlowModel`): a rectified-flow transformer over the object latent (8192 tokens of 16)
//! and the input camera (1 token of 5), conditioned on the image tokens. The latent, image and camera tokens are
//! refined on their own (two blocks each), then run through the joint blocks with every token's rotary position
//! predicted from its own features (`RePo3D`). Blocks in the reference's f16 (Linear only), everything else f32.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const scope_mod = @import("../../mlx/scope.zig");
const L = @import("layers.zig");
const glue = @import("glue.zig");
const device = @import("../../mlx/device.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");

const A = L.A;
const Scope = L.Scope;
const Linear = L.Linear;
const Weights = L.Weights;
const NONE = L.NONE;

/// Lets a test run identical image tokens uncollapsed.
pub var collapse_disabled_for_test = false;

pub const Cfg = struct {
    q_token_length: u32 = 8192,
    in_channels: u32 = 16,
    cam_channels: u32 = 5,
    model_channels: u32 = 1024,
    cond_channels: u32 = 1280,
    cond2_channels: u32 = 128,
    num_blocks: u32 = 24,
    num_refiner_blocks: u32 = 2,
    num_heads: u32 = 16,
    mlp_ratio: u32 = 4,
};

const Block = struct {
    shift_table: ?A,
    norm1_w: ?A,
    norm1_b: ?A,
    norm2_w: ?A,
    norm2_b: ?A,
    qkv: Linear,
    out: Linear,
    q_gamma: A,
    k_gamma: A,
    mlp: L.Mlp,

    fn load(w: *const Weights, comptime fmt: []const u8, i: usize, c: u32, hidden: u32, modulated: bool) !Block {
        var pbuf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, fmt, .{i});
        return .{
            .shift_table = if (modulated) try scope_mod.getW(w, "{s}.shift_table", .{p}) else null,
            .norm1_w = if (modulated) null else try scope_mod.getW(w, "{s}.norm1.weight", .{p}),
            .norm1_b = if (modulated) null else try scope_mod.getW(w, "{s}.norm1.bias", .{p}),
            .norm2_w = if (modulated) null else try scope_mod.getW(w, "{s}.norm2.weight", .{p}),
            .norm2_b = if (modulated) null else try scope_mod.getW(w, "{s}.norm2.bias", .{p}),
            .qkv = try scope_mod.linear(w, "{s}.attn.qkv", .{p}, c),
            .out = try scope_mod.linear(w, "{s}.attn.out", .{p}, c),
            .q_gamma = try scope_mod.getW(w, "{s}.attn.q_norm.gamma", .{p}),
            .k_gamma = try scope_mod.getW(w, "{s}.attn.k_norm.gamma", .{p}),
            .mlp = try L.Mlp.load(w, "{s}.mlp", .{p}, c, hidden),
        };
    }
};

/// `RePo3DRotaryEmbedding`: each token's features predict a 3D offset per head, turned into rotary angles.
const Repo = struct {
    norm_w: A,
    norm_b: A,
    gate: Linear,
    content: Linear,
    final: Linear,
    freqs: [3]A,
    /// What the fused angle pass reads: the final map dense f32 `[hidden, 3H]`, the 32 frequencies and their axis.
    wf: A = NONE,
    fq: A = NONE,
    axis: A = NONE,
    /// The norm folded into the gate | content maps (dense packs): `fold` f32 `[C, PARTS * 2 hidden]` the maps with the
    /// norm's weight folded in, `cs` its column sums and `bw` the norm's bias through the maps (`glue.repoFold`). h's
    /// f16 values are exact in TF32, so where MLX rounds f32 matmuls to TF32 only the maps split: their TF32-exact part
    /// beside the rest (PARTS 2) keeps the f32 reference's precision in one GEMM. The mean comes off after the GEMM:
    /// the fold assumes a token's mean is small against its spread, as Idol's hidden states are.
    fold: A = NONE,
    cs: A = NONE,
    bw: A = NONE,

    fn load(w: *const Weights, comptime name: []const u8, i: usize, c: u32) !Repo {
        const hidden: u32 = c / 8;
        return .{
            .norm_w = try scope_mod.getW(w, name ++ ".{d}.norm.weight", .{i}),
            .norm_b = try scope_mod.getW(w, name ++ ".{d}.norm.bias", .{i}),
            .gate = try scope_mod.linear(w, name ++ ".{d}.gate_map", .{i}, c),
            .content = try scope_mod.linear(w, name ++ ".{d}.content_map", .{i}, c),
            .final = try scope_mod.linear(w, name ++ ".{d}.final_map", .{i}, hidden),
            .freqs = .{
                try scope_mod.getW(w, name ++ ".{d}.freqs_0", .{i}),
                try scope_mod.getW(w, name ++ ".{d}.freqs_1", .{i}),
                try scope_mod.getW(w, name ++ ".{d}.freqs_2", .{i}),
            },
        };
    }

    /// The fused pass's inputs, owned by `owned`; the final map dequantized where the pack quantizes it.
    fn fuse(self: *Repo, gpa: std.mem.Allocator, owned: *std.ArrayList(A)) !void {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer {
            _ = mlx.mlx_stream_free(sc.s);
            sc.deinit();
        }
        const f = self.final;
        var dense = f.w;
        if (f.scales) |scales| {
            var o = mlx.mlx_array_new();
            dense = try sc.res(mlx.mlx_dequantize(&o, f.w, scales, f.biases, .some(@intCast(f.group_size)), .some(@intCast(f.bits)), "affine", NONE, .{ .value = .float32, .has_value = true }, sc.s), &o);
        }
        var t = mlx.mlx_array_new();
        const wf = try sc.res(mlx.mlx_contiguous(&t, try sc.transpose(try L.asDtype(&sc, dense, .float32), &.{ 1, 0 }), false, sc.s), &t);
        const fq = try L.asDtype(&sc, try sc.concat(&self.freqs, 0), .float32);
        var axis: [32]i32 = undefined;
        var at: usize = 0;
        for (self.freqs, 0..) |q, a| for (0..mlx.mlx_array_size(q)) |_| {
            if (at == axis.len) return error.UnsupportedIdolRepo;
            axis[at] = @intCast(a);
            at += 1;
        };
        if (at != axis.len) return error.UnsupportedIdolRepo;
        const ax = try sc.fromI32(&axis, &.{32});
        var fold = [3]A{ NONE, NONE, NONE };
        const g = self.gate;
        const c = self.content;
        if (g.scales == null and c.scales == null and g.bias == null and c.bias == null and g.w8 == null and c.w8 == null and !g.transposed and !c.transposed) {
            const stacked = try L.asDtype(&sc, try sc.concat(&.{ g.w, c.w }, 0), .float32);
            const folded = try sc.mul(stacked, try L.asDtype(&sc, self.norm_w, .float32));
            var t2 = mlx.mlx_array_new();
            const wt = try sc.res(mlx.mlx_contiguous(&t2, try sc.transpose(folded, &.{ 1, 0 }), false, sc.s), &t2);
            fold[0] = if (device.tf32Matmul()) blk: {
                const p = try tf32Split(&sc, wt);
                break :blk try sc.concat(&p, 1);
            } else wt;
            fold[1] = try sc.sumAxis(folded, 1);
            fold[2] = try sc.sumAxis(try sc.mul(stacked, try L.asDtype(&sc, self.norm_b, .float32)), 1);
        }
        for ([_]*A{ &self.wf, &self.fq, &self.axis, &self.fold, &self.cs, &self.bw }, [_]A{ wf, fq, ax, fold[0], fold[1], fold[2] }) |slot, v| {
            if (v.ctx == null) continue;
            try mlx.check(mlx.mlx_array_eval(v));
            const kept = sc.out(v);
            errdefer _ = mlx.mlx_array_free(kept);
            try owned.append(gpa, kept);
            slot.* = kept;
        }
    }

    /// cos and sin `[1, L, H, 32]` f32 for `h` `[1, L, C]`.
    fn angles(self: *const Repo, sc: *Scope, h: A, heads: c_int) !struct { cos: A, sin: A } {
        const l = mlx.getShape(h)[1];
        if (self.fold.ctx != null) {
            const p = try sc.matmul(try L.asDtype(sc, h, .float32), self.fold);
            if (try glue.repoFold(p, h, self.cs, self.bw, 1e-5, sc.s)) |gc| {
                if (try glue.repoAngles(try sc.keep(gc), self.wf, self.fq, self.axis, heads, sc.s)) |t| return .{ .cos = try sc.keep(t.cos), .sin = try sc.keep(t.sin) };
            }
        }
        const n = try sc.layerNorm(try L.asDtype(sc, h, .float32), self.norm_w, self.norm_b, 1e-5);
        const g = try self.gate.forward(sc, n);
        const c = try self.content.forward(sc, n);
        const act = try sc.mul(try sc.silu(g), c);
        const delta = try sc.reshape(try self.final.forward(sc, act), &.{ 1, l, heads, 3 });
        var parts: [3]A = undefined;
        for (&parts, 0..) |*p, axis| {
            const d = try sc.slice(delta, 3, @intCast(axis), @intCast(axis + 1));
            p.* = try sc.mulS(try sc.mul(d, self.freqs[axis]), std.math.pi);
        }
        const ang = try sc.concat(&parts, 3);
        return .{ .cos = try sc.cos(ang), .sin = try sc.sin(ang) };
    }
};

/// `a` f32 as hi + lo, hi exact in TF32 (its low 13 mantissa bits cleared): the operands of a 3xTF32 product.
fn tf32Split(sc: *Scope, a: A) ![2]A {
    const mask_bits: u32 = 0xFFFFE000;
    const mask = try sc.keep(mlx.mlx_array_new_data(&mask_bits, &[_]c_int{}, 0, .uint32));
    var u = mlx.mlx_array_new();
    const bits = try sc.res(mlx.mlx_view(&u, a, .uint32, sc.s), &u);
    var m = mlx.mlx_array_new();
    const kept = try sc.res(mlx.mlx_bitwise_and(&m, bits, mask, sc.s), &m);
    var f = mlx.mlx_array_new();
    const hi = try sc.res(mlx.mlx_view(&f, kept, .float32, sc.s), &f);
    return .{ hi, try sc.sub(a, hi) };
}

const queue = scope_mod.queue;

/// `LayerNorm(x) * (1 + scale) + shift`, one pass on the GPU.
fn normMod(sc: *Scope, x: A, scale: A, shift: A) !A {
    if (try glue.normMod(x, scale, shift, L.LN_EPS, sc.s)) |n| return sc.keep(n);
    return L.modulate(sc, try sc.layerNorm(x, NONE, NONE, L.LN_EPS), shift, scale);
}

/// `x + lin(y) * gate`: the projection's bias, the gate and the residual add in one pass on the GPU.
fn residual(sc: *Scope, x: A, lin: *const Linear, y: A, gate: A) !A {
    const p = try lin.project(sc, y);
    if (lin.bias) |bias| if (try glue.gateResidual(x, p, bias, gate, sc.s)) |r| return sc.keep(r);
    return sc.add(x, try sc.mul(if (lin.bias) |bias| try sc.add(p, bias) else p, gate));
}

/// Whether every token of `y` `[1, L, C]` equals the first.
fn sameRows(sc: *Scope, y: A) !bool {
    var eq = mlx.mlx_array_new();
    const e = try sc.res(mlx.mlx_equal(&eq, y, try sc.slice(y, 1, 0, 1), sc.s), &eq);
    var all = mlx.mlx_array_new();
    const r = try sc.res(mlx.mlx_all(&all, e, false, sc.s), &all);
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, r));
    return same;
}

/// `x` `[1, L, H, 64]` rotated pairwise (even, odd) by `cos`/`sin` `[1, L, H, 32]`, f32 math, x's dtype out.
pub fn applyRope(sc: *Scope, x: A, cos: A, sin: A) !A {
    const dt = mlx.mlx_array_dtype(x);
    const sh = mlx.getShape(x);
    const pair = [_]c_int{ sh[0], sh[1], sh[2], @divExact(sh[3], 2), 2 };
    const p = try sc.reshape(try L.asDtype(sc, x, .float32), &pair);
    const x0 = try sc.slice(p, 4, 0, 1);
    const x1 = try sc.slice(p, 4, 1, 2);
    const c = try sc.reshape(cos, &.{ sh[0], sh[1], sh[2], pair[3], 1 });
    const s = try sc.reshape(sin, &.{ sh[0], sh[1], sh[2], pair[3], 1 });
    const o0 = try sc.sub(try sc.mul(x0, c), try sc.mul(x1, s));
    const o1 = try sc.add(try sc.mul(x0, s), try sc.mul(x1, c));
    return L.asDtype(sc, try sc.reshape(try sc.concat(&.{ o0, o1 }, 4), sh), dt);
}

pub const Denoiser = struct {
    gpa: std.mem.Allocator,
    cfg: Cfg,
    act: mlx.mlx_dtype,
    input: Linear,
    cond1: Linear,
    cond2: Linear,
    t_fc0: Linear,
    t_fc2: Linear,
    ada: Linear,
    out: Linear,
    cam_out: Linear,
    cam0: Linear,
    cam1: Linear,
    shift_table: A,
    pos: A,
    noise_blocks: []Block,
    context_blocks: []Block,
    blocks: []Block,
    noise_repo: []Repo,
    context_repo: []Repo,
    repo: []Repo,
    ones: A,
    owned: std.ArrayList(A) = .empty,

    /// `w` is the denoiser's weight map (the caller keeps it alive); `act` the blocks' dtype (f16 as published).
    pub fn load(gpa: std.mem.Allocator, w: *const Weights, cfg: Cfg, act: mlx.mlx_dtype) !Denoiser {
        const c = cfg.model_channels;
        const hidden = c * cfg.mlp_ratio;
        var d: Denoiser = .{
            .gpa = gpa, .cfg = cfg, .act = act,
            .input = try scope_mod.linear(w, "input_layer", .{}, cfg.in_channels),
            .cond1 = try scope_mod.linear(w, "cond_embedder", .{}, cfg.cond_channels),
            .cond2 = try scope_mod.linear(w, "cond_embedder2", .{}, cfg.cond2_channels),
            .t_fc0 = try scope_mod.linear(w, "t_embedder.mlp.0", .{}, 256),
            .t_fc2 = try scope_mod.linear(w, "t_embedder.mlp.2", .{}, c),
            .ada = try scope_mod.linear(w, "adaLN_modulation.1", .{}, c),
            .out = try scope_mod.linear(w, "out_layer", .{}, c),
            .cam_out = try scope_mod.linear(w, "cam_out_layer", .{}, c),
            .cam0 = try scope_mod.linear(w, "cam_refiner.mlp.0", .{}, cfg.cam_channels),
            .cam1 = try scope_mod.linear(w, "cam_refiner.mlp.2", .{}, c),
            .shift_table = try scope_mod.getW(w, "shift_table", .{}),
            .pos = undefined,
            .noise_blocks = &.{}, .context_blocks = &.{}, .blocks = &.{},
            .noise_repo = &.{}, .context_repo = &.{}, .repo = &.{},
            .ones = undefined,
        };
        errdefer d.deinit();
        const n = cfg.num_blocks;
        const r = cfg.num_refiner_blocks;
        d.noise_blocks = try gpa.alloc(Block, r);
        d.context_blocks = try gpa.alloc(Block, r);
        d.blocks = try gpa.alloc(Block, n);
        d.noise_repo = try gpa.alloc(Repo, r);
        d.context_repo = try gpa.alloc(Repo, r);
        d.repo = try gpa.alloc(Repo, n);
        for (d.noise_blocks, 0..) |*b, i| b.* = try Block.load(w, "noise_refiner.{d}", i, c, hidden, true);
        for (d.context_blocks, 0..) |*b, i| b.* = try Block.load(w, "context_refiner.{d}", i, c, hidden, false);
        for (d.blocks, 0..) |*b, i| b.* = try Block.load(w, "blocks.{d}", i, c, hidden, true);
        for (d.noise_repo, 0..) |*b, i| b.* = try Repo.load(w, "noise_repo_layers", i, c);
        for (d.context_repo, 0..) |*b, i| b.* = try Repo.load(w, "context_repo_layers", i, c);
        for (d.repo, 0..) |*b, i| b.* = try Repo.load(w, "repo_layers", i, c);
        for ([_][]Repo{ d.noise_repo, d.context_repo, d.repo }) |set| for (set) |*rp| try rp.fuse(gpa, &d.owned);

        // The latent tokens' fixed positions, embedded once.
        const pe = try scope_mod.getW(w, "pos_pe", .{});
        try mlx.check(mlx.mlx_array_eval(pe));
        const host = try gpa.alloc(f32, cfg.q_token_length * 3);
        defer gpa.free(host);
        const src = mlx.mlx_array_data_float32(pe) orelse return error.NoData;
        @memcpy(host, src[0..host.len]);
        const emb = try L.absolutePosition(gpa, host, c);
        defer gpa.free(emb);
        d.pos = mlx.mlx_array_new_data(emb.ptr, &[_]c_int{ 1, @intCast(cfg.q_token_length), @intCast(c) }, 3, .float32);
        try d.owned.append(gpa, d.pos);
        const ones = try gpa.alloc(f32, L.HEAD_DIM);
        defer gpa.free(ones);
        @memset(ones, 1);
        d.ones = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{L.HEAD_DIM}, 1, .float32);
        try d.owned.append(gpa, d.ones);
        return d;
    }

    pub fn deinit(self: *Denoiser) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.noise_blocks);
        self.gpa.free(self.context_blocks);
        self.gpa.free(self.blocks);
        self.gpa.free(self.noise_repo);
        self.gpa.free(self.context_repo);
        self.gpa.free(self.repo);
    }

    fn heads(self: *const Denoiser) c_int {
        return @intCast(self.cfg.num_heads);
    }

    /// Attention of the normalized tokens `n`: the qkv projection, its bias, q and k rotated, then RMS-normalized
    /// per head (fused into one pass on the GPU), heads-major, then attention (the biased pass on `glue.attention`).
    /// Returns `[1, L, C]`.
    fn attention(self: *const Denoiser, sc: *Scope, b: *const Block, n: A, rope: anytype, bias: KeyBias) !A {
        const h = self.heads();
        const d: c_int = L.HEAD_DIM;
        const y = try b.qkv.project(sc, n);
        if (b.qkv.bias) |qb| if (try glue.qkvPrep(y, qb, b.q_gamma, b.k_gamma, rope.cos, rope.sin, L.HEAD_NORM_EPS, h, sc.s)) |f| {
            const q = try sc.keep(f.q);
            const k = try sc.keep(f.k);
            const v = try sc.keep(f.v);
            if (bias.base2.ctx != null) if (try glue.attention(q, k, v, 1.0 / @sqrt(@as(f32, L.HEAD_DIM)), bias.base2, sc.s)) |o| {
                return sc.reshape(try sc.keep(o), &.{ 1, mlx.getShape(q)[2], h * d });
            };
            return L.attendHeads(sc, q, k, v, bias.mask);
        };
        const yb = if (b.qkv.bias) |qb| try sc.add(y, qb) else y;
        const q = try L.headNorm(sc, try applyRope(sc, try L.part(sc, yb, 3, 0, h, d), rope.cos, rope.sin), b.q_gamma, self.ones);
        const k = try L.headNorm(sc, try applyRope(sc, try L.part(sc, yb, 3, 1, h, d), rope.cos, rope.sin), b.k_gamma, self.ones);
        return L.attend(sc, q, k, try L.part(sc, yb, 3, 2, h, d), bias.mask);
    }

    fn block(self: *const Denoiser, sc: *Scope, b: *const Block, x: A, mod: ?A, rope: anytype, bias: KeyBias) !A {
        const c = self.cfg.model_channels;
        if (mod == null) {
            const n1 = try L.asDtype(sc, try sc.layerNorm(try L.asDtype(sc, x, .float32), b.norm1_w.?, b.norm1_b.?, L.LN_EPS), self.act);
            const x1 = try sc.add(x, try b.out.forward(sc, try self.attention(sc, b, n1, rope, bias)));
            const n2 = try L.asDtype(sc, try sc.layerNorm(try L.asDtype(sc, x1, .float32), b.norm2_w.?, b.norm2_b.?, L.LN_EPS), self.act);
            return sc.add(x1, try b.mlp.forward(sc, n2));
        }
        const m = try sc.add(mod.?, try L.asLike(sc, b.shift_table.?, mod.?));
        const shift_msa, const scale_msa, const gate_msa, const shift_mlp, const scale_mlp, const gate_mlp = try L.chunks6(sc, m, c);
        const x1 = try residual(sc, x, &b.out, try self.attention(sc, b, try normMod(sc, x, scale_msa, shift_msa), rope, bias), gate_msa);
        return residual(sc, x1, &b.mlp.fc2, try b.mlp.activate(sc, try normMod(sc, x1, scale_mlp, shift_mlp)), gate_mlp);
    }

    /// An additive key bias in the two forms the attentions read: sdpa's mask `[1, 1, 1, Lk]` (f16) and the int8
    /// attention's `[Lk]` (f32, times log2 e); null handles for none.
    pub const KeyBias = struct { mask: A = NONE, base2: A = NONE };

    /// The refined image tokens, and the key bias the joint blocks add to their scores (latent, image, camera keys).
    pub const Ctx = struct { tokens: A, bias: KeyBias };

    /// The image tokens refined once (they depend on neither the noisy latent nor the time): `feature1`
    /// `[1, Lc, 1280]` and `feature2` `[1, Lc, 128]` f32 in, `[1, Lc, C]` in the block dtype out. Both belong to `parent`.
    /// Identical tokens (the unconditional branch's all-zero features) stay identical through every block, RoPE coming
    /// from each token's own features: they run as two, their count carried by the key bias (`keyBias`).
    pub fn context(self: *const Denoiser, parent: *Scope, f1: A, f2: A) !Ctx {
        var cur: ?A = null;
        errdefer if (cur) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var bias: KeyBias = .{};
        {
            var sc = Scope.init(parent.a, parent.s);
            defer sc.deinit();
            const e = try sc.add(try self.cond1.forward(&sc, f1), try self.cond2.forward(&sc, f2));
            var y = try L.asDtype(&sc, e, self.act);
            const n = mlx.getShape(y)[1];
            if (!collapse_disabled_for_test and n > 2 and try sameRows(&sc, y)) {
                y = try sc.slice(y, 1, 0, 2);
                bias = try self.keyBias(parent, n);
            }
            try mlx.check(mlx.mlx_array_eval(y));
            cur = sc.out(y);
        }
        for (self.context_blocks, self.context_repo) |*b, *r| {
            var sc = Scope.init(parent.a, parent.s);
            defer sc.deinit();
            const rope = try r.angles(&sc, cur.?, self.heads());
            const y = try self.block(&sc, b, cur.?, null, rope, .{});
            try mlx.check(mlx.mlx_array_eval(y));
            _ = mlx.mlx_array_free(cur.?);
            cur = sc.out(y);
        }
        const tokens = try parent.keep(cur.?);
        cur = null;
        return .{ .tokens = tokens, .bias = bias };
    }

    /// n copies of a key attend like one key with ln(n) added to its score. Two keys whose f16 biases satisfy
    /// e^b1 + e^b2 = n carry it (one f16 bias is off by up to 0.4%); laid out as the joint blocks' tokens: latent,
    /// the two image tokens, camera.
    fn keyBias(self: *const Denoiser, parent: *Scope, n: c_int) !KeyBias {
        const nf: f64 = @floatFromInt(n);
        var b1: f16 = @floatCast(@log(nf));
        if (@exp(@as(f64, b1)) > nf) b1 = std.math.nextAfter(f16, b1, 0);
        const rest = nf - @exp(@as(f64, b1));
        const lx = self.cfg.q_token_length;
        const bias = try parent.a.alloc(f16, lx + 3);
        defer parent.a.free(bias);
        const base2 = try parent.a.alloc(f32, lx + 3);
        defer parent.a.free(base2);
        @memset(bias, 0);
        bias[lx] = b1;
        bias[lx + 1] = if (rest > 0) @floatCast(@log(rest)) else -std.math.inf(f16);
        for (base2, bias) |*o, v| o.* = @as(f32, v) * std.math.log2e;
        return .{
            .mask = try parent.keep(mlx.mlx_array_new_data(bias.ptr, &[_]c_int{ 1, 1, 1, @intCast(lx + 3) }, 4, .float16)),
            .base2 = try parent.keep(mlx.mlx_array_new_data(base2.ptr, &[_]c_int{@intCast(lx + 3)}, 1, .float32)),
        };
    }

    /// What a step's two guidance passes share: the latent tokens through the noise refiner, the camera token and
    /// the time embeddings (none of them sees the condition).
    pub const Stream = struct { hx: A, hc: A, t_emb: A, t_mod: A };

    /// `z` `[1, 8192, 16]`, `cam` `[1, 1, 5]` f32 at model time `t` (1000 x flow time); everything belongs to `sc`.
    pub fn stream(self: *const Denoiser, sc: *Scope, z: A, cam: A, t: f32) !Stream {
        var tbuf: [256]f32 = undefined;
        L.sinusoid(&tbuf, t, 10000, false);
        const e0 = try sc.fromF32(&tbuf, &.{ 1, 256 });
        const t_emb = try self.t_fc2.forward(sc, try sc.silu(try self.t_fc0.forward(sc, e0)));
        const t_mod = try L.asDtype(sc, try self.ada.forward(sc, try sc.silu(t_emb)), self.act);

        // The reference casts both terms to the block dtype before adding.
        const h_in = try L.asDtype(sc, try self.input.forward(sc, z), self.act);
        var hx = try sc.add(h_in, try L.asDtype(sc, self.pos, self.act));
        for (self.noise_blocks, self.noise_repo) |*b, *r| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const rope = try r.angles(&bs, hx, self.heads());
            const y = try self.block(&bs, b, hx, t_mod, rope, .{});
            try queue(y, hx);
            hx = try sc.keep(bs.out(y));
        }
        const hc = try L.asDtype(sc, try self.cam1.forward(sc, try L.geluTanh(sc, try self.cam0.forward(sc, cam))), self.act);
        return .{ .hx = hx, .hc = hc, .t_emb = t_emb, .t_mod = t_mod };
    }

    pub const Velocity = struct { latent: A, camera: A };

    /// The velocity from a step's `stream` and one condition's `context`; both outputs belong to `sc`.
    pub fn joint(self: *const Denoiser, sc: *Scope, st: Stream, ctx: Ctx) !Velocity {
        const c: c_int = @intCast(self.cfg.model_channels);
        const nz: c_int = @intCast(self.cfg.q_token_length);
        var h = try sc.concat(&.{ st.hx, ctx.tokens, st.hc }, 1);
        for (self.blocks, self.repo) |*b, *r| {
            var bs = Scope.init(sc.a, sc.s);
            defer bs.deinit();
            const rope = try r.angles(&bs, h, self.heads());
            const y = try self.block(&bs, b, h, st.t_mod, rope, ctx.bias);
            try queue(y, h);
            h = try sc.keep(bs.out(y));
        }
        const total = mlx.getShape(h)[1];
        const lat = try sc.layerNorm(try L.asDtype(sc, try sc.slice(h, 1, 0, nz), .float32), NONE, NONE, L.FINAL_LN_EPS);
        const cm = try sc.layerNorm(try L.asDtype(sc, try sc.slice(h, 1, total - 1, total), .float32), NONE, NONE, L.FINAL_LN_EPS);
        const te = try sc.reshape(st.t_emb, &.{ 1, 1, c });
        const sh = try sc.add(self.shift_table, te);
        const shift = try sc.slice(sh, 1, 0, 1);
        const scale = try sc.slice(sh, 1, 1, 2);
        const one = try sc.scalar(1);
        const lat_m = try sc.add(try sc.mul(lat, try sc.add(scale, one)), shift);
        const cam_m = try sc.add(try sc.mul(cm, try sc.add(scale, one)), shift);
        return .{ .latent = try self.out.forward(sc, lat_m), .camera = try self.cam_out.forward(sc, cam_m) };
    }

    /// One velocity prediction: `stream` then `joint`.
    pub fn velocity(self: *const Denoiser, sc: *Scope, z: A, cam: A, t: f32, ctx: Ctx) !Velocity {
        return self.joint(sc, try self.stream(sc, z, cam, t), ctx);
    }
};

test "idol: a dense RePo layer's rotary tables keep f32 precision where MLX's matmul rounds to TF32" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !device.tf32Matmul()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const c = 256;
    const k = c / 8;
    const l = 300;
    var sc = Scope.init(a, mlx.gpuStream());
    defer {
        _ = mlx.mlx_stream_free(sc.s);
        sc.deinit();
    }
    var rng = std.Random.DefaultPrng.init(3);
    const vals = try a.alloc(f32, l * c + 2 * c + 2 * k * c + 24 * k + 32);
    defer a.free(vals);
    for (vals) |*v| v.* = rng.random().floatNorm(f32);
    // The maps at their checkpoint scale (unit outputs), so the angles stay within a few turns.
    for (vals[l * c + 2 * c ..][0 .. 2 * k * c]) |*v| v.* /= 16;
    for (vals[l * c + 2 * c + 2 * k * c ..][0 .. 24 * k]) |*v| v.* /= 6;
    var at: usize = 0;
    const take = struct {
        fn f(s: *Scope, v: []const f32, i: *usize, shape: []const c_int) !A {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            defer i.* += n;
            return s.fromF32(v[i.*..][0..n], shape);
        }
    }.f;
    const h = try L.asDtype(&sc, try take(&sc, vals, &at, &.{ 1, l, c }), .float16);
    var r = Repo{
        .norm_w = try take(&sc, vals, &at, &.{c}),
        .norm_b = try take(&sc, vals, &at, &.{c}),
        .gate = .{ .w = try take(&sc, vals, &at, &.{ k, c }) },
        .content = .{ .w = try take(&sc, vals, &at, &.{ k, c }) },
        .final = .{ .w = try take(&sc, vals, &at, &.{ 24, k }) },
        .freqs = .{ try take(&sc, vals, &at, &.{10}), try take(&sc, vals, &at, &.{10}), try take(&sc, vals, &at, &.{12}) },
    };
    var owned: std.ArrayList(A) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    try r.fuse(a, &owned);
    // The chain on the CPU: its f32 products are exact FMAs.
    var cpu = Scope.init(a, mlx.mlx_default_cpu_stream_new());
    defer {
        _ = mlx.mlx_stream_free(cpu.s);
        cpu.deinit();
    }
    // Centred tokens and tokens whose mean is twice their spread; plain TF32 misses the bar.
    for ([_]f32{ 0, 2 }) |offset| {
        const x = try L.asDtype(&sc, try sc.add(try L.asDtype(&sc, h, .float32), try sc.scalar(offset)), .float16);
        const got = try r.angles(&sc, x, 8);
        const want = try r.angles(&cpu, x, 8);
        try std.testing.expect(try w8a8.relRms(got.sin, want.sin, cpu.s) < 1e-4);
        try std.testing.expect(try w8a8.relRms(got.cos, want.cos, cpu.s) < 1e-4);
    }
}
