//! Hy4-flash's gated MLA attention: q_a -> RMSNorm -> q_b (heads x (nope+rope)), kv_a -> RMSNorm -> latent
//! (kv_lora) + k_rot (rope), interleaved RoPE on the trailing rope channels, MLA absorption (q_nope folded into
//! the latent by `kv_b`'s K half), learned per-head sinks, the latent cache, and the sublayer's close:
//! `sigmoid(linear_gate(x))` gates the expanded values, then `post_proj_norm(o_proj(..))`.
//!
//! Reference (`HYV4FlashAttention`): keys are `[latent | k_rot]`, values are the latent, the sinks are ONE extra
//! per-head logit with a zero value, and V is lifted from the attended latent by `kv_b`'s V half AFTER the
//! reduction. Sliding layers attend their last `sliding_window` keys; sparse layers attend the full causal
//! history below the recall budget and the compressed indexer's mask beyond it.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const block = @import("../../../sdk/bases/block.zig");
const kv = @import("../../../sdk/bases/kv.zig");
const sliding = @import("../../../sdk/bases/blocks/sliding.zig");
const sliding_ring = @import("../../../sdk/config/sliding_ring.zig");
const proj = @import("proj.zig");
const indexer_mod = @import("indexer.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = block.Ctx;
const Proj = proj.Proj;
const gpa = std.heap.c_allocator;

pub const Attn = struct {
    q_a: Proj,
    q_a_norm: Array,
    q_b: Proj,
    kv_a: Proj,
    kv_a_norm: Array,
    /// `kv_b`'s K half, dense `[heads, nope, rank]`: folds `q_nope` into the latent.
    wk: Array,
    /// `kv_b`'s V half, dense `[heads, v_dim, rank]`: lifts the attended latent to values.
    wv: Array,
    linear_gate: Proj,
    o_proj: Proj,
    post_proj_norm: Array,
    /// `[heads]` f32 learned sink logits.
    sink: Array,
    heads: u32,
    nope: u32,
    rope_dim: u32,
    v_dim: u32,
    rank: u32,
    /// RoPE base: sparse layers rotate long (1e7), sliding ones short (1e4).
    theta: f32,
    /// Interleaved M-RoPE frequency split [t,h,w] for image prompts.
    mrope_sections: [3]u32 = .{ 0, 0, 0 },
    /// Keys a sliding layer attends; 0 = the full causal history.
    window: u32,
    /// The compressed indexer (DSA layers); null = attend the full causal history.
    indexer: ?*const indexer_mod.Indexer = null,
    rms_eps: f32,

    pub fn forwards(self: *const Attn, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const s = c.rows;
        const heads: c_int = @intCast(self.heads);
        const nope: c_int = @intCast(self.nope);
        const rope_dim: c_int = @intCast(self.rope_dim);
        const rank: c_int = @intCast(self.rank);
        const vd: c_int = @intCast(self.v_dim);
        const qk: c_int = nope + rope_dim;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
        const act = ops.dtype(x);

        var t = g.scope();
        defer t.deinit();

        const q_resid = try t.rmsNorm(try self.q_a.forward(&t, x), self.q_a_norm, self.rms_eps);
        const q_flat = try self.q_b.forward(&t, q_resid);
        const q4 = try t.transpose(try t.reshape(q_flat, &.{ ops.shape(x)[0], s, heads, qk }), &.{ 0, 2, 1, 3 });
        const q_nope = try t.slice(q4, &.{ 0, 0, 0, 0 }, &.{ ops.shape(x)[0], heads, s, nope });
        const q_rot = try t.slice(q4, &.{ 0, 0, 0, nope }, &.{ ops.shape(x)[0], heads, s, qk });

        const kv_a = try self.kv_a.forward(&t, x);
        const lat_raw = try t.slice(kv_a, &.{ 0, 0, 0 }, &.{ ops.shape(x)[0], s, rank });
        const k_rot = try t.slice(kv_a, &.{ 0, 0, rank }, &.{ ops.shape(x)[0], s, rank + rope_dim });
        const latent = try t.rmsNorm(lat_raw, self.kv_a_norm, self.rms_eps);

        const off = c.offset();
        const q_rot_r = if (c.mrope_positions) |mp|
            try ropeM(&t, q_rot, rope_dim, self.theta, self.mrope_sections, mp)
        else
            try rope(&t, q_rot, rope_dim, self.theta, @intCast(off));
        const k_rot_r = if (c.mrope_positions) |mp|
            try ropeM(&t, k_rot, rope_dim, self.theta, self.mrope_sections, mp)
        else
            try rope(&t, k_rot, rope_dim, self.theta, @intCast(off));

        // MLA absorption: q_nope @ Wk lands the query in the latent; V is lifted after the reduction.
        const q_abs = try t.matmul(q_nope, self.wk); // [slots, heads, rows, rank]
        const q_cat = try t.concat(&.{ q_abs, q_rot_r }, -1);

        // A DSA layer caches its token keys beside the latent (the indexer pools them); the attention keys are
        // the leading `rank + rope` channels.
        const ix_dim: c_int = if (self.indexer) |ix| @intCast(ix.dim) else 0;
        const token_keys = if (self.indexer) |ix| try ix.tokenKeys(&t, x) else Array{};
        const k_row = if (ix_dim > 0)
            try t.reshape(try t.concat(&.{ latent, k_rot_r, token_keys }, -1),
                &.{ ops.shape(x)[0], 1, s, rank + rope_dim + ix_dim })
        else
            try t.reshape(try t.concat(&.{ latent, k_rot_r }, -1), &.{ ops.shape(x)[0], 1, s, rank + rope_dim });
        const v_row = try t.reshape(latent, &.{ ops.shape(x)[0], 1, s, rank });

        const total: c_int = @as(c_int, @intCast(off)) + s;
        const sinks = try t.astype(self.sink, act);
        var o_lat: Array = .{};
        if (self.window == 0) {
            var view = try kv.append(c, k_row, v_row, 0);
            defer kv.release(c, &view);
            const dv = kv.dense(view);
            const kv_len: c_int = ops.shape(dv.k)[2];
            const keys = if (ix_dim > 0)
                try t.slice(dv.k, &.{ 0, 0, 0, 0 }, &.{ 1, 1, kv_len, rank + rope_dim })
            else dv.k;
            const budget: c_int = if (self.indexer) |ix| @intCast(ix.budget) else 0;
            if (self.indexer != null and kv_len > budget) {
                // Beyond the recall budget the compressed indexer picks which slots each query may read.
                const tk_all = try t.slice(dv.k, &.{ 0, 0, 0, rank + rope_dim }, &.{ 1, 1, kv_len, rank + rope_dim + ix_dim });
                const base_pos: c_int = @as(c_int, @intCast(off)) - (kv_len - s);
                const mask = try self.indexer.?.recallMask(&t, x, q_resid, tk_all, kv_len, base_pos, c.positions);
                o_lat = try sdpa(&t, q_cat, keys, dv.v, scale, sinks, "array", mask);
            } else o_lat = try sdpa(&t, q_cat, keys, dv.v, scale, sinks, if (s > 1) "causal" else "", .{});
        } else {
            const wc: c_int = @intCast(self.window);
            const sv = sliding.view(self.window, qk, total, s);
            var view = try kv.appendSliding(c, c.cache(), k_row, v_row, sv.span, self.window, sliding_ring.rows(self.window));
            defer kv.release(c, &view);
            const dv = kv.dense(view);
            const kv_len: c_int = ops.shape(dv.k)[2];
            const keys_w = dv.k;
            if (s > 1 and total <= wc) {
                o_lat = try sdpa(&t, q_cat, keys_w, dv.v, scale, sinks, "causal", .{});
            } else if (s > 1) {
                o_lat = try sdpa(&t, q_cat, keys_w, dv.v, scale, sinks, "array", try sliding.bandMask(g, s, kv_len, wc));
            } else if (total <= wc) {
                o_lat = try sdpa(&t, q_cat, keys_w, dv.v, scale, sinks, "", .{});
            } else {
                o_lat = try sdpa(&t, q_cat, keys_w, dv.v, scale, sinks, "array", try sliding.decodeMask(g, kv_len, wc));
            }
        }

        // Lift the attended latent to values per head, gate on the input, then the sandwich norm.
        const wv_t = try t.transpose(self.wv, &.{ 0, 2, 1 }); // [heads, rank, v_dim]
        const values = try t.matmul(o_lat, wv_t); // [slots, heads, rows, v_dim]
        const flat = try t.reshape(try t.transpose(values, &.{ 0, 2, 1, 3 }),
            &.{ ops.shape(x)[0], s, heads * vd });
        const gate = try t.sigmoid(try self.linear_gate.forward(&t, x));
        const gated = try t.mul(flat, gate);
        const out = try t.rmsNorm(try self.o_proj.forward(&t, gated), self.post_proj_norm, self.rms_eps);
        return g.adopt(t.escape(out));
    }
};

/// Interleaved M-RoPE on the trailing `dim` channels from RAW 3-D positions `[3, rows]` int32: frequency index
/// `i` uses the time plane unless it falls in the height (`i % 3 == 1`, first `section[1]*3`) or width
/// (`i % 3 == 2`, first `section[2]*3`) slice; each layer kind supplies its own base.
fn ropeM(t: *Graph, x: Array, dim: c_int, base: f32, sections: [3]u32, pos: Array) !Array {
    const sh = ops.shape(x);
    const rank = sh.len;
    if (rank < 3) return error.BadRopeShape;
    const rows: usize = @intCast(sh[rank - 2]);
    const dt = ops.dtype(x);
    const half = @divExact(@as(usize, @intCast(dim)), 2);
    try ops.eval(pos);
    const p = mlx.mlx_array_data_int32(pos) orelse return error.NoData;
    const cos_buf = try gpa.alloc(f32, rows * half);
    defer gpa.free(cos_buf);
    const sin_buf = try gpa.alloc(f32, rows * half);
    defer gpa.free(sin_buf);
    for (0..rows) |r| for (0..half) |i| {
        const axis: usize = if (i < sections[1] * 3 and i % 3 == 1) 1 else if (i < sections[2] * 3 and i % 3 == 2) 2 else 0;
        const ang = @as(f32, @floatFromInt(p[axis * rows + r])) * std.math.pow(f32, base, -@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        cos_buf[r * half + i] = @cos(ang);
        sin_buf[r * half + i] = @sin(ang);
    };
    const cos_dt = try t.astype(try t.fromSlice(f32, cos_buf, &.{ @intCast(rows), @intCast(half) }), dt);
    const sin_dt = try t.astype(try t.fromSlice(f32, sin_buf, &.{ @intCast(rows), @intCast(half) }), dt);

    var esh: [8]c_int = @splat(1);
    esh[rank - 2] = @intCast(rows);
    esh[rank - 1] = @intCast(half);
    var csh: [8]c_int = esh;
    csh[rank] = 1;
    const cf = try t.reshape(try t.astype(cos_dt, .float32), csh[0 .. rank + 1]);
    const sf = try t.reshape(try t.astype(sin_dt, .float32), csh[0 .. rank + 1]);

    var psh: [8]c_int = undefined;
    @memcpy(psh[0 .. rank - 1], sh[0 .. rank - 1]);
    psh[rank - 1] = @intCast(half);
    psh[rank] = 2;
    const pairs = try t.reshape(x, psh[0 .. rank + 1]);
    var lo0: [8]c_int = @splat(0);
    var hi0: [8]c_int = psh;
    hi0[rank] = 1;
    var lo1: [8]c_int = lo0;
    lo1[rank] = 1;
    var hi1: [8]c_int = psh;
    hi1[rank] = 2;
    const x0 = try t.astype(try t.slice(pairs, lo0[0 .. rank + 1], hi0[0 .. rank + 1]), .float32);
    const x1 = try t.astype(try t.slice(pairs, lo1[0 .. rank + 1], hi1[0 .. rank + 1]), .float32);
    const o0 = try t.sub(try t.mul(x0, cf), try t.mul(x1, sf));
    const o1 = try t.add(try t.mul(x0, sf), try t.mul(x1, cf));
    const joined = try t.concat(&.{ o0, o1 }, -1);
    return t.astype(try t.reshape(joined, sh), dt);
}

/// RoPE on the trailing `dim` channels of `x` (`[slots, heads, rows, dim]` or `[slots, rows, dim]`), adjacent-pair
/// (interleaved) rotation: MLX's `traditional` arm rounds the angle table to the activation dtype before it rotates,
/// as the reference does.
fn rope(t: *Graph, x: Array, dim: c_int, base: f32, offset: c_int) !Array {
    return t.call(mlx.mlx_fast_rope, .{ x, dim, true, mlx.mlx_optional_float.some(base), 1.0, offset, Array{} });
}

fn sdpa(t: *Graph, q: Array, k: Array, v: Array, scale: f32, sinks: Array, mode: [*:0]const u8, mask: Array) !Array {
    return t.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, k, v, scale, mode, mask, sinks, false });
}

// ── tests ──

const testing = std.testing;

/// An array's values as f32: bf16/f16 casts first (`mlx_array_data_float32` reads f32 buffers only).
fn readF32(a: Array) ![]const f32 {
    if (mlx.mlx_array_dtype(a) == .float32) {
        try ops.eval(a);
        const n = mlx.mlx_array_size(a);
        const data = mlx.mlx_array_data_float32(a) orelse return error.NoData;
        return data[0..n];
    }
    var cast = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, a, .float32, mlx.gpuStream()));
    try ops.eval(cast);
    const n = mlx.mlx_array_size(cast);
    const data = mlx.mlx_array_data_float32(cast) orelse return error.NoData;
    return data[0..n];
}

test "hy_v4 mla: interleaved RoPE on a pair rotates as the reference does (adjacent pairs, rounded table)" {
    // One row, one head, rope width 4: pairs (0,1) and (2,3) rotate by theta_0 and theta_1.
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();

    const base: f32 = 10000.0;
    const dim: usize = 4;
    const pos: usize = 3;
    var x = [_]f32{ 0.7, -0.3, 1.1, 0.2 };
    const x_arr = try g.fromSlice(f32, &x, &.{ 1, 1, dim });
    defer g.drop(x_arr);
    const out = try rope(&g, x_arr, @intCast(dim), base, @intCast(pos));
    defer g.drop(out);
    const got = try readF32(out);

    // inv_freq[i] = base^(-i/(dim/2)); theta_i = pos * inv_freq[i]; the table rounds to f32, the rotation is f32.
    const half = dim / 2;
    for (0..half) |i| {
        const inv = std.math.pow(f64, @as(f64, base), -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half)));
        const theta = @as(f64, @floatFromInt(pos)) * inv;
        const c: f32 = @floatCast(@cos(theta));
        const sn: f32 = @floatCast(@sin(theta));
        const x1 = x[2 * i];
        const x2 = x[2 * i + 1];
        try testing.expectApproxEqAbs(x1 * c - x2 * sn, got[2 * i], 1e-5);
        try testing.expectApproxEqAbs(x1 * sn + x2 * c, got[2 * i + 1], 1e-5);
    }
}

test "hy_v4 mla: absorbed attention, the sink logit, the gate and the sandwich norm match f64" {
    // Tiny geometry (2 heads, nope 2 + rope 2, rank 2, v 2, hidden 4), every projection dense (pre-transposed),
    // a real `TestCache`: prefill 3 rows, then a decode row, each compared to the reference math in f64.
    const hidden: usize = 4;
    const heads: usize = 2;
    const nope: usize = 2;
    const rope_dim: usize = 2;
    const rank: usize = 2;
    const vd: usize = 2;
    const q_lora: usize = 2;
    const qk = nope + rope_dim;
    const eps: f64 = 1e-5;
    const theta: f32 = 10000.0;

    var rng = std.Random.DefaultPrng.init(77);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const fill = struct {
        fn f(a: std.mem.Allocator, n: usize, r: *std.Random.DefaultPrng) ![]f64 {
            const o = try a.alloc(f64, n);
            for (o) |*v| v.* = r.random().float(f64) * 2 - 1;
            return o;
        }
        fn near1(a: std.mem.Allocator, n: usize, r: *std.Random.DefaultPrng) ![]f64 {
            const o = try a.alloc(f64, n);
            for (o) |*v| v.* = 1 + r.random().float(f64) * 0.2 - 0.1;
            return o;
        }
    };
    const qa = try fill.f(aa, q_lora * hidden, &rng);
    const qb = try fill.f(aa, heads * qk * q_lora, &rng);
    const kva = try fill.f(aa, (rank + rope_dim) * hidden, &rng);
    const wk = try fill.f(aa, heads * nope * rank, &rng);
    const wv = try fill.f(aa, heads * vd * rank, &rng);
    const gate_w = try fill.f(aa, heads * vd * hidden, &rng);
    const ow = try fill.f(aa, hidden * heads * vd, &rng);
    const qan = try fill.near1(aa, q_lora, &rng);
    const kvan = try fill.near1(aa, rank, &rng);
    const ppn = try fill.near1(aa, hidden, &rng);
    const sink = try fill.f(aa, heads, &rng);
    const rows_in = try aa.alloc([hidden]f64, 4);
    for (rows_in) |*row| {
        for (row) |*v| v.* = rng.random().float(f64) * 2 - 1;
    }

    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();

    // A dense projection stored `[out, in]` becomes MLX's pre-transposed `[in, out]`.
    const mk = struct {
        fn dense(graph: *Graph, w: []const f64, od: usize, id: usize, a: std.mem.Allocator) !Proj {
            const t = try a.alloc(f32, w.len);
            for (0..od) |o| {
                for (0..id) |i| t[i * od + o] = @floatCast(w[o * id + i]);
            }
            return .{ .w = try graph.fromSlice(f32, t, &.{ @intCast(id), @intCast(od) }) };
        }
        fn vec(graph: *Graph, w: []const f64, a: std.mem.Allocator) !Array {
            const t = try a.alloc(f32, w.len);
            for (w, t) |v, *o| o.* = @floatCast(v);
            return graph.fromSlice(f32, t, &.{@intCast(w.len)});
        }
    };

    const attn = Attn{
        .q_a = try mk.dense(&g, qa, q_lora, hidden, aa),
        .q_a_norm = try mk.vec(&g, qan, aa),
        .q_b = try mk.dense(&g, qb, heads * qk, q_lora, aa),
        .kv_a = try mk.dense(&g, kva, rank + rope_dim, hidden, aa),
        .kv_a_norm = try mk.vec(&g, kvan, aa),
        .wk = try g.fromSlice(f32, blk: {
            const t = try aa.alloc(f32, wk.len);
            for (wk, t) |v, *o| o.* = @floatCast(v);
            break :blk t;
        }, &.{ @intCast(heads), @intCast(nope), @intCast(rank) }),
        .wv = try g.fromSlice(f32, blk: {
            const t = try aa.alloc(f32, wv.len);
            for (wv, t) |v, *o| o.* = @floatCast(v);
            break :blk t;
        }, &.{ @intCast(heads), @intCast(vd), @intCast(rank) }),
        .linear_gate = try mk.dense(&g, gate_w, heads * vd, hidden, aa),
        .o_proj = try mk.dense(&g, ow, hidden, heads * vd, aa),
        .post_proj_norm = try mk.vec(&g, ppn, aa),
        .sink = try mk.vec(&g, sink, aa),
        .heads = heads,
        .nope = nope,
        .rope_dim = rope_dim,
        .v_dim = vd,
        .rank = rank,
        .theta = theta,
        .window = 0,
        .rms_eps = @floatCast(eps),
    };
    defer {
        inline for (.{ attn.q_a, attn.q_b, attn.kv_a, attn.linear_gate, attn.o_proj }) |p| g.drop(p.w);
        for ([_]Array{ attn.q_a_norm, attn.kv_a_norm, attn.wk, attn.wv, attn.post_proj_norm, attn.sink }) |x| g.drop(x);
    }

    const cached = @import("../../../sdk/testing/cache_ops.zig");
    var cache = try cached.TestCache.init(testing.allocator, 1);
    defer cache.deinit();
    var shared = block.Shared.init(@floatCast(eps), @floatCast(eps));
    defer shared.deinit();

    // ── the oracle ──
    const orc = MlaOracle{
        .hidden = hidden, .heads = heads, .nope = nope, .rope_dim = rope_dim, .rank = rank, .vd = vd,
        .q_lora = q_lora, .qk = qk, .eps = eps, .theta = theta,
        .qa = qa, .qb = qb, .kva = kva, .wk = wk, .wv = wv, .gate_w = gate_w, .ow = ow,
        .qan = qan, .kvan = kvan, .ppn = ppn, .sink = sink, .rows_in = rows_in,
    };
    var lat_all: [4][rank]f64 = undefined;
    var krot_all: [4][rope_dim]f64 = undefined;
    var qfull_all: [4][heads * qk]f64 = undefined;
    for (0..4) |r| orc.row(r, &qfull_all[r], &lat_all[r], &krot_all[r]);

    // ── the component: prefill rows 0..2, then decode row 3 ──
    var pos: [1]u32 = .{0};
    var ctx: block.Ctx = .{
        .g = &g, .mode = .prefill, .rows = 3, .positions = &pos,
        .caches = &.{cache.handle()}, .cache_ops = &cached.ops, .shared = &shared,
    };
    var got_all: [4][hidden]f64 = undefined;
    {
        const t3 = try aa.alloc(f32, 3 * hidden);
        for (0..3) |r| {
            for (0..hidden) |i| t3[r * hidden + i] = @floatCast(rows_in[r][i]);
        }
        const x3 = try g.fromSlice(f32, t3, &.{ 1, 3, @intCast(hidden) });
        defer g.drop(x3);
        const o = try attn.forwards(&ctx, x3);
        defer g.drop(o);
        const got = try readF32(o);
        for (0..3) |r| {
            for (0..hidden) |i| got_all[r][i] = got[r * hidden + i];
        }

        const t1 = try aa.alloc(f32, hidden);
        for (0..hidden) |i| t1[i] = @floatCast(rows_in[3][i]);
        const x1 = try g.fromSlice(f32, t1, &.{ 1, 1, @intCast(hidden) });
        defer g.drop(x1);
        ctx.rows = 1;
        ctx.mode = .decode;
        pos = .{3};
        const o1 = try attn.forwards(&ctx, x1);
        defer g.drop(o1);
        const got1 = try readF32(o1);
        for (0..hidden) |i| got_all[3][i] = got1[i];
    }

    for (0..4) |r| {
        var want: [hidden]f64 = undefined;
        orc.out(r, &qfull_all[r], &lat_all, &krot_all, &want);
        for (0..hidden) |i| try testing.expectApproxEqAbs(@as(f32, @floatCast(want[i])), @as(f32, @floatCast(got_all[r][i])), 3e-4);
    }
}

/// The reference's per-row math in f64 over the test's matrices (`[out, in]` layouts, all dense).
const MlaOracle = struct {
    hidden: usize,
    heads: usize,
    nope: usize,
    rope_dim: usize,
    rank: usize,
    vd: usize,
    q_lora: usize,
    qk: usize,
    eps: f64,
    theta: f32,
    qa: []const f64,
    qb: []const f64,
    kva: []const f64,
    wk: []const f64,
    wv: []const f64,
    gate_w: []const f64,
    ow: []const f64,
    qan: []const f64,
    kvan: []const f64,
    ppn: []const f64,
    sink: []const f64,
    rows_in: []const [4]f64,

    fn norm(v: []const f64, w: []const f64, n: usize, eps_: f64, dst: []f64) void {
        var ms: f64 = 0;
        for (v[0..n]) |q| ms += q * q;
        ms /= @floatFromInt(n);
        const inv = 1.0 / @sqrt(ms + eps_);
        for (0..n) |i| dst[i] = v[i] * inv * w[i];
    }

    fn lin(x: []const f64, w: []const f64, od: usize, id: usize, dst: []f64) void {
        for (0..od) |o_| {
            var acc: f64 = 0;
            for (0..id) |i| acc += x[i] * w[o_ * id + i];
            dst[o_] = acc;
        }
    }

    /// Adjacent-pair rotation at `pos` for a 2-wide tail (rope_dim == 2 in this fixture).
    fn rope2(v: [2]f64, pos: usize, base: f64) [2]f64 {
        const ang = @as(f64, @floatFromInt(pos)) * std.math.pow(f64, base, -0.0);
        const c = @cos(ang);
        const sn = @sin(ang);
        return .{ v[0] * c - v[1] * sn, v[0] * sn + v[1] * c };
    }

    /// Row `r`'s roped query `[heads*qk]`, its latent `[rank]` and `k_rot[rope_dim]`.
    fn row(o: *const MlaOracle, r: usize, qfull: []f64, lat: []f64, krot: []f64) void {
        var qres_raw: [8]f64 = undefined;
        lin(o.rows_in[r][0..], o.qa, o.q_lora, o.hidden, qres_raw[0..o.q_lora]);
        var qres: [8]f64 = undefined;
        norm(qres_raw[0..o.q_lora], o.qan, o.q_lora, o.eps, qres[0..o.q_lora]);
        lin(qres[0..o.q_lora], o.qb, o.heads * o.qk, o.q_lora, qfull);
        var kvraw: [16]f64 = undefined;
        lin(o.rows_in[r][0..], o.kva, o.rank + o.rope_dim, o.hidden, kvraw[0 .. o.rank + o.rope_dim]);
        norm(kvraw[0..o.rank], o.kvan, o.rank, o.eps, lat);
        const kr = rope2(.{ kvraw[o.rank], kvraw[o.rank + 1] }, r, @as(f64, o.theta));
        krot[0] = kr[0];
        krot[1] = kr[1];
        for (0..o.heads) |h| {
            const base = h * o.qk + o.nope;
            const qr = rope2(.{ qfull[base], qfull[base + 1] }, r, @as(f64, o.theta));
            qfull[base] = qr[0];
            qfull[base + 1] = qr[1];
        }
    }

    /// Query row `r` attending keys `0..r` with the sink, then the gate, `o_proj` and the sandwich norm.
    fn out(o: *const MlaOracle, r: usize, qfull: []const f64, lat: []const [2]f64, krot: []const [2]f64, y: []f64) void {
        var vals: [16]f64 = undefined;
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(o.qk)));
        for (0..o.heads) |h| {
            var qabs: [8]f64 = undefined;
            for (0..o.rank) |k| {
                var acc: f64 = 0;
                for (0..o.nope) |i| acc += qfull[h * o.qk + i] * o.wk[h * o.nope * o.rank + i * o.rank + k];
                qabs[k] = acc;
            }
            var logits: [33]f64 = undefined;
            var lo: f64 = -std.math.inf(f64);
            for (0..r + 1) |j| {
                var acc: f64 = 0;
                for (0..o.rank) |k| acc += qabs[k] * lat[j][k];
                for (0..o.rope_dim) |k| acc += qfull[h * o.qk + o.nope + k] * krot[j][k];
                logits[j] = acc * scale;
                lo = @max(lo, logits[j]);
            }
            logits[r + 1] = o.sink[h];
            lo = @max(lo, o.sink[h]);
            var sum: f64 = 0;
            for (0..r + 2) |j| {
                logits[j] = @exp(logits[j] - lo);
                sum += logits[j];
            }
            var ot: [8]f64 = @splat(0);
            for (0..r + 1) |j| {
                for (0..o.rank) |k| ot[k] += logits[j] / sum * lat[j][k];
            }
            for (0..o.vd) |d| {
                var acc: f64 = 0;
                for (0..o.rank) |k| acc += ot[k] * o.wv[h * o.vd * o.rank + d * o.rank + k];
                var gv: f64 = 0;
                for (0..o.hidden) |i| gv += o.rows_in[r][i] * o.gate_w[(h * o.vd + d) * o.hidden + i];
                vals[h * o.vd + d] = acc / (1.0 + @exp(-gv));
            }
        }
        var full: [8]f64 = undefined;
        lin(vals[0 .. o.heads * o.vd], o.ow, o.hidden, o.heads * o.vd, full[0..o.hidden]);
        norm(full[0..o.hidden], o.ppn, o.hidden, o.eps, y);
    }
};

test "hy_v4 mla: ropeM picks the time plane on i=0 and the height plane on i=1" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var g = Graph.init(testing.allocator, stream);
    defer g.deinit();
    var pos_buf = [_]i32{ 5, 6, 2, 3, 7, 8 }; // [3, rows]: t, h, w
    const pos = try g.fromSlice(i32, &pos_buf, &.{ 3, 2 });
    defer g.drop(pos);
    var x = [_]f32{ 1.0, -1.0, 0.5, 2.0, 0.0, 1.0, 0.25, -3.0 };
    const xa = try g.fromSlice(f32, &x, &.{ 1, 2, 4 });
    defer g.drop(xa);
    const out = try ropeM(&g, xa, 4, 10000.0, .{ 1, 1, 1 }, pos);
    defer g.drop(out);
    const got = try readF32(out);
    // row 0: pair (0,1) by t=5, pair (2,3) by h=2; row 1: t=6, h=3.
    const inv0: f32 = 1.0;
    const inv1: f32 = std.math.pow(f32, 10000.0, -1.0 / 2.0);
    for (0..2) |r| {
        const t: f32 = if (r == 0) 5 else 6;
        const h: f32 = if (r == 0) 2 else 3;
        const a0 = t * inv0;
        const a1 = h * inv1;
        const x0 = x[r * 4];
        const x1 = x[r * 4 + 1];
        const y0 = x[r * 4 + 2];
        const y1 = x[r * 4 + 3];
        try testing.expectApproxEqAbs(x0 * @cos(a0) - x1 * @sin(a0), got[r * 4], 1e-5);
        try testing.expectApproxEqAbs(x0 * @sin(a0) + x1 * @cos(a0), got[r * 4 + 1], 1e-5);
        try testing.expectApproxEqAbs(y0 * @cos(a1) - y1 * @sin(a1), got[r * 4 + 2], 1e-5);
        try testing.expectApproxEqAbs(y0 * @sin(a1) + y1 * @cos(a1), got[r * 4 + 3], 1e-5);
    }
}
