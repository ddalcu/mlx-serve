//! k2 `wq_b` (`~/plans/dsv41-decode-kernels-spec.md`): q_norm, then wq_b (and the indexer's) with the rope tails, the window row and the attention split.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const system = @import("../../../sdk/util/system.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_wq_b.metal");
pub const HEADER = dsv41.FP8_HEADER ++ kernelSource("plugins/deepseek_v41/kernels/dec_wq_b_header.metal");
pub const INS = [_][*:0]const u8{ "qa", "qn", "w_q", "s_q", "w_i", "s_i", "kv", "kvn", "periods", "periods_i", "pos", "z", "eps" };
pub const OUTS = [_][*:0]const u8{ "q", "q_idx", "kv_row", "pre", "post", "comb" };

/// The query latent's width and the kv row's.
const K = 1280;
const KV = 512;

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_wq_b", &INS, &OUTS, SOURCE, HEADER);
}

pub const Out = struct { q: A, q_idx: ?A, kv_row: A, sa: dsv41.Split };

/// Simdgroups a threadgroup holds, and the threadgroups that walk the units: two a GPU core (one where the GPU's
/// cores are unknown).
const SGS = 8;

pub fn groups() usize {
    const cores = system.gpuCoreCount();
    return if (cores > 0) 2 * cores else 160;
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dim(x: A, axis: usize) usize {
    return @intCast(mlx.getShape(x)[axis]);
}

fn mxfp8Rows(l: dsv41.Lin, n: usize) bool {
    if (l.qp.mode != .mxfp8 or l.qp.bits != 8 or l.qp.group_size != 32) return false;
    if (mlx.mlx_array_dtype(l.w) != .uint32 or mlx.mlx_array_dtype(l.s) != .uint8) return false;
    const ws = mlx.getShape(l.w);
    const ss = mlx.getShape(l.s);
    return ws.len == 2 and ss.len == 2 and ws[0] == n and ss[0] == n and ws[1] * 4 == K and ss[1] * 32 == K;
}

/// The rows of one decode step at positions pos .. pos + rows: `qa` and `kv` `[rows, K]`, `[rows, 512]` bf16 (the
/// `q_kv` GEMV's), `z` `[rows, 24]` f32 (this stage's hc mixes), `pos` an int32 scalar. Null where the shape or the
/// weights' format is not served.
pub fn run(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, kern: mlx.mlx_fast_metal_kernel, ly: *const dsv41.Layer, li: usize, qa: A, kv: A, z: A, pos: A) !?Out {
    return launch(sc, m, kern, ly, li, qa, kv, z, pos, groups());
}

/// `run` with `g` threadgroups walking the units.
pub fn launch(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, kern: mlx.mlx_fast_metal_kernel, ly: *const dsv41.Layer, li: usize, qa: A, kv: A, z: A, pos: A, g: usize) !?Out {
    const c = &m.cfg;
    if (m.qat_off or m.act != .bfloat16 or !mlx.streamIsGpu(sc.s)) return null;
    const rows = dim(qa, 0);
    const heads = c.num_attention_heads;
    const hd = c.head_dim;
    const hc = c.dsv4_hc_mult;
    if (rows < 1 or rows > decode.max_rows or dim(qa, 1) != K or hd != KV or c.dsv4_rope_head_dim != 64 or hc != 4) return null;
    if (c.dsv4_q_lora_rank != K or !mxfp8Rows(ly.wq_b, heads * hd)) return null;
    const idx_n = if (ly.idx != null) c.dsv4_index_n_heads * c.dsv4_index_head_dim else 0;
    if (ly.idx) |iw| if (c.dsv4_index_head_dim != 128 or idx_n % 32 != 0 or !mxfp8Rows(iw.wq_b, idx_n)) return null;
    if (heads * hd % (SGS * 2) != 0 or g < rows + 1) return null;
    const iw_q = if (ly.idx) |iw| iw.wq_b else ly.wq_b;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const act = mlx.mlx_array_dtype(qa);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(heads), ci(hd) }, 3, act));
    if (ly.idx != null)
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(c.dsv4_index_n_heads), ci(c.dsv4_index_head_dim) }, 3, .float32))
    else
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), KV }, 2, act));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc), ci(hc) }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(g * SGS * 32), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(SGS * 32), 1, 1));
    inline for (.{
        .{ "R", rows },
        .{ "SGS", SGS },
        .{ "G", g },
        .{ "NQ", heads * hd },
        .{ "NI", idx_n },
        .{ "HC", hc },
        .{ "ITERS", c.dsv4_hc_sinkhorn_iters },
        .{ "RD", c.dsv4_rope_head_dim },
    }) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], ci(t[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", act));
    const eps = try dsv41.hcEps(sc, m);
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ qa, ly.q_norm, ly.wq_b.w, ly.wq_b.s, iw_q.w, iw_q.s, kv, ly.kv_norm, dsv41.ropeFor(m, li).fwd, m.rope_yarn.fwd, pos, z, eps }, INS.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o: [OUTS.len]A = undefined;
    for (&o, 0..) |*a, k| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, ov, k));
        _ = try sc.keep(a.*);
    }
    return .{ .q = o[0], .q_idx = if (ly.idx != null) o[1] else null, .kv_row = o[2], .sa = .{ .pre = o[3], .post = o[4], .comb = o[5] } };
}

// ── kernel contract ──

const cx = @import("../contract.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const metal = kit.metal;
const spec = kit.spec;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const Iv = cx.Iv;

const wq_b_l = metal.Launch{
    .name = "dsv41_dec_wq_b",
    .inputs = &INS,
    .outputs = &OUTS,
    .source = SOURCE,
    .header = HEADER,
    // xs (the latent of 6 rows as float4 chunks), ps and t_s.
    .tg_bytes = 4 * (decode.max_rows * 4 * 320 + decode.max_rows * SGS + 16),
    .m1 = 512,
    .m2 = 512,
};

/// The kernel's inputs, in `INS` order.
const I = struct {
    const qa = 0;
    const qn = 1;
    const wq = 2;
    const wi = 4;
    const kv = 6;
    const kvn = 7;
    const periods = 8;
    const periods_i = 9;
    const pos = 10;
    const z = 11;
    const eps = 12;
};

/// An e4m3 byte's value (the OCP fn variant; S.1111.111 is never drawn).
fn e4m3(b: u8) f64 {
    const e: i32 = (b >> 3) & 15;
    const m: f64 = @floatFromInt(b & 7);
    const v = if (e == 0) m / 8 * 0x1p-6 else std.math.ldexp(1 + m / 8, e - 7);
    return if (b & 0x80 != 0) -v else v;
}

/// Row `c` of the mxfp8 linear at inputs `at` (words) and `at + 1` (exponents), dequantized in float64.
fn weightRow(k: anytype, at: usize, c: usize, out: []f64) void {
    const words = k.in(at)[c * (K / 4) ..][0 .. K / 4];
    const exps = k.in(at + 1)[c * (K / 32) ..][0 .. K / 32];
    for (out, 0..) |*o, j| {
        const w: u32 = @intFromFloat(words[j / 4]);
        const byte: u8 = @truncate(w >> @intCast(8 * (j % 4)));
        o.* = std.math.ldexp(e4m3(byte), @as(i32, @intFromFloat(exps[j / 32])) - 127);
    }
}

/// One row's `T(w * T(x * inv))` through the f32 rms: intervals, or float64 points in `point`.
fn normed(k: anytype, x: []const f64, w: []const f64, sum: Sum, point: bool, out: []Iv) !void {
    const a = k.scratch();
    const sq = try a.alloc(f64, x.len);
    defer a.free(sq);
    for (sq, x) |*q, v| q.* = v * v;
    const tot = sum.of(sq);
    const d: f64 = @floatFromInt(x.len);
    const g = if (point) 0 else spec.gamma(d, cx.u);
    const eps = spec.roundTo(.f32, k.in(I.eps)[0]);
    const inv = Iv{ .lo = 1 / @sqrt(tot * (1 + g) / d + eps), .hi = 1 / @sqrt(tot * (1 - g) / d + eps) };
    const iw = if (point) Iv.pt((inv.lo + inv.hi) / 2) else inv.widen(2 * cx.u, 0);
    const r: f64 = if (point) 0 else cx.u;
    for (out, x, w) |*o, xv, wv| o.* = Iv.pt(xv).mul(iw).widen(r, 0).round(cx.act).mul(Iv.pt(wv)).widen(r, 0).round(cx.act);
}

/// `T(qr . W[c])` for the `n` rows of the linear at input `at` against `rows` latents `qr` [rows, K]: intervals
/// through the f32 sum, or points in `point` (summed as `sum` adds them; `mg` the terms' magnitude).
fn gemv(k: anytype, at: usize, n: usize, rows: usize, qr: []const Iv, sum: Sum, point: bool, y: []Iv, mg: []f64) !void {
    const a = k.scratch();
    const w = try a.alloc(f64, K);
    defer a.free(w);
    const t = try a.alloc(f64, K);
    defer a.free(t);
    const gm = spec.gamma(K + 8, cx.u);
    for (0..n) |c| {
        weightRow(k, at, c, w);
        for (0..rows) |r| {
            const x = qr[r * K ..][0..K];
            var m: f64 = 0;
            if (point) {
                for (t, x, w) |*o, q, wv| o.* = q.lo * wv;
                for (t) |v| m += @abs(v);
                y[r * n + c] = Iv.pt(cx.bf(sum.of(t)));
            } else {
                var lo: f64 = 0;
                var hi: f64 = 0;
                for (x, w) |q, wv| {
                    const p0 = q.lo * wv;
                    const p1 = q.hi * wv;
                    lo += @min(p0, p1);
                    hi += @max(p0, p1);
                    m += @max(@abs(p0), @abs(p1));
                }
                y[r * n + c] = (Iv{ .lo = lo - gm * m, .hi = hi + gm * m }).round(cx.act);
            }
            mg[r * n + c] = m;
        }
    }
}

/// The tail rope of the T-rounded `y` [rows, n] (heads of `hd`) at position row + pos with the periods at input `at`.
fn rope(k: anytype, y: []const Iv, rows: usize, n: usize, hd: usize, at: usize, point: bool, out: []Iv) void {
    const head = hd - cx.RD;
    for (0..rows) |r| for (0..n) |c| {
        const i = r * n + c;
        const col = c % hd;
        if (col < head) {
            out[i] = y[i];
            continue;
        }
        const b = i - (col - head) % 2;
        const th = (@as(f64, @floatFromInt(r)) + k.in(I.pos)[0]) / spec.roundTo(.f32, k.in(at)[(col - head) / 2]);
        const even = (col - head) % 2 == 0;
        out[i] = if (point) blk: {
            const x1 = y[b].lo;
            const x2 = y[b + 1].lo;
            break :blk Iv.pt(cx.bf(if (even) x1 * @cos(th) - x2 * @sin(th) else x1 * @sin(th) + x2 * @cos(th)));
        } else cx.ropeIv(y[b], y[b + 1], th, even).round(cx.act);
    };
}

/// Float64 points through the round trip `q` over blocks of 32 along each row.
fn qatRows(q: cx.Qk, v: []f64) void {
    var i: usize = 0;
    while (i < v.len) : (i += 32) cx.qatPoint(q, v[i..][0..32]);
}

/// The packed e4m3 words of integer weights `w` (|w| <= 15, exact in e4m3), four to a word.
fn packCodes(w: []const f64, words: []f64) void {
    var tab: [16]u8 = undefined;
    for (&tab, 0..) |*c, v| c.* = for (0..127) |b| {
        if (e4m3(@intCast(b)) == @as(f64, @floatFromInt(v))) break @intCast(b);
    } else unreachable;
    for (words, 0..) |*o, i| {
        var v: u32 = 0;
        for (0..4) |t| {
            const x: i32 = @intFromFloat(w[4 * i + t]);
            const code = tab[@abs(x)] | @as(u8, if (x < 0) 0x80 else 0);
            v |= @as(u32, code) << @intCast(8 * t);
        }
        o.* = @floatFromInt(v);
    }
}

/// Random e4m3 words (never S.1111.111) and e8m0 exponents around 1.
fn randomWeights(rng: std.Random, words: []f64, exps: []f64) void {
    for (words) |*o| {
        var b: [4]u8 = undefined;
        for (&b) |*x| {
            x.* = rng.int(u8);
            if (x.* & 0x7f == 0x7f) x.* -= 1;
        }
        o.* = @floatFromInt(std.mem.readInt(u32, &b, .little));
    }
    for (exps) |*e| e.* = @floatFromInt(rng.intRangeAtMost(u8, 120, 130));
}

fn depthWqb(_: Shape) usize {
    return K + 8;
}
fn qN(s: Shape) usize {
    return s.rows * s.n;
}
fn idxN(s: Shape) usize {
    return s.rows * s.k;
}
fn kvN(s: Shape) usize {
    return s.rows * KV;
}
fn preN(s: Shape) usize {
    return s.rows * cx.HC;
}
fn combN(s: Shape) usize {
    return s.rows * cx.HC * cx.HC;
}

/// Contract shapes: `rows` the step's rows, `n` the q rows (heads x 512), `k` the indexer's rows (0: none); the
/// latent is always K = 1280 wide. `g`: threadgroups that walk the units.
const Wqb = struct {
    pub const Params = struct { g: usize };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 1 or s.rows > decode.max_rows or s.n == 0 or s.n % KV != 0 or s.k % cx.IHD != 0 or p.g < s.rows + 1) return null;
        var g = metal.Geometry{ .grid = .{ ci(p.g * SGS * 32), 1, 1 }, .tg = .{ SGS * 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n / KV), KV }, cx.act);
        if (s.k > 0) {
            g.out(&.{ ci(s.rows), ci(s.k / cx.IHD), cx.IHD }, .float32);
        } else {
            g.out(&.{1}, .float32);
            g.unread = 1 << 1;
        }
        g.out(&.{ ci(s.rows), KV }, cx.act);
        g.out(&.{ ci(s.rows), cx.HC }, .float32);
        g.out(&.{ ci(s.rows), cx.HC }, .float32);
        g.out(&.{ ci(s.rows), cx.HC, cx.HC }, .float32);
        inline for (.{ .{ "R", s.rows }, .{ "SGS", SGS }, .{ "G", p.g }, .{ "NQ", s.n }, .{ "NI", s.k }, .{ "HC", cx.HC }, .{ "ITERS", cx.ITERS }, .{ "RD", cx.RD } }) |t| g.int(t[0], t[1]);
        g.dtype("T", cx.act);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const R = k.shape.rows;
        const nq = k.shape.n;
        const ni = k.shape.k;
        const qa = try k.add(cx.act, &.{ R, K });
        const qn = try k.add(cx.act, &.{K});
        const wq = try k.add(.uint32, &.{ nq, K / 4 });
        const sq = try k.add(.uint8, &.{ nq, K / 32 });
        // Without the indexer the kernel is still bound a (never read) pair.
        const wi = try k.add(.uint32, &.{ @max(ni, 1), K / 4 });
        const si = try k.add(.uint8, &.{ @max(ni, 1), K / 32 });
        const kv = try k.add(cx.act, &.{ R, KV });
        const kvn = try k.add(cx.act, &.{KV});
        const periods = try k.add(.float32, &.{cx.RD / 2});
        const periods_i = try k.add(.float32, &.{cx.RD / 2});
        const pos = try k.add(.int32, &.{});
        const z = try k.add(.float32, &.{ R, cx.MIX });
        const eps = try k.add(.float32, &.{2});
        kit.uniform(rng, qa, -3, 3);
        kit.uniform(rng, qn, 0.5, 1.5);
        kit.uniform(rng, kv, -3, 3);
        kit.uniform(rng, kvn, 0.5, 1.5);
        kit.uniform(rng, z, -3, 3);
        cx.periodsFill(periods, 1);
        cx.periodsFill(periods_i, 1);
        for (periods_i) |*v| v.* *= 0.75;
        pos[0] = 3001;
        eps[0] = cx.EPS_RMS;
        eps[1] = cx.EPS_HC;
        randomWeights(rng, wq, sq);
        randomWeights(rng, wi, si);
        switch (f) {
            .random => {},
            .nonfinite => {
                kit.plantRows(qa, R, K);
                kit.plantRows(kv, R, KV);
                kit.plantRows(z, R, cx.MIX);
            },
            .stress => {
                // The latent is q_norm itself (|qa| = 1 is a unit rms, eps 0): a mirrored integer row (`stressDot`)
                // against integer weights in unit-scale codes, so every sum is exact in fp32 in any order. Angle 0,
                // the window row +-32 (an exact rsqrt) and z 0 keep the other stages inside `ops`.
                eps[0] = 0;
                @memset(qa, 1);
                const a = k.scratch();
                const w = try a.alloc(f64, (nq + ni) * K);
                defer a.free(w);
                try cx.stressDot(rng, qn, w, nq + ni, K, 255, 15, a);
                packCodes(w[0 .. nq * K], wq);
                packCodes(w[nq * K ..], wi[0 .. ni * K / 4]);
                @memset(sq, 127);
                @memset(si, 127);
                @memset(periods, kit.inf);
                @memset(periods_i, kit.inf);
                for (kv) |*v| v.* = if (rng.boolean()) 32 else -32;
                @memset(kvn, 1);
                @memset(z, 0);
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * (s.n + s.k + KV + cx.MIX);
    }

    /// The pre | post | comb outputs, one row of `z` after another.
    fn splits(k: anytype, out: []f64, sum: Sum) void {
        const R = k.shape.rows;
        const H = cx.HC;
        var pk: [2 * H + H * H]f64 = undefined;
        for (0..R) |r| {
            cx.splitRow(k.in(I.z)[r * cx.MIX ..][0..cx.MIX], spec.roundTo(.f32, k.in(I.eps)[1]), sum, &pk);
            @memcpy(out[r * H ..][0..H], pk[0..H]);
            @memcpy(out[R * H + r * H ..][0..H], pk[H .. 2 * H]);
            @memcpy(out[2 * R * H + r * H * H ..][0 .. H * H], pk[2 * H ..]);
        }
    }

    /// The window row's values before the round trip, for every row.
    fn window(k: anytype, sum: Sum, point: bool, y: []Iv, o: []Iv) !void {
        for (0..k.shape.rows) |r| try normed(k, k.in(I.kv)[r * KV ..][0..KV], k.in(I.kvn), sum, point, y[r * KV ..][0..KV]);
        rope(k, y, k.shape.rows, KV, KV, I.periods, point, o);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const s = k.shape;
        const R = s.rows;
        const a = k.scratch();
        const wide = @max(s.n, s.k, KV);
        const qr = try a.alloc(Iv, R * K);
        defer a.free(qr);
        const y = try a.alloc(Iv, R * wide);
        defer a.free(y);
        const o = try a.alloc(Iv, R * wide);
        defer a.free(o);
        const mg = try a.alloc(f64, R * wide);
        defer a.free(mg);
        @memset(mag, 0);
        for (0..R) |r| try normed(k, k.in(I.qa)[r * K ..][0..K], k.in(I.qn), sum, true, qr[r * K ..][0..K]);
        try gemv(k, I.wq, s.n, R, qr, sum, true, y, mg);
        rope(k, y, R, s.n, KV, I.periods, true, o);
        for (out[0 .. R * s.n], o[0 .. R * s.n]) |*v, iv| v.* = iv.lo;
        @memcpy(mag[0 .. R * s.n], mg[0 .. R * s.n]);
        if (s.k > 0) {
            try gemv(k, I.wi, s.k, R, qr, sum, true, y, mg);
            rope(k, y, R, s.k, cx.IHD, I.periods_i, true, o);
            const dst = out[R * s.n ..][0 .. R * s.k];
            for (dst, o[0 .. R * s.k]) |*v, iv| v.* = iv.lo;
            qatRows(.fp4, dst);
            for (dst) |*v| v.* = cx.bf(v.*);
        }
        try window(k, sum, true, y, o);
        const kv = out[R * (s.n + s.k) ..][0 .. R * KV];
        for (kv, o[0 .. R * KV]) |*v, iv| v.* = iv.lo;
        qatRows(.fp8, kv);
        const at = R * (s.n + s.k + KV);
        splits(k, out[at..], sum);
        for (mag[at..], out[at..]) |*m, w| m.* = @abs(w);
    }

    /// Every output as the last run wrote it.
    fn outputs(k: anytype, got: []f64) void {
        const s = k.shape;
        const bytes = k.rig.outputBytes();
        var off: usize = 0;
        var at: usize = 0;
        inline for (.{ .{ spec.Elem.bf16, qN(s) }, .{ spec.Elem.f32, idxN(s) }, .{ spec.Elem.bf16, kvN(s) }, .{ spec.Elem.f32, s.rows * cx.MIX } }) |part| {
            for (0..part[1]) |i| got[at + i] = spec.read(part[0], bytes[off..], i);
            off += part[1] * spec.size(part[0]);
            at += part[1];
        }
    }

    pub fn check(k: anytype, gate: spec.Gate) !?[]const u8 {
        const s = k.shape;
        const R = s.rows;
        const a = k.scratch();
        const n = count(k.p, s);
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        const got = buf[0..n];
        outputs(k, got);
        const at = R * (s.n + s.k + KV);
        if (gate == .nonfinite) {
            const want = buf[n .. 2 * n];
            try compute(k, want, buf[2 * n ..], .{});
            // The quantized outputs last: a NaN row's codes are the known finding.
            if (cx.classes(&k.msg, got[0 .. R * s.n], want[0 .. R * s.n])) |why| return why;
            if (cx.classes(&k.msg, got[at..], want[at..])) |why| return why;
            return cx.classes(&k.msg, got[R * s.n .. at], want[R * s.n .. at]);
        }
        splits(k, buf[n + at .. 2 * n], .{});
        if (cx.splitOk(&k.msg, got[at..], buf[n + at .. 2 * n])) |why| return why;
        const wide = @max(s.n, s.k, KV);
        const qr = try a.alloc(Iv, R * K);
        defer a.free(qr);
        const y = try a.alloc(Iv, R * wide);
        defer a.free(y);
        const o = try a.alloc(Iv, R * wide);
        defer a.free(o);
        const mg = try a.alloc(f64, R * wide);
        defer a.free(mg);
        for (0..R) |r| try normed(k, k.in(I.qa)[r * K ..][0..K], k.in(I.qn), .{}, false, qr[r * K ..][0..K]);
        try gemv(k, I.wq, s.n, R, qr, .{}, false, y, mg);
        rope(k, y, R, s.n, KV, I.periods, false, o);
        for (o[0 .. R * s.n], got[0 .. R * s.n], 0..) |iv, g, i| if (!iv.has(g))
            return std.fmt.bufPrint(&k.msg, "q output {d} is {e} outside [{e}, {e}]", .{ i, g, iv.lo, iv.hi }) catch "q";
        if (s.k > 0) {
            try gemv(k, I.wi, s.k, R, qr, .{}, false, y, mg);
            rope(k, y, R, s.k, cx.IHD, I.periods_i, false, o);
            const idx = got[R * s.n ..][0 .. R * s.k];
            var b: usize = 0;
            while (b < idx.len) : (b += 32) if (cx.qatOk(.fp4, o[b..][0..32], idx[b..][0..32], cx.act, &k.msg, b)) |why| return why;
        }
        try window(k, .{}, false, y, o);
        const kv = got[R * (s.n + s.k) ..][0 .. R * KV];
        var b: usize = 0;
        while (b < kv.len) : (b += 32) if (cx.qatOk(.fp8, o[b..][0..32], kv[b..][0..32], cx.act, &k.msg, b)) |why| return why;
        return null;
    }
};

pub fn register(set: *kc.Set) !void {
    try kit.add(set, Wqb, &wq_b_l, .{ .g = 9 }, .{
        .name = "dsv41 dec wq_b (rms, wq_b + rope, indexer q fp4, window row fp8, attention split)",
        .accum = .f32,
        .depth = depthWqb,
        .ops = cx.sinkhorn_ops,
        .out = .bf16,
        .outs = &.{ .{ .elem = .bf16, .count = qN }, .{ .elem = .f32, .count = idxN }, .{ .elem = .bf16, .count = kvN }, .{ .elem = .f32, .count = preN }, .{ .elem = .f32, .count = preN }, .{ .elem = .f32, .count = combN } },
        .nan = cx.quant_nan,
        .known = cx.nan_code,
        .tiles = .{ .{ .size = 4 }, .{}, .{ .size = cx.IHD, .step = cx.IHD, .min = cx.IHD } },
        .served = &.{ .{ .rows = 1, .n = 2 * KV, .k = 2 * cx.IHD }, .{ .rows = 6, .n = KV, .k = cx.IHD }, .{ .rows = 1, .n = KV, .k = 0 }, .{ .rows = 6, .n = 2 * KV, .k = 0 } },
    });
}

// ── tests ──

const testing = std.testing;

/// The chain the kernel replaces, on the same inputs: `qkvProj`'s rms, wq_b and rope, `kvRow`, the indexer's wq_b and
/// `ropeQat`, and the attention's hc split.
fn chain(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, qa: A, kv: A, z: A, pos: A) !Out {
    const c = &m.cfg;
    const rows = dim(qa, 0);
    var o = mlx.mlx_array_new();
    const qr = try sc.res(mlx.mlx_fast_rms_norm(&o, qa, ly.q_norm, c.rms_norm_eps, sc.s), &o);
    const y = try ly.wq_b.apply(sc, qr);
    const q = try dsv41.ropeTail(sc, m, try sc.reshape(y, &.{ ci(rows), ci(c.num_attention_heads), ci(c.head_dim) }), dsv41.ropeFor(m, li).fwd, pos, 1.0);
    const q_idx = if (ly.idx) |iw|
        try dsv41.ropeQat(sc, m, try sc.reshape(try iw.wq_b.apply(sc, qr), &.{ ci(rows), ci(c.dsv4_index_n_heads), ci(c.dsv4_index_head_dim) }), m.rope_yarn.fwd, pos, 1.0, .fp4_ue8m0, .float32)
    else
        null;
    return .{ .q = q, .q_idx = q_idx, .kv_row = try dsv41.kvRow(sc, m, ly, dsv41.ropeFor(m, li).fwd, kv, pos), .sa = try split(sc, m, z) };
}

/// `hcStage`'s split of the mixes `z`.
fn split(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, z: A) !dsv41.Split {
    const rows = dim(z, 0);
    const hc = m.cfg.dsv4_hc_mult;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(hc), ci(hc) }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, ci(rows), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", ci(hc)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(m.cfg.dsv4_hc_sinkhorn_iters)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ z, try dsv41.hcEps(sc, m) }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.hc_split_kernel, iv, cfg, sc.s));
    var g: [3]A = undefined;
    for (&g, 0..) |*a, k| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, ov, k));
        _ = try sc.keep(a.*);
    }
    return .{ .pre = g[0], .post = g[1], .comb = g[2] };
}

/// `src` with `deps` its dependencies: evaluating it evaluates them (a chain's step carries this on).
fn after(sc: *dsv41.Scope, src: A, deps: []const A) !A {
    const iv = mlx.mlx_vector_array_new_data(&[_]A{src}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    const dv = mlx.mlx_vector_array_new_data(deps.ptr, deps.len);
    defer _ = mlx.mlx_vector_array_free(dv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_depends(&ov, iv, dv));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn randn(sc: *dsv41.Scope, key: u64, shape: []const c_int, dt: mlx.mlx_dtype, std_: f32) !A {
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_random_key(&k, key));
    var o = mlx.mlx_array_new();
    const a = try sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, std_, k, sc.s), &o);
    try mlx.check(mlx.mlx_array_eval(a));
    return a;
}

fn quantized(sc: *dsv41.Scope, key: u64, n: usize) !dsv41.Lin {
    const w = try randn(sc, key, &.{ ci(n), K }, .bfloat16, 0.03);
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, sc.s));
    var t: [2]A = undefined;
    for (&t, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, qv, i));
        _ = try sc.keep(a.*);
    }
    return .{ .w = t[0], .s = t[1], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
}

/// A model and `n` layers of it whose projections sit in distinct buffers with distinct random content, so a chain
/// over them reads memory, not a cache (`mlx_copy` shares the buffer).
const Bench = struct {
    fx: *fixture.Fixture,
    sc: dsv41.Scope,
    layers: []dsv41.Layer,
    li: usize = 20,

    fn init(n: usize) !*Bench {
        const b = try testing.allocator.create(Bench);
        b.* = .{ .fx = try fixture.Fixture.init(testing.allocator, .{ .lo = 20, .hi = 21, .experts = 2, .ctx = 0 }), .sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream()), .layers = try testing.allocator.alloc(dsv41.Layer, n) };
        errdefer b.deinit();
        var pending: std.ArrayList(A) = .empty;
        defer pending.deinit(testing.allocator);
        for (b.layers, 0..) |*ly, i| {
            ly.* = b.fx.layer(20).*;
            ly.wq_b = try quantized(&b.sc, 100 + 2 * i, 32768);
            ly.idx.?.wq_b = try quantized(&b.sc, 101 + 2 * i, 4096);
            try pending.appendSlice(testing.allocator, &.{ ly.wq_b.w, ly.wq_b.s, ly.idx.?.wq_b.w, ly.idx.?.wq_b.s });
            if (pending.items.len >= 64 or i + 1 == n) {
                const vec = mlx.mlx_vector_array_new_data(pending.items.ptr, pending.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_eval(vec));
                pending.clearRetainingCapacity();
            }
        }
        return b;
    }

    fn deinit(b: *Bench) void {
        b.sc.deinit();
        testing.allocator.free(b.layers);
        b.fx.deinit();
        testing.allocator.destroy(b);
    }
};

/// One chain's inputs for `rows` rows, and the step the chain driver calls.
const Steps = struct {
    b: *Bench,
    sc: dsv41.Scope,
    kern: mlx.mlx_fast_metal_kernel,
    g: usize,
    with_idx: bool,
    replaced: bool,
    qa: A,
    kv0: A,
    z: A,
    pos: A,
    kv: A = none,

    fn init(b: *Bench, kern: mlx.mlx_fast_metal_kernel, g: usize, rows: usize, with_idx: bool, replaced: bool) !Steps {
        var s: Steps = .{ .b = b, .sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream()), .kern = kern, .g = g, .with_idx = with_idx, .replaced = replaced, .qa = undefined, .kv0 = undefined, .z = undefined, .pos = undefined };
        errdefer s.sc.deinit();
        const r = ci(rows);
        s.qa = try randn(&b.sc, 11, &.{ r, K }, .bfloat16, 2);
        s.kv0 = try randn(&b.sc, 12, &.{ r, KV }, .bfloat16, 2);
        s.z = try randn(&b.sc, 13, &.{ r, 24 }, .float32, 1.5);
        s.pos = try b.sc.keep(mlx.mlx_array_new_int(3001));
        return s;
    }

    fn deinit(s: *Steps) void {
        s.sc.deinit();
    }

    pub fn step(s: *Steps, i: usize) !A {
        var ly = s.b.layers[i % s.b.layers.len];
        if (!s.with_idx) ly.idx = null;
        if (i == 0) s.kv = s.kv0;
        const out: Out = if (s.replaced) try chain(&s.sc, s.b.fx.m, &ly, s.b.li, s.qa, s.kv, s.z, s.pos) else (try launch(&s.sc, s.b.fx.m, s.kern, &ly, s.b.li, s.qa, s.kv, s.z, s.pos, s.g)).?;
        var deps: [5]A = .{ out.q, out.sa.pre, out.sa.post, out.sa.comb, none };
        const n: usize = if (out.q_idx) |qi| blk: {
            deps[4] = qi;
            break :blk 5;
        } else 4;
        s.kv = try after(&s.sc, out.kv_row, deps[0..n]);
        return s.kv;
    }
};

/// The elements of `a` and `b` (any dtype, read as f32) that differ.
fn differing(a: A, b: A) !usize {
    const x = try dsv41.hostF32(a);
    defer testing.allocator.free(x);
    const y = try dsv41.hostF32(b);
    defer testing.allocator.free(y);
    try testing.expectEqual(x.len, y.len);
    var n: usize = 0;
    for (x, y) |u, v| n += @intFromBool(@as(u32, @bitCast(u)) != @as(u32, @bitCast(v)));
    return n;
}

fn scalar(sc: *dsv41.Scope, v: i32) !A {
    return sc.keep(mlx.mlx_array_new_int(v));
}

fn f32Array(sc: *dsv41.Scope, host: []const f32, shape: []const c_int, dt: mlx.mlx_dtype) !A {
    const f = try sc.keep(mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32));
    if (dt == .float32) return f;
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_astype(&o, f, dt, sc.s), &o);
}

/// A linear of `n` rows whose e4m3 codes hold small integers (scales 1).
fn integerLin(sc: *dsv41.Scope, rng: std.Random, n: usize) !dsv41.Lin {
    const codes = [_]u8{ 0x00, 0x38, 0xB8, 0x40, 0xC0, 0x44, 0xC4, 0x48, 0xC8 };
    const w = try testing.allocator.alloc(u8, n * K);
    defer testing.allocator.free(w);
    for (w) |*c| c.* = codes[rng.uintLessThan(usize, codes.len)];
    const s = try testing.allocator.alloc(u8, n * K / 32);
    defer testing.allocator.free(s);
    @memset(s, 127);
    return .{
        .w = try sc.keep(mlx.mlx_array_new_data(w.ptr, &[_]c_int{ ci(n), K / 4 }, 2, .uint32)),
        .s = try sc.keep(mlx.mlx_array_new_data(s.ptr, &[_]c_int{ ci(n), K / 32 }, 2, .uint8)),
        .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 },
    };
}

/// Threadgroup counts the tests launch: the machine's (0), and few enough that each walks many units (7 is the least
/// that holds 6 window rows and the split).
const test_groups = [_]usize{ 0, 7, 19 };

test "dsv41 dec wq_b: bit-equal to the chain where every sum is exact (small integers), 1..6 rows, with and without the indexer" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const fx = try fixture.Fixture.init(testing.allocator, .{ .lo = 20, .hi = 21, .experts = 2, .ctx = 0 });
    defer fx.deinit();
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rng = prng.random();
    var ly = fx.layer(20).*;
    ly.wq_b = try integerLin(&sc, rng, 32768);
    ly.idx.?.wq_b = try integerLin(&sc, rng, 4096);
    var qn: [K]f32 = undefined;
    for (&qn) |*v| v.* = ([_]f32{ 1, 2, 0.5, -1, 3 })[rng.uintLessThan(usize, 5)];
    ly.q_norm = try f32Array(&sc, &qn, &.{K}, .bfloat16);
    var kn: [KV]f32 = undefined;
    for (&kn) |*v| v.* = 0.5 + rng.float(f32);
    ly.kv_norm = try f32Array(&sc, &kn, &.{KV}, .bfloat16);
    for ([_]bool{ true, false }) |with_idx| {
        var l = ly;
        if (!with_idx) l.idx = null;
        for (1..decode.max_rows + 1) |rows| {
            // |qa| = 1: the rms is 1 and the normed rows are the weights' own integers.
            var qa_h: [decode.max_rows * K]f32 = undefined;
            for (qa_h[0 .. rows * K]) |*v| v.* = if (rng.boolean()) 1 else -1;
            const qa = try f32Array(&sc, qa_h[0 .. rows * K], &.{ ci(rows), K }, .bfloat16);
            const kv = try randn(&sc, 21 + rows, &.{ ci(rows), KV }, .bfloat16, 2);
            const z = try randn(&sc, 31 + rows, &.{ ci(rows), 24 }, .float32, 1.5);
            const pos = try scalar(&sc, @intCast(3001 + 17 * rows));
            const want = try chain(&sc, fx.m, &l, 20, qa, kv, z, pos);
            for (test_groups) |g| {
                const got = (try launch(&sc, fx.m, kern, &l, 20, qa, kv, z, pos, if (g == 0) groups() else g)) orelse return error.Declined;
                try testing.expectEqual(@as(usize, 0), try differing(got.q, want.q));
                if (with_idx) try testing.expectEqual(@as(usize, 0), try differing(got.q_idx.?, want.q_idx.?));
                try testing.expectEqual(@as(usize, 0), try differing(got.kv_row, want.kv_row));
                try testing.expectEqual(@as(usize, 0), try differing(got.sa.pre, want.sa.pre));
                try testing.expectEqual(@as(usize, 0), try differing(got.sa.post, want.sa.post));
                try testing.expectEqual(@as(usize, 0), try differing(got.sa.comb, want.sa.comb));
            }
        }
    }
}

fn f32Of(sc: *dsv41.Scope, a: A) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_astype(&o, a, .float32, sc.s), &o);
}

/// `x [rows, K]` against the dequantized rows of `l`, in f32.
fn denseDot(sc: *dsv41.Scope, x: A, l: dsv41.Lin) !A {
    var w = mlx.mlx_array_new();
    const wd = try sc.res(mlx.mlx_dequantize(&w, l.w, l.s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, sc.s), &w);
    var t = mlx.mlx_array_new();
    const wt = try sc.res(mlx.mlx_transpose(&t, wd, sc.s), &t);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_matmul(&o, x, wt, sc.s), &o);
}

/// f32 over the dequantized weights with no intermediate rounding: the values the pipeline approximates.
fn truth(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, ly: *const dsv41.Layer, li: usize, qa: A, pos: A) !struct { q: A, q_idx: ?A } {
    const c = &m.cfg;
    const rows = ci(dim(qa, 0));
    var o = mlx.mlx_array_new();
    const qr = try sc.res(mlx.mlx_fast_rms_norm(&o, try f32Of(sc, qa), try f32Of(sc, ly.q_norm), c.rms_norm_eps, sc.s), &o);
    const q = try dsv41.ropeTail(sc, m, try sc.reshape(try denseDot(sc, qr, ly.wq_b), &.{ rows, ci(c.num_attention_heads), ci(c.head_dim) }), dsv41.ropeFor(m, li).fwd, pos, 1.0);
    const q_idx = if (ly.idx) |iw|
        try dsv41.ropeQat(sc, m, try sc.reshape(try denseDot(sc, qr, iw.wq_b), &.{ rows, ci(c.dsv4_index_n_heads), ci(c.dsv4_index_head_dim) }), m.rope_yarn.fwd, pos, 1.0, .fp4_ue8m0, .float32)
    else
        null;
    return .{ .q = q, .q_idx = q_idx };
}

fn noWorse(label: []const u8, got: A, ch: A, tr: A) !void {
    const g = try dsv41.hostF32(got);
    defer testing.allocator.free(g);
    const w = try dsv41.hostF32(ch);
    defer testing.allocator.free(w);
    const t = try dsv41.hostF32(tr);
    defer testing.allocator.free(t);
    try decode.expectNoWorse(label, g, w, t);
}

test "dsv41 dec wq_b: no further from f32 truth than the chain on the fixture's layers, 1, 2, 4 and 6 rows; kv row and split equal" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    for ([_][2]usize{ .{ 20, 25 }, .{ 2, 4 } }) |range| {
        const fx = try fixture.Fixture.init(testing.allocator, .{ .lo = range[0], .hi = range[1], .experts = 2, .ctx = 0 });
        defer fx.deinit();
        for (range[0]..range[1]) |li| {
            if (li == 22 or li == 23) continue;
            const ly = fx.layer(li);
            for ([_]usize{ 1, 2, 4, 6 }) |rows| {
                var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
                defer sc.deinit();
                const qa = try randn(&sc, 41 + rows, &.{ ci(rows), K }, .bfloat16, 2);
                const kv = try randn(&sc, 51 + rows, &.{ ci(rows), KV }, .bfloat16, 2);
                const z = try randn(&sc, 61 + rows, &.{ ci(rows), 24 }, .float32, 1.5);
                const pos = try scalar(&sc, @intCast(3001 + 13 * rows));
                const want = try chain(&sc, fx.m, ly, li, qa, kv, z, pos);
                const tr = try truth(&sc, fx.m, ly, li, qa, pos);
                const got = (try run(&sc, fx.m, kern, ly, li, qa, kv, z, pos)) orelse return error.Declined;
                var label_buf: [64]u8 = undefined;
                const nq = try differing(got.q, want.q);
                std.debug.print("[wq_b] layer {d}, rows {d}: q differs in {d} of {d}", .{ li, rows, nq, rows * 32768 });
                try noWorse(try std.fmt.bufPrint(&label_buf, "layer {d} rows {d} q", .{ li, rows }), got.q, want.q, tr.q);
                if (ly.idx != null) {
                    const ni = try differing(got.q_idx.?, want.q_idx.?);
                    std.debug.print("; q_idx differs in {d} of {d}\n", .{ ni, rows * 4096 });
                    try noWorse(try std.fmt.bufPrint(&label_buf, "layer {d} rows {d} q_idx", .{ li, rows }), got.q_idx.?, want.q_idx.?, tr.q_idx.?);
                    try testing.expect(ni <= rows * 4096 / 500);
                } else try testing.expect(got.q_idx == null);
                try testing.expect(nq <= rows * 32768 / 500);
                try testing.expectEqual(@as(usize, 0), try differing(got.kv_row, want.kv_row));
                try testing.expectEqual(@as(usize, 0), try differing(got.sa.pre, want.sa.pre) + try differing(got.sa.post, want.sa.post) + try differing(got.sa.comb, want.sa.comb));
            }
        }
    }
}

test "dsv41 dec wq_b: the declared launch is the wrapper's dispatch, with and without the indexer; it registers" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    try testing.expectEqual(@as(usize, 1), set.kernels.items.len);
    const fx = try fixture.Fixture.init(testing.allocator, .{ .lo = 20, .hi = 21, .experts = 2, .ctx = 0 });
    defer fx.deinit();
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    var rig = metal.Rig.init(testing.allocator);
    defer rig.deinit();
    const eps = try dsv41.hcEps(&sc, fx.m);
    for ([_]bool{ true, false }) |with_idx| for ([_]usize{ 1, 6 }) |rows| {
        var ly = fx.layer(20).*;
        if (!with_idx) ly.idx = null;
        const qa = try randn(&sc, 71 + rows, &.{ ci(rows), K }, .bfloat16, 2);
        const kv = try randn(&sc, 81 + rows, &.{ ci(rows), KV }, .bfloat16, 2);
        const z = try randn(&sc, 91 + rows, &.{ ci(rows), 24 }, .float32, 1.5);
        const pos = try scalar(&sc, 3001);
        const got = (try launch(&sc, fx.m, kern, &ly, 20, qa, kv, z, pos, 9)) orelse return error.Declined;
        const iw = if (ly.idx) |i| i.wq_b else ly.wq_b;
        const args = [_]A{ qa, ly.q_norm, ly.wq_b.w, ly.wq_b.s, iw.w, iw.s, kv, ly.kv_norm, dsv41.ropeFor(fx.m, 20).fwd, fx.m.rope_yarn.fwd, pos, z, eps };
        const shape = Shape{ .rows = rows, .n = 32768, .k = if (with_idx) 4096 else 0 };
        try rig.launch(&wq_b_l, Wqb.geometry(.{ .g = 9 }, shape).?, &args);
        const declared = try testing.allocator.dupe(u8, rig.outputBytes());
        defer testing.allocator.free(declared);
        var outs: [6]A = undefined;
        var n: usize = 0;
        for ([_]?A{ got.q, got.q_idx, got.kv_row, got.sa.pre, got.sa.post, got.sa.comb }) |o| if (o) |a| {
            outs[n] = a;
            n += 1;
        };
        try testing.expectEqualSlices(u8, try rig.bytesOf(outs[0..n]), declared);
    };
}

test "dsv41 dec wq_b: gates 1-4 of the kernel contract hold for the registration" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(testing.allocator, set.kernels.items, 0xc0de, &t);
    try testing.expectEqual(@as(usize, 0), t.problems());
}

fn chainTime(b: *Bench, kern: mlx.mlx_fast_metal_kernel, g: usize, rows: usize, with_idx: bool, replaced: bool) !f64 {
    var s = try Steps.init(b, kern, g, rows, with_idx, replaced);
    defer s.deinit();
    return decode.chainUs(&s, b.layers.len);
}

fn nowUs() i64 {
    const iou = @import("../../../sdk/util/io_util.zig");
    return iou.nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

/// What a step costs the host to build, and the device and its encoding to run once built (microseconds a step).
fn splitTime(b: *Bench, kern: mlx.mlx_fast_metal_kernel, g: usize, rows: usize, with_idx: bool, replaced: bool) ![2]f64 {
    var best: [2]f64 = .{ 1e9, 1e9 };
    for (0..5) |_| {
        var s = try Steps.init(b, kern, g, rows, with_idx, replaced);
        defer s.deinit();
        const t0 = nowUs();
        var last: A = none;
        for (0..b.layers.len) |i| last = try s.step(i);
        const t1 = nowUs();
        try mlx.check(mlx.mlx_array_eval(last));
        const t2 = nowUs();
        const n: f64 = @floatFromInt(b.layers.len);
        best[0] = @min(best[0], @as(f64, @floatFromInt(t1 - t0)) / n);
        best[1] = @min(best[1], @as(f64, @floatFromInt(t2 - t1)) / n);
    }
    return best;
}

test "dsv41 dec wq_b: ubench" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const b = try Bench.init(49);
    defer b.deinit();
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    // The GPU's clocks settle under load.
    for (0..4) |_| _ = try chainTime(b, kern, groups(), 1, true, false);
    for ([_]bool{ false, true }) |with_idx| for ([_]usize{ 1, 2, 4, 6 }) |rows| {
        const bytes: f64 = if (with_idx) 48_660_480 else 43_253_760;
        const us = try chainTime(b, kern, groups(), rows, with_idx, false);
        const ch = try chainTime(b, kern, groups(), rows, with_idx, true);
        decode.report(if (with_idx) "wq_b+idx" else "wq_b", rows, us, bytes, ch);
        const sp = try splitTime(b, kern, groups(), rows, with_idx, false);
        const sc_ = try splitTime(b, kern, groups(), rows, with_idx, true);
        std.debug.print("[dec-ubench-eval] {s} R={d}: host build {d:.1} us of the above, device alone {d:.1} us ({d:.0} GB/s); replaced chain {d:.1} + {d:.1}\n", .{ if (with_idx) "wq_b+idx" else "wq_b", rows, sp[0], sp[1], bytes / sp[1] / 1e3, sc_[0], sc_[1] });
    };
}
