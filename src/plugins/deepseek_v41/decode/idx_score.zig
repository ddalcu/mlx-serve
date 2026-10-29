//! k3 `idx_score` (`~/plans/dsv41-decode-kernels-spec.md`): the compressor's pool step at a KV source, and the index
//! scores of 1..6 decode rows, gathered to the candidate blocks past the candidate source.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const nax = @import("../../../mlx/device.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const c41 = @import("../contract.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const fixture = @import("fixture.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dim(x: A, axis: usize) usize {
    return @intCast(mlx.getShape(x)[axis]);
}

// ── the kernels ──────────────────────────────────────────────────────────────

/// Scores `[R, nb]` (or `[R, 8 cand slots]`): sum over the 32 heads of weight * relu(q_h . key_n), `-inf` past the
/// keys a row sees ((p0 + r + 1) / ratio) and for padded candidate blocks. 16 keys a NAX op.
pub const SCORE_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_idx_score.metal");
const MPP_HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");
pub const SCORE_HEADER = "#define GATHER 0\n" ++ MPP_HEADER;
pub const GATHER_HEADER = "#define GATHER 1\n" ++ MPP_HEADER;
pub const SCORE_INS = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };
pub const GATHER_INS = SCORE_INS ++ [_][*:0]const u8{"cand"};

/// The pool step: the groups a step completes at a KV source (pooled at ratio 2), their comp rows and, as this
/// source owns index keys, their index keys; ratio 2 also hands back the open group's row.
pub const POOL_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_idx_score_pool.metal");
pub const POOL2_HEADER = "#define POOL 1\n#define RATIO 2\n" ++ dsv41.FP8_HEADER;
pub const POOL1_HEADER = "#define POOL 0\n#define RATIO 1\n" ++ dsv41.FP8_HEADER;
pub const POOL2_INS = [_][*:0]const u8{ "rows", "pend_kv", "pend_gate", "norm", "wk", "knorm", "periods", "meta", "eps" };
pub const POOL1_INS = [_][*:0]const u8{ "rows", "norm", "wk", "knorm", "periods", "meta", "eps" };
pub const POOL2_OUTS = [_][*:0]const u8{ "comp", "idx", "pend_kv_o", "pend_gate_o" };
pub const POOL1_OUTS = [_][*:0]const u8{ "comp", "idx" };

pub const Kernels = struct {
    score: mlx.mlx_fast_metal_kernel,
    gather: mlx.mlx_fast_metal_kernel,
    pool2: mlx.mlx_fast_metal_kernel,
    pool1: mlx.mlx_fast_metal_kernel,

    pub fn deinit(k: *Kernels) void {
        for ([_]mlx.mlx_fast_metal_kernel{ k.score, k.gather, k.pool2, k.pool1 }) |x| _ = mlx.mlx_fast_metal_kernel_free(x);
    }
};

pub fn build() !Kernels {
    const b = dsv41.buildKernel;
    var k: Kernels = undefined;
    k.score = try b("dsv41_dec_idx_score", &SCORE_INS, &.{"scores"}, SCORE_SOURCE, SCORE_HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(k.score);
    k.gather = try b("dsv41_dec_idx_gather", &GATHER_INS, &.{"scores"}, SCORE_SOURCE, GATHER_HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(k.gather);
    k.pool2 = try b("dsv41_dec_idx_pool2", &POOL2_INS, &POOL2_OUTS, POOL_SOURCE, POOL2_HEADER);
    errdefer _ = mlx.mlx_fast_metal_kernel_free(k.pool2);
    k.pool1 = try b("dsv41_dec_idx_pool1", &POOL1_INS, &POOL1_OUTS, POOL_SOURCE, POOL1_HEADER);
    return k;
}

// ── the scores ───────────────────────────────────────────────────────────────

/// The heads and channels the indexer's scores run over (`indexer`: 1 / sqrt(channels) / sqrt(heads)).
const heads = 32;
const channels = 128;
pub const wscale: f32 = 1.0 / @sqrt(@as(f32, channels)) / @sqrt(@as(f32, heads));

/// Simdgroups a threadgroup; a row's tiles are cut into at most 160 ranges of threadgroups (320 beside other rows).
const sg_per_tg = 4;

fn ranges(rows: usize, ncols: usize) usize {
    const tiles = (ncols + 15) / 16;
    const cap: usize = if (rows == 1) 160 else 320;
    return @min(cap, (tiles + sg_per_tg - 1) / sg_per_tg);
}

/// Index scores of the rows `q` `[R, 32, 128]` (f32, or bf16: the fp4 round trip's output, exact in bf16) with
/// weights `w` `[R, 32]` bf16 at positions p0..p0 + R - 1, against `keys` `[nb, 128]` bf16 (this step's new keys
/// included): `[R, nb]` f32. With `cand` `[R, n]` i32 (n even; each row's picked block ids ascending, -1 padded) the
/// result is gathered: `[R, 8 n]`, slot s the key `cand[r][s / 8] * 8 + s % 8`. Null: declined (not NAX, QAT off,
/// shapes or dtypes outside the kernel's).
pub fn scores(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, k: *const Kernels, q: A, w: A, keys: A, p0: usize, ratio: usize, cand: ?A) !?A {
    if (!mlx.streamIsGpu(sc.s) or !nax.naxAvailable() or m.qat_off) return null;
    const rows = dim(q, 0);
    if (rows == 0 or rows > decode.max_rows or ratio == 0) return null;
    const qdt = mlx.mlx_array_dtype(q);
    if ((qdt != .float32 and qdt != .bfloat16) or mlx.mlx_array_dtype(keys) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return null;
    if (mlx.getShape(q).len != 3 or dim(q, 1) != heads or dim(q, 2) != channels) return null;
    if (mlx.getShape(keys).len != 2 or dim(keys, 1) != channels or dim(keys, 0) == 0) return null;
    if (mlx.getShape(w).len != 2 or dim(w, 0) != rows or dim(w, 1) != heads) return null;
    const nb = dim(keys, 0);
    var ncols = nb;
    if (cand) |c| {
        if (mlx.mlx_array_dtype(c) != .int32 or mlx.getShape(c).len != 2 or dim(c, 0) != rows or dim(c, 1) % 2 != 0) return null;
        ncols = dim(c, 1) * 8;
    }
    const meta = try sc.ints(&.{ @intCast(p0), @intCast(ratio) }, &.{2});
    const wsc = try sc.keep(mlx.mlx_array_new_data(&[_]f32{wscale}, &[_]c_int{1}, 1, .float32));
    const inputs = [_]A{ q, keys, w, meta, wsc, if (cand) |c| c else none };
    return try launchScores(sc, if (cand != null) k.gather else k.score, inputs[0..if (cand != null) 6 else 5], rows, ncols);
}

/// One launch of a scores kernel over `inputs` (its inputs in order, then any it ignores): a threadgroup a (row, range
/// of tiles), the rows of a range adjacent.
fn launchScores(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, inputs: []const A, rows: usize, ncols: usize) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(ncols) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(ranges(rows, ncols) * rows * sg_per_tg * 32), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, sg_per_tg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "QT", mlx.mlx_array_dtype(inputs[0])));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

// ── the pool step ────────────────────────────────────────────────────────────

/// The groups a step completed: comp rows `[G, 512]` and index keys `[G, 128]` (bf16, QAT round trips applied), none
/// when G is 0; ratio 2 also the open group's kv and gate rows `[1, 512]` f32, none when the step closed it.
pub const Pooled = struct { comp: A, idx_k: A, pend_kv: A, pend_gate: A };

/// What a KV source's pool step runs: its kernel's inputs in order, the groups the step completes, whether the step
/// leaves the open group a row.
const PoolPlan = struct { pooling: bool, groups: usize, leftover: bool, ins: [9]A, n_ins: usize };

fn poolPlan(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, li: usize, rows: A, pend_kv: A, pend_gate: A, p0: usize) !?PoolPlan {
    if (!mlx.streamIsGpu(sc.s) or m.qat_off) return null;
    const ly = &m.layers[li];
    const cw = ly.comp orelse return null;
    const iw = ly.idx orelse return null;
    const wk = iw.wk orelse return null;
    const ratio: usize = ly.ratio;
    const r = dim(rows, 0);
    if (r == 0 or r > decode.max_rows or (ratio != 1 and ratio != 2) or m.cfg.head_dim != 512 or m.cfg.dsv4_index_head_dim != channels) return null;
    if (mlx.mlx_array_dtype(wk.w) != .bfloat16 or wk.s.ctx != null or mlx.mlx_array_dtype(cw.norm) != .bfloat16 or mlx.mlx_array_dtype(iw.k_norm) != .bfloat16) return null;
    const pooling = ratio == 2;
    if (dim(rows, 1) != (if (pooling) @as(usize, 1024) else 512) or mlx.mlx_array_dtype(rows) != (if (pooling) mlx.mlx_dtype.float32 else .bfloat16)) return null;
    const np: usize = if (pooling) p0 % 2 else 0;
    if (pooling and (np == 0) != (pend_kv.ctx == null)) return null;
    const total = np + r;
    var plan: PoolPlan = .{ .pooling = pooling, .groups = if (pooling) total / 2 else r, .leftover = pooling and total % 2 == 1, .ins = undefined, .n_ins = 0 };
    if (plan.groups == 0) return plan;
    const meta = try sc.ints(&.{ ci(np), ci(p0 / ratio), ci(r) }, &.{3});
    const eps = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.rms_norm_eps}, &[_]c_int{1}, 1, .float32));
    const kv = if (pend_kv.ctx != null) pend_kv else rows;
    const gate = if (pend_gate.ctx != null) pend_gate else rows;
    const ins2 = [_]A{ rows, kv, gate, cw.norm, wk.w, iw.k_norm, m.rope_yarn.fwd, meta, eps };
    const ins1 = [_]A{ rows, cw.norm, wk.w, iw.k_norm, m.rope_yarn.fwd, meta, eps };
    plan.n_ins = if (pooling) ins2.len else ins1.len;
    if (pooling) plan.ins = ins2 else @memcpy(plan.ins[0..ins1.len], &ins1);
    return plan;
}

/// KV source `li`'s pool step for a step of R rows from position p0: `rows` the compressor projection of the step
/// (ratio 2: `[R, 1024]` f32, kv then gate; ratio 1: `[R, 512]` bf16), `pend_kv` / `pend_gate` the open group's
/// rows so far (`[p0 % 2, 512]` f32, none when empty). Arrays belong to `sc`. Null: declined.
pub fn pool(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, k: *const Kernels, li: usize, rows: A, pend_kv: A, pend_gate: A, p0: usize) !?Pooled {
    const plan = (try poolPlan(sc, m, li, rows, pend_kv, pend_gate, p0)) orelse return null;
    if (plan.groups == 0) {
        const last = dim(rows, 0) - 1;
        return .{ .comp = none, .idx_k = none, .pend_kv = try sliceRow(sc, rows, last, 0), .pend_gate = try sliceRow(sc, rows, last, 512) };
    }
    return try launchPool(sc, if (plan.pooling) k.pool2 else k.pool1, plan.ins[0..plan.n_ins], plan.groups, plan.pooling, plan.leftover);
}

/// One launch of a pool kernel over `inputs` (its inputs in order, then any it ignores): a threadgroup a group, and
/// for the open group's row one more.
fn launchPool(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, inputs: []const A, groups: usize, pooling: bool, leftover: bool) !Pooled {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(groups), 512 }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(groups), channels }, 2, .bfloat16));
    if (pooling) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 512 }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 512 }, 2, .float32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((groups + @intFromBool(leftover)) * 128), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var out: Pooled = .{ .comp = none, .idx_k = none, .pend_kv = none, .pend_gate = none };
    inline for (.{ "comp", "idx_k", "pend_kv", "pend_gate" }, 0..) |name, i| {
        if (i < 2 or leftover) {
            var o = mlx.mlx_array_new();
            @field(out, name) = try sc.res(mlx.mlx_vector_array_get(&o, ov, i), &o);
        }
    }
    return out;
}

/// Row `r` of `x`, columns lo..lo + 512.
fn sliceRow(sc: *dsv41.Scope, x: A, r: usize, lo: usize) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_slice(&o, x, &[_]c_int{ ci(r), ci(lo) }, 2, &[_]c_int{ ci(r + 1), ci(lo + 512) }, 2, &[_]c_int{ 1, 1 }, 2, sc.s), &o);
}

// ── the kernel contract ──────────────────────────────────────────────────────

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

const score_l = metal.Launch{ .name = "dsv41_dec_idx_score", .inputs = &SCORE_INS, .outputs = &.{"scores"}, .source = SCORE_SOURCE, .header = SCORE_HEADER, .tg_bytes = 0, .offline = false, .min_gen = 17 };
const gather_l = metal.Launch{ .name = "dsv41_dec_idx_gather", .inputs = &GATHER_INS, .outputs = &.{"scores"}, .source = SCORE_SOURCE, .header = GATHER_HEADER, .tg_bytes = 0, .offline = false, .min_gen = 17 };
const pool2_l = metal.Launch{ .name = "dsv41_dec_idx_pool2", .inputs = &POOL2_INS, .outputs = &POOL2_OUTS, .source = POOL_SOURCE, .header = POOL2_HEADER, .tg_bytes = pool_tg_bytes, .m1 = 1024, .m2 = 1024 };
const pool1_l = metal.Launch{ .name = "dsv41_dec_idx_pool1", .inputs = &POOL1_INS, .outputs = &POOL1_OUTS, .source = POOL_SOURCE, .header = POOL1_HEADER, .tg_bytes = pool_tg_bytes, .m1 = 1024, .m2 = 1024 };

/// The scores of R rows over n keys, or over the keys of `nblk` candidate blocks a row (`gather`): today's indexer
/// formula, so its fill and float64 reference serve (keys and queries as bf16, the fp4 round trip's output).
const ScoreCase = struct {
    pub const Params = struct { ih: usize = heads, gather: bool = false };
    const nblk = 64;

    fn cols(p: Params, s: Shape) usize {
        return if (p.gather) nblk * 8 else s.n;
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > decode.max_rows or s.n == 0 or s.k != channels) return null;
        var g = metal.Geometry{ .grid = .{ ci(ranges(s.rows, cols(p, s)) * s.rows * sg_per_tg * 32), 1, 1 }, .tg = .{ sg_per_tg * 32, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(cols(p, s)) }, .float32);
        g.dtype("QT", .bfloat16);
        return g;
    }

    /// Gathered: each row's candidate blocks, an ascending sample of the key blocks padded with -1.
    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        try c41.idxFill(k, f, rng, .bfloat16, .bfloat16);
        if (!k.p.gather) return;
        const cand = try k.add(.int32, &.{ k.shape.rows, nblk });
        const blocks = (k.shape.n + 7) / 8;
        const want = @min(nblk - 3, blocks);
        for (0..k.shape.rows) |r| {
            var n: usize = 0;
            var b: usize = 0;
            while (b < blocks and n < want) : (b += 1) if (rng.uintLessThan(usize, blocks - b) < want - n) {
                cand[r * nblk + n] = @floatFromInt(b);
                n += 1;
            };
            @memset(cand[r * nblk + n ..][0 .. nblk - n], -1);
        }
    }

    pub fn count(p: Params, s: Shape) usize {
        return s.rows * cols(p, s);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        if (!k.p.gather) return c41.idxCompute(k, out, mag, sum);
        const L = k.shape.rows;
        const nb = k.shape.n;
        const all = try k.scratch().alloc(f64, 2 * L * nb);
        defer k.scratch().free(all);
        @memset(all[L * nb ..], 0);
        try c41.idxCompute(k, all[0 .. L * nb], all[L * nb ..], sum);
        for (0..L) |l| for (0..nblk * 8) |s| {
            const b = k.in(5)[l * nblk + s / 8];
            const at = if (b < 0) nb else @as(usize, @intFromFloat(b)) * 8 + s % 8;
            out[l * nblk * 8 + s] = if (at < nb) all[l * nb + at] else -kit.inf;
            mag[l * nblk * 8 + s] = if (at < nb) all[L * nb + l * nb + at] else 0;
        };
    }
};

fn depthScore(s: Shape) usize {
    return s.k + heads;
}

/// Threadgroup bytes of the pool kernel: the latent row (bf16), the key row, the simdgroup sums, the rms scale.
const pool_tg_bytes = 64 * 16 + 128 * 2 + 32 * 4 + 4;

/// A KV source's pool step as the shipped chain computes it. Shape: rows = the step's R, n = the open group's rows
/// (0 or 1; ratio 2 only), k = 512. Outputs: comp rows, index keys, then (ratio 2, an odd total) the open row.
const PoolCase = struct {
    pub const Params = struct { pooling: bool };
    const g0 = 1500;
    const eps: f64 = 1e-6;

    fn groups(p: Params, s: Shape) usize {
        return if (p.pooling) (s.n + s.rows) / 2 else s.rows;
    }

    fn leftover(p: Params, s: Shape) bool {
        return p.pooling and (s.n + s.rows) % 2 == 1;
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.rows > decode.max_rows or s.k != 512 or s.n > @intFromBool(p.pooling) or groups(p, s) == 0) return null;
        const G = groups(p, s);
        var g = metal.Geometry{ .grid = .{ ci((G + @intFromBool(leftover(p, s))) * 128), 1, 1 }, .tg = .{ 128, 1, 1 } };
        g.out(&.{ ci(G), 512 }, .bfloat16);
        g.out(&.{ ci(G), channels }, .bfloat16);
        if (p.pooling) {
            g.out(&.{ 1, 512 }, .float32);
            g.out(&.{ 1, 512 }, .float32);
            if (!leftover(p, s)) g.unread = 0b1100;
        }
        return g;
    }

    /// wk is sparse (8 entries of +-1 a row) so its f32 sums are exact in any order and the chain's bytes can be
    /// compared; its dense accuracy is the equivalence test's.
    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const R = k.shape.rows;
        const np = k.shape.n;
        const w: usize = if (k.p.pooling) 1024 else 512;
        const rows = try k.add(if (k.p.pooling) .float32 else .bfloat16, &.{ R, w });
        kit.uniform(rng, rows, -3, 3);
        if (f == .nonfinite) kit.plantRows(rows, R, w);
        if (k.p.pooling) for (0..2) |_| kit.uniform(rng, try k.add(.float32, &.{ @max(np, 1), 512 }), -3, 3);
        kit.uniform(rng, try k.add(.bfloat16, &.{512}), 0.5, 1.5);
        const wk = try k.add(.bfloat16, &.{ channels, 512 });
        for (0..channels) |o| for (0..8) |_| {
            wk[o * 512 + rng.uintLessThan(usize, 512)] = if (rng.boolean()) 1 else -1;
        };
        kit.uniform(rng, try k.add(.bfloat16, &.{channels}), 0.5, 1.5);
        const periods = try k.add(.float32, &.{32});
        for (periods, 0..) |*v, i| v.* = std.math.pow(f64, 10000, @as(f64, @floatFromInt(i)) / 32);
        const meta = try k.add(.int32, &.{3});
        meta[0] = @floatFromInt(np);
        meta[1] = g0;
        meta[2] = @floatFromInt(R);
        (try k.add(.float32, &.{1}))[0] = eps;
    }

    pub fn count(p: Params, s: Shape) usize {
        return groups(p, s) * (512 + channels) + if (leftover(p, s)) @as(usize, 1024) else 0;
    }

    /// Float64's classes only: a group with a non-finite pooled value is NaN throughout (its rms is), else finite.
    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const G = groups(k.p, k.shape);
        const np = k.shape.n;
        const rows = k.in(0);
        @memset(mag, 0);
        for (0..G) |g| {
            var bad = false;
            for (0..512) |i| {
                if (!k.p.pooling) {
                    bad = bad or !std.math.isFinite(rows[g * 512 + i]);
                    continue;
                }
                var kv: [2]f64 = undefined;
                var ga: [2]f64 = undefined;
                for (0..2) |j| {
                    const c = 2 * g + j;
                    kv[j] = if (c < np) k.in(1)[c * 512 + i] else rows[(c - np) * 1024 + i];
                    ga[j] = if (c < np) k.in(2)[c * 512 + i] else rows[(c - np) * 1024 + 512 + i];
                }
                const m = kit.maxN(ga[0], ga[1]);
                const e0 = @exp(ga[0] - m);
                const e1 = @exp(ga[1] - m);
                bad = bad or !std.math.isFinite(kv[0] * (e0 / (e0 + e1)) + kv[1] * (e1 / (e0 + e1)));
            }
            @memset(out[g * 512 ..][0..512], if (bad) kit.nan else 0);
            @memset(out[G * 512 + g * channels ..][0..channels], if (bad) kit.nan else 0);
        }
        if (leftover(k.p, k.shape)) {
            const last = (k.shape.rows - 1) * 1024;
            @memcpy(out[G * (512 + channels) ..][0..1024], rows[last..][0..1024]);
        }
    }

    /// The shipped chain on the same inputs: pool in f32, rms, then wk, rms, and the two rope + QAT kernels.
    pub fn replaced(k: anytype) ![]const u8 {
        var sc = dsv41.Scope.init(k.scratch(), k.rig.stream());
        defer sc.deinit();
        const np = k.shape.n;
        const G = groups(k.p, k.shape);
        const o: usize = if (k.p.pooling) 2 else 0;
        const norm, const wk, const knorm, const periods = .{ k.args[1 + o], k.args[2 + o], k.args[3 + o], k.args[4 + o] };
        const e: f32 = @floatCast(eps);
        const ratio: f64 = if (k.p.pooling) 2 else 1;
        var held: [4]A = undefined;
        var latent = k.args[0];
        if (k.p.pooling) {
            var kv = try slice(&sc, k.args[0], 1, 0, 512);
            var gate = try slice(&sc, k.args[0], 1, 512, 1024);
            if (np == 1) {
                kv = try concat(&sc, k.args[1], kv);
                gate = try concat(&sc, k.args[2], gate);
            }
            const shape = [_]c_int{ ci(G), 2, 512 };
            const kg = try mx(&sc, mlx.mlx_reshape, .{ try slice(&sc, kv, 0, 0, 2 * G), &shape, 3 });
            const gg = try mx(&sc, mlx.mlx_reshape, .{ try slice(&sc, gate, 0, 0, 2 * G), &shape, 3 });
            const pooled = try mx(&sc, mlx.mlx_sum_axis, .{ try mx(&sc, mlx.mlx_multiply, .{ kg, try mx(&sc, mlx.mlx_softmax_axis, .{ gg, 1, true }) }), 1, false });
            latent = try mx(&sc, mlx.mlx_astype, .{ pooled, mlx.mlx_dtype.bfloat16 });
            held[2] = try slice(&sc, kv, 0, 2 * G, 2 * G + 1);
            held[3] = try slice(&sc, gate, 0, 2 * G, 2 * G + 1);
        }
        latent = try mx(&sc, mlx.mlx_fast_rms_norm, .{ latent, norm, e });
        const lin: dsv41.Lin = .{ .w = wk };
        const y = try mx(&sc, mlx.mlx_fast_rms_norm, .{ try lin.apply(&sc, latent), knorm, e });
        var rig = metal.Rig.init(k.scratch());
        defer rig.deinit();
        held[0] = try rope(&sc, &rig, latent, .{ .qk = .fp4_e4m3, .scal = ratio, .off = g0 }, periods);
        held[1] = try rope(&sc, &rig, y, .{ .qk = .fp4, .scal = ratio, .off = g0 }, periods);
        return k.rig.bytesOf(held[0..if (leftover(k.p, k.shape)) @as(usize, 4) else 2]);
    }
};

/// One MLX op on `sc`'s stream: `f(&out, args..., stream)`.
fn mx(sc: *dsv41.Scope, comptime f: anytype, args: anytype) !A {
    var o = mlx.mlx_array_new();
    return sc.res(@call(.auto, f, .{&o} ++ args ++ .{sc.s}), &o);
}

fn concat(sc: *dsv41.Scope, a: A, b: A) !A {
    const v = mlx.mlx_vector_array_new_data(&[_]A{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(v);
    return mx(sc, mlx.mlx_concatenate_axis, .{ v, 0 });
}

/// `x[lo..hi]` along `axis` of a matrix.
fn slice(sc: *dsv41.Scope, x: A, axis: usize, lo: usize, hi: usize) !A {
    var start = [_]c_int{ 0, 0 };
    var stop = [_]c_int{ ci(dim(x, 0)), ci(dim(x, 1)) };
    start[axis] = ci(lo);
    stop[axis] = ci(hi);
    return mx(sc, mlx.mlx_slice, .{ x, &start, 2, &stop, 2, &[_]c_int{ 1, 1 }, 2 });
}

/// The shipped rope + QAT kernel (the contract's launch of it) on the rows of `x`.
fn rope(sc: *dsv41.Scope, rig: *metal.Rig, x: A, p: c41.RopeTail.Params, periods: A) !A {
    const shape: Shape = .{ .rows = dim(x, 0), .n = 1, .k = dim(x, 1) };
    const off = try sc.ints(&.{p.off}, &.{});
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{@floatCast(p.scal)}, &[_]c_int{1}, 1, .float32));
    try rig.launch(&c41.rope_l, c41.RopeTail.geometry(p, shape).?, &.{ x, periods, off, scal });
    return sc.keep(sc.out(rig.output(0)));
}

fn pool2Comp(s: Shape) usize {
    return (s.n + s.rows) / 2 * (512 + channels);
}

pub fn register(set: *kc.Set) !void {
    const tiles: [3]spec.Tile = .{ .{}, .{ .size = 16 }, .{} };
    const scored: []const Shape = &.{ .{ .rows = 6, .n = 600, .k = channels }, .{ .rows = 1, .n = 10304, .k = channels }, .{ .rows = 2, .n = 1030, .k = channels } };
    try kit.add(set, ScoreCase, &score_l, .{}, .{
        .name = "dsv41 dec idx_score (NAX scores, every key)",
        .accum = .f32,
        .depth = depthScore,
        .out = .f32,
        .tiles = tiles,
        .served = scored,
        .nax = true,
        .known = c41.relu_nan,
    });
    try kit.add(set, ScoreCase, &gather_l, .{ .gather = true }, .{
        .name = "dsv41 dec idx_score (NAX scores, candidate blocks)",
        .accum = .f32,
        .depth = depthScore,
        .out = .f32,
        .tiles = tiles,
        .served = &.{ .{ .rows = 6, .n = 600, .k = channels }, .{ .rows = 2, .n = 1030, .k = channels } },
        .nax = true,
        .known = c41.relu_nan,
    });
    const pool_chain = "the compress chain (pool, rms, wk, rms, rope + QAT)";
    try kit.add(set, PoolCase, &pool2_l, .{ .pooling = true }, .{
        .name = "dsv41 dec idx_score pool (ratio 2: pairs of rows, index keys)",
        .accum = .none,
        .depth = c41.depthOne,
        .out = .bf16,
        .outs = &.{ .{ .elem = .bf16, .count = pool2Comp }, .{ .elem = .f32, .count = pool2Comp } },
        .replaces = pool_chain,
        .served = &.{ .{ .rows = 6, .n = 0, .k = 512 }, .{ .rows = 1, .n = 1, .k = 512 }, .{ .rows = 2, .n = 1, .k = 512 }, .{ .rows = 3, .n = 0, .k = 512 }, .{ .rows = 4, .n = 1, .k = 512 }, .{ .rows = 5, .n = 0, .k = 512 } },
        .known = c41.nan_code,
    });
    try kit.add(set, PoolCase, &pool1_l, .{ .pooling = false }, .{
        .name = "dsv41 dec idx_score pool (ratio 1: a group a row, index keys)",
        .accum = .none,
        .depth = c41.depthOne,
        .out = .bf16,
        .replaces = pool_chain,
        .served = &.{ .{ .rows = 6, .n = 0, .k = 512 }, .{ .rows = 1, .n = 0, .k = 512 }, .{ .rows = 3, .n = 0, .k = 512 } },
        .known = c41.nan_code,
    });
}

// ── tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

/// Host copy of `x` as f64 (any float dtype).
fn hostF64(x: A) ![]f64 {
    const f = try dsv41.hostF32(x);
    defer testing.allocator.free(f);
    const out = try testing.allocator.alloc(f64, f.len);
    for (out, f) |*o, v| o.* = v;
    return out;
}

/// Rows [0, n) of the layer's index keys.
fn keyRows(sc: *dsv41.Scope, ls: *const dsv41.LState, n: usize) !A {
    const rows = ls.idx_k.?;
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_slice(&o, rows.buf, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), ci(rows.width) }, 2, &[_]c_int{ 1, 1 }, 2, sc.s), &o);
}

/// The unit roundoff of f32 and the bound on a sum of n terms: gamma_n = n u / (1 - n u).
const u32_: f64 = 0x1p-24;
fn gamma(n: usize) f64 {
    const nu = @as(f64, @floatFromInt(n)) * u32_;
    return nu / (1 - nu);
}

const Truth = struct { val: []f64, mag: []f64 };

/// The scores in float64 from the host copies of q, w and keys (slot `s` of row r reads key `key_of(r, s)` or -inf).
fn truthScores(q: []const f64, w: []const f64, keys: []const f64, rows: usize, nb: usize, p0: usize, ratio: usize, cand: ?[]const i32, ncols: usize) !Truth {
    const val = try testing.allocator.alloc(f64, rows * ncols);
    const mag = try testing.allocator.alloc(f64, rows * ncols);
    const wsc: f64 = wscale;
    for (0..rows) |r| for (0..ncols) |s| {
        const n: ?usize = if (cand) |c| blk: {
            const b = c[r * (ncols / 8) + s / 8];
            break :blk if (b < 0) null else @as(usize, @intCast(b)) * 8 + s % 8;
        } else s;
        const seen = (p0 + r + 1) / ratio;
        if (n == null or n.? >= seen or n.? >= nb) {
            val[r * ncols + s] = -std.math.inf(f64);
            mag[r * ncols + s] = 0;
            continue;
        }
        var acc: f64 = 0;
        var am: f64 = 0;
        for (0..heads) |h| {
            var d: f64 = 0;
            for (0..channels) |j| d += q[(r * heads + h) * channels + j] * keys[n.? * channels + j];
            const wh = @as(f64, @as(f32, @floatCast(w[r * heads + h] * wsc)));
            const t = wh * @max(d, 0);
            acc += t;
            am += @abs(t);
        }
        val[r * ncols + s] = acc;
        mag[r * ncols + s] = am;
    };
    return .{ .val = val, .mag = mag };
}

/// q `[R, 32, 128]` as the indexer makes it (rope + fp4 round trip, f32), w `[R, 32]` bf16.
fn randomQ(fx: *fixture.Fixture, sc: *dsv41.Scope, rows: usize, p0: usize) !struct { q: A, w: A } {
    const x = try fx.normal(&.{ ci(rows), heads, channels }, .float32, 0, 1.5);
    defer _ = mlx.mlx_array_free(x);
    const off = try sc.keep(mlx.mlx_array_new_int(@intCast(p0)));
    const q = try dsv41.ropeQat(sc, fx.m, x, fx.m.rope_yarn.fwd, off, 1.0, .fp4_ue8m0, .float32);
    return .{ .q = q, .w = try sc.keep(try fx.normal(&.{ ci(rows), heads }, .bfloat16, 0, 1)) };
}

/// Our scores (full or gathered) against today's `indexScoresScalar` (masked to the candidates for the gathered) and
/// float64 truth.
fn scoresCase(fx: *fixture.Fixture, k: *const Kernels, li: usize, rows: usize, gather: bool) !void {
    const gpa = testing.allocator;
    const m = fx.m;
    const ls = &fx.state().layers[li];
    const ratio: usize = m.layers[li].ratio;
    const nb = ls.idx_k.?.used;
    const p0 = nb * ratio - rows;
    var sc = dsv41.Scope.init(gpa, m.s);
    defer sc.deinit();
    const inp = try randomQ(fx, &sc, rows, p0);
    const keys = try keyRows(&sc, ls, nb);
    const chain = try dsv41.indexScoresScalar(&sc, m, inp.q, keys, inp.w, wscale, p0, ratio);
    // Candidates: per row an ascending random subset of the blocks, -1 padded.
    const nblk = 64;
    const nblocks = (nb + 7) / 8;
    var cand_h: [decode.max_rows * nblk]i32 = @splat(-1);
    var prng = std.Random.DefaultPrng.init(0x1d5c0 + rows);
    for (0..rows) |r| {
        var picked: [nblk]i32 = undefined;
        var n: usize = 0;
        while (n < nblk - 5) {
            const b: i32 = @intCast(prng.random().uintLessThan(usize, nblocks));
            if (std.mem.indexOfScalar(i32, picked[0..n], b) == null) {
                picked[n] = b;
                n += 1;
            }
        }
        std.mem.sort(i32, picked[0..n], {}, std.sort.asc(i32));
        @memcpy(cand_h[r * nblk ..][0..n], picked[0..n]);
    }
    const cand = if (gather) try sc.ints(cand_h[0 .. rows * nblk], &.{ ci(rows), nblk }) else null;
    const got = (try scores(&sc, m, k, inp.q, inp.w, keys, p0, ratio, cand)) orelse return error.Declined;
    const ncols: usize = if (gather) nblk * 8 else nb;
    const gv = try dsv41.hostF32(got);
    defer gpa.free(gv);
    const cv = try dsv41.hostF32(chain);
    defer gpa.free(cv);
    const qh = try hostF64(inp.q);
    defer gpa.free(qh);
    const wh = try hostF64(inp.w);
    defer gpa.free(wh);
    const kh = try hostF64(keys);
    defer gpa.free(kh);
    const t = try truthScores(qh, wh, kh, rows, nb, p0, ratio, if (gather) cand_h[0 .. rows * nblk] else null, ncols);
    defer gpa.free(t.val);
    defer gpa.free(t.mag);
    // Today's chain at the same slots: its full scores, gathered by the candidates.
    const want = try gpa.alloc(f32, rows * ncols);
    defer gpa.free(want);
    for (0..rows) |r| for (0..ncols) |s| {
        const n: ?usize = if (gather) blk: {
            const b = cand_h[r * nblk + s / 8];
            break :blk if (b < 0) null else @as(usize, @intCast(b)) * 8 + s % 8;
        } else s;
        want[r * ncols + s] = if (n) |nn| (if (nn < nb) cv[r * nb + nn] else -std.math.inf(f32)) else -std.math.inf(f32);
    };
    var finite: usize = 0;
    var worst: f64 = 0;
    var worst_truth: f64 = 0;
    for (gv, want, t.val, t.mag) |g, c, tv, tm| {
        if (std.math.isInf(c) or std.math.isInf(tv)) {
            try testing.expectEqual(c, g);
            continue;
        }
        finite += 1;
        // Both are f32 sums of the same 32 terms: each within gamma_34 of the terms' magnitude from the truth.
        const bound = 2 * gamma(34) * tm + 1e-30;
        worst = @max(worst, @abs(@as(f64, g) - c) / bound);
        worst_truth = @max(worst_truth, @abs(@as(f64, g) - tv) / (gamma(34) * tm + 1e-30));
        try testing.expect(@abs(@as(f64, g) - c) <= bound);
        try testing.expect(@abs(@as(f64, g) - tv) <= gamma(34) * tm + 1e-30);
    }
    std.debug.print("[dec] idx_score {s} layer {d} R={d} nb={d}: {d} finite scores, worst new-vs-chain {d:.2} of its f32 bound, new-vs-truth {d:.2}\n", .{ if (gather) "gathered" else "full", li, rows, nb, finite, worst, worst_truth });
    try testing.expect(finite > 0);
    const tf = try gpa.alloc(f32, finite);
    defer gpa.free(tf);
    const nf = try gpa.alloc(f32, finite);
    defer gpa.free(nf);
    const cf = try gpa.alloc(f32, finite);
    defer gpa.free(cf);
    var i: usize = 0;
    for (gv, want, t.val) |g, c, tv| if (!std.math.isInf(c)) {
        tf[i] = @floatCast(tv);
        nf[i] = g;
        cf[i] = c;
        i += 1;
    };
    try decode.expectNoWorse("idx_score", nf, cf, tf);
    // The top half of each row's finite scores: picks that differ from today's are near-ties.
    const diff = try pickDiffs(gpa, gv, want, t.mag, rows, ncols);
    std.debug.print("[dec] idx_score: top-half picks of {d} rows: {d} differ from today's, {d} not near-ties\n", .{ rows, diff[0], diff[1] });
    try testing.expectEqual(@as(usize, 0), diff[1]);
}

const ByScore = struct {
    s: []const f32,
    fn before(c: ByScore, a: usize, b: usize) bool {
        return c.s[a] > c.s[b] or (c.s[a] == c.s[b] and a < b);
    }
};

/// Per row, the picks of `x`'s top half (finite scores, ties to the lower index) that `y`'s top half lacks, and how many
/// of those lie further from the half's last score than f32 reassociation moves a score (`mag`: its terms' magnitude).
fn pickDiffs(gpa: std.mem.Allocator, x: []const f32, y: []const f32, mag: []const f64, rows: usize, ncols: usize) ![2]usize {
    const order = try gpa.alloc(usize, ncols);
    defer gpa.free(order);
    const in_y = try gpa.alloc(bool, ncols);
    defer gpa.free(in_y);
    var out = [2]usize{ 0, 0 };
    for (0..rows) |r| {
        const xs = x[r * ncols ..][0..ncols];
        const ys = y[r * ncols ..][0..ncols];
        var finite: usize = 0;
        var row_mag: f64 = 0;
        for (ys, mag[r * ncols ..][0..ncols]) |v, m| {
            finite += @intFromBool(!std.math.isInf(v));
            row_mag = @max(row_mag, m);
        }
        const half = finite / 2;
        if (half == 0) continue;
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, ByScore{ .s = ys }, ByScore.before);
        @memset(in_y, false);
        for (order[0..half]) |i| in_y[i] = true;
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, ByScore{ .s = xs }, ByScore.before);
        const last: f64 = xs[order[half - 1]];
        for (order[0..half]) |i| if (!in_y[i]) {
            out[0] += 1;
            out[1] += @intFromBool(@abs(@as(f64, xs[i]) - last) > 8 * gamma(34) * row_mag);
        };
    }
    return out;
}

test "dsv41 dec idx_score: the scores equal today's within f32 reassociation, full and gathered, R = 1..6" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !nax.naxAvailable()) return error.SkipZigTest;
    const fx = try fixture.Fixture.init(testing.allocator, .{ .ctx = 1030 });
    defer fx.deinit();
    var k = try build();
    defer k.deinit();
    for ([_]usize{ 1, 2, 3, 4, 5, 6 }) |rows| for ([_]bool{ false, true }) |gather| try scoresCase(fx, &k, 20, rows, gather);
}

fn toF32(sc: *dsv41.Scope, x: A) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_astype(&o, x, .float32, sc.s), &o);
}

/// Rows [from, to) of an append-only state buffer.
fn bufRows(sc: *dsv41.Scope, r: dsv41.Rows, from: usize, to: usize) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_slice(&o, r.buf, &[_]c_int{ ci(from), 0 }, 2, &[_]c_int{ ci(to), ci(r.width) }, 2, &[_]c_int{ 1, 1 }, 2, sc.s), &o);
}

/// Our pool step of KV source `li` against `compress` on the same state and step: comp rows and pending rows
/// bit-equal; index keys checked and differing values, where `wk`'s f32 sum rounds another way across a bf16 boundary.
fn poolCase(fx: *fixture.Fixture, k: *const Kernels, li: usize, rows: usize) ![2]usize {
    const gpa = testing.allocator;
    const m = fx.m;
    const st = fx.state();
    const ls = &st.layers[li];
    const ratio: usize = m.layers[li].ratio;
    const p0 = st.n;
    var sc = dsv41.Scope.init(gpa, m.s);
    defer sc.deinit();
    const x = try sc.keep(try fx.normal(&.{ ci(rows), 5120 }, .bfloat16, 0, 1));
    const cw = m.layers[li].comp.?;
    const proj = if (ratio == 2) try dsv41.denseF32x(&sc, m, &cw.wkv, try toF32(&sc, x)) else try cw.wkv.apply(&sc, x);
    const old_kv = if (ls.pend_kv.ctx != null) sc.out(ls.pend_kv) else none;
    const old_gate = if (ls.pend_gate.ctx != null) sc.out(ls.pend_gate) else none;
    const mine = (try pool(&sc, m, k, li, proj, ls.pend_kv, ls.pend_gate, p0)) orelse return error.Declined;
    const comp_used = ls.comp.?.used;
    const idx_used = ls.idx_k.?.used;
    var sc2 = dsv41.Scope.init(gpa, m.s);
    defer sc2.deinit();
    try dsv41.compress(&sc2, m, st, li, &m.layers[li].comp.?, x, p0);
    const groups = ls.comp.?.used - comp_used;
    const np = if (ratio == 2) p0 % 2 else 0;
    try testing.expectEqual(if (ratio == 2) (np + rows) / 2 else rows, groups);
    var checked: usize = 0;
    var flips: usize = 0;
    if (groups > 0) {
        const wc = try dsv41.hostF32(try bufRows(&sc2, ls.comp.?, comp_used, comp_used + groups));
        defer gpa.free(wc);
        const gc = try dsv41.hostF32(mine.comp);
        defer gpa.free(gc);
        try testing.expectEqualSlices(f32, wc, gc);
        const wi = try dsv41.hostF32(try bufRows(&sc2, ls.idx_k.?, idx_used, idx_used + groups));
        defer gpa.free(wi);
        const gi = try dsv41.hostF32(mine.idx_k);
        defer gpa.free(gi);
        for (wi, gi) |a, b| {
            checked += 1;
            if (a != b) flips += 1;
        }
    }
    // The open group's rows.
    const left = (if (ratio == 2) (np + rows) % 2 else 0) == 1;
    try testing.expectEqual(left, ls.pend_kv.ctx != null);
    if (left) {
        const wk_ = try dsv41.hostF32(ls.pend_kv);
        defer gpa.free(wk_);
        const gk_ = try dsv41.hostF32(mine.pend_kv);
        defer gpa.free(gk_);
        try testing.expectEqualSlices(f32, wk_, gk_);
        const wg_ = try dsv41.hostF32(ls.pend_gate);
        defer gpa.free(wg_);
        const gg_ = try dsv41.hostF32(mine.pend_gate);
        defer gpa.free(gg_);
        try testing.expectEqualSlices(f32, wg_, gg_);
    } else try testing.expect(mine.pend_kv.ctx == null);
    // Back to the state before the step.
    ls.comp.?.used = comp_used;
    ls.idx_k.?.used = idx_used;
    if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
    if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
    ls.pend_kv = old_kv;
    ls.pend_gate = old_gate;
    return .{ checked, flips };
}

/// `poolCase` over `rows` steps (and `again` more of the widest), the index keys' tally printed and held to 0.2%.
fn poolCases(fx: *fixture.Fixture, k: *const Kernels, li: usize, rows: []const usize, again: usize) !void {
    var tally = [2]usize{ 0, 0 };
    for (rows) |r| for (try poolCase(fx, k, li, r), &tally) |t, *sum| {
        sum.* += t;
    };
    for (0..again) |_| for (try poolCase(fx, k, li, rows[rows.len - 1]), &tally) |t, *sum| {
        sum.* += t;
    };
    std.debug.print("[dec] idx_score pool layer {d} at p0={d}: comp and pending rows bit-equal, index keys {d} of {d} values differ\n", .{ li, fx.state().n, tally[1], tally[0] });
    try testing.expect(tally[1] * 500 <= tally[0]);
}

test "dsv41 dec idx_score: the pool step equals compress at ratio 2 (open group of 0 and 1 rows) and ratio 1, R = 1, 2, 3, 4, 6" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var k = try build();
    defer k.deinit();
    for ([_]usize{ 300, 301 }) |ctx| {
        const fx = try fixture.Fixture.init(testing.allocator, .{ .lo = 2, .hi = 4, .ctx = ctx });
        defer fx.deinit();
        try poolCases(fx, &k, 2, &.{ 1, 2, 3, 4, 6 }, 8);
    }
    const fx = try fixture.Fixture.init(testing.allocator, .{ .ctx = 300 });
    defer fx.deinit();
    try poolCases(fx, &k, 20, &.{ 1, 2, 4, 6 }, 40);
}

test "dsv41 dec idx_score: the declared launches are the shipped wrappers', byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var k = try build();
    defer k.deinit();
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    if (nax.naxAvailable()) for ([_]bool{ false, true }) |gather| {
        var c = kit.Case(ScoreCase).init(gpa, if (gather) &gather_l else &score_l, .{ .gather = gather });
        defer c.deinit();
        try testing.expect(try c.prepare(.{ .rows = 3, .n = 600, .k = channels }, .random, 7));
        try c.run();
        const theirs = try launchScores(&sc, if (gather) k.gather else k.score, c.args[0..if (gather) 6 else 5], 3, ScoreCase.cols(c.p, c.shape));
        try testing.expectEqualSlices(u8, try c.rig.bytesOf(&.{theirs}), c.rig.outputBytes());
    };
    // Ratio 2 with an open row left over, and without; ratio 1.
    for ([_]struct { pooling: bool, rows: usize, n: usize }{ .{ .pooling = true, .rows = 4, .n = 1 }, .{ .pooling = true, .rows = 5, .n = 1 }, .{ .pooling = false, .rows = 3, .n = 0 } }) |t| {
        var c = kit.Case(PoolCase).init(gpa, if (t.pooling) &pool2_l else &pool1_l, .{ .pooling = t.pooling });
        defer c.deinit();
        try testing.expect(try c.prepare(.{ .rows = t.rows, .n = t.n, .k = 512 }, .random, 7));
        try c.run();
        const left = PoolCase.leftover(c.p, c.shape);
        const theirs = try launchPool(&sc, if (t.pooling) k.pool2 else k.pool1, c.args[0..if (t.pooling) 9 else 7], PoolCase.groups(c.p, c.shape), t.pooling, left);
        const outs = [_]A{ theirs.comp, theirs.idx_k, theirs.pend_kv, theirs.pend_gate };
        try testing.expectEqualSlices(u8, try c.rig.bytesOf(outs[0..if (left) 4 else 2]), c.rig.outputBytes());
    }
}

test "dsv41 dec idx_score: the registered kernels pass the contract's gates 1-4" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    var set = kc.Set.init(testing.allocator);
    defer set.deinit();
    try register(&set);
    var t: kit.gates.Tally = .{};
    try kit.gates.runAll(testing.allocator, set.kernels.items, 0xc0de, &t);
    try testing.expectEqual(@as(usize, 0), t.problems());
}

// ── ubench ───────────────────────────────────────────────────────────────────

const dep_in = [_][*:0]const u8{"dep"};

/// Twins of the kernels with a trailing input they ignore: binding the previous launch's output there makes the next
/// launch wait for it, a dependent chain without an op between.
const Twins = struct {
    score: mlx.mlx_fast_metal_kernel,
    gather: mlx.mlx_fast_metal_kernel,
    scalar: mlx.mlx_fast_metal_kernel,
    pool2: mlx.mlx_fast_metal_kernel,
    pool1: mlx.mlx_fast_metal_kernel,

    fn init() !Twins {
        const b = dsv41.buildKernel;
        return .{
            .score = try b("dsv41_dec_idx_score_dep", &(SCORE_INS ++ dep_in), &.{"scores"}, SCORE_SOURCE, SCORE_HEADER),
            .gather = try b("dsv41_dec_idx_gather_dep", &(GATHER_INS ++ dep_in), &.{"scores"}, SCORE_SOURCE, GATHER_HEADER),
            .scalar = try b("dsv41_index_scores_dep", &(dsv41.IDX_INS ++ dep_in), &.{"scores"}, dsv41.IDX_SCORE_SOURCE, ""),
            .pool2 = try b("dsv41_dec_idx_pool2_dep", &(POOL2_INS ++ dep_in), &POOL2_OUTS, POOL_SOURCE, POOL2_HEADER),
            .pool1 = try b("dsv41_dec_idx_pool1_dep", &(POOL1_INS ++ dep_in), &POOL1_OUTS, POOL_SOURCE, POOL1_HEADER),
        };
    }

    fn deinit(t: *Twins) void {
        for ([_]mlx.mlx_fast_metal_kernel{ t.score, t.gather, t.scalar, t.pool2, t.pool1 }) |x| _ = mlx.mlx_fast_metal_kernel_free(x);
    }
};

fn nowUs() i64 {
    return @import("../../../sdk/util/io_util.zig").nowUsMonotonic(std.Io.Threaded.global_single_threaded.io());
}

/// Median over 7 evaluations (after 2) of the `ctx.n` steps `ctx.build` makes off the clock (the arrays it returns
/// are evaluated), in microseconds in all; the host's graph building is not timed.
fn evalTotalUs(gpa: std.mem.Allocator, s: mlx.mlx_stream, ctx: anytype) !f64 {
    var times: [7]f64 = undefined;
    for (0..2 + times.len) |rep| {
        var sc = dsv41.Scope.init(gpa, s);
        defer sc.deinit();
        const outs = try ctx.build(&sc);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (outs) |o| if (o.ctx != null) try mlx.check(mlx.mlx_vector_array_append_value(vec, o));
        const t0 = nowUs();
        try mlx.check(mlx.mlx_eval(vec));
        if (rep >= 2) times[rep - 2] = @floatFromInt(nowUs() - t0);
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

/// Microseconds a step of a dependent chain on the GPU: the difference of a chain of `ctx.n` steps and one of half as
/// many, over the steps between, which leaves out the commit and wait every evaluation pays.
fn evalUs(gpa: std.mem.Allocator, s: mlx.mlx_stream, ctx: anytype) !f64 {
    var shorter = ctx;
    shorter.n = ctx.n / 2;
    const long = try evalTotalUs(gpa, s, ctx);
    const short = try evalTotalUs(gpa, s, shorter);
    return (long - short) / @as(f64, @floatFromInt(ctx.n - shorter.n));
}

/// Today's scalar scores kernel over `inputs` (its inputs in order, then any it ignores).
fn launchScalar(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, inputs: []const A, rows: usize, nb: usize) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((nb + 255) / 256 * 256), ci(rows), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IH", heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IHD", channels));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "KT", .bfloat16));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// `n` launches of the scores kernel (or today's scalar kernel, and gathered its candidate mask) over distinct key
/// blocks, each waiting for the one before.
const ScoreBench = struct {
    twins: *const Twins,
    q: A,
    w: A,
    keys: []const A,
    cand: ?A,
    /// Today's candidate mask `[R, blocks]` bool.
    mask: A,
    rows: usize,
    n: usize,
    replaced: bool,

    fn build(b: ScoreBench, sc: *dsv41.Scope) ![4]A {
        const nb = dim(b.keys[0], 0);
        const meta = try sc.ints(&.{ @intCast(nb - b.rows), 1 }, &.{2});
        const wsc = try sc.keep(mlx.mlx_array_new_data(&[_]f32{wscale}, &[_]c_int{1}, 1, .float32));
        const ninf = try sc.keep(mlx.mlx_array_new_float(-std.math.inf(f32)));
        var dep = try sc.keep(mlx.mlx_array_new_data(&[_]f32{0}, &[_]c_int{1}, 1, .float32));
        for (0..b.n) |i| {
            const keys = b.keys[i % b.keys.len];
            if (b.replaced) {
                dep = try launchScalar(sc, b.twins.scalar, &.{ b.q, keys, b.w, meta, wsc, dep }, b.rows, nb);
                if (b.cand != null) {
                    var rep = mlx.mlx_array_new();
                    const wide = try sc.res(mlx.mlx_repeat_axis(&rep, b.mask, 8, -1, sc.s), &rep);
                    var o = mlx.mlx_array_new();
                    dep = try sc.res(mlx.mlx_where(&o, wide, dep, ninf, sc.s), &o);
                }
            } else if (b.cand) |c| {
                dep = try launchScores(sc, b.twins.gather, &.{ b.q, keys, b.w, meta, wsc, c, dep }, b.rows, dim(c, 1) * 8);
            } else {
                dep = try launchScores(sc, b.twins.score, &.{ b.q, keys, b.w, meta, wsc, dep }, b.rows, nb);
            }
        }
        return .{ dep, none, none, none };
    }
};

/// A candidate list of `blocks` ascending random block ids of `nblocks`, the same for every row, and today's mask.
fn benchCand(sc: *dsv41.Scope, rows: usize, blocks: usize, nblocks: usize) !struct { cand: A, mask: A } {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xc4d);
    const picked = try gpa.alloc(i32, blocks);
    defer gpa.free(picked);
    const seen = try gpa.alloc(bool, nblocks);
    defer gpa.free(seen);
    @memset(seen, false);
    var n: usize = 0;
    while (n < blocks) {
        const b = prng.random().uintLessThan(usize, nblocks);
        if (!seen[b]) {
            seen[b] = true;
            n += 1;
        }
    }
    n = 0;
    for (seen, 0..) |on, b| if (on) {
        picked[n] = @intCast(b);
        n += 1;
    };
    const cand_h = try gpa.alloc(i32, rows * blocks);
    defer gpa.free(cand_h);
    const mask_h = try gpa.alloc(bool, rows * nblocks);
    defer gpa.free(mask_h);
    for (0..rows) |r| {
        @memcpy(cand_h[r * blocks ..][0..blocks], picked);
        @memcpy(mask_h[r * nblocks ..][0..nblocks], seen);
    }
    return .{
        .cand = try sc.ints(cand_h, &.{ ci(rows), ci(blocks) }),
        .mask = try sc.keep(mlx.mlx_array_new_data(mask_h.ptr, &[_]c_int{ ci(rows), ci(nblocks) }, 2, .bool_)),
    };
}

fn scoreBench(gpa: std.mem.Allocator, fx: *fixture.Fixture, twins: *const Twins, nb: usize, gather: bool) !void {
    var sc = dsv41.Scope.init(gpa, fx.m.s);
    defer sc.deinit();
    const bytes_per: f64 = if (gather) 2048 * 8 * 256 else @floatFromInt(nb * 256);
    const copies: usize = @max(1, @min(1024, 2_000_000_000 / (nb * 256)));
    const keys = try gpa.alloc(A, copies);
    defer {
        for (keys) |x| _ = mlx.mlx_array_free(x);
        gpa.free(keys);
    }
    for (keys) |*x| x.* = try fx.normal(&.{ ci(nb), channels }, .bfloat16, 0, 1);
    for ([_]usize{ 1, 6 }) |rows| {
        const inp = try randomQ(fx, &sc, rows, nb - rows);
        const cm = if (gather) try benchCand(&sc, rows, 2048, nb / 8) else null;
        var b: ScoreBench = .{ .twins = twins, .q = inp.q, .w = inp.w, .keys = keys, .cand = if (cm) |c| c.cand else null, .mask = if (cm) |c| c.mask else none, .rows = rows, .n = copies, .replaced = false };
        const us = try evalUs(gpa, fx.m.s, b);
        b.replaced = true;
        const chain = try evalUs(gpa, fx.m.s, b);
        var name_buf: [64]u8 = undefined;
        decode.report(try std.fmt.bufPrint(&name_buf, "idx_score {s} {d} keys", .{ if (gather) "gathered" else "full", nb }), rows, us, bytes_per, chain);
    }
}

/// `n` pool steps of KV source `li` (the twin kernel, each waiting for the one before, or `compress` on the layer's
/// state, which the state chains by itself), a step of `rows` rows from position `p0` (+ rows a step).
const PoolBench = struct {
    twins: *const Twins,
    m: *dsv41.Dsv41Model,
    li: usize,
    x: A,
    proj: A,
    pend_kv: A,
    pend_gate: A,
    p0: usize,
    rows: usize,
    n: usize,
    replaced: bool,

    /// The layer's state as `compress` finds it: the rows used, the open group.
    fn rewind(b: PoolBench, ls: *dsv41.LState, comp_used: usize, idx_used: usize, kv: A, gate: A) void {
        _ = b;
        ls.comp.?.used = comp_used;
        ls.idx_k.?.used = idx_used;
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = if (kv.ctx != null) copyOf(kv) else none;
        ls.pend_gate = if (gate.ctx != null) copyOf(gate) else none;
    }

    fn copyOf(x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }

    fn build(b: PoolBench, sc: *dsv41.Scope) ![4]A {
        const m = b.m;
        const ls = &m.dec_state.?.layers[b.li];
        if (b.replaced) {
            const comp_used = ls.comp.?.used;
            const idx_used = ls.idx_k.?.used;
            for (0..b.n) |i| try dsv41.compress(sc, m, &m.dec_state.?, b.li, &m.layers[b.li].comp.?, b.x, b.p0 + i * b.rows);
            const out = [4]A{ ls.comp.?.buf, ls.idx_k.?.buf, ls.pend_kv, ls.pend_gate };
            // Handles the scope frees with it; the state goes back to its start at the next build.
            var held: [4]A = undefined;
            for (&held, out) |*h, o| h.* = if (o.ctx != null) try sc.keep(sc.out(o)) else none;
            ls.comp.?.used = comp_used;
            ls.idx_k.?.used = idx_used;
            return held;
        }
        const plan = (try poolPlan(sc, m, b.li, b.proj, b.pend_kv, b.pend_gate, b.p0)) orelse return error.Declined;
        var dep = try sc.keep(mlx.mlx_array_new_data(&[_]f32{0}, &[_]c_int{1}, 1, .float32));
        var last: Pooled = undefined;
        for (0..b.n) |_| {
            var ins: [10]A = undefined;
            @memcpy(ins[0..plan.n_ins], plan.ins[0..plan.n_ins]);
            ins[plan.n_ins] = dep;
            last = try launchPool(sc, if (plan.pooling) b.twins.pool2 else b.twins.pool1, ins[0 .. plan.n_ins + 1], plan.groups, plan.pooling, plan.leftover);
            dep = last.comp;
        }
        return .{ last.comp, last.idx_k, last.pend_kv, last.pend_gate };
    }
};

fn poolBench(gpa: std.mem.Allocator, fx: *fixture.Fixture, twins: *const Twins, li: usize) !void {
    const m = fx.m;
    const ls = &fx.state().layers[li];
    const ratio: usize = m.layers[li].ratio;
    for ([_]usize{ 1, 6 }) |rows| {
        var sc = dsv41.Scope.init(gpa, m.s);
        defer sc.deinit();
        const x = try sc.keep(try fx.normal(&.{ ci(rows), 5120 }, .bfloat16, 0, 1));
        const cw = m.layers[li].comp.?;
        const proj = if (ratio == 2) try dsv41.denseF32x(&sc, m, &cw.wkv, try toF32(&sc, x)) else try cw.wkv.apply(&sc, x);
        // A step that completes a group and, at ratio 2 with 6 rows, leaves one open: p0 odd with a row open.
        const p0 = fx.state().n;
        var b: PoolBench = .{ .twins = twins, .m = m, .li = li, .x = x, .proj = proj, .pend_kv = ls.pend_kv, .pend_gate = ls.pend_gate, .p0 = p0, .rows = rows, .n = 128, .replaced = false };
        const us = try evalUs(gpa, m.s, b);
        // `compress` mutates the state it chains through: start every build from the fixture's.
        const comp_used = ls.comp.?.used;
        const idx_used = ls.idx_k.?.used;
        const kv = if (ls.pend_kv.ctx != null) sc.out(ls.pend_kv) else none;
        const gate = if (ls.pend_gate.ctx != null) sc.out(ls.pend_gate) else none;
        defer {
            if (kv.ctx != null) _ = mlx.mlx_array_free(kv);
            if (gate.ctx != null) _ = mlx.mlx_array_free(gate);
        }
        b.replaced = true;
        b.n = 32;
        const chain = try evalUsRewound(gpa, m.s, b, ls, comp_used, idx_used, kv, gate);
        var name_buf: [64]u8 = undefined;
        decode.report(try std.fmt.bufPrint(&name_buf, "idx_score pool ratio {d}", .{ratio}), rows, us, @floatFromInt(131072 + rows * 4096), chain);
    }
}

/// `evalUs` of a `compress` chain, the layer's state put back before every build.
fn evalUsRewound(gpa: std.mem.Allocator, s: mlx.mlx_stream, b: PoolBench, ls: *dsv41.LState, comp_used: usize, idx_used: usize, kv: A, gate: A) !f64 {
    const Rewound = struct {
        b: PoolBench,
        n: usize,
        ls: *dsv41.LState,
        comp_used: usize,
        idx_used: usize,
        kv: A,
        gate: A,
        fn build(r: @This(), sc: *dsv41.Scope) ![4]A {
            r.b.rewind(r.ls, r.comp_used, r.idx_used, r.kv, r.gate);
            var bb = r.b;
            bb.n = r.n;
            return bb.build(sc);
        }
    };
    return evalUs(gpa, s, Rewound{ .b = b, .n = b.n, .ls = ls, .comp_used = comp_used, .idx_used = idx_used, .kv = kv, .gate = gate });
}

test "dsv41 dec idx_score: ubench" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !nax.naxAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var twins = try Twins.init();
    defer twins.deinit();
    {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .ctx = 0 });
        defer fx.deinit();
        for ([_]usize{ 4096, 32768, 131072, 1048576 }) |nb| try scoreBench(gpa, fx, &twins, nb, false);
        try scoreBench(gpa, fx, &twins, 131072, true);
    }
    {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 2, .hi = 4, .ctx = 301 });
        defer fx.deinit();
        try poolBench(gpa, fx, &twins, 2);
    }
    {
        const fx = try fixture.Fixture.init(gpa, .{ .ctx = 300 });
        defer fx.deinit();
        try poolBench(gpa, fx, &twins, 20);
    }
}
