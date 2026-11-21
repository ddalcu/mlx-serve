//! k5 `attn` (`~/plans/dsv41-decode-kernels-spec.md`): a decode step's sparse attention over the window and the picked
//! compressed rows, as the split kernel's partials (`dec_attn.metal`, on the tensor units). Whoever merges them is the
//! consumer (`attn_merge`'s math).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const device = @import("../../../mlx/device.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const contract = @import("../contract.zig");
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;
const none: A = .{ .ctx = null };

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_attn.metal");
pub const HEADER = kernelSource("kernels/common/mpp_tensor_ops.metal");
pub const INS = [_][*:0]const u8{ "q", "ring", "comp", "topk", "meta", "scal" };
pub const OUTS = [_][*:0]const u8{ "pacc", "pm", "pl" };

const HD = 512;
/// The most keys a split holds, the most splits a (row, head group) has, a threadgroup's threads and its bytes.
const BK = 128;
const MAX_SPLITS = 8;
const TG = 128;
const TG_BYTES = 4 * (BK + 2 * 4 * 16) + 2 * 2 * (BK / 16) * 32 * 8;

pub fn build() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_attn", &INS, &OUTS, SOURCE, HEADER);
}

/// The merge variant (`MLX_SERVE_DSV41_FUSE_ATTN_MERGE`): the same kernel, whose last split of each (row, 16 heads)
/// merges the splits with the sink once, so wo_a reads one merged split instead of re-merging the splits in each of
/// its 16 output tiles of the head.
pub const MERGE_HEADER = HEADER ++ "#define DSV41_ATTN_MERGE 1\n";
pub const MERGE_INS = INS ++ [_][*:0]const u8{ "sink", "counter" };
pub const MERGE_OUTS = OUTS ++ [_][*:0]const u8{ "merged", "pmo", "plo" };

pub fn buildMerge() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_dec_attn_merge", &MERGE_INS, &MERGE_OUTS, SOURCE, MERGE_HEADER);
}

/// `MLX_SERVE_DSV41_FUSE_ATTN_MERGE` (read each call): =1 the decode attention merges its splits (`partialsMerged`).
pub fn mergeOn() bool {
    const v = env.get(.MLX_SERVE_DSV41_FUSE_ATTN_MERGE) orelse return false;
    return v[0] == '1';
}

/// Words of the merge variant's arrival counter: a word a (row, 16 heads), zeroed once; each launch's last arrival
/// zeroes its word again.
pub const counter_words = 256;

pub fn newCounter() A {
    return mlx.mlx_array_new_data(&std.mem.zeroes([counter_words]u32), &[_]c_int{counter_words}, 1, .uint32);
}

/// The sink wo_a reads against a merged split: so low that its weight is 0 (exp underflows), so the merged split
/// passes through unchanged (max 0, weight sum 1).
pub fn noSink(heads: usize) A {
    var v: [256]f32 = undefined;
    @memset(&v, -3.0e38);
    return mlx.mlx_array_new_data(&v, &[_]c_int{@intCast(heads)}, 1, .float32);
}

/// What `attn_split_epilogue.metal` writes: each split's unnormalised weighted keys `acc` [R, H, splits, 512], its max
/// score `m` and weight sum `l` [R, H, splits], all f32. `m` is -inf and `l` 0 where no key counted.
pub const Partials = struct { acc: A, m: A, l: A, splits: c_int };

fn dimOf(x: A, axis: usize) c_int {
    return mlx.getShape(x)[axis];
}

/// The splits for a list of `keys`: at most BK keys each, and for a short list about 32, which is all the parallelism
/// a row's 16-head groups get. Null: the list needs more than MAX_SPLITS.
fn splitsFor(keys: usize) ?c_int {
    const least = (keys + BK - 1) / BK;
    if (least > MAX_SPLITS) return null;
    const s: c_int = @intCast(@max(least, @min(MAX_SPLITS, (keys + 31) / 32)));
    return s;
}

/// Rows q [R, H, 512] bf16 (roped, 1..`decode.max_rows` of them, H a multiple of 16) at positions p0.., the window
/// ring already holding their keys, the latest KV source's compressed rows and the rows' picks `topk` [R, K] (-1 =
/// none; absent at ratio 0). Null: the shape, dtype or stream is not one it serves (the caller keeps today's path).
pub fn partials(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, m: *const dsv41.Dsv41Model, q: A, win: dsv41.Ring, comp_rows: A, topk: A, p0: usize, scale: f32) !?Partials {
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32));
    return run(sc, kern, m, q, win, comp_rows, topk, p0, scal);
}

/// `partials` on the merge variant: the splits merged with `sink` [H] (f32) by their last arrival, as ONE split
/// (`acc` [R, H, 1, 512], `m` 0, `l` 1) for a consumer that merges against `noSink`. Null as `partials`.
pub fn partialsMerged(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, m: *const dsv41.Dsv41Model, q: A, win: dsv41.Ring, comp_rows: A, topk: A, p0: usize, scale: f32, sink: A, counter: A) !?Partials {
    const qs = mlx.getShape(q);
    if (qs.len != 3 or mlx.mlx_array_size(sink) != qs[1] or mlx.mlx_array_dtype(sink) != .float32 or qs[0] * @divTrunc(qs[1], 16) > counter_words) return null;
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32));
    return runWith(sc, kern, m, q, win, comp_rows, topk, p0, scal, .{ sink, counter });
}

fn run(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, m: *const dsv41.Dsv41Model, q: A, win: dsv41.Ring, comp_rows: A, topk: A, p0: usize, scal: A) !?Partials {
    return runWith(sc, kern, m, q, win, comp_rows, topk, p0, scal, null);
}

fn runWith(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, m: *const dsv41.Dsv41Model, q: A, win: dsv41.Ring, comp_rows: A, topk: A, p0: usize, scal: A, merge: ?[2]A) !?Partials {
    if (!mlx.streamIsGpu(sc.s) or !device.naxAvailable()) return null;
    const qs = mlx.getShape(q);
    if (qs.len != 3 or qs[2] != HD or @mod(qs[1], 16) != 0 or qs[0] < 1 or qs[0] > decode.max_rows) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(win.buf) != .bfloat16 or win.width != HD) return null;
    const R = qs[0];
    const H = qs[1];
    const K: c_int = if (comp_rows.ctx != null) dimOf(topk, 1) else 0;
    const nc: c_int = if (K > 0) dimOf(comp_rows, 0) else 0;
    if (K > 0 and nc == 0) return null;
    const S = splitsFor(m.cfg.sliding_window + @as(usize, @intCast(K))) orelse return null;
    const meta_v = [_]i32{ @intCast(p0), nc, @intCast(m.cfg.sliding_window), @intCast(win.cap) };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const comp = if (K > 0) try sc.astype(comp_rows, .bfloat16) else win.buf;
    const ids = if (K > 0) try sc.astype(topk, .int32) else meta;

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, H, S, HD }, 4, .float32));
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, H, S }, 3, .float32));
    if (merge != null) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, H, 1, HD }, 4, .float32));
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ R, H, 1 }, 3, .float32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(H, 16) * TG, R, S));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", S));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", K));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", H));
    const inputs = [_]A{ q, win.buf, comp, ids, meta, scal, if (merge) |g| g[0] else none, if (merge) |g| g[1] else none };
    const iv = mlx.mlx_vector_array_new_data(&inputs, if (merge != null) MERGE_INS.len else INS.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var parts: [MERGE_OUTS.len]A = undefined;
    for (parts[0..if (merge != null) MERGE_OUTS.len else OUTS.len], 0..) |*x, k| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(x, ov, k));
        _ = try sc.keep(x.*);
    }
    if (merge != null) return .{ .acc = parts[3], .m = parts[4], .l = parts[5], .splits = 1 };
    return .{ .acc = parts[0], .m = parts[1], .l = parts[2], .splits = S };
}

// ── the kernel contract: the split kernel's reference and rule, on this kernel's launch ──

const WINDOW = 128;
const act: mlx.mlx_dtype = .bfloat16;

fn ci(v: usize) c_int {
    return @intCast(v);
}

const launch_decl = kit.metal.Launch{
    .name = "dsv41_dec_attn",
    .inputs = &INS,
    .outputs = &OUTS,
    .source = SOURCE,
    .header = HEADER,
    .tg_bytes = TG_BYTES,
    .offline = false,
    .min_gen = 17,
};

/// The partials of `s` splits of rows of `n` heads with `k` picks: `contract.AttnSplit`'s values, split bounds and
/// rule, over this kernel's inputs (q, ring, comp, picks, meta, scale).
const DecAttn = struct {
    pub const Params = contract.AttnSplit.Params;

    pub fn geometry(p: Params, s: kit.Shape) ?kit.metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.n % 16 != 0 or p.s == 0 or (WINDOW + s.k + p.s - 1) / p.s > BK) return null;
        var g = kit.metal.Geometry{ .grid = .{ ci(s.n / 16 * TG), ci(s.rows), ci(p.s) }, .tg = .{ TG, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s), HD }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s) }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), ci(p.s) }, .float32);
        g.dtype("T", act);
        g.int("S", p.s);
        g.int("K", s.k);
        g.int("H", s.n);
        return g;
    }

    /// q [L, H, HD], the ring [cap, HD], the compressed rows [nc, HD], picks [L, K] (some unset, some past the cache),
    /// meta, the scale.
    pub fn fill(k: anytype, f: kit.Fill, rng: std.Random) !void {
        const g = k.p.g;
        const L = k.shape.rows;
        const H = k.shape.n;
        const K = k.shape.k;
        const q = try k.add(act, &.{ L, H, HD });
        const ring = try k.add(act, &.{ g.cap, HD });
        const comp = try k.add(act, if (K > 0) &.{ g.nc, HD } else &.{ g.cap, HD });
        const topk = try k.add(.int32, if (K > 0) &.{ L, K } else &.{4});
        const meta = try k.add(.int32, &.{4});
        const scal = try k.add(.float32, &.{1});
        kit.uniform(rng, q, -2, 2);
        kit.uniform(rng, ring, -2, 2);
        kit.uniform(rng, comp, -2, 2);
        @memcpy(meta, &[_]f64{ @floatFromInt(g.p0), if (K > 0) @floatFromInt(g.nc) else 0, WINDOW, @floatFromInt(g.cap) });
        if (K > 0) {
            for (topk, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else if (i % 41 == 7) @floatFromInt(g.nc + 200) else @floatFromInt((i * 13) % g.nc);
        }
        scal[0] = g.scale;
        if (f == .nonfinite) kit.plantRows(q, L * H, HD);
    }

    pub const count = contract.AttnSplit.count;
    pub const compute = contract.AttnSplit.compute;
    pub const check = contract.AttnSplit.check;
};

const merge_decl = kit.metal.Launch{
    .name = "dsv41_dec_attn_merge",
    .inputs = &MERGE_INS,
    .outputs = &MERGE_OUTS,
    .source = SOURCE,
    .header = MERGE_HEADER,
    .tg_bytes = TG_BYTES + 4 * 16 * (MAX_SPLITS + 1) + 4,
    .offline = false,
    .min_gen = 17,
};

/// The merge variant: `DecAttn`'s partials, then the splits of each (row, head) merged with the sink (input 6) as one
/// split `merged` [R, H, 1, 512] f32 with max 0 and sum 1. Its rule: the partials are `DecAttn`'s, the merge is the
/// float64 merge of the launch's own partials within the merge's rounding, and the max and sum are exactly 0 and 1.
const MergeAttn = struct {
    pub const Params = DecAttn.Params;

    pub fn geometry(p: Params, s: kit.Shape) ?kit.metal.Geometry {
        var g = DecAttn.geometry(p, s) orelse return null;
        if (s.rows * s.n / 16 > counter_words) return null;
        g.out(&.{ ci(s.rows), ci(s.n), 1, HD }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), 1 }, .float32);
        g.out(&.{ ci(s.rows), ci(s.n), 1 }, .float32);
        return g;
    }

    pub fn fill(k: anytype, f: kit.Fill, rng: std.Random) !void {
        try DecAttn.fill(k, f, rng);
        kit.uniform(rng, try k.add(.float32, &.{k.shape.n}), -3, 3);
        _ = try k.add(.uint32, &.{counter_words});
    }

    fn splitCount(p: Params, s: kit.Shape) usize {
        return contract.AttnSplit.count(p, s);
    }

    pub fn count(p: Params, s: kit.Shape) usize {
        return splitCount(p, s) + s.rows * s.n * (HD + 2);
    }

    /// Head (i, h)'s merge of the partials `part` (pacc, then pm, then pl, as the kernel lays them out) with the sink,
    /// in float64: the channels into `o` and each one's bound (the merge's f32 rounding) into `e`.
    fn mergeHead(k: anytype, part: []const f64, i: usize, h: usize, o: *[HD]f64, e: *[HD]f64) void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const S = k.p.s;
        const nm = L * H * S * HD;
        const base = (i * H + h) * S;
        const sk = k.in(6)[h];
        var M = sk;
        for (0..S) |j| M = if (std.math.isNan(part[nm + base + j])) M else @max(M, part[nm + base + j]);
        var total = @exp(sk - M);
        var om: [HD]f64 = @splat(0);
        o.* = @splat(0);
        var spread: f64 = @abs(sk - M);
        for (0..S) |j| {
            const l = part[nm + L * H * S + base + j];
            const w = if (l > 0) @exp(part[nm + base + j] - M) else 0;
            if (l > 0) spread = @max(spread, @abs(part[nm + base + j] - M));
            total += l * w;
            for (0..HD) |d| {
                o[d] += part[(base + j) * HD + d] * w;
                om[d] += @abs(part[(base + j) * HD + d] * w);
            }
        }
        const rel = (@as(f64, @floatFromInt(S)) + 8 + spread) * contract.u;
        for (o, e, om) |*v, *b, m| {
            v.* /= total;
            b.* = m / total * rel + @abs(v.*) * rel;
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: kit.Sum) !void {
        const L = k.shape.rows;
        const H = k.shape.n;
        const ns = splitCount(k.p, k.shape);
        try contract.AttnSplit.compute(k, out[0..ns], mag[0..ns], sum);
        var o: [HD]f64 = undefined;
        var e: [HD]f64 = undefined;
        for (0..L) |i| for (0..H) |h| {
            mergeHead(k, out[0..ns], i, h, &o, &e);
            @memcpy(out[ns + (i * H + h) * HD ..][0..HD], &o);
        };
        @memset(out[ns + L * H * HD ..][0 .. L * H], 0);
        @memset(out[ns + L * H * HD + L * H ..][0 .. L * H], 1);
        @memset(mag, 0);
    }

    pub fn check(k: anytype, at: kit.spec.Gate) !?[]const u8 {
        if (try contract.AttnSplit.check(k, at)) |why| return why;
        const L = k.shape.rows;
        const H = k.shape.n;
        const ns = splitCount(k.p, k.shape);
        const a = k.scratch();
        const part = try a.alloc(f64, ns);
        defer a.free(part);
        const bytes = k.rig.outputBytes();
        for (part, 0..) |*v, i| v.* = kit.spec.read(.f32, bytes, i);
        var o: [HD]f64 = undefined;
        var e: [HD]f64 = undefined;
        for (0..L) |i| for (0..H) |h| {
            mergeHead(k, part, i, h, &o, &e);
            for (o, e, 0..) |want, b, d| {
                const got = kit.spec.read(.f32, bytes, ns + (i * H + h) * HD + d);
                const ok = if (at == .nonfinite) kit.spec.classOf(got) == kit.spec.classOf(want) else @abs(got - want) <= b + 0x1p-140;
                if (!ok) return std.fmt.bufPrint(&k.msg, "row {d} head {d} channel {d} merged {e}, the partials' merge {e}", .{ i, h, d, got, want }) catch "merge";
            }
            if (kit.spec.read(.f32, bytes, ns + L * H * HD + i * H + h) != 0 or kit.spec.read(.f32, bytes, ns + L * H * HD + L * H + i * H + h) != 1)
                return std.fmt.bufPrint(&k.msg, "row {d} head {d}: the merged split's max and sum are not 0 and 1", .{ i, h }) catch "ml";
        };
        return null;
    }
};

pub fn register(set: *kc.Set) !void {
    const nan: kit.spec.Nan = .{ .stated = "a split whose weights go NaN reports max -inf and a NaN partial output and sum, which the merge propagates; else classes as float64 gives them" };
    try kit.add(set, DecAttn, &launch_decl, .{ .s = 8 }, .{
        .name = "dsv41 dec attn S=8",
        .accum = .none,
        .depth = contract.depthAttn,
        .out = .f32,
        .nan = nan,
        .nax = true,
        .served = &.{ .{ .rows = 1, .n = 16, .k = 512 }, .{ .rows = 6, .n = 32, .k = 512 }, .{ .rows = 2, .n = 16, .k = 100 } },
    });
    try kit.add(set, DecAttn, &launch_decl, .{ .s = 4 }, .{
        .name = "dsv41 dec attn S=4, window only",
        .accum = .none,
        .depth = contract.depthAttn,
        .out = .f32,
        .nan = nan,
        .nax = true,
        .served = &.{ .{ .rows = 1, .n = 16, .k = 0 }, .{ .rows = 6, .n = 16, .k = 0 } },
    });
    try kit.add(set, MergeAttn, &merge_decl, .{ .s = 8 }, .{
        .name = "dsv41 dec attn merge S=8 (the last split merges the splits with the sink)",
        .accum = .none,
        .depth = contract.depthAttn,
        .out = .f32,
        .nan = nan,
        .nax = true,
        .served = &.{ .{ .rows = 1, .n = 16, .k = 512 }, .{ .rows = 6, .n = 32, .k = 512 }, .{ .rows = 2, .n = 16, .k = 100 } },
    });
    try kit.add(set, MergeAttn, &merge_decl, .{ .s = 4 }, .{
        .name = "dsv41 dec attn merge S=4, window only",
        .accum = .none,
        .depth = contract.depthAttn,
        .out = .f32,
        .nan = nan,
        .nax = true,
        .served = &.{ .{ .rows = 1, .n = 16, .k = 0 }, .{ .rows = 6, .n = 16, .k = 0 } },
    });
}

// ── tests ─────────────────────────────────────────────────────────────────

const testing = std.testing;
const fixture = @import("fixture.zig");

const HEADS = 64;
const SCALE: f32 = 1.0 / @sqrt(@as(f32, HD));

/// The attention inputs of a decode step at the release's shapes: `copies` distinct compressed-row buffers and `steps`
/// pick sets, so a ubench's picked rows come cold from memory; the ring and q are the previous kernels' output, warm.
const Rig = struct {
    rows: usize,
    k: usize,
    copies: usize,
    steps: usize,
    p0: usize,
    q: A,
    sink: A,
    ring: A,
    comps: []A,
    picks: []A,

    fn randn(sc: *Scope, shape: []const c_int, seed: u64) !A {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed));
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .bfloat16, 0, 1, key, sc.s), &o);
    }

    fn init(sc: *Scope, rows: usize, n_comp: usize, k: usize, copies: usize, steps: usize, p0: usize, seed: u64) !*Rig {
        const gpa = testing.allocator;
        const rig = try gpa.create(Rig);
        errdefer gpa.destroy(rig);
        rig.* = .{ .rows = rows, .k = k, .copies = copies, .steps = steps, .p0 = p0, .q = undefined, .sink = undefined, .ring = undefined, .comps = try gpa.alloc(A, copies), .picks = try gpa.alloc(A, steps) };
        rig.q = try randn(sc, &.{ @intCast(rows), HEADS, HD }, seed);
        rig.ring = try randn(sc, &.{ 144, HD }, seed + 1);
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, seed + 2));
        var sk = mlx.mlx_array_new();
        rig.sink = try sc.res(mlx.mlx_random_normal(&sk, &[_]c_int{ 1, HEADS, 1 }, 3, .float32, 0, 1, key, sc.s), &sk);
        const comp_all = if (k > 0) try randn(sc, &.{ @intCast(copies * n_comp), HD }, seed + 3) else none;
        try mlx.check(mlx.mlx_array_eval(rig.q));
        try mlx.check(mlx.mlx_array_eval(rig.ring));
        try mlx.check(mlx.mlx_array_eval(rig.sink));
        if (k > 0) try mlx.check(mlx.mlx_array_eval(comp_all));
        for (rig.comps, 0..) |*c, i| c.* = if (k > 0) try sliceRows(sc, comp_all, i * n_comp, (i + 1) * n_comp) else none;
        // Each step's rows pick k distinct compressed rows, the tail unset where the cache has fewer.
        const idx = try gpa.alloc(i32, rows * k);
        defer gpa.free(idx);
        const perm = try gpa.alloc(i32, n_comp);
        defer gpa.free(perm);
        for (perm, 0..) |*v, i| v.* = @intCast(i);
        var prng = std.Random.DefaultPrng.init(seed + 4);
        const rnd = prng.random();
        for (rig.picks) |*pk| {
            for (0..rows) |r| {
                for (0..@min(k, n_comp)) |i| std.mem.swap(i32, &perm[i], &perm[i + rnd.uintLessThan(usize, n_comp - i)]);
                for (0..k) |i| idx[r * k + i] = if (i < n_comp) perm[i] else -1;
            }
            pk.* = if (k > 0) try sc.keep(mlx.mlx_array_new_data(idx.ptr, &[_]c_int{ @intCast(rows), @intCast(k) }, 2, .int32)) else none;
        }
        return rig;
    }

    fn deinit(rig: *Rig) void {
        const gpa = testing.allocator;
        gpa.free(rig.comps);
        gpa.free(rig.picks);
        gpa.destroy(rig);
    }

    fn win(rig: *const Rig) dsv41.Ring {
        return .{ .buf = rig.ring, .cap = 144, .width = HD };
    }
};

fn sliceRows(sc: *Scope, x: A, lo: usize, hi: usize) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_slice(&o, x, &[_]c_int{ @intCast(lo), 0 }, 2, &[_]c_int{ @intCast(hi), HD }, 2, &[_]c_int{ 1, 1 }, 2, sc.s), &o);
}

/// Today's kernels and the model fields `attnDecodeSplit` reads.
fn todayModel() !dsv41.Dsv41Model {
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = WINDOW;
    m.act = .bfloat16;
    m.attn_dec_kernel = try dsv41.buildKernel("dsv41_attn_decode", &dsv41.ATTN_DEC_INS, &.{"out"}, dsv41.ATTN_DEC_SOURCE, dsv41.UNROPE_HEADER);
    m.attn_split_kernel = try dsv41.buildKernel("dsv41_attn_split", &dsv41.ATTN_INS, &.{ "pacc", "pm", "pl" }, dsv41.ATTN_SPLIT_SOURCE, "");
    m.attn_merge_kernel = try dsv41.buildKernel("dsv41_attn_merge", &dsv41.ATTN_MERGE_INS, &.{"out"}, dsv41.ATTN_MERGE_SOURCE, dsv41.UNROPE_HEADER);
    return m;
}

fn freeTodayModel(m: *dsv41.Dsv41Model) void {
    _ = mlx.mlx_fast_metal_kernel_free(m.attn_dec_kernel);
    _ = mlx.mlx_fast_metal_kernel_free(m.attn_split_kernel);
    _ = mlx.mlx_fast_metal_kernel_free(m.attn_merge_kernel);
}

/// Today's merge of `p` with the sink: the heads [R, H, 512] bf16, no inverse rope.
fn mergeToday(sc: *Scope, m: *const dsv41.Dsv41Model, p: Partials, sink: A, p0: usize) !A {
    const R = dimOf(p.acc, 0);
    const H = dimOf(p.acc, 1);
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, &[_]c_int{ R, H, HD }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, HD, H, R));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, HD, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "D", HD));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "S", p.splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "RD", 0));
    const meta = try sc.ints(&[_]i32{ @intCast(p0), 0, 0, 0, 0, 0 }, &.{6});
    const periods = try sc.zeros(&.{1}, .float32);
    const mv = mlx.mlx_vector_array_new_data(&[_]A{ p.acc, p.m, p.l, sink, meta, periods }, 6);
    defer _ = mlx.mlx_vector_array_free(mv);
    var mo = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(mo);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&mo, m.attn_merge_kernel, mv, mc, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, mo, 0), &o);
}

test "dsv41 dec attn: matches today's split and merge" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m = try todayModel();
    defer freeTodayModel(&m);
    var ly: dsv41.Layer = undefined;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    for ([_]usize{ 512, 100, 0 }) |k| for ([_]usize{ 1, 2, 4, 6 }) |R| for ([_]usize{ 300, 5000 }) |ctx| {
        const rig = try Rig.init(&sc, R, ctx, k, 1, 1, ctx, 9);
        defer rig.deinit();
        ly.sink = rig.sink;
        const want = try dsv41.hostF32(try dsv41.attnDecodeSplit(&sc, &m, &ly, rig.q, rig.win(), rig.comps[0], rig.picks[0], rig.p0, SCALE, dsv41.attnSplits(WINDOW + k), null));
        defer gpa.free(want);
        const p = (try partials(&sc, kern, &m, rig.q, rig.win(), rig.comps[0], rig.picks[0], rig.p0, SCALE)) orelse return error.Declined;
        const got = try dsv41.hostF32(try mergeToday(&sc, &m, p, rig.sink, rig.p0));
        defer gpa.free(got);
        const e = decode.relRms(got, want);
        if (e >= 1e-3) {
            std.debug.print("[dec] attn K={d} R={d} ctx={d}: rel rms vs today {e:.2}\n", .{ k, R, ctx, e });
            return error.TestUnexpectedResult;
        }
    };
}

/// f64 attention of each (row, head) over its window rows and picked rows, the sink in the denominator only: heads
/// [R, H, 512].
fn truthHeads(gpa: std.mem.Allocator, q: []const f32, ring: []const f32, comp: []const f32, picks: []const i32, sink: []const f32, rows: usize, k: usize, nc: usize, p0: usize) ![]f32 {
    const out = try gpa.alloc(f32, rows * HEADS * HD);
    const s = try gpa.alloc(f64, WINDOW + k);
    defer gpa.free(s);
    const keys = try gpa.alloc(?[]const f32, WINDOW + k);
    defer gpa.free(keys);
    const acc = try gpa.alloc(f64, HD);
    defer gpa.free(acc);
    for (0..rows) |i| {
        const pos = p0 + i;
        var n: usize = 0;
        for ((pos + 1) -| WINDOW..pos + 1) |p| {
            keys[n] = ring[(p % 144) * HD ..][0..HD];
            n += 1;
        }
        for (0..k) |j| {
            const c = picks[i * k + j];
            keys[n] = if (c >= 0 and @as(usize, @intCast(c)) < nc) comp[@as(usize, @intCast(c)) * HD ..][0..HD] else null;
            n += 1;
        }
        for (0..HEADS) |h| {
            const qh = q[(i * HEADS + h) * HD ..][0..HD];
            var top: f64 = sink[h];
            for (keys[0..n], s[0..n]) |key, *sv| {
                if (key) |kr| {
                    var d: f64 = 0;
                    for (qh, kr) |a, b| d += @as(f64, a) * @as(f64, b);
                    sv.* = d * SCALE;
                    top = @max(top, sv.*);
                } else sv.* = -std.math.inf(f64);
            }
            var denom = @exp(@as(f64, sink[h]) - top);
            @memset(acc, 0);
            for (keys[0..n], s[0..n]) |key, sv| {
                if (key) |kr| {
                    const w = @exp(sv - top);
                    denom += w;
                    for (acc, kr) |*a, b| a.* += w * @as(f64, b);
                }
            }
            for (acc, 0..) |a, d| out[(i * HEADS + h) * HD + d] = @floatCast(a / denom);
        }
    }
    return out;
}

/// `buf` [144, 512] with the rows of positions p0.. written at their slots.
fn ringWith(sc: *Scope, buf: A, new_rows: A, p0: usize) !A {
    var cur = buf;
    for (0..@intCast(dimOf(new_rows, 0))) |r| {
        const slot: c_int = @intCast((p0 + r) % 144);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, new_rows, &[_]c_int{ @intCast(r), 0 }, 2, &[_]c_int{ @intCast(r + 1), HD }, 2, &[_]c_int{ 1, 1 }, 2, sc.s));
        var up = mlx.mlx_array_new();
        cur = try sc.res(mlx.mlx_slice_update(&up, cur, row, &[_]c_int{ slot, 0 }, 2, &[_]c_int{ slot + 1, HD }, 2, &[_]c_int{ 1, 1 }, 2, sc.s), &up);
    }
    return cur;
}

test "dsv41 dec attn: rows 1, 2, 4, 6 on the fixture equal today's split and merge, no further from truth" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const Case = struct { lo: usize, hi: usize, ctx: usize };
    for ([_]Case{ .{ .lo = 20, .hi = 25, .ctx = 300 }, .{ .lo = 20, .hi = 25, .ctx = 4200 }, .{ .lo = 0, .hi = 1, .ctx = 300 }, .{ .lo = 0, .hi = 1, .ctx = 4200 } }) |c| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = c.lo, .hi = c.hi, .experts = 16, .ctx = c.ctx });
        defer fx.deinit();
        const m = fx.m;
        const kern = try build();
        defer _ = mlx.mlx_fast_metal_kernel_free(kern);
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const st = fx.state();
        const p0 = st.n;
        for (c.lo..c.hi) |li| {
            const ly = &m.layers[li];
            // The compressed rows of the nearest layer at or before this one that holds them.
            var src: ?usize = null;
            if (ly.ratio > 0) for (c.lo..li + 1) |l| {
                if (st.layers[l].comp != null) src = l;
            };
            const comp_rows = if (src) |l| try sliceRows(&sc, st.layers[l].comp.?.buf, 0, st.layers[l].comp.?.used) else none;
            const nc: usize = if (src) |l| st.layers[l].comp.?.used else 0;
            const k: usize = if (src != null) 512 else 0;
            for ([_]usize{ 1, 2, 4, 6 }) |R| {
                const q = try Rig.randn(&sc, &.{ @intCast(R), HEADS, HD }, 100 + R);
                const kv = try Rig.randn(&sc, &.{ @intCast(R), HD }, 200 + R);
                const ring: dsv41.Ring = .{ .buf = try ringWith(&sc, st.layers[li].win.buf, kv, p0), .cap = 144, .width = HD };
                const picks = try gpa.alloc(i32, R * k);
                defer gpa.free(picks);
                // Distinct picks, the rest unset: a row's tail, and on odd rows holes between picks too.
                for (0..R) |r| for (0..k) |j| {
                    picks[r * k + j] = if (j < nc and !(r % 2 == 1 and j % 7 == 3)) @intCast((j * 7919 + r * 131) % nc) else -1;
                };
                const topk = if (k > 0) try sc.ints(picks, &.{ @intCast(R), @intCast(k) }) else none;
                const want = try dsv41.hostF32(try dsv41.attnDecodeSplit(&sc, m, ly, q, ring, comp_rows, topk, p0, SCALE, dsv41.attnSplits(WINDOW + k), null));
                defer gpa.free(want);
                const p = (try partials(&sc, kern, m, q, ring, comp_rows, topk, p0, SCALE)) orelse return error.Declined;
                const got = try dsv41.hostF32(try mergeToday(&sc, m, p, ly.sink, p0));
                defer gpa.free(got);
                const hq = try dsv41.hostF32(q);
                defer gpa.free(hq);
                const hr = try dsv41.hostF32(ring.buf);
                defer gpa.free(hr);
                const hc = if (k > 0) try dsv41.hostF32(comp_rows) else try gpa.alloc(f32, 0);
                defer gpa.free(hc);
                const hs = try dsv41.hostF32(ly.sink);
                defer gpa.free(hs);
                const truth = try truthHeads(gpa, hq, hr, hc, picks, hs, R, k, nc, p0);
                defer gpa.free(truth);
                var label: [96]u8 = undefined;
                try decode.expectNoWorse(try std.fmt.bufPrint(&label, "attn layer {d} ctx {d} R={d}", .{ li, c.ctx, R }), got, want, truth);
            }
        }
    }
}

test "dsv41 dec attn: the declared launch is the wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var m = try todayModel();
    defer freeTodayModel(&m);
    var sc = Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    inline for (.{ .{ 8, 6, 64, 512 }, .{ 8, 1, 16, 100 }, .{ 4, 3, 32, 0 } }) |c| {
        const cs = try a.create(kit.Case(DecAttn));
        defer {
            cs.deinit();
            a.destroy(cs);
        }
        cs.* = .init(a, &launch_decl, .{ .s = c[0] });
        try testing.expect(try cs.prepare(.{ .rows = c[1], .n = c[2], .k = c[3] }, .random, 7));
        try cs.run();
        const g = cs.p.g;
        const win: dsv41.Ring = .{ .buf = cs.args[1], .cap = g.cap, .width = HD };
        const p = (try run(&sc, kern, &m, cs.args[0], win, if (c[3] > 0) cs.args[2] else none, cs.args[3], g.p0, cs.args[5])) orelse return error.Declined;
        try testing.expectEqual(@as(c_int, c[0]), p.splits);
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(a);
        for ([_]A{ p.acc, p.m, p.l }) |x| {
            const h = try dsv41.hostF32(x);
            defer a.free(h);
            try got.appendSlice(a, std.mem.sliceAsBytes(h));
        }
        try testing.expectEqualSlices(u8, cs.rig.outputBytes(), got.items);
    }
}

/// A dependent chain of decode steps for `decode.chainUs`: step i is fed by step i - 1, and a chain's arrays are freed
/// when the next one starts, so a long chain does not pin the allocator's buffers.
const TodayChain = struct {
    gpa: std.mem.Allocator,
    sc: Scope,
    m: *const dsv41.Dsv41Model,
    ly: *const dsv41.Layer,
    rig: *const Rig,
    q: A,

    pub fn step(self: *TodayChain, i: usize) !A {
        if (i == 0) {
            self.sc.deinit();
            self.sc = Scope.init(self.gpa, mlx.gpuStream());
            self.q = self.rig.q;
        }
        self.q = try dsv41.attnDecodeSplit(&self.sc, self.m, self.ly, self.q, self.rig.win(), self.rig.comps[i % self.rig.copies], self.rig.picks[i % self.rig.steps], self.rig.p0, SCALE, dsv41.attnSplits(WINDOW + self.rig.k), null);
        return self.q;
    }
};

/// This kernel's chain: the next step's scale input is the last step's weight sums, so each launch waits for the one
/// before it (the kernel reads only element 0 of it; the values are not meaningful).
const NewChain = struct {
    gpa: std.mem.Allocator,
    sc: Scope,
    kern: mlx.mlx_fast_metal_kernel,
    m: *const dsv41.Dsv41Model,
    rig: *const Rig,
    scal0: A,
    scal: A,

    pub fn step(self: *NewChain, i: usize) !A {
        if (i == 0) {
            self.sc.deinit();
            self.sc = Scope.init(self.gpa, mlx.gpuStream());
            self.scal = self.scal0;
        }
        const p = (try run(&self.sc, self.kern, self.m, self.rig.q, self.rig.win(), self.rig.comps[i % self.rig.copies], self.rig.picks[i % self.rig.steps], self.rig.p0, self.scal)) orelse return error.Declined;
        self.scal = p.l;
        return p.acc;
    }
};

/// The launch floor: a one-thread kernel in a dependent chain.
const NoopChain = struct {
    gpa: std.mem.Allocator,
    sc: Scope,
    kern: mlx.mlx_fast_metal_kernel,
    x0: A,
    x: A,

    pub fn step(self: *NoopChain, i: usize) !A {
        if (i == 0) {
            self.sc.deinit();
            self.sc = Scope.init(self.gpa, mlx.gpuStream());
            self.x = self.x0;
        }
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{1}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
        const iv = mlx.mlx_vector_array_new_data(&[_]A{self.x}, 1);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, self.kern, iv, cfg, self.sc.s));
        var o = mlx.mlx_array_new();
        self.x = try self.sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
        return self.x;
    }
};

test "dsv41 dec attn: ubench" {
    if (!decode.ubenchOn()) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var m = try todayModel();
    defer freeTodayModel(&m);
    var ly: dsv41.Layer = undefined;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const noop = try dsv41.buildKernel("dsv41_dec_noop", &.{"x"}, &.{"out"}, "if (thread_position_in_grid.x == 0) out[0] = x[0] + 1.0f;", "");
    defer _ = mlx.mlx_fast_metal_kernel_free(noop);
    {
        var keep = Scope.init(gpa, mlx.gpuStream());
        defer keep.deinit();
        const x0 = try keep.keep(mlx.mlx_array_new_data(&[_]f32{1}, &[_]c_int{1}, 1, .float32));
        var chain: NoopChain = .{ .gpa = gpa, .sc = Scope.init(gpa, mlx.gpuStream()), .kern = noop, .x0 = x0, .x = x0 };
        defer chain.sc.deinit();
        std.debug.print("[dec-ubench] attn: a dependent one-thread launch is {d:.1} us a step\n", .{try decode.chainUs(&chain, 400)});
    }
    // Picks over a compressed-row buffer of 1k, 32k and 128k rows (copies of it cycle over 2 GB), then window only.
    for ([_]usize{ 512, 0 }) |k| for ([_]usize{ 1, 6 }) |R| for ([_]usize{ 1024, 32768, 131072 }) |ctx| {
        if (k == 0 and ctx != 1024) continue;
        var keep = Scope.init(gpa, mlx.gpuStream());
        defer keep.deinit();
        const copies = @max(2, @min(2000, (2 << 30) / (ctx * HD * 2)));
        const rig = try Rig.init(&keep, R, ctx, k, copies, 400, ctx, 5);
        defer rig.deinit();
        ly.sink = rig.sink;
        var today: TodayChain = .{ .gpa = gpa, .sc = Scope.init(gpa, mlx.gpuStream()), .m = &m, .ly = &ly, .rig = rig, .q = rig.q };
        defer today.sc.deinit();
        const scal0 = try keep.keep(mlx.mlx_array_new_data(&[_]f32{SCALE}, &[_]c_int{1}, 1, .float32));
        var new: NewChain = .{ .gpa = gpa, .sc = Scope.init(gpa, mlx.gpuStream()), .kern = kern, .m = &m, .rig = rig, .scal0 = scal0, .scal = scal0 };
        defer new.sc.deinit();
        const chain_us = try decode.chainUs(&today, 400);
        const us = try decode.chainUs(&new, 400);
        var name: [48]u8 = undefined;
        decode.report(try std.fmt.bufPrint(&name, "attn K={d} ctx={d}", .{ k, ctx }), R, us, @floatFromInt(R * k * HD * 2 + WINDOW * HD * 2 + R * HEADS * HD * 2), chain_us);
    };
}

/// Random mxfp8 wo_a weights `[groups, 1024, 8 * 512]` (never S.1111.111, scales around 1) and the inverse periods.
fn woaWeights(sc: *Scope, seed: u64) ![3]A {
    const gpa = testing.allocator;
    const G = 8;
    const OL = 1024;
    const KW = 8 * HD;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const words = try gpa.alloc(u32, G * OL * KW / 4);
    defer gpa.free(words);
    for (words) |*w| {
        var b: [4]u8 = undefined;
        for (&b) |*x| {
            x.* = rnd.int(u8);
            if (x.* & 0x7f == 0x7f) x.* -= 1;
        }
        w.* = std.mem.readInt(u32, &b, .little);
    }
    const sc8 = try gpa.alloc(u8, G * OL * KW / 32);
    defer gpa.free(sc8);
    for (sc8) |*e| e.* = rnd.intRangeAtMost(u8, 118, 124);
    var per: [32]f32 = undefined;
    for (&per, 0..) |*v, i| v.* = std.math.pow(f32, 10000, @as(f32, @floatFromInt(2 * i)) / 64.0);
    return .{
        try sc.keep(mlx.mlx_array_new_data(words.ptr, &[_]c_int{ G, OL, KW / 4 }, 3, .uint32)),
        try sc.keep(mlx.mlx_array_new_data(sc8.ptr, &[_]c_int{ G, OL, KW / 32 }, 3, .uint8)),
        try sc.keep(mlx.mlx_array_new_data(&per, &[_]c_int{32}, 1, .float32)),
    };
}

/// The merged split `pm` against the host's float64 merge of the plain kernel's partials `pl` with `sink`: within 1e-5
/// of each head's scale, its max 0 and its sum 1 (the first misses printed).
fn mergedIsHostMerge(sc: *Scope, pl: Partials, pm: Partials, sink: A) !void {
    _ = sc;
    const gpa = testing.allocator;
    const acc = try dsv41.hostF32(pl.acc);
    defer gpa.free(acc);
    const mm = try dsv41.hostF32(pl.m);
    defer gpa.free(mm);
    const ll = try dsv41.hostF32(pl.l);
    defer gpa.free(ll);
    const sk = try dsv41.hostF32(sink);
    defer gpa.free(sk);
    const got = try dsv41.hostF32(pm.acc);
    defer gpa.free(got);
    const gm = try dsv41.hostF32(pm.m);
    defer gpa.free(gm);
    const gl = try dsv41.hostF32(pm.l);
    defer gpa.free(gl);
    const S: usize = @intCast(pl.splits);
    const H = sk.len;
    const R = got.len / (H * HD);
    var misses: usize = 0;
    for (0..R) |i| for (0..H) |h| {
        const base = (i * H + h) * S;
        var M: f64 = sk[h];
        for (0..S) |j| M = @max(M, mm[base + j]);
        var total = @exp(sk[h] - M);
        var w: [MAX_SPLITS]f64 = undefined;
        for (0..S) |j| {
            w[j] = if (ll[base + j] > 0) @exp(mm[base + j] - M) else 0;
            total += ll[base + j] * w[j];
        }
        var scale: f64 = 1e-30;
        var o: [HD]f64 = @splat(0);
        for (0..HD) |d| {
            for (0..S) |j| o[d] += acc[(base + j) * HD + d] * w[j];
            o[d] /= total;
            scale = @max(scale, @abs(o[d]));
        }
        for (0..HD) |d| {
            const g = got[(i * H + h) * HD + d];
            if (!(@abs(g - o[d]) <= 1e-5 * scale)) {
                if (misses < 8) std.debug.print("[attn merge] row {d} head {d} ch {d}: merged {e} host {e} (max {e} sum {e})\n", .{ i, h, d, g, o[d], gm[i * H + h], gl[i * H + h] });
                misses += 1;
            }
        }
        if (gm[i * H + h] != 0 or gl[i * H + h] != 1) {
            if (misses < 8) std.debug.print("[attn merge] row {d} head {d}: max {e} sum {e}\n", .{ i, h, gm[i * H + h], gl[i * H + h] });
            misses += 1;
        }
    };
    if (misses > 0) std.debug.print("[attn merge] {d} misses of {d}\n", .{ misses, R * H * (HD + 1) });
    try testing.expectEqual(@as(usize, 0), misses);
}

test "dsv41 dec attn merge: wo_a over the merged split is wo_a over the splits (rel rms 1e-3), rows 1, 2, 4, 6; the partials are the plain kernel's; the sink counted twice fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const wo_a = @import("wo_a.zig");
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = WINDOW;
    m.act = .bfloat16;
    const kern = try build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const km = try buildMerge();
    defer _ = mlx.mlx_fast_metal_kernel_free(km);
    const kw = try wo_a.build();
    defer _ = mlx.mlx_fast_metal_kernel_free(kw);
    const counter = try sc.keep(newCounter());
    // A counter each wo_a launch: launches without a dependency between them may run at once.
    const wcounters = [3]A{ try sc.keep(wo_a.newCounter()), try sc.keep(wo_a.newCounter()), try sc.keep(wo_a.newCounter()) };
    const nosink = try sc.keep(noSink(HEADS));
    const wts = try woaWeights(&sc, 77);
    var worst: f64 = 0;
    var least_bad: f64 = 1e9;
    for ([_]usize{ 512, 100, 0 }) |k| for ([_]usize{ 1, 2, 4, 6 }) |R| for ([_]usize{ 300, 5000 }) |ctx| {
        const rig = try Rig.init(&sc, R, ctx, k, 1, 1, ctx, 11 + R + k);
        defer rig.deinit();
        const sink = try sc.reshape(rig.sink, &.{HEADS});
        const meta = try sc.ints(&[_]i32{ @intCast(ctx), @intCast(R), @intCast(k), WINDOW, 144, @intCast(ctx) }, &.{6});
        const pl = (try partials(&sc, kern, &m, rig.q, rig.win(), rig.comps[0], rig.picks[0], rig.p0, SCALE)) orelse return error.Declined;
        const want = (try wo_a.launch(&sc, kw, 64, .bfloat16, .{ pl.acc, pl.m, pl.l, sink, meta, wts[2], wts[0], wts[1] }, wcounters[0], null)) orelse return error.Declined;
        const pm = (try partialsMerged(&sc, km, &m, rig.q, rig.win(), rig.comps[0], rig.picks[0], rig.p0, SCALE, sink, counter)) orelse return error.Declined;
        try testing.expectEqual(@as(c_int, 1), pm.splits);
        try mergedIsHostMerge(&sc, pl, pm, sink);
        const got = (try wo_a.launch(&sc, kw, 64, .bfloat16, .{ pm.acc, pm.m, pm.l, nosink, meta, wts[2], wts[0], wts[1] }, wcounters[1], null)) orelse return error.Declined;
        // The planted bug: wo_a weighs the layer's sink against the merged split too (the sink counted twice).
        const bad = (try wo_a.launch(&sc, kw, 64, .bfloat16, .{ pm.acc, pm.m, pm.l, sink, meta, wts[2], wts[0], wts[1] }, wcounters[2], null)) orelse return error.Declined;
        const w = try dsv41.hostF32(want);
        defer gpa.free(w);
        const g = try dsv41.hostF32(got);
        defer gpa.free(g);
        const b = try dsv41.hostF32(bad);
        defer gpa.free(b);
        for (g) |v| if (!std.math.isFinite(v)) return error.NonFinite;
        const e = decode.relRms(g, w);
        const eb = decode.relRms(b, w);
        std.debug.print("[attn merge] k {d} R {d} ctx {d}: oa rel rms merged {e:.2}, sink twice {e:.2}\n", .{ k, R, ctx, e, eb });
        try testing.expect(e <= 1e-3);
        try testing.expect(eb > 1e-2);
        worst = @max(worst, e);
        least_bad = @min(least_bad, eb);
    };
    std.debug.print("[attn merge] worst merged {e:.2}, least planted {e:.2}\n", .{ worst, least_bad });
}

test "dsv41 dec attn merge: the declared launch is the wrapper's, byte for byte; the registrations add" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !device.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    try testing.expectEqual(@as(usize, 4), set.kernels.items.len);
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = WINDOW;
    m.act = .bfloat16;
    var sc = Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const km = try buildMerge();
    defer _ = mlx.mlx_fast_metal_kernel_free(km);
    inline for (.{ .{ 8, 6, 64, 512 }, .{ 8, 1, 16, 100 }, .{ 4, 3, 32, 0 } }) |c| {
        const cs = try a.create(kit.Case(MergeAttn));
        defer {
            cs.deinit();
            a.destroy(cs);
        }
        cs.* = .init(a, &merge_decl, .{ .s = c[0] });
        try testing.expect(try cs.prepare(.{ .rows = c[1], .n = c[2], .k = c[3] }, .random, 7));
        try cs.run();
        const declared = try a.dupe(u8, cs.rig.outputBytes());
        defer a.free(declared);
        const g = cs.p.g;
        const win: dsv41.Ring = .{ .buf = cs.args[1], .cap = g.cap, .width = HD };
        const p = (try runWith(&sc, km, &m, cs.args[0], win, if (c[3] > 0) cs.args[2] else none, cs.args[3], g.p0, cs.args[5], .{ cs.args[6], cs.args[7] })) orelse return error.Declined;
        try testing.expectEqual(@as(c_int, 1), p.splits);
        const h = try dsv41.hostF32(p.acc);
        defer a.free(h);
        const ns = MergeAttn.splitCount(cs.p, .{ .rows = c[1], .n = c[2], .k = c[3] });
        try testing.expectEqualSlices(u8, declared[4 * ns ..][0 .. 4 * h.len], std.mem.sliceAsBytes(h));
    }
}

