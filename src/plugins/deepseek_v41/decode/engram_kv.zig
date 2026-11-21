//! An Engram layer's wkv over its decode rows straight from the table's codes (`dec_engram_kv.metal`, `fuse.engram`):
//! the rows' dequant, their cast and the mxfp8 projection in one launch, the rows never in memory. The rows are
//! picked by id out of any byte buffer (`Geo`): the rows copied out of the table, or the whole resident table
//! wrapped as one GPU buffer (`fuse.engram_gather`), so the gather is the kernel's own load.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;

pub const SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_engram_kv.metal");
/// The e2m1, e4m3 and e8m0 reads (`fp4_scale.metal` and the mxfp8 pair decode).
pub const HEADER = @import("../../../sdk/blocks/qmm/mxfp8_qmv.zig").HEADER;
pub const INS = [_][*:0]const u8{ "codes", "tsc", "ids", "geo", "w", "wsc" };
/// A threadgroup's output rows (the kernel's SG x RT), its threads, and the inputs it stages at a time (CH).
pub const cols = 16;
pub const threads = 256;
pub const chunk = 1024;

/// The table rows' geometry: `hd` inputs a row, `tb` bits a code (4: e2m1, 8: e4m3), an e8m0 scale per 32.
pub const Table = struct { hd: usize, tb: usize };

/// Where table row `id`'s codes and scales sit in their buffers: byte offset of row 0 and the row stride of each.
pub const Geo = struct {
    w_base: u64,
    w_stride: u64,
    s_base: u64,
    s_stride: u64,

    /// The kernel's `geo` input: each base as two 32-bit halves, then its stride.
    pub fn words(g: Geo) [6]u32 {
        return .{ @truncate(g.w_base), @truncate(g.w_base >> 32), @intCast(g.w_stride), @truncate(g.s_base), @truncate(g.s_base >> 32), @intCast(g.s_stride) };
    }
};

/// Built at first use (one per `-Dunits` unit, as the SDK's mxfp8 qmv).
var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn dims(a: A, axis: usize) usize {
    return @intCast(mlx.getShape(a)[axis]);
}

fn is(a: A, dt: mlx.mlx_dtype, want: []const usize) bool {
    const sh = mlx.getShape(a);
    if (a.ctx == null or mlx.mlx_array_dtype(a) != dt or sh.len != want.len) return false;
    for (sh, want) |d, x| if (d != ci(x)) return false;
    return true;
}

/// kv `[rows, N]` bf16 = x . W^T, row r of x the `K / hd` table rows `ids[r * K / hd ..]` (u32), each read out of the
/// byte buffers `codes` and `tsc` where `geo` (`Geo.words`, u32 [6]) puts it; W mxfp8 `w` `[N, K / 4]` u32 with e8m0
/// `wsc` `[N, K / 32]`. The caller vouches that every id's bytes lie inside the buffers (4-aligned rows of codes).
/// Null where the kernel does not serve: not a GPU stream, more rows than a decode step, K off the stage width or the
/// table rows, N off the threadgroup's.
pub fn launch(sc: *dsv41.Scope, t: Table, rows: usize, codes: A, tsc: A, ids: A, geo: A, w: A, wsc: A) !?A {
    if (!mlx.streamIsGpu(sc.s) or rows < 1 or rows > decode.max_rows) return null;
    if (!(t.tb == 4 or t.tb == 8) or t.hd == 0 or t.hd % 32 != 0) return null;
    if (mlx.getShape(w).len != 2 or codes.ctx == null or tsc.ctx == null) return null;
    const N = dims(w, 0);
    const K = dims(w, 1) * 4;
    if (K == 0 or K % chunk != 0 or K % t.hd != 0 or N == 0 or N % cols != 0) return null;
    if (!is(ids, .uint32, &.{rows * K / t.hd}) or !is(geo, .uint32, &.{6})) return null;
    if (!is(w, .uint32, &.{ N, K / 4 }) or !is(wsc, .uint8, &.{ N, K / 32 })) return null;
    if (kernel == null) kernel = try dsv41.buildKernel("dsv41_dec_engram_kv", &INS, &.{"kv"}, SOURCE, HEADER);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(N) }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(threads * (N / cols)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    inline for (.{ .{ "R", rows }, .{ "K", K }, .{ "N", N }, .{ "HD", t.hd }, .{ "TB", t.tb } }) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, a[0], ci(a[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ codes, tsc, ids, geo, w, wsc };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

pub const ROWS_SOURCE = kernelSource("plugins/deepseek_v41/kernels/engram_rows.metal");
pub const ROWS_INS = [_][*:0]const u8{ "codes", "tsc", "ids", "geo" };
/// Threads a threadgroup of the rows' gather (16 values a thread).
pub const rows_threads = 256;
var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `ids.len` table rows `[n, hd]` bf16, each gathered by id out of `codes` / `tsc` at `geo` and dequantized (a
/// prompt chunk's Engram rows, the GEMM's input, from the resident table). Null off a GPU stream or off the layouts.
pub fn gatherRows(sc: *dsv41.Scope, t: Table, codes: A, tsc: A, ids: A, geo: A) !?A {
    if (!mlx.streamIsGpu(sc.s) or !(t.tb == 4 or t.tb == 8) or t.hd == 0 or t.hd % 32 != 0) return null;
    if (codes.ctx == null or tsc.ctx == null or mlx.getShape(ids).len != 1 or mlx.mlx_array_dtype(ids) != .uint32 or !is(geo, .uint32, &.{6})) return null;
    const n = dims(ids, 0);
    if (n == 0) return null;
    if (rows_kernel == null) rows_kernel = try dsv41.buildKernel("dsv41_engram_rows", &ROWS_INS, &.{"out"}, ROWS_SOURCE, HEADER);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(n), ci(t.hd) }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(n * t.hd / 16), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, rows_threads, 1, 1));
    inline for (.{ .{ "HD", t.hd }, .{ "TB", t.tb } }) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, a[0], ci(a[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const ins = [_]A{ codes, tsc, ids, geo };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, rows_kernel.?, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return try sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

// ── the kernel contract ──

const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const cv = @import("../contract.zig");
const moe_fp4 = @import("moe_fp4.zig");

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

/// The release's Engram rows: 256 inputs, 24 of them a decode row (6144).
const HD = 256;

fn e8m0(b: f64) f64 {
    return if (b == 255) kit.nan else std.math.pow(f64, 2, b - 127);
}

/// e2m1 magnitudes by code bits 0..2; bit 3 is the sign.
const e2m1 = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

/// The e2m1 code of the integer magnitude `c` (1, 2, 3, 4 or 6).
fn e2m1Int(c: u32) u32 {
    return switch (c) {
        1 => 2,
        2 => 4,
        3 => 5,
        4 => 6,
        else => 7,
    };
}

/// Integers 0..8 as e4m3 codes (all exact).
const e4m3_int = [_]u8{ 0x00, 0x38, 0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58 };

const K_ = struct {
    pub const Params = struct { tb: usize = 4 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 1 or s.rows > decode.max_rows or s.k == 0 or s.k % chunk != 0 or s.n == 0 or s.n % cols != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(threads * (s.n / cols)), 1, 1 }, .tg = .{ threads, 1, 1 } };
        g.out(&.{ ci(s.rows), ci(s.n) }, .bfloat16);
        g.int("R", s.rows);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("HD", HD);
        g.int("TB", p.tb);
        g.dtype("T", .bfloat16);
        return g;
    }

    /// Row-of-x `r`'s codes for the values `v` (integers the code holds, or random codes when null), under `scale`.
    fn packRow(tb: usize, codes: []f64, tsc: []f64, r: usize, K: usize, v: ?[]const f64, scale: ?f64, rng: std.Random) void {
        const tw = HD * tb / 32;
        for (0..K / HD) |j| {
            const row = r * (K / HD) + j;
            for (0..HD / 32) |g| tsc[row * (HD / 32) + g] = scale orelse @floatFromInt(120 + rng.uintLessThan(u32, 7));
            for (0..tw) |wi| {
                var word: u32 = 0;
                const per: usize = 32 / tb;
                for (0..per) |b| {
                    const i = j * HD + wi * per + b;
                    const c: u32 = if (v) |vals| blk: {
                        const mag: u32 = @intFromFloat(@abs(vals[i]));
                        const sign: u32 = if (vals[i] < 0) 1 else 0;
                        break :blk if (tb == 4) e2m1Int(mag) | sign << 3 else @as(u32, e4m3_int[mag]) | sign << 7;
                    } else if (tb == 4) rng.uintLessThan(u32, 16) else rng.uintLessThan(u32, 2) << 7 | (5 + rng.uintLessThan(u32, 5)) << 3 | rng.uintLessThan(u32, 8);
                    word |= c << @intCast(tb * b);
                }
                codes[row * tw + wi] = @floatFromInt(word);
            }
        }
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const tb = k.p.tb;
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const tr = R * K / HD;
        const codes = try k.add(.uint32, &.{ tr, HD * tb / 32 });
        const tsc = try k.add(.uint8, &.{ tr, HD / 32 });
        // The rows read through a shuffle of their ids, as a gather out of the whole table reads them (the stress
        // fill keeps each row's own, its mirrored halves in one row of x).
        const ids = try k.add(.uint32, &.{tr});
        const geo = try k.add(.uint32, &.{6});
        for (ids, 0..) |*v, i| v.* = @floatFromInt(i);
        if (f != .stress) rng.shuffle(f64, ids);
        for (geo, (Geo{ .w_base = 0, .w_stride = HD * tb / 8, .s_base = 0, .s_stride = HD / 32 }).words()) |*g, v| g.* = @floatFromInt(v);
        const w = try k.add(.uint32, &.{ N, K / 4 });
        const wsc = try k.add(.uint8, &.{ N, K / 32 });
        const perm = try k.scratch().alloc(usize, K / 2);
        defer k.scratch().free(perm);
        switch (f) {
            .random, .nonfinite => {
                for (0..R) |r| packRow(tb, codes, tsc, r, K, null, null, rng);
                try moe_fp4.fillShared(rng, w, wsc, N, K, f, perm, k.scratch());
                // A NaN weight (e4m3 0x7f, MLX's NaN code) on every fifth output row: its outputs are NaN for every row of x.
                if (f == .nonfinite) {
                    var n: usize = 1;
                    while (n < N) : (n += 5) {
                        const word: u32 = @intFromFloat(w[n * (K / 4)]);
                        w[n * (K / 4)] = @floatFromInt((word & 0xFFFFFF00) | 0x7F);
                    }
                }
            },
            .stress => {
                // Each row of x mirrored halves of integers e2m1 / e4m3 hold (8 entries one apart) against the
                // weights' mirrored integers under unit scales: each output the sum of 8 negated ones, exact.
                const x = try k.scratch().alloc(f64, K);
                defer k.scratch().free(x);
                for (perm, 0..) |*q, i| q.* = i;
                rng.shuffle(usize, perm);
                const h = K / 2;
                const big = [_]f64{ 1, 2, 3, 4, 6 };
                for (0..R) |r| {
                    for (0..h) |i| {
                        x[i] = if (i < 8) @floatFromInt(rng.intRangeAtMost(u32, 1, 3)) else big[rng.uintLessThan(usize, big.len)];
                        x[h + perm[i]] = x[i] + @as(f64, if (i < 8) 1 else 0);
                    }
                    packRow(tb, codes, tsc, r, K, x, 127, rng);
                }
                try moe_fp4.fillShared(rng, w, wsc, N, K, f, perm, k.scratch());
            },
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n;
    }

    fn xAt(k: anytype, r: usize, i: usize) f64 {
        const tb = k.p.tb;
        const K = k.shape.k;
        const row: usize = @intFromFloat(k.in(2)[r * (K / HD) + i / HD]);
        const p = i % HD;
        const s = e8m0(k.in(1)[row * (HD / 32) + p / 32]);
        const per: usize = 32 / tb;
        const word: u32 = @intFromFloat(k.in(0)[row * (HD * tb / 32) + p / per]);
        const c = (word >> @intCast(tb * (p % per))) & ((@as(u32, 1) << @intCast(tb)) - 1);
        const v = if (tb == 4) (if (c & 8 != 0) -e2m1[c & 7] else e2m1[c & 7]) else kc.host.e4m3(@intCast(c));
        return v * s;
    }

    /// W's input i of row n as the kernel reads it: MLX's e4m3 (S.1111.111 NaN, which the shared host table reads as
    /// 480) times its e8m0 exponent.
    fn wAt(words: []const f64, exps: []const f64, n: usize, i: usize, K: usize) f64 {
        const word: u32 = @intFromFloat(words[n * (K / 4) + i / 4]);
        const byte: u8 = @truncate(word >> @intCast(8 * (i % 4)));
        if (byte & 0x7f == 0x7f) return kit.nan;
        return kc.host.e4m3(byte) * e8m0(exps[n * (K / 32) + i / 32]);
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        const R = k.shape.rows;
        const N = k.shape.n;
        const K = k.shape.k;
        const a = k.scratch();
        const t = try a.alloc(f64, K);
        defer a.free(t);
        const x = try a.alloc(f64, R * K);
        defer a.free(x);
        const terms = try a.alloc(f64, K);
        defer a.free(terms);
        for (0..R) |r| for (0..K) |i| {
            x[r * K + i] = xAt(k, r, i);
        };
        for (0..N) |n| {
            for (0..K) |i| t[i] = wAt(k.in(4), k.in(5), n, i, K);
            for (0..R) |r| {
                var m: f64 = 0;
                for (0..K) |i| {
                    terms[i] = x[r * K + i] * t[i];
                    m += @abs(terms[i]);
                }
                out[r * N + n] = sum.of(terms);
                mag[r * N + n] = m;
            }
        }
    }
};

/// The rows' gather: `s.rows` x `s.n` table rows of 256 (rows of x by n-gram rows) through shuffled ids, dequantized.
const RowsK = struct {
    pub const Params = struct { tb: usize = 4 };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.k != HD) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.rows * s.n * HD / 16), 1, 1 }, .tg = .{ rows_threads, 1, 1 } };
        g.out(&.{ ci(s.rows * s.n), HD }, .bfloat16);
        g.int("HD", HD);
        g.int("TB", p.tb);
        g.dtype("T", .bfloat16);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const tb = k.p.tb;
        const n = k.shape.rows * k.shape.n;
        const codes = try k.add(.uint32, &.{ n, HD * tb / 32 });
        const tsc = try k.add(.uint8, &.{ n, HD / 32 });
        const ids = try k.add(.uint32, &.{n});
        const geo = try k.add(.uint32, &.{6});
        for (0..n) |r| K_.packRow(tb, codes, tsc, r, HD, null, null, rng);
        for (ids, 0..) |*v, i| v.* = @floatFromInt(i);
        rng.shuffle(f64, ids);
        for (geo, (Geo{ .w_base = 0, .w_stride = HD * tb / 8, .s_base = 0, .s_stride = HD / 32 }).words()) |*g, v| g.* = @floatFromInt(v);
        // NaN scales (e8m0 0xff) on every seventh row: its values are NaN.
        if (f == .nonfinite) {
            var r: usize = 3;
            while (r < n) : (r += 7) @memset(tsc[r * (HD / 32) ..][0 .. HD / 32], 255);
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows * s.n * HD;
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, sum: Sum) !void {
        _ = sum;
        const tb = k.p.tb;
        const per: usize = 32 / tb;
        for (0..k.shape.rows * k.shape.n) |i| {
            const row: usize = @intFromFloat(k.in(2)[i]);
            for (0..HD) |p| {
                const s = e8m0(k.in(1)[row * (HD / 32) + p / 32]);
                const word: u32 = @intFromFloat(k.in(0)[row * (HD * tb / 32) + p / per]);
                const c = (word >> @intCast(tb * (p % per))) & ((@as(u32, 1) << @intCast(tb)) - 1);
                const v = if (tb == 4) (if (c & 8 != 0) -e2m1[c & 7] else e2m1[c & 7]) else kc.host.e4m3(@intCast(c));
                out[i * HD + p] = v * s;
                mag[i * HD + p] = @abs(v * s);
            }
        }
    }
};

fn depth(s: Shape) usize {
    return s.k;
}

const kv_l = cv.launch("dsv41_dec_engram_kv", &INS, &.{"kv"}, SOURCE, HEADER, 4 * decode.max_rows * chunk, threads, false);
const rows_l = cv.launch("dsv41_engram_rows", &ROWS_INS, &.{"out"}, ROWS_SOURCE, HEADER, 0, rows_threads, false);

/// Served shapes cut the output width to a few threadgroups (each runs the same code at its own rows); the reduction
/// length is the release's (24 table rows of 256), the rows 1..6. One case a table format.
pub fn register(set: *kc.Set) !void {
    const cases = [_]struct { [:0]const u8, K_.Params }{
        .{ "dsv41 dec engram_kv (mxfp4 table rows, mxfp8 wkv)", .{ .tb = 4 } },
        .{ "dsv41 dec engram_kv (mxfp8 table rows, mxfp8 wkv)", .{ .tb = 8 } },
    };
    for (cases) |c| try kit.add(set, K_, &kv_l, c[1], .{
        .name = c[0],
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{ .{}, .{ .size = cols, .step = cols, .min = cols }, .{ .size = chunk, .step = chunk, .min = chunk } },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 6144 }, .{ .rows = 3, .n = 16, .k = 6144 }, .{ .rows = decode.max_rows, .n = 32, .k = 6144 } },
        .sums = .{ .f16 = 30, .bf16 = 30, .why = "mirrored halves: an exact sum of -8 behind partial sums near 10^5" },
    });
    const rows = [_]struct { [:0]const u8, RowsK.Params }{
        .{ "dsv41 engram_rows (mxfp4 table rows gathered by id to bf16)", .{ .tb = 4 } },
        .{ "dsv41 engram_rows (mxfp8 table rows gathered by id to bf16)", .{ .tb = 8 } },
    };
    for (rows) |c| try kit.add(set, RowsK, &rows_l, c[1], .{
        .name = c[0],
        .accum = .none,
        .depth = depthOne,
        .out = .bf16,
        .served = &.{ .{ .rows = 1, .n = 24, .k = HD }, .{ .rows = 64, .n = 24, .k = HD } },
    });
}

fn depthOne(_: Shape) usize {
    return 1;
}

test "dsv41 dec engram_kv: the contract's launch is the wrapper's, byte for byte (mxfp4 and mxfp8 table rows, R = 1..6)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]usize{ 4, 8 }) |tb| for (1..decode.max_rows + 1) |R| {
        var c = kit.Case(K_).init(a, &kv_l, .{ .tb = tb });
        defer c.deinit();
        try std.testing.expect(try c.prepare(.{ .rows = R, .n = 32, .k = 6144 }, .random, 11));
        try c.run();
        const mine = try a.dupe(u8, c.outBytes());
        defer a.free(mine);
        var sc = dsv41.Scope.init(a, c.rig.stream());
        defer sc.deinit();
        const y = (try launch(&sc, .{ .hd = HD, .tb = tb }, R, c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5])) orelse return error.Declined;
        const theirs = try c.rig.bytesOf(&.{y});
        try std.testing.expectEqualSlices(u8, mine, theirs);
    };
}
