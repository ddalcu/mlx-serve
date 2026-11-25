//! An Engram layer's wkv over its decode rows straight from the table's codes (`dec_engram_kv.metal`, `engram_ids`):
//! the rows' dequant, their cast and the mxfp8 projection in one launch, the rows never in memory. The rows are
//! picked by id out of any byte buffer (`Geo`): the rows copied out of the table, or the whole resident table
//! wrapped as one GPU buffer (`fuse.wrapHeld`), so the gather is the kernel's own load.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const log = @import("../../../sdk/util/log.zig");

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
/// table rows, N off the threadgroup's. One row runs slot-blocked under `MLX_SERVE_DSV41_ENGRAM_KV_SLOTS` where that
/// kernel serves (`launchSlots`).
pub fn launch(sc: *dsv41.Scope, t: Table, rows: usize, codes: A, tsc: A, ids: A, geo: A, w: A, wsc: A) !?A {
    if (rows == 1 and slotsOn()) if (try launchSlots(sc, t, codes, tsc, ids, geo, w, wsc, null)) |kv| {
        if (!slots_logged.swap(true, .monotonic)) log.info(lev.line(.ENGRAM_KV_SLOTS) ++ ": a decode row's Engram wkv slot-blocked\n", .{});
        return kv;
    };
    return launchRows(sc, t, rows, codes, tsc, ids, geo, w, wsc);
}

pub fn slotsOn() bool {
    return lev.onUnlessZero(.ENGRAM_KV_SLOTS);
}

fn launchRows(sc: *dsv41.Scope, t: Table, rows: usize, codes: A, tsc: A, ids: A, geo: A, w: A, wsc: A) !?A {
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

pub const SLOTS_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_engram_kv_slots.metal");
/// The reads, and the slot kernel's two-sum: `s` the rounded sum, `e` its exact error (MLX's Safe math mode keeps
/// the operations as written).
pub const SLOTS_HEADER = HEADER ++
    \\inline float2 dsv41_two_sum(float a, float b) {
    \\  const float s = a + b;
    \\  const float bb = s - a;
    \\  return float2(s, (a - (s - bb)) + (b - bb));
    \\}
    \\
;
/// The slot kernel reading and filling the contribution cache (`engram_kvc.zig`).
pub const SLOTS_KVC_SOURCE = "#define KVC 1\n" ++ SLOTS_SOURCE;
pub const SLOTS_KVC_INS = INS ++ [_][*:0]const u8{ "plan", "vals" };
/// The slot kernel's output rows a threadgroup (SG x RT) and the table row it reads a half simdgroup at a time.
pub const slot_cols = 32;
pub const slot_hd = 256;
var slots_kernel: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var slots_logged = std.atomic.Value(bool).init(false);

/// The contribution cache's part of one launch: `plan` u32 `[K / 256]`, each slot's entry of `vals` f32
/// `[entries, N]` (bit 31: read it, else compute and store it).
pub const Kvc = struct { plan: A, vals: A };

/// `launch` of one row, slot-blocked (`dec_engram_kv_slots.metal`): each 256-input table row's partial sum reduced on
/// its own and the partials added in slot order, so a partial read from `cache` gives the same bits as computing it.
/// Null where it does not serve: not a GPU stream, table rows other than 256 inputs, K off 512, N off 32.
pub fn launchSlots(sc: *dsv41.Scope, t: Table, codes: A, tsc: A, ids: A, geo: A, w: A, wsc: A, cache: ?Kvc) !?A {
    if (!mlx.streamIsGpu(sc.s) or !(t.tb == 4 or t.tb == 8) or t.hd != slot_hd) return null;
    if (mlx.getShape(w).len != 2 or codes.ctx == null or tsc.ctx == null) return null;
    const N = dims(w, 0);
    const K = dims(w, 1) * 4;
    if (K == 0 or K % 512 != 0 or N == 0 or N % slot_cols != 0) return null;
    if (!is(ids, .uint32, &.{K / t.hd}) or !is(geo, .uint32, &.{6})) return null;
    if (!is(w, .uint32, &.{ N, K / 4 }) or !is(wsc, .uint8, &.{ N, K / 32 })) return null;
    if (cache) |c| if (!is(c.plan, .uint32, &.{K / t.hd}) or mlx.mlx_array_dtype(c.vals) != .float32 or mlx.getShape(c.vals).len != 2 or dims(c.vals, 1) != N) return null;
    const v: usize = @intFromBool(cache != null);
    if (slots_kernel[v] == null) slots_kernel[v] = try if (cache != null)
        dsv41.buildKernel("dsv41_dec_engram_kv_slots_kvc", &SLOTS_KVC_INS, &.{"kv"}, SLOTS_KVC_SOURCE, SLOTS_HEADER)
    else
        dsv41.buildKernel("dsv41_dec_engram_kv_slots", &INS, &.{"kv"}, SLOTS_SOURCE, SLOTS_HEADER);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, ci(N) }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(threads * (N / slot_cols)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    inline for (.{ .{ "K", K }, .{ "N", N }, .{ "HD", t.hd }, .{ "TB", t.tb } }) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, a[0], ci(a[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    var ins: [8]A = .{ codes, tsc, ids, geo, w, wsc, undefined, undefined };
    if (cache) |c| {
        ins[6] = c.plan;
        ins[7] = c.vals;
    }
    const iv = mlx.mlx_vector_array_new_data(&ins, if (cache != null) 8 else 6);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, slots_kernel[v].?, iv, cfg, sc.s));
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
    /// `f32_out`: the sums as f32, not the served bf16; `spread`: random scales over the range real tables and wkv
    /// span (the float64 distance test).
    pub const Params = struct { tb: usize = 4, f32_out: bool = false, spread: bool = false };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows < 1 or s.rows > decode.max_rows or s.k == 0 or s.k % chunk != 0 or s.n == 0 or s.n % cols != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(threads * (s.n / cols)), 1, 1 }, .tg = .{ threads, 1, 1 } };
        const dt: mlx.mlx_dtype = if (p.f32_out) .float32 else .bfloat16;
        g.out(&.{ ci(s.rows), ci(s.n) }, dt);
        g.int("R", s.rows);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("HD", HD);
        g.int("TB", p.tb);
        g.dtype("T", dt);
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
                if (k.p.spread) {
                    for (tsc) |*v| v.* = @floatFromInt(112 + rng.uintLessThan(u32, 24));
                    for (wsc) |*v| v.* = @floatFromInt(112 + rng.uintLessThan(u32, 16));
                }
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

/// The slot kernel (`launchSlots`): one row of x (a gate's rows read as 1), K_'s values and reference; with `kvc`
/// every slot a cache miss stored into its own entry of `vals` (bound, written in place), the cache's computing arm.
const KS = struct {
    pub const Params = struct { tb: usize = 4, kvc: bool = false, f32_out: bool = false, spread: bool = false };

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.k == 0 or s.k % 512 != 0 or s.n == 0 or s.n % slot_cols != 0) return null;
        var g = metal.Geometry{ .grid = .{ ci(threads * (s.n / slot_cols)), 1, 1 }, .tg = .{ threads, 1, 1 } };
        const dt: mlx.mlx_dtype = if (p.f32_out) .float32 else .bfloat16;
        g.out(&.{ 1, ci(s.n) }, dt);
        g.int("K", s.k);
        g.int("N", s.n);
        g.int("HD", HD);
        g.int("TB", p.tb);
        g.dtype("T", dt);
        return g;
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        k.shape.rows = 1;
        try K_.fill(k, f, rng);
        if (!k.p.kvc) return;
        const ns = k.shape.k / HD;
        const plan = try k.add(.uint32, &.{ns});
        for (plan, 0..) |*v, j| v.* = @floatFromInt(j);
        @memset(try k.add(.float32, &.{ ns, k.shape.n }), kit.nan);
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.n;
    }

    pub const compute = K_.compute;
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
const slots_l = cv.launch("dsv41_dec_engram_kv_slots", &INS, &.{"kv"}, SLOTS_SOURCE, SLOTS_HEADER, 2 * 6144, threads, false);
const slots_kvc_l = cv.launch("dsv41_dec_engram_kv_slots_kvc", &SLOTS_KVC_INS, &.{"kv"}, SLOTS_KVC_SOURCE, SLOTS_HEADER, 2 * 6144, threads, false);
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
    const slot_cases = [_]struct { [:0]const u8, KS.Params }{
        .{ "dsv41 dec engram_kv slots (mxfp4 table rows, mxfp8 wkv)", .{ .tb = 4 } },
        .{ "dsv41 dec engram_kv slots (mxfp8 table rows, mxfp8 wkv)", .{ .tb = 8 } },
        .{ "dsv41 dec engram_kv slots, cache misses stored (mxfp4 table rows)", .{ .tb = 4, .kvc = true } },
        .{ "dsv41 dec engram_kv slots, cache misses stored (mxfp8 table rows)", .{ .tb = 8, .kvc = true } },
    };
    for (slot_cases) |c| try kit.add(set, KS, if (c[1].kvc) &slots_kvc_l else &slots_l, c[1], .{
        .name = c[0],
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .in_place = c[1].kvc,
        .tiles = .{ .{}, .{ .size = slot_cols, .step = slot_cols, .min = slot_cols }, .{ .size = 512, .step = 512, .min = 512 } },
        .served = &.{ .{ .rows = 1, .n = 64, .k = 6144 }, .{ .rows = 1, .n = 32, .k = 4096 } },
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
        const y = (try launchRows(&sc, .{ .hd = HD, .tb = tb }, R, c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5])) orelse return error.Declined;
        const theirs = try c.rig.bytesOf(&.{y});
        try std.testing.expectEqualSlices(u8, mine, theirs);
    };
}

test "dsv41 dec engram_kv slots: the contract's launch is the wrapper's, byte for byte (mxfp4 and mxfp8 table rows, with and without the cache's stores)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    for ([_]usize{ 4, 8 }) |tb| for ([_]bool{ false, true }) |kvc| {
        var c = kit.Case(KS).init(a, if (kvc) &slots_kvc_l else &slots_l, .{ .tb = tb, .kvc = kvc });
        defer c.deinit();
        try std.testing.expect(try c.prepare(.{ .rows = 1, .n = 64, .k = 6144 }, .random, 13));
        try c.run();
        const mine = try a.dupe(u8, c.outBytes());
        defer a.free(mine);
        var sc = dsv41.Scope.init(a, c.rig.stream());
        defer sc.deinit();
        const cache: ?Kvc = if (kvc) .{ .plan = c.args[6], .vals = c.args[7] } else null;
        const y = (try launchSlots(&sc, .{ .hd = HD, .tb = tb }, c.args[0], c.args[1], c.args[2], c.args[3], c.args[4], c.args[5], cache)) orelse return error.Declined;
        const theirs = try c.rig.bytesOf(&.{y});
        try std.testing.expectEqualSlices(u8, mine, theirs);
    };
}

/// Squared error, squared reference and largest absolute error of a launch's [1, N] output (f32 or bf16) against
/// float64 `want`, added into `acc`.
const Dist = struct {
    se: f64 = 0,
    ss: f64 = 0,
    max: f64 = 0,

    fn add(d: *Dist, got: []const u8, f32_out: bool, want: []const f64) void {
        for (want, 0..) |w, n| {
            const g: f64 = if (f32_out) @as(f32, @bitCast(std.mem.readInt(u32, got[4 * n ..][0..4], .little))) else @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, got[2 * n ..][0..2], .little)) << 16));
            d.se += (g - w) * (g - w);
            d.ss += w * w;
            d.max = @max(d.max, @abs(g - w));
        }
    }

    fn l2(d: Dist) f64 {
        return @sqrt(d.se / d.ss);
    }

    /// No farther from float64 than `o`, in both measures.
    fn within(d: Dist, o: Dist) bool {
        return d.se <= o.se and d.max <= o.max;
    }
};

test "dsv41 dec engram_kv slots: no farther from float64 than the served rows kernel at the release's decode row (N 25600, K 6144, mxfp4 and mxfp8 table rows, narrow and spread scales), l2 and max, on the f32 sums and the served bf16 output; a planted coarser sum fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const shape: Shape = .{ .rows = 1, .n = 25600, .k = 6144 };
    var e4m3: [256]f64 = undefined;
    for (&e4m3, 0..) |*v, b| v.* = kc.host.e4m3(@intCast(b));
    var near = true;
    var plant_caught = true;
    for ([_]bool{ false, true }) |spread| for ([_]usize{ 4, 8 }) |tb| {
        // [out][kernel]: out 0 the f32 sums, 1 the bf16 output; kernel 0 rows, 1 slots, 2 the plant.
        var d: [2][3]Dist = @splat(@splat(.{}));
        for (0..2) |seed| {
            var c = kit.Case(K_).init(a, &kv_l, .{ .tb = tb, .spread = spread });
            defer c.deinit();
            try std.testing.expect(try c.prepare(shape, .random, 0xF64 + seed));
            const K = shape.k;
            const x = try a.alloc(f64, K);
            defer a.free(x);
            for (x, 0..) |*v, i| v.* = K_.xAt(&c, 0, i);
            const want = try a.alloc(f64, shape.n);
            defer a.free(want);
            const words = c.in(4);
            const exps = c.in(5);
            for (want, 0..) |*o, n| {
                var acc: f64 = 0;
                for (0..K / 32) |g| {
                    const sc = std.math.ldexp(@as(f64, 1), @as(i32, @intFromFloat(exps[n * (K / 32) + g])) - 127);
                    for (0..8) |q| {
                        const wd: u32 = @intFromFloat(words[n * (K / 4) + g * 8 + q]);
                        for (0..4) |b| acc += x[g * 32 + q * 4 + b] * e4m3[(wd >> @intCast(8 * b)) & 0xff] * sc;
                    }
                }
                o.* = acc;
            }
            for ([_]bool{ true, false }, 0..) |f32_out, oi| {
                try c.rig.launch(&kv_l, K_.geometry(.{ .tb = tb, .f32_out = f32_out }, shape).?, c.args[0..6]);
                d[oi][0].add(c.rig.outputBytes(), f32_out, want);
                try c.rig.launch(&slots_l, KS.geometry(.{ .tb = tb, .f32_out = f32_out }, shape).?, c.args[0..6]);
                const got = try a.dupe(u8, c.rig.outputBytes());
                defer a.free(got);
                d[oi][1].add(got, f32_out, want);
                // The plant: the slot kernel's sums kept to 15 mantissa bits (f32), or cut to bf16 toward zero.
                if (f32_out) {
                    for (0..shape.n) |n| std.mem.writeInt(u32, got[4 * n ..][0..4], std.mem.readInt(u32, got[4 * n ..][0..4], .little) & 0xFFFFFF00, .little);
                    d[0][2].add(got, true, want);
                } else {
                    try c.rig.launch(&slots_l, KS.geometry(.{ .tb = tb, .f32_out = true }, shape).?, c.args[0..6]);
                    const f = c.rig.outputBytes();
                    const cut = try a.alloc(u8, 2 * shape.n);
                    defer a.free(cut);
                    for (0..shape.n) |n| std.mem.writeInt(u16, cut[2 * n ..][0..2], @truncate(std.mem.readInt(u32, f[4 * n ..][0..4], .little) >> 16), .little);
                    d[1][2].add(cut, false, want);
                }
            }
        }
        const data = if (spread) "spread scales" else "narrow scales";
        for ([_][]const u8{ "f32 sums", "bf16 output" }, d) |what, row| std.debug.print("[engram-kv f64] {s} tb {d} {s}: rows kernel l2 {e:.4} max {e:.4}; slot kernel l2 {e:.4} max {e:.4}; plant l2 {e:.4} max {e:.4}\n", .{ data, tb, what, row[0].l2(), row[0].max, row[1].l2(), row[1].max, row[2].l2(), row[2].max });
        for (d) |row| {
            near = near and row[1].within(row[0]);
            // A plant on sums the rows kernel gets exact is caught by any error at all.
            plant_caught = plant_caught and !row[2].within(row[0]);
        }
    };
    try std.testing.expect(near);
    try std.testing.expect(plant_caught);
}
