//! The gather_qmv decode kernels on fp4 banks (NVFP4, NVFP4 with ModelOpt's per-expert scales, GLM's clamp; MXFP4
//! declines them)
//! held bit for bit to MLX's composed ops: `gather_qmm`, the scale multiplies, MLX's SwiGLU / clampedSwiGLU, the
//! routing multiply and the slot sum. Inputs make every float32 sum exact (x in halves, e2m1 codes, scales near 1),
//! so any summation order gives the same bits and a difference is a decode or epilogue difference. Edge banks put
//! every e4m3 scale code on its own output row.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const env = @import("../../../env.zig");
const fp4 = @import("../../quant/fp4.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const activations = @import("../mlp/activations.zig");
const testRandUniformBf16 = @import("../../testing/parity.zig").testRandUniformBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const gq = @import("gather_qmv.zig");
const gqr = @import("gather_qmv_rows.zig");
const dr = @import("down_reduce.zig");

const Array = mlx.mlx_array;
const testing = std.testing;
const none: Array = .{ .ctx = null };

const Bank = struct { w: Array, sc: Array };

fn modeName(f: fp4.Format) [*:0]const u8 {
    return if (f == .mxfp4) "mxfp4" else "nvfp4";
}

/// A random scale code in [0.25, 2): e8m0 2^-2..2^0, e4m3 any mantissa (either sign).
fn nearOneCode(rng: std.Random, f: fp4.Format) u8 {
    if (f == .mxfp4) return 125 + rng.uintLessThan(u8, 3);
    const code: u8 = ((5 + rng.uintLessThan(u8, 3)) << 3) | rng.uintLessThan(u8, 8);
    return if (rng.boolean()) code | 0x80 else code;
}

/// An fp4 bank [E, N, K]: `code(e, n, k)` the e2m1 nibble, `scale(e, n, group)` the group's code.
fn bankFrom(g: *ops.Graph, f: fp4.Format, e: usize, n: usize, k: usize, ctx: anytype, comptime code: fn (@TypeOf(ctx), usize, usize, usize) u8, comptime scale: fn (@TypeOf(ctx), usize, usize, usize) u8) !Bank {
    const a = testing.allocator;
    const gs = f.group();
    const words = try a.alloc(u32, e * n * k / 8);
    defer a.free(words);
    const sc = try a.alloc(u8, e * n * k / gs);
    defer a.free(sc);
    for (0..e) |ei| for (0..n) |ni| {
        const row = ei * n + ni;
        for (0..k / 8) |wi| {
            var v: u32 = 0;
            for (0..8) |j| v |= @as(u32, code(ctx, ei, ni, wi * 8 + j) & 15) << @intCast(4 * j);
            words[row * (k / 8) + wi] = v;
        }
        for (0..k / gs) |gi| sc[row * (k / gs) + gi] = scale(ctx, ei, ni, gi);
    };
    const ci = struct {
        fn c(v: usize) c_int {
            return @intCast(v);
        }
    }.c;
    return .{
        .w = try g.adopt(mlx.mlx_array_new_data(words.ptr, &[_]c_int{ ci(e), ci(n), ci(k / 8) }, 3, .uint32)),
        .sc = try g.adopt(mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ ci(e), ci(n), ci(k / gs) }, 3, .uint8)),
    };
}

const RandCtx = struct { rng: std.Random, f: fp4.Format };
fn randCode(c: RandCtx, _: usize, _: usize, _: usize) u8 {
    return c.rng.int(u8) & 15;
}
fn randScale(c: RandCtx, _: usize, _: usize, _: usize) u8 {
    return nearOneCode(c.rng, c.f);
}

fn randBank(g: *ops.Graph, rng: std.Random, f: fp4.Format, e: usize, n: usize, k: usize) !Bank {
    return bankFrom(g, f, e, n, k, RandCtx{ .rng = rng, .f = f }, randCode, randScale);
}

/// x in halves within [-1, 1] (or [0.5, 1] when `positive`), rounded to `dt`.
fn halvesX(g: *ops.Graph, rng: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, positive: bool) !Array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    for (v) |*o| o.* = if (positive) 0.5 * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 2))) else 0.5 * (@as(f32, @floatFromInt(rng.uintLessThan(u32, 5))) - 2);
    return g.astype(try g.fromSlice(f32, v, shape), dt);
}

fn u32s(g: *ops.Graph, v: []const u32, shape: []const c_int) !Array {
    return g.fromSlice(u32, v, shape);
}

/// MLX's gather_qmm, transposed: x3 [B, 1, K] rows picked by `lhs`, experts by `rhs` (both [S]) -> [S, N].
fn gatherRef(g: *ops.Graph, f: fp4.Format, x3: Array, b: Bank, lhs: Array, rhs: Array) !Array {
    const y = try g.call(mlx.mlx_gather_qmm, .{ x3, b.w, b.sc, none, lhs, rhs, true, mlx.mlx_optional_int.some(@intCast(f.group())), mlx.mlx_optional_int.some(4), modeName(f), false });
    const sh = mlx.getShape(y);
    return g.reshape(y, &.{ sh[0], sh[sh.len - 1] });
}

/// The split path's SwiGLU over gathered gate / up [S, N]: each expert's scale multiplied in (`inds` [S]), then
/// MLX's clampedSwiGLU (limit > 0) or silu(g) * u.
fn swigluRef(g: *ops.Graph, gate: Array, up: Array, inds: Array, scales: gq.Scales) !Array {
    const slots = mlx.getShape(gate)[0];
    var pair = [2]Array{ gate, up };
    for ([_]?Array{ scales.gate, scales.up }, 0..) |sc, i| if (sc) |a| {
        pair[i] = try g.mul(pair[i], try g.reshape(try g.take(a, inds, 0), &.{ slots, 1 }));
    };
    if (scales.clamp > 0) return g.adopt(try activations.clampedSwiGLU(g.s, pair[0], pair[1], @floatFromInt(scales.clamp)));
    return g.mul(try g.mul(pair[0], try g.sigmoid(pair[0])), pair[1]);
}

/// The composed weighted sum the down+reduce replaces: per slot `down * score` in x's dtype, then the slots added in
/// order from zero. `d` [R * TOPK, H], `scores` [R * TOPK] -> [R, H].
fn slotSumRef(g: *ops.Graph, d: Array, scores: Array, rows: c_int, topk: c_int) !Array {
    const h = mlx.getShape(d)[1];
    const p = try g.mul(d, try g.reshape(scores, &.{ rows * topk, 1 }));
    var out: [8]Array = undefined;
    for (0..@intCast(rows)) |r| {
        var total = try g.zeros(&.{ 1, h }, mlx.mlx_array_dtype(d));
        for (0..@intCast(topk)) |k| {
            const at: c_int = @intCast(r * @as(usize, @intCast(topk)) + k);
            total = try g.add(total, try g.sliceAxis(p, 0, at, at + 1));
        }
        out[r] = total;
    }
    return g.concat(out[0..@intCast(rows)], 0);
}

/// Same bits (any NaN equals any NaN), element by element.
fn expectSameBits(label: []const u8, want: Array, got: Array, s: mlx.mlx_stream) !void {
    const a = testing.allocator;
    const n = mlx.mlx_array_size(want);
    try testing.expectEqual(n, mlx.mlx_array_size(got));
    const w = try a.alloc(f32, n);
    defer a.free(w);
    const o = try a.alloc(f32, n);
    defer a.free(o);
    try testReadF32(want, w, s);
    try testReadF32(got, o, s);
    for (w, o, 0..) |x, y, i| {
        if (std.math.isNan(x) and std.math.isNan(y)) continue;
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("\n[gqmv-fp4] {s}: element {d} of {d}: MLX {e} kernel {e}\n", .{ label, i, n, x, y });
            return error.NotBitExact;
        }
    }
}

const Shape = struct { name: []const u8, e: usize, hidden: usize, inter: usize, topk: usize };
/// The contract's served shapes, then GLM-5.3 Pro (6144 -> 2048, top-8 of 256) and DeepSeek-V4.1-Flash
/// (5120 -> 2304, top-6 of 384), their banks cut to 16 experts (the arithmetic reads one expert's rows).
const shapes = [_]Shape{
    .{ .name = "contract", .e = 8, .hidden = 2048, .inter = 256, .topk = 8 },
    .{ .name = "contract down", .e = 8, .hidden = 1024, .inter = 512, .topk = 8 },
    .{ .name = "GLM-5.3 Pro", .e = 16, .hidden = 6144, .inter = 2048, .topk = 8 },
    .{ .name = "DeepSeek-V4.1-Flash", .e = 16, .hidden = 5120, .inter = 2304, .topk = 6 },
};
const Variant = struct { name: []const u8, f: fp4.Format, gscale: bool = false, uscale: bool = false, clamp: u32 = 0 };
const variants = [_]Variant{
    .{ .name = "NVFP4", .f = .nvfp4 },
    .{ .name = "NVFP4 + gate scale", .f = .nvfp4, .gscale = true },
    .{ .name = "NVFP4 + gate, up scales, clamp 10", .f = .nvfp4, .gscale = true, .uscale = true, .clamp = 10 },
    .{ .name = "NVFP4 clamp 7", .f = .nvfp4, .clamp = 7 },
};

fn routes(rng: std.Random, out: []u32, e: usize, topk: usize) void {
    var perm: [64]u32 = undefined;
    for (0..out.len / topk) |r| {
        for (perm[0..e], 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(u32, perm[0..e]);
        @memcpy(out[r * topk ..][0..topk], perm[0..topk]);
    }
}

fn skip() bool {
    return mlx.noGpuBackend() or mlx.virtualGpu();
}

/// One shape and variant through every kernel: gather_qmv (shared x and x per expert), the gate+up+SwiGLU at one
/// row and at 4, the down+reduce at one row (under an expert width of 2048) and at 4; one row equals rows at R = 1.
fn matrixCase(sh: Shape, v: Variant, seed: u64) !void {
    const s = mlx.gpuStream();
    var g = ops.Graph.init(testing.allocator, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const f = v.f;
    const mode = f.mode();
    const gs = f.group();
    const E: c_int = @intCast(sh.e);
    const H: c_int = @intCast(sh.hidden);
    const I: c_int = @intCast(sh.inter);
    const TOPK: c_int = @intCast(sh.topk);
    const R: c_int = 4;
    const dt: mlx.mlx_dtype = .bfloat16;
    const gate = try randBank(&g, rng, f, sh.e, sh.inter, sh.hidden);
    const up = try randBank(&g, rng, f, sh.e, sh.inter, sh.hidden);
    const down = try randBank(&g, rng, f, sh.e, sh.hidden, sh.inter);
    const scales: gq.Scales = .{
        .gate = if (v.gscale) try g.adopt(try testRandUniformBf16(rng, &.{E}, 0.5, 2.0, s)) else null,
        .up = if (v.uscale) try g.adopt(try testRandUniformBf16(rng, &.{E}, 0.5, 2.0, s)) else null,
        .clamp = v.clamp,
    };
    var label_buf: [160]u8 = undefined;

    // ── one row ──
    var ids: [64]u32 = undefined;
    routes(rng, ids[0..sh.topk], sh.e, sh.topk);
    const inds = try u32s(&g, ids[0..sh.topk], &.{TOPK});
    const zeros: [64]u32 = @splat(0);
    const lhs0 = try u32s(&g, zeros[0..sh.topk], &.{TOPK});
    var seq: [64]u32 = undefined;
    for (&seq, 0..) |*o, i| o.* = @intCast(i);
    const lhs_seq = try u32s(&g, seq[0..sh.topk], &.{TOPK});
    const x = try halvesX(&g, rng, &.{H}, dt, false);
    const x3 = try g.reshape(x, &.{ 1, 1, H });
    const g_ref = try gatherRef(&g, f, x3, gate, lhs0, inds);
    const u_ref = try gatherRef(&g, f, x3, up, lhs0, inds);

    // gather_qmv, shared x: MLX's gather_qmm.
    const g_got = (try gq.gatherQmv(s, x, gate.w, gate.sc, none, inds, 4, gs, mode, false)) orelse return error.GatherQmvDeclined;
    _ = try g.adopt(g_got);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} gather_qmv", .{ sh.name, v.name }), g_ref, g_got, s);

    // gate+up+SwiGLU, one row.
    const act_ref = try swigluRef(&g, g_ref, u_ref, inds, scales);
    const act = (try gq.gatherQmvGateUpScaled(s, x, gate.w, gate.sc, none, up.w, up.sc, none, inds, 4, gs, mode, scales)) orelse return error.GateUpDeclined;
    _ = try g.adopt(act);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} gateup", .{ sh.name, v.name }), act_ref, act, s);

    // gather_qmv with x per expert (the down projection) and the down+reduce on the same exact activations.
    const xd = try halvesX(&g, rng, &.{ TOPK, I }, dt, false);
    const xd3 = try g.reshape(xd, &.{ TOPK, 1, I });
    const d_ref = try gatherRef(&g, f, xd3, down, lhs_seq, inds);
    const d_got = (try gq.gatherQmv(s, xd, down.w, down.sc, none, inds, 4, gs, mode, true)) orelse return error.GatherQmvDeclined;
    _ = try g.adopt(d_got);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} gather_qmv x per expert", .{ sh.name, v.name }), d_ref, d_got, s);
    const scores = try g.adopt(try testRandUniformBf16(rng, &.{TOPK}, 0.05, 0.6, s));
    const sum_ref = try slotSumRef(&g, d_ref, scores, 1, TOPK);
    const solo_dn = try dr.gatherQmvDownReduce(s, xd, down.w, down.sc, none, inds, scores, 4, gs, mode);
    if (sh.inter >= 2048) {
        // From a 2048-wide expert the solo kernel steps aside by design (gathered down + composed sum).
        try testing.expect(solo_dn == null);
    } else {
        const got = solo_dn orelse return error.DownReduceDeclined;
        _ = try g.adopt(got);
        try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} down_reduce", .{ sh.name, v.name }), sum_ref, got, s);
    }

    // ── R rows (the batched tick / verify) ──
    var rids: [64]u32 = undefined;
    routes(rng, rids[0 .. 4 * sh.topk], sh.e, sh.topk);
    const n_pairs: c_int = R * TOPK;
    const rinds = try u32s(&g, rids[0 .. 4 * sh.topk], &.{ R, TOPK });
    const rflat = try u32s(&g, rids[0 .. 4 * sh.topk], &.{n_pairs});
    var rowof: [64]u32 = undefined;
    for (&rowof, 0..) |*o, i| o.* = @intCast(i / sh.topk);
    const lhs_rows = try u32s(&g, rowof[0 .. 4 * sh.topk], &.{n_pairs});
    const xr = try halvesX(&g, rng, &.{ R, H }, dt, false);
    const xr3 = try g.reshape(xr, &.{ R, 1, H });
    const gr_ref = try gatherRef(&g, f, xr3, gate, lhs_rows, rflat);
    const ur_ref = try gatherRef(&g, f, xr3, up, lhs_rows, rflat);
    const ract_ref = try swigluRef(&g, gr_ref, ur_ref, rflat, scales);
    const ract = (try gqr.gatherQmvGateUpRowsScaled(s, xr, gate.w, gate.sc, none, up.w, up.sc, none, rinds, 4, gs, mode, scales)) orelse return error.GateUpRowsDeclined;
    _ = try g.adopt(ract);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} gateup_rows", .{ sh.name, v.name }), ract_ref, ract, s);

    const xdr = try halvesX(&g, rng, &.{ R, TOPK, I }, dt, false);
    const xdr3 = try g.reshape(xdr, &.{ n_pairs, 1, I });
    const dr_ref = try gatherRef(&g, f, xdr3, down, lhs_seq_n: {
        var sq: [64]u32 = undefined;
        for (&sq, 0..) |*o, i| o.* = @intCast(i);
        break :lhs_seq_n try u32s(&g, sq[0 .. 4 * sh.topk], &.{n_pairs});
    }, rflat);
    const rscores = try g.adopt(try testRandUniformBf16(rng, &.{ R, TOPK }, 0.05, 0.6, s));
    const rsum_ref = try slotSumRef(&g, dr_ref, try g.reshape(rscores, &.{n_pairs}), R, TOPK);
    const rsum = (try dr.gatherQmvDownReduceRows(s, xdr, down.w, down.sc, none, rinds, rscores, 4, gs, mode)) orelse return error.DownRowsDeclined;
    _ = try g.adopt(rsum);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} down_reduce_rows", .{ sh.name, v.name }), rsum_ref, rsum, s);

    // One kernel serves one row and rows: the rows launch at R = 1 is the one-row launch's bytes.
    const act1 = (try gqr.gatherQmvGateUpRowsScaled(s, try g.reshape(x, &.{ 1, H }), gate.w, gate.sc, none, up.w, up.sc, none, try g.reshape(inds, &.{ 1, TOPK }), 4, gs, mode, scales)) orelse return error.GateUpRowsDeclined;
    _ = try g.adopt(act1);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} gateup rows at 1", .{ sh.name, v.name }), act, act1, s);
    if (solo_dn) |sd| {
        const dn1 = (try dr.gatherQmvDownReduceRows(s, try g.reshape(xd, &.{ 1, TOPK, I }), down.w, down.sc, none, try g.reshape(inds, &.{ 1, TOPK }), try g.reshape(scores, &.{ 1, TOPK }), 4, gs, mode)) orelse return error.DownRowsDeclined;
        _ = try g.adopt(dn1);
        try expectSameBits(try std.fmt.bufPrint(&label_buf, "{s} {s} down_reduce rows at 1", .{ sh.name, v.name }), sd, dn1, s);
    }
}

/// Every variant at shape `si`, each seeded as it always was.
fn matrixShape(si: usize) !void {
    if (skip()) return error.SkipZigTest;
    for (variants, 0..) |v, vi| try matrixCase(shapes[si], v, 0x6F40 + si * 16 + vi);
}

comptime {
    if (shapes.len != 4) @compileError("gqmv fp4 matrix: one test per shape below");
}

// One test per shape, so the shards spread them.
test "gqmv fp4 matrix: every kernel x {NVFP4, NVFP4 scaled, clamped} equals MLX's composed ops bit for bit [contract]" {
    try matrixShape(0);
}

test "gqmv fp4 matrix: every kernel x {NVFP4, NVFP4 scaled, clamped} equals MLX's composed ops bit for bit [contract down]" {
    try matrixShape(1);
}

test "gqmv fp4 matrix: every kernel x {NVFP4, NVFP4 scaled, clamped} equals MLX's composed ops bit for bit [GLM-5.3 Pro]" {
    try matrixShape(2);
}

test "gqmv fp4 matrix: every kernel x {NVFP4, NVFP4 scaled, clamped} equals MLX's composed ops bit for bit [DeepSeek-V4.1-Flash]" {
    try matrixShape(3);
}

// ── edge codes ──

const EdgeCtx = struct { f: fp4.Format, neg: bool };
/// Expert 0 positive e2m1 codes, expert 1 negative: each row's terms share a sign, so its partial sums only grow and
/// an overflow is order-free.
fn edgeCode(c: EdgeCtx, e: usize, n: usize, k: usize) u8 {
    const v: u8 = @intCast((n * 7 + k * 3) % 8);
    return if (e == 1 or c.neg) v | 8 else v;
}
/// Row n of expert 0 carries scale code n, expert 1 code 255 - n: all 256 codes on each.
fn edgeScale(_: EdgeCtx, e: usize, n: usize, _: usize) u8 {
    return @intCast(if (e == 0) n else 255 - n);
}

/// float32 -> bfloat16 bits, round to nearest even (NaN kept).
fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Each row's exact value: x . e2m1 times the row's scale (f64 is exact here), to float32 then bfloat16.
fn edgeTruth(f: fp4.Format, xs: []const f32, e: usize, n: usize, k: usize) f32 {
    const ctx = EdgeCtx{ .f = f, .neg = false };
    var sum: f64 = 0;
    for (0..k) |i| {
        const c = edgeCode(ctx, e, n, i);
        const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
        const w = if (c & 8 != 0) -mag[c & 7] else mag[c & 7];
        sum += xs[i] * w;
    }
    const code = edgeScale(ctx, e, n, 0);
    const scale: f64 = if (f == .mxfp4) blk: {
        if (code == 0xff) return std.math.nan(f32);
        break :blk std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, code) - 127)));
    } else blk: {
        if (code & 0x7f == 0x7f) return std.math.nan(f32);
        const ex: i32 = (code >> 3) & 0xf;
        const m: f64 = @floatFromInt(code & 7);
        const val = if (ex == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.pow(f64, 2, @floatFromInt(ex - 7));
        break :blk if (code & 0x80 != 0) -val else val;
    };
    // MLX's sums start at +0, so a zero row is +0 whatever the signs of its terms.
    const p = sum * scale;
    const v: f32 = if (p == 0) 0 else @floatCast(p);
    const bits: u32 = @as(u32, bf16Bits(v)) << 16;
    return @bitCast(bits);
}

fn edgeCase(f: fp4.Format) !void {
    const s = mlx.gpuStream();
    var g = ops.Graph.init(testing.allocator, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xED6E);
    const rng = prng.random();
    const mode = f.mode();
    const gs = f.group();
    const K: c_int = 64;
    const ctx = EdgeCtx{ .f = f, .neg = false };
    const bank = try bankFrom(&g, f, 2, 256, 64, ctx, edgeCode, edgeScale);
    var xs: [64]f32 = undefined;
    for (&xs) |*o| o.* = 0.5 * @as(f32, @floatFromInt(1 + rng.uintLessThan(u32, 2)));
    const x = try g.astype(try g.fromSlice(f32, &xs, &.{K}), .bfloat16);
    const inds = try u32s(&g, &.{ 0, 1 }, &.{2});
    const lhs0 = try u32s(&g, &.{ 0, 0 }, &.{2});
    const x3 = try g.reshape(x, &.{ 1, 1, K });
    var label_buf: [96]u8 = undefined;
    const name = if (f == .mxfp4) "e8m0" else "e4m3";

    // gather_qmv: MLX's gather_qmm, code by code (MLX's fp8.h read is the standard). Rows where both differ from
    // exact arithmetic (a subnormal the GPU flushes) are counted, not failed.
    const ref = try gatherRef(&g, f, x3, bank, lhs0, inds);
    const got = (try gq.gatherQmv(s, x, bank.w, bank.sc, none, inds, 4, gs, mode, false)) orelse return error.GatherQmvDeclined;
    _ = try g.adopt(got);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "every {s} code: gather_qmv vs gather_qmm", .{name}), ref, got, s);
    var host: [512]f32 = undefined;
    try testReadF32(got, &host, s);
    var inexact: usize = 0;
    for (0..2) |e| for (0..256) |n| {
        const want = edgeTruth(f, &xs, e, n, 64);
        const have = host[e * 256 + n];
        if (std.math.isNan(want) and std.math.isNan(have)) continue;
        if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(have))) {
            inexact += 1;
            std.debug.print("[gqmv-fp4] every {s} code: expert {d} code {x}: exact {e}, MLX and the kernel {e}\n", .{ name, e, edgeScale(ctx, e, n, 0), want, have });
        }
    };
    std.debug.print("[gqmv-fp4] every {s} code: {d} of 512 rows off exact arithmetic (MLX's bits on all)\n", .{ name, inexact });

    // gate+up+SwiGLU with and without the clamp and scales: gate rows walk the codes up, up rows down (NaN and the
    // largest values meet the clamp from both banks).
    const up = try bankFrom(&g, f, 2, 256, 64, EdgeCtx{ .f = f, .neg = true }, edgeCode, struct {
        fn sc(_: EdgeCtx, e: usize, n: usize, _: usize) u8 {
            return @intCast(if (e == 1) n else 255 - n);
        }
    }.sc);
    const uref = try gatherRef(&g, f, x3, up, lhs0, inds);
    const gsc = try g.adopt(try testRandUniformBf16(rng, &.{2}, 0.5, 2.0, s));
    const usc = try g.adopt(try testRandUniformBf16(rng, &.{2}, 0.5, 2.0, s));
    for ([_]gq.Scales{ .{}, .{ .clamp = 10 }, .{ .gate = gsc, .up = usc, .clamp = 10 }, .{ .gate = gsc } }) |sc| {
        const want = try swigluRef(&g, ref, uref, inds, sc);
        const have = (try gq.gatherQmvGateUpScaled(s, x, bank.w, bank.sc, none, up.w, up.sc, none, inds, 4, gs, mode, sc)) orelse return error.GateUpDeclined;
        _ = try g.adopt(have);
        try expectSameBits(try std.fmt.bufPrint(&label_buf, "every {s} code: gateup clamp {d}", .{ name, sc.clamp }), want, have, s);
        const rows = (try gqr.gatherQmvGateUpRowsScaled(s, try g.reshape(x, &.{ 1, K }), bank.w, bank.sc, none, up.w, up.sc, none, try g.reshape(inds, &.{ 1, 2 }), 4, gs, mode, sc)) orelse return error.GateUpRowsDeclined;
        _ = try g.adopt(rows);
        try expectSameBits(try std.fmt.bufPrint(&label_buf, "every {s} code: gateup_rows clamp {d}", .{ name, sc.clamp }), want, rows, s);
    }

    // down+reduce: two slots over the edge bank (codes 0..255 and 255..0 per output row), scores 1 and 0.5.
    const xd = try g.astype(try g.fromSlice(f32, &(xs ++ xs), &.{ 2, K }), .bfloat16);
    const d_ref = try gatherRef(&g, f, try g.reshape(xd, &.{ 2, 1, K }), bank, try u32s(&g, &.{ 0, 1 }, &.{2}), inds);
    const scores = try g.astype(try g.fromSlice(f32, &[_]f32{ 1.0, 0.5 }, &.{2}), .bfloat16);
    const sum_ref = try slotSumRef(&g, d_ref, scores, 1, 2);
    const sum = (try dr.gatherQmvDownReduce(s, xd, bank.w, bank.sc, none, inds, scores, 4, gs, mode)) orelse return error.DownReduceDeclined;
    _ = try g.adopt(sum);
    try expectSameBits(try std.fmt.bufPrint(&label_buf, "every {s} code: down_reduce", .{name}), sum_ref, sum, s);
}

test "gqmv fp4 edges: every e4m3 code (subnormals, -0, NaN 0x7f / 0xff) reads as MLX reads it through every kernel" {
    if (skip()) return error.SkipZigTest;
    try edgeCase(.nvfp4);
}

const AllCodes = struct {};
/// Each word walks all 16 e2m1 codes (-0 included); one row is all -0.
fn allCodes(_: AllCodes, _: usize, n: usize, k: usize) u8 {
    return if (n == 7) 8 else @intCast((k + n) % 16);
}
/// e4m3 0x38 = 1.0.
fn oneScale(_: AllCodes, _: usize, _: usize, _: usize) u8 {
    return 0x38;
}

test "gqmv fp4 edges: all 16 e2m1 codes, -0 kept, as MLX reads them" {
    if (skip()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var g = ops.Graph.init(testing.allocator, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0xE2);
    const bank = try bankFrom(&g, .nvfp4, 1, 8, 64, AllCodes{}, allCodes, oneScale);
    const x = try halvesX(&g, prng.random(), &.{64}, .bfloat16, false);
    const inds = try u32s(&g, &.{0}, &.{1});
    const ref = try gatherRef(&g, .nvfp4, try g.reshape(x, &.{ 1, 1, 64 }), bank, try u32s(&g, &.{0}, &.{1}), inds);
    const got = (try gq.gatherQmv(s, x, bank.w, bank.sc, none, inds, 4, 16, .nvfp4, false)) orelse return error.GatherQmvDeclined;
    _ = try g.adopt(got);
    try expectSameBits("all e2m1 codes", ref, got, s);
}

test "gqmv fp4: MXFP4 banks decline these kernels; NVFP4 serves" {
    if (skip()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var g = ops.Graph.init(testing.allocator, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x5317);
    const rng = prng.random();
    const Qp = struct { bits: u32, group_size: u32, mode: QuantMode };
    const Act = enum { silu };
    const mx: Qp = .{ .bits = 4, .group_size = 32, .mode = .mxfp4 };
    const nv: Qp = .{ .bits = 4, .group_size = 16, .mode = .nvfp4 };
    for ([_]fp4.Format{ .mxfp4, .nvfp4 }) |f| {
        const b = try randBank(&g, rng, f, 8, 64, 256);
        const x = try halvesX(&g, rng, &.{256}, .bfloat16, false);
        const inds = try u32s(&g, &.{ 1, 4 }, &.{2});
        const y = try gq.gatherQmv(s, x, b.w, b.sc, none, inds, 4, f.group(), f.mode(), false);
        if (y) |a| _ = try g.adopt(a);
        const gu = try gq.gatherQmvGateUp(s, x, b.w, b.sc, none, b.w, b.sc, none, inds, 4, f.group(), f.mode(), 0);
        if (gu) |a| _ = try g.adopt(a);
        try testing.expectEqual(f == .nvfp4, y != null);
        try testing.expectEqual(f == .nvfp4, gu != null);
    }
    try testing.expect(!dr.gatherQmvDownReduceRowsEligible(Act.silu, mx, mx, mx, 1024));
    try testing.expect(dr.gatherQmvDownReduceRowsEligible(Act.silu, nv, nv, nv, 1024));
}

test "gqmv fp4: refusals are by shape and format" {
    if (skip()) return error.SkipZigTest;
    // No dequant here: MXFP4 at any group, NVFP4 at 32, mxfp8, an fp4 mode at 8 bits.
    try testing.expect(gq.modeDequant(.mxfp4, 4, 16) == null);
    try testing.expect(gq.modeDequant(.nvfp4, 4, 32) == null);
    try testing.expect(gq.modeDequant(.mxfp8, 8, 32) == null);
    try testing.expect(gq.modeDequant(.nvfp4, 8, 16) == null);
    try testing.expect(gq.modeDequant(.mxfp4, 4, 32) == null);
    try testing.expectEqual(gq.Dequant.fp4, gq.modeDequant(.nvfp4, 4, 16).?);
    try testing.expectEqual(gq.Dequant.affine, gq.modeDequant(.affine, 4, 64).?);
    // A per-expert scale that is not [E] in x's dtype.
    const s = mlx.gpuStream();
    var g = ops.Graph.init(testing.allocator, s);
    defer g.deinit();
    var prng = std.Random.DefaultPrng.init(0x7EF);
    const rng = prng.random();
    const b = try randBank(&g, rng, .nvfp4, 8, 64, 256);
    const x = try halvesX(&g, rng, &.{256}, .bfloat16, false);
    const inds = try u32s(&g, &.{ 1, 4 }, &.{2});
    const short = try g.adopt(try testRandUniformBf16(rng, &.{4}, 0.5, 2.0, s));
    const f32sc = try g.astype(try g.adopt(try testRandUniformBf16(rng, &.{8}, 0.5, 2.0, s)), .float32);
    try testing.expect((try gq.gatherQmvGateUpScaled(s, x, b.w, b.sc, none, b.w, b.sc, none, inds, 4, 16, .nvfp4, .{ .gate = short })) == null);
    try testing.expect((try gq.gatherQmvGateUpScaled(s, x, b.w, b.sc, none, b.w, b.sc, none, inds, 4, 16, .nvfp4, .{ .up = f32sc })) == null);
}
