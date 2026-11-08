//! The exact DSpark verify's gate (`MLX_SERVE_DSV41_EXACT`, `Dsv41Model.exact`): at the release's shapes and formats
//! (bf16 activations, the iQ pack's affine 2/3-bit experts, mxfp8 projections, the fp8-exact wo_a, the fused decode
//! kernels), a verify block of R = 2..6 rows through a worker's layers gives every row's stream and mix bit for bit as
//! R plain steps do, and, rolled back to any kept prefix, leaves the state those steps leave: a plain step after it
//! gives the same bits. The positions straddle every attention band edge (the split count's 1 -> 2, 2 -> 3 and
//! saturation for the ratio-1 and ratio-2 layers) and the compressor's groups. Must run on the GPU it certifies: a
//! NAX GPU (M5) takes other kernels than an M3 (`Dec.init`), and exact holds only on the non-NAX ones.
//!
//! `DSV41_EXACT_GATE=1` runs it; `DSV41_EXACT_GATE_OFF=1` (the same run with exact off) prints how far today's verify
//! is from the plain steps instead of asserting.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const dsv41 = @import("../deepseek_v41.zig");
const fixture = @import("fixture.zig");
const device = @import("../../../mlx/device.zig");

const A = mlx.mlx_array;

const Range = struct { lo: usize, hi: usize, p0s: []const usize };

/// Layer ranges a worker may hold (each starts at a source layer) and the positions each is checked at.
const ranges = [_]Range{
    // Ratio 1 (KV source, index owner, candidate source, an index non-owner): S 1 -> 2 at p 191, 2 -> 3 at 351,
    // saturation at 511.
    // 190: a two-row verify straddles 191. 286: the rows wrap the window ring (cap 128 + 16 = 144 slots). 16382: the
    // rows straddle 16384 groups, past which the candidate source pins blocks (2048 blocks of 8).
    .{ .lo = 20, .hi = 25, .p0s = &.{ 60, 187, 190, 260, 286, 348, 509, 700, 16382 } },
    // Ratio 2 (a KV source pooling pairs): S 1 -> 2 at p 383, 2 -> 3 at 703, saturation at 1023; pairs straddle.
    .{ .lo = 2, .hi = 4, .p0s = &.{ 61, 380, 382, 701, 1021, 1100 } },
    // Ratio 0: the window alone, filling at 128.
    .{ .lo = 0, .hi = 1, .p0s = &.{ 60, 126 } },
};

const max_rows = 6;

fn freeAll(xs: []const A) void {
    for (xs) |x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x);
    };
}

fn rows(x: A, lo: usize, hi: usize) !A {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const st: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[0] = @intCast(lo);
    stop[0] = @intCast(hi);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, mlx.gpuStream()));
    return o;
}

/// Elements whose bits differ (f32 of bf16 or f32 values: exact either way). The reference must be finite: two equal
/// NaN rows would pass vacuously.
fn differing(a: A, b: A) !usize {
    const ha = try dsv41.hostF32(a);
    defer std.testing.allocator.free(ha);
    const hb = try dsv41.hostF32(b);
    defer std.testing.allocator.free(hb);
    return diffF32(ha, hb);
}

fn diffF32(ha: []const f32, hb: []const f32) !usize {
    if (ha.len != hb.len) return error.ShapeMismatch;
    var n: usize = 0;
    for (ha, hb) |x, y| {
        if (!std.math.isFinite(y)) return error.NonFinite;
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) n += 1;
    }
    return n;
}

/// The state the next plain step reads, on the host: per held layer the window's last rows, the compressed rows,
/// the index keys and the open pooling group (`dsv41.stateRowsForTest`; an absent part is an empty slice).
fn snapState(fx: *fixture.Fixture) ![][]f32 {
    const gpa = std.testing.allocator;
    const p = fx.m.pair.?;
    var parts: std.ArrayList([]f32) = .empty;
    errdefer {
        for (parts.items) |x| gpa.free(x);
        parts.deinit(gpa);
    }
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    for (p.lo..p.hi) |li| for (try dsv41.stateRowsForTest(&sc, fx.m, fx.state(), li)) |a| {
        try parts.append(gpa, if (a.ctx != null and mlx.mlx_array_size(a) > 0) try dsv41.hostF32(a) else try gpa.alloc(f32, 0));
    };
    return parts.toOwnedSlice(gpa);
}

fn freeSnap(s: [][]f32) void {
    for (s) |x| std.testing.allocator.free(x);
    std.testing.allocator.free(s);
}

/// Elements that differ between two `snapState`s (a part of another length counts whole).
fn stateDiffs(a: []const []f32, b: []const []f32) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += if (x.len != y.len) @max(x.len, y.len) else diffF32(x, y) catch x.len;
    return n;
}

const Inputs = struct {
    x: A, // [R, hc, dim]
    pre: A, // [R, hc]
    zx: A, // the probe step after the kept prefix
    zpre: A,
    fn deinit(in: Inputs) void {
        freeAll(&.{ in.x, in.pre, in.zx, in.zpre });
    }
};

/// One step of the worker (caller frees both).
fn step(fx: *fixture.Fixture, x: A, pre: A, p: usize, verify: bool) ![2]A {
    return dsv41.workerStep(fx.m, std.testing.allocator, x, pre, p, false, verify);
}

const Tally = struct { cases: usize = 0, row_diffs: usize = 0, probe_diffs: usize = 0, state_diffs: usize = 0, misses: u64 = 0 };

/// R plain steps then the probe, against one R-row verify kept to `keep` then the probe, from the snapshot at `p0`.
fn checkCase(fx: *fixture.Fixture, in: Inputs, p0: usize, R: usize, keep: usize, tally: *Tally, label: []const u8) !void {
    // Serial: rows 0..keep-1 (all R when keep == R, for the row comparison), then the probe at p0 + keep.
    var serial: [max_rows][2]A = undefined;
    var n_serial: usize = 0;
    defer for (serial[0..n_serial]) |o| freeAll(&o);
    for (0..keep) |i| {
        const xi = try rows(in.x, i, i + 1);
        defer _ = mlx.mlx_array_free(xi);
        const pi = try rows(in.pre, i, i + 1);
        defer _ = mlx.mlx_array_free(pi);
        serial[i] = try step(fx, xi, pi, p0 + i, false);
        n_serial += 1;
    }
    const zs = try step(fx, in.zx, in.zpre, p0 + keep, false);
    defer freeAll(&zs);
    const ss = try snapState(fx);
    defer freeSnap(ss);

    // Verify: all R rows armed, then the probe at p0 + keep (the worker rolls back to `keep` kept rows).
    const xr = try rows(in.x, 0, R);
    defer _ = mlx.mlx_array_free(xr);
    const pr = try rows(in.pre, 0, R);
    defer _ = mlx.mlx_array_free(pr);
    const v = try step(fx, xr, pr, p0, true);
    defer freeAll(&v);
    const zv = try step(fx, in.zx, in.zpre, p0 + keep, false);
    defer freeAll(&zv);
    const sv = try snapState(fx);
    defer freeSnap(sv);

    tally.cases += 1;
    for (0..keep) |i| {
        const vx = try rows(v[0], i, i + 1);
        defer _ = mlx.mlx_array_free(vx);
        const vp = try rows(v[1], i, i + 1);
        defer _ = mlx.mlx_array_free(vp);
        const dx = try differing(vx, serial[i][0]);
        const dp = try differing(vp, serial[i][1]);
        if (dx + dp > 0) {
            tally.row_diffs += 1;
            std.debug.print("[exact-gate] {s} p0={d} R={d} keep={d}: row {d} differs (stream {d}, mix {d} elements)\n", .{ label, p0, R, keep, i, dx, dp });
        }
    }
    const zx_d = try differing(zv[0], zs[0]);
    const zp_d = try differing(zv[1], zs[1]);
    if (zx_d + zp_d > 0) {
        tally.probe_diffs += 1;
        std.debug.print("[exact-gate] {s} p0={d} R={d} keep={d}: the step after the kept rows differs (stream {d}, mix {d})\n", .{ label, p0, R, keep, zx_d, zp_d });
    }
    // The state itself (window, compressed rows, index keys, pooling group), which the probe reads only in part (with
    // every group visible, the index keys never change the picks).
    const sd = stateDiffs(ss, sv);
    if (sd > 0) {
        tally.state_diffs += 1;
        std.debug.print("[exact-gate] {s} p0={d} R={d} keep={d}: the state after the probe differs ({d} elements)\n", .{ label, p0, R, keep, sd });
    }
}

fn runGate(exact_on: bool) !Tally {
    const gpa = std.testing.allocator;
    var tally: Tally = .{};
    var ab: [64]u8 = undefined;
    std.debug.print("[exact-gate] gpu {s}, NAX {any}, exact={any}\n", .{ device.gpuArchitecture(&ab) orelse "?", device.naxAvailable(), exact_on });
    var misses: u64 = 0;
    for (ranges) |rg| for (rg.p0s) |p0| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = rg.lo, .hi = rg.hi, .ctx = p0 - 1, .seed = 7 + p0, .act = .bfloat16, .fused = true });
        defer fx.deinit();
        fx.m.exact = .{ .on = exact_on };
        fx.m.exact_pair_ok = true;
        // One plain step to p0 with a snapshot there: every case rewinds to it.
        {
            const x = try fx.stream(1);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(1);
            defer _ = mlx.mlx_array_free(pre);
            const o = try dsv41.workerStep(fx.m, gpa, x, pre, p0 - 1, true, false);
            freeAll(&o);
        }
        const in: Inputs = .{ .x = try fx.stream(max_rows), .pre = try fx.cpre(max_rows), .zx = try fx.stream(1), .zpre = try fx.cpre(1) };
        defer in.deinit();
        var label_buf: [48]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "layers {d}..{d}", .{ rg.lo, rg.hi });
        for (2..max_rows + 1) |R| for (1..R + 1) |keep| try checkCase(fx, in, p0, R, keep, &tally, label);
        misses += fx.m.exact_misses;
    };
    tally.misses = misses;
    std.debug.print("[exact-gate] exact={any}: {d} cases, {d} rows differ, {d} probe steps differ, {d} states differ, {d} exact misses\n", .{ exact_on, tally.cases, tally.row_diffs, tally.probe_diffs, tally.state_diffs, misses });
    return tally;
}

test "dsv41 exact: verify rows and the state they leave equal plain steps bit for bit (DSV41_EXACT_GATE)" {
    if (env.get(.DSV41_EXACT_GATE) == null) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    // Exact is the non-NAX kernels' (`Dsv41Model.exact` stays off on NAX): on an M5, run it with
    // MLX_SERVE_DIAG=force-gpu-family-fallback=1, which takes this plugin's non-NAX kernels (MLX's own stay NAX's).
    if (device.naxAvailable()) {
        std.debug.print("[exact-gate] skipped: NAX GPU (set MLX_SERVE_DIAG=force-gpu-family-fallback=1)\n", .{});
        return error.SkipZigTest;
    }
    if (env.get(.DSV41_EXACT_GATE_OFF) != null) {
        _ = try runGate(false);
        return;
    }
    const t = try runGate(true);
    try std.testing.expectEqual(@as(usize, 0), t.row_diffs);
    try std.testing.expectEqual(@as(usize, 0), t.probe_diffs);
    try std.testing.expectEqual(@as(usize, 0), t.state_diffs);
    try std.testing.expectEqual(@as(u64, 0), t.misses);
}

fn randBf16(shape: []const c_int, std_: f32, seed: u64) !A {
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_random_key(&k, seed));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .bfloat16, 0, std_, k, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// Rows `lo..hi` of axis `axis`.
fn sliceAxis(x: A, axis: usize, lo: usize, hi: usize) !A {
    const sh = mlx.getShape(x);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    const st: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[axis] = @intCast(lo);
    stop[axis] = @intCast(hi);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, mlx.gpuStream()));
    return o;
}

test "dsv41 exact: the one-row kernel over R rows gives each row's one-row quantized_matmul bits (DSV41_EXACT_GATE)" {
    if (env.get(.DSV41_EXACT_GATE) == null) return error.SkipZigTest;
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    // The release's quantized projections a verify row runs: the 8-bit head, wo_b, the Engram's wkv, the grouped wo_a;
    // and two small ones (a partial output tile, a grouped affine).
    const Case = struct { mode: [*:0]const u8, bits: c_int, gs: c_int, b: c_int, n: c_int, k: c_int, label: []const u8 };
    const cases = [_]Case{
        .{ .mode = "affine", .bits = 8, .gs = 64, .b = 0, .n = 129280, .k = 5120, .label = "head 8-bit" },
        .{ .mode = "mxfp8", .bits = 8, .gs = 32, .b = 0, .n = 5120, .k = 8192, .label = "wo_b" },
        .{ .mode = "mxfp8", .bits = 8, .gs = 32, .b = 0, .n = 25600, .k = 6144, .label = "engram wkv" },
        .{ .mode = "mxfp8", .bits = 8, .gs = 32, .b = 8, .n = 1024, .k = 4096, .label = "wo_a grouped" },
        .{ .mode = "mxfp8", .bits = 8, .gs = 32, .b = 0, .n = 1028, .k = 512, .label = "mxfp8 partial tile" },
        .{ .mode = "affine", .bits = 8, .gs = 64, .b = 3, .n = 520, .k = 1024, .label = "affine grouped, partial tile" },
    };
    var sc = dsv41.Scope.init(std.testing.allocator, s);
    defer sc.deinit();
    var bad: usize = 0;
    for (cases, 0..) |cs, ci| {
        const wshape: []const c_int = if (cs.b > 0) &.{ cs.b, cs.n, cs.k } else &.{ cs.n, cs.k };
        const wf = try randBf16(wshape, 0.02, 100 + ci);
        defer _ = mlx.mlx_array_free(wf);
        const q = try fixture.quantize(wf, cs.gs, cs.bits, cs.mode);
        defer freeAll(&q);
        const mode: @import("../../../sdk/quant/params.zig").QuantMode = if (cs.mode[0] == 'a') .affine else .mxfp8;
        const lin: dsv41.Lin = .{ .w = q[0], .s = q[1], .b = q[2], .qp = .{ .bits = @intCast(cs.bits), .group_size = @intCast(cs.gs), .mode = mode } };
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 16 }) |R| {
            // A grouped input comes as attn_out makes it: [R, B, K] transposed to [B, R, K], a view.
            const xshape: []const c_int = if (cs.b > 0) &.{ @intCast(R), cs.b, cs.k } else &.{ @intCast(R), cs.k };
            const xr = try randBf16(xshape, 1.0, 1000 + ci * 31 + R);
            defer _ = mlx.mlx_array_free(xr);
            var x = xr;
            var xt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xt);
            if (cs.b > 0) {
                try mlx.check(mlx.mlx_transpose_axes(&xt, xr, &[_]c_int{ 1, 0, 2 }, 3, s));
                x = xt;
            }
            const got = (try dsv41.rowQmv(&sc, &lin, x)) orelse return error.Declined;
            const row_axis: usize = if (cs.b > 0) 1 else 0;
            var diff: usize = 0;
            for (0..R) |r| {
                const x1 = try sliceAxis(x, row_axis, r, r + 1);
                defer _ = mlx.mlx_array_free(x1);
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try mlx.check(mlx.mlx_quantized_matmul(&want, x1, q[0], q[1], q[2], true, mlx.mlx_optional_int.some(cs.gs), mlx.mlx_optional_int.some(cs.bits), cs.mode, s));
                const gr = try sliceAxis(got, row_axis, r, r + 1);
                defer _ = mlx.mlx_array_free(gr);
                diff += try differing(gr, want);
            }
            if (diff > 0) bad += 1;
            std.debug.print("[exact-gate] row_qmv {s} R={d}: {d} elements differ from the one-row calls\n", .{ cs.label, R, diff });
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}
