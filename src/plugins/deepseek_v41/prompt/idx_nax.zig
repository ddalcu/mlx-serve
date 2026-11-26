//! `MLX_SERVE_DSV41_PF_IDX_NAX` (on by default; `=0` idx_tile): a wide prompt span's indexer scores on the matrix
//! units (`kernels/pf_idx_score.metal`), not `dsv41_index_tile`'s 8 x 8 simdgroup matrices.
//!
//! idx_tile runs a threadgroup a (row, 64 keys): each row re-reads every key tile and its queries per tile, at 22
//! TFLOPS on the 17.6K card (81 ms of 1.9 TFLOP). Here a threadgroup is 8 adjacent rows over a range of 16-key tiles:
//! dec_idx_score's tile op (16 keys x 16 channels against the row's 32 heads in registers), the 8 rows reading each
//! tile at once, the next tile loading under the current one's contraction, relu, the head weight and the 32-head sum
//! in registers. Shapes: 32 heads x 128 channels, bf16 keys and queries (the wrapper casts q as idx_tile's does),
//! M = the span's rows, N = the keys it sees. Floor: 2 x rows x 32 x keys x 128 FLOP at 120 TFLOPS (compute-bound:
//! 4 KB of keys a 16-key tile per 131 KFLOP); the scores out are 4 B a cell. No dequant or requant. The same scores
//! up to the f32 sums' order (the dot over 128 channels, then the 32 heads), so near-tie picks may differ.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const log = @import("../../../sdk/util/log.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");

const A = mlx.mlx_array;

pub const source = kernelSource("plugins/deepseek_v41/kernels/pf_idx_score.metal");
pub const header = kernelSource("kernels/common/mpp_tensor_ops.metal");
/// The planted control: each row scores one key past what it sees.
pub const plant_header = "#define DSV41_PF_IDX_PLANT 1\n" ++ header;
pub const ins = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };

/// The release's indexer shapes, the kernel's constants.
pub const heads: usize = 32;
pub const channels: usize = 128;
/// Rows a threadgroup (a simdgroup a row).
pub const NSG: usize = 8;
/// Threadgroups a launch aims at: about 8 a GPU core on the M5 Ultra's 80, a machine constant.
const target_tgs: usize = 640;
/// Tiles a range takes at least, so a row's queries (8 KB) load once per 32 KB of keys or more.
const min_tiles: usize = 8;

/// Key ranges a launch over `rows` rows and `nb` keys cuts each row group's tiles into.
pub fn ranges(rows: usize, nb: usize) usize {
    const tiles = (nb + 15) / 16;
    const groups = (rows + NSG - 1) / NSG;
    return @max(1, @min((target_tgs + groups - 1) / groups, tiles / min_tiles));
}

const State = struct { k: ?mlx.mlx_fast_metal_kernel = null, logged: bool = false };
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_idx_nax", .{});
inline fn ps() *State {
    return shared_state.get();
}

pub fn kernel() !mlx.mlx_fast_metal_kernel {
    if (ps().k) |k| return k;
    const k = try dsv41.buildKernel("dsv41_pf_idx_score", &ins, &.{"scores"}, source, header);
    ps().k = k;
    return k;
}

fn dim(x: A, axis: usize) usize {
    return @intCast(mlx.getShape(x)[axis]);
}
fn ci(v: usize) c_int {
    return @intCast(v);
}

/// One launch of `kern` over `inputs` (q bf16 `[L, 32, 128]`, keys bf16 `[nb, 128]`, w bf16 `[L, 32]`, meta {p0, ratio},
/// the weights' scale): the scores `[L, nb]` f32.
pub fn launch(sc: *dsv41.Scope, kern: mlx.mlx_fast_metal_kernel, in: []const A) !A {
    const L = dim(in[0], 0);
    const nb = dim(in[1], 0);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(ranges(L, nb) * NSG * 32), ci((L + NSG - 1) / NSG), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(NSG * 32), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", ci(NSG)));
    const iv = mlx.mlx_vector_array_new_data(in.ptr, in.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The kernel's inputs from `indexScoresTiled`'s arguments (q cast to bf16 as idx_tile casts it to the keys' dtype).
pub fn inputs(sc: *dsv41.Scope, q: A, keys: A, weights: A, wscale: f32, p0: usize, ratio: usize) ![5]A {
    const meta = try sc.ints(&.{ @intCast(p0), @intCast(ratio) }, &.{2});
    const wsc = try sc.keep(mlx.mlx_array_new_data(&[_]f32{wscale}, &[_]c_int{1}, 1, .float32));
    return .{ try sc.astype(q, .bfloat16), keys, weights, meta, wsc };
}

/// `indexScoresTiled` on the matrix units; null (nothing launched) where the switch is off or the shapes, dtypes or GPU
/// are not the kernel's.
pub fn scores(sc: *dsv41.Scope, q: A, keys: A, weights: A, wscale: f32, p0: usize, ratio: usize) !?A {
    if (!lev.onUnlessZero(.PF_IDX_NAX) or !device.naxAvailable() or !mlx.streamIsGpu(sc.s) or ratio == 0) return null;
    if (mlx.getShape(q).len != 3 or dim(q, 1) != heads or dim(q, 2) != channels or dim(q, 0) == 0) return null;
    if (mlx.getShape(keys).len != 2 or dim(keys, 1) != channels or dim(keys, 0) == 0 or mlx.mlx_array_dtype(keys) != .bfloat16) return null;
    if (mlx.getShape(weights).len != 2 or dim(weights, 0) != dim(q, 0) or dim(weights, 1) != heads or mlx.mlx_array_dtype(weights) != .bfloat16) return null;
    if (!ps().logged) {
        ps().logged = true;
        log.info(lev.line(.PF_IDX_NAX) ++ " (rows {d}, keys {d}, ranges {d})\n", .{ dim(q, 0), dim(keys, 0), ranges(dim(q, 0), dim(keys, 0)) });
    }
    const in = try inputs(sc, q, keys, weights, wscale, p0, ratio);
    return try launch(sc, try kernel(), &in);
}

// ── tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;
const nax = device;
const fixture = @import("../decode/fixture.zig");
const decode = @import("../decode/decode.zig");
const idx = @import("../decode/idx_score.zig");

test "dsv41 pf idx_nax: key ranges fill the machine, at least 8 tiles a range, never zero" {
    try testing.expectEqual(@as(usize, 20), ranges(256, 8820)); // 32 row groups x 20 ranges of 28 tiles
    try testing.expectEqual(@as(usize, 4), ranges(256, 515));
    try testing.expectEqual(@as(usize, 1), ranges(256, 100));
    try testing.expectEqual(@as(usize, 1), ranges(17, 1));
    try testing.expectEqual(@as(usize, 69), ranges(17, 8820)); // 3 row groups: 214 ranges cap at 552 / 8
}

/// The kernel (or its planted twin) against today's `indexScoresTiled` and the float64 truth on a span of `rows` rows
/// at the end of layer `li`'s keys: each finite score within f32 reassociation of both, the -inf cells the same, the
/// top half's differing picks all near-ties. Errors on the first failing check.
fn spanCase(fx: *fixture.Fixture, kern: mlx.mlx_fast_metal_kernel, li: usize, rows: usize) !void {
    const gpa = testing.allocator;
    const m = fx.m;
    const ls = &fx.state().layers[li];
    const ratio: usize = m.layers[li].ratio;
    const nb = ls.idx_k.?.used;
    const p0 = nb * ratio - rows;
    var sc = dsv41.Scope.init(gpa, m.s);
    defer sc.deinit();
    const inp = try idx.randomQ(fx, &sc, rows, p0);
    const keys = try idx.keyRows(&sc, ls, nb);
    const chain = try dsv41.indexScoresTiled(&sc, m, inp.q, keys, inp.w, idx.wscale, p0, ratio);
    const in = try inputs(&sc, inp.q, keys, inp.w, idx.wscale, p0, ratio);
    const got = try launch(&sc, kern, &in);
    const gv = try dsv41.hostF32(got);
    defer gpa.free(gv);
    const cv = try dsv41.hostF32(chain);
    defer gpa.free(cv);
    const qh = try idx.hostF64(inp.q);
    defer gpa.free(qh);
    const wh = try idx.hostF64(inp.w);
    defer gpa.free(wh);
    const kh = try idx.hostF64(keys);
    defer gpa.free(kh);
    const t = try idx.truthScores(qh, wh, kh, rows, nb, p0, ratio, null, nb);
    defer gpa.free(t.val);
    defer gpa.free(t.mag);
    var finite: usize = 0;
    var worst: f64 = 0;
    for (gv, cv, t.val, t.mag) |g, c, tv, tm| {
        if (std.math.isInf(c) or std.math.isInf(tv)) {
            if (g != c) return error.MaskDiffers;
            continue;
        }
        finite += 1;
        const bound = 2 * idx.gamma(34) * tm + 1e-30;
        worst = @max(worst, @abs(@as(f64, g) - c) / bound);
        if (@abs(@as(f64, g) - c) > bound) return error.ScoreOffChain;
        if (@abs(@as(f64, g) - tv) > idx.gamma(34) * tm + 1e-30) return error.ScoreOffTruth;
    }
    std.debug.print("[pf] idx_nax layer {d} rows={d} nb={d} ranges={d}: {d} finite scores, worst vs idx_tile {d:.2} of its f32 bound\n", .{ li, rows, nb, ranges(rows, nb), finite, worst });
    if (finite == 0) return error.NoScores;
    const diff = try idx.pickDiffs(gpa, gv, cv, t.mag, rows, nb);
    std.debug.print("[pf] idx_nax: top-half picks of {d} rows: {d} differ from idx_tile's, {d} not near-ties\n", .{ rows, diff[0], diff[1] });
    if (diff[1] != 0) return error.PicksDiffer;
}

test "dsv41 pf idx_nax: a span's scores are idx_tile's within f32 reassociation at prompt widths, ratio 2 and the replay's ratio 1, ragged rows and tiles; the planted off-by-one mask fails" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !nax.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const fx = try fixture.Fixture.init(testing.allocator, .{ .lo = 2, .hi = 4, .ctx = 1030 });
    defer fx.deinit();
    const k = try kernel();
    // 515 keys: 33 tiles, the last of 3 keys, in 4 ranges; 300 and 17 rows leave a partial row group.
    for ([_]usize{ 256, 300, 17 }) |rows| try spanCase(fx, k, 2, rows);
    const plant = try dsv41.buildKernel("dsv41_pf_idx_score_plant", &ins, &.{"scores"}, source, plant_header);
    defer _ = mlx.mlx_fast_metal_kernel_free(plant);
    try testing.expectError(error.MaskDiffers, spanCase(fx, plant, 2, 256));
    // Ratio 1 (the decoder's index layers, a chunk's 128-row replay): 600 keys.
    const fx1 = try fixture.Fixture.init(testing.allocator, .{ .lo = 20, .hi = 21, .ctx = 600 });
    defer fx1.deinit();
    try spanCase(fx1, k, 20, 128);
}

/// `n` (at most 32) launches of the kernel (or `indexScoresTiled`, `replaced`) over one span's inputs, all evaluated at
/// once.
const Bench = struct {
    m: *const dsv41.Dsv41Model,
    q: A,
    keys: A,
    w: A,
    p0: usize,
    n: usize,
    replaced: bool,
    pub fn build(b: @This(), sc: *dsv41.Scope) ![32]A {
        var outs: [32]A = @splat(.{ .ctx = null });
        for (outs[0..b.n]) |*o| {
            if (b.replaced) {
                o.* = try dsv41.indexScoresTiled(sc, b.m, b.q, b.keys, b.w, idx.wscale, b.p0, 2);
            } else {
                const in = try inputs(sc, b.q, b.keys, b.w, idx.wscale, b.p0, 2);
                o.* = try launch(sc, try kernel(), &in);
            }
        }
        return outs;
    }
};

test "dsv41 pf idx_nax: ubench, a 256-row span at the 17.6K card's key counts against idx_tile" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or !nax.naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 2, .hi = 3, .ctx = 0 });
    defer fx.deinit();
    const rows = 256;
    for ([_]usize{ 1024, 4410, 8820 }) |nb| {
        var sc = dsv41.Scope.init(gpa, fx.m.s);
        defer sc.deinit();
        const keys = try sc.keep(try fx.normal(&.{ ci(nb), channels }, .bfloat16, 0, 1));
        const inp = try idx.randomQ(fx, &sc, rows, 2 * nb - rows);
        const q = try sc.astype(inp.q, .bfloat16);
        try mlx.check(mlx.mlx_array_eval(q));
        var b: Bench = .{ .m = fx.m, .q = q, .keys = keys, .w = inp.w, .p0 = 2 * nb - rows, .n = 32, .replaced = false };
        const us = try idx.evalTotalUs(gpa, fx.m.s, b) / 32;
        b.replaced = true;
        const chain = try idx.evalTotalUs(gpa, fx.m.s, b) / 32;
        const flop: f64 = 2.0 * 256.0 * 32.0 * 128.0 * @as(f64, @floatFromInt(nb));
        const floor_us = flop / 120e6;
        std.debug.print("[pf-ubench] idx_nax rows={d} keys={d}: {d:.1} us ({d:.1} TFLOPS, {d:.0}% of the 120 TFLOPS floor {d:.1} us), idx_tile {d:.1} us ({d:.1} TFLOPS)\n", .{ rows, nb, us, flop / us / 1e6, 100 * floor_us / us, floor_us, chain, flop / chain / 1e6 });
    }
}
