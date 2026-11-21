//! The wide prompt layer's exact rewrites behind their default-off switches against the stock path, byte for byte, at
//! the release's widths and every prompt row count: each kernel's own, then each switch alone and all of them through a
//! release-shape pair worker's prefill (`decode/fixture.zig`).

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const mpair = @import("moe_pair.zig");

const testing = std.testing;
const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const hidden: usize = 5120;
const inter: usize = 2304;
const top_k = 6;

fn skip() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable();
}

fn bytesOf(a: std.mem.Allocator, x: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const raw = mlx.mlx_array_data_uint8(c) orelse return error.NoData;
    return a.dupe(u8, raw[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
}

/// `x` and `y` of one dtype and shape, holding the same bytes (a NaN's payload and a zero's sign included).
fn sameBytes(x: A, y: A) !bool {
    if (mlx.mlx_array_dtype(x) != mlx.mlx_array_dtype(y) or !std.mem.eql(c_int, mlx.getShape(x), mlx.getShape(y))) return false;
    const a = testing.allocator;
    const bx = try bytesOf(a, x);
    defer a.free(bx);
    const by = try bytesOf(a, y);
    defer a.free(by);
    return std.mem.eql(u8, bx, by);
}

fn key(sc: *dsv41.Scope, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_key(&o, seed), &o);
}

fn normal(sc: *dsv41.Scope, shape: []const c_int, dt: mlx.mlx_dtype, std_: f32, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, std_, try key(sc, seed), sc.s), &o);
}

fn uniform(sc: *dsv41.Scope, shape: []const c_int, dt: mlx.mlx_dtype, lo: f32, hi: f32, seed: u64) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_uniform(&o, try sc.f(lo), try sc.f(hi), shape.ptr, shape.len, dt, try key(sc, seed), sc.s), &o);
}

// ── the paired fp4 gate and up (moe_pair.zig) ──

/// Random fp4 codes `[experts, rows, cols]` over every e2m1 value, with scale codes drawn from [lo, hi): e8m0 2^-6 to
/// 2^-4 a 32-group (MXFP4) or e4m3 2^-6 to 0.117 a 16-group (NVFP4), so many gate and up products pass the clamp.
fn fp4Bank(sc: *dsv41.Scope, seed: u64, experts: c_int, rows_: usize, cols_: usize, nv: bool) !dsv41.Bank {
    const group: c_int = if (nv) 16 else 32;
    const rows: c_int = @intCast(rows_);
    const cols: c_int = @intCast(cols_);
    var w = mlx.mlx_array_new();
    const wa = try sc.res(mlx.mlx_random_bits(&w, &[_]c_int{ experts, rows, @divExact(cols, 8) }, 3, 4, try key(sc, seed), sc.s), &w);
    var s = mlx.mlx_array_new();
    const lo: f32 = if (nv) 0x08 else 121;
    const hi: f32 = if (nv) 0x20 else 124;
    const sa = try sc.res(mlx.mlx_random_randint(&s, try sc.f(lo), try sc.f(hi), &[_]c_int{ experts, rows, @divExact(cols, group) }, 3, .uint8, try key(sc, seed + 1), sc.s), &s);
    try dsv41.evalAll(&.{ wa, sa });
    return .{ .w = wa, .s = sa, .qp = .{ .bits = 4, .group_size = @intCast(group), .mode = if (nv) .nvfp4 else .mxfp4 } };
}

const Call = struct { x: A, idx: A, wts: A, sh: A };

/// `L` rows' normed input, routes (six distinct experts a row drawn as floor(u^2 E): expert 0 takes about sqrt(E) times
/// the mean, the high ids run short), f32 weights in [0.05, 1.05) and shared-expert rows; `nan`: row 3 all NaN, row 7
/// one NaN.
fn call(sc: *dsv41.Scope, L: usize, experts: usize, seed: u64, nan: bool) !Call {
    const a = testing.allocator;
    const shape = [_]c_int{ @intCast(L), @intCast(hidden) };
    var x = try normal(sc, &shape, .bfloat16, 1, seed);
    if (nan) {
        try dsv41.evalAll(&.{x});
        const h = try a.dupe(u16, (mlx.mlx_array_data_bfloat16(x) orelse return error.NoData)[0 .. L * hidden]);
        defer a.free(h);
        @memset(h[3 * hidden ..][0..hidden], 0x7FC0);
        h[7 * hidden + 100] = 0x7FC0;
        x = try sc.keep(mlx.mlx_array_new_data(h.ptr, &shape, 2, .bfloat16));
    }
    const ids = try a.alloc(u32, L * top_k);
    defer a.free(ids);
    const ws = try a.alloc(f32, L * top_k);
    defer a.free(ws);
    var rng = std.Random.DefaultPrng.init(seed);
    for (0..L) |l| {
        var j: usize = 0;
        while (j < top_k) {
            const u = rng.random().float(f32);
            const e: u32 = @intCast(@min(experts - 1, @as(usize, @intFromFloat(u * u * @as(f32, @floatFromInt(experts))))));
            if (std.mem.indexOfScalar(u32, ids[l * top_k ..][0..j], e) != null) continue;
            ids[l * top_k + j] = e;
            ws[l * top_k + j] = 0.05 + rng.random().float(f32);
            j += 1;
        }
    }
    const idx = try sc.keep(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ @intCast(L), top_k }, 2, .uint32));
    const wts = try sc.keep(mlx.mlx_array_new_data(ws.ptr, &[_]c_int{ @intCast(L), top_k }, 2, .float32));
    return .{ .x = x, .idx = idx, .wts = wts, .sh = try normal(sc, &shape, .bfloat16, 1, seed + 1) };
}

/// `mpair.act` on `c` as `moePrefill` calls it (its pairs sorted by expert).
fn actOf(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x: A, idx: A, wts: A) !?A {
    const L: usize = @intCast(mlx.getShape(x)[0]);
    const n: c_int = @intCast(L * top_k);
    const flat = try sc.reshape(try sc.astype(idx, .uint32), &.{n});
    var o = mlx.mlx_array_new();
    const order = try sc.res(mlx.mlx_argsort_axis(&o, flat, 0, sc.s), &o);
    const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(top_k));
    const w_sorted = try sc.take(try sc.reshape(wts, &.{n}), order, 0);
    return mpair.act(sc, m, e, try sc.reshape(x, &.{ @intCast(L), 1, @intCast(hidden) }), rows, try sc.take(flat, order, 0), w_sorted);
}

test "dsv41 wide exact: the paired fp4 gate+up with moe_act in its epilogue gives the split path's bytes at the release widths, MXFP4 and NVFP4 with per-expert scales, every prompt row count to 8192" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    // The release's 384 experts under `-Dslow-tests` (7 GB of banks); else 48, whose rows an expert reach every plan.
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const unset = env.override(.MLX_SERVE_DSV41_MOE_PAIR, null);
    defer unset.restore();
    const unset_scaled = env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, null);
    defer unset_scaled.restore();
    // A format whose bytes differ is reported and the other still runs.
    var differs = false;
    inline for (.{ false, true }) |nv| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_wt_kernel = null;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        // Per-expert scales over two octaves either side of 1.
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 17, 256, 500, 3072, 5001, 8064, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x1000 + ci, L == 500);
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            // The other kind's switch leaves this bank on the split path.
            {
                const other = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR, "1") else env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "1");
                defer other.restore();
                try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) == null);
            }
            const sw = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "1") else env.override(.MLX_SERVE_DSV41_MOE_PAIR, "1");
            defer sw.restore();
            // Served: the paired kernel takes this call.
            try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) != null);
            const got = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            if (!try sameBytes(want, got)) {
                std.debug.print("moe pair {s} rows {d}: the paired path's bytes differ from the split path's\n", .{ if (nv) "NVFP4" else "MXFP4", L });
                differs = true;
                break;
            }
            // Planted: the paired path clamped at 9 must not pass for the split path's bytes at 10.
            m.cfg.dsv4_swiglu_limit = 9;
            const planted = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            m.cfg.dsv4_swiglu_limit = 10;
            try testing.expect(!try sameBytes(want, planted));
        }
    }
    if (differs) return error.TestExpectedEqual;
}
