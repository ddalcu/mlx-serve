//! The wide prompt layer's default-off rewrites behind their switches against the stock (unfused) path at the
//! release's widths and prompt row counts: each kernel's own, then each switch alone and all of them through a
//! release-shape pair worker's prefill (`decode/fixture.zig`). A rewrite may round in another order, so each output is
//! held to a tolerance (`Err`), not to the stock path's bytes; a planted semantic bug (a weight on the wrong expert,
//! swapped per-expert scales, a dropped norm weight or mix scale, the routed term at half its scale) must fail it.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const world = @import("../../../sdk/testing/world.zig");
const dsv41 = @import("../deepseek_v41.zig");
const mpair = @import("moe_pair.zig");
const pfroute = @import("pf_route.zig");
const fp4gemm = @import("fp4_gemm.zig");
const gather_nax = @import("../../../sdk/blocks/moe/gather_qmm_nax.zig");
const verify_qmm_nax = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const mtail = @import("moe_tail.zig");
const hv2 = @import("hc_v2.zig");
const fixture = @import("../decode/fixture.zig");
const ggemm = @import("gate_gemm.zig");
const kw = @import("keep_wide.zig");

const testing = std.testing;
const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

const hidden: usize = 5120;
const inter: usize = 2304;
const top_k = 6;

fn skip() bool {
    return !mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable();
}

/// How far `got` is from `want`, in f32 over the elements neither holds a NaN at: `l2` the difference's norm over the
/// base's, `max` its largest magnitude over the base's largest. The base is `want`, or `want - ref` where a residual
/// `ref` would hide the change under test. `nan_same`: the two hold NaNs at the same elements.
const Err = struct {
    l2: f32,
    max: f32,
    nan_same: bool,

    /// Reordered rounding: within `l2_tol` in norm and 5% of the base's largest element anywhere (bf16 keeps 8 bits, so
    /// one step either side is 0.4%; a fused epilogue drops one rounding, a retiled sum reorders them).
    fn ok(e: Err, l2_tol: f32) bool {
        return e.nan_same and e.l2 <= l2_tol and e.max <= 0.05;
    }
};

const tol_kernel: f32 = 1e-2;

fn scalar(x: A) !f32 {
    try dsv41.evalAll(&.{x});
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, x));
    return v;
}

fn errOf(want: A, got: A, ref: ?A) !Err {
    if (mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got) or !std.mem.eql(c_int, mlx.getShape(want), mlx.getShape(got))) return error.ShapeOrDtypeDiffers;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const w = try sc.astype(want, .float32);
    const g = try sc.astype(got, .float32);
    const nw = try sc.op1(mlx.mlx_logical_not, try sc.op2(mlx.mlx_equal, w, w));
    const ng = try sc.op1(mlx.mlx_logical_not, try sc.op2(mlx.mlx_equal, g, g));
    var eq = mlx.mlx_array_new();
    const same = try sc.res(mlx.mlx_array_equal(&eq, nw, ng, false, sc.s), &eq);
    try dsv41.evalAll(&.{same});
    var nan_same: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&nan_same, same));
    const zero = try sc.f(0);
    const w0 = try sc.where(nw, zero, w);
    const g0 = try sc.where(ng, zero, g);
    const d = try sc.sub(g0, w0);
    const base = if (ref) |r| try sc.sub(w0, try sc.where(nw, zero, try sc.astype(r, .float32))) else w0;
    var o1 = mlx.mlx_array_new();
    const dd = try sc.res(mlx.mlx_sum(&o1, try sc.op1(mlx.mlx_square, d), false, sc.s), &o1);
    var o2 = mlx.mlx_array_new();
    const bb = try sc.res(mlx.mlx_sum(&o2, try sc.op1(mlx.mlx_square, base), false, sc.s), &o2);
    var o3 = mlx.mlx_array_new();
    const dm = try sc.res(mlx.mlx_max(&o3, try sc.op1(mlx.mlx_abs, d), false, sc.s), &o3);
    var o4 = mlx.mlx_array_new();
    const bm = try sc.res(mlx.mlx_max(&o4, try sc.op1(mlx.mlx_abs, base), false, sc.s), &o4);
    const vb = try scalar(bb);
    const vm = try scalar(bm);
    if (!(vb > 0) or !(vm > 0)) return error.EmptyBase;
    return .{ .l2 = @sqrt(try scalar(dd) / vb), .max = try scalar(dm) / vm, .nan_same = nan_same };
}

/// `got` within `l2_tol` of `want` (printed either way, so a run shows its margin).
fn close(what: []const u8, want: A, got: A, ref: ?A, l2_tol: f32) !bool {
    const e = try errOf(want, got, ref);
    std.debug.print("{s}: l2 {e:.3} max {e:.3} nan_same {}\n", .{ what, e.l2, e.max, e.nan_same });
    return e.ok(l2_tol);
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

/// `moePrefill`'s pairs sorted by expert: the flat experts, each sorted pair's row, expert and routing weight, and each
/// pair's place among them.
const Pairs = struct { flat: A, rows: A, sidx: A, w: A, inv: A };
fn pairsOf(sc: *dsv41.Scope, idx: A, wts: A) !Pairs {
    const n: c_int = @intCast(mlx.mlx_array_size(idx));
    const flat = try sc.reshape(try sc.astype(idx, .uint32), &.{n});
    var o = mlx.mlx_array_new();
    const order = try sc.res(mlx.mlx_argsort_axis(&o, flat, 0, sc.s), &o);
    var oi = mlx.mlx_array_new();
    const inv = try sc.res(mlx.mlx_argsort_axis(&oi, order, 0, sc.s), &oi);
    return .{ .flat = flat, .rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(top_k)), .sidx = try sc.take(flat, order, 0), .w = try sc.take(try sc.reshape(wts, &.{n}), order, 0), .inv = inv };
}

/// `mpair.act` on `c` as `moePrefill` calls it (its pairs sorted by expert).
fn actOf(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x: A, idx: A, wts: A) !?A {
    const L: usize = @intCast(mlx.getShape(x)[0]);
    const p = try pairsOf(sc, idx, wts);
    return mpair.act(sc, m, e, try sc.reshape(x, &.{ @intCast(L), 1, @intCast(hidden) }), p.rows, p.sidx, p.w);
}

/// `moePrefill`'s activation on the split path (`MLX_SERVE_DSV41_MOE_PAIR_FP4=0`): the fused bank's sorted gather,
/// each half times its expert's scale where the bank has them (NVFP4: `expertScaled`, both rounded), then moe_act.
fn splitAct(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.Experts, x3: A, p: Pairs) !A {
    const gu = &e.gu.?;
    const n = mlx.getShape(p.sidx)[0];
    const mapped = if (verify_qmm_nax.verifyQmmNaxAvailable()) try gather_nax.sortedGatherMappedPrompt(x3, p.rows, gu.w, gu.s, gu.b, p.sidx, gu.qp.bits, gu.qp.group_size, gu.qp.mode.isFp4(), true, sc.s) else null;
    const y = if (mapped) |v| try sc.keep(v) else try dsv41.gatherMm(sc, try sc.take(x3, p.rows, 0), gu, none, p.sidx, true);
    const ni = @divExact(mlx.getShape(y)[2], 2);
    const nu: usize = @intCast(ni);
    var halves: [2]A = undefined;
    for (&halves, 0..) |*hf, i| {
        const half = try sc.slice(y, -1, i * nu, (i + 1) * nu);
        hf.* = if (e.scale[i]) |s| try sc.mul(half, try sc.reshape(try sc.take(s, p.sidx, 0), &.{ n, 1, 1 })) else half;
    }
    const lim = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.dsv4_swiglu_limit}, &[_]c_int{1}, 1, .float32));
    const yy = try sc.reshape(try sc.concat(&halves, -1), &.{ n, 2 * ni });
    return dsv41.flatKernel(sc, m.moe_act_kernel, &.{ yy, p.w, lim }, &.{ n, 1, ni }, .bfloat16, @intCast(n * ni), &.{.{ .name = "N", .v = ni }});
}

/// `moePrefillParts`' down and combine (PF_FP4_GEMM's down in pair order, served by default) over the activation `act`.
fn moeFromAct(sc: *dsv41.Scope, ly: *const dsv41.Layer, act: A, p: Pairs, sh: A) !A {
    const n = mlx.getShape(p.sidx)[0];
    const L = @divExact(n, top_k);
    const dn = (try fp4gemm.downTok(sc, &ly.experts.down, act, p.sidx, p.rows, p.flat, top_k)) orelse return error.DownTokDeclined;
    return dsv41.flatKernel(sc, try dsv41.moeCombineTok(), &.{ try sc.reshape(dn, &.{ n, @intCast(hidden) }), p.inv, sh }, &.{ L, @intCast(hidden) }, .bfloat16, @intCast(L * @as(c_int, @intCast(hidden))), &.{ .{ .name = "D", .v = @intCast(hidden) }, .{ .name = "K", .v = top_k } });
}

/// How far bf16 `a`'s words are from `b`'s, a row of the last axis at a time: the largest difference over one bf16
/// step at its row's largest `b` magnitude (`worst`), the largest distance between words as integers (1: neighbouring
/// values of one sign), and the words more than one step apart: their count and their largest `b` magnitude over
/// their row's.
const Steps = struct { worst: f64, max: usize, far: usize, far_mag: f64 };
fn rowSteps(sc: *dsv41.Scope, a: A, b: A) !Steps {
    const sh = mlx.getShape(a);
    const w = sh[sh.len - 1];
    const af = try sc.reshape(try sc.astype(a, .float32), &.{ -1, w });
    const bf = try sc.reshape(try sc.astype(b, .float32), &.{ -1, w });
    const d = try sc.reshape(try sc.op1(mlx.mlx_abs, try sc.sub(try sc.astype(try sc.view(a, .uint16), .int32), try sc.astype(try sc.view(b, .uint16), .int32))), &.{ -1, w });
    const far = try sc.op2(mlx.mlx_greater, d, try sc.i(1));
    const bm = try sc.op1(mlx.mlx_abs, bf);
    var o = mlx.mlx_array_new();
    const dmax = try sc.res(mlx.mlx_max(&o, d, false, sc.s), &o);
    try dsv41.evalAll(&.{dmax});
    var mx: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&mx, dmax));
    var rows: [4][]f32 = undefined;
    for (rows[0..3], [_]A{ bm, try sc.op1(mlx.mlx_abs, try sc.sub(af, bf)), try sc.where(far, bm, try sc.f(0)) }) |*r, x| {
        r.* = try dsv41.hostF32(try sc.reduce(mlx.mlx_max_axis, x, 1, false));
    }
    rows[3] = try dsv41.hostF32(try sc.reduce(mlx.mlx_sum_axis, try sc.astype(far, .float32), 1, false));
    defer for (rows) |r| testing.allocator.free(r);
    var st: Steps = .{ .worst = 0, .max = @intCast(mx), .far = 0, .far_mag = 0 };
    for (rows[0], rows[1], rows[2], rows[3]) |peak, dv, fm, fc| {
        st.far += @intFromFloat(fc);
        if (dv == 0) continue;
        const step = if (peak > 0) std.math.ldexp(@as(f64, 1), std.math.ilogb(@as(f64, peak)) - 7) else 0;
        st.worst = @max(st.worst, if (step > 0) @as(f64, dv) / step else std.math.inf(f64));
        if (fc > 0) st.far_mag = @max(st.far_mag, @as(f64, fm) / @as(f64, peak));
    }
    return st;
}

/// The words `a` and `b` (the same dtype and shape, bf16 or f32) differ in.
fn wordsDiffer(sc: *dsv41.Scope, a: A, b: A) !usize {
    const wt: mlx.mlx_dtype = if (mlx.mlx_array_dtype(a) == .float32) .uint32 else .uint16;
    const ne = try sc.astype(try sc.op1(mlx.mlx_logical_not, try sc.op2(mlx.mlx_equal, try sc.view(a, wt), try sc.view(b, wt))), .int32);
    var o = mlx.mlx_array_new();
    const s = try sc.res(mlx.mlx_sum(&o, ne, false, sc.s), &o);
    try dsv41.evalAll(&.{s});
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, s));
    return @intCast(v);
}

/// Each row's routing weights rotated by one (a weight on the wrong expert): the planted MoE bug.
fn rolled(sc: *dsv41.Scope, wts: A) !A {
    return sc.concat(&.{ try sc.slice(wts, 1, 1, top_k), try sc.slice(wts, 1, 0, 1) }, 1);
}

test "dsv41 wide tol: the paired fp4 gate+up with moe_act in its epilogue is the split path's MoE within 1% at the release widths, prompt row counts to 8192, NVFP4 (per-expert scales) and MXFP4 (none); a weight on the wrong expert or swapped scales fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    // The release's 384 experts under `-Dslow-tests` (7 GB of banks); else 48, whose rows an expert reach every plan.
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const unset_scaled = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "0");
    defer unset_scaled.restore();
    // Every format and row count runs; each out of tolerance is reported.
    var ok = true;
    inline for (.{ false, true }) |nv| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
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
            // No shared expert, so the routed sum is all the output holds.
            const sh = try csc.zeros(&.{ @intCast(L), @intCast(hidden) }, .bfloat16);
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
            // MOE_PAIR_FP4=0 leaves the bank on the split path.
            try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) == null);
            const sw = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "1");
            defer sw.restore();
            // Served: the paired kernel takes this call.
            try testing.expect(try actOf(&csc, &m, &ly.experts, c.x, c.idx, c.wts) != null);
            var lb: [64]u8 = undefined;
            const label = try std.fmt.bufPrint(&lb, "moe pair {s} rows {d}", .{ if (nv) "NVFP4" else "MXFP4", L });
            const got = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
            if (!try close(label, want, got, null, tol_kernel)) {
                std.debug.print("{s}: the paired path is outside the split path's tolerance\n", .{label});
                ok = false;
            }
            // Planted: each row's weights on the wrong experts.
            const bad_w = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, try rolled(&csc, c.wts), sh);
            try testing.expect(!try close("  planted: weights rolled", want, bad_w, null, tol_kernel));
            // Planted: the gate's per-expert scale on up and up's on the gate.
            if (nv) {
                const s = ly.experts.scale;
                ly.experts.scale = .{ s[1], s[0], s[2] };
                defer ly.experts.scale = s;
                const bad_s = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
                try testing.expect(!try close("  planted: gate and up scales swapped", want, bad_s, null, tol_kernel));
            }
        }
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: MLX_SERVE_DSV41_MOE_PAIR_FP4's paired fp4 gate+up with moe_act in its epilogue is a reorder of the split path's activation (=0), NVFP4 with per-expert scales and MXFP4 without: each differing word within one bf16 step of its row's largest value, under 1% of the words differing, float64 l2 within 1.05x of =0's, at the release widths, prompt rows to 8192, each arm's activation its moePrefill's; the gate and up scales swapped (NVFP4) or each pair's weight on the next pair (MXFP4) fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    try pairReorder(true);
    try pairReorder(false);
}

/// The reorder bounds of `MOE_PAIR_FP4` on an NVFP4 bank with per-expert scales (`nv`) or an MXFP4 bank without.
fn pairReorder(comptime nv: bool) !void {
    const fmt = if (nv) "NVFP4" else "MXFP4";
    const gpa = testing.allocator;
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.dsv4_swiglu_limit = 10;
    m.act = .bfloat16;
    m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
    m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
    var ly: dsv41.Layer = undefined;
    ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
    if (nv) for (&ly.experts.scale, 0..) |*v, i| {
        v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
    };
    const gu = &ly.experts.gu.?;
    // The float64 reference's per-expert scales: the bank's, or 1 (MXFP4 has none).
    var es: [2][]f32 = undefined;
    for (&es, 0..) |*v, i| v.* = if (nv) try dsv41.hostF32(ly.experts.scale[i].?) else blk: {
        const ones = try gpa.alloc(f32, @intCast(experts));
        @memset(ones, 1);
        break :blk ones;
    };
    defer for (es) |v| gpa.free(v);
    // The float64 check's 256 gate columns (every ninth) and their up columns.
    const NC = 256;
    var sel_h: [2 * NC]i32 = undefined;
    for (0..NC) |j| {
        sel_h[j] = @intCast(9 * j);
        sel_h[NC + j] = @intCast(inter + 9 * j);
    }
    const sel = try sc.ints(&sel_h, &.{2 * NC});
    const P = 32;
    var ok = true;
    for ([_]usize{ 17, 256, 500, 3072, 5001, 8064, 8192 }, 0..) |L, ci| {
        var csc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer csc.deinit();
        const c = try call(&csc, L, @intCast(experts), 0x3100 + ci, false);
        const sh = try csc.zeros(&.{ @intCast(L), @intCast(hidden) }, .bfloat16);
        const p = try pairsOf(&csc, c.idx, c.wts);
        const x3 = try csc.reshape(c.x, &.{ @intCast(L), 1, @intCast(hidden) });
        const paired = blk: {
            const on = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, null);
            defer on.restore();
            break :blk (try mpair.act(&csc, &m, &ly.experts, x3, p.rows, p.sidx, p.w)) orelse return error.PairDeclined;
        };
        const split = try splitAct(&csc, &m, &ly.experts, x3, p);
        // Each arm's activation is its served one: through the down and combine, its moePrefill's words.
        for ([_]?[*:0]const u8{ null, "0" }, [_]A{ paired, split }) |v, a| {
            const o = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, v);
            defer o.restore();
            try testing.expect(try sameWords(try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh), try moeFromAct(&csc, &ly, a, p, sh)));
        }
        // Planted: the gate's per-expert scale on up and up's on the gate (NVFP4), or each sorted pair's routing weight
        // on the next pair (MXFP4, which has no scales to swap).
        const bad = blk: {
            if (!nv) {
                const nw = mlx.getShape(p.w)[0];
                const w_next = try csc.concat(&.{ try csc.slice(p.w, 0, 1, @intCast(nw)), try csc.slice(p.w, 0, 0, 1) }, 0);
                break :blk (try mpair.act(&csc, &m, &ly.experts, x3, p.rows, p.sidx, w_next)) orelse return error.PairDeclined;
            }
            const s = ly.experts.scale;
            ly.experts.scale = .{ s[1], s[0], s[2] };
            defer ly.experts.scale = s;
            break :blk (try mpair.act(&csc, &m, &ly.experts, x3, p.rows, p.sidx, p.w)) orelse return error.PairDeclined;
        };
        const differ = try wordsDiffer(&csc, paired, split);
        // float64 at P pairs spread over the sorted order and the checked columns.
        const n = L * top_k;
        const sidx = try dsv41.hostF32(p.sidx);
        defer gpa.free(sidx);
        const prow = try dsv41.hostF32(p.rows);
        defer gpa.free(prow);
        const pw = try dsv41.hostF32(p.w);
        defer gpa.free(pw);
        var q_h: [P]i32 = undefined;
        var r_h: [P]i32 = undefined;
        for (&q_h, &r_h, 0..) |*q, *r, i| {
            q.* = @intCast(i * (n - 1) / (P - 1));
            r.* = @intFromFloat(prow[@intCast(q.*)]);
        }
        const xs = try dsv41.hostF32(try csc.take(c.x, try csc.ints(&r_h, &.{P}), 0));
        defer gpa.free(xs);
        const q_at = try csc.ints(&q_h, &.{P});
        var got: [3][]f32 = undefined;
        for (&got, [_]A{ paired, split, bad }) |*g, a| g.* = try dsv41.hostF32(try csc.take(try csc.reshape(a, &.{ @intCast(n), @intCast(inter) }), q_at, 0));
        defer for (got) |g| gpa.free(g);
        const ref = try gpa.alloc(f64, P * NC);
        defer gpa.free(ref);
        var at: [3][]f32 = undefined;
        for (&at) |*v| v.* = try gpa.alloc(f32, P * NC);
        defer for (at) |v| gpa.free(v);
        var i: usize = 0;
        while (i < P) {
            const e: usize = @intFromFloat(sidx[@intCast(q_h[i])]);
            // The expert's checked rows, dequantized (e2m1 times e4m3 is exact in f32).
            var esc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer esc.deinit();
            const wsel = try esc.reshape(try esc.take(try esc.slice(gu.w, 0, e, e + 1), sel, 1), &.{ 2 * NC, -1 });
            const ssel = try esc.reshape(try esc.take(try esc.slice(gu.s, 0, e, e + 1), sel, 1), &.{ 2 * NC, -1 });
            var o = mlx.mlx_array_new();
            const wd = try dsv41.hostF32(try esc.res(mlx.mlx_dequantize(&o, wsel, ssel, none, mlx.mlx_optional_int.some(@intCast(gu.qp.group_size)), mlx.mlx_optional_int.some(@intCast(gu.qp.bits)), gu.qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, esc.s), &o));
            defer gpa.free(wd);
            while (i < P and @as(usize, @intFromFloat(sidx[@intCast(q_h[i])])) == e) : (i += 1) {
                const xr = xs[i * hidden ..][0..hidden];
                const lim: f64 = m.cfg.dsv4_swiglu_limit;
                for (0..NC) |j| {
                    var g: f64 = 0;
                    var u: f64 = 0;
                    for (xr, wd[j * hidden ..][0..hidden], wd[(NC + j) * hidden ..][0..hidden]) |xv, wg, wu| {
                        g += @as(f64, xv) * @as(f64, wg);
                        u += @as(f64, xv) * @as(f64, wu);
                    }
                    g = @min(g * @as(f64, es[0][e]), lim);
                    u = std.math.clamp(u * @as(f64, es[1][e]), -lim, lim);
                    ref[i * NC + j] = g / (1 + @exp(-g)) * u * @as(f64, pw[@intCast(q_h[i])]);
                    for (at, got) |v, gv| v[i * NC + j] = gv[i * inter + 9 * j];
                }
            }
        }
        const ep = f64Err(at[0], ref);
        const es_ = f64Err(at[1], ref);
        const eb = f64Err(at[2], ref);
        // Where the two arms' words differ: an equal word adds the same error to both, so over every differing word the
        // two arms' l2 and max order as over the whole activation (here up to DS of them, spread over it).
        const DS = 4096;
        var el: [DS]usize = undefined;
        var nd: usize = 0;
        {
            const ne = try csc.astype(try csc.op1(mlx.mlx_logical_not, try csc.op2(mlx.mlx_equal, try csc.view(paired, .uint16), try csc.view(split, .uint16))), .uint8);
            var cm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cm);
            try mlx.check(mlx.mlx_contiguous(&cm, try csc.reshape(ne, &.{-1}), false, csc.s));
            try mlx.check(mlx.mlx_array_eval(cm));
            const mask = (mlx.mlx_array_data_uint8(cm) orelse return error.NoData)[0 .. n * inter];
            const stride = @max(1, (differ + DS - 1) / DS);
            var seen: usize = 0;
            for (mask, 0..) |b, k| if (b != 0) {
                if (seen % stride == 0 and nd < DS) {
                    el[nd] = k;
                    nd += 1;
                }
                seen += 1;
            };
        }
        var nearer = [2]usize{ 0, 0 };
        var dp: F64Err = .{ .l2 = 0, .max = 0 };
        var ds: F64Err = .{ .l2 = 0, .max = 0 };
        if (nd > 0) {
            var dsc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer dsc.deinit();
            const wr = try gpa.alloc(i32, 2 * nd);
            defer gpa.free(wr);
            const xr = try gpa.alloc(i32, nd);
            defer gpa.free(xr);
            const ar = try gpa.alloc(i32, nd);
            defer gpa.free(ar);
            for (el[0..nd], 0..) |k, t| {
                const q = k / inter;
                const e: usize = @intFromFloat(sidx[q]);
                wr[t] = @intCast(e * 2 * inter + k % inter);
                wr[nd + t] = @intCast(e * 2 * inter + inter + k % inter);
                xr[t] = @intFromFloat(prow[q]);
                ar[t] = @intCast(k);
            }
            const wa = try dsc.ints(wr, &.{@intCast(2 * nd)});
            const wsel = try dsc.take(try dsc.reshape(gu.w, &.{ experts * @as(c_int, @intCast(2 * inter)), -1 }), wa, 0);
            const ssel = try dsc.take(try dsc.reshape(gu.s, &.{ experts * @as(c_int, @intCast(2 * inter)), -1 }), wa, 0);
            var o = mlx.mlx_array_new();
            const wd = try dsv41.hostF32(try dsc.res(mlx.mlx_dequantize(&o, wsel, ssel, none, mlx.mlx_optional_int.some(@intCast(gu.qp.group_size)), mlx.mlx_optional_int.some(@intCast(gu.qp.bits)), gu.qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, dsc.s), &o));
            defer gpa.free(wd);
            const xd = try dsv41.hostF32(try dsc.take(c.x, try dsc.ints(xr, &.{@intCast(nd)}), 0));
            defer gpa.free(xd);
            const aa = try dsc.ints(ar, &.{@intCast(nd)});
            var dg: [2][]f32 = undefined;
            for (&dg, [_]A{ paired, split }) |*g, a| g.* = try dsv41.hostF32(try dsc.take(try dsc.reshape(a, &.{-1}), aa, 0));
            defer for (dg) |g| gpa.free(g);
            const dref = try gpa.alloc(f64, nd);
            defer gpa.free(dref);
            const lim: f64 = m.cfg.dsv4_swiglu_limit;
            for (el[0..nd], 0..) |k, t| {
                const q = k / inter;
                const e: usize = @intFromFloat(sidx[q]);
                var g: f64 = 0;
                var u: f64 = 0;
                for (xd[t * hidden ..][0..hidden], wd[t * hidden ..][0..hidden], wd[(nd + t) * hidden ..][0..hidden]) |xv, wg, wu| {
                    g += @as(f64, xv) * @as(f64, wg);
                    u += @as(f64, xv) * @as(f64, wu);
                }
                g = @min(g * @as(f64, es[0][e]), lim);
                u = std.math.clamp(u * @as(f64, es[1][e]), -lim, lim);
                dref[t] = g / (1 + @exp(-g)) * u * @as(f64, pw[q]);
            }
            dp = f64Err(dg[0], dref);
            ds = f64Err(dg[1], dref);
            for (dg[0], dg[1], dref) |a, b, t| {
                const ua = @abs(@as(f64, a) - t);
                const ub = @abs(@as(f64, b) - t);
                if (ua < ub) nearer[0] += 1 else if (ub < ua) nearer[1] += 1;
            }
            // The largest error of each arm (the words l2 and max weigh most).
            var big: [2]usize = .{ 0, 0 };
            for (dg[0], dg[1], dref, 0..) |a, b, t, q| {
                if (@abs(@as(f64, a) - t) > @abs(@as(f64, dg[0][big[0]]) - dref[big[0]])) big[0] = q;
                if (@abs(@as(f64, b) - t) > @abs(@as(f64, dg[1][big[1]]) - dref[big[1]])) big[1] = q;
            }
            for (big, 0..) |q, w| std.debug.print("[moe-pair-diag] rows {d} worst {s}: float64 {e:.6}, MOE_PAIR_FP4 {e:.6}, =0 {e:.6}\n", .{ L, if (w == 0) "MOE_PAIR_FP4" else "=0", dref[q], dg[0][q], dg[1][q] });
        }
        std.debug.print("[moe-pair-f64] " ++ fmt ++ " rows {d}: {d} of {d} activation words differ between MOE_PAIR_FP4 and =0; from float64 at {d} pairs x {d} columns: MOE_PAIR_FP4 l2 {e:.6} max {e:.6}, =0 (split) l2 {e:.6} max {e:.6}, planted ({s}) l2 {e:.3} max {e:.3}; at {d} of the differing words: MOE_PAIR_FP4 l2 {e:.6} max {e:.6}, =0 l2 {e:.6} max {e:.6}; nearer at {d} vs {d} of them\n", .{ L, differ, n * inter, P, NC, ep.l2, ep.max, es_.l2, es_.max, if (nv) @as([]const u8, "scales swapped") else "weights on the next pair", eb.l2, eb.max, nd, dp.l2, dp.max, ds.l2, ds.max, nearer[0], nearer[1] });
        const total = n * inter;
        const st = try rowSteps(&csc, paired, split);
        const differ_b = try wordsDiffer(&csc, bad, split);
        const st_b = try rowSteps(&csc, bad, split);
        std.debug.print("[moe-pair-f64] " ++ fmt ++ " rows {d}: {d:.4}% of words differ from =0 (planted {d:.2}%); the largest difference {d:.3} bf16 steps at its row's largest value (planted {d:.3}); the largest word step {d} (planted {d}); {d} words more than one bf16 step apart, their largest =0 word {e:.3} of its row's largest; float64 l2 ratio to =0 {d:.4} over the sample (planted {d:.1}), {d:.3} over the differing words\n", .{ L, 100 * @as(f64, @floatFromInt(differ)) / @as(f64, @floatFromInt(total)), 100 * @as(f64, @floatFromInt(differ_b)) / @as(f64, @floatFromInt(total)), st.worst, st_b.worst, st.max, st_b.max, st.far, st.far_mag, ep.l2 / es_.l2, eb.l2 / es_.l2, if (ds.l2 > 0) dp.l2 / ds.l2 else 0 });
        const in_bound = struct {
            fn of(sv: Steps, nw: usize, na: usize, ev: F64Err, bv: F64Err) bool {
                return sv.worst <= 1 and nw * 100 < na and ev.l2 <= 1.05 * bv.l2;
            }
        }.of;
        if (!in_bound(st, differ, total, ep, es_)) ok = false;
        try testing.expect(!in_bound(st_b, differ_b, total, eb, es_));
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: the MoE combine and hc expand in one launch are moePrefill then hcExpand within 1% at the release widths, NVFP4 with per-expert scales, prompt row counts to 8192; an unsort off by a row or swapped post fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = 48;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.dsv4_swiglu_limit = 10;
    m.act = .bfloat16;
    m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
    m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
    var ly: dsv41.Layer = undefined;
    ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x7A11, experts, 2 * inter, hidden, true), .down = try fp4Bank(&sc, 0x7A12, experts, hidden, inter, true) };
    for (&ly.experts.scale, 0..) |*v, i| v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x7A13 + i);
    var ok = true;
    for ([_]usize{ 17, 500, 3072, 8192 }, 0..) |L, ci| {
        var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer csc.deinit();
        const c = try call(&csc, L, @intCast(experts), 0x2000 + ci, L == 500);
        const Lc: c_int = @intCast(L);
        const x1 = try normal(&csc, &.{ Lc, 4, @intCast(hidden) }, .bfloat16, 1, 0x2100 + ci);
        const sp: dsv41.Split = .{ .pre = none, .post = try uniform(&csc, &.{ Lc, 4 }, .float32, 0.1, 2, 0x2200 + ci), .comb = try uniform(&csc, &.{ Lc, 4, 4 }, .float32, 0, 1, 0x2300 + ci) };
        const want = try dsv41.hcExpand(&csc, x1, try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh), &sp);
        const p = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
        var lb: [64]u8 = undefined;
        if (!try close(try std.fmt.bufPrint(&lb, "moe tail rows {d}", .{L}), want, try mtail.launch(&csc, p, x1, &sp), x1, tol_kernel)) ok = false;
        // Planted: each row combining the next row's pairs (the unsort off by a row), then each row's post on the wrong
        // copies.
        const n: usize = L * top_k;
        // (Under PF_DOWN_TOKORD the rows are in pair order: the off-by-a-row read is the rows rolled by a token.)
        const off_row: dsv41.MoeParts = if (p.tok)
            .{ .dn = try csc.concat(&.{ try csc.slice(p.dn, 0, top_k, n), try csc.slice(p.dn, 0, 0, top_k) }, 0), .inv = p.inv, .sh = p.sh, .tok = true }
        else
            .{ .dn = p.dn, .inv = try csc.concat(&.{ try csc.slice(p.inv, 0, top_k, n), try csc.slice(p.inv, 0, 0, top_k) }, 0), .sh = p.sh };
        try testing.expect(!try close("  planted: the unsort off by a row", want, try mtail.launch(&csc, off_row, x1, &sp), x1, tol_kernel));
        const rolled_post: dsv41.Split = .{ .pre = none, .post = try csc.concat(&.{ try csc.slice(sp.post, 1, 1, 4), try csc.slice(sp.post, 1, 0, 1) }, 1), .comb = sp.comb };
        try testing.expect(!try close("  planted: post on the wrong copies", want, try mtail.launch(&csc, p, x1, &rolled_post), x1, tol_kernel));
    }
    try testing.expect(ok);
}

test "dsv41 wide: MLX_SERVE_DSV41_MOE_TAIL's one launch is the two launches' words (moePrefill's combine, then hcExpand) at the release widths, NVFP4 with per-expert scales, the down rows in pair order (PF_DOWN_TOKORD, served) and sorted, prompt rows to 8192; an unsort off by a row fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = 48;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var m: dsv41.Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.dsv4_swiglu_limit = 10;
    m.act = .bfloat16;
    m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
    m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
    var ly: dsv41.Layer = undefined;
    ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x7A11, experts, 2 * inter, hidden, true), .down = try fp4Bank(&sc, 0x7A12, experts, hidden, inter, true) };
    for (&ly.experts.scale, 0..) |*v, i| v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x7A13 + i);
    var ok = true;
    for ([_]usize{ 17, 500, 3072, 8192 }, 0..) |L, ci| for ([_]bool{ true, false }) |tok| {
        var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer csc.deinit();
        const tk = env.override(.MLX_SERVE_DSV41_PF_DOWN_TOKORD, if (tok) null else "0");
        defer tk.restore();
        const c = try call(&csc, L, @intCast(experts), 0x2B00 + ci, L == 500);
        const Lc: c_int = @intCast(L);
        const x1 = try normal(&csc, &.{ Lc, 4, @intCast(hidden) }, .bfloat16, 1, 0x2C00 + ci);
        const sp: dsv41.Split = .{ .pre = none, .post = try uniform(&csc, &.{ Lc, 4 }, .float32, 0.1, 2, 0x2D00 + ci), .comb = try uniform(&csc, &.{ Lc, 4, 4 }, .float32, 0, 1, 0x2E00 + ci) };
        const p = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
        try testing.expectEqual(tok, p.tok);
        const combine = struct {
            fn of(s: *dsv41.Scope, mm: *const dsv41.Dsv41Model, q: dsv41.MoeParts, rows: c_int) !A {
                const tmpl = [_]dsv41.Tmpl{ .{ .name = "D", .v = @intCast(hidden) }, .{ .name = "K", .v = top_k } };
                return dsv41.flatKernel(s, if (q.tok) try dsv41.moeCombineTok() else mm.moe_combine_kernel, &.{ q.dn, q.inv, q.sh }, &.{ rows, @intCast(hidden) }, .bfloat16, @intCast(rows * @as(c_int, @intCast(hidden))), &tmpl);
            }
        }.of;
        const want = try dsv41.hcExpand(&csc, x1, try combine(&csc, &m, p, Lc), &sp);
        const got = try mtail.launch(&csc, p, x1, &sp);
        const differ = try wordsDiffer(&csc, want, got);
        // Planted: each row combining the next row's pairs.
        const n: usize = L * top_k;
        const off_row: dsv41.MoeParts = if (p.tok)
            .{ .dn = try csc.concat(&.{ try csc.slice(p.dn, 0, top_k, n), try csc.slice(p.dn, 0, 0, top_k) }, 0), .inv = p.inv, .sh = p.sh, .tok = true }
        else
            .{ .dn = p.dn, .inv = try csc.concat(&.{ try csc.slice(p.inv, 0, top_k, n), try csc.slice(p.inv, 0, 0, top_k) }, 0), .sh = p.sh };
        const differ_p = try wordsDiffer(&csc, want, try mtail.launch(&csc, off_row, x1, &sp));
        std.debug.print("[moe-tail-words] rows {d}, down rows {s}: {d} of {d} words differ from the two launches; planted (unsort off by a row) {d}\n", .{ L, if (tok) "in pair order" else "sorted", differ, mlx.mlx_array_size(want), differ_p });
        if (differ != 0) ok = false;
        try testing.expect(differ_p > 0);
    };
    try testing.expect(ok);
}

// ── the routed pairs' counting sort (pf_route.zig) ──

/// `a` and `b` the same dtype, shape and words.
fn sameWords(a: A, b: A) !bool {
    if (mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(b) or !std.mem.eql(c_int, mlx.getShape(a), mlx.getShape(b))) return false;
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, mlx.gpuStream()));
    try dsv41.evalAll(&.{eq});
    var v = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, eq));
    return v;
}

test "dsv41 wide: the counting-sort route pack gives the argsorts' pair order, rows, experts and routing weights word for word at the release's 384 experts, top-6, replay to chunk widths; weights put by the wrong pair fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts = 384;
    for ([_]usize{ 33, 128, 500, 3072, 5001, 8192 }, 0..) |L, ci| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const c = try call(&sc, L, experts, 0x5017 + ci, false);
        const n: c_int = @intCast(L * top_k);
        const flat = try sc.reshape(c.idx, &.{n});
        const wf = try sc.reshape(c.wts, &.{n});
        var oo = mlx.mlx_array_new();
        const order = try sc.res(mlx.mlx_argsort_axis(&oo, flat, 0, sc.s), &oo);
        var oi = mlx.mlx_array_new();
        const inv = try sc.res(mlx.mlx_argsort_axis(&oi, order, 0, sc.s), &oi);
        const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(top_k));
        const got = (try pfroute.pack(&sc, flat, wf, top_k, experts)) orelse return error.PackDeclined;
        const ok = [_]bool{ try sameWords(inv, got.inv), try sameWords(rows, got.rows), try sameWords(try sc.take(flat, order, 0), got.sidx), try sameWords(try sc.take(wf, order, 0), got.w) };
        std.debug.print("route pack rows {d}: inv {} rows {} experts {} weights {}\n", .{ L, ok[0], ok[1], ok[2], ok[3] });
        try testing.expect(ok[0] and ok[1] and ok[2] and ok[3]);
        // Planted: each row's weights on the wrong experts.
        const bad = (try pfroute.pack(&sc, flat, try sc.reshape(try rolled(&sc, c.wts), &.{n}), top_k, experts)).?;
        try testing.expect(!try sameWords(try sc.take(wf, order, 0), bad.w));
    }
}

test "dsv41 wide: MLX_SERVE_DSV41_PF_ROUTE_PACK's route pack (its scan on one threadgroup) is today's pack word for word (pair order, rows, experts, weights) at 384 experts, top-6, replay to pass widths; a planted first offset differs" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts = 384;
    for ([_]usize{ 33, 128, 500, 3072, 5001, 8192, 17_638 }, 0..) |L, ci| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const c = try call(&sc, L, experts, 0x5117 + ci, false);
        const n: c_int = @intCast(L * top_k);
        const flat = try sc.reshape(c.idx, &.{n});
        const wf = try sc.reshape(c.wts, &.{n});
        const want = (try pfroute.pack(&sc, flat, wf, top_k, experts)) orelse return error.PackDeclined;
        const s0 = pfroute.served.load(.monotonic);
        const got = (try pfroute.packFast(&sc, flat, wf, top_k, experts, false)) orelse return error.PackDeclined;
        try testing.expect(pfroute.served.load(.monotonic) == s0 + 1);
        const ok = [_]bool{ try sameWords(want.inv, got.inv), try sameWords(want.rows, got.rows), try sameWords(want.sidx, got.sidx), try sameWords(want.w, got.w) };
        const bad = (try pfroute.packFast(&sc, flat, wf, top_k, experts, true)).?;
        const caught = !try sameWords(want.inv, bad.inv);
        std.debug.print("route pack fast rows {d}: inv {} rows {} experts {} weights {}; planted offset caught {}\n", .{ L, ok[0], ok[1], ok[2], ok[3], caught });
        try testing.expect(ok[0] and ok[1] and ok[2] and ok[3] and caught);
    }
}

test "dsv41 wide: MLX_SERVE_DSV41_PF_ROUTE_PACK keeps a pair worker's prefill streams and mixes word for word (128 experts, the top-6 and the pack's scan both served), MXFP4 and NVFP4 experts with per-expert scales" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    // The switch is on by default: the served arm needs the explicit 0, and the test leaves it as it found it.
    const keep = env.override(.MLX_SERVE_DSV41_PF_ROUTE_PACK, "0");
    defer keep.restore();
    for ([_]bool{ false, true }) |nv| {
        const rows = 4608;
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 22, .ctx = 0, .experts = 128, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none });
        defer fx.deinit();
        const x = try fx.stream(rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(rows);
        defer _ = mlx.mlx_array_free(pre);
        var want_tap: dsv41.RouteTap = .{ .gpa = gpa };
        defer freeTap(&want_tap);
        const s0 = pfroute.served.load(.monotonic);
        const want = try prefillTapped(fx, x, pre, &want_tap);
        defer for (want) |o| {
            _ = mlx.mlx_array_free(o);
        };
        try testing.expectEqual(s0, pfroute.served.load(.monotonic));
        env.set(.MLX_SERVE_DSV41_PF_ROUTE_PACK, "1");
        var tap: dsv41.RouteTap = .{ .gpa = gpa };
        defer freeTap(&tap);
        const got = try prefillTapped(fx, x, pre, &tap);
        defer for (got) |o| {
            _ = mlx.mlx_array_free(o);
        };
        env.set(.MLX_SERVE_DSV41_PF_ROUTE_PACK, "0");
        // Two layers, each its top-6 and its pack.
        const n = pfroute.served.load(.monotonic) - s0;
        const same = [_]bool{ try sameWords(want[0], got[0]), try sameWords(want[1], got[1]) };
        std.debug.print("wide layers 20-21 {s} at {d} rows, PF_ROUTE_PACK: served {d} calls, streams {} mixes {}\n", .{ if (nv) "NVFP4" else "MXFP4", rows, n, same[0], same[1] });
        try testing.expect(n >= 4 and same[0] and same[1]);
    }
}

/// route.metal as `routeFrom` launches it: today's prompt top-6.
fn routeToday(sc: *dsv41.Scope, logits: A, bias: A, scal: A, k: usize, norm: bool) ![2]A {
    const kern = try dsv41.buildKernel("dsv41_route", &.{ "logits", "bias", "scal" }, &.{ "idx", "wts" }, dsv41.ROUTE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(kern);
    const L: c_int = mlx.getShape(logits)[0];
    const E: usize = @intCast(mlx.getShape(logits)[1]);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ L, @intCast(k) }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ L, @intCast(k) }, 2, .float32));
    const tg: c_int = @intCast((E + 31) / 32 * 32);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg, L, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", @intCast(k)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", norm));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ logits, bias, scal }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var out: [2]A = undefined;
    for (&out, 0..) |*o, i| {
        var a = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&a, ov, i), &a);
    }
    return out;
}

test "dsv41 wide: MLX_SERVE_DSV41_PF_ROUTE_PACK's top-6 on a simdgroup a row is route.metal's picks and weights word for word at 384 experts, prompt rows 33 to 17,638, logits on a coarse grid (ties broken to the lower expert), one bias row and a bias a row (image spans), norm on and off, with and without NVFP4's per-expert down scale (routeScaled's take and multiply); the planted pick order differs" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const a = testing.allocator;
    const E = 384;
    for ([_]usize{ 33, 128, 2048, 17_638 }, 0..) |L, li| for ([_]bool{ false, true }) |per_row| for ([_]bool{ true, false }) |norm| for ([_]bool{ false, true }) |ds| {
        var sc = dsv41.Scope.init(a, mlx.gpuStream());
        defer sc.deinit();
        var rng = std.Random.DefaultPrng.init(0x6017 + li);
        const z = try a.alloc(f32, L * E);
        defer a.free(z);
        // Router logits spread as served (a few units), on a 1/8 grid so equal scores occur within a row.
        for (z) |*v| v.* = @round(rng.random().floatNorm(f32) * 2.5 * 8) / 8;
        const br: usize = if (per_row) L else 1;
        const b = try a.alloc(f32, br * E);
        defer a.free(b);
        for (b) |*v| v.* = @round(rng.random().floatNorm(f32) * 0.3 * 16) / 16;
        const logits = try sc.keep(mlx.mlx_array_new_data(z.ptr, &[_]c_int{ @intCast(L), E }, 2, .float32));
        const bias = try sc.keep(mlx.mlx_array_new_data(b.ptr, &[_]c_int{ @intCast(br), E }, 2, .float32));
        const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{1.5}, &[_]c_int{1}, 1, .float32));
        var dsc: ?A = null;
        if (ds) {
            const d = try a.alloc(f32, E);
            defer a.free(d);
            for (d) |*v| v.* = 0.25 + rng.random().float(f32);
            dsc = try sc.astype(try sc.keep(mlx.mlx_array_new_data(d.ptr, &[_]c_int{E}, 1, .float32)), .bfloat16);
        }
        var want = try routeToday(&sc, logits, bias, scal, top_k, norm);
        if (dsc) |d| want[1] = try sc.mul(want[1], try sc.take(d, want[0], 0));
        const got = (try pfroute.routeRows(&sc, logits, bias, scal, dsc, top_k, norm, false)) orelse return error.RouteDeclined;
        const bad = (try pfroute.routeRows(&sc, logits, bias, scal, dsc, top_k, norm, true)).?;
        const ok = [_]bool{ try sameWords(want[0], got[0]), try sameWords(want[1], got[1]) };
        const caught = !try sameWords(want[0], bad[0]);
        std.debug.print("route rows {d} (bias {s}, norm {}, down scale {}): picks {} weights {}; planted order caught {}\n", .{ L, if (per_row) "a row each" else "one row", norm, ds, ok[0], ok[1], caught });
        try testing.expect(ok[0] and ok[1] and caught);
    };
}

test "dsv41 wide: moePrefill under MLX_SERVE_DSV41_PF_MOE_EPI is the argsort path's output word for word, MXFP4 and NVFP4 with per-expert scales, paired and split gate/up, prompt rows to 8192" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    // On by default: the argsort arm needs the explicit 0.
    const unset = env.override(.MLX_SERVE_DSV41_PF_MOE_EPI, "0");
    defer unset.restore();
    // EPI's own pack (`pfroute.pack`), not PF_ROUTE_PACK's (on by default), is the path under test.
    const own_pack = env.override(.MLX_SERVE_DSV41_PF_ROUTE_PACK, "0");
    defer own_pack.restore();
    inline for (.{ false, true }) |nv| inline for (.{ false, true }) |paired| {
        const pk = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, if (paired) "1" else "0");
        defer pk.restore();
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        // No router weight: the pack takes its expert count from the banks, as a caller building the banks alone needs.
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 40, 500, 3072, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x2000 + ci, false);
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const on = env.override(.MLX_SERVE_DSV41_PF_MOE_EPI, "1");
            defer on.restore();
            const got = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            // MOE_TAIL reads the parts: the same down rows and unsort.
            const pw = blk: {
                const off = env.override(.MLX_SERVE_DSV41_PF_MOE_EPI, "0");
                defer off.restore();
                break :blk try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            };
            const pg = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const eq = try sameWords(want, got) and try sameWords(pw.inv, pg.inv) and try sameWords(pw.dn, pg.dn);
            std.debug.print("pf route moePrefill {s} {s} rows {d}: same words {}\n", .{ if (nv) "NVFP4" else "MXFP4", if (paired) "paired" else "split", L, eq });
            try testing.expect(eq);
            // Planted: each row's weights on the wrong experts.
            const bad = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, try rolled(&csc, c.wts), c.sh);
            try testing.expect(!try sameWords(want, bad));
        }
    };
}

test "dsv41 wide: moePrefill under MLX_SERVE_DSV41_PF_FP4_GEMM is today's output word for word, NVFP4 with per-expert scales, paired gate/up, odd prompt rows to 8192" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    // Off as "0", not unset: a run with the switch exported would unsetenv before the first stderr print, which
    // std's lazy environ scan then reads past.
    const unset = env.override(.MLX_SERVE_DSV41_PF_FP4_GEMM, "0");
    defer unset.restore();
    // The paired gate/up is NVFP4's alone: an MXFP4 bank takes the split path, which stages no gate/up.
    inline for (.{true}) |nv| {
        const pk = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "1");
        defer pk.restore();
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        ly.gate_w = try sc.zeros(&.{ experts, 8 }, .bfloat16);
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 40, 500, 777, 1254, 4097, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x2100 + ci, false);
            const want = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            inline for (.{ false, true }) |seg| {
                const gs = env.override(.MLX_SERVE_DSV41_PF_GU_SEG, if (seg) "1" else "0");
                defer gs.restore();
                const on = env.override(.MLX_SERVE_DSV41_PF_FP4_GEMM, "1");
                defer on.restore();
                const before = fp4gemm.launches.load(.monotonic);
                const got = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
                const served = fp4gemm.launches.load(.monotonic) - before;
                // Under PF_DOWN_TOKORD the staged down's rows are in pair order: today's sorted rows unsorted.
                const eq = try sameWords(if (got.tok) try csc.take(want.dn, want.inv, 0) else want.dn, got.dn) and try sameWords(want.inv, got.inv);
                std.debug.print("pf fp4 gemm moePrefill {s} rows {d} gu_seg {}: staged launches {d}, same words {}\n", .{ if (nv) "NVFP4" else "MXFP4", L, seg, served, eq });
                // Down always staged; under PF_GU_SEG gate/up only where the seg 128 schedule does not serve the call.
                const launches: u32 = if (!seg or gather_nax.stagedServesGateUp(@intCast(L * top_k), experts, @intCast(hidden), @intCast(inter))) 2 else 1;
                try testing.expectEqual(launches, served);
                try testing.expect(eq);
                // Planted: each row's weights on the wrong experts.
                const bad = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, try rolled(&csc, c.wts), c.sh);
                try testing.expect(!try sameWords(want.dn, bad.dn));
            }
        }
    }
}

test "dsv41 wide: MLX_SERVE_DSV41_PF_DOWN_TOKORD: the staged down's rows at their pairs' slots are the sorted rows unsorted, and moePrefill, MOE_TAIL and the hc_v2 combining tails reading them in place are today's output, word for word, NVFP4 with per-expert scales and MXFP4, prompt rows to 8192; each row handed the next token fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const off = env.override(.MLX_SERVE_DSV41_PF_DOWN_TOKORD, "0");
    defer off.restore();
    inline for (.{ true, false }) |nv| {
        // The NVFP4 bank's gate/up on the paired kernel; an MXFP4 bank's on the split path.
        const pk = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "1");
        defer pk.restore();
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        ly.gate_w = try sc.zeros(&.{ experts, 8 }, .bfloat16);
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 40, 500, 777, 1254, 4097, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x2400 + ci, false);
            const Lc: c_int = @intCast(L);
            const x1 = try normal(&csc, &.{ Lc, 4, @intCast(hidden) }, .bfloat16, 1, 0x2500 + ci);
            const sp: dsv41.Split = .{ .pre = none, .post = try uniform(&csc, &.{ Lc, 4 }, .float32, 0.1, 2, 0x2600 + ci), .comb = try uniform(&csc, &.{ Lc, 4, 4 }, .float32, 0, 1, 0x2700 + ci) };
            const y = try normal(&csc, &.{ Lc, @intCast(hidden) }, .bfloat16, 1, 0x2800 + ci);
            const cpre = try uniform(&csc, &.{ Lc, 4 }, .float32, 0.1, 1, 0x2900 + ci);
            const fnw = try normal(&csc, &.{ @intCast(hv2.MIX), @intCast(hv2.N) }, .float32, 0.01, 0x2A00 + ci);
            const S = hv2.slicesFor(L);
            // Today's: the sorted down rows, their unsort, and every reader of them.
            const want = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const want_y = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const want_t = try mtail.launch(&csc, want, x1, &sp);
            const tin = [_]A{ x1, y, sp.post, sp.comb, want.dn, want.inv, try csc.astype(want.sh, .bfloat16), sp.post, sp.comb, cpre, fnw };
            const want_c = try hv2.mixLaunch(&csc, .tail_c, &tin, L, S);
            const want_o = try hv2.mixLaunch(&csc, .tail_only_c, tin[0..9], L, S);
            const on = env.override(.MLX_SERVE_DSV41_PF_DOWN_TOKORD, "1");
            defer on.restore();
            const before = fp4gemm.tok_launches.load(.monotonic);
            const got = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const served = fp4gemm.tok_launches.load(.monotonic) - before;
            const got_y = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const got_t = try mtail.launch(&csc, got, x1, &sp);
            var gin = tin;
            gin[4] = got.dn;
            gin[5] = got.inv;
            const got_c = try hv2.mixLaunchTo(&csc, .tail_c, &gin, L, S, true);
            const got_o = try hv2.mixLaunchTo(&csc, .tail_only_c, gin[0..9], L, S, true);
            const rows_eq = try sameWords(try csc.take(want.dn, want.inv, 0), got.dn);
            const eq = [_]bool{ rows_eq, try sameWords(want_y, got_y), try sameWords(want_t, got_t), try sameWords(want_c[0], got_c[0]) and try sameWords(want_c[1], got_c[1]) and try sameWords(want_c[2], got_c[2]), try sameWords(want_o[0], got_o[0]) };
            std.debug.print("[down-tokord] {s} rows {d}: tok {} launches {d}; down rows {}, moePrefill {}, moe_tail {}, hc_v2 tail_c {}, tail_only_c {}\n", .{ if (nv) "NVFP4" else "MXFP4", L, got.tok, served, eq[0], eq[1], eq[2], eq[3], eq[4] });
            try testing.expect(got.tok);
            // The one moePrefillParts call's down.
            try testing.expectEqual(@as(u32, 1), served);
            for (eq) |v| try testing.expect(v);
            // Planted: each sorted row handed the next token (its value in another token's slots).
            fp4gemm.tok_plant = true;
            defer fp4gemm.tok_plant = false;
            const bad = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            try testing.expect(!try sameWords(try csc.take(want.dn, want.inv, 0), bad.dn));
            try testing.expect(!try sameWords(want_y, try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh)));
        }
    }
}

test "dsv41 wide: MLX_SERVE_DSV41_PF_DOWN_BK32: the down on the 32-deep tile gives moePrefill's words at every prompt width, NVFP4 with per-expert scales and MXFP4; the calls of 96 rows an expert and more only; each down row handed the next token fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const off = env.override(.MLX_SERVE_DSV41_PF_DOWN_BK32, "0");
    defer off.restore();
    inline for (.{ true, false }) |nv| {
        // The NVFP4 bank's gate/up on the paired kernel; an MXFP4 bank's on the split path. The staged down serves both.
        const pk = env.override(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "1");
        defer pk.restore();
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        var m: dsv41.Dsv41Model = undefined;
        m.cfg = .{};
        m.cfg.dsv4_swiglu_limit = 10;
        m.act = .bfloat16;
        m.moe_act_kernel = try dsv41.buildKernel("dsv41_moe_act", &dsv41.MOE_ACT_INS, &.{"out"}, dsv41.MOE_ACT_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        m.moe_combine_kernel = try dsv41.buildKernel("dsv41_moe_combine", &dsv41.MOE_COMBINE_INS, &.{"out"}, dsv41.MOE_COMBINE_SOURCE, "");
        defer _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
        var ly: dsv41.Layer = undefined;
        ly.gate_w = try sc.zeros(&.{ experts, 8 }, .bfloat16);
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 40, 777, 4097, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x3400 + ci, false);
            const wide = @divTrunc(@as(c_int, @intCast(L * top_k)), experts) >= 96;
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const on = env.override(.MLX_SERVE_DSV41_PF_DOWN_BK32, "1");
            defer on.restore();
            const before = fp4gemm.bk32_launches.load(.monotonic);
            const got = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const served = fp4gemm.bk32_launches.load(.monotonic) - before;
            const eq = try sameWords(want, got);
            std.debug.print("[down-bk32] {s} rows {d}: launches {d}, moePrefill {}\n", .{ if (nv) "NVFP4" else "MXFP4", L, served, eq });
            try testing.expect(eq);
            // The down, at the wide calls only.
            try testing.expectEqual(@as(u32, if (wide) 1 else 0), served);
            if (!wide) continue;
            fp4gemm.bk32_plant = true;
            defer fp4gemm.bk32_plant = false;
            try testing.expect(!try sameWords(want, try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh)));
        }
    }
}

// ── the hc stages ──

/// A model holding only the hc kernels, at the release's widths, the fused ones built and probed.
fn hcModel(m: *dsv41.Dsv41Model) !void {
    m.cfg = .{};
    m.cfg.hidden_size = hidden;
    m.cfg.dsv4_hc_mult = 4;
    m.cfg.dsv4_hc_sinkhorn_iters = 20;
    m.s = mlx.gpuStream();
    m.act = .bfloat16;
    m.hc_tg = 1024;
    m.hc_mix_kernel = try dsv41.buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, dsv41.HC_MIX_SOURCE, "");
    m.hc_norm_kernel = try dsv41.buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, dsv41.HC_NORM_SOURCE, "");
    m.hc_split_kernel = try dsv41.buildKernel("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, dsv41.HC_SPLIT_SOURCE, "");
    try dsv41.hcFuseInit(testing.allocator, m);
}

fn hcModelFree(m: *dsv41.Dsv41Model) void {
    for ([_]mlx.mlx_fast_metal_kernel{ m.hc_mix_kernel, m.hc_norm_kernel, m.hc_split_kernel }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
}

/// `L` rows of a wide layer's hc inputs: the stream, the block output, its split, the previous mix, the mix weights
/// and the norm.
const HcRows = struct { x: A, y: A, sp: dsv41.Split, cpre: A, hc: dsv41.Hc, w: A };

fn hcRows(sc: *dsv41.Scope, L: c_int, seed: u64) !HcRows {
    const d: c_int = @intCast(hidden);
    return .{
        .x = try normal(sc, &.{ L, 4, d }, .bfloat16, 1, seed),
        .y = try normal(sc, &.{ L, d }, .bfloat16, 1, seed + 1),
        .sp = .{ .pre = none, .post = try uniform(sc, &.{ L, 4 }, .float32, 0.1, 2, seed + 2), .comb = try uniform(sc, &.{ L, 4, 4 }, .float32, 0, 1, seed + 3) },
        .cpre = try uniform(sc, &.{ L, 4 }, .float32, 0.05, 0.45, seed + 4),
        .hc = .{ .fn_w = try normal(sc, &.{ 24, 4 * d }, .float32, 0.01, seed + 5), .scale = try uniform(sc, &.{24}, .float32, 0.5, 1.5, seed + 6), .base = try uniform(sc, &.{24}, .float32, -0.5, 0.5, seed + 7) },
        .w = try uniform(sc, &.{d}, .bfloat16, 0.5, 1.5, seed + 8),
    };
}

test "dsv41 wide tol: the FFN-side hc expand and norm in one launch are the stock launches' stream, split and norm within 1% at prompt widths; a dropped norm weight fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    const unset = env.override(.MLX_SERVE_DSV41_HC_EXPAND_NORM, "0");
    defer unset.restore();
    var ok = true;
    for ([_]c_int{ 17, 256, 1027, 4096, 8192 }, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0x4C00 + 16 * @as(u64, i));
        const want = try dsv41.hcExpandPreNorm(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w);
        const on = env.override(.MLX_SERVE_DSV41_HC_EXPAND_NORM, "1");
        defer on.restore();
        const got = (try dsv41.hcExpandNormFresh(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w)) orelse return error.FusedHcRefused;
        const served = try dsv41.hcExpandPreNorm(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, r.w);
        const names = [_][]const u8{ "x1", "h", "pre", "post", "comb" };
        for ([_]dsv41.Staged{ got, served }) |g| for ([_][2]A{ .{ want.x1, g.x1 }, .{ want.h, g.h }, .{ want.sp.pre, g.sp.pre }, .{ want.sp.post, g.sp.post }, .{ want.sp.comb, g.sp.comb } }, names) |pr, nm| {
            var lb: [64]u8 = undefined;
            if (!try close(try std.fmt.bufPrint(&lb, "hc expand norm rows {d} {s}", .{ L, nm }), pr[0], pr[1], null, tol_kernel)) ok = false;
        };
        // Planted: the fused launch without the norm's weight (ones in its place).
        const bad = (try dsv41.hcExpandNormFresh(&sc, &m, r.x, r.y, &r.sp, r.cpre, &r.hc, try sc.ones(&.{@intCast(hidden)}, .bfloat16))) orelse return error.FusedHcRefused;
        try testing.expect(!try close("  planted: norm weight dropped", want.h, bad.h, null, tol_kernel));
    }
    try testing.expect(ok);
}

/// hc_v2's prompt widths: below a threadgroup's 64 rows, the chunk tail, a lead half, an odd width, a full chunk.
const v2_widths = [_]c_int{ 37, 1254, 4096, 5001, 8192 };

fn v2Close(lb: []u8, what: []const u8, L: c_int, nm: []const u8, want: A, got: A) !bool {
    return close(try std.fmt.bufPrint(lb, "hc_v2 {s} rows {d} {s}", .{ what, L, nm }), want, got, null, tol_kernel);
}

/// The mixes, split and normed collapse of `got` against the stock launches' (`want`), each printed.
fn v2Same(what: []const u8, L: c_int, want_z: A, want: dsv41.Normed, got: hv2.Res) !bool {
    var lb: [96]u8 = undefined;
    var ok = true;
    const names = [_][]const u8{ "z", "h", "pre", "post", "comb" };
    for ([_][2]A{ .{ want_z, got.z }, .{ want.h, got.n.h }, .{ want.sp.pre, got.n.sp.pre }, .{ want.sp.post, got.n.sp.post }, .{ want.sp.comb, got.n.sp.comb } }, names) |pr, nm| {
        if (pr[0].ctx == null) continue;
        if (!try v2Close(&lb, what, L, nm, pr[0], pr[1])) ok = false;
    }
    return ok;
}

test "dsv41 wide tol: hc_v2's attention side is hc_mix_norm's mixes, split and norm within 1% at prompt widths; the wrong previous mix fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    var ok = true;
    for (v2_widths, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0xA200 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        const want = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, null, eps);
        const got = try hv2.attn(&sc, &m, r.x, r.cpre, &r.hc, r.w);
        if (!try v2Same("attn", L, want.z, want.n, got)) ok = false;
        // Planted: the collapse with this mix's own pre in place of the previous mix's.
        const bad = try hv2.attn(&sc, &m, r.x, try sc.add(r.cpre, try sc.f(0.5)), &r.hc, r.w);
        try testing.expect(!try close("  planted: the wrong previous mix", want.n.h, bad.n.h, null, tol_kernel));
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: hc_v2's FFN side is the stock expand, norm and mixes' split and norm within 1% at prompt widths; a dropped norm weight and mix weights on the wrong copy fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    var ok = true;
    for (v2_widths, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0xA400 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        // The stock launches: glm5's expand, hc_mix over x1, hc_norm's split and normed collapse.
        const x1 = try dsv41.hcExpand(&sc, r.x, r.y, &r.sp);
        const z = try dsv41.hcMixWide(&sc, &m, x1, &r.hc, eps);
        const want = try dsv41.hcNorm(&sc, &m, z, x1, r.cpre, r.w, eps);
        const sa: dsv41.Split = .{ .pre = r.cpre, .post = r.sp.post, .comb = r.sp.comb };
        const got = try hv2.ffn(&sc, &m, r.x, r.y, &sa, &r.hc, r.w);
        if (!try v2Same("ffn", L, z, want, got)) ok = false;
        const bad = try hv2.ffn(&sc, &m, r.x, r.y, &sa, &r.hc, try sc.ones(&.{@intCast(hidden)}, .bfloat16));
        try testing.expect(!try close("  planted: norm weight dropped", want.h, bad.n.h, null, tol_kernel));
        // Planted: the mix weights read one copy over (fn's columns rolled by 5120).
        var hc_bad = r.hc;
        const n4: usize = 4 * hidden;
        hc_bad.fn_w = try sc.concat(&.{ try sc.slice(r.hc.fn_w, 1, hidden, n4), try sc.slice(r.hc.fn_w, 1, 0, hidden) }, 1);
        const bad2 = try hv2.ffn(&sc, &m, r.x, r.y, &sa, &hc_bad, r.w);
        try testing.expect(!try close("  planted: mix weights on the wrong copy", z, bad2.z, null, tol_kernel));
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: hc_v2's tail is glm5's two expands' stream and the next layer's hc_mix_norm within 1% at prompt widths; a transposed comb fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    var ok = true;
    for (v2_widths, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0xA600 + 16 * @as(u64, i));
        const f = try hcRows(&sc, L, 0xA700 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        const sa: dsv41.Split = .{ .pre = r.cpre, .post = r.sp.post, .comb = r.sp.comb };
        const sf: dsv41.Split = .{ .pre = f.cpre, .post = f.sp.post, .comb = f.sp.comb };
        // The stock launches: x1, then the tail's x2, then the next layer's fused mix and norm over x2.
        const x1 = try dsv41.hcExpand(&sc, r.x, r.y, &sa);
        const x2 = try dsv41.hcExpand(&sc, x1, f.y, &sf);
        const want = try dsv41.hcMixNormFused(&sc, &m, x2, f.cpre, &f.hc, f.w, null, eps);
        const got = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .rows = f.y }, &sf, .{ .hc = &f.hc, .norm_w = f.w });
        var lb: [96]u8 = undefined;
        if (!try v2Close(&lb, "tail", L, "x2", x2, got.x2)) ok = false;
        if (!try v2Same("tail next", L, want.z, want.n, got.next.?)) ok = false;
        const only = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .rows = f.y }, &sf, null);
        if (!try v2Close(&lb, "tail only", L, "x2", x2, only.x2)) ok = false;
        // Planted: the FFN split's comb transposed (the residual mixed the wrong way round).
        const bad_sf: dsv41.Split = .{ .pre = f.cpre, .post = f.sp.post, .comb = try sc.transpose(f.sp.comb, &.{ 0, 2, 1 }) };
        const bad = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .rows = f.y }, &bad_sf, null);
        try testing.expect(!try close("  planted: comb transposed", x2, bad.x2, x1, tol_kernel));
    }
    try testing.expect(ok);
}

/// A wide layer's MoE parts over `L` rows: sorted down rows, the unsort (a random permutation) and shared rows; and
/// moe_tail's combine of them as ops (f32 route sum plus the shared row, rounded to bf16).
fn moeParts(sc: *dsv41.Scope, L: c_int, seed: u64) !struct { p: dsv41.MoeParts, y: A } {
    const d: c_int = @intCast(hidden);
    const n: usize = @as(usize, @intCast(L)) * hv2.K;
    const perm = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(perm);
    for (perm, 0..) |*v, q| v.* = @intCast(q);
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().shuffle(u32, perm);
    const inv = try sc.keep(mlx.mlx_array_new_data(perm.ptr, &[_]c_int{@intCast(n)}, 1, .uint32));
    const dn = try normal(sc, &.{ @intCast(n), d }, .bfloat16, 1, seed + 1);
    const sh = try normal(sc, &.{ L, d }, .bfloat16, 1, seed + 2);
    const rows = try sc.reshape(try sc.astype(try sc.take(dn, inv, 0), .float32), &.{ L, @intCast(hv2.K), d });
    var o = mlx.mlx_array_new();
    const sum = try sc.res(mlx.mlx_sum_axis(&o, rows, 1, false, sc.s), &o);
    const y = try sc.astype(try sc.add(sum, try sc.astype(sh, .float32)), .bfloat16);
    return .{ .p = .{ .dn = dn, .inv = inv, .sh = sh }, .y = y };
}

test "dsv41 wide tol: hc_v2's combining tail is moe_tail's combine, glm5's expands and the next layer's hc_mix_norm within 1% at prompt widths; the unsort off by a row fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    var ok = true;
    for (v2_widths, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0xA800 + 16 * @as(u64, i));
        const f = try hcRows(&sc, L, 0xA900 + 16 * @as(u64, i));
        const mp = try moeParts(&sc, L, 0xAA00 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        const sa: dsv41.Split = .{ .pre = r.cpre, .post = r.sp.post, .comb = r.sp.comb };
        const sf: dsv41.Split = .{ .pre = f.cpre, .post = f.sp.post, .comb = f.sp.comb };
        const x1 = try dsv41.hcExpand(&sc, r.x, r.y, &sa);
        const x2 = try dsv41.hcExpand(&sc, x1, mp.y, &sf);
        const want = try dsv41.hcMixNormFused(&sc, &m, x2, f.cpre, &f.hc, f.w, null, eps);
        const got = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .parts = mp.p }, &sf, .{ .hc = &f.hc, .norm_w = f.w });
        var lb: [96]u8 = undefined;
        if (!try v2Close(&lb, "tail combine", L, "x2", x2, got.x2)) ok = false;
        if (!try v2Same("tail combine next", L, want.z, want.n, got.next.?)) ok = false;
        const only = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .parts = mp.p }, &sf, null);
        if (!try v2Close(&lb, "tail combine only", L, "x2", x2, only.x2)) ok = false;
        // Planted: the unsort off by a row (each row combines its neighbour's routes).
        const n: c_int = L * @as(c_int, @intCast(hv2.K));
        const rolled_inv = try sc.concat(&.{ try sc.slice(mp.p.inv, 0, hv2.K, @intCast(n)), try sc.slice(mp.p.inv, 0, 0, hv2.K) }, 0);
        var bad_p = mp.p;
        bad_p.inv = rolled_inv;
        const bad = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .parts = bad_p }, &sf, null);
        try testing.expect(!try close("  planted: unsort off by a row", x2, bad.x2, x1, tol_kernel));
    }
    try testing.expect(ok);
}

/// hc's float64 math over one row's stream `s` [HC * D]: its mixes under `fnw`, `scale` and `base` (`z`), their split,
/// and the collapse with `cpre` rms-normed by `w` (`h`); `eps` the rms's and the split's.
const HcRef = struct { z: [hv2.MIX]f64, pre: [hv2.HC]f64, post: [hv2.HC]f64, comb: [hv2.HC * hv2.HC]f64 };
fn hcRef(s: []const f64, fnw: []const f32, scale: []const f32, base: []const f32, cpre: []const f32, w: []const f32, eps: []const f32, iters: usize, o: *HcRef, h: []f64) void {
    const HC = hv2.HC;
    const D = hv2.D;
    const N = hv2.N;
    var sq: f64 = 0;
    for (s) |sv| sq += sv * sv;
    const rr = 1 / @sqrt(sq / @as(f64, @floatFromInt(N)) + @as(f64, eps[0]));
    for (0..hv2.MIX) |mi| {
        var d: f64 = 0;
        for (s, fnw[mi * N ..][0..N]) |sv, fv| d += sv * @as(f64, fv);
        o.z[mi] = d * rr * @as(f64, scale[mi]) + @as(f64, base[mi]);
    }
    const e: f64 = eps[1];
    for (0..HC) |j| {
        o.pre[j] = 1 / (1 + @exp(-o.z[j])) + e;
        o.post[j] = 2 / (1 + @exp(-o.z[HC + j]));
    }
    const v = &o.comb;
    for (0..HC) |a| {
        const zr = o.z[2 * HC + a * HC ..][0..HC];
        const mx = @max(@max(zr[0], zr[1]), @max(zr[2], zr[3]));
        var t: f64 = 0;
        for (0..HC) |b| {
            v[a * HC + b] = @exp(zr[b] - mx);
            t += v[a * HC + b];
        }
        for (0..HC) |b| v[a * HC + b] = v[a * HC + b] / t + e;
    }
    // Sinkhorn: 2 iters - 1 normalizations, columns first.
    for (0..2 * iters - 1) |it| {
        const by_row = it % 2 == 1;
        var sums: [HC]f64 = @splat(0);
        for (0..HC) |a| for (0..HC) |b| {
            sums[if (by_row) a else b] += v[a * HC + b];
        };
        for (0..HC) |a| for (0..HC) |b| {
            v[a * HC + b] /= sums[if (by_row) a else b] + e;
        };
    }
    var hs: f64 = 0;
    for (0..D) |d| {
        var x: f64 = 0;
        for (0..HC) |i| x += @as(f64, cpre[i]) * s[i * D + d];
        h[d] = x;
        hs += x * x;
    }
    const inv = 1 / @sqrt(hs / @as(f64, @floatFromInt(D)) + @as(f64, eps[0]));
    for (h, w) |*x, wv| x.* *= inv * @as(f64, wv);
}

/// `out` = post y + comb^T x over one row in float64 (`x`, `out` [HC * D]; comb [j][i] row-major).
fn expandRef(post: []const f32, comb: []const f32, y: []const f64, x: []const f64, out: []f64) void {
    const HC = hv2.HC;
    const D = hv2.D;
    for (0..HC) |i| for (0..D) |d| {
        var acc = @as(f64, post[i]) * y[d];
        for (0..HC) |j| acc += @as(f64, comb[j * HC + i]) * x[j * D + d];
        out[i * D + d] = acc;
    };
}

/// PF_HC_V2's reorder bound on output `k` (`names` order) of side `si` (`sides` order), from float64 (`e`) against the
/// five passes' (`ref`): the attention side's f32 mixes and split within 1e-5, x2 the five passes' words (`differ`
/// none), every other output's l2 within 1.01x of theirs.
fn v2Bound(si: usize, k: usize, e: F64Err, ref: F64Err, differ: usize) bool {
    if (k == 5) return differ == 0;
    if (si == 0 and k < 4) return e.l2 < 1e-5 and e.max < 1e-5;
    return e.l2 <= 1.01 * ref.l2;
}

/// The rows a float64 check reads at `L` rows: the ends, a threadgroup's edge and 14 evenly between (ascending, distinct).
fn checkRows(buf: *[32]i32, L: usize) []i32 {
    var n: usize = 0;
    const edge = [_]usize{ 0, 1, 2, 63, 64, 65, L - 2, L - 1 };
    for (edge) |r| if (r < L) {
        buf[n] = @intCast(r);
        n += 1;
    };
    for (0..14) |i| {
        buf[n] = @intCast(i * (L - 1) / 13);
        n += 1;
    }
    std.mem.sort(i32, buf[0..n], {}, std.sort.asc(i32));
    var k: usize = 0;
    for (buf[0..n]) |r| if (k == 0 or buf[k - 1] != r) {
        buf[k] = r;
        k += 1;
    };
    return buf[0..k];
}

test "dsv41 wide tol: MLX_SERVE_DSV41_PF_HC_V2's three passes (the attention side, the FFN side, the combining tail with the next layer's attention side) are a reorder of the five passes (=0): the attention side's f32 mixes and split within 1e-5 of float64 in l2 and max, every other output within 1.01x of the five passes' float64 l2, the tail's x2 their words, at prompt widths; the wrong previous mix, a dropped norm weight and a transposed comb fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    const HC = hv2.HC;
    const D = hv2.D;
    const K = hv2.K;
    const iters: usize = m.cfg.dsv4_hc_sinkhorn_iters;
    // Each side's outputs: the mixes, the split, the normed collapse, and the tail's x2.
    const names = [_][]const u8{ "z", "pre", "post", "comb", "h", "x2" };
    const width = [_]usize{ hv2.MIX, HC, HC, HC * HC, D, HC * D };
    const sides = [_][]const u8{ "attention side", "FFN side", "combining tail" };
    var ok = true;
    for (v2_widths, 0..) |Lc, wi| {
        const L: usize = @intCast(Lc);
        var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, Lc, 0xB000 + 16 * @as(u64, wi));
        const f = try hcRows(&sc, Lc, 0xB100 + 16 * @as(u64, wi));
        const mp = try moeParts(&sc, Lc, 0xB200 + 16 * @as(u64, wi));
        const eps = try dsv41.hcEps(&sc, &m);
        const sa: dsv41.Split = .{ .pre = r.cpre, .post = r.sp.post, .comb = r.sp.comb };
        const sf: dsv41.Split = .{ .pre = f.cpre, .post = f.sp.post, .comb = f.sp.comb };
        const nx: hv2.Next = .{ .hc = &f.hc, .norm_w = f.w };
        // The switch's passes, the five passes as a wide layer serves them at =0, and a planted bug in each side.
        const n_a = try hv2.attn(&sc, &m, r.x, r.cpre, &r.hc, r.w);
        const n_f = try hv2.ffn(&sc, &m, r.x, r.y, &sa, &r.hc, r.w);
        const n_t = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .parts = mp.p }, &sf, nx);
        const o_a = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, null, eps);
        const o_f = try dsv41.hcExpandPreNorm(&sc, &m, r.x, r.y, &sa, sa.pre, &r.hc, r.w);
        const o_fz = try dsv41.hcMixWide(&sc, &m, o_f.x1, &r.hc, eps);
        const o_x2 = try mtail.launch(&sc, mp.p, o_f.x1, &sf);
        const o_n = try dsv41.hcMixNormFused(&sc, &m, o_x2, f.cpre, &f.hc, f.w, null, eps);
        const p_a = try hv2.attn(&sc, &m, r.x, try sc.add(r.cpre, try sc.f(0.5)), &r.hc, r.w);
        const p_f = try hv2.ffn(&sc, &m, r.x, r.y, &sa, &r.hc, try sc.ones(&.{@intCast(hidden)}, .bfloat16));
        const sf_t: dsv41.Split = .{ .pre = f.cpre, .post = f.sp.post, .comb = try sc.transpose(f.sp.comb, &.{ 0, 2, 1 }) };
        const p_t = try hv2.tail(&sc, &m, r.x, r.y, &sa, .{ .parts = mp.p }, &sf_t, nx);
        const nt = n_t.next.?;
        const pt = p_t.next.?;
        // [side][arm: switch, =0, planted][output]; the attention and FFN sides hold no x2.
        const outs = [3][3][6]A{
            .{ .{ n_a.z, n_a.n.sp.pre, n_a.n.sp.post, n_a.n.sp.comb, n_a.n.h, none }, .{ o_a.z, o_a.n.sp.pre, o_a.n.sp.post, o_a.n.sp.comb, o_a.n.h, none }, .{ p_a.z, p_a.n.sp.pre, p_a.n.sp.post, p_a.n.sp.comb, p_a.n.h, none } },
            .{ .{ n_f.z, n_f.n.sp.pre, n_f.n.sp.post, n_f.n.sp.comb, n_f.n.h, none }, .{ o_fz, o_f.sp.pre, o_f.sp.post, o_f.sp.comb, o_f.h, none }, .{ p_f.z, p_f.n.sp.pre, p_f.n.sp.post, p_f.n.sp.comb, p_f.n.h, none } },
            .{ .{ nt.z, nt.n.sp.pre, nt.n.sp.post, nt.n.sp.comb, nt.n.h, n_t.x2 }, .{ o_n.z, o_n.n.sp.pre, o_n.n.sp.post, o_n.n.sp.comb, o_n.n.h, o_x2 }, .{ pt.z, pt.n.sp.pre, pt.n.sp.post, pt.n.sp.comb, pt.n.h, p_t.x2 } },
        };
        var rb: [32]i32 = undefined;
        const rows = checkRows(&rb, L);
        const R = rows.len;
        const at = try sc.ints(rows, &.{@intCast(R)});
        const H = struct {
            fn rowsOf(s: *dsv41.Scope, x: A, a: A) ![]f32 {
                return dsv41.hostF32(try s.take(x, a, 0));
            }
        };
        // The inputs at the check rows.
        const xs = try H.rowsOf(&sc, r.x, at);
        defer gpa.free(xs);
        const ys = try H.rowsOf(&sc, r.y, at);
        defer gpa.free(ys);
        const sa_post = try H.rowsOf(&sc, r.sp.post, at);
        defer gpa.free(sa_post);
        const sa_comb = try H.rowsOf(&sc, r.sp.comb, at);
        defer gpa.free(sa_comb);
        const sa_pre = try H.rowsOf(&sc, r.cpre, at);
        defer gpa.free(sa_pre);
        const sf_post = try H.rowsOf(&sc, f.sp.post, at);
        defer gpa.free(sf_post);
        const sf_comb = try H.rowsOf(&sc, f.sp.comb, at);
        defer gpa.free(sf_comb);
        const sf_pre = try H.rowsOf(&sc, f.cpre, at);
        defer gpa.free(sf_pre);
        const inv_at = try sc.reshape(try sc.take(try sc.reshape(mp.p.inv, &.{ Lc, @intCast(K) }), at, 0), &.{@intCast(R * K)});
        const dns = try H.rowsOf(&sc, mp.p.dn, inv_at);
        defer gpa.free(dns);
        const shs = try H.rowsOf(&sc, mp.p.sh, at);
        defer gpa.free(shs);
        var hw: [2][4][]f32 = undefined;
        for ([_]*const HcRows{ &r, &f }, &hw) |src, *dst| for ([_]A{ src.hc.fn_w, src.hc.scale, src.hc.base, src.w }, dst) |a, *d| {
            d.* = try dsv41.hostF32(a);
        };
        defer for (hw) |q| for (q) |d| gpa.free(d);
        const epsh = try dsv41.hostF32(eps);
        defer gpa.free(epsh);
        // float64 at the check rows, [side][output].
        var want: [3][6][]f64 = undefined;
        for (&want) |*sd| for (sd, width) |*w, wd| {
            w.* = try gpa.alloc(f64, R * wd);
        };
        defer for (want) |sd| for (sd) |w| gpa.free(w);
        const buf = try gpa.alloc(f64, 4 * HC * D + D);
        defer gpa.free(buf);
        const s0 = buf[0 .. HC * D];
        const s1 = buf[HC * D ..][0 .. HC * D];
        const s2 = buf[2 * HC * D ..][0 .. HC * D];
        const yv = buf[3 * HC * D ..][0..D];
        for (0..R) |ri| {
            for (s0, xs[ri * HC * D ..][0 .. HC * D]) |*d, v| d.* = v;
            for (yv, ys[ri * D ..][0..D]) |*d, v| d.* = v;
            expandRef(sa_post[ri * HC ..][0..HC], sa_comb[ri * HC * HC ..][0 .. HC * HC], yv, s0, s1);
            for (yv, 0..) |*d, c| {
                var acc: f64 = shs[ri * D + c];
                for (0..K) |k| acc += dns[(ri * K + k) * D + c];
                d.* = acc;
            }
            expandRef(sf_post[ri * HC ..][0..HC], sf_comb[ri * HC * HC ..][0 .. HC * HC], yv, s1, s2);
            @memcpy(want[2][5][ri * HC * D ..][0 .. HC * D], s2);
            for ([_][]const f64{ s0, s1, s2 }, [_][]const f32{ sa_pre, sa_pre, sf_pre }, [_]usize{ 0, 0, 1 }, 0..) |st, cp, wsrc, si| {
                var o: HcRef = undefined;
                const hw_ = hw[wsrc];
                hcRef(st, hw_[0], hw_[1], hw_[2], cp[ri * HC ..][0..HC], hw_[3], epsh, iters, &o, want[si][4][ri * D ..][0..D]);
                @memcpy(want[si][0][ri * hv2.MIX ..][0..hv2.MIX], &o.z);
                @memcpy(want[si][1][ri * HC ..][0..HC], &o.pre);
                @memcpy(want[si][2][ri * HC ..][0..HC], &o.post);
                @memcpy(want[si][3][ri * HC * HC ..][0 .. HC * HC], &o.comb);
            }
        }
        for (0..3) |si| {
            var side_ok = true;
            var planted_ok = true;
            for (0..6) |k| {
                if (outs[si][0][k].ctx == null) continue;
                var g: [3][]f32 = undefined;
                for (&g, 0..) |*v, ai| v.* = try H.rowsOf(&sc, outs[si][ai][k], at);
                defer for (g) |v| gpa.free(v);
                const en = f64Err(g[0], want[si][k]);
                const er = f64Err(g[1], want[si][k]);
                const ep = f64Err(g[2], want[si][k]);
                const differ = try wordsDiffer(&sc, outs[si][0][k], outs[si][1][k]);
                const differ_p = if (k == 5) try wordsDiffer(&sc, outs[si][2][k], outs[si][1][k]) else 0;
                const in_bound = v2Bound(si, k, en, er, differ);
                std.debug.print("[hcv2-f64] rows {d} {s} {s}: from float64 ({d} rows) PF_HC_V2 l2 {e:.4} max {e:.4}, =0 l2 {e:.4} max {e:.4}, planted l2 {e:.3} max {e:.3}; {d} of {d} words differ from =0{s}\n", .{ L, sides[si], names[k], R, en.l2, en.max, er.l2, er.max, ep.l2, ep.max, differ, mlx.mlx_array_size(outs[si][0][k]), if (in_bound) "" else "; PF_HC_V2 OUTSIDE ITS REORDER BOUND" });
                if (!in_bound) side_ok = false;
                if (!v2Bound(si, k, ep, er, differ_p)) planted_ok = false;
            }
            if (!side_ok) ok = false;
            try testing.expect(!planted_ok);
        }
    }
    try testing.expect(ok);
}

// ── a release-shape pair worker's prefill ──

const switches = .{ .MLX_SERVE_DSV41_MOE_PAIR_FP4, .MLX_SERVE_DSV41_HC_EXPAND_NORM, .MLX_SERVE_DSV41_WOA_JOIN, .MLX_SERVE_DSV41_MOE_TAIL, .MLX_SERVE_DSV41_PF_HC_V2, .MLX_SERVE_DSV41_PF_MOE_EPI, .MLX_SERVE_DSV41_PF_IDX_NAX };

/// The flash kernel on `fx`'s model as its load sets it (`plugin.configureLoad`): the gather's kernels rebuilt with
/// MLX_SERVE_DSV41_FLASH_ATTN on (and MLX_SERVE_DSV41_FLASH_TRIM, MLX_SERVE_DSV41_FLASH_PSUM as asked) and the ring
/// off; off puts the stock gather back.
fn flashOn(fx: *fixture.Fixture, on: bool, trim: bool, psum: bool) !void {
    fa.ps().mode = if (on) .on else .off;
    fa.ps().trim = on and trim;
    fa.ps().psum = on and psum;
    fa.ps().explicit = false;
    sat.ps().ring_regions = !on;
    if (fx.m.attn_gather) |*g| g.deinit();
    fx.m.attn_gather = null;
    if (device.naxAvailable()) fx.m.attn_gather = try dsv41.GatherAttn.init();
}

/// The pair worker's streams and mixes after its layers over `x` / `pre`, from a fresh state (caller frees).
fn prefill(fx: *fixture.Fixture, x: A, pre: A) ![2]A {
    return dsv41.workerStep(fx.m, testing.allocator, x, pre, 0, false, false);
}

/// A pair worker's layers move its stream through rounded bf16 sums over two layers, so its tolerance is 5% of the
/// layers' change to the stream (and of the mixes), in norm only: a reordered sum may flip a near-tie top-6 route, which
/// moves that row by one expert's share.
const tol_layers: f32 = 5e-2;

/// `prefill` with every route's ids recorded in `tap`.
fn prefillTapped(fx: *fixture.Fixture, x: A, pre: A, tap: *dsv41.RouteTap) ![2]A {
    dsv41.route_tap = tap;
    defer dsv41.route_tap = null;
    return prefill(fx, x, pre);
}

fn freeTap(t: *dsv41.RouteTap) void {
    for (t.ids.items) |a| _ = mlx.mlx_array_free(a);
    t.ids.deinit(t.gpa);
}

fn idsOf(gpa: std.mem.Allocator, x: A) ![]u32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const raw = mlx.mlx_array_data_uint32(c) orelse return error.NoData;
    return gpa.dupe(u32, raw[0..mlx.mlx_array_size(c)]);
}

/// Each route call's rows whose top-k expert set differs between `want` and `got` (near-tie flips under a reorder).
fn printFlips(what: []const u8, want: *const dsv41.RouteTap, got: *const dsv41.RouteTap) !void {
    if (want.ids.items.len != got.ids.items.len) {
        std.debug.print("{s}: route calls {d} vs the stock path's {d}\n", .{ what, got.ids.items.len, want.ids.items.len });
        return;
    }
    const gpa = testing.allocator;
    for (want.ids.items, got.ids.items, 0..) |w, g, i| {
        const sh = mlx.getShape(w);
        const L: usize = @intCast(sh[0]);
        const k: usize = @intCast(sh[1]);
        const a = try idsOf(gpa, w);
        defer gpa.free(a);
        const b = try idsOf(gpa, g);
        defer gpa.free(b);
        var n: usize = 0;
        for (0..L) |l| {
            const ra = a[l * k ..][0..k];
            const rb = b[l * k ..][0..k];
            std.mem.sort(u32, ra, {}, std.sort.asc(u32));
            std.mem.sort(u32, rb, {}, std.sort.asc(u32));
            if (!std.mem.eql(u32, ra, rb)) n += 1;
        }
        std.debug.print("{s}: route call {d}: top-{d} flipped on {d} of {d} rows\n", .{ what, i, k, n, L });
    }
}

/// Whether the pair worker's prefill is within `tol_layers` of `want` (the stream measured on its change from `x`);
/// prints each route call's flipped rows against `want_tap`.
fn closeAs(fx: *fixture.Fixture, x: A, pre: A, want: [2]A, want_tap: *const dsv41.RouteTap, label: []const u8, what: []const u8) !bool {
    var tap: dsv41.RouteTap = .{ .gpa = testing.allocator };
    defer freeTap(&tap);
    const got = try prefillTapped(fx, x, pre, &tap);
    defer for (got) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var lb: [160]u8 = undefined;
    try printFlips(try std.fmt.bufPrint(&lb, "wide layer {s}, {s}", .{ label, what }), want_tap, &tap);
    var ok = true;
    for (want, got, 0..) |w, g, k| {
        const ref: ?A = if (k == 0 and std.mem.eql(c_int, mlx.getShape(w), mlx.getShape(x))) x else null;
        const e = try errOf(w, g, ref);
        std.debug.print("wide layer {s}, {s}, output {d}: l2 {e:.3} max {e:.3} nan_same {}\n", .{ label, what, k, e.l2, e.max, e.nan_same });
        if (!(e.nan_same and e.l2 <= tol_layers)) ok = false;
    }
    return ok;
}

test "dsv41 wide tol: each switch, and all of them, keep a release-shape pair worker's prefill streams and mixes within 5% of the stock path's, MXFP4 and NVFP4 experts with per-expert scales; the routed term at half its scale fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    inline for (switches) |n| env.set(n, "0");
    // A ratio-2 pair (layers 8, 9) and a ratio-1 pair (20, 21), each range its own KV and index source and no Engram
    // layer (14 reads Engram, which a pair worker's fixture does not hold).
    const Case = struct { lo: usize, rows: usize };
    // Every case and switch runs; each out of tolerance is reported.
    var ok = true;
    for ([_]Case{ .{ .lo = 8, .rows = 5000 }, .{ .lo = 20, .rows = 8192 } }) |c| inline for (.{ false, true }) |nv| {
        var lb: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&lb, "{d}-{d} {s} at {d} rows", .{ c.lo, c.lo + 1, if (nv) "NVFP4" else "MXFP4", c.rows });
        const fx = fixture.Fixture.init(gpa, .{ .lo = c.lo, .hi = c.lo + 2, .ctx = 0, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none }) catch |e| {
            std.debug.print("wide layer {s}: the fixture refused: {s}\n", .{ label, @errorName(e) });
            return e;
        };
        defer fx.deinit();
        const x = try fx.stream(c.rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(c.rows);
        defer _ = mlx.mlx_array_free(pre);
        var want_tap: dsv41.RouteTap = .{ .gpa = gpa };
        defer freeTap(&want_tap);
        const want = prefillTapped(fx, x, pre, &want_tap) catch |e| {
            std.debug.print("wide layer {s}: the stock prefill failed: {s}\n", .{ label, @errorName(e) });
            return e;
        };
        defer for (want) |o| {
            _ = mlx.mlx_array_free(o);
        };
        // The layer's loaded fp4 experts, NVFP4 or MXFP4, take the paired kernel under MOE_PAIR_FP4.
        {
            var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            const ly = fx.layer(c.lo);
            const h = try normal(&sc, &.{ @intCast(c.rows), @intCast(hidden) }, .bfloat16, 1, 0x77);
            const r = dsv41.route(&sc, fx.m, ly, h, none) catch |e| {
                std.debug.print("wide layer {s}: the route failed: {s}\n", .{ label, @errorName(e) });
                return e;
            };
            env.set(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "1");
            defer env.set(.MLX_SERVE_DSV41_MOE_PAIR_FP4, "0");
            const served = (actOf(&sc, fx.m, &ly.experts, h, r.idx, r.wts) catch |e| {
                std.debug.print("wide layer {s}: the paired kernel failed: {s}\n", .{ label, @errorName(e) });
                return e;
            }) != null;
            if (!served) std.debug.print("wide layer {s}: the paired kernel {s} the loaded experts (gu {}, mode {s})\n", .{ label, if (served) "takes" else "declines", ly.experts.gu != null, if (ly.experts.gu) |g| @tagName(g.qp.mode) else "-" });
            try testing.expect(served);
        }
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "the stock path again")) ok = false;
        inline for (switches) |n| {
            env.set(n, "1");
            defer env.set(n, "0");
            if (!try closeAs(fx, x, pre, want, &want_tap, label, @tagName(n))) ok = false;
        }
        // FLASH_TRIM is FLASH_ATTN's values (on by default where the plugin loads): its prefill is FLASH_ATTN's, word
        // for word.
        {
            try flashOn(fx, true, false, false);
            defer flashOn(fx, false, false, false) catch {};
            const want_fa = try prefill(fx, x, pre);
            defer for (want_fa) |o| {
                _ = mlx.mlx_array_free(o);
            };
            try flashOn(fx, true, true, false);
            const got = try prefill(fx, x, pre);
            defer for (got) |o| {
                _ = mlx.mlx_array_free(o);
            };
            for (want_fa, got, 0..) |w, g, k| {
                const same = try sameWords(w, g);
                std.debug.print("wide layer {s}, MLX_SERVE_DSV41_FLASH_TRIM against MLX_SERVE_DSV41_FLASH_ATTN, output {d}: word for word {}\n", .{ label, k, same });
                if (!same) ok = false;
            }
        }
        // FLASH_PSUM replaces FLASH_ATTN's sums, alone and beside FLASH_TRIM, so it is held to FLASH_ATTN's prefill;
        // FLASH_ATTN's own distance from the gather is printed beside it.
        {
            var fa_tap: dsv41.RouteTap = .{ .gpa = gpa };
            defer freeTap(&fa_tap);
            try flashOn(fx, true, false, false);
            const want_fa = prefillTapped(fx, x, pre, &fa_tap) catch |e| {
                std.debug.print("wide layer {s}: the FLASH_ATTN prefill failed: {s}\n", .{ label, @errorName(e) });
                return e;
            };
            defer for (want_fa) |o| {
                _ = mlx.mlx_array_free(o);
            };
            try flashOn(fx, false, false, false);
            _ = try closeAs(fx, x, pre, want_fa, &fa_tap, label, "the gather against MLX_SERVE_DSV41_FLASH_ATTN (not gated)");
            defer flashOn(fx, false, false, false) catch {};
            try flashOn(fx, true, false, true);
            if (!try closeAs(fx, x, pre, want_fa, &fa_tap, label, "MLX_SERVE_DSV41_FLASH_PSUM against MLX_SERVE_DSV41_FLASH_ATTN")) ok = false;
            try flashOn(fx, true, true, true);
            if (!try closeAs(fx, x, pre, want_fa, &fa_tap, label, "MLX_SERVE_DSV41_FLASH_TRIM + MLX_SERVE_DSV41_FLASH_PSUM against MLX_SERVE_DSV41_FLASH_ATTN")) ok = false;
        }
        inline for (switches) |n| env.set(n, "1");
        defer {
            inline for (switches) |n| env.set(n, "0");
        }
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "every switch")) ok = false;
        // Planted: every switch on with the routed experts' sum at half its scale.
        const scale = fx.m.cfg.router_scaling_factor;
        fx.m.cfg.router_scaling_factor = scale / 2;
        defer fx.m.cfg.router_scaling_factor = scale;
        try testing.expect(!try closeAs(fx, x, pre, want, &want_tap, label, "planted: the routed term at half its scale"));
    };
    try testing.expect(ok);
}

// ── the flash kernel (flash_attn.zig) at the release's spans: float64 references and the softmax reorders' ubench ──

const fa = @import("flash_attn.zig");
const sat = @import("scratch_attn.zig");
const dattn = @import("../decode/attn.zig");
const ddec = @import("../decode/decode.zig");

const HD: usize = 512;
const HEADS: usize = 64;
const WIN: usize = 128;
const OG: usize = 8;
/// The dims a head keeps unroped (the epilogue's fast trig rotates the last 64).
const PLAIN: usize = HD - 64;

/// A wide span as the prompt layer hands it to the gather: `L` rows at window offset `wp`, `K` picks among `nc`
/// compressed rows (some -1); queries 4x wider than the keys so the scaled logits spread like attention's (std ~4).
const SpanCase = struct { L: usize, wp: usize, nc: usize, K: usize };
const Span = struct {
    c: SpanCase,
    sink: A,
    q: A,
    qt: A,
    keys_w: A,
    comp: A,
    picks: A,
    periods: A,
    meta: A,
    scal: A,
    h_q: []f32,
    h_kw: []f32,
    h_comp: []const f32,
    h_picks: []i32,
    h_sink: []f32,

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HD)));
    const p0: usize = 1000;

    fn init(sc: *dsv41.Scope, c: SpanCase, seed: u64) !Span {
        const gpa = testing.allocator;
        const sink = try normal(sc, &.{ 1, @intCast(HEADS), 1 }, .float32, 1, seed);
        const q = try normal(sc, &.{ @intCast(c.L), @intCast(HEADS), @intCast(HD) }, .bfloat16, 4, seed + 1);
        const qt = try normal(sc, &.{ @intCast(c.L), @intCast(HEADS), 64 }, .bfloat16, 4, seed + 2);
        const keys_w = try normal(sc, &.{ @intCast(c.wp + c.L), @intCast(HD) }, .bfloat16, 1, seed + 3);
        const comp = if (c.K > 0) try normal(sc, &.{ @intCast(c.nc), @intCast(HD) }, .bfloat16, 1, seed + 4) else none;
        const picks_h = try gpa.alloc(i32, @max(c.L * c.K, 1));
        for (picks_h, 0..) |*v, i| v.* = if (c.K == 0) 0 else if (i % 37 == 5) -1 else @intCast((i * 13 + i / c.K * 7) % c.nc);
        const topk = if (c.K > 0) try sc.ints(picks_h, &.{ @intCast(c.L), @intCast(c.K) }) else none;
        var per: [32]f32 = undefined;
        for (&per, 0..) |*v, i| v.* = std.math.pow(f32, 10000, @as(f32, @floatFromInt(i)) / 32);
        const periods = try sc.keep(mlx.mlx_array_new_data(&per, &[_]c_int{32}, 1, .float32));
        const meta = try sc.ints(&.{ @intCast(c.wp), @intCast(WIN), @intCast(c.wp + c.L), @intCast(c.K), @intCast(HEADS), @intCast(HD), @intCast(p0), @intCast(c.L), @intCast(OG) }, &.{9});
        const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32));
        return .{ .c = c, .sink = sink, .q = q, .qt = qt, .keys_w = keys_w, .comp = comp, .picks = if (c.K > 0) topk else meta, .periods = periods, .meta = meta, .scal = scal, .h_q = try dsv41.hostF32(q), .h_kw = try dsv41.hostF32(keys_w), .h_comp = if (c.K > 0) try dsv41.hostF32(comp) else &.{}, .h_picks = picks_h, .h_sink = try dsv41.hostF32(sink) };
    }

    fn deinit(s: *Span) void {
        const gpa = testing.allocator;
        gpa.free(s.h_q);
        gpa.free(s.h_kw);
        if (s.c.K > 0) gpa.free(s.h_comp);
        gpa.free(s.h_picks);
        gpa.free(s.h_sink);
    }

    /// Key `c` of row `l`'s list: a window row, a pick, or absent.
    fn key(s: *const Span, l: usize, c: usize) ?[]const f32 {
        if (c < WIN) {
            const r = @as(isize, @intCast(s.c.wp + l)) - @as(isize, WIN - 1) + @as(isize, @intCast(c));
            if (r < 0 or r >= s.c.wp + s.c.L) return null;
            return s.h_kw[@as(usize, @intCast(r)) * HD ..][0..HD];
        }
        const r = s.h_picks[l * s.c.K + (c - WIN)];
        if (r < 0) return null;
        return s.h_comp[@as(usize, @intCast(r)) * HD ..][0..HD];
    }

    /// Row `l`'s heads in float64 (the sink in the denominator), their unroped dims, into `out` [HEADS, PLAIN].
    fn reference(s: *const Span, l: usize, out: []f64) void {
        const nk = WIN + s.c.K;
        var sv: [WIN + 1024]f64 = undefined;
        for (0..HEADS) |h| {
            const qv = s.h_q[(l * HEADS + h) * HD ..][0..HD];
            var m: f64 = s.h_sink[h];
            for (0..nk) |c| {
                sv[c] = -std.math.inf(f64);
                const k = s.key(l, c) orelse continue;
                var d: f64 = 0;
                for (qv, k) |a, b| d += @as(f64, a) * b;
                sv[c] = d * scale;
                m = @max(m, sv[c]);
            }
            var den = @exp(@as(f64, s.h_sink[h]) - m);
            const o = out[h * PLAIN ..][0..PLAIN];
            @memset(o, 0);
            for (0..nk) |c| {
                const k = s.key(l, c) orelse continue;
                const p = @exp(sv[c] - m);
                den += p;
                for (o, k[0..PLAIN]) |*x, v| x.* += p * v;
            }
            for (o) |*x| x.* /= den;
        }
    }

    /// The flash kernel `k` over the span, heads grouped as `attnOut` reads them, in `ot`.
    fn flash(s: *const Span, sc: *dsv41.Scope, k: mlx.mlx_fast_metal_kernel, qt_on: bool, ot: mlx.mlx_dtype) !A {
        const ins = [_]A{ s.q, s.keys_w, if (s.c.K > 0) s.comp else s.keys_w, s.picks, s.meta, if (qt_on) s.qt else s.q, s.sink, s.scal, s.periods };
        var o: [1]A = .{none};
        try dsv41.flashLaunch(sc, k, &ins, &.{.{ .shape = &.{ @intCast(OG), @intCast(s.c.L), @intCast(HEADS / OG * HD) }, .dt = ot }}, s.c.L, HEADS, qt_on, ot, &o);
        return o[0];
    }
};

/// FLASH_ATTN's text, under its own name.
fn flash1Kernel() !mlx.mlx_fast_metal_kernel {
    return dsv41.buildKernel("dsv41_attn_flash_t1", &dsv41.GatherAttn.FLASH_INS, &.{"o"}, fa.source, dattn.HEADER);
}
/// The flash kernel under FA_TRIM / FA_PSUM (`plant` 1: the window's keys a row late), under its own name.
fn flashKernel(comptime trim: bool, comptime psum: bool, comptime plant: u1) !mlx.mlx_fast_metal_kernel {
    const name = std.fmt.comptimePrint("dsv41_attn_flash_t_{d}{d}{d}", .{ @intFromBool(trim), @intFromBool(psum), plant });
    return dsv41.buildKernel(name, &dsv41.GatherAttn.FLASH_INS, &.{"o"}, fa.source, comptime dattn.HEADER ++ fa.plan.defines(trim, psum, plant));
}
fn flashTrimKernel(comptime plant: u1) !mlx.mlx_fast_metal_kernel {
    return flashKernel(true, false, plant);
}

/// `got` (grouped [OG, L, H / OG * D]) against the float64 heads at rows `rows`, unroped dims: the error's norm over the
/// reference's and its largest element over the reference's largest.
const FErr = struct { l2: f64, max: f64 };
fn spanErr(s: *const Span, got: A, rows: []const usize) !FErr {
    const ref = try spanRefs(s, rows);
    defer testing.allocator.free(ref);
    return spanErrTo(s, got, rows, ref);
}

/// The float64 heads at `rows` (`Span.reference`), [rows, HEADS, PLAIN]; the caller frees.
fn spanRefs(s: *const Span, rows: []const usize) ![]f64 {
    const ref = try testing.allocator.alloc(f64, rows.len * HEADS * PLAIN);
    for (rows, 0..) |l, i| s.reference(l, ref[i * HEADS * PLAIN ..][0 .. HEADS * PLAIN]);
    return ref;
}

/// `spanErr` against the float64 heads `ref` (`spanRefs`).
fn spanErrTo(s: *const Span, got: A, rows: []const usize, ref: []const f64) !FErr {
    const g = try dsv41.hostF32(got);
    defer testing.allocator.free(g);
    var dd: f64 = 0;
    var rr: f64 = 0;
    var dm: f64 = 0;
    var rm: f64 = 0;
    const hg = HEADS / OG;
    for (rows, 0..) |l, i| {
        for (0..HEADS) |h| for (0..PLAIN) |d| {
            const w = ref[(i * HEADS + h) * PLAIN + d];
            const x: f64 = g[((h / hg) * s.c.L + l) * hg * HD + (h % hg) * HD + d];
            dd += (x - w) * (x - w);
            rr += w * w;
            dm = @max(dm, @abs(x - w));
            rm = @max(rm, @abs(w));
        };
    }
    return .{ .l2 = @sqrt(dd / rr), .max = dm / rm };
}

/// `a` against `b` (both grouped) over every row, all dims (the roped tail too), over `b`'s norm.
fn spanDist(a: A, b: A) !f64 {
    const ga = try dsv41.hostF32(a);
    defer testing.allocator.free(ga);
    const gb = try dsv41.hostF32(b);
    defer testing.allocator.free(gb);
    var dd: f64 = 0;
    var rr: f64 = 0;
    for (ga, gb) |x, y| {
        dd += (@as(f64, x) - y) * (@as(f64, x) - y);
        rr += @as(f64, y) * y;
    }
    return @sqrt(dd / rr);
}

const span_cases = [_]SpanCase{
    .{ .L = 256, .wp = 127, .nc = 4096, .K = 512 }, // a full span: the window and 512 picks
    .{ .L = 256, .wp = 0, .nc = 0, .K = 0 }, // the first span, window only
    .{ .L = 200, .wp = 40, .nc = 300, .K = 512 }, // a short window, a short last span
    .{ .L = 64, .wp = 500, .nc = 900, .K = 200 }, // a part-filled last block
};

test "dsv41 wide: FLASH_TRIM's heads are FLASH_ATTN's word for word at the release's spans, both query tails, f32 and bf16 out, so no further from float64; window keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const k1 = try flash1Kernel();
    defer _ = mlx.mlx_fast_metal_kernel_free(k1);
    const k2 = try flashTrimKernel(0);
    defer _ = mlx.mlx_fast_metal_kernel_free(k2);
    const kp = try flashTrimKernel(1);
    defer _ = mlx.mlx_fast_metal_kernel_free(kp);
    var ok = true;
    for (span_cases, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 40 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        var same = true;
        for ([_]bool{ false, true }) |qt_on| for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |ot| {
            if (!try sameWords(try s.flash(&sc, k2, qt_on, ot), try s.flash(&sc, k1, qt_on, ot))) same = false;
        };
        const e1 = try spanErr(&s, try s.flash(&sc, k1, false, .float32), &rows);
        const e2 = try spanErr(&s, try s.flash(&sc, k2, false, .float32), &rows);
        const ep = try spanErr(&s, try s.flash(&sc, kp, false, .float32), &rows);
        std.debug.print("[ftrim] L={d} wp={d} K={d}: FLASH_ATTN's words {}; rel l2 / max vs float64 (f32 out): FLASH_ATTN {e:.3}/{e:.3} FLASH_TRIM {e:.3}/{e:.3} planted {e:.3}/{e:.3}\n", .{ c.L, c.wp, c.K, same, e1.l2, e1.max, e2.l2, e2.max, ep.l2, ep.max });
        ok = ok and same and e2.l2 <= e1.l2 and e2.max <= e1.max;
        // planted: window keys a row late leave FLASH_ATTN's own error far behind
        ok = ok and ep.l2 > 10 * e1.l2;
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: FLASH_PSUM's heads, alone and beside FLASH_TRIM, are no further from float64 than FLASH_ATTN's (norm and worst element) and within its own error of it, at the release's spans, both query tails, f32 and bf16 out; window keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const k1 = try flash1Kernel();
    defer _ = mlx.mlx_fast_metal_kernel_free(k1);
    const Cand = struct { name: []const u8, k: mlx.mlx_fast_metal_kernel, kp: mlx.mlx_fast_metal_kernel };
    const cands = [_]Cand{
        .{ .name = "FLASH_PSUM", .k = try flashKernel(false, true, 0), .kp = try flashKernel(false, true, 1) },
        .{ .name = "FLASH_TRIM + FLASH_PSUM", .k = try flashKernel(true, true, 0), .kp = try flashKernel(true, true, 1) },
    };
    defer for (cands) |c| {
        _ = mlx.mlx_fast_metal_kernel_free(c.k);
        _ = mlx.mlx_fast_metal_kernel_free(c.kp);
    };
    var ok = true;
    for (span_cases, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 40 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        // the float64 reference reads q's own last 64 dims: the unroped dims compared are the same either way
        const e1 = try spanErr(&s, try s.flash(&sc, k1, false, .float32), &rows);
        const b1 = try spanErr(&s, try s.flash(&sc, k1, false, .bfloat16), &rows);
        for (cands) |cd| {
            const e2 = try spanErr(&s, try s.flash(&sc, cd.k, false, .float32), &rows);
            const ep = try spanErr(&s, try s.flash(&sc, cd.kp, false, .float32), &rows);
            var dist: [2]f64 = undefined;
            for ([_]bool{ false, true }, 0..) |qt_on, i| {
                dist[i] = try spanDist(try s.flash(&sc, cd.k, qt_on, .float32), try s.flash(&sc, k1, qt_on, .float32));
            }
            const b2 = try spanErr(&s, try s.flash(&sc, cd.k, false, .bfloat16), &rows);
            std.debug.print("[fpsum] L={d} wp={d} K={d}: rel l2 / max vs float64, f32 out: FLASH_ATTN {e:.3}/{e:.3} {s} {e:.3}/{e:.3} planted {e:.3}/{e:.3}; bf16 out: {e:.3}/{e:.3} vs {e:.3}/{e:.3}; to FLASH_ATTN (all dims, f32): {e:.3}, qt tail {e:.3}\n", .{ c.L, c.wp, c.K, e1.l2, e1.max, cd.name, e2.l2, e2.max, ep.l2, ep.max, b1.l2, b1.max, b2.l2, b2.max, dist[0], dist[1] });
            // no further from float64 than FLASH_ATTN in norm and at its worst element, within two of its errors of
            // it, the served bf16 error kept
            ok = ok and e2.l2 <= e1.l2 and e2.max <= e1.max and dist[0] <= 2 * e1.l2 and dist[1] <= 2 * e1.l2;
            ok = ok and b2.l2 <= b1.l2 * 1.02;
            // planted: window keys a row late leave FLASH_ATTN's own error far behind
            ok = ok and ep.l2 > 10 * e1.l2;
        }
    }
    try testing.expect(ok);
}

test "dsv41 wide tol: MLX_SERVE_DSV41_FLASH_ATTN's heads, alone and beside FLASH_TRIM (served), are no farther from float64 than the gather's (=0) in l2 and max, f32 and bf16 out, at the release's spans the gather serves, through attnGather as the load builds it; window keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const saved = fa.ps().*;
    defer fa.ps().* = saved;
    const arms = [_][]const u8{ "the gather (=0)", "FLASH_ATTN", "FLASH_ATTN + FLASH_TRIM" };
    var g: [3]dsv41.GatherAttn = undefined;
    for (&g, [_]fa.Mode{ .off, .on, .on }, [_]bool{ false, false, true }) |*a, mode, trim| {
        fa.ps().* = .{ .mode = mode, .trim = trim };
        a.* = try dsv41.GatherAttn.init();
    }
    defer for (&g) |*a| a.deinit();
    const kp = try flashKernel(false, false, 1);
    defer _ = mlx.mlx_fast_metal_kernel_free(kp);
    var m: dsv41.Dsv41Model = undefined;
    attnModel(&m, .bfloat16);
    var ok = true;
    for (span_cases, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 40 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        const ref = try spanRefs(&s, &rows);
        defer gpa.free(ref);
        const comp = if (c.K > 0) s.comp else none;
        const topk = if (c.K > 0) s.picks else none;
        var e: [2][3]FErr = undefined;
        var o: [2][3]A = undefined;
        var served: usize = 0;
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }, 0..) |ot, oi| for (&g, 0..) |*a, ai| {
            m.attn_gather = a.*;
            m.act = ot;
            o[oi][ai] = (try dsv41.attnGather(&sc, &m, s.sink, s.q, none, s.keys_w, c.wp, comp, topk, Span.scale, .{ .periods = s.periods, .p0 = Span.p0 })) orelse continue;
            e[oi][ai] = try spanErrTo(&s, o[oi][ai], &rows, ref);
            served += 1;
        };
        // Picks not a multiple of 64: the gather declines and only the flash arms serve (held against the f32 chain by
        // the odd-span test).
        if (c.K % 64 != 0) {
            std.debug.print("[fa-gather] L={d} wp={d} K={d}: the gather declines, the flash arms serve\n", .{ c.L, c.wp, c.K });
            try testing.expectEqual(@as(usize, 4), served);
            continue;
        }
        try testing.expectEqual(@as(usize, 6), served);
        const ep = try spanErrTo(&s, try s.flash(&sc, kp, false, .float32), &rows, ref);
        for (0..2) |oi| {
            const dt = if (oi == 0) "f32" else "bf16";
            for (1..3) |ai| {
                const near = e[oi][ai].l2 <= e[oi][0].l2 and e[oi][ai].max <= e[oi][0].max;
                std.debug.print("[fa-gather] L={d} wp={d} K={d} {s} out: rel l2 / max vs float64: {s} {e:.3}/{e:.3}, {s} {e:.3}/{e:.3}; {d} of {d} words differ from the gather; no farther {}\n", .{ c.L, c.wp, c.K, dt, arms[0], e[oi][0].l2, e[oi][0].max, arms[ai], e[oi][ai].l2, e[oi][ai].max, try wordsDiffer(&sc, o[oi][ai], o[oi][0]), mlx.mlx_array_size(o[oi][0]), near });
                if (!near) ok = false;
            }
        }
        std.debug.print("[fa-gather] L={d} wp={d} K={d}: planted (window keys a row late) f32 {e:.3}/{e:.3}\n", .{ c.L, c.wp, c.K, ep.l2, ep.max });
        try testing.expect(!(ep.l2 <= e[0][0].l2 and ep.max <= e[0][0].max));
    }
    try testing.expect(ok);
}

/// Spans whose picks are not a multiple of 64: a short prompt's last span on a ratio-2 layer (its compressed rows all
/// picked, nc = K) at 300, 500, 700, 1,000 and 1,023 rows, and `span_cases`' part-filled block.
const odd_spans = [_]SpanCase{
    .{ .L = 44, .wp = 127, .nc = 150, .K = 150 },
    .{ .L = 244, .wp = 127, .nc = 250, .K = 250 },
    .{ .L = 188, .wp = 127, .nc = 350, .K = 350 },
    .{ .L = 232, .wp = 127, .nc = 500, .K = 500 },
    .{ .L = 255, .wp = 127, .nc = 511, .K = 511 },
    span_cases[3],
};

/// Spans a call outside a wide layer hands the flash kernel (`attnCore`): first turns of 18, 100 and 200 tokens on a
/// ratio-2 and a ratio-1 layer (no window rows before the call, so fewer keys than the window), rewinds of 7 and 12
/// rows, a 450-token prompt's CED replay, a 600-token rewind's replay, a 150-row turn on a 549-token cached prefix.
const served_spans = [_]SpanCase{
    .{ .L = 17, .wp = 0, .nc = 8, .K = 8 },
    .{ .L = 17, .wp = 0, .nc = 17, .K = 17 },
    .{ .L = 99, .wp = 0, .nc = 49, .K = 49 },
    .{ .L = 199, .wp = 0, .nc = 99, .K = 99 },
    .{ .L = 199, .wp = 0, .nc = 199, .K = 199 },
    .{ .L = 7, .wp = 0, .nc = 3, .K = 3 },
    .{ .L = 12, .wp = 0, .nc = 12, .K = 12 },
    .{ .L = 128, .wp = 0, .nc = 449, .K = 449 },
    .{ .L = 128, .wp = 0, .nc = 300, .K = 300 },
    .{ .L = 150, .wp = 127, .nc = 349, .K = 349 },
};

/// `attnGather`'s kernels as the load builds them: with `flash` the flash kernel under FLASH_ATTN and FLASH_TRIM (the
/// release's defaults), else the gather alone (FLASH_ATTN=0).
fn gatherAttn(flash: bool) !dsv41.GatherAttn {
    const saved = fa.ps().*;
    defer fa.ps().* = saved;
    fa.ps().* = .{ .mode = if (flash) .on else .off, .trim = flash };
    return dsv41.GatherAttn.init();
}

/// A model holding what the wide attention cores read: the release's window, groups and rope, `act` out.
fn attnModel(m: *dsv41.Dsv41Model, act: mlx.mlx_dtype) void {
    m.cfg = .{};
    m.cfg.sliding_window = WIN;
    m.cfg.dsv4_o_groups = OG;
    m.cfg.dsv4_rope_head_dim = 64;
    m.act = act;
    m.trace = null;
    m.attn_f16 = false;
}

/// `[L, H, D]` heads laid out as the grouped kernels store them (`spanErrTo` reads that layout).
fn groupedOf(sc: *dsv41.Scope, o: A, L: usize) !A {
    return sc.transpose(try sc.reshape(o, &.{ @intCast(L), @intCast(OG), @intCast(HEADS / OG * HD) }), &.{ 1, 0, 2 });
}

/// The span's grouped heads through attnGather on `g` in `ot`, or null where it declines (the caller's core serves).
fn spanGather(sc: *dsv41.Scope, m: *dsv41.Dsv41Model, g: *const dsv41.GatherAttn, s: *const Span, ot: mlx.mlx_dtype) !?A {
    m.attn_gather = g.*;
    m.act = ot;
    return dsv41.attnGather(sc, m, s.sink, s.q, none, s.keys_w, s.c.wp, if (s.c.K > 0) s.comp else none, if (s.c.K > 0) s.picks else none, Span.scale, .{ .periods = s.periods, .p0 = Span.p0 });
}

test "dsv41 wide: at the release's defaults on NAX a span whose picks are not a multiple of 64 takes the flash kernel (FLASH_TRIM's words), never the f16 core; under FLASH_ATTN=0 the gather declines it" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var on = try gatherAttn(true);
    defer on.deinit();
    var off = try gatherAttn(false);
    defer off.deinit();
    const kt = try flashTrimKernel(0);
    defer _ = mlx.mlx_fast_metal_kernel_free(kt);
    var m: dsv41.Dsv41Model = undefined;
    attnModel(&m, .bfloat16);
    for (odd_spans ++ [_]SpanCase{span_cases[0]}, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 60 + 10 * ci_);
        defer s.deinit();
        const got = (try spanGather(&sc, &m, &on, &s, .bfloat16)) orelse return error.FlashDeclined;
        const same = try sameWords(got, try s.flash(&sc, kt, false, .bfloat16));
        const off_serves = (try spanGather(&sc, &m, &off, &s, .bfloat16)) != null;
        std.debug.print("[fa-route] L={d} wp={d} K={d}: flash serves, FLASH_TRIM's words {}; under FLASH_ATTN=0 the gather serves {}\n", .{ c.L, c.wp, c.K, same, off_serves });
        try testing.expect(same);
        try testing.expectEqual(c.K % 64 == 0, off_serves);
    }
}

/// Each core's prompt-attention calls so far (`dsv41.core_calls`).
fn coreCounts() [4]usize {
    var o: [4]usize = undefined;
    for (&o, &dsv41.core_calls) |*v, *c| v.* = c.load(.monotonic);
    return o;
}
fn coresSince(from: [4]usize) [4]usize {
    var o = coreCounts();
    for (&o, from) |*v, f| v.* -= f;
    return o;
}
fn core(d: [4]usize, c: dsv41.Core) usize {
    return d[@intFromEnum(c)];
}

/// `n` prompt tokens of a request (`salt` keeps two requests' prefixes apart); the caller frees.
fn tokens(n: usize, salt: usize, vocab: usize) ![]u32 {
    const p = try testing.allocator.alloc(u32, n);
    for (p, 0..) |*t, i| t.* = @intCast((i * 7919 + salt * 104729 + 11) % vocab);
    return p;
}

/// The plugin's prompt calls on its live state as the engine makes them for a resident pack: `resumePrompt`, the rows
/// past what it restored but the last token as one call, then the last token alone.
const Served = struct {
    m: *dsv41.Dsv41Model,

    fn call(sv: Served, ids: []const u32) !void {
        const out = try dsv41.extendResumable(sv.m, testing.allocator, ids, null);
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
    }
    fn turn(sv: Served, prompt: []const u32) !usize {
        const n = try dsv41.resumePrompt(sv.m, prompt, null);
        if (prompt.len - 1 > n) try sv.call(prompt[n .. prompt.len - 1]);
        try sv.call(prompt[prompt.len - 1 ..]);
        return n;
    }
};

const Route = enum { first18, first100, first200, first450, follow700, rewind600, rewind12, chunks200, logprobs200, flash_named200 };

/// `r` on the served model: each core's prompt-attention calls it made.
fn servedRoute(sv: Served, r: Route, salt: usize) ![4]usize {
    const V = sv.m.vocab;
    const before = coreCounts();
    switch (r) {
        .first18, .first100, .first200, .first450, .flash_named200 => {
            const n: usize = switch (r) {
                .first18 => 18,
                .first100 => 100,
                .first450 => 450,
                else => 200,
            };
            const p = try tokens(n, salt, V);
            defer testing.allocator.free(p);
            const explicit = fa.ps().explicit;
            defer fa.ps().explicit = explicit;
            fa.ps().explicit = r == .flash_named200;
            try testing.expectEqual(@as(usize, 0), try sv.turn(p));
        },
        .follow700 => {
            const p = try tokens(700, salt, V);
            defer testing.allocator.free(p);
            try testing.expectEqual(@as(usize, 0), try sv.turn(p[0..550]));
            try testing.expectEqual(@as(usize, 549), try sv.turn(p));
        },
        .rewind600, .rewind12 => {
            const p = try tokens(if (r == .rewind600) 600 else 12, salt, V);
            defer testing.allocator.free(p);
            _ = try sv.turn(p);
            try dsv41.rewind(sv.m, testing.allocator, &sv.m.dec_state.?, p);
            try sv.call(p[0..1]);
        },
        .chunks200 => {
            const p = try tokens(601, salt, V);
            defer testing.allocator.free(p);
            _ = try dsv41.resumePrompt(sv.m, p, null);
            var at: usize = 0;
            while (at < 600) : (at += 200) try sv.call(p[at .. at + 200]);
            try sv.call(p[600..]);
        },
        .logprobs200 => {
            const p = try tokens(200, salt, V);
            defer testing.allocator.free(p);
            _ = try dsv41.resumePrompt(sv.m, p, null);
            const out = try dsv41.extendAllResumable(sv.m, testing.allocator, p);
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        },
    }
    return coresSince(before);
}

test "dsv41 wide: on NAX at the release's defaults every prompt-attention call of a served request takes a lossless core (flash, the gather or the f32 chain), never the f16 core: first turns of 18, 100, 200 and 450 tokens (the CED replay), a cached-prefix turn to 700, rewinds of 600 and 12 rows, 200-row chunks, prompt logprobs, FLASH_ATTN=1 named; the fall-through before attnCore (planted) takes the f16 core" {
    if (skip()) return error.SkipZigTest;
    const saved = fa.ps().*;
    defer fa.ps().* = saved;
    fa.ps().* = .{ .mode = .on, .trim = true };
    const fx = try fixture.Fixture.initFull(testing.allocator, 8, 1024, 1);
    defer fx.deinit();
    try testing.expect(fx.m.ced_split == 3 and !fx.m.attn_f16);
    const sv: Served = .{ .m = fx.m };
    for (std.enums.values(Route), 0..) |r, ri| {
        const d = try servedRoute(sv, r, ri);
        std.debug.print("[attn-core] {s}: flash {d}, gather {d}, f32 chain {d}, f16 core {d}\n", .{ @tagName(r), core(d, .flash), core(d, .gather), core(d, .f32), core(d, .f16) });
        try testing.expectEqual(@as(usize, 0), core(d, .f16));
        try testing.expect(core(d, .flash) > 0);
    }
    dsv41.core_plant = true;
    defer dsv41.core_plant = false;
    const d = try servedRoute(sv, .first200, 100);
    std.debug.print("[attn-core] planted first200: gather {d}, f16 core {d}\n", .{ core(d, .gather), core(d, .f16) });
    try testing.expect(core(d, .f16) > 0);
}

test "dsv41 wide: under FLASH_ATTN=0 a served 300-token prompt's spans whose picks are not a multiple of 64 take the f32 chain on NAX, never the f16 core; the fall-through before attnCore (planted) takes the f16 core" {
    if (skip()) return error.SkipZigTest;
    const saved = fa.ps().*;
    defer fa.ps().* = saved;
    fa.ps().* = .{};
    const fx = try fixture.Fixture.initFull(testing.allocator, 8, 1024, 2);
    defer fx.deinit();
    const sv: Served = .{ .m = fx.m };
    for ([_]bool{ false, true }) |plant| {
        dsv41.core_plant = plant;
        defer dsv41.core_plant = false;
        const p = try tokens(300, @intFromBool(plant), fx.m.vocab);
        defer testing.allocator.free(p);
        const before = coreCounts();
        _ = try sv.turn(p);
        const d = coresSince(before);
        std.debug.print("[attn-core] FLASH_ATTN=0 first300 (planted {}): gather {d}, f32 chain {d}, f16 core {d}\n", .{ plant, core(d, .gather), core(d, .f32), core(d, .f16) });
        try testing.expectEqual(@as(usize, 0), core(d, .flash));
        try testing.expect(core(d, .gather) > 0);
        if (plant) try testing.expect(core(d, .f16) > 0) else try testing.expect(core(d, .f16) == 0 and core(d, .f32) > 0);
    }
}

test "dsv41 wide: a tensor split's prompt-attention calls (32 heads a rank: no gather or flash shape) take the f32 chain on NAX, never the f16 core, one span and wide; the fall-through before attnCore (planted) takes the f16 core" {
    if (skip()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const saved = fa.ps().*;
    defer fa.ps().* = saved;
    fa.ps().* = .{ .mode = .on, .trim = true };
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 22, .experts = 8, .ctx = 0, .experts_fp4 = true, .tp = .{ .rank = 1, .size = 2 } });
    defer fx.deinit();
    // The other rank's halves are not summed in: the cores, not the values, are under test.
    const own = struct {
        fn sum(x: A) anyerror!A {
            var o = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&o, x);
            return o;
        }
    }.sum;
    dsv41.clusterHooks().tp_sum = &own;
    defer dsv41.clusterHooks().tp_sum = null;
    var p0: usize = 0;
    for ([_]bool{ false, true }) |plant| {
        dsv41.core_plant = plant;
        defer dsv41.core_plant = false;
        const before = coreCounts();
        for ([_]usize{ 100, 2 * dsv41.SPAN + 8 }) |rows| {
            const x = try fx.stream(rows);
            defer _ = mlx.mlx_array_free(x);
            const pre = try fx.cpre(rows);
            defer _ = mlx.mlx_array_free(pre);
            const out = try dsv41.workerStep(fx.m, gpa, x, pre, p0, false, false);
            defer for (out) |o| {
                _ = mlx.mlx_array_free(o);
            };
            try mlx.check(mlx.mlx_array_eval(out[0]));
            p0 += rows;
        }
        const d = coresSince(before);
        std.debug.print("[attn-core] tensor split (planted {}): flash {d}, gather {d}, f32 chain {d}, f16 core {d}\n", .{ plant, core(d, .flash), core(d, .gather), core(d, .f32), core(d, .f16) });
        try testing.expect(core(d, .flash) == 0 and core(d, .gather) == 0);
        if (plant) try testing.expect(core(d, .f16) > 0) else try testing.expect(core(d, .f16) == 0 and core(d, .f32) > 0);
    }
}

test "dsv41 wide tol: FLASH_ATTN's heads at a pick count not a multiple of 64 (a short prompt's last span, and the calls outside a wide layer it serves: no window rows before the call, 7 to 199 rows, picks 3 to 449) are no farther from float64 than the f32 chain (=0) in l2 at f32 and bf16 out and in max at f32 out, under M5's TF32 (TF32 off printed); window keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var on = try gatherAttn(true);
    defer on.deinit();
    const kp = try flashKernel(true, false, 1);
    defer _ = mlx.mlx_fast_metal_kernel_free(kp);
    var m: dsv41.Dsv41Model = undefined;
    attnModel(&m, .bfloat16);
    m.attn_softmax_kernel = try dsv41.buildKernel("dsv41_attn_softmax", &dsv41.ATTN_SOFTMAX_INS, &.{ "pw", "pc" }, dsv41.ATTN_SOFTMAX_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_softmax_kernel);
    // The f32 chain's GEMMs are TF32 on NAX unless MLX_ENABLE_TF32=0 (read at the process's first matmul): M5's =0.
    const tf32 = world.tf32();
    var ok = true;
    for (odd_spans ++ served_spans, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 80 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        const ref = try spanRefs(&s, &rows);
        defer gpa.free(ref);
        const comp = if (c.K > 0) s.comp else none;
        const topk = if (c.K > 0) s.picks else none;
        const band = try dsv41.windowBand(&sc, c.wp, c.L, WIN);
        // [out: bf16, f32][arm: the f32 chain, flash, the f16 core]
        var e: [2][3]FErr = undefined;
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |ot, oi| {
            e[oi][0] = try spanErrTo(&s, try groupedOf(&sc, try dsv41.attnWideF32(&sc, s.sink, s.q, s.keys_w, band, comp, topk, Span.scale, ot), c.L), &rows, ref);
            e[oi][1] = try spanErrTo(&s, (try spanGather(&sc, &m, &on, &s, ot)) orelse return error.FlashDeclined, &rows, ref);
            e[oi][2] = try spanErrTo(&s, try groupedOf(&sc, try dsv41.attnWide(&sc, &m, s.sink, s.q, s.keys_w, c.wp, comp, topk, Span.scale, ot), c.L), &rows, ref);
        }
        const ep = try spanErrTo(&s, try s.flash(&sc, kp, false, .bfloat16), &rows, ref);
        // Max is held at f32 out, where the kernel's own error shows: a bf16-out max is one element at the output
        // rounding both arms share, and which side of a bf16 boundary it lands on is a rounding flip (printed).
        const near = e[0][1].l2 <= e[0][0].l2 and e[1][1].l2 <= e[1][0].l2 and e[1][1].max <= e[1][0].max;
        std.debug.print("[fa-odd] L={d} wp={d} K={d} (f32 chain TF32 {}): rel l2 / max vs float64, bf16 out: =0 {e:.4}/{e:.4}, flash {e:.4}/{e:.4}, f16 core {e:.4}/{e:.4}; f32 out: =0 {e:.3}/{e:.3}, flash {e:.3}/{e:.3}, f16 core {e:.3}/{e:.3}; flash no farther {}; planted (keys a row late) {e:.3}/{e:.3}\n", .{ c.L, c.wp, c.K, tf32, e[0][0].l2, e[0][0].max, e[0][1].l2, e[0][1].max, e[0][2].l2, e[0][2].max, e[1][0].l2, e[1][0].max, e[1][1].l2, e[1][1].max, e[1][2].l2, e[1][2].max, near, ep.l2, ep.max });
        if (tf32 and !near) ok = false;
        try testing.expect(!(ep.l2 <= e[0][0].l2 and ep.max <= e[0][0].max));
    }
    try testing.expect(ok);
}

/// `got`, plain heads `[L, nh, HD]` (the f32 chain's), against the float64 heads `ref` (`spanRefs`) of its first `nh`
/// heads at `rows`, unroped dims.
fn plainErrTo(got: A, nh: usize, rows: []const usize, ref: []const f64) !FErr {
    const g = try dsv41.hostF32(got);
    defer testing.allocator.free(g);
    var dd: f64 = 0;
    var rr: f64 = 0;
    var dm: f64 = 0;
    var rm: f64 = 0;
    for (rows, 0..) |l, i| for (0..nh) |h| for (0..PLAIN) |d| {
        const w = ref[(i * HEADS + h) * PLAIN + d];
        const x: f64 = g[(l * nh + h) * HD + d];
        dd += (x - w) * (x - w);
        rr += w * w;
        dm = @max(dm, @abs(x - w));
        rm = @max(rm, @abs(w));
    };
    return .{ .l2 = @sqrt(dd / rr), .max = dm / rm };
}

test "dsv41 wide tol: the f32 chain where attnCore's lossless cores end on NAX (a tensor split's 32 heads, f16 and f32 activations) is as close to float64 as at the release's 64 bf16 heads (=0 of FLASH_ATTN and ATTN_F16) on the same heads, within 1.1x in l2 at f32 and bf16 out and in max at f32 out; keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const roll = struct {
        fn of(sc_: *dsv41.Scope, x: A) !A {
            const n: usize = @intCast(mlx.getShape(x)[0]);
            return sc_.concat(&.{ try sc_.slice(x, 0, 1, n), try sc_.slice(x, 0, 0, 1) }, 0);
        }
    }.of;
    const Arm = struct { name: []const u8, nh: usize, in: mlx.mlx_dtype };
    const arms = [_]Arm{ .{ .name = "tensor split, 32 heads", .nh = 32, .in = .bfloat16 }, .{ .name = "f16 activations", .nh = HEADS, .in = .float16 }, .{ .name = "f32 activations", .nh = HEADS, .in = .float32 } };
    const tf32 = world.tf32();
    var ok = true;
    for ([_]SpanCase{ span_cases[0], span_cases[1], odd_spans[0], odd_spans[4], served_spans[0], served_spans[7] }, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 140 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        const ref = try spanRefs(&s, &rows);
        defer gpa.free(ref);
        const band = try dsv41.windowBand(&sc, c.wp, c.L, WIN);
        const topk = if (c.K > 0) s.picks else none;
        for (arms) |arm| {
            const q = try sc.astype(try sc.slice(s.q, 1, 0, arm.nh), arm.in);
            const sink = try sc.slice(s.sink, 1, 0, arm.nh);
            const keys = try sc.astype(s.keys_w, arm.in);
            const comp = if (c.K > 0) try sc.astype(s.comp, arm.in) else none;
            // [out: bf16, f32][the release's chain, this arm]
            var e: [2][2]FErr = undefined;
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |ot, oi| {
                e[oi][0] = try plainErrTo(try sc.slice(try dsv41.attnWideF32(&sc, s.sink, s.q, s.keys_w, band, if (c.K > 0) s.comp else none, topk, Span.scale, ot), 1, 0, arm.nh), arm.nh, &rows, ref);
                e[oi][1] = try plainErrTo(try dsv41.attnWideF32(&sc, sink, q, keys, band, comp, topk, Span.scale, ot), arm.nh, &rows, ref);
            }
            const bad = try plainErrTo(try dsv41.attnWideF32(&sc, sink, q, try roll(&sc, keys), band, if (c.K > 0) try roll(&sc, comp) else none, topk, Span.scale, .float32), arm.nh, &rows, ref);
            const near = e[0][1].l2 <= 1.1 * e[0][0].l2 and e[1][1].l2 <= 1.1 * e[1][0].l2 and e[1][1].max <= 1.1 * e[1][0].max;
            std.debug.print("[f32-chain] L={d} wp={d} K={d} {s} (TF32 {}): rel l2 / max vs float64, bf16 out: 64 bf16 heads {e:.4}/{e:.4}, this {e:.4}/{e:.4}; f32 out: 64 bf16 heads {e:.3}/{e:.3}, this {e:.3}/{e:.3}; within 1.1x {}; planted (keys a row late) {e:.3}/{e:.3}\n", .{ c.L, c.wp, c.K, arm.name, tf32, e[0][0].l2, e[0][0].max, e[0][1].l2, e[0][1].max, e[1][0].l2, e[1][0].max, e[1][1].l2, e[1][1].max, near, bad.l2, bad.max });
            if (!near) ok = false;
            try testing.expect(!(bad.l2 <= 1.1 * e[1][0].l2 and bad.max <= 1.1 * e[1][0].max));
        }
    }
    try testing.expect(ok);
}

test "dsv41 wide meas: MLX_SERVE_DSV41_ATTN_F16's f16 core (LOSSY; it serves without NAX, on NAX only where ATTN_F16 is named on) against float64 beside the f32 chain (=0), printed, not gated, at the release's spans; keys a row late fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: dsv41.Dsv41Model = undefined;
    attnModel(&m, .bfloat16);
    m.attn_softmax_kernel = try dsv41.buildKernel("dsv41_attn_softmax", &dsv41.ATTN_SOFTMAX_INS, &.{ "pw", "pc" }, dsv41.ATTN_SOFTMAX_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_softmax_kernel);
    const tf32 = world.tf32();
    for (span_cases ++ odd_spans[0..5].*, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 40 + 10 * ci_);
        defer s.deinit();
        const rows = [_]usize{ 0, 1, 2, c.L / 3, c.L / 2, c.L - 2, c.L - 1 };
        const ref = try spanRefs(&s, &rows);
        defer gpa.free(ref);
        const comp = if (c.K > 0) s.comp else none;
        const topk = if (c.K > 0) s.picks else none;
        const band = try dsv41.windowBand(&sc, c.wp, c.L, WIN);
        var e: [2][2]FErr = undefined;
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }, 0..) |ot, oi| {
            e[oi][0] = try spanErrTo(&s, try groupedOf(&sc, try dsv41.attnWideF32(&sc, s.sink, s.q, s.keys_w, band, comp, topk, Span.scale, ot), c.L), &rows, ref);
            e[oi][1] = try spanErrTo(&s, try groupedOf(&sc, try dsv41.attnWide(&sc, &m, s.sink, s.q, s.keys_w, c.wp, comp, topk, Span.scale, ot), c.L), &rows, ref);
        }
        // Planted: the f16 core over the keys a row late (each window and compressed row the next one's).
        const roll = struct {
            fn of(sc_: *dsv41.Scope, x: A) !A {
                const n: usize = @intCast(mlx.getShape(x)[0]);
                return sc_.concat(&.{ try sc_.slice(x, 0, 1, n), try sc_.slice(x, 0, 0, 1) }, 0);
            }
        }.of;
        const bad = try dsv41.attnWide(&sc, &m, s.sink, s.q, try roll(&sc, s.keys_w), c.wp, if (c.K > 0) try roll(&sc, s.comp) else none, topk, Span.scale, .bfloat16);
        const ep = try spanErrTo(&s, try groupedOf(&sc, bad, c.L), &rows, ref);
        std.debug.print("[attn-f16] L={d} wp={d} K={d} (f32 chain TF32 {}): rel l2 / max vs float64, bf16 out: =0 {e:.4}/{e:.4}, ATTN_F16 {e:.4}/{e:.4}; f32 out: =0 {e:.3}/{e:.3}, ATTN_F16 {e:.3}/{e:.3}; planted (keys a row late) {e:.3}/{e:.3}\n", .{ c.L, c.wp, c.K, tf32, e[0][0].l2, e[0][0].max, e[0][1].l2, e[0][1].max, e[1][0].l2, e[1][0].max, e[1][1].l2, e[1][1].max, ep.l2, ep.max });
        try testing.expect(!(ep.l2 <= e[0][0].l2 and ep.max <= e[0][0].max));
    }
}

test "dsv41 wide ubench: FLASH_ATTN and FLASH_TRIM at the release's spans (DSV41_DEC_UBENCH)" {
    if (!ddec.ubenchOn() or skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("../../../sdk/util/io_util.zig");
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const Arm = struct { name: []const u8, k: mlx.mlx_fast_metal_kernel };
    var arms = [_]Arm{
        .{ .name = "FLASH_ATTN", .k = try flash1Kernel() },
        .{ .name = "FLASH_TRIM", .k = try flashTrimKernel(0) },
        .{ .name = "FLASH_TRIM + FLASH_PSUM", .k = try flashKernel(true, true, 0) },
    };
    defer for (arms) |a| {
        _ = mlx.mlx_fast_metal_kernel_free(a.k);
    };
    const n = 16;
    for ([_]SpanCase{ span_cases[0], span_cases[1] }, 0..) |c, ci_| {
        var s = try Span.init(&sc, c, 7 + ci_);
        defer s.deinit();
        const flops = 4.0 * @as(f64, @floatFromInt(c.L * HEADS * (WIN + c.K) * HD));
        // four rounds over the arms, each arm's median of 7 within a round
        for (0..4) |round| for (&arms) |a| {
            var times: [7]f64 = undefined;
            for (0..2 + times.len) |rep| {
                var inner = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
                defer inner.deinit();
                var outs: [n]A = undefined;
                const t0 = iou.nowUsMonotonic(io);
                for (&outs) |*o| o.* = try s.flash(&inner, a.k, true, .bfloat16);
                try dsv41.evalAll(&outs);
                const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / n;
                if (rep >= 2) times[rep - 2] = us;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            const us = times[times.len / 2];
            std.debug.print("[ftrim-ubench] round {d} L={d} keys={d} {s}: {d:.1} us (min {d:.1} max {d:.1}), {d:.1} TFLOPS useful, matrix floor {d:.1} us at 120\n", .{ round, c.L, WIN + c.K, a.name, us, times[0], times[times.len - 1], flops / us / 1e6, flops / 120e6 });
        };
    }
}

test "dsv41 wide tol: gate_gemm's bf16 MMA (PF_COMP_BF16, PF_ROUTER_BF16) is the f32 chain within f32 reordering at the compressor's and router's shapes, K 5,120, prompt rows 257 to 17,639; a weight row moved fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var ok = true;
    for ([_]usize{ 512, 384 }) |n| for ([_]usize{ 257, 4097, 17_639 }) |rows| {
        const x = try normal(&sc, &.{ @intCast(rows), @intCast(hidden) }, .bfloat16, 1, 0x51 + rows);
        const w = try normal(&sc, &.{ @intCast(n), @intCast(hidden) }, .bfloat16, 1.0 / @sqrt(@as(f32, hidden)), 0x52 + n);
        const want = try sc.matmul(try sc.astype(x, .float32), try sc.transpose(try sc.astype(w, .float32), &.{ 1, 0 }));
        const got = (try ggemm.bf16F32(sc.s, x, w, true, .single)) orelse return error.Declined;
        _ = try sc.keep(got);
        var lb: [96]u8 = undefined;
        if (!try close(try std.fmt.bufPrint(&lb, "bf16 MMA, f32 out: {d} rows x {d}", .{ rows, n }), want, got, null, 1e-5)) ok = false;
        // Planted: weight row 0 replaced by row 1.
        const wp = try sc.concat(&.{ try sc.slice(w, 0, 1, 2), try sc.slice(w, 0, 1, n) }, 0);
        const bad = (try ggemm.bf16F32(sc.s, x, wp, true, .single)) orelse return error.Declined;
        _ = try sc.keep(bad);
        try testing.expect(!try close(try std.fmt.bufPrint(&lb, "planted: row 0 moved, {d} rows x {d}", .{ rows, n }), want, bad, null, 1e-5));
    };
    // Declines: another type, no NAX.
    const xf = try normal(&sc, &.{ 300, @intCast(hidden) }, .float32, 1, 9);
    const wb = try normal(&sc, &.{ 384, @intCast(hidden) }, .bfloat16, 1, 10);
    try testing.expect((try ggemm.bf16F32(sc.s, xf, wb, true, .single)) == null);
    try testing.expect((try ggemm.bf16F32(sc.s, try sc.astype(xf, .bfloat16), wb, false, .single)) == null);
    try testing.expect(ok);
}

test "dsv41 wide tol: MLX_SERVE_DSV41_PF_COMP_BF16 and MLX_SERVE_DSV41_PF_ROUTER_BF16, each and both, keep a pair worker's prefill within 5% of the stock path's, MXFP4 and NVFP4 experts with per-expert scales, a ratio-2 and a ratio-1 pair; the compressor's kv and gate swapped and the logits negated fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    const sw = .{ .MLX_SERVE_DSV41_PF_COMP_BF16, .MLX_SERVE_DSV41_PF_ROUTER_BF16 };
    inline for (sw) |n| env.set(n, "0");
    defer {
        inline for (sw) |n| env.set(n, "0");
    }
    const Case = struct { lo: usize, rows: usize, nv: bool };
    var ok = true;
    for ([_]Case{ .{ .lo = 8, .rows = 5000, .nv = false }, .{ .lo = 20, .rows = 4608, .nv = true } }) |c| {
        var lb: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&lb, "{d}-{d} {s} at {d} rows", .{ c.lo, c.lo + 1, if (c.nv) "NVFP4" else "MXFP4", c.rows });
        const fx = try fixture.Fixture.init(gpa, .{ .lo = c.lo, .hi = c.lo + 2, .ctx = 0, .experts_fp4 = !c.nv, .experts_nvfp4 = c.nv, .expert_scales = if (c.nv) .all else .none });
        defer fx.deinit();
        const x = try fx.stream(c.rows);
        defer _ = mlx.mlx_array_free(x);
        const pre = try fx.cpre(c.rows);
        defer _ = mlx.mlx_array_free(pre);
        var want_tap: dsv41.RouteTap = .{ .gpa = gpa };
        defer freeTap(&want_tap);
        const c0 = dsv41.comp_bf16_count.load(.monotonic);
        const r0 = dsv41.router_bf16_count.load(.monotonic);
        const want = try prefillTapped(fx, x, pre, &want_tap);
        defer for (want) |o| {
            _ = mlx.mlx_array_free(o);
        };
        try testing.expectEqual(c0, dsv41.comp_bf16_count.load(.monotonic));
        try testing.expectEqual(r0, dsv41.router_bf16_count.load(.monotonic));
        const has_comp = fx.m.layers[c.lo].ratio > 1 or fx.m.layers[c.lo + 1].ratio > 1;
        // Each alone, engaged: the compressor on a ratio-2 KV source's layers, the router on both layers.
        env.set(.MLX_SERVE_DSV41_PF_COMP_BF16, "1");
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "PF_COMP_BF16")) ok = false;
        std.debug.print("wide layer {s}: PF_COMP_BF16 served {d} layers\n", .{ label, dsv41.comp_bf16_count.load(.monotonic) - c0 });
        if (has_comp) try testing.expect(dsv41.comp_bf16_count.load(.monotonic) > c0);
        env.set(.MLX_SERVE_DSV41_PF_COMP_BF16, "0");
        env.set(.MLX_SERVE_DSV41_PF_ROUTER_BF16, "1");
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "PF_ROUTER_BF16")) ok = false;
        std.debug.print("wide layer {s}: PF_ROUTER_BF16 served {d} calls\n", .{ label, dsv41.router_bf16_count.load(.monotonic) - r0 });
        try testing.expect(dsv41.router_bf16_count.load(.monotonic) > r0);
        env.set(.MLX_SERVE_DSV41_PF_COMP_BF16, "1");
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "both")) ok = false;
        // Planted: the logits negated; the compressor's kv and gate swapped (where it serves).
        {
            dsv41.router_bf16_plant = true;
            defer dsv41.router_bf16_plant = false;
            try testing.expect(!try closeAs(fx, x, pre, want, &want_tap, label, "planted: the logits negated"));
        }
        if (has_comp) {
            dsv41.comp_bf16_plant = true;
            defer dsv41.comp_bf16_plant = false;
            try testing.expect(!try closeAs(fx, x, pre, want, &want_tap, label, "planted: the compressor's kv and gate swapped"));
        }
        inline for (sw) |n| env.set(n, "0");
    }
    try testing.expect(ok);
}

/// `got` against float64 `ref` (both `n`): the error's norm over `ref`'s, its largest magnitude over `ref`'s largest.
const F64Err = struct { l2: f64, max: f64 };
fn f64Err(got: []const f32, ref: []const f64) F64Err {
    var dd: f64 = 0;
    var rr: f64 = 0;
    var dm: f64 = 0;
    var rm: f64 = 0;
    for (got, ref) |g, r| {
        const d = @as(f64, g) - r;
        dd += d * d;
        rr += r * r;
        dm = @max(dm, @abs(d));
        rm = @max(rm, @abs(r));
    }
    return .{ .l2 = @sqrt(dd / rr), .max = dm / rm };
}

test "dsv41 wide tol: PF_COMP_BF16's compressor projection over a 4,096-row layer is no farther from float64 than today's span-by-span TF32 chain in l2 and max (one 256-row span, the release's fused kv and gate rows); the single running sum (planted) is" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 8, .hi = 10, .ctx = 0, .experts_fp4 = true });
    defer fx.deinit();
    // The pair's KV source above ratio 1.
    const li: usize = if (fx.layer(8).comp != null and fx.layer(8).ratio > 1) 8 else 9;
    const cw = &fx.layer(li).comp.?;
    try testing.expect(fx.layer(li).ratio > 1 and cw.wgate == null);
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const rows = 4096;
    const span = 256;
    const k0 = 5 * span;
    const x = try normal(&sc, &.{ rows, @intCast(hidden) }, .bfloat16, 1, 0x6b);
    try dsv41.evalAll(&.{x});
    // Today's: the span's own call (`compProj`, MLX's f32 matmul at M = 256: its split K). The switch's: one launch over
    // the layer's rows, the span's rows of it.
    const old = try dsv41.hostF32(try dsv41.compProj(&sc, fx.m, li, cw, try sc.slice(x, 0, k0, k0 + span)));
    defer gpa.free(old);
    const all = (try ggemm.bf16F32(sc.s, x, cw.wkv.w, true, .blocked)) orelse return error.Declined;
    _ = try sc.keep(all);
    const new = try dsv41.hostF32(try sc.slice(all, 0, k0, k0 + span));
    defer gpa.free(new);
    // Planted: the blocked sum's control, one running sum over K.
    ggemm.blocked_plant = true;
    const single = (try ggemm.bf16F32(sc.s, x, cw.wkv.w, true, .blocked)) orelse return error.Declined;
    ggemm.blocked_plant = false;
    _ = try sc.keep(single);
    const one = try dsv41.hostF32(try sc.slice(single, 0, k0, k0 + span));
    defer gpa.free(one);
    // float64: the bf16 operands widened exactly, each output summed in float64.
    const xs = try dsv41.hostF32(try sc.slice(x, 0, k0, k0 + span));
    defer gpa.free(xs);
    const ws = try dsv41.hostF32(cw.wkv.w);
    defer gpa.free(ws);
    const N = ws.len / hidden;
    const ref = try gpa.alloc(f64, span * N);
    defer gpa.free(ref);
    for (0..span) |r| for (0..N) |c| {
        var acc: f64 = 0;
        const xr = xs[r * hidden ..][0..hidden];
        const wr = ws[c * hidden ..][0..hidden];
        for (xr, wr) |a, b| acc += @as(f64, a) * @as(f64, b);
        ref[r * N + c] = acc;
    };
    var differ: usize = 0;
    for (old, new) |a, b| differ += @intFromBool(a != b);
    const eo = f64Err(old, ref);
    const en = f64Err(new, ref);
    const ep = f64Err(one, ref);
    std.debug.print("compressor projection, {d} rows x {d}: {d} of {d} words differ between today's and PF_COMP_BF16; from float64: today l2 {e:.3} max {e:.3}, PF_COMP_BF16 (blocked) l2 {e:.3} max {e:.3}, planted single sum l2 {e:.3} max {e:.3}\n", .{ span, N, differ, old.len, eo.l2, eo.max, en.l2, en.max, ep.l2, ep.max });
    try testing.expect(en.l2 <= eo.l2 and en.max <= eo.max);
    try testing.expect(!(ep.l2 <= eo.l2 and ep.max <= eo.max));
}

test "dsv41 wide tol: PF_ROUTER_BF16 serves only where today runs gate_gemm's single sum (1,707 rows and up at 384 experts, K 5,120), today's words there; below it today's split-K path, which the single sum (planted) is farther from float64 than" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const E = 384;
    const w = try normal(&sc, &.{ E, @intCast(hidden) }, .bfloat16, 1.0 / @sqrt(@as(f32, hidden)), 0x72);
    try dsv41.evalAll(&.{w});
    const ws = try dsv41.hostF32(w);
    defer gpa.free(ws);
    var planted_farther = false;
    // The CED replay's 128, a verify block, a short prompt, the split-K edge each side, a chunk and the one-pass width.
    for ([_]usize{ 33, 128, 1024, 1706, 1707, 4097, 17_638 }) |rows| {
        const x = try normal(&sc, &.{ @intCast(rows), @intCast(hidden) }, .bfloat16, 1, 0x71 + rows);
        try dsv41.evalAll(&.{x});
        // Today's served call: gate_gemm's launch where MLX runs its regular kernel, else MLX's f32 matmul (split K).
        const today = if (try ggemm.router(sc.s, x, w, true)) |y| try sc.keep(y) else try sc.matmul(try sc.astype(x, .float32), try sc.transpose(try sc.astype(w, .float32), &.{ 1, 0 }));
        const sum = dsv41.routerSum(rows, E, hidden);
        const got = if (sum) |sm| try sc.keep((try ggemm.bf16F32(sc.s, x, w, true, sm)) orelse return error.Declined) else today;
        dsv41.router_sum_plant = true;
        const planted = try sc.keep((try ggemm.bf16F32(sc.s, x, w, true, dsv41.routerSum(rows, E, hidden).?)) orelse return error.Declined);
        dsv41.router_sum_plant = false;
        // float64 over (at most) 256 rows from the middle.
        const n: usize = @min(rows, 256);
        const r0 = (rows - n) / 2;
        const a = try dsv41.hostF32(try sc.slice(today, 0, r0, r0 + n));
        defer gpa.free(a);
        const b = try dsv41.hostF32(try sc.slice(got, 0, r0, r0 + n));
        defer gpa.free(b);
        const c = try dsv41.hostF32(try sc.slice(planted, 0, r0, r0 + n));
        defer gpa.free(c);
        const xs = try dsv41.hostF32(try sc.slice(x, 0, r0, r0 + n));
        defer gpa.free(xs);
        const ref = try gpa.alloc(f64, n * E);
        defer gpa.free(ref);
        for (0..n) |r| for (0..E) |col| {
            var acc: f64 = 0;
            for (xs[r * hidden ..][0..hidden], ws[col * hidden ..][0..hidden]) |p, q| acc += @as(f64, p) * @as(f64, q);
            ref[r * E + col] = acc;
        };
        var differ: usize = 0;
        for (a, b) |p, q| differ += @intFromBool(p != q);
        const et = f64Err(a, ref);
        const eg = f64Err(b, ref);
        const ep = f64Err(c, ref);
        const regular = ggemm.mlxRegular(@intCast(rows), E, hidden);
        std.debug.print("router logits, {d} rows (today {s}): PF_ROUTER_BF16 {s}, {d} of {d} words differ from today's; from float64: today l2 {e:.3} max {e:.3}, served l2 {e:.3} max {e:.3}, the single sum here l2 {e:.3} max {e:.3}\n", .{ rows, if (regular) "gate_gemm's single sum" else "MLX's split K", if (sum != null) "serves" else "declines", differ, a.len, et.l2, et.max, eg.l2, eg.max, ep.l2, ep.max });
        try testing.expectEqual(regular, sum != null);
        try testing.expectEqual(@as(usize, 0), differ);
        if (!(ep.l2 <= et.l2 and ep.max <= et.max)) planted_farther = true;
    }
    try testing.expect(planted_farther);
}

fn freePair(xs: [2]A) void {
    for (xs) |o| _ = mlx.mlx_array_free(o);
}

/// Whether the pair worker's prefill over `x` / `pre` is `want` word for word, from an empty MLX cache (as the engine
/// leaves it at each request's end).
fn sameAs(fx: *fixture.Fixture, x: A, pre: A, want: [2]A) !bool {
    _ = mlx.mlx_synchronize(mlx.gpuStream());
    try mlx.check(mlx.mlx_clear_cache());
    const got = try prefill(fx, x, pre);
    defer freePair(got);
    return try sameWords(want[0], got[0]) and try sameWords(want[1], got[1]);
}

test "dsv41 wide: under MLX_SERVE_DSV41_PF_KEEP_WIDE's reserve a release-shape pair worker's prefill is the same words at the pinned width and narrower, its buffers holding the last pass's words, MXFP4 and NVFP4 experts; a planted aliasing reserve fails" {
    if (skip()) return error.SkipZigTest;
    if (!kw.available()) {
        std.debug.print("keep wide: the loaded libmlx has no mlx_metal_wide_reserve: skipped\n", .{});
        return error.SkipZigTest;
    }
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    kw.disarm();
    defer kw.disarm();
    // The fixture's widest prompt pass is the pinned width; a narrower one takes the same buffers.
    const pinned_rows: usize = 8192;
    const narrow_rows: usize = 5000;
    inline for (.{ false, true }) |nv| {
        const fx = try fixture.Fixture.init(gpa, .{ .lo = 8, .hi = 10, .ctx = 0, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none });
        defer fx.deinit();
        const xw = try fx.stream(pinned_rows);
        defer _ = mlx.mlx_array_free(xw);
        const pw = try fx.cpre(pinned_rows);
        defer _ = mlx.mlx_array_free(pw);
        const xn = try fx.stream(narrow_rows);
        defer _ = mlx.mlx_array_free(xn);
        const pn = try fx.cpre(narrow_rows);
        defer _ = mlx.mlx_array_free(pn);
        const want_w = try prefill(fx, xw, pw);
        defer freePair(want_w);
        const want_n = try prefill(fx, xn, pn);
        defer freePair(want_n);
        var bytes: [kw.max_buffers]usize = undefined;
        const n = kw.sizes(&fx.m.cfg, pinned_rows, &bytes);
        try testing.expect(n > 0);
        try testing.expectEqual(@as(c_int, 0), kw.arm(bytes[0..n]));
        defer kw.disarm();
        const label = if (nv) "NVFP4" else "MXFP4";
        // The pinned width, a narrower pass, then the pinned width over buffers that hold the last passes' words.
        const h0 = kw.stats().hits;
        try testing.expect(try sameAs(fx, xw, pw, want_w));
        const h1 = kw.stats().hits;
        try testing.expect(try sameAs(fx, xn, pn, want_n));
        const h2 = kw.stats().hits;
        try testing.expect(try sameAs(fx, xw, pw, want_w));
        const st = kw.stats();
        std.debug.print("keep wide {s}: {d} buffers {d} MB; hits {d} at {d} rows, {d} at {d} rows; misses {d} ({d} MB)\n", .{ label, st.buffers, st.bytes >> 20, h1 - h0, pinned_rows, h2 - h1, narrow_rows, st.misses, st.missed_bytes >> 20 });
        // Every reserve buffer serves the pinned width's pass.
        try testing.expect(h1 - h0 >= n and h2 > h1);
        // Planted: the reserve hands out buffers it keeps idle, so live arrays share them.
        kw.plant(true);
        const bad = sameAs(fx, xw, pw, want_w);
        kw.plant(false);
        _ = mlx.mlx_synchronize(mlx.gpuStream());
        try testing.expect(!(try bad));
    }
}

// ── MLX_SERVE_DSV41_PF_COMP_PASS and MLX_SERVE_DSV41_PF_IDX_PASS: the compressor and the indexer once over the pass ──

test "dsv41 wide: MLX_SERVE_DSV41_PF_COMP_PASS keeps a release-shape KV source and index layer's prefill streams and mixes word for word (layers 8-9, an odd row count's pending row), NVFP4 experts with per-expert scales; the groups' rope a row late fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    const keep = env.override(.MLX_SERVE_DSV41_PF_COMP_PASS, "0");
    defer keep.restore();
    const rows = 4999;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 8, .hi = 10, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all });
    defer fx.deinit();
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(pre);
    const c0 = dsv41.comp_pass_count.load(.monotonic);
    const want = try prefill(fx, x, pre);
    defer for (want) |o| {
        _ = mlx.mlx_array_free(o);
    };
    try testing.expectEqual(c0, dsv41.comp_pass_count.load(.monotonic));
    env.set(.MLX_SERVE_DSV41_PF_COMP_PASS, "1");
    const got = try prefill(fx, x, pre);
    defer for (got) |o| {
        _ = mlx.mlx_array_free(o);
    };
    const n = dsv41.comp_pass_count.load(.monotonic) - c0;
    dsv41.comp_pass_plant = true;
    const bad = try prefill(fx, x, pre);
    dsv41.comp_pass_plant = false;
    defer for (bad) |o| {
        _ = mlx.mlx_array_free(o);
    };
    env.set(.MLX_SERVE_DSV41_PF_COMP_PASS, "0");
    const same = [_]bool{ try sameWords(want[0], got[0]), try sameWords(want[1], got[1]) };
    const planted = try sameWords(want[0], bad[0]);
    std.debug.print("wide layers 8-9 at {d} rows, PF_COMP_PASS: served {d} layers, streams {} mixes {}; planted streams the same {}\n", .{ rows, n, same[0], same[1], planted });
    try testing.expect(n >= 1 and same[0] and same[1] and !planted);
}

test "dsv41 wide tol: MLX_SERVE_DSV41_PF_IDX_PASS keeps a release-shape index layer's prefill streams and mixes within the pair worker's tolerance (layers 8-9), and its queries' word-for-word half (the pass's wq_b, rope and fp4 round trip, the head weights per span) gives the same words; the queries' rope a row late fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    const keep = env.override(.MLX_SERVE_DSV41_PF_IDX_PASS, "0");
    defer keep.restore();
    const rows = 4999;
    const label = "8-9 NVFP4 at 4999 rows";
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 8, .hi = 10, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all });
    defer fx.deinit();
    const x = try fx.stream(rows);
    defer _ = mlx.mlx_array_free(x);
    const pre = try fx.cpre(rows);
    defer _ = mlx.mlx_array_free(pre);
    var want_tap: dsv41.RouteTap = .{ .gpa = gpa };
    defer freeTap(&want_tap);
    const c0 = dsv41.idx_pass_count.load(.monotonic);
    const want = try prefillTapped(fx, x, pre, &want_tap);
    defer for (want) |o| {
        _ = mlx.mlx_array_free(o);
    };
    try testing.expectEqual(c0, dsv41.idx_pass_count.load(.monotonic));
    env.set(.MLX_SERVE_DSV41_PF_IDX_PASS, "1");
    const ok = try closeAs(fx, x, pre, want, &want_tap, label, "PF_IDX_PASS");
    const n = dsv41.idx_pass_count.load(.monotonic) - c0;
    // the head weights from MLX's split GEMM per span, the rest over the pass: the same words as span by span
    dsv41.idx_pass_mlx_w = true;
    const exact = try prefill(fx, x, pre);
    defer for (exact) |o| {
        _ = mlx.mlx_array_free(o);
    };
    dsv41.idx_pass_mlx_w = false;
    dsv41.idx_pass_plant = true;
    const planted = try closeAs(fx, x, pre, want, &want_tap, label, "PF_IDX_PASS planted");
    dsv41.idx_pass_plant = false;
    env.set(.MLX_SERVE_DSV41_PF_IDX_PASS, "0");
    const same = [_]bool{ try sameWords(want[0], exact[0]), try sameWords(want[1], exact[1]) };
    std.debug.print("wide layers {s}, PF_IDX_PASS: served {d} layers, within tolerance {}; queries over the pass with per-span head weights: streams {} mixes {}; planted within tolerance {}\n", .{ label, n, ok, same[0], same[1], planted });
    try testing.expect(n >= 1 and ok and same[0] and same[1] and !planted);
}

test "dsv41 wide tol: PF_IDX_PASS's head weights (kernels/idx_wproj.metal) are no farther from float64 than MLX's GEMM per 256-row span (the served call) in l2 and max over every output, 32 heads x 5,120, bf16 out, 257 to 17,639 rows; a weight row moved fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var ok = true;
    for ([_]usize{ 257, 4097, 17_639 }, 0..) |rows, ci_| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const x = try normal(&sc, &.{ @intCast(rows), 5120 }, .bfloat16, 1, 0x91 + ci_);
        const w = try normal(&sc, &.{ 32, 5120 }, .bfloat16, 1.0 / @sqrt(5120.0), 0x92 + ci_);
        const lin: dsv41.Lin = .{ .w = w };
        // served: MLX's matmul on each 256-row span, as `rowLinear` runs the span's rows
        var parts: std.ArrayList(A) = .empty;
        defer parts.deinit(testing.allocator);
        var lo: usize = 0;
        while (lo < rows) : (lo += 256) try parts.append(testing.allocator, try lin.apply(&sc, try sc.slice(x, 0, lo, @min(rows, lo + 256))));
        const served = try sc.concat(parts.items, 0);
        const got = (try dsv41.idxHeadWeights(&sc, &lin, x)) orelse return error.Declined;
        const wp = try sc.concat(&.{ try sc.slice(w, 0, 1, 2), try sc.slice(w, 0, 1, 32) }, 0);
        const bad = (try dsv41.idxHeadWeights(&sc, &.{ .w = wp }, x)) orelse return error.Declined;
        try dsv41.evalAll(&.{ served, got, bad });
        const hx = try dsv41.hostF32(x);
        defer testing.allocator.free(hx);
        const hw = try dsv41.hostF32(w);
        defer testing.allocator.free(hw);
        // every 7th row and the last, all 32 heads, in float64
        var idx: std.ArrayList(usize) = .empty;
        defer idx.deinit(testing.allocator);
        var r: usize = 0;
        while (r < rows) : (r += 7) try idx.append(testing.allocator, r);
        try idx.append(testing.allocator, rows - 1);
        const ref = try testing.allocator.alloc(f64, idx.items.len * 32);
        defer testing.allocator.free(ref);
        for (idx.items, 0..) |row, i| for (0..32) |h| {
            var d: f64 = 0;
            for (hx[row * 5120 ..][0..5120], hw[h * 5120 ..][0..5120]) |a, b| d += @as(f64, a) * b;
            ref[i * 32 + h] = d;
        };
        var e: [3]F64Err = undefined;
        var differ: usize = 0;
        // Where the two bf16 words are equal their errors are equal, so l2 and max over every output compare on the
        // words that differ: each one's float64 value, both errors.
        var sq = [2]f64{ 0, 0 };
        var mx = [2]f64{ 0, 0 };
        var nearer: usize = 0;
        for ([_]A{ served, got, bad }, 0..) |o, k| {
            const ho = try dsv41.hostF32(o);
            defer testing.allocator.free(ho);
            const sel = try testing.allocator.alloc(f32, idx.items.len * 32);
            defer testing.allocator.free(sel);
            for (idx.items, 0..) |row, i| @memcpy(sel[i * 32 ..][0..32], ho[row * 32 ..][0..32]);
            e[k] = f64Err(sel, ref);
        }
        {
            const hs = try dsv41.hostF32(served);
            defer testing.allocator.free(hs);
            const hg = try dsv41.hostF32(got);
            defer testing.allocator.free(hg);
            for (hs, hg, 0..) |a, b, i| if (a != b) {
                differ += 1;
                const row = i / 32;
                const h = i % 32;
                var d: f64 = 0;
                for (hx[row * 5120 ..][0..5120], hw[h * 5120 ..][0..5120]) |u, v| d += @as(f64, u) * v;
                const es = @abs(@as(f64, a) - d);
                const em = @abs(@as(f64, b) - d);
                sq[0] += es * es;
                sq[1] += em * em;
                mx[0] = @max(mx[0], es);
                mx[1] = @max(mx[1], em);
                nearer += @intFromBool(em < es);
            };
        }
        std.debug.print("[idx-wproj] {d} rows: rel l2 / max vs float64 over sampled rows (bf16 out): served per-span GEMM {e:.4}/{e:.4}, idx_wproj {e:.4}/{e:.4}; {d} of {d} words differ, {d} of them nearer float64 in idx_wproj; over them squared error served {e:.6} idx_wproj {e:.6}, worst served {e:.6} idx_wproj {e:.6}; planted {e:.3}/{e:.3}\n", .{ rows, e[0].l2, e[0].max, e[1].l2, e[1].max, differ, rows * 32, nearer, sq[0], sq[1], mx[0], mx[1], e[2].l2, e[2].max });
        // the worst element over the same words for both: the sampled rows (f64Err's max is over their largest
        // reference) and every word that differs
        var rm: f64 = 0;
        for (ref) |v| rm = @max(rm, @abs(v));
        const worst = [2]f64{ @max(e[0].max * rm, mx[0]), @max(e[1].max * rm, mx[1]) };
        std.debug.print("[idx-wproj] {d} rows: worst element over the sampled rows and the differing words: served {e:.6} idx_wproj {e:.6}\n", .{ rows, worst[0], worst[1] });
        ok = ok and sq[1] <= sq[0] and worst[1] <= worst[0] and e[2].l2 > 10 * e[0].l2;
    }
    try testing.expect(ok);
}
