//! The wide prompt layer's default-off rewrites behind their switches against the stock (unfused) path at the
//! release's widths and prompt row counts: each kernel's own, then each switch alone and all of them through a
//! release-shape pair worker's prefill (`decode/fixture.zig`). A rewrite may round in another order, so each output is
//! held to a tolerance (`Err`), not to the stock path's bytes; a planted semantic bug (a weight on the wrong expert,
//! swapped per-expert scales, a dropped norm weight or mix scale, the routed term at half its scale) must fail it.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const dsv41 = @import("../deepseek_v41.zig");
const mpair = @import("moe_pair.zig");
const pfroute = @import("pf_route.zig");
const fp4gemm = @import("fp4_gemm.zig");
const mtail = @import("moe_tail.zig");
const hv2 = @import("hc_v2.zig");
const fixture = @import("../decode/fixture.zig");

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

/// Each row's routing weights rotated by one (a weight on the wrong expert): the planted MoE bug.
fn rolled(sc: *dsv41.Scope, wts: A) !A {
    return sc.concat(&.{ try sc.slice(wts, 1, 1, top_k), try sc.slice(wts, 1, 0, 1) }, 1);
}

test "dsv41 wide tol: the paired fp4 gate+up with moe_act in its epilogue is the split path's MoE within 1% at the release widths, MXFP4 and NVFP4 with per-expert scales, prompt row counts to 8192; a weight on the wrong expert or swapped scales fail" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    // The release's 384 experts under `-Dslow-tests` (7 GB of banks); else 48, whose rows an expert reach every plan.
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    const unset = env.override(.MLX_SERVE_DSV41_MOE_PAIR, null);
    defer unset.restore();
    const unset_scaled = env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "0");
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
            // No shared expert, so the routed sum is all the output holds.
            const sh = try csc.zeros(&.{ @intCast(L), @intCast(hidden) }, .bfloat16);
            const want = try dsv41.moePrefill(&csc, &m, &ly, c.x, c.idx, c.wts, sh);
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
    m.moe_wt_kernel = null;
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
        const off_row: dsv41.MoeParts = .{ .dn = p.dn, .inv = try csc.concat(&.{ try csc.slice(p.inv, 0, top_k, n), try csc.slice(p.inv, 0, 0, top_k) }, 0), .sh = p.sh };
        try testing.expect(!try close("  planted: the unsort off by a row", want, try mtail.launch(&csc, off_row, x1, &sp), x1, tol_kernel));
        const rolled_post: dsv41.Split = .{ .pre = none, .post = try csc.concat(&.{ try csc.slice(sp.post, 1, 1, 4), try csc.slice(sp.post, 1, 0, 1) }, 1), .comb = sp.comb };
        try testing.expect(!try close("  planted: post on the wrong copies", want, try mtail.launch(&csc, p, x1, &rolled_post), x1, tol_kernel));
    }
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

test "dsv41 wide: moePrefill under MLX_SERVE_DSV41_PF_MOE_EPI is the argsort path's output word for word, MXFP4 and NVFP4 with per-expert scales, paired and split gate/up, prompt rows to 8192" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    // On by default: the argsort arm needs the explicit 0.
    const unset = env.override(.MLX_SERVE_DSV41_PF_MOE_EPI, "0");
    defer unset.restore();
    inline for (.{ false, true }) |nv| inline for (.{ false, true }) |paired| {
        const pk = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, if (paired) "1" else "0") else env.override(.MLX_SERVE_DSV41_MOE_PAIR, if (paired) "1" else null);
        defer pk.restore();
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

test "dsv41 wide: moePrefill under MLX_SERVE_DSV41_PF_FP4_GEMM is today's output word for word, NVFP4 with per-expert scales and MXFP4, paired gate/up, odd prompt rows to 8192" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const experts: c_int = if (@import("build_options").slow_tests) 384 else 48;
    // Off as "0", not unset: a run with the switch exported would unsetenv before the first stderr print, which
    // std's lazy environ scan then reads past.
    const unset = env.override(.MLX_SERVE_DSV41_PF_FP4_GEMM, "0");
    defer unset.restore();
    inline for (.{ true, false }) |nv| {
        const pk = if (nv) env.override(.MLX_SERVE_DSV41_MOE_PAIR_SCALED, "1") else env.override(.MLX_SERVE_DSV41_MOE_PAIR, "1");
        defer pk.restore();
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
        ly.gate_w = try sc.zeros(&.{ experts, 8 }, .bfloat16);
        ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .gu = try fp4Bank(&sc, 0x6A7E, experts, 2 * inter, hidden, nv), .down = try fp4Bank(&sc, 0xD0E1, experts, hidden, inter, nv) };
        if (nv) for (&ly.experts.scale, 0..) |*v, i| {
            v.* = try uniform(&sc, &.{experts}, .bfloat16, 0.25, 4.0, 0x5CA1E + i);
        };
        for ([_]usize{ 40, 777, 1254, 4097, 8192 }, 0..) |L, ci| {
            var csc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
            defer csc.deinit();
            const c = try call(&csc, L, @intCast(experts), 0x2100 + ci, false);
            const want = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const on = env.override(.MLX_SERVE_DSV41_PF_FP4_GEMM, "1");
            defer on.restore();
            const before = fp4gemm.launches.load(.monotonic);
            const got = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, c.wts, c.sh);
            const served = fp4gemm.launches.load(.monotonic) - before;
            const eq = try sameWords(want.dn, got.dn) and try sameWords(want.inv, got.inv);
            std.debug.print("pf fp4 gemm moePrefill {s} rows {d}: staged launches {d}, same words {}\n", .{ if (nv) "NVFP4" else "MXFP4", L, served, eq });
            try testing.expectEqual(@as(u32, 2), served);
            try testing.expect(eq);
            // Planted: each row's weights on the wrong experts.
            const bad = try dsv41.moePrefillParts(&csc, &m, &ly, c.x, c.idx, try rolled(&csc, c.wts), c.sh);
            try testing.expect(!try sameWords(want.dn, bad.dn));
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

test "dsv41 wide tol: the hc mixes at 8 rows a threadgroup are the 4-row launches' mixes, split and norms within 1% at prompt widths; a dropped mix scale fails" {
    if (skip()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: dsv41.Dsv41Model = undefined;
    try hcModel(&m);
    defer hcModelFree(&m);
    const unset = env.override(.MLX_SERVE_DSV41_HC_MIX8, null);
    defer unset.restore();
    var ok = true;
    for ([_]c_int{ 17, 1027, 5001, 8192 }, 0..) |L, i| {
        var sc = dsv41.Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        const r = try hcRows(&sc, L, 0x8800 + 16 * @as(u64, i));
        const eps = try dsv41.hcEps(&sc, &m);
        const z4 = try dsv41.hcMixWide(&sc, &m, r.x, &r.hc, eps);
        const f4 = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, r.w, eps);
        const on = env.override(.MLX_SERVE_DSV41_HC_MIX8, "1");
        defer on.restore();
        const z8 = try dsv41.hcMixWide(&sc, &m, r.x, &r.hc, eps);
        const f8 = try dsv41.hcMixNormFused(&sc, &m, r.x, r.cpre, &r.hc, r.w, r.w, eps);
        const names = [_][]const u8{ "z", "fused z", "h", "h2", "comb" };
        for ([_][2]A{ .{ z4, z8 }, .{ f4.z, f8.z }, .{ f4.n.h, f8.n.h }, .{ f4.h2, f8.h2 }, .{ f4.n.sp.comb, f8.n.sp.comb } }, names) |pr, nm| {
            var lb: [64]u8 = undefined;
            if (!try close(try std.fmt.bufPrint(&lb, "hc mix8 rows {d} {s}", .{ L, nm }), pr[0], pr[1], null, tol_kernel)) ok = false;
        }
        // Planted: 8 rows a threadgroup with the mixes' scale dropped (ones in its place).
        var hc_bad = r.hc;
        hc_bad.scale = try sc.ones(mlx.getShape(r.hc.scale), .float32);
        const bad = try dsv41.hcMixWide(&sc, &m, r.x, &hc_bad, eps);
        try testing.expect(!try close("  planted: mix scale dropped", z4, bad, null, tol_kernel));
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

// ── a release-shape pair worker's prefill ──

const switches = .{ .MLX_SERVE_DSV41_MOE_PAIR, .MLX_SERVE_DSV41_MOE_PAIR_SCALED, .MLX_SERVE_DSV41_HC_EXPAND_NORM, .MLX_SERVE_DSV41_HC_MIX8, .MLX_SERVE_DSV41_WOA_JOIN, .MLX_SERVE_DSV41_MOE_TAIL, .MLX_SERVE_DSV41_PF_HC_V2, .MLX_SERVE_DSV41_PF_MOE_EPI };

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
        // The layer's loaded experts take the paired kernel under their kind's switch.
        {
            var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            const ly = fx.layer(c.lo);
            const h = try normal(&sc, &.{ @intCast(c.rows), @intCast(hidden) }, .bfloat16, 1, 0x77);
            const r = dsv41.route(&sc, fx.m, ly, h, none) catch |e| {
                std.debug.print("wide layer {s}: the route failed: {s}\n", .{ label, @errorName(e) });
                return e;
            };
            env.set(if (nv) .MLX_SERVE_DSV41_MOE_PAIR_SCALED else .MLX_SERVE_DSV41_MOE_PAIR, "1");
            defer env.set(if (nv) .MLX_SERVE_DSV41_MOE_PAIR_SCALED else .MLX_SERVE_DSV41_MOE_PAIR, "0");
            const served = (actOf(&sc, fx.m, &ly.experts, h, r.idx, r.wts) catch |e| {
                std.debug.print("wide layer {s}: the paired kernel failed: {s}\n", .{ label, @errorName(e) });
                return e;
            }) != null;
            if (!served) std.debug.print("wide layer {s}: the paired kernel declines the loaded experts (gu {}, mode {s})\n", .{ label, ly.experts.gu != null, if (ly.experts.gu) |g| @tagName(g.qp.mode) else "-" });
            try testing.expect(served);
        }
        if (!try closeAs(fx, x, pre, want, &want_tap, label, "the stock path again")) ok = false;
        inline for (switches) |n| {
            env.set(n, "1");
            defer env.set(n, "0");
            if (!try closeAs(fx, x, pre, want, &want_tap, label, @tagName(n))) ok = false;
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
