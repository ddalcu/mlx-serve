//! A batched twin of MLX's M=1 gemv for a plain (unquantized) half-type linear's decode rows: y [L, N] = x [L, K] w
//! [N, K]^T (w row-major), each row's words the words MLX's matmul gives that row alone, the weight read once a launch.
//!
//! MLX (the kit's lib/mlx-src, 64ea011) runs x [1, K] w^T as `gemv_<T>_bm*_bn*_sm*_sn*_tm*_tn*_nc0_axpby0`
//! (mlx/backend/metal/kernels/gemv.h `GEMVKernel::run`, its parameters `gemv_axbpy`'s in matmul.cpp), but 2..15 rows as
//! gemv_wide (Matmul::eval_gpu folds a broadcast weight's batch into M; M == 1 is the one shape that stays with gemv) and
//! 16 or more as its GEMM: a verify's rows round apart from the same rows decoded one at a time. kernels/gemv_twin.metal
//! is GEMVKernel::run's body over up to `chunk_rows` rows a threadgroup: MLX's threadgroup (32, bn, bm) over MLX's
//! output block, each thread's matrix elements loaded once a step and multiplied into every row's own accumulators, so a
//! row's loads, products, sums, shuffles and roundings are MLX's one-row call's on that row, in its order, and the matrix
//! is read once a launch as the one-row call reads it once (grid z the chunks of rows). `plan` is gemv_axbpy's rule for a
//! row-major weight (MLX-free).
//!
//! Shape-generic: any N and K (the kernel reads them from its arrays' shapes, as MLX's from its arguments), bf16 or f16
//! (x and w one dtype): the product of two 8- or 11-bit significands is exact in f32, so a fused multiply-add and a
//! rounded multiply then add give the same sum, and no contraction choice of either compiler can move a bit (f32 rows
//! keep MLX's matmul). A launch's free axes (`Variant`: the rows a threadgroup, a thread's output rows, the threadgroup's
//! simdgroup rows, the chunks' place in the grid) move no word: a thread's K elements and the order its partials meet are
//! fixed by MLX's sn, tn and bn alone. A launch shape (the dtype, MLX's plan with the variant's tiling, the rows a
//! threadgroup) is one kernel with its constants in its header, made at its first call and compiled at its first
//! evaluation (`warm`: a weight's at load), its launch configs kept by row count, so a call builds no template argument or
//! config. `linear` serves `served`'s variant for the call's shape and rows; `linearWith` takes any. One row is declined: MLX's matmul of one
//! row is that row's gemv (the same words, MLX's own launch).
//! MLX only (no import of the model), so its MLX-free tests run in `zig build test`.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

const A = mlx.mlx_array;

/// One launch's MLX parameters: threadgroup rows and columns in simdgroups (bm, bn), a simdgroup's rows and columns in
/// threads (sm, sn), a thread's rows and columns in elements (tm, tn), and the threadgroups a row (MLX's n_tgp).
pub const Plan = struct { bm: u32, bn: u32, sm: u32, sn: u32, tm: u32, tn: u32, tgs: u32 };

/// gemv_axbpy's rule for out_vector_len `n`, in_vector_len `k` and a matrix that is not transposed (a row-major [n, k]).
pub fn plan(n: usize, k: usize) Plan {
    var bm: u32 = if (n >= 4096) 8 else 4;
    var bn: u32 = 1;
    var sm: u32 = 1;
    var sn: u32 = 32;
    if (k <= 64) {
        bm = 1;
        sm = 8;
        sn = 4;
    } else if (k >= 16 * n) {
        bm = 1;
        bn = 8;
    }
    const tm: u32 = if (n < 4) 1 else 4;
    const per: usize = bm * sm * tm;
    return .{ .bm = bm, .bn = bn, .sm = sm, .sn = sn, .tm = tm, .tn = 4, .tgs = @intCast((n + per - 1) / per) };
}

/// The MLX kernel name `plan(n, k)` selects for dtype name `t` (as matmul.cpp spells it), for log lines and tests.
pub fn mlxName(buf: []u8, t: []const u8, p: Plan) []const u8 {
    return std.fmt.bufPrint(buf, "gemv_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_nc0_axpby0", .{ t, p.bm, p.bn, p.sm, p.sn, p.tm, p.tn }) catch buf[0..0];
}

/// The most rows a call takes: MLX's gemv_wide widths (2..15) and its GEMM's first.
pub const max_rows: usize = 16;
/// The most rows a threadgroup takes (a launch's grid the chunks of them).
pub const chunk_rows: usize = 8;

/// A launch's free axes, each row's words MLX's M=1 words at every one of them: lane l of simdgroup column g reads the K
/// offsets (g sn + l) tn + i bn sn tn and MLX's simdgroup shuffles and (bn > 1) threadgroup partials meet in its order, all
/// fixed by MLX's sn, tn and bn, so the rows a threadgroup takes (`nr`, 1..`chunk_rows`), a thread's output rows (`tm`,
/// MLX's or fewer), the threadgroup's simdgroup rows (`bm`, 32 bn bm threads at most 1024) and the chunks' place in the
/// grid (`ilv`: along x beside their output tile, so a tile's chunks run together and share its weight reads; else along
/// z) choose only the launch's shape.
pub const Variant = struct { nr: u32, tm: u32, bm: u32, ilv: bool = false };

/// `p` (MLX's plan for an [n, k] weight) with `v`'s thread and threadgroup rows, its threadgroups an output row recounted;
/// null where `v` is outside the free axes.
pub fn tiled(p: Plan, n: usize, v: Variant) ?Plan {
    if (v.nr < 1 or v.nr > chunk_rows) return null;
    if (v.tm < 1 or v.tm > p.tm or (v.tm != 1 and v.tm != 2 and v.tm != 4)) return null;
    if (v.bm < 1 or 32 * p.bn * v.bm > 1024) return null;
    var q = p;
    q.tm = v.tm;
    q.bm = v.bm;
    const per: usize = q.bm * q.sm * q.tm;
    q.tgs = @intCast((n + per - 1) / per);
    return q;
}

/// A weight shape's served launches at 2..6 rows (7..16 take the 6-row one).
pub const Choice = struct { n: usize, k: usize, v: [5]Variant };

/// The served launches, read from this machine's timings (dvi4u, w-dvi4u-20261102-0415: a dependent chain of calls alone
/// and beside an independent GEMM, the median of 3 rotated rounds each; arms MLX's own matmul, dvi2's batched launch,
/// dvi1's per-row geometry and dvi3u's leaders). DeepSeek-V4.1-Flash's indexer weights_proj (32 x 5120) and index key wk
/// (128 x 512): the fastest beside the GEMM among the launches at or below MLX's matmul in both modes, one row a
/// threadgroup at MLX's tiling (dvi1's geometry), at 6 rows one row a thread. Engram's wkv (25600 x 6144), above MLX's
/// matmul at every launch: the nearest to it, every row in one threadgroup one row a thread at 2 to 4 rows (3 rows at
/// half the simdgroup rows), chunks of 3 rows two rows a thread beside their tile at 5 and 6.
pub const choices = [_]Choice{
    .{ .n = 32, .k = 5120, .v = .{
        .{ .nr = 1, .tm = 4, .bm = 1 },
        .{ .nr = 1, .tm = 4, .bm = 1 },
        .{ .nr = 1, .tm = 4, .bm = 1 },
        .{ .nr = 1, .tm = 4, .bm = 1 },
        .{ .nr = 1, .tm = 1, .bm = 1 },
    } },
    .{ .n = 128, .k = 512, .v = .{
        .{ .nr = 1, .tm = 4, .bm = 4 },
        .{ .nr = 1, .tm = 4, .bm = 4 },
        .{ .nr = 1, .tm = 4, .bm = 4 },
        .{ .nr = 1, .tm = 4, .bm = 4 },
        .{ .nr = 1, .tm = 1, .bm = 4 },
    } },
    .{ .n = 25600, .k = 6144, .v = .{
        .{ .nr = 2, .tm = 1, .bm = 8 },
        .{ .nr = 3, .tm = 1, .bm = 4 },
        .{ .nr = 4, .tm = 1, .bm = 8 },
        .{ .nr = 3, .tm = 2, .bm = 8, .ilv = true },
        .{ .nr = 3, .tm = 2, .bm = 8, .ilv = true },
    } },
};

/// The variant `linear` serves for an [n, k] weight at `rows` rows: a shape `choices` names takes its entry (rows past 6
/// the 6-row one); another shape every row of a call of up to `chunk_rows` rows in one threadgroup, MLX's own tiling, the
/// chunks along z (dvi2's launch).
pub fn served(p: Plan, n: usize, k: usize, rows: usize) Variant {
    for (choices) |c| {
        if (c.n == n and c.k == k) return c.v[@min(@max(rows, 2), 6) - 2];
    }
    return .{ .nr = @intCast(@min(rows, chunk_rows)), .tm = p.tm, .bm = p.bm };
}

pub const source = kernelSource("plugins/deepseek_v41/kernels/gemv_twin.metal");
const ins: []const [*:0]const u8 = &.{ "x", "w" };
const outs: []const [*:0]const u8 = &.{"y"};

/// The Metal type a dtype's rows take, as MLX's kernels spell it; null outside bf16 and f16.
pub fn typeName(dt: mlx.mlx_dtype) ?[]const u8 {
    return switch (dt) {
        .bfloat16 => "bfloat16_t",
        .float16 => "half",
        else => null,
    };
}

/// A launch shape's kernel header (its template arguments as constants, MLX's MLX_MTL_CONST spelling), 0-terminated in
/// `buf`: the element type `t`, the (tiled) plan, `nr` rows a threadgroup and the chunks' place (`ilv`).
pub fn header(buf: []u8, t: []const u8, p: Plan, nr: usize, ilv: bool) ![:0]const u8 {
    const h = try std.fmt.bufPrint(buf[0 .. buf.len - 1], "typedef {s} T;\n" ++
        "static constant constexpr const int BM = {d};\nstatic constant constexpr const int BN = {d};\n" ++
        "static constant constexpr const int SM = {d};\nstatic constant constexpr const int SN = {d};\n" ++
        "static constant constexpr const int TM = {d};\nstatic constant constexpr const int TN = {d};\n" ++
        "static constant constexpr const int NR = {d};\nstatic constant constexpr const int ILV = {d};\n", .{ t, p.bm, p.bn, p.sm, p.sn, p.tm, p.tn, nr, @intFromBool(ilv) });
    buf[h.len] = 0;
    return buf[0..h.len :0];
}

/// A launch shape: x and w's dtype, its (n, k), the variant and MLX's plan tiled by it; its kernel, and its launch configs
/// by row count (each made at its first call).
const Shape = struct {
    n: usize,
    k: usize,
    dt: mlx.mlx_dtype,
    v: Variant,
    p: Plan,
    kern: mlx.mlx_fast_metal_kernel,
    cfgs: [max_rows + 1]mlx.mlx_fast_metal_kernel_config,
    warm: bool,
};
var shapes: [48]Shape = undefined;
var n_shapes: usize = 0;

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The launch shape for (n, k, dt, v), made at its first call; null (declined) when the table is full, the dtype is
/// neither bf16 nor f16 or `v` is outside the free axes. The kernel's name is its dtype, tiled plan, rows and chunk place:
/// two weight shapes on one of them share one compiled kernel (MLX keys its libraries by name and source).
fn shapeFor(n: usize, k: usize, dt: mlx.mlx_dtype, v: Variant) !?*Shape {
    for (shapes[0..n_shapes]) |*sh| {
        if (sh.n == n and sh.k == k and sh.dt == dt and sh.v.nr == v.nr and sh.v.tm == v.tm and sh.v.bm == v.bm and sh.v.ilv == v.ilv) return sh;
    }
    if (n_shapes == shapes.len) return null;
    const t = typeName(dt) orelse return null;
    const p = tiled(plan(n, k), n, v) orelse return null;
    var hb: [512]u8 = undefined;
    const h = try header(&hb, t, p, v.nr, v.ilv);
    var nb: [128]u8 = undefined;
    const nm = try std.fmt.bufPrint(nb[0 .. nb.len - 1], "dsv41_gemv_twin_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_r{d}{s}", .{ if (dt == .bfloat16) "bf16" else "f16", p.bm, p.bn, p.sm, p.sn, p.tm, p.tn, v.nr, if (v.ilv) "_i" else "" });
    nb[nm.len] = 0;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kern = mlx.mlx_fast_metal_kernel_new(nb[0..nm.len :0].ptr, iv, ov, source, h.ptr, true, false);
    if (kern.ctx == null) return error.MetalKernelCompileFailed;
    const sh = &shapes[n_shapes];
    sh.* = .{ .n = n, .k = k, .dt = dt, .v = v, .p = p, .kern = kern, .cfgs = undefined, .warm = false };
    for (&sh.cfgs) |*c| c.* = .{ .ctx = null };
    n_shapes += 1;
    return sh;
}

/// `sh`'s launch config at `rows` rows (the output [rows, n], MLX's grid over the chunks of `sh.v.nr` rows: along z, or
/// with `ilv` along x, each tile's chunks adjacent), made at its first call and kept.
fn configFor(sh: *Shape, rows: usize) !mlx.mlx_fast_metal_kernel_config {
    if (sh.cfgs[rows].ctx != null) return sh.cfgs[rows];
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const chunks: usize = (rows + sh.v.nr - 1) / sh.v.nr;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), ci(sh.n) }, 2, sh.dt));
    if (sh.v.ilv) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(sh.p.tgs * chunks * 32), ci(sh.p.bn), ci(sh.p.bm)));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(sh.p.tgs * 32), ci(sh.p.bn), ci(chunks * sh.p.bm)));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, ci(sh.p.bn), ci(sh.p.bm)));
    sh.cfgs[rows] = cfg;
    return cfg;
}

/// Every launch shape's kernel and configs freed (the model's deinit; a second call finds none).
pub fn deinit() void {
    for (shapes[0..n_shapes]) |*sh| {
        for (&sh.cfgs) |*c| {
            if (c.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(c.*);
            c.* = .{ .ctx = null };
        }
        _ = mlx.mlx_fast_metal_kernel_free(sh.kern);
    }
    n_shapes = 0;
}

/// x [L, K] times w [N, K]^T (w an evaluated row-major array, the dtype of x, bf16 or f16), L of 2..`max_rows`: y [L, N]
/// in that dtype, each row MLX's M=1 words, a new array the caller owns. Null (nothing launched) where it declines: not
/// the GPU, one row (MLX's own gemv), a rank, shape, layout or dtype outside it, a full shape table.
pub fn linear(s: mlx.mlx_stream, x: A, w: A) !?A {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[0] < 2 or ws[0] < 1 or ws[1] < 1) return null;
    return linearWith(s, x, w, served(plan(@intCast(ws[0]), @intCast(ws[1])), @intCast(ws[0]), @intCast(ws[1]), @intCast(xs[0])));
}

/// `linear` on variant `v` (any of the free axes: the same words), for the microbench and the served choice's table.
pub fn linearWith(s: mlx.mlx_stream, x: A, w: A, v: Variant) !?A {
    if (!mlx.streamIsGpu(s)) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[1] != ws[1] or xs[1] < 1 or ws[0] < 1 or xs[0] < 2) return null;
    const rows: usize = @intCast(xs[0]);
    if (rows > max_rows) return null;
    const dt = mlx.mlx_array_dtype(x);
    if ((dt != .bfloat16 and dt != .float16) or mlx.mlx_array_dtype(w) != dt) return null;
    // a loaded (evaluated) row-major weight, its strides (K, 1): a view would be copied whole every call (the kernel takes
    // its arrays row contiguous), and an unevaluated array's strides are not its layout yet
    var w_ok = false;
    if (mlx._mlx_array_is_available(&w_ok, w) != 0 or !w_ok) return null;
    const wst = mlx.mlx_array_strides(w);
    if (wst[1] != 1 or (wst[0] != @as(usize, @intCast(ws[1])) and ws[0] != 1)) return null;
    const sh = (try shapeFor(@intCast(ws[0]), @intCast(xs[1]), dt, v)) orelse return null;
    const cfg = try configFor(sh, rows);
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, w }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, sh.kern, iv, cfg, s));
    var y = mlx.mlx_array_new();
    mlx.check(mlx.mlx_vector_array_get(&y, ov, 0)) catch |e| {
        _ = mlx.mlx_array_free(y);
        return e;
    };
    return y;
}

/// The launch shapes `w`'s calls take (`served`'s at 2..`max_rows` rows) made and compiled now, one evaluated call of
/// zeros each, so a request's first decode steps pay no JIT; a shape already compiled (another row count's, another
/// layer's weight of the same shape) is skipped.
pub fn warm(s: mlx.mlx_stream, w: A) !void {
    const ws = mlx.getShape(w);
    if (ws.len != 2 or !mlx.streamIsGpu(s)) return;
    const dt = mlx.mlx_array_dtype(w);
    const p = plan(@intCast(ws[0]), @intCast(ws[1]));
    for (2..max_rows + 1) |rows| {
        const v = served(p, @intCast(ws[0]), @intCast(ws[1]), rows);
        const sh = (try shapeFor(@intCast(ws[0]), @intCast(ws[1]), dt, v)) orelse return;
        if (sh.warm) continue;
        var z = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z);
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(rows), ws[1] }, 2, dt, s));
        const y = (try linearWith(s, z, w, v)) orelse return;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        sh.warm = true;
    }
}

test "gemv_twin plan: matmul.cpp's gemv_axbpy rule (not transposed) on the served shapes and its edges" {
    const t = std.testing;
    // indexer weights_proj (32 heads x 5120): K >= 16 N, so one simdgroup row and eight K slices, the tgp reduction.
    try t.expectEqual(Plan{ .bm = 1, .bn = 8, .sm = 1, .sn = 32, .tm = 4, .tn = 4, .tgs = 8 }, plan(32, 5120));
    // index key wk (128 x 512): K < 16 N.
    try t.expectEqual(Plan{ .bm = 4, .bn = 1, .sm = 1, .sn = 32, .tm = 4, .tn = 4, .tgs = 8 }, plan(128, 512));
    // Engram wkv (25600 x 6144): N >= 4096.
    try t.expectEqual(Plan{ .bm = 8, .bn = 1, .sm = 1, .sn = 32, .tm = 4, .tn = 4, .tgs = 800 }, plan(25600, 6144));
    // K <= 64 takes the 8 x 4 simdgroup shape whatever N is; N < 4 one row a thread.
    try t.expectEqual(Plan{ .bm = 1, .bn = 1, .sm = 8, .sn = 4, .tm = 4, .tn = 4, .tgs = 2 }, plan(64, 64));
    try t.expectEqual(Plan{ .bm = 1, .bn = 8, .sm = 1, .sn = 32, .tm = 1, .tn = 4, .tgs = 3 }, plan(3, 4096));
    // The K >= 16 N boundary and the N >= 4096 one.
    try t.expectEqual(@as(u32, 8), plan(32, 512).bn);
    try t.expectEqual(@as(u32, 1), plan(32, 511).bn);
    try t.expectEqual(@as(u32, 4), plan(4095, 8192).bm);
    try t.expectEqual(@as(u32, 8), plan(4096, 8192).bm);
    // Every plan is one MLX instantiates (gemv.metal's instantiate_gemv_blocks).
    const shipped = [_][6]u32{ .{ 1, 8, 1, 32, 4, 4 }, .{ 1, 8, 1, 32, 1, 4 }, .{ 1, 1, 8, 4, 4, 4 }, .{ 1, 1, 8, 4, 1, 4 }, .{ 4, 1, 1, 32, 1, 4 }, .{ 4, 1, 1, 32, 4, 4 }, .{ 8, 1, 1, 32, 4, 4 } };
    for ([_]usize{ 1, 3, 4, 32, 128, 511, 4096, 25600 }) |n| for ([_]usize{ 16, 64, 65, 512, 5120, 6144, 8192 }) |k| {
        const p = plan(n, k);
        var hit = false;
        for (shipped) |c| hit = hit or std.mem.eql(u32, &c, &.{ p.bm, p.bn, p.sm, p.sn, p.tm, p.tn });
        try t.expect(hit);
    };
    var buf: [96]u8 = undefined;
    try t.expectEqualStrings("gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0", mlxName(&buf, "bfloat16", plan(32, 5120)));
}

test "gemv_twin header: the plan, the rows a threadgroup and the element type as the kernel's constants" {
    const t = std.testing;
    var buf: [512]u8 = undefined;
    const h = try header(&buf, typeName(.bfloat16).?, plan(32, 5120), 6, false);
    try t.expectEqualStrings("typedef bfloat16_t T;\n" ++
        "static constant constexpr const int BM = 1;\nstatic constant constexpr const int BN = 8;\n" ++
        "static constant constexpr const int SM = 1;\nstatic constant constexpr const int SN = 32;\n" ++
        "static constant constexpr const int TM = 4;\nstatic constant constexpr const int TN = 4;\n" ++
        "static constant constexpr const int NR = 6;\nstatic constant constexpr const int ILV = 0;\n", h);
    try t.expectEqual(@as(u8, 0), buf[h.len]);
    try t.expectEqualStrings("half", typeName(.float16).?);
    try t.expect(typeName(.float32) == null);
    // the kernel reads every constant the header sets, and its sizes from its arrays' shapes, as MLX's from its arguments
    for ([_][]const u8{ "BM", "BN", "SM", "SN", "TM", "TN", "NR", "ILV", "x_shape[1]", "w_shape[0]", "x_shape[0]" }) |name| try t.expect(std.mem.indexOf(u8, source, name) != null);
}

test "gemv_twin variants: the free axes tile MLX's plan and move only its row tiling" {
    const t = std.testing;
    // Engram's wkv at a thread's one row: four times the threadgroups, the K split (sn, tn, bn) MLX's
    const e = plan(25600, 6144);
    try t.expectEqual(Plan{ .bm = 8, .bn = 1, .sm = 1, .sn = 32, .tm = 1, .tn = 4, .tgs = 3200 }, tiled(e, 25600, .{ .nr = 6, .tm = 1, .bm = 8 }).?);
    try t.expectEqual(Plan{ .bm = 2, .bn = 1, .sm = 1, .sn = 32, .tm = 2, .tn = 4, .tgs = 6400 }, tiled(e, 25600, .{ .nr = 2, .tm = 2, .bm = 2 }).?);
    // weights_proj (bn 8): a thread's one row, its partials still the eight simdgroups' in order; 32 x 8 x 4 = 1024 threads
    const wp = plan(32, 5120);
    try t.expectEqual(Plan{ .bm = 1, .bn = 8, .sm = 1, .sn = 32, .tm = 1, .tn = 4, .tgs = 32 }, tiled(wp, 32, .{ .nr = 1, .tm = 1, .bm = 1 }).?);
    try t.expect(tiled(wp, 32, .{ .nr = 2, .tm = 4, .bm = 4 }) != null);
    try t.expect(tiled(wp, 32, .{ .nr = 2, .tm = 4, .bm = 5 }) == null);
    // outside the axes: more rows a thread than MLX's, three, no rows, more than a chunk
    try t.expect(tiled(e, 25600, .{ .nr = 2, .tm = 8, .bm = 8 }) == null);
    try t.expect(tiled(e, 25600, .{ .nr = 2, .tm = 3, .bm = 8 }) == null);
    try t.expect(tiled(plan(3, 4096), 3, .{ .nr = 2, .tm = 2, .bm = 1 }) == null);
    try t.expect(tiled(e, 25600, .{ .nr = 0, .tm = 4, .bm = 8 }) == null);
    try t.expect(tiled(e, 25600, .{ .nr = 9, .tm = 4, .bm = 8 }) == null);
    // the served launches: a named shape's entries (rows past 6 the 6-row one), another shape the batched launch
    try t.expectEqual(Variant{ .nr = 3, .tm = 2, .bm = 8, .ilv = true }, served(e, 25600, 6144, 6));
    try t.expectEqual(Variant{ .nr = 3, .tm = 2, .bm = 8, .ilv = true }, served(e, 25600, 6144, 13));
    try t.expectEqual(Variant{ .nr = 2, .tm = 1, .bm = 8 }, served(e, 25600, 6144, 2));
    try t.expectEqual(Variant{ .nr = 3, .tm = 1, .bm = 4 }, served(e, 25600, 6144, 3));
    try t.expectEqual(Variant{ .nr = 1, .tm = 4, .bm = 1 }, served(wp, 32, 5120, 2));
    try t.expectEqual(Variant{ .nr = 1, .tm = 1, .bm = 1 }, served(wp, 32, 5120, 6));
    try t.expectEqual(Variant{ .nr = 1, .tm = 4, .bm = 4 }, served(plan(128, 512), 128, 512, 4));
    try t.expectEqual(Variant{ .nr = 1, .tm = 1, .bm = 4 }, served(plan(128, 512), 128, 512, 6));
    try t.expectEqual(Variant{ .nr = 8, .tm = 4, .bm = 1 }, served(plan(32, 4096), 32, 4096, 13));
    // one row a threadgroup at MLX's tiling is MLX's own plan (dvi1's geometry), the chunks the rows
    try t.expectEqual(wp, tiled(wp, 32, served(wp, 32, 5120, 3)).?);
    try t.expectEqual(plan(128, 512), tiled(plan(128, 512), 128, served(plan(128, 512), 128, 512, 2)).?);
    // every choice is a launch the free axes take
    for (choices) |c| for (c.v, 2..) |v, rows| {
        try t.expect(tiled(plan(c.n, c.k), c.n, v) != null);
        try t.expect(v.nr <= rows);
    };
    var buf: [512]u8 = undefined;
    try t.expect(std.mem.endsWith(u8, try header(&buf, "half", tiled(e, 25600, .{ .nr = 3, .tm = 1, .bm = 8, .ilv = true }).?, 3, true), "TM = 1;\nstatic constant constexpr const int TN = 4;\nstatic constant constexpr const int NR = 3;\nstatic constant constexpr const int ILV = 1;\n"));
}
