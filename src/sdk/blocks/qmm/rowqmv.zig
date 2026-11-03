//! Row-exact 4/6/8-bit affine matvecs, ported from TensorFold's `rows.py` (MIT, see NOTICE).
//! Every input row is its own simdgroup running one row's loop, so a row's bits never
//! depend on how many rows share the call: serial decode and a verify window agree.
const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");

pub const MAX_ROWS = 16;
/// Output rows a simdgroup.
const RPS = 4;

/// MLX's `qmv_fast` loop (load_vector + qdot) at BITS 4, 6 or 8. A lane
/// loads VPL inputs a block (16 at 4-bit, else 8), pre-divided so masked
/// fields need no shift; the 4/6-bit sum adds each run of 4 in bf16 first, as
/// MLX does on bfloat16_t, the 8-bit one adds each input in float. The tail of
/// a K that is not a multiple of the 32 * VPL block runs one more chunk on the
/// lanes below (K % block) / VPL. Reads nothing but its own row of x.
pub const HEADER = kernelSource("kernels/qmm/rowqmv_header.metal");

/// One simdgroup per input row, each over BLK blocks of RPS output rows; the
/// row count is a launch dimension only.
pub const QMV_SOURCE = kernelSource("kernels/qmm/rowqmv_qmv.metal");

var qmv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { rows: c_int, n: c_int, k: c_int, bits: u32, gs: u32, relu2: bool };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};

pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn qmvConfig(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    if (cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ key.rows, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .bfloat16));
    const blocks: c_int = if (key.rows <= 8) 2 else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * key.rows, @divExact(key.n, RPS), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * key.rows, blocks, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RPS", RPS));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(key.bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "RELU2", key.relu2));
    try cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// A 4, 6 or 8-bit affine bf16 matrix [..., N, K * bits / 32] the kernels
/// read: groups of 32, 64 or 128, K a multiple of 64, N a multiple of 8.
pub fn fits(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) bool {
    if ((bits != 4 and bits != 6 and bits != 8) or bi.ctx == null) return false;
    if (group_size != 32 and group_size != 64 and group_size != 128) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    if (ws.len < 2) return false;
    const packed_bits = ws[ws.len - 1] * 32;
    const b: c_int = @intCast(bits);
    return @rem(packed_bits, b) == 0 and @rem(@divTrunc(packed_bits, b), 64) == 0 and @rem(ws[ws.len - 2], 8) == 0;
}

/// Input width K of a matrix `fits` admits.
fn inWidth(w: mlx.mlx_array, bits: u32) c_int {
    const ws = mlx.getShape(w);
    return @divTrunc(ws[ws.len - 1] * 32, @as(c_int, @intCast(bits)));
}

/// `x [..., K] @ w.T` for 1..MAX_ROWS rows, or null outside the kernel.
pub fn qmv(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    return qmvAct(x, w, sc, bi, bits, group_size, false, s);
}

/// `qmv` with mlx-lm's relu2 on the bf16 output (`relu2`): bf16(max(bf16(acc), 0)^2).
pub fn qmvAct(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, relu2: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !fits(w, sc, bi, bits, group_size) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const ws = mlx.getShape(w);
    if (ws.len != 2) return null;
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs.len > 8) return null;
    const k = xs[xs.len - 1];
    if (k != inWidth(w, bits)) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;
    const n = ws[0];

    if (qmv_kernel == null) qmv_kernel = try newKernel("msv_rowqmv", &.{ "x", "w", "scales", "biases" }, &.{"y"}, QMV_SOURCE);
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, qmv_kernel.?, in_vec, try qmvConfig(.{ .rows = rows, .n = n, .k = k, .bits = bits, .gs = group_size, .relu2 = relu2 }), s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

/// Routed experts, one (row, slot) pair a threadgroup column: pair p is row
/// p / TOPK's slot p % TOPK and reads expert IDS[p]; simdgroup g of
/// threadgroup (b, p) takes the expert's output rows RPS (SG b + g) ..
/// fc1 applies mlx-lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2).
pub const EXPERT_UP_SOURCE = kernelSource("kernels/qmm/rowqmv_expert_up.metal");

/// fc2 over each pair's own activation row.
pub const EXPERT_DOWN_SOURCE = kernelSource("kernels/qmm/rowqmv_expert_down.metal");

const EXPERT_SG = 2;
var up_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ExpertKey = struct { up: bool, pairs: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32 };
var expert_cfgs: std.AutoHashMapUnmanaged(ExpertKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn expertConfig(key: ExpertKey) !mlx.mlx_fast_metal_kernel_config {
    if (expert_cfgs.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ key.pairs, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * EXPERT_SG, @divExact(key.n, RPS * EXPERT_SG), key.pairs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * EXPERT_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RPS", RPS));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SG", EXPERT_SG));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(key.bits)));
    if (key.up) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPK", key.topk));
    try expert_cfgs.put(std.heap.c_allocator, key, config);
    return config;
}

pub fn expertLaunch(up: bool, x: mlx.mlx_array, ids: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, pairs: c_int, topk: c_int, q: [2]u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const kslot = if (up) &up_kernel else &down_kernel;
    if (kslot.* == null) kslot.* = try newKernel(if (up) "msv_rowqmv_expert_up" else "msv_rowqmv_expert_down", &.{ "x", "ids", "w", "scales", "biases" }, &.{"y"}, if (up) EXPERT_UP_SOURCE else EXPERT_DOWN_SOURCE);
    const ws = mlx.getShape(w);
    const inputs = [_]mlx.mlx_array{ x, ids, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    const key = ExpertKey{ .up = up, .pairs = pairs, .topk = topk, .n = ws[1], .k = inWidth(w, q[0]), .bits = q[0], .gs = q[1] };
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kslot.*.?, in_vec, try expertConfig(key), s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

/// mlx-lm's SwitchMLP (fc1, relu2, fc2) on x [R, D] bf16 for the uint32
/// expert ids [R * TOPK] (row-major): [R * TOPK, D] bf16, or null outside the
/// kernel. Each (row, slot) pair is computed on its own, the same way at any R.
/// `q1`/`q2` = (bits, group size) of fc1/fc2.
pub fn experts(x: mlx.mlx_array, ids: mlx.mlx_array, topk: c_int, fc1: [3]mlx.mlx_array, fc2: [3]mlx.mlx_array, q1: [2]u32, q2: [2]u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !fits(fc1[0], fc1[1], fc1[2], q1[0], q1[1]) or !fits(fc2[0], fc2[1], fc2[2], q2[0], q2[1])) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const w1 = mlx.getShape(fc1[0]);
    const w2 = mlx.getShape(fc2[0]);
    if (xs.len != 2 or w1.len != 3 or w2.len != 3) return null;
    const rows = xs[0];
    const pairs = rows * topk;
    if (rows < 1 or rows > MAX_ROWS or mlx.mlx_array_size(ids) != @as(usize, @intCast(pairs))) return null;
    const block = RPS * EXPERT_SG;
    if (xs[1] != inWidth(fc1[0], q1[0]) or inWidth(fc2[0], q2[0]) != w1[1] or @rem(w1[1], block) != 0 or @rem(w2[1], block) != 0) return null;
    const act = try expertLaunch(true, x, ids, fc1[0], fc1[1], fc1[2], pairs, topk, q1, s);
    defer _ = mlx.mlx_array_free(act);
    return try expertLaunch(false, act, ids, fc2[0], fc2[1], fc2[2], pairs, topk, q2, s);
}

/// Dense bf16 router logits for R rows, w [E, D]: one threadgroup of SG
/// simdgroups per expert; simdgroup g sums its D / SG inputs (lane l takes 4
/// consecutive inputs at a time, 128 apart), then the simdgroup sums add in
/// order. The row count is a runtime value, so a row's bits never depend on it.
pub const ROUTER_SOURCE = kernelSource("kernels/qmm/rowqmv_router.metal");

const ROUTER_SG = 8;
var router_kernel: ?mlx.mlx_fast_metal_kernel = null;
const RouterKey = struct { rows: c_int, e: c_int, d: c_int };
var router_cfgs: std.AutoHashMapUnmanaged(RouterKey, mlx.mlx_fast_metal_kernel_config) = .{};

/// `x [R, D] @ w.T` for a dense bf16 w [E, D] (a router), 1..MAX_ROWS rows,
/// or null outside the kernel.
pub fn router(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[1] != ws[1] or @rem(ws[1], 4 * ROUTER_SG) != 0) return null;
    if (xs[0] < 1 or xs[0] > MAX_ROWS) return null;
    const key = RouterKey{ .rows = xs[0], .e = ws[0], .d = ws[1] };
    if (router_kernel == null) router_kernel = try newKernel("msv_rowqmv_router", &.{ "x", "w", "rows" }, &.{"y"}, ROUTER_SOURCE);
    const config = router_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ key.rows, key.e }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * ROUTER_SG, key.e, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * ROUTER_SG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "D", key.d));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "E", key.e));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", ROUTER_SG));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "MAX_ROWS", MAX_ROWS));
        try router_cfgs.put(std.heap.c_allocator, key, c);
        break :blk c;
    };
    const rows = mlx.mlx_array_new_int(key.rows);
    defer _ = mlx.mlx_array_free(rows);
    const inputs = [_]mlx.mlx_array{ x, w, rows };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, router_kernel.?, in_vec, config, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

// ── tests ──

const testing = std.testing;

fn randBf16(a: std.mem.Allocator, shape: []const c_int, scale: f32, rnd: std.Random, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf) |*v| v.* = scale * rnd.floatNorm(f32);
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

const Quant = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    fn deinit(q: Quant) void {
        _ = mlx.mlx_array_free(q.w);
        _ = mlx.mlx_array_free(q.sc);
        _ = mlx.mlx_array_free(q.bi);
    }
};

fn quantize(wf: mlx.mlx_array, gs: c_int, bits: c_int, s: mlx.mlx_stream) !Quant {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q = Quant{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .bi = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q.sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q.bi, triple, 2));
    return q;
}

fn sliceRows(x: mlx.mlx_array, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const k = mlx.getShape(x)[1];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ lo, 0 }, 2, &[_]c_int{ hi, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

fn expectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

/// Max |got - truth| relative to max |truth|, truth = f32 x @ dequant(w).T.
fn relErr(got: mlx.mlx_array, x: mlx.mlx_array, q: Quant, gs: c_int, bits: c_int, s: mlx.mlx_stream) !f32 {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_dequantize(&wt, q.w, q.sc, q.bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var wtt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wtt);
    try mlx.check(mlx.mlx_transpose(&wtt, wt, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wtt, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var at = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(at);
    try mlx.check(mlx.mlx_abs(&at, truth, s));
    var md = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(md);
    try mlx.check(mlx.mlx_max(&md, ad, false, s));
    var mt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mt);
    try mlx.check(mlx.mlx_max(&mt, at, false, s));
    var a: f32 = 0;
    var b: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&a, md));
    try mlx.check(mlx.mlx_array_item_float32(&b, mt));
    return a / b;
}

test "rowqmv.qmv: every row of an R-row call equals its one-row call bit for bit, and matches f32 truth" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7f);
    const rnd = prng.random();
    // Nemotron-H's shapes: K = 2688 (not a multiple of 512), 4096, 1856, 3712.
    const Shape = struct { n: c_int, k: c_int, gs: c_int };
    for ([_]u32{ 4, 6, 8 }) |bits| for ([_]Shape{ .{ .n = 1024, .k = 2688, .gs = 64 }, .{ .n = 512, .k = 4096, .gs = 64 }, .{ .n = 2688, .k = 1856, .gs = 64 }, .{ .n = 256, .k = 3712, .gs = 32 }, .{ .n = 264, .k = 640, .gs = 128 } }) |sh| {
        const wf = try randBf16(a, &.{ sh.n, sh.k }, 0.02, rnd, s);
        defer _ = mlx.mlx_array_free(wf);
        const q = try quantize(wf, sh.gs, @intCast(bits), s);
        defer q.deinit();
        const x = try randBf16(a, &.{ MAX_ROWS, sh.k }, 1.0, rnd, s);
        defer _ = mlx.mlx_array_free(x);

        const all = (try qmv(x, q.w, q.sc, q.bi, bits, @intCast(sh.gs), s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try relErr(all, x, q, sh.gs, @intCast(bits), s) < 2e-2);
        var r: c_int = 0;
        while (r < MAX_ROWS) : (r += 1) {
            const one_x = try sliceRows(x, r, r + 1, s);
            defer _ = mlx.mlx_array_free(one_x);
            const one = (try qmv(one_x, q.w, q.sc, q.bi, bits, @intCast(sh.gs), s)).?;
            defer _ = mlx.mlx_array_free(one);
            const row = try sliceRows(all, r, r + 1, s);
            defer _ = mlx.mlx_array_free(row);
            try expectBitEqual(one, row, s);
        }
        // A 3-row window starting mid-batch gives the same rows too.
        const win_x = try sliceRows(x, 5, 8, s);
        defer _ = mlx.mlx_array_free(win_x);
        const win = (try qmv(win_x, q.w, q.sc, q.bi, bits, @intCast(sh.gs), s)).?;
        defer _ = mlx.mlx_array_free(win);
        const want = try sliceRows(all, 5, 8, s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(win, want, s);
    };
}

test "rowqmv.qmv: a one-row call is bit-equal to stock quantized_matmul at 4, 6 and 8 bits" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x27b);
    const rnd = prng.random();
    // Qwen3.8-27B's widths: K = 5120 (hidden) and 17408 (MLP).
    const Shape = struct { n: c_int, k: c_int, gs: c_int };
    for ([_]u32{ 4, 6, 8 }) |bits| for ([_]Shape{ .{ .n = 512, .k = 5120, .gs = 64 }, .{ .n = 256, .k = 17408, .gs = 64 }, .{ .n = 264, .k = 5120, .gs = 32 }, .{ .n = 136, .k = 5120, .gs = 128 } }) |sh| {
        const wf = try randBf16(a, &.{ sh.n, sh.k }, 0.02, rnd, s);
        defer _ = mlx.mlx_array_free(wf);
        const q = try quantize(wf, sh.gs, @intCast(bits), s);
        defer q.deinit();
        const x = try randBf16(a, &.{ 1, sh.k }, 1.0, rnd, s);
        defer _ = mlx.mlx_array_free(x);
        const got = (try qmv(x, q.w, q.sc, q.bi, bits, @intCast(sh.gs), s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_quantized_matmul(&want, x, q.w, q.sc, q.bi, true, mlx.mlx_optional_int.some(sh.gs), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
        try expectBitEqual(got, want, s);
    };
}

test "rowqmv.qmvAct: relu2 is max(bf16(y), 0)^2 rounded to bf16, rows width-invariant" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const wf = try randBf16(a, &.{ 3712, 2688 }, 0.02, prng.random(), s);
    defer _ = mlx.mlx_array_free(wf);
    const q = try quantize(wf, 64, 4, s);
    defer q.deinit();
    const x = try randBf16(a, &.{ 5, 2688 }, 1.0, prng.random(), s);
    defer _ = mlx.mlx_array_free(x);
    const plain = (try qmv(x, q.w, q.sc, q.bi, 4, 64, s)).?;
    defer _ = mlx.mlx_array_free(plain);
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var zb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zb);
    try mlx.check(mlx.mlx_astype(&zb, zero, .bfloat16, s));
    var mx_ = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_);
    try mlx.check(mlx.mlx_maximum(&mx_, plain, zb, s));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_square(&want, mx_, s));
    const got = (try qmvAct(x, q.w, q.sc, q.bi, 4, 64, true, s)).?;
    defer _ = mlx.mlx_array_free(got);
    try expectBitEqual(got, want, s);
    const x1 = try sliceRows(x, 3, 4, s);
    defer _ = mlx.mlx_array_free(x1);
    const one = (try qmvAct(x1, q.w, q.sc, q.bi, 4, 64, true, s)).?;
    defer _ = mlx.mlx_array_free(one);
    const row = try sliceRows(got, 3, 4, s);
    defer _ = mlx.mlx_array_free(row);
    try expectBitEqual(one, row, s);
}

test "rowqmv.qmv: declines what it cannot read" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const wf = try randBf16(a, &.{ 64, 512 }, 0.02, prng.random(), s);
    defer _ = mlx.mlx_array_free(wf);
    const q = try quantize(wf, 64, 4, s);
    defer q.deinit();
    const x = try randBf16(a, &.{ MAX_ROWS + 1, 512 }, 1.0, prng.random(), s);
    defer _ = mlx.mlx_array_free(x);
    try testing.expect((try qmv(x, q.w, q.sc, q.bi, 4, 64, s)) == null);
    const x1 = try sliceRows(x, 0, 1, s);
    defer _ = mlx.mlx_array_free(x1);
    try testing.expect((try qmv(x1, q.w, q.sc, q.bi, 3, 64, s)) == null);
    try testing.expect((try qmv(x1, q.w, q.sc, q.bi, 8, 64, s)) == null);
}

fn sumSq(v: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, v, s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_sum(&t, sq, false, s));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, t));
    return out;
}

fn takeExpert(bank: mlx.mlx_array, e: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(bank);
    var sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl);
    try mlx.check(mlx.mlx_slice(&sl, bank, &[_]c_int{ e, 0, 0 }, 3, &[_]c_int{ e + 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, sl, &[_]c_int{ sh[1], sh[2] }, 2, s));
    return out;
}

fn dequantF32(q: [3]mlx.mlx_array, gs: c_int, bits: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&w, q[0], q[1], q[2], mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    return w;
}

test "rowqmv.experts: rows are width-invariant bit for bit and match the f32 SwitchMLP" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE1);
    const rnd = prng.random();
    // Nemotron-H's geometry: D = 2688 and I = 1856 are not multiples of 512.
    const E: c_int = 8;
    const D: c_int = 2688;
    const I: c_int = 1856;
    const TOPK: c_int = 6;
    for ([_]u32{ 4, 6, 8 }) |bits| {
    const f1 = try randBf16(a, &.{ E, I, D }, 0.03, rnd, s);
    defer _ = mlx.mlx_array_free(f1);
    const f2 = try randBf16(a, &.{ E, D, I }, 0.03, rnd, s);
    defer _ = mlx.mlx_array_free(f2);
    const q1 = try quantize(f1, 64, @intCast(bits), s);
    defer q1.deinit();
    const q2 = try quantize(f2, 64, @intCast(bits), s);
    defer q2.deinit();
    const fc1 = [3]mlx.mlx_array{ q1.w, q1.sc, q1.bi };
    const fc2 = [3]mlx.mlx_array{ q2.w, q2.sc, q2.bi };
    const x = try randBf16(a, &.{ MAX_ROWS, D }, 1.0, rnd, s);
    defer _ = mlx.mlx_array_free(x);
    var idv: [MAX_ROWS * TOPK]u32 = undefined;
    for (&idv) |*v| v.* = rnd.uintLessThan(u32, E);
    const ids = mlx.mlx_array_new_data(&idv, &[_]c_int{MAX_ROWS * TOPK}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);

    const all = (try experts(x, ids, TOPK, fc1, fc2, .{ bits, 64 }, .{ bits, 64 }, s)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(all);
    try testing.expectEqualSlices(c_int, &.{ MAX_ROWS * TOPK, D }, mlx.getShape(all));

    // Width invariance: a 1-row call and a 4-row window give the same rows.
    for ([_][2]c_int{ .{ 0, 1 }, .{ 7, 8 }, .{ 3, 7 } }) |win| {
        const xw = try sliceRows(x, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(xw);
        var idw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idw);
        try mlx.check(mlx.mlx_slice(&idw, ids, &[_]c_int{win[0] * TOPK}, 1, &[_]c_int{win[1] * TOPK}, 1, &[_]c_int{1}, 1, s));
        const got = (try experts(xw, idw, TOPK, fc1, fc2, .{ bits, 64 }, .{ bits, 64 }, s)).?;
        defer _ = mlx.mlx_array_free(got);
        const want = try sliceRows(all, win[0] * TOPK, win[1] * TOPK, s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(got, want, s);
    }

    // Truth per pair: f32 x @ W1[e].T, relu^2, @ W2[e].T.
    var worst: f32 = 0;
    for (0..MAX_ROWS * TOPK) |p| {
        const e: c_int = @intCast(idv[p]);
        const r: c_int = @intCast(p / @as(usize, TOPK));
        var w1s: [3]mlx.mlx_array = undefined;
        var w2s: [3]mlx.mlx_array = undefined;
        for (0..3) |i| {
            w1s[i] = try takeExpert(fc1[i], e, s);
            w2s[i] = try takeExpert(fc2[i], e, s);
        }
        defer for (w1s ++ w2s) |t| {
            _ = mlx.mlx_array_free(t);
        };
        const d1 = try dequantF32(w1s, 64, @intCast(bits), s);
        defer _ = mlx.mlx_array_free(d1);
        const d2 = try dequantF32(w2s, 64, @intCast(bits), s);
        defer _ = mlx.mlx_array_free(d2);
        const xr = try sliceRows(x, r, r + 1, s);
        defer _ = mlx.mlx_array_free(xr);
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, xr, .float32, s));
        var t1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1);
        try mlx.check(mlx.mlx_transpose(&t1, d1, s));
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_matmul(&h, xf, t1, s));
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        var hr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hr);
        try mlx.check(mlx.mlx_maximum(&hr, h, zero, s));
        var h2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h2);
        try mlx.check(mlx.mlx_square(&h2, hr, s));
        var t2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t2);
        try mlx.check(mlx.mlx_transpose(&t2, d2, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, h2, t2, s));
        const got = try sliceRows(all, @intCast(p), @intCast(p + 1), s);
        defer _ = mlx.mlx_array_free(got);
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
        worst = @max(worst, @sqrt(try sumSq(d, s) / try sumSq(truth, s)));
    }
    try testing.expect(worst < 2e-2);
    }
}

test "rowqmv.router: rows are width-invariant bit for bit and match f32 truth" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA7);
    const rnd = prng.random();
    const w = try randBf16(a, &.{ 128, 2688 }, 0.05, rnd, s);
    defer _ = mlx.mlx_array_free(w);
    const x = try randBf16(a, &.{ MAX_ROWS, 2688 }, 1.0, rnd, s);
    defer _ = mlx.mlx_array_free(x);
    const all = (try router(x, w, s)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(all);
    for ([_][2]c_int{ .{ 0, 1 }, .{ 9, 10 }, .{ 2, 5 } }) |win| {
        const xw = try sliceRows(x, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(xw);
        const got = (try router(xw, w, s)).?;
        defer _ = mlx.mlx_array_free(got);
        const want = try sliceRows(all, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(got, want, s);
    }
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_astype(&wf, w, .float32, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wf, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&gf, all, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
    try testing.expect(@sqrt(try sumSq(d, s) / try sumSq(truth, s)) < 1e-2);
}
